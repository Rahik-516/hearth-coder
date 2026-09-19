"""Vector storage and exact similarity search.

There is no vector database (ADR 0001). Vectors live as float16 BLOBs in the same SQLite
file as everything else, and search is an exact dot product over an in-memory NumPy
matrix. At Hearth's target scale that is fast enough by a wide margin, it has no recall
cliff, no index build step, and no second store to keep consistent.

Two things make this work:

* **Vectors are L2-normalized on write**, so cosine similarity is a plain matmul.
* **The matrix is a lazy cache.** It is rebuilt on first search after a write, not on
  every write — indexing a repository performs thousands of inserts and one search.

Vectors are keyed by ``embed_text_hash``, not by chunk id. Identical text anywhere in the
repository therefore embeds once, which is what makes a branch switch or a moved file
nearly free (docs/system-design.md §6.7).

``float16`` halves memory at a cosine-similarity cost far below the noise floor of
retrieval: 100K chunks x 1024 dims is ~200 MB rather than ~400 MB.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class VectorHit:
    """One similarity hit."""

    embed_text_hash: str
    score: float


class VectorIndex(Protocol):
    """Pluggable vector backend.

    A scale backend (sqlite-vec, LanceDB) can be added behind this protocol if a
    repository ever exceeds ~1M chunks, with no change above it (docs/system-design.md §18).
    """

    def upsert(self, vectors: Mapping[str, np.ndarray]) -> int: ...

    def delete(self, hashes: Iterable[str]) -> int: ...

    def search(
        self,
        query: np.ndarray,
        *,
        k: int = 40,
        candidates: set[str] | None = None,
    ) -> list[VectorHit]: ...

    def count(self) -> int: ...


class NumpyVectorIndex:
    """Exact dot-product search over float16 vectors stored in SQLite."""

    def __init__(self, connection: sqlite3.Connection, *, model_id: str, dims: int) -> None:
        self._connection = connection
        self._model_id = model_id
        self._dims = dims
        self._matrix: np.ndarray | None = None
        self._hashes: list[str] = []
        self._dirty = True
        self._coverage: tuple[int, int] | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def commit(self) -> None:
        """Commit pending writes.

        The embedding phase calls this per batch. That is what makes it resumable: a
        cancelled run loses at most the batch in flight, and the next run recomputes its
        work list from what is actually missing rather than from a saved cursor.
        """
        self._connection.commit()

    # ------------------------------------------------------------------ write

    def upsert(self, vectors: Mapping[str, np.ndarray]) -> int:
        """Store vectors, normalizing and downcasting to float16.

        Caller supplies ``embed_text_hash -> vector``. Existing rows are replaced, so this
        is safe to re-run over a partially embedded index.
        """
        if not vectors:
            return 0

        rows = []
        for text_hash, vector in vectors.items():
            prepared = _normalize(np.asarray(vector, dtype=np.float32))
            if prepared.shape[0] != self._dims:
                raise ValueError(
                    f"vector for {text_hash[:12]} has {prepared.shape[0]} dims, expected {self._dims}"
                )
            rows.append((text_hash, self._model_id, self._dims, prepared.astype(np.float16).tobytes()))

        self._connection.executemany(
            "INSERT INTO embeddings(embed_text_hash, model_id, dims, vector) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(embed_text_hash, model_id, dims) DO UPDATE SET vector = excluded.vector",
            rows,
        )
        self._dirty = True
        self._coverage = None
        return len(rows)

    def delete(self, hashes: Iterable[str]) -> int:
        targets = list(hashes)
        if not targets:
            return 0

        cursor = self._connection.executemany(
            "DELETE FROM embeddings WHERE embed_text_hash = ? AND model_id = ? AND dims = ?",
            [(text_hash, self._model_id, self._dims) for text_hash in targets],
        )
        self._dirty = True
        self._coverage = None
        return cursor.rowcount if cursor.rowcount > 0 else 0

    def prune_orphans(self) -> int:
        """Delete vectors whose text no longer appears in any chunk.

        Embeddings are keyed by content, so a deleted file does not remove them. Left
        alone they would accumulate across every edit of every file.
        """
        cursor = self._connection.execute(
            "DELETE FROM embeddings WHERE embed_text_hash NOT IN (SELECT embed_text_hash FROM chunks)"
        )
        removed = cursor.rowcount if cursor.rowcount > 0 else 0
        if removed:
            self._dirty = True
            self._coverage = None
        return removed

    # ----------------------------------------------------------------- search

    def search(
        self,
        query: np.ndarray,
        *,
        k: int = 40,
        candidates: set[str] | None = None,
    ) -> list[VectorHit]:
        """Top-k by cosine similarity.

        ``candidates`` restricts the search to a set of hashes — used for language or path
        prefiltering. Masking the score vector is far cheaper than building a second index
        per filter.
        """
        matrix, hashes = self._ensure_matrix()
        if matrix.size == 0 or k <= 0:
            return []

        vector = _normalize(np.asarray(query, dtype=np.float32))
        if vector.shape[0] != matrix.shape[1]:
            raise ValueError(f"query has {vector.shape[0]} dims, index has {matrix.shape[1]}")

        scores = matrix @ vector

        if candidates is not None:
            if not candidates:
                return []
            mask = np.fromiter((h in candidates for h in hashes), dtype=bool, count=len(hashes))
            scores = np.where(mask, scores, -np.inf)

        limit = min(k, scores.shape[0])
        # argpartition finds the top-k without sorting everything, then only those are
        # sorted. At 100K vectors that is the difference between ~1ms and ~10ms.
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]

        return [
            VectorHit(embed_text_hash=hashes[int(i)], score=float(scores[int(i)]))
            for i in top
            if np.isfinite(scores[int(i)])
        ]

    # ------------------------------------------------------------------ state

    def count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE model_id = ? AND dims = ?",
            (self._model_id, self._dims),
        ).fetchone()
        return int(row[0])

    def coverage(self) -> tuple[int, int]:
        """``(embedded, total)`` distinct chunk texts.

        Retrieval degrades gracefully rather than failing while embeddings are still
        building, and the CLI reports this so the user knows dense results are partial
        (docs/system-design.md §6.1).

        **Cached**, because these are two ``COUNT(DISTINCT)`` queries over every chunk.
        Recomputing them on every retrieve cost ~100 ms at 100K chunks — most of the
        retrieval budget, spent on a diagnostic the caller usually only displays. The
        cache is invalidated by any write through this class.
        """
        if self._coverage is None:
            total = int(
                self._connection.execute("SELECT COUNT(DISTINCT embed_text_hash) FROM chunks").fetchone()[0]
            )
            embedded = int(
                self._connection.execute(
                    "SELECT COUNT(DISTINCT c.embed_text_hash) FROM chunks c "
                    "JOIN embeddings e ON e.embed_text_hash = c.embed_text_hash "
                    "WHERE e.model_id = ? AND e.dims = ?",
                    (self._model_id, self._dims),
                ).fetchone()[0]
            )
            self._coverage = (embedded, total)
        return self._coverage

    def missing_hashes(self, limit: int | None = None) -> list[str]:
        """Chunk texts with no vector yet, for the resumable embedding phase."""
        sql = (
            "SELECT DISTINCT c.embed_text_hash FROM chunks c "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM embeddings e "
            "  WHERE e.embed_text_hash = c.embed_text_hash AND e.model_id = ? AND e.dims = ?"
            ") ORDER BY c.embed_text_hash"
        )
        params: tuple[object, ...] = (self._model_id, self._dims)
        if limit is not None:
            sql += " LIMIT ?"
            params = (*params, limit)

        rows = self._connection.execute(sql, params).fetchall()
        return [str(row[0]) for row in rows]

    def invalidate(self) -> None:
        """Drop the in-memory caches. Call after external writes to ``embeddings`` or
        ``chunks`` — indexing new files changes coverage without going through this class.
        """
        self._matrix = None
        self._hashes = []
        self._dirty = True
        self._coverage = None

    # -------------------------------------------------------------- internals

    def _ensure_matrix(self) -> tuple[np.ndarray, Sequence[str]]:
        if not self._dirty and self._matrix is not None:
            return self._matrix, self._hashes

        rows = self._connection.execute(
            "SELECT embed_text_hash, vector FROM embeddings "
            "WHERE model_id = ? AND dims = ? ORDER BY embed_text_hash",
            (self._model_id, self._dims),
        ).fetchall()

        if not rows:
            self._matrix = np.zeros((0, self._dims), dtype=np.float32)
            self._hashes = []
        else:
            self._hashes = [str(row[0]) for row in rows]
            # float32 for the matmul: float16 accumulation loses precision, and the
            # matrix is transient while the stored form stays float16.
            self._matrix = np.stack(
                [np.frombuffer(row[1], dtype=np.float16).astype(np.float32) for row in rows]
            )

        self._dirty = False
        return self._matrix, self._hashes


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Scale to unit length, leaving an all-zero vector untouched."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm
