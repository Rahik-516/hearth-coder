"""Read/write access to ``index.db``.

**The invariant this module exists to hold:** ``chunks_fts`` is a *contentless* FTS5
table, so SQLite stores no copy of the text and maintains nothing automatically. Every
insert and delete must be mirrored here, with a matching rowid, inside the same
transaction as the ``chunks`` row (docs/system-design.md §13.1).

Get that wrong and search returns rowids for chunks that no longer exist — which surfaces
as citations pointing at deleted code, long after the bug.

All writes go through ``replace_file``, which is the only safe way to keep the two in
step.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, TypedDict, cast

from hearth.storage.db import transaction
from hearth.util.text import build_search_text


@dataclass
class FileRecord:
    """A file as stored in the index."""

    path: str
    language: str | None
    size_bytes: int
    mtime_ns: int
    content_hash: str
    is_generated: bool = False
    parse_status: str = "ok"
    id: int | None = None
    indexed_at: int = 0


@dataclass
class ChunkRecord:
    """One indexed chunk. ``search_text`` is derived, not stored on the chunk itself."""

    kind: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    text: str
    embed_text_hash: str
    token_estimate: int
    symbol_path: str | None = None
    id: int | None = None


@dataclass
class SymbolRecord:
    name: str
    kind: str
    start_line: int
    end_line: int
    signature: str | None = None
    exported: bool | None = None
    parent_index: int | None = None  # index into the same batch, resolved on write
    id: int | None = None


@dataclass
class RefRecord:
    name: str
    line: int
    kind: str


@dataclass
class ImportRecord:
    module_spec: str
    names: list[str] = field(default_factory=list)


class SearchHit(TypedDict):
    """One lexical search result.

    A TypedDict rather than a dataclass so callers keep subscript access while the fields
    stay typed — retrieval in M2 fuses these with dense and symbol hits, and untyped
    ``object`` values there would be a constant source of casts.
    """

    chunk_id: int
    path: str
    kind: str
    symbol_path: str | None
    start_line: int
    end_line: int
    text: str
    raw_score: float
    score: float


@dataclass
class FilePayload:
    """Everything extracted from one file, written as a unit."""

    file: FileRecord
    chunks: list[ChunkRecord] = field(default_factory=list)
    symbols: list[SymbolRecord] = field(default_factory=list)
    refs: list[RefRecord] = field(default_factory=list)
    imports: list[ImportRecord] = field(default_factory=list)


class IndexRepository:
    """Queries and writes over ``index.db``."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    # ------------------------------------------------------------------ meta

    def set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def get_meta(self, key: str) -> str | None:
        row = self._connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    # ----------------------------------------------------------------- files

    def get_file(self, path: str) -> FileRecord | None:
        row = self._connection.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
        return None if row is None else _file_from_row(row)

    def all_files(self) -> list[FileRecord]:
        rows = self._connection.execute("SELECT * FROM files ORDER BY path")
        return [_file_from_row(row) for row in rows]

    def file_paths(self) -> set[str]:
        rows = self._connection.execute("SELECT path FROM files")
        return {str(row["path"]) for row in rows}

    def file_states(self) -> dict[str, tuple[int, int, str]]:
        """``path -> (size_bytes, mtime_ns, content_hash)`` for cheap change detection."""
        rows = self._connection.execute("SELECT path, size_bytes, mtime_ns, content_hash FROM files")
        return {
            str(r["path"]): (int(r["size_bytes"]), int(r["mtime_ns"]), str(r["content_hash"])) for r in rows
        }

    def paths_now_parseable(self, parseable: Collection[str]) -> set[str]:
        """Indexed files that were skipped for want of a grammar that now exists.

        Installing a language grammar changes what Hearth can extract from files whose
        *contents* never changed, so change detection — which compares size, mtime and
        hash — sees nothing to do. Without this the symbols for that language stay empty
        until someone thinks to run `hearth index --rebuild`, and nothing suggests they
        should.
        """
        if not parseable:
            return set()

        placeholders = ",".join("?" * len(parseable))
        rows = self._connection.execute(
            f"SELECT path FROM files WHERE parse_status = 'skipped' AND language IN ({placeholders})",  # noqa: S608 - placeholders only
            tuple(parseable),
        ).fetchall()
        return {str(row["path"]) for row in rows}

    def count_files(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM files").fetchone()[0])

    def count_chunks(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def count_symbols(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM symbols").fetchone()[0])

    # ----------------------------------------------------------------- write

    def replace_file(self, payload: FilePayload) -> int:
        """Insert or replace one file and everything derived from it.

        Deletes first, so this is idempotent and safe to call on an unchanged file.
        ``ON DELETE CASCADE`` removes chunks, symbols, refs and imports, but the
        contentless FTS rows are *not* cascaded and must be removed explicitly first —
        which is why this method, not the caller, owns the ordering.
        """
        connection = self._connection
        path = payload.file.path

        self._delete_fts_rows_for_path(path)
        connection.execute("DELETE FROM files WHERE path = ?", (path,))

        now = int(time.time())
        cursor = connection.execute(
            "INSERT INTO files(path, language, size_bytes, mtime_ns, content_hash, "
            "is_generated, parse_status, indexed_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                path,
                payload.file.language,
                payload.file.size_bytes,
                payload.file.mtime_ns,
                payload.file.content_hash,
                int(payload.file.is_generated),
                payload.file.parse_status,
                now,
            ),
        )
        file_id = int(cursor.lastrowid or 0)

        self._insert_chunks(file_id, path, payload.chunks)
        self._insert_symbols(file_id, payload.symbols)
        self._insert_refs(file_id, payload.refs)
        self._insert_imports(file_id, payload.imports)
        return file_id

    def replace_files(self, payloads: Iterable[FilePayload]) -> int:
        """Write a batch in one transaction (docs/system-design.md §6.1)."""
        count = 0
        with transaction(self._connection):
            for payload in payloads:
                self.replace_file(payload)
                count += 1
        return count

    def delete_file(self, path: str) -> bool:
        """Remove a file and everything derived from it. False if it was not indexed."""
        self._delete_fts_rows_for_path(path)
        cursor = self._connection.execute("DELETE FROM files WHERE path = ?", (path,))
        return cursor.rowcount > 0

    def delete_files(self, paths: Iterable[str]) -> int:
        removed = 0
        with transaction(self._connection):
            for path in paths:
                if self.delete_file(path):
                    removed += 1
        return removed

    # --------------------------------------------------------------- internals

    def _delete_fts_rows_for_path(self, path: str) -> None:
        """Drop FTS rows for a path before its chunks disappear.

        Must run *before* the cascade deletes the chunks: the rowids come from ``chunks``,
        and once those rows are gone there is no way to find the FTS entries again.

        This relies on ``contentless_delete=1``. The older contentless form would require
        replaying each row's original column values here, which would silently break the
        moment ``build_search_text`` changed.
        """
        self._connection.execute(
            "DELETE FROM chunks_fts WHERE rowid IN ("
            "  SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = ?"
            ")",
            (path,),
        )

    def _insert_chunks(self, file_id: int, path: str, chunks: Sequence[ChunkRecord]) -> None:
        for chunk in chunks:
            cursor = self._connection.execute(
                "INSERT INTO chunks(file_id, kind, symbol_path, start_line, end_line, "
                "start_byte, end_byte, text, embed_text_hash, token_estimate) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    file_id,
                    chunk.kind,
                    chunk.symbol_path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.start_byte,
                    chunk.end_byte,
                    chunk.text,
                    chunk.embed_text_hash,
                    chunk.token_estimate,
                ),
            )
            chunk_id = int(cursor.lastrowid or 0)
            chunk.id = chunk_id

            # Same transaction, matching rowid. This is the invariant.
            #
            # All three columns are identifier-expanded, not just the body. The tokenizer
            # treats `InvoiceService` as one token and `invoice_service.py` as another, so
            # without expansion a query for "invoice" matches neither the symbol column
            # nor the path column — and their BM25 weights, which exist precisely to rank
            # a name match above a body mention, would never fire.
            self._connection.execute(
                "INSERT INTO chunks_fts(rowid, search_text, symbol_path, path) VALUES(?, ?, ?, ?)",
                (
                    chunk_id,
                    build_search_text(chunk.text),
                    build_search_text(chunk.symbol_path or ""),
                    build_search_text(path),
                ),
            )

    def _insert_symbols(self, file_id: int, symbols: Sequence[SymbolRecord]) -> None:
        assigned: list[int] = []
        for symbol in symbols:
            parent_id = (
                assigned[symbol.parent_index]
                if symbol.parent_index is not None and symbol.parent_index < len(assigned)
                else None
            )
            cursor = self._connection.execute(
                "INSERT INTO symbols(file_id, name, kind, parent_id, start_line, end_line, "
                "signature, exported) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    file_id,
                    symbol.name,
                    symbol.kind,
                    parent_id,
                    symbol.start_line,
                    symbol.end_line,
                    symbol.signature,
                    None if symbol.exported is None else int(symbol.exported),
                ),
            )
            symbol_id = int(cursor.lastrowid or 0)
            symbol.id = symbol_id
            assigned.append(symbol_id)

    def _insert_refs(self, file_id: int, refs: Sequence[RefRecord]) -> None:
        self._connection.executemany(
            "INSERT INTO refs(file_id, name, line, kind) VALUES(?, ?, ?, ?)",
            [(file_id, ref.name, ref.line, ref.kind) for ref in refs],
        )

    def _insert_imports(self, file_id: int, imports: Sequence[ImportRecord]) -> None:
        self._connection.executemany(
            "INSERT INTO imports(file_id, module_spec, names) VALUES(?, ?, ?)",
            [(file_id, imp.module_spec, json.dumps(imp.names)) for imp in imports],
        )

    # ---------------------------------------------------------------- queries

    #: Per-kind multipliers applied to the BM25 score.
    #:
    #: BM25 rewards term density in short chunks, which systematically favours a file's
    #: import-and-docstring preamble over the definition it describes: the preamble
    #: *mentions* `InvoiceService.finalize` in a few lines, while the class that *is* it
    #: runs for eighty. Someone searching for a symbol wants the code, so real definitions
    #: are boosted and navigational or incidental chunks are demoted.
    #:
    #: SQLite's bm25() returns negative scores where lower is better, so a multiplier
    #: above 1.0 promotes and below 1.0 demotes.
    _KIND_WEIGHTS: ClassVar[dict[str, float]] = {
        "class": 1.35,
        "function": 1.35,
        "method": 1.35,
        "class_skeleton": 1.05,
        "section": 1.0,
        "window": 1.0,
        "module_skeleton": 0.70,  # useful for "what's in this file", not for a symbol
        "preamble": 0.80,  # imports and docstrings are rarely the answer
    }

    def search_lexical(self, query: str, *, limit: int = 20) -> list[SearchHit]:
        """BM25 search over ``chunks_fts``, joined back to chunk and file rows.

        Column weights favour a symbol-path hit over a body hit: matching the *name* of a
        definition is a far stronger signal than the same word appearing inside it. A
        per-kind weight then corrects BM25's bias toward short chunks.

        This is deliberately simple. Proper multi-retriever fusion with rank-based scoring
        arrives in M2 (docs/system-design.md §7); until then this keeps lexical-only
        results defensible.
        """
        cases = " ".join(f"WHEN '{kind}' THEN {weight}" for kind, weight in self._KIND_WEIGHTS.items())
        rows = self._connection.execute(
            f"""
            SELECT
              c.id            AS chunk_id,
              f.path          AS path,
              c.kind          AS kind,
              c.symbol_path   AS symbol_path,
              c.start_line    AS start_line,
              c.end_line      AS end_line,
              c.text          AS text,
              bm25(chunks_fts, 1.0, 4.0, 2.0) AS raw_score,
              bm25(chunks_fts, 1.0, 4.0, 2.0)
                * (CASE c.kind {cases} ELSE 1.0 END) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN files  f ON f.id = c.file_id
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,  # noqa: S608 — `cases` is built from this class's own literal table
            (query, limit),
        ).fetchall()
        return [cast("SearchHit", dict(row)) for row in rows]

    def find_references(self, name: str, *, limit: int = 200) -> list[dict[str, object]]:
        """Places a name is used: ``path``, ``line`` and ``kind`` (call, type, attribute…).

        By simple name, so two unrelated ``parse`` functions share references. That is the
        index's real precision, and callers present the result as "known references, may be
        incomplete or include namesakes" rather than as a call graph.
        """
        rows = self._connection.execute(
            """
            SELECT f.path, r.line, r.kind
            FROM refs r JOIN files f ON f.id = r.file_id
            WHERE r.name = ? COLLATE NOCASE
            ORDER BY f.path, r.line
            LIMIT ?
            """,
            (name, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def find_symbol(self, name: str, *, limit: int = 20) -> list[dict[str, object]]:
        rows = self._connection.execute(
            """
            SELECT s.name, s.kind, s.signature, s.start_line, s.end_line, f.path
            FROM symbols s JOIN files f ON f.id = s.file_id
            WHERE s.name = ? COLLATE NOCASE
            ORDER BY f.path, s.start_line
            LIMIT ?
            """,
            (name, limit),
        ).fetchall()
        return [dict(row) for row in rows]


def _file_from_row(row: sqlite3.Row) -> FileRecord:
    return FileRecord(
        id=int(row["id"]),
        path=str(row["path"]),
        language=row["language"],
        size_bytes=int(row["size_bytes"]),
        mtime_ns=int(row["mtime_ns"]),
        content_hash=str(row["content_hash"]),
        is_generated=bool(row["is_generated"]),
        parse_status=str(row["parse_status"]),
        indexed_at=int(row["indexed_at"]),
    )
