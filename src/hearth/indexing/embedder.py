"""The embedding phase.

Runs after the lexical phase, in the background. Retrieval works on whatever is available,
which is what lets a developer ask questions a minute into indexing a large repository
rather than waiting for a full embed (docs/system-design.md §6.1).

Three properties this module is responsible for:

* **Cached by content.** Vectors are keyed by ``(model, dims, embed_text_hash)``. Identical
  text embeds once, so a branch switch or a moved file costs nothing.
* **Resumable.** Progress commits per batch, so Ctrl+C loses at most one batch and a
  restart re-embeds nothing already stored — an M2 acceptance criterion.
* **Placement-aware.** During an interactive session, query embedding is pinned to CPU so
  it cannot evict the chat model from a 6 GB GPU (docs/system-design.md §6.7).

Prompt templates come from the model profile, never from code: Qwen3-Embedding wants an
instruction prefix on queries only, other families differ, and getting it wrong degrades
retrieval silently.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from hearth.indexing.enrich import build_context_header, compose_embed_text
from hearth.llm.errors import LLMError
from hearth.llm.profiles import ModelProfile
from hearth.llm.provider import LLMProvider
from hearth.storage.vector_index import NumpyVectorIndex

#: Items per /api/embed call. Ollama serializes per model, so larger batches mostly trade
#: latency for throughput; 32 is a reasonable middle on the reference machine.
DEFAULT_BATCH_SIZE = 32

#: Ollama serializes per model unless configured otherwise, so concurrency above ~2 buys
#: nothing and risks timeouts (docs/system-design.md §6.7).
DEFAULT_CONCURRENCY = 2

EmbedProgress = Callable[["EmbedStats"], None]


@dataclass
class EmbedStats:
    """What an embedding run did."""

    total_missing: int = 0
    embedded: int = 0
    batches: int = 0
    failed: int = 0
    skipped_empty: int = 0
    duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.total_missing - self.embedded - self.failed)

    @property
    def complete(self) -> bool:
        return self.remaining == 0 and not self.failed

    def summary(self) -> str:
        return (
            f"{self.embedded}/{self.total_missing} embedded in {self.batches} batch(es), {self.failed} failed"
        )


def apply_query_template(profile: ModelProfile | None, query: str) -> str:
    """Wrap a query in the model's instruction template.

    Qwen3-Embedding applies an instruction prefix to queries but not documents; applying
    it to both, or to neither, measurably degrades retrieval
    (docs/model-recommendations.md §6.1).
    """
    if profile is None or profile.query_template is None:
        return query
    return profile.query_template.replace("{text}", query)


def apply_document_template(profile: ModelProfile | None, text: str) -> str:
    if profile is None or profile.document_template is None:
        return text
    return profile.document_template.replace("{text}", text)


class Embedder:
    """Fills in missing vectors for an index."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        vector_index: NumpyVectorIndex,
        model: str,
        dimensions: int | None = None,
        profile: ModelProfile | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        concurrency: int = DEFAULT_CONCURRENCY,
        on_cpu: bool = False,
        on_progress: EmbedProgress | None = None,
    ) -> None:
        self._provider = provider
        self._index = vector_index
        self._model = model
        self._dimensions = dimensions
        self._profile = profile
        self._batch_size = max(1, batch_size)
        self._concurrency = max(1, concurrency)
        self._on_cpu = on_cpu
        self._on_progress = on_progress

    async def run(self, *, limit: int | None = None) -> EmbedStats:
        """Embed every chunk text that has no vector yet.

        Safe to call repeatedly and safe to interrupt: each batch commits before the next
        starts, and the work list is recomputed from what is actually missing.
        """
        import time

        started = time.monotonic()
        stats = EmbedStats()

        missing = self._index.missing_hashes(limit=limit)
        stats.total_missing = len(missing)
        if not missing:
            stats.duration_s = time.monotonic() - started
            return stats

        texts = self._load_texts(missing)
        batches = [
            [h for h in missing[i : i + self._batch_size] if texts.get(h, "").strip()]
            for i in range(0, len(missing), self._batch_size)
        ]
        stats.skipped_empty = len(missing) - sum(len(b) for b in batches)

        semaphore = asyncio.Semaphore(self._concurrency)

        async def process(batch: list[str]) -> None:
            if not batch:
                return
            async with semaphore:
                await self._embed_batch(batch, texts, stats)

        # Batches run concurrently but each commits on its own, so cancellation mid-run
        # leaves the already-committed ones intact.
        await asyncio.gather(*(process(batch) for batch in batches))

        stats.duration_s = time.monotonic() - started
        return stats

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query, with the instruction template and CPU pinning applied."""
        prepared = apply_query_template(self._profile, query)
        vectors = await self._provider.embed(
            [prepared],
            model=self._model,
            dimensions=self._dimensions,
            on_cpu=self._on_cpu,
        )
        return np.asarray(vectors[0], dtype=np.float32)

    # -------------------------------------------------------------- internals

    async def _embed_batch(self, batch: list[str], texts: dict[str, str], stats: EmbedStats) -> None:
        payload = [apply_document_template(self._profile, texts[h]) for h in batch]

        try:
            vectors = await self._provider.embed(
                payload,
                model=self._model,
                dimensions=self._dimensions,
                on_cpu=self._on_cpu,
            )
        except LLMError as exc:
            # One bad batch must not abandon the rest; the next run retries it, because
            # the work list is derived from what is missing rather than from a cursor.
            stats.failed += len(batch)
            stats.errors.append(str(exc))
            self._report(stats)
            return

        arrays = [np.asarray(vector, dtype=np.float32) for vector in vectors]
        self._index.upsert(dict(zip(batch, arrays, strict=False)))
        self._index.commit()

        stats.embedded += len(batch)
        stats.batches += 1
        self._report(stats)

    def _load_texts(self, hashes: Sequence[str]) -> dict[str, str]:
        """Rebuild the embedding text for each hash.

        **The header must be reconstructed, not skipped.** ``embed_text_hash`` is the hash
        of *header + body*, so embedding the bare chunk text would store a vector under a
        key describing different content — the cache would look full while holding vectors
        for text that was never embedded. The header is rebuilt from the chunk and file
        rows rather than stored, which keeps it out of the database twice.

        Identical embedding text shares a hash by construction, so any representative row
        is correct.
        """
        texts: dict[str, str] = {}
        connection = self._index.connection

        window_size = 500
        for start in range(0, len(hashes), window_size):
            window = hashes[start : start + window_size]
            placeholders = ",".join("?" * len(window))
            rows = connection.execute(
                "SELECT c.embed_text_hash, c.text, c.kind, c.symbol_path, f.path, f.language "  # noqa: S608 - placeholders only
                "FROM chunks c JOIN files f ON f.id = c.file_id "
                f"WHERE c.embed_text_hash IN ({placeholders})",
                tuple(window),
            ).fetchall()

            for row in rows:
                text_hash = str(row[0])
                if text_hash in texts:
                    continue
                header = build_context_header(
                    path=str(row[4]),
                    language=row[5],
                    symbol_path=row[3],
                    kind=str(row[2]),
                )
                texts[text_hash] = compose_embed_text(header, str(row[1]))
        return texts

    def _report(self, stats: EmbedStats) -> None:
        if self._on_progress is not None:
            self._on_progress(stats)
