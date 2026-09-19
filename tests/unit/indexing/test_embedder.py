"""Embedder behaviour: caching, resumability and prompt templates.

The resumability tests back an M2 acceptance criterion: *"Interrupting embedding with
Ctrl+C and resuming re-embeds nothing that was already committed."* That property comes
from two decisions — work is derived from what is *missing* rather than from a saved
cursor, and each batch commits before the next begins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hearth.indexing.embedder import (
    Embedder,
    apply_document_template,
    apply_query_template,
)
from hearth.indexing.pipeline import Indexer
from hearth.llm.errors import ProviderUnavailableError
from hearth.llm.profiles import ProfileRegistry
from hearth.llm.scripted_provider import ScriptedProvider
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate
from hearth.storage.vector_index import NumpyVectorIndex

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "repos"
DIMS = 64


class CountingProvider(ScriptedProvider):
    """Records how many texts were embedded, and can fail on demand."""

    def __init__(self, *, fail_after: int | None = None) -> None:
        super().__init__()
        self.embedded_texts: list[str] = []
        self.batch_count = 0
        self._fail_after = fail_after

    async def embed(self, texts, *, model, dimensions=None, on_cpu=False):
        self.batch_count += 1
        if self._fail_after is not None and self.batch_count > self._fail_after:
            raise ProviderUnavailableError("simulated outage")
        self.embedded_texts.extend(texts)
        return await super().embed(texts, model=model, dimensions=dimensions, on_cpu=on_cpu)


@pytest.fixture
def indexed(tmp_path: Path):
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    Indexer(root=FIXTURES / "py_small", repository=IndexRepository(connection)).run()
    return connection


@pytest.fixture
def index(indexed) -> NumpyVectorIndex:
    return NumpyVectorIndex(indexed, model_id="test-embed", dims=DIMS)


def make_embedder(provider, index: NumpyVectorIndex, **kwargs) -> Embedder:
    return Embedder(
        provider=provider,
        vector_index=index,
        model="test-embed",
        dimensions=DIMS,
        **kwargs,
    )


# ------------------------------------------------------------------ templates


def test_query_template_is_applied() -> None:
    """Qwen3-Embedding wants an instruction prefix on queries but not documents."""
    profile = ProfileRegistry.load().for_model("qwen3-embedding:0.6b")

    prepared = apply_query_template(profile, "how do we throttle calls?")

    assert "Instruct:" in prepared
    assert "how do we throttle calls?" in prepared


def test_document_template_leaves_text_alone() -> None:
    profile = ProfileRegistry.load().for_model("qwen3-embedding:0.6b")

    assert apply_document_template(profile, "def f(): ...") == "def f(): ..."


def test_missing_profile_is_a_passthrough() -> None:
    assert apply_query_template(None, "text") == "text"
    assert apply_document_template(None, "text") == "text"


# ------------------------------------------------------------------- embedding


async def test_embeds_every_missing_chunk(index: NumpyVectorIndex) -> None:
    provider = CountingProvider()
    stats = await make_embedder(provider, index).run()

    assert stats.embedded == stats.total_missing > 0
    assert stats.failed == 0
    assert index.coverage()[0] == index.coverage()[1]


async def test_embedded_text_carries_the_context_header(index: NumpyVectorIndex) -> None:
    """The cache key is the hash of header+body, so the header must be what is sent."""
    provider = CountingProvider()
    await make_embedder(provider, index).run()

    assert any("path: src/billing/" in text for text in provider.embedded_texts)
    assert any("language: python" in text for text in provider.embedded_texts)


async def test_rerun_embeds_nothing(index: NumpyVectorIndex) -> None:
    """M2 acceptance: resuming must re-embed nothing already committed."""
    await make_embedder(CountingProvider(), index).run()

    second = CountingProvider()
    stats = await make_embedder(second, index).run()

    assert stats.total_missing == 0
    assert stats.embedded == 0
    assert second.embedded_texts == []


async def test_partial_run_resumes_where_it_stopped(index: NumpyVectorIndex) -> None:
    """The interrupted-embedding case, without needing an actual Ctrl+C."""
    total = len(index.missing_hashes())
    assert total > 8

    first = await make_embedder(CountingProvider(), index, batch_size=4).run(limit=8)
    assert first.embedded == 8

    provider = CountingProvider()
    second = await make_embedder(provider, index, batch_size=4).run()

    assert second.total_missing == total - 8
    assert second.embedded == total - 8
    assert index.coverage()[0] == index.coverage()[1]


async def test_failed_batch_does_not_abandon_the_rest(index: NumpyVectorIndex) -> None:
    """One provider error must not lose the batches that already succeeded."""
    provider = CountingProvider(fail_after=1)
    stats = await make_embedder(provider, index, batch_size=4, concurrency=1).run()

    assert stats.embedded > 0
    assert stats.failed > 0
    assert stats.errors
    assert not stats.complete


async def test_failed_work_is_retried_on_the_next_run(index: NumpyVectorIndex) -> None:
    """Work is derived from what is missing, so failures need no separate bookkeeping."""
    await make_embedder(CountingProvider(fail_after=1), index, batch_size=4, concurrency=1).run()
    before = len(index.missing_hashes())
    assert before > 0

    stats = await make_embedder(CountingProvider(), index, batch_size=4).run()

    assert stats.embedded == before
    assert index.missing_hashes() == []


async def test_identical_text_is_embedded_once(tmp_path: Path) -> None:
    """Content-keyed caching is what makes a branch switch or a moved file nearly free."""
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    repo = IndexRepository(connection)

    from hearth.storage.index_repo import ChunkRecord, FilePayload, FileRecord

    shared = ChunkRecord(
        kind="function",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=5,
        text="def f(): ...",
        embed_text_hash="same-hash",
        token_estimate=4,
    )
    repo.replace_files(
        [
            FilePayload(
                file=FileRecord(path=p, language="python", size_bytes=1, mtime_ns=1, content_hash=p),
                chunks=[shared],
            )
            for p in ("a.py", "b.py", "c.py")
        ]
    )

    index = NumpyVectorIndex(connection, model_id="test-embed", dims=DIMS)
    provider = CountingProvider()
    stats = await make_embedder(provider, index).run()

    assert stats.embedded == 1, "three files, one distinct text, one embedding"
    assert len(provider.embedded_texts) == 1


async def test_cpu_placement_is_forwarded(index: NumpyVectorIndex) -> None:
    """Query embedding during a session must not evict the chat model from a small GPU."""
    provider = CountingProvider()
    await make_embedder(provider, index, on_cpu=True).run(limit=2)

    assert provider.embed_calls
    assert all(call[3] is True for call in provider.embed_calls)


async def test_embed_query_applies_the_template(index: NumpyVectorIndex) -> None:
    provider = CountingProvider()
    embedder = make_embedder(
        provider,
        index,
        profile=ProfileRegistry.load().for_model("qwen3-embedding:0.6b"),
    )

    vector = await embedder.embed_query("how do we throttle calls?")

    assert vector.shape == (DIMS,)
    assert "Instruct:" in provider.embedded_texts[0]


async def test_vectors_are_stored_normalized(index: NumpyVectorIndex) -> None:
    await make_embedder(CountingProvider(), index).run(limit=4)

    matrix, _ = index._ensure_matrix()
    norms = np.linalg.norm(matrix, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-2)
