"""Vector storage, similarity search and the embedding cache key.

The round-trip test is the important one: ``embed_text_hash`` is the hash of *header +
body*, and the embedder rebuilds that string from the database rather than storing it. If
the two sides ever disagreed the cache would look full while holding vectors for text that
was never embedded — silently, and visible only as poor retrieval.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hearth.indexing.chunker import Chunk
from hearth.indexing.enrich import build_context_header, compose_embed_text, enrich
from hearth.storage.db import connect
from hearth.storage.index_repo import (
    ChunkRecord,
    FilePayload,
    FileRecord,
    IndexRepository,
)
from hearth.storage.migrate import migrate
from hearth.storage.vector_index import NumpyVectorIndex
from hearth.util.hashing import blob_hash

DIMS = 16


@pytest.fixture
def repo(tmp_path: Path) -> IndexRepository:
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    return IndexRepository(connection)


@pytest.fixture
def index(repo: IndexRepository) -> NumpyVectorIndex:
    return NumpyVectorIndex(repo.connection, model_id="test-embed", dims=DIMS)


def vector(seed: int, dims: int = DIMS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dims).astype(np.float32)


def store_chunks(repo: IndexRepository, *hashes: str) -> None:
    repo.replace_files(
        [
            FilePayload(
                file=FileRecord(
                    path="src/a.py",
                    language="python",
                    size_bytes=1,
                    mtime_ns=1,
                    content_hash="h",
                ),
                chunks=[
                    ChunkRecord(
                        kind="function",
                        start_line=i + 1,
                        end_line=i + 2,
                        start_byte=0,
                        end_byte=1,
                        text=f"body {i}",
                        embed_text_hash=text_hash,
                        token_estimate=1,
                    )
                    for i, text_hash in enumerate(hashes)
                ],
            )
        ]
    )


# ------------------------------------------------------------- store and search


def test_upsert_and_count(index: NumpyVectorIndex) -> None:
    assert index.upsert({"h1": vector(1), "h2": vector(2)}) == 2
    assert index.count() == 2


def test_vectors_are_normalized_on_write(index: NumpyVectorIndex) -> None:
    """Search is a plain dot product, which is only cosine similarity if vectors are unit."""
    index.upsert({"h1": vector(1) * 37.0})

    hit = index.search(vector(1), k=1)[0]
    assert hit.score == pytest.approx(1.0, abs=1e-2)


def test_search_ranks_by_similarity(index: NumpyVectorIndex) -> None:
    query = vector(1)
    index.upsert({"same": query, "other": vector(99), "negated": -query})

    hits = index.search(query, k=3)

    assert hits[0].embed_text_hash == "same"
    assert hits[-1].embed_text_hash == "negated"
    assert hits[0].score > hits[-1].score


def test_search_respects_k(index: NumpyVectorIndex) -> None:
    index.upsert({f"h{i}": vector(i) for i in range(20)})
    assert len(index.search(vector(1), k=5)) == 5


def test_search_on_empty_index_returns_nothing(index: NumpyVectorIndex) -> None:
    assert index.search(vector(1), k=5) == []


def test_candidate_filter_restricts_results(index: NumpyVectorIndex) -> None:
    """Masking scores is how language and path prefiltering works, without a second index."""
    index.upsert({f"h{i}": vector(i) for i in range(10)})

    hits = index.search(vector(1), k=10, candidates={"h3", "h7"})

    assert {h.embed_text_hash for h in hits} == {"h3", "h7"}


def test_empty_candidate_set_returns_nothing(index: NumpyVectorIndex) -> None:
    index.upsert({"h1": vector(1)})
    assert index.search(vector(1), k=5, candidates=set()) == []


def test_dimension_mismatch_is_rejected(index: NumpyVectorIndex) -> None:
    with pytest.raises(ValueError, match="dims"):
        index.upsert({"bad": np.ones(DIMS + 1, dtype=np.float32)})

    index.upsert({"ok": vector(1)})
    with pytest.raises(ValueError, match="dims"):
        index.search(np.ones(DIMS + 3, dtype=np.float32))


def test_upsert_replaces_rather_than_duplicates(index: NumpyVectorIndex) -> None:
    index.upsert({"h1": vector(1)})
    index.upsert({"h1": vector(2)})

    assert index.count() == 1
    assert index.search(vector(2), k=1)[0].score == pytest.approx(1.0, abs=1e-2)


def test_matrix_cache_refreshes_after_write(index: NumpyVectorIndex) -> None:
    """The cache is lazy; a stale one would silently hide newly embedded chunks."""
    index.upsert({"h1": vector(1)})
    assert len(index.search(vector(1), k=10)) == 1

    index.upsert({"h2": vector(2)})
    assert len(index.search(vector(1), k=10)) == 2


def test_delete_removes_vectors(index: NumpyVectorIndex) -> None:
    index.upsert({"h1": vector(1), "h2": vector(2)})
    index.delete(["h1"])

    assert {h.embed_text_hash for h in index.search(vector(1), k=10)} == {"h2"}


def test_float16_storage_keeps_similarity_usable(index: NumpyVectorIndex) -> None:
    """float16 halves memory; the precision loss must stay far below retrieval noise."""
    query = vector(5, dims=256)
    big = NumpyVectorIndex(index.connection, model_id="wide", dims=256)
    big.upsert({"target": query})

    assert big.search(query, k=1)[0].score == pytest.approx(1.0, abs=1e-2)


# ----------------------------------------------------------------- coverage


def test_coverage_reports_partial_progress(repo: IndexRepository, index: NumpyVectorIndex) -> None:
    """Search must work while embeddings are still building, and say how far along it is."""
    store_chunks(repo, "a", "b", "c")
    assert index.coverage() == (0, 3)

    index.upsert({"a": vector(1)})
    assert index.coverage() == (1, 3)


def test_missing_hashes_drives_resumption(repo: IndexRepository, index: NumpyVectorIndex) -> None:
    store_chunks(repo, "a", "b", "c")
    index.upsert({"a": vector(1)})

    assert index.missing_hashes() == ["b", "c"]


def test_missing_hashes_is_empty_when_complete(repo: IndexRepository, index: NumpyVectorIndex) -> None:
    store_chunks(repo, "a")
    index.upsert({"a": vector(1)})

    assert index.missing_hashes() == []


def test_vectors_for_another_model_do_not_count(repo: IndexRepository, index: NumpyVectorIndex) -> None:
    """Changing the embedding model must invalidate vectors, not silently reuse them."""
    store_chunks(repo, "a")
    other = NumpyVectorIndex(repo.connection, model_id="different-model", dims=DIMS)
    other.upsert({"a": vector(1)})

    assert index.coverage() == (0, 1)
    assert index.missing_hashes() == ["a"]


def test_prune_orphans_drops_vectors_for_deleted_text(repo: IndexRepository, index: NumpyVectorIndex) -> None:
    """Content-keyed vectors are not cascaded by a file delete, so they need sweeping."""
    store_chunks(repo, "a")
    index.upsert({"a": vector(1), "stale": vector(2)})

    assert index.prune_orphans() == 1
    assert index.count() == 1


# ------------------------------------------------------- the cache-key contract


def test_embed_text_hash_round_trips_between_writer_and_embedder() -> None:
    """The writer's hash and the embedder's reconstruction must agree exactly."""
    chunk = Chunk(
        kind="method",
        text="def finalize(self):\n    return 1",
        start_line=10,
        end_line=11,
        start_byte=0,
        end_byte=30,
        symbol_path="InvoiceService > finalize",
    )

    # What the indexing pipeline stores.
    written = enrich(chunk, path="src/billing/invoice_service.py", language="python")

    # What the embedder rebuilds from the database rows.
    rebuilt_header = build_context_header(
        path="src/billing/invoice_service.py",
        language="python",
        symbol_path="InvoiceService > finalize",
        kind="method",
    )
    rebuilt = compose_embed_text(rebuilt_header, chunk.text)

    assert rebuilt == written.embed_text
    assert blob_hash(rebuilt) == written.embed_text_hash


def test_embed_text_includes_the_context_header() -> None:
    """A short method embedded bare is nearly contentless; scope is what makes it findable."""
    chunk = Chunk(
        kind="method",
        text="def finalize(self): ...",
        start_line=1,
        end_line=1,
        start_byte=0,
        end_byte=10,
        symbol_path="InvoiceService > finalize",
    )
    enriched = enrich(chunk, path="src/billing/invoice_service.py", language="python")

    assert "path: src/billing/invoice_service.py" in enriched.embed_text
    assert "language: python" in enriched.embed_text
    assert "scope: InvoiceService > finalize" in enriched.embed_text
    assert enriched.embed_text.endswith(chunk.text)
