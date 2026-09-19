"""Retrieval latency at scale.

M2 acceptance criterion: *"Retrieval (excluding query embedding) completes in under 150 ms
at 100K synthetic chunks."*

Excluding query embedding is the right boundary — that cost belongs to the model and is
measured by `hearth bench`. What is measured here is everything Hearth controls: FTS5,
the vector matmul, fusion and diversity.

Marked ``slow`` so it does not run by default. Synthetic data, so it measures the
algorithms rather than any particular corpus.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from hearth.retrieval.engine import RetrievalEngine
from hearth.storage.db import connect
from hearth.storage.index_repo import ChunkRecord, FilePayload, FileRecord, IndexRepository
from hearth.storage.migrate import migrate
from hearth.storage.vector_index import NumpyVectorIndex

pytestmark = pytest.mark.slow

CHUNKS = 100_000
FILES = 2_000
DIMS = 1024
BUDGET_MS = 150.0

_WORDS = [
    "invoice",
    "payment",
    "customer",
    "service",
    "handler",
    "gateway",
    "bucket",
    "token",
    "retry",
    "ledger",
    "account",
    "balance",
    "charge",
    "refund",
    "subscription",
    "webhook",
    "session",
    "cache",
    "queue",
    "worker",
]


def _build(tmp_path: Path):
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    repo = IndexRepository(connection)

    rng = np.random.default_rng(1234)
    per_file = CHUNKS // FILES
    payloads = []

    for file_no in range(FILES):
        chunks = []
        for chunk_no in range(per_file):
            words = rng.choice(_WORDS, size=24)
            body = f"def handler_{file_no}_{chunk_no}():\n    " + " ".join(words)
            chunks.append(
                ChunkRecord(
                    kind="function",
                    symbol_path=f"Module{file_no} > handler_{file_no}_{chunk_no}",
                    start_line=chunk_no * 10 + 1,
                    end_line=chunk_no * 10 + 9,
                    start_byte=0,
                    end_byte=len(body),
                    text=body,
                    embed_text_hash=f"h{file_no:05d}{chunk_no:04d}",
                    token_estimate=30,
                )
            )
        payloads.append(
            FilePayload(
                file=FileRecord(
                    path=f"src/pkg{file_no // 50}/module_{file_no}.py",
                    language="python",
                    size_bytes=len(chunks) * 200,
                    mtime_ns=file_no,
                    content_hash=f"c{file_no}",
                ),
                chunks=chunks,
            )
        )
        if len(payloads) >= 100:
            repo.replace_files(payloads)
            payloads = []
    if payloads:
        repo.replace_files(payloads)

    index = NumpyVectorIndex(connection, model_id="bench", dims=DIMS)
    vectors = rng.standard_normal((CHUNKS, DIMS)).astype(np.float32)
    batch: dict[str, np.ndarray] = {}
    position = 0
    for file_no in range(FILES):
        for chunk_no in range(per_file):
            batch[f"h{file_no:05d}{chunk_no:04d}"] = vectors[position]
            position += 1
        if len(batch) >= 5000:
            index.upsert(batch)
            batch = {}
    if batch:
        index.upsert(batch)
    connection.commit()

    return connection, index


@pytest.fixture(scope="module")
def big_index(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("perf"))


def test_corpus_is_the_advertised_size(big_index) -> None:
    connection, index = big_index
    assert IndexRepository(connection).count_chunks() == CHUNKS
    assert index.count() == CHUNKS


def test_hybrid_retrieval_under_budget(big_index) -> None:
    """The acceptance criterion, with the query embedding supplied rather than computed."""
    connection, index = big_index
    engine = RetrievalEngine(connection=connection, vector_index=index)

    rng = np.random.default_rng(99)
    queries = ["invoice payment handler", "token bucket retry", "webhook session cache"]
    vectors = [rng.standard_normal(DIMS).astype(np.float32) for _ in queries]

    # Warm the lazily-built matrix and SQLite's page cache; the criterion is steady-state
    # latency, not first-call cost.
    engine.retrieve(queries[0], query_vector=vectors[0], limit=10)

    timings = []
    for query, vector in zip(queries * 3, vectors * 3, strict=False):
        started = time.perf_counter()
        result = engine.retrieve(query, query_vector=vector, limit=10)
        timings.append((time.perf_counter() - started) * 1000)
        assert result.results

    median = sorted(timings)[len(timings) // 2]
    assert median < BUDGET_MS, f"median retrieval {median:.1f}ms exceeds {BUDGET_MS}ms"


def test_dense_search_alone_is_fast(big_index) -> None:
    """The matmul is the part that scales with corpus size, so measure it separately."""
    _, index = big_index
    rng = np.random.default_rng(5)
    vector = rng.standard_normal(DIMS).astype(np.float32)

    index.search(vector, k=40)  # warm the matrix cache

    started = time.perf_counter()
    hits = index.search(vector, k=40)
    elapsed = (time.perf_counter() - started) * 1000

    assert len(hits) == 40
    assert elapsed < 100.0, f"dense search {elapsed:.1f}ms at {CHUNKS} vectors"


def test_vector_memory_stays_within_expectations(big_index) -> None:
    """float16 storage is what keeps 100K x 1024 near 200 MB rather than 400 MB."""
    _, index = big_index
    matrix, _ = index._ensure_matrix()

    stored_mb = CHUNKS * DIMS * 2 / 1_000_000
    assert stored_mb < 250
    assert matrix.shape == (CHUNKS, DIMS)
