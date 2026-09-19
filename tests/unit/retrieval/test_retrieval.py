"""Query analysis, fusion and the retrieval engine.

Uses deterministic fake vectors rather than a model, so the whole file runs offline in
milliseconds. Whether embeddings are *semantically* good is the eval's job
(`hearth eval retrieval`); these tests cover the mechanics around them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hearth.indexing.pipeline import Indexer
from hearth.retrieval.engine import RetrievalEngine
from hearth.retrieval.fusion import FusionContext, FusionWeights, fuse
from hearth.retrieval.query_analysis import Intent, analyze
from hearth.retrieval.types import Candidate, Retriever
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate
from hearth.storage.vector_index import NumpyVectorIndex

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "repos"
DIMS = 32


@pytest.fixture
def indexed(tmp_path: Path):
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    Indexer(root=FIXTURES / "py_small", repository=IndexRepository(connection)).run()
    return connection


@pytest.fixture
def engine(indexed) -> RetrievalEngine:
    return RetrievalEngine(
        connection=indexed,
        vector_index=NumpyVectorIndex(indexed, model_id="fake", dims=DIMS),
    )


def candidate(
    chunk_id: int, rank: int, retriever: Retriever, *, path: str = "a.py", kind: str = "function"
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        path=path,
        kind=kind,
        symbol_path=f"sym{chunk_id}",
        start_line=1,
        end_line=2,
        text="body",
        score=1.0,
        rank=rank,
        retriever=retriever,
    )


# ------------------------------------------------------------- query analysis


@pytest.mark.parametrize(
    ("query", "intent"),
    [
        ("how do we handle failed payments?", Intent.CONCEPTUAL),
        ("src/billing/payments.py", Intent.PATH),
        ("where are the tests for invoices", Intent.TEST),
        ("where are invoice totals tested?", Intent.TEST),
        ("check the readme", Intent.DOCS),
        ("is this documented anywhere", Intent.DOCS),
    ],
)
def test_intent_classification(query: str, intent: Intent) -> None:
    assert analyze(query).intent is intent


def test_tested_is_recognised_as_a_test_query() -> None:
    """Regression: exact-word matching missed "tested", so a question about tests got the
    test-file *penalty* — the opposite of what was wanted."""
    assert analyze("where are invoice totals tested?").intent is Intent.TEST
    assert analyze("what does the spec cover").intent is Intent.TEST


def test_identifiers_are_extracted() -> None:
    analysis = analyze("what does InvoiceService.finalize do?")
    assert "InvoiceService.finalize" in analysis.identifiers or "InvoiceService" in analysis.identifiers


def test_prose_words_are_not_identifiers() -> None:
    """Otherwise every question becomes a symbol query and the symbol weight misfires."""
    analysis = analyze("how does the system handle failures")
    assert analysis.identifiers == []


def test_backticked_terms_are_always_identifiers() -> None:
    assert "finalize" in analyze("what does `finalize` do?").identifiers


def test_paths_are_extracted() -> None:
    analysis = analyze("look at src/billing/payments.py please")
    assert analysis.paths == ["src/billing/payments.py"]


def test_language_hints() -> None:
    assert analyze("the TypeScript frontend").languages == ["typescript"]


def test_known_symbols_promote_to_symbol_intent(engine: RetrievalEngine) -> None:
    """A confirmed symbol is the strongest signal available without a model."""
    analysis = engine.analyze_query("what does InvoiceService do?")

    assert "InvoiceService" in analysis.known_symbols
    assert analysis.intent is Intent.SYMBOL


def test_unknown_identifiers_do_not_promote(engine: RetrievalEngine) -> None:
    analysis = engine.analyze_query("what does NoSuchClass do?")

    assert analysis.known_symbols == []
    assert analysis.intent is not Intent.SYMBOL


# -------------------------------------------------------------------- fusion


def test_rrf_rewards_agreement_between_retrievers() -> None:
    """The core claim of hybrid retrieval: two signals agreeing beats one signal alone."""
    agreed = [candidate(1, 0, Retriever.BM25)], [candidate(1, 0, Retriever.DENSE)]
    solo = [candidate(2, 0, Retriever.BM25)]

    fused = fuse([*agreed, solo], limit=5)
    ranked = {r.chunk_id: r.score for r in fused}

    assert ranked[1] > ranked[2]


def test_fusion_uses_best_rank_per_retriever() -> None:
    fused = fuse([[candidate(1, 5, Retriever.BM25), candidate(1, 0, Retriever.BM25)]], limit=5)
    assert fused[0].ranks[Retriever.BM25] == 0


def test_symbol_weight_rises_for_symbol_intent() -> None:
    lists = [[candidate(1, 0, Retriever.SYMBOL)], [candidate(2, 0, Retriever.BM25)]]

    conceptual = {r.chunk_id: r.score for r in fuse(lists, intent=Intent.CONCEPTUAL, limit=5)}
    symbolic = {r.chunk_id: r.score for r in fuse(lists, intent=Intent.SYMBOL, limit=5)}

    assert symbolic[1] / symbolic[2] > conceptual[1] / conceptual[2]


def test_definition_chunks_are_boosted() -> None:
    lists = [
        [candidate(1, 0, Retriever.BM25, kind="class"), candidate(2, 0, Retriever.BM25, kind="preamble")]
    ]
    scores = {r.chunk_id: r.score for r in fuse(lists, limit=5)}

    assert scores[1] > scores[2]


def test_test_files_are_penalised_unless_asked_for() -> None:
    lists = [[candidate(1, 0, Retriever.BM25, path="tests/test_billing.py")]]

    penalised = fuse(lists, intent=Intent.CONCEPTUAL, limit=5)[0]
    exempt = fuse(lists, intent=Intent.TEST, limit=5)[0]

    assert penalised.score < exempt.score
    assert any(label == "test-file" for label, _ in penalised.adjustments)
    assert not any(label == "test-file" for label, _ in exempt.adjustments)


def test_docs_are_penalised_unless_asked_for() -> None:
    lists = [[candidate(1, 0, Retriever.BM25, path="README.md", kind="section")]]

    assert (
        fuse(lists, intent=Intent.CONCEPTUAL, limit=5)[0].score
        < fuse(lists, intent=Intent.DOCS, limit=5)[0].score
    )


def test_pinned_files_outrank_strangers() -> None:
    lists = [[candidate(1, 0, Retriever.BM25, path="a.py"), candidate(2, 0, Retriever.BM25, path="b.py")]]
    context = FusionContext(pinned_paths={"b.py"})

    fused = fuse(lists, context=context, limit=5)
    assert fused[0].path == "b.py"


def test_adjustments_are_recorded_for_explain() -> None:
    lists = [[candidate(1, 0, Retriever.BM25, path="tests/test_x.py", kind="class")]]
    result = fuse(lists, limit=5)[0]

    labels = {label for label, _ in result.adjustments}
    assert {"definition", "test-file"} <= labels


def test_weights_are_configurable() -> None:
    lists = [[candidate(1, 0, Retriever.DENSE)], [candidate(2, 0, Retriever.BM25)]]

    dense_heavy = FusionWeights(dense=5.0, bm25=1.0)
    scores = {r.chunk_id: r.score for r in fuse(lists, weights=dense_heavy, limit=5)}

    assert scores[1] > scores[2]


def test_fusion_of_nothing_is_empty() -> None:
    assert fuse([], limit=5) == []


# -------------------------------------------------------------------- engine


def test_lexical_mode_skips_dense(engine: RetrievalEngine) -> None:
    result = engine.retrieve("InvoiceService", mode="lexical")

    assert result.dense_used is False
    assert result.dense_skipped_reason == "lexical-only requested"
    assert result.results


def test_retrieval_degrades_without_embeddings(engine: RetrievalEngine) -> None:
    """M2 acceptance: search must work while embeddings are still building."""
    result = engine.retrieve("InvoiceService finalize")

    assert result.results, "lexical and symbol retrievers must carry the query"
    assert result.dense_used is False
    assert result.coverage[0] == 0


def test_coverage_is_reported(engine: RetrievalEngine, indexed) -> None:
    """Partial dense results should be visible, not silently worse."""
    result = engine.retrieve("anything")

    embedded, total = result.coverage
    assert total > 0
    assert embedded == 0
    assert result.coverage_fraction == 0.0
    assert result.is_fully_embedded is False


def test_dense_participates_once_embedded(engine: RetrievalEngine, indexed) -> None:
    index = NumpyVectorIndex(indexed, model_id="fake", dims=DIMS)
    rng = np.random.default_rng(7)
    hashes = [str(row[0]) for row in indexed.execute("SELECT DISTINCT embed_text_hash FROM chunks")]
    index.upsert({h: rng.standard_normal(DIMS).astype(np.float32) for h in hashes})

    result = engine.retrieve("anything", query_vector=rng.standard_normal(DIMS).astype(np.float32))

    assert result.dense_used is True
    assert result.is_fully_embedded


def test_diversity_caps_chunks_per_file(engine: RetrievalEngine) -> None:
    """One large file must not monopolise the budget."""
    result = engine.retrieve("invoice", limit=10)

    counts: dict[str, int] = {}
    for found in result.results:
        counts[found.path] = counts.get(found.path, 0) + 1

    assert max(counts.values()) <= 3


def test_explicitly_named_file_is_exempt_from_the_cap(engine: RetrievalEngine) -> None:
    """If someone asks about one file, filling the list from it is the right answer."""
    result = engine.retrieve("src/billing/payments.py", limit=8)
    from_target = [r for r in result.results if r.path == "src/billing/payments.py"]

    assert len(from_target) > 3


def test_results_carry_provenance(engine: RetrievalEngine) -> None:
    result = engine.retrieve("InvoiceService finalize")
    top = result.results[0]

    assert top.retrievers, "every result must record which retrievers found it"
    assert top.citation.startswith("src/billing/invoice_service.py:")


def test_path_query_returns_the_named_file_first(engine: RetrievalEngine) -> None:
    """Regression: ordering path matches by length put errors.py above payments.py."""
    result = engine.retrieve("src/billing/payments.py", limit=5)

    assert result.results[0].path == "src/billing/payments.py"


def test_symbol_query_finds_the_definition(engine: RetrievalEngine) -> None:
    result = engine.retrieve("where is TokenBucket defined?", limit=5)

    assert result.results[0].path == "src/billing/payments.py"


def test_empty_query_does_not_crash(engine: RetrievalEngine) -> None:
    assert engine.retrieve("", limit=5).results == []


def test_query_with_only_punctuation_does_not_crash(engine: RetrievalEngine) -> None:
    assert engine.retrieve("?!?", limit=5) is not None
