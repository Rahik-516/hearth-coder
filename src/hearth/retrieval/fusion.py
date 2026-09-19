"""Weighted Reciprocal Rank Fusion (docs/system-design.md §7.3).

    score(chunk) = Sum_r  w_r / (k + rank_r(chunk))          k = 60

RRF fuses **ranks, not scores**, which is the point: BM25 returns unbounded negative
numbers, cosine similarity returns 0 to 1, and symbol matching returns something arbitrary.
Normalising those onto a common scale means inventing a calibration that no data supports.
Ranks are directly comparable, and a chunk found by several retrievers accumulates
contributions — which is precisely the signal hybrid retrieval exists to capture.

Boosts and penalties are multiplicative and applied after fusion. Every number here is a
default to be tuned against the eval, not a truth.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from hearth.retrieval.query_analysis import Intent
from hearth.retrieval.types import Candidate, FusedResult, Retriever

#: The RRF constant. Larger values flatten the curve, so rank 1 and rank 10 differ less.
DEFAULT_RRF_K = 60


@dataclass
class FusionWeights:
    """Per-retriever weights and the post-fusion adjustments."""

    dense: float = 1.0
    bm25: float = 1.0
    symbol: float = 1.5
    path: float = 0.8

    #: When the query names a symbol that exists, that signal dominates.
    symbol_intent_boost: float = 2.5

    # Multiplicative adjustments.
    definition_boost: float = 1.1
    pinned_boost: float = 1.5
    edited_boost: float = 1.3
    test_penalty: float = 0.6
    generated_penalty: float = 0.3
    docs_penalty: float = 0.8

    def for_retriever(self, retriever: Retriever, intent: Intent) -> float:
        if retriever is Retriever.SYMBOL:
            return self.symbol_intent_boost if intent is Intent.SYMBOL else self.symbol
        return {
            Retriever.DENSE: self.dense,
            Retriever.BM25: self.bm25,
            Retriever.PATH: self.path,
        }[retriever]


@dataclass
class FusionContext:
    """Session state that influences ranking.

    Empty by default. `hearth search` has no conversation; the agent loop will populate
    these so files the user pinned or just edited rank above equally-relevant strangers.
    """

    pinned_paths: set[str] = field(default_factory=set)
    edited_paths: set[str] = field(default_factory=set)
    generated_paths: set[str] = field(default_factory=set)


#: Definition-bearing chunk kinds. A definition outranks a chunk that merely mentions a
#: name, all else being equal.
_DEFINITION_KINDS = frozenset({"class", "method", "function", "class_skeleton"})

#: Path fragments that mark a file as tests. Cheap and imperfect, but tests are numerous
#: and usually not the answer — unless the question is about tests, hence the intent check.
_TEST_MARKERS = ("test_", "_test.", "/tests/", "/test/", ".test.", ".spec.", "spec_")

_DOC_EXTENSIONS = (".md", ".rst", ".txt", ".markdown")


def fuse(
    candidate_lists: Iterable[Sequence[Candidate]],
    *,
    intent: Intent = Intent.CONCEPTUAL,
    weights: FusionWeights | None = None,
    context: FusionContext | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    limit: int = 20,
) -> list[FusedResult]:
    """Fuse ranked candidate lists into one ordered result list."""
    active_weights = weights or FusionWeights()
    active_context = context or FusionContext()

    fused: dict[int, FusedResult] = {}

    for candidates in candidate_lists:
        for candidate in candidates:
            result = fused.get(candidate.chunk_id)
            if result is None:
                result = FusedResult(
                    chunk_id=candidate.chunk_id,
                    path=candidate.path,
                    kind=candidate.kind,
                    symbol_path=candidate.symbol_path,
                    start_line=candidate.start_line,
                    end_line=candidate.end_line,
                    text=candidate.text,
                    language=candidate.language,
                )
                fused[candidate.chunk_id] = result

            weight = active_weights.for_retriever(candidate.retriever, intent)
            contribution = weight / (rrf_k + candidate.rank + 1)

            # A retriever can surface the same chunk more than once; keep its best rank.
            previous = result.ranks.get(candidate.retriever)
            if previous is None or candidate.rank < previous:
                result.ranks[candidate.retriever] = candidate.rank
                result.contributions[candidate.retriever] = contribution

    for result in fused.values():
        result.score = sum(result.contributions.values())
        _apply_adjustments(result, intent=intent, weights=active_weights, context=active_context)

    ordered = sorted(fused.values(), key=lambda r: r.score, reverse=True)
    return ordered[:limit]


def _apply_adjustments(
    result: FusedResult,
    *,
    intent: Intent,
    weights: FusionWeights,
    context: FusionContext,
) -> None:
    """Apply multiplicative boosts and penalties, recording each for `--explain`."""
    path = result.path
    lowered = path.lower()

    if path in context.pinned_paths:
        _adjust(result, "pinned", weights.pinned_boost)
    if path in context.edited_paths:
        _adjust(result, "edited-this-session", weights.edited_boost)
    if result.kind in _DEFINITION_KINDS:
        _adjust(result, "definition", weights.definition_boost)

    if path in context.generated_paths:
        _adjust(result, "generated", weights.generated_penalty)

    if any(marker in lowered for marker in _TEST_MARKERS) and intent is not Intent.TEST:
        _adjust(result, "test-file", weights.test_penalty)

    if lowered.endswith(_DOC_EXTENSIONS) and intent is not Intent.DOCS:
        _adjust(result, "docs", weights.docs_penalty)


def _adjust(result: FusedResult, label: str, factor: float) -> None:
    if factor == 1.0:
        return
    result.score *= factor
    result.adjustments.append((label, factor))
