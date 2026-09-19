"""Retrieval evaluation harness.

Every later tuning decision — fusion weights, chunk sizes, embedding model — is settled by
this, not by intuition (docs/implementation-roadmap.md §0, principle 3). Vendor benchmarks
do not tell you how a model ranks *your* code.

Metrics are recall@5, recall@10 and MRR. Each question declares the files (and optionally
symbols) that should be found; a result counts as a hit when it comes from an expected
file, or names an expected symbol.

Runs each mode separately — lexical-only, dense-only, hybrid — because the acceptance bar
is not just "hybrid is good enough" but "hybrid beats both of its parts". A hybrid that
merely matched lexical-only would mean the dense half was paying for itself in latency and
memory while contributing nothing.

YAML is read through the ``evals`` optional extra (ADR 0003); ``import yaml`` is lazy so a
default install never needs it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hearth.retrieval.engine import RetrievalEngine
from hearth.retrieval.types import FusedResult

#: Recall depths reported. @1 and @3 matter most on small fixtures: with ten files,
#: recall@10 is near 1.0 for any working retriever and cannot distinguish two modes.
RECALL_DEPTHS = (1, 3, 5, 10)

#: Embeds a query string for dense retrieval.
QueryEmbedder = Callable[[str], np.ndarray]


class EvalDependencyError(RuntimeError):
    """The evals extra is not installed."""


class EvalDataError(ValueError):
    """A question set is malformed."""


@dataclass(frozen=True)
class EvalQuestion:
    """One question and what should be found for it."""

    question: str
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    kind: str = "conceptual"
    note: str | None = None

    def is_hit(self, result: FusedResult) -> bool:
        """Whether a result answers this question.

        A file match counts on its own: chunk boundaries move as the chunker evolves, so
        pinning an expected line range would make the eval measure the chunker rather than
        retrieval.
        """
        if any(result.path == expected or result.path.endswith(expected) for expected in self.files):
            return True
        if self.symbols and result.symbol_path:
            leaf = result.symbol_path.split(">")[-1].strip()
            return any(leaf == symbol or symbol in result.symbol_path for symbol in self.symbols)
        return False


@dataclass
class QuestionOutcome:
    question: EvalQuestion
    first_hit_rank: int | None
    hit_depths: dict[int, bool] = field(default_factory=dict)
    top_paths: list[str] = field(default_factory=list)

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.first_hit_rank is None else 1.0 / (self.first_hit_rank + 1)


@dataclass
class ModeReport:
    """Aggregate metrics for one retrieval mode."""

    mode: str
    outcomes: list[QuestionOutcome] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.outcomes)

    def recall_at(self, depth: int) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o.hit_depths.get(depth)) / len(self.outcomes)

    @property
    def mrr(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.reciprocal_rank for o in self.outcomes) / len(self.outcomes)

    @property
    def misses(self) -> list[QuestionOutcome]:
        """Questions with no hit in the top 10. The list worth reading after a run."""
        return [o for o in self.outcomes if not o.hit_depths.get(10)]

    def mrr_for_kind(self, kind: str) -> float:
        """MRR restricted to one question kind.

        The aggregate hides the trade that matters: dense should win on conceptual
        questions and lose on exact-name ones. On a small fixture only this split shows it.
        """
        items = [o for o in self.outcomes if o.question.kind == kind]
        if not items:
            return 0.0
        return sum(o.reciprocal_rank for o in items) / len(items)

    def by_kind(self) -> dict[str, float]:
        """recall@10 split by question kind, which is where regressions localise."""
        grouped: dict[str, list[QuestionOutcome]] = {}
        for outcome in self.outcomes:
            grouped.setdefault(outcome.question.kind, []).append(outcome)
        return {
            kind: sum(1 for o in items if o.hit_depths.get(10)) / len(items)
            for kind, items in sorted(grouped.items())
        }


@dataclass
class EvalReport:
    """One eval set across every mode."""

    name: str
    modes: dict[str, ModeReport] = field(default_factory=dict)
    coverage: tuple[int, int] = (0, 0)

    @property
    def hybrid(self) -> ModeReport | None:
        return self.modes.get("hybrid")

    def hybrid_beats_parts(self) -> bool:
        """The acceptance bar: hybrid must beat both single-signal baselines."""
        hybrid = self.modes.get("hybrid")
        lexical = self.modes.get("lexical")
        dense = self.modes.get("dense")
        if hybrid is None or lexical is None or dense is None:
            return False
        return (
            hybrid.recall_at(10) >= lexical.recall_at(10)
            and hybrid.recall_at(10) >= dense.recall_at(10)
            and (hybrid.recall_at(10) > lexical.recall_at(10) or hybrid.mrr > lexical.mrr)
        )


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read a YAML question set.

    Raises:
        EvalDependencyError: when the ``evals`` extra is not installed.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on install shape
        raise EvalDependencyError(
            "Eval data requires the evals extra. Install with: uv sync --extra evals"
        ) from exc

    # safe_load, always: eval files are repository data, and the default loader
    # constructs arbitrary Python objects (ADR 0003).
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        # A malformed question set should name the file and the problem, not produce a
        # parser traceback the reader has to decode.
        raise EvalDataError(f"{path}: invalid YAML: {exc}") from exc

    entries = raw.get("questions", []) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise EvalDataError(f"{path}: expected a list of questions")

    questions: list[EvalQuestion] = []
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or "question" not in entry:
            raise EvalDataError(f"{path}: entry {position} has no 'question' field")
        questions.append(
            EvalQuestion(
                question=str(entry["question"]),
                files=tuple(entry.get("files", []) or ()),
                symbols=tuple(entry.get("symbols", []) or ()),
                kind=str(entry.get("kind", "conceptual")),
                note=entry.get("note"),
            )
        )
    return questions


def evaluate_mode(
    engine: RetrievalEngine,
    questions: Sequence[EvalQuestion],
    *,
    mode: str,
    embed_query: QueryEmbedder | None = None,
    limit: int = 10,
) -> ModeReport:
    """Run one mode over a question set."""
    import time

    report = ModeReport(mode=mode)
    started = time.perf_counter()

    for question in questions:
        vector = embed_query(question.question) if embed_query is not None else None
        result = engine.retrieve(
            question.question,
            limit=limit,
            query_vector=vector,
            mode=mode,  # type: ignore[arg-type]
        )

        outcome = QuestionOutcome(
            question=question,
            first_hit_rank=None,
            top_paths=[r.path for r in result.results[:5]],
        )
        for rank, found in enumerate(result.results):
            if question.is_hit(found):
                outcome.first_hit_rank = rank
                break

        for depth in RECALL_DEPTHS:
            outcome.hit_depths[depth] = outcome.first_hit_rank is not None and outcome.first_hit_rank < depth
        report.outcomes.append(outcome)

    report.duration_ms = (time.perf_counter() - started) * 1000
    return report


def format_report(report: EvalReport, *, show_misses: bool = True) -> str:
    """Render a report as plain text, modes side by side."""
    lines = [f"Eval: {report.name}"]

    embedded, total = report.coverage
    if total:
        lines.append(f"Embedding coverage: {embedded}/{total} ({embedded / total:.0%})")
    lines.append("")

    header = f"{'mode':<10}{'r@1':>7}{'r@3':>7}{'r@5':>7}{'r@10':>7}{'MRR':>8}{'ms':>9}"
    lines.append(header)
    lines.append("-" * len(header))

    for mode in ("lexical", "dense", "hybrid"):
        entry = report.modes.get(mode)
        if entry is None:
            continue
        lines.append(
            f"{mode:<10}{entry.recall_at(1):>7.2f}{entry.recall_at(3):>7.2f}"
            f"{entry.recall_at(5):>7.2f}{entry.recall_at(10):>7.2f}"
            f"{entry.mrr:>8.3f}{entry.duration_ms:>9.0f}"
        )

    hybrid = report.hybrid
    if hybrid is not None:
        lines.append("")
        lines.append("MRR by question kind (this is where the modes actually differ):")
        kinds = sorted({o.question.kind for o in hybrid.outcomes})
        lines.append(f"  {'kind':<14}" + "".join(f"{m:>10}" for m in ("lexical", "dense", "hybrid")))
        for kind in kinds:
            row = f"  {kind:<14}"
            for mode in ("lexical", "dense", "hybrid"):
                entry = report.modes.get(mode)
                row += f"{entry.mrr_for_kind(kind):>10.3f}" if entry else f"{'-':>10}"
            lines.append(row)

        if show_misses and hybrid.misses:
            lines.append("")
            lines.append(f"Missed ({len(hybrid.misses)}):")
            for outcome in hybrid.misses:
                lines.append(f"  · {outcome.question.question}")
                lines.append(f"      expected: {', '.join(outcome.question.files) or '-'}")
                lines.append(f"      got:      {', '.join(outcome.top_paths[:3]) or '-'}")

    return "\n".join(lines)


def symbol_exists(connection: sqlite3.Connection, name: str) -> bool:
    """Whether a symbol is in the index. Used to validate question sets."""
    row = connection.execute(
        "SELECT 1 FROM symbols WHERE name = ? COLLATE NOCASE LIMIT 1", (name,)
    ).fetchone()
    return row is not None
