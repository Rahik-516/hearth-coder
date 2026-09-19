"""Context budget allocation (docs/system-design.md §9.1).

``num_ctx`` is split into named segments and held constant for a session. The point is not
tidiness: exceeding ``num_ctx`` makes Ollama truncate silently, taking the system prompt or
tool definitions with it, so the budget has to be enforced *before* sending rather than
discovered afterwards.

Two rules make the table work in practice:

* **Unused budget flows forward.** A short system prompt should let retrieval use the
  slack rather than wasting it.
* **The output reserve is never borrowed from.** It is the room the model needs to answer;
  spending it on context produces a truncated reply, which is worse than a thinner one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Segment(StrEnum):
    """Named parts of a request, in prompt order."""

    SYSTEM = "system"
    PROJECT = "project"
    REPO_MAP = "repo_map"
    HISTORY = "history"
    RETRIEVAL = "retrieval"
    OUTPUT = "output"


#: Per-tier segment sizes, keyed by num_ctx (docs/system-design.md §9.1). The 12K row is
#: the reference dev laptop: a 6 GB GPU running a 4B model.
_TABLES: dict[int, dict[Segment, int]] = {
    12_288: {
        Segment.SYSTEM: 1_500,
        Segment.PROJECT: 400,
        Segment.REPO_MAP: 800,
        Segment.HISTORY: 3_000,
        Segment.RETRIEVAL: 4_000,
        Segment.OUTPUT: 2_500,
    },
    16_384: {
        Segment.SYSTEM: 2_000,
        Segment.PROJECT: 500,
        Segment.REPO_MAP: 1_000,
        Segment.HISTORY: 4_000,
        Segment.RETRIEVAL: 5_000,
        Segment.OUTPUT: 3_500,
    },
    32_768: {
        Segment.SYSTEM: 2_500,
        Segment.PROJECT: 1_000,
        Segment.REPO_MAP: 2_000,
        Segment.HISTORY: 9_000,
        Segment.RETRIEVAL: 12_000,
        Segment.OUTPUT: 5_500,
    },
    65_536: {
        Segment.SYSTEM: 3_000,
        Segment.PROJECT: 1_500,
        Segment.REPO_MAP: 4_000,
        Segment.HISTORY: 20_000,
        Segment.RETRIEVAL: 26_000,
        Segment.OUTPUT: 9_500,
    },
    131_072: {
        Segment.SYSTEM: 3_000,
        Segment.PROJECT: 2_000,
        Segment.REPO_MAP: 6_000,
        Segment.HISTORY: 45_000,
        Segment.RETRIEVAL: 60_000,
        Segment.OUTPUT: 12_000,
    },
}

#: Compaction thresholds, as a fraction of (num_ctx - output reserve). Small windows
#: compact earlier, because a 4B model in 12K has little room to recover from a near-full
#: context (docs/system-design.md §9.1).
COMPACTION_THRESHOLD_SMALL = 0.65
COMPACTION_THRESHOLD_DEFAULT = 0.75
SMALL_CONTEXT_CEILING = 16_384


@dataclass(frozen=True)
class ContextBudget:
    """How many tokens each segment may use."""

    num_ctx: int
    limits: dict[Segment, int]

    def limit(self, segment: Segment) -> int:
        return self.limits.get(segment, 0)

    @property
    def output_reserve(self) -> int:
        return self.limits[Segment.OUTPUT]

    @property
    def input_budget(self) -> int:
        """Everything except the output reserve."""
        return self.num_ctx - self.output_reserve

    @property
    def compaction_threshold(self) -> int:
        """Estimated input size at which compaction should run."""
        fraction = (
            COMPACTION_THRESHOLD_SMALL
            if self.num_ctx <= SMALL_CONTEXT_CEILING
            else COMPACTION_THRESHOLD_DEFAULT
        )
        return int(self.input_budget * fraction)

    def with_slack_from(self, used: dict[Segment, int]) -> ContextBudget:
        """Redistribute unused budget to history and retrieval.

        A short system prompt should make room for more context, not go to waste. The
        output reserve is excluded on both sides: it is never a donor and never a
        recipient.
        """
        donors = (Segment.SYSTEM, Segment.PROJECT, Segment.REPO_MAP)
        slack = sum(max(0, self.limit(s) - used.get(s, 0)) for s in donors)
        if slack <= 0:
            return self

        adjusted = dict(self.limits)
        # Retrieval benefits more than history: fresh context for the current question is
        # usually worth more than older turns.
        adjusted[Segment.RETRIEVAL] += int(slack * 0.6)
        adjusted[Segment.HISTORY] += slack - int(slack * 0.6)
        return ContextBudget(num_ctx=self.num_ctx, limits=adjusted)


def budget_for(num_ctx: int) -> ContextBudget:
    """Budget for a context size, interpolating between the tier rows.

    An unlisted ``num_ctx`` is scaled from the nearest row rather than rejected: users pick
    odd values to fit VRAM, and refusing them would be unhelpful.
    """
    if num_ctx in _TABLES:
        return ContextBudget(num_ctx=num_ctx, limits=dict(_TABLES[num_ctx]))

    nearest = min(_TABLES, key=lambda size: abs(size - num_ctx))
    scale = num_ctx / nearest
    scaled = {segment: max(1, int(value * scale)) for segment, value in _TABLES[nearest].items()}

    # Rounding must never let the segments exceed the window.
    overflow = sum(scaled.values()) - num_ctx
    if overflow > 0:
        scaled[Segment.RETRIEVAL] = max(1, scaled[Segment.RETRIEVAL] - overflow)

    return ContextBudget(num_ctx=num_ctx, limits=scaled)


@dataclass
class BudgetUsage:
    """Actual token use per segment, for `/context` and the stats line."""

    budget: ContextBudget
    used: dict[Segment, int]

    @property
    def total_input(self) -> int:
        return sum(value for segment, value in self.used.items() if segment is not Segment.OUTPUT)

    @property
    def fraction(self) -> float:
        return self.total_input / max(1, self.budget.input_budget)

    @property
    def needs_compaction(self) -> bool:
        return self.total_input >= self.budget.compaction_threshold

    def over_budget(self) -> list[Segment]:
        return [s for s, value in self.used.items() if value > self.budget.limit(s)]

    def rows(self) -> list[tuple[Segment, int, int]]:
        """``(segment, used, limit)`` in prompt order, for rendering."""
        return [(s, self.used.get(s, 0), self.budget.limit(s)) for s in Segment]
