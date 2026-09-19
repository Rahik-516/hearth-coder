"""Token estimation, budgets and prompt assembly.

The prefix-stability tests matter most. Breaking the cached prefix costs seconds of
prefill per turn on a CPU-split model and produces no error — the session just gets slow,
and nothing points at the cause (docs/system-design.md §9.2).
"""

from __future__ import annotations

import pytest

from hearth.core.context.budget import Segment, budget_for
from hearth.core.context.builder import CONTEXT_CLOSE, CONTEXT_OPEN, ContextBuilder
from hearth.core.context.tokens import (
    DEFAULT_CHARS_PER_TOKEN,
    TokenEstimator,
    observe,
)
from hearth.llm.types import Message
from hearth.retrieval.types import FusedResult


def result(path: str, *, text: str = "def f():\n    return 1", start: int = 1, end: int = 2) -> FusedResult:
    return FusedResult(
        chunk_id=hash(path) % 10_000,
        path=path,
        kind="function",
        symbol_path="f",
        start_line=start,
        end_line=end,
        text=text,
        language="python",
    )


@pytest.fixture
def builder() -> ContextBuilder:
    return ContextBuilder(budget=budget_for(12_288), estimator=TokenEstimator())


# ---------------------------------------------------------------- estimation


def test_estimate_scales_with_length() -> None:
    estimator = TokenEstimator()
    assert estimator.estimate("") == 0
    assert estimator.estimate("x") >= 1
    assert estimator.estimate("x" * 360) > estimator.estimate("x" * 36)


def test_calibration_moves_toward_observed_ratio() -> None:
    """The estimator learns the loaded model's tokenization from reported counts."""
    estimator = TokenEstimator()
    before = estimator.chars_per_token

    # Server counted 25% more tokens than estimated: each token covers fewer characters.
    estimator.calibrate(estimated=1000, actual=1250)

    assert estimator.chars_per_token < before
    assert estimator.is_calibrated


def test_calibration_converges() -> None:
    estimator = TokenEstimator()
    for _ in range(10):
        estimated = estimator.estimate("x" * 3600)
        estimator.calibrate(estimated=estimated, actual=1200)

    # 3600 chars reported as 1200 tokens => 3.0 chars/token.
    assert estimator.chars_per_token == pytest.approx(3.0, abs=0.2)


def test_implausible_ratios_are_rejected() -> None:
    """A truncated prompt reports far fewer tokens; absorbing that would poison the model."""
    estimator = TokenEstimator()
    before = estimator.chars_per_token

    assert estimator.calibrate(estimated=1000, actual=100) is False
    assert estimator.calibrate(estimated=1000, actual=9000) is False
    assert estimator.chars_per_token == before


def test_truncation_is_detected() -> None:
    estimator = TokenEstimator()
    assert estimator.looks_truncated(estimated=10_000, actual=4_000) is True
    assert estimator.looks_truncated(estimated=10_000, actual=9_900) is False


def test_over_reporting_is_not_truncation() -> None:
    """More tokens than estimated is a calibration matter, not lost content."""
    estimator = TokenEstimator()
    assert estimator.looks_truncated(estimated=1_000, actual=1_400) is False


def test_observe_reports_and_does_not_calibrate_on_truncation() -> None:
    estimator = TokenEstimator()
    report = observe(estimator, estimated=10_000, actual=3_000)

    assert report.truncated is True
    assert report.calibrated is False
    assert report.shortfall == 7_000
    assert "truncated" in report.message()
    assert estimator.is_calibrated is False


def test_reset_restores_defaults() -> None:
    estimator = TokenEstimator()
    estimator.calibrate(estimated=100, actual=120)
    estimator.reset()

    assert estimator.chars_per_token == DEFAULT_CHARS_PER_TOKEN
    assert estimator.is_calibrated is False


# ------------------------------------------------------------------- budgets


def test_documented_tier_table() -> None:
    """The 12K row is the reference dev laptop (docs/system-design.md §9.1)."""
    budget = budget_for(12_288)

    assert budget.limit(Segment.SYSTEM) == 1_500
    assert budget.limit(Segment.RETRIEVAL) == 4_000
    assert budget.limit(Segment.OUTPUT) == 2_500


def test_segments_fit_the_window() -> None:
    for num_ctx in (12_288, 16_384, 32_768, 65_536, 131_072):
        budget = budget_for(num_ctx)
        assert sum(budget.limits.values()) <= num_ctx


def test_unlisted_context_size_is_scaled() -> None:
    budget = budget_for(20_000)

    assert sum(budget.limits.values()) <= 20_000
    assert budget.limit(Segment.RETRIEVAL) > 0


def test_small_windows_compact_earlier() -> None:
    """A 4B model in 12K has little room to recover from a nearly full context."""
    small = budget_for(12_288)
    large = budget_for(65_536)

    assert small.compaction_threshold / small.input_budget < (large.compaction_threshold / large.input_budget)


def test_slack_flows_to_retrieval_and_history() -> None:
    budget = budget_for(12_288)
    used = {Segment.SYSTEM: 500, Segment.PROJECT: 0, Segment.REPO_MAP: 0}

    adjusted = budget.with_slack_from(used)

    assert adjusted.limit(Segment.RETRIEVAL) > budget.limit(Segment.RETRIEVAL)
    assert adjusted.limit(Segment.HISTORY) > budget.limit(Segment.HISTORY)


def test_output_reserve_is_never_borrowed() -> None:
    """Spending the reserve yields a truncated answer, which is worse than thinner context."""
    budget = budget_for(12_288)
    adjusted = budget.with_slack_from({Segment.SYSTEM: 0, Segment.PROJECT: 0, Segment.REPO_MAP: 0})

    assert adjusted.limit(Segment.OUTPUT) == budget.limit(Segment.OUTPUT)


# ------------------------------------------------------------------- layout


def test_message_order_is_cache_stable(builder: ContextBuilder) -> None:
    """system, system, history…, user — the order the cache depends on."""
    built = builder.build(
        system_prompt="core rules",
        project_instructions="project conventions",
        user_message="what does this do?",
        history=[Message(role="user", content="earlier"), Message(role="assistant", content="reply")],
    )
    roles = [m.role for m in built.messages]

    assert roles[0] == "system"
    assert roles[1] == "system"
    assert roles[-1] == "user"
    assert roles[2:-1] == ["user", "assistant"]


def test_prefix_is_identical_across_turns(builder: ContextBuilder) -> None:
    """The whole point: two turns in a session must share a byte-identical prefix."""
    first = builder.build(system_prompt="core rules", user_message="question one")
    second = builder.build(
        system_prompt="core rules",
        user_message="question two",
        history=[Message(role="user", content="question one"), Message(role="assistant", content="a")],
    )

    assert [m.content for m in first.stable_prefix] == [m.content for m in second.stable_prefix]


def test_retrieved_context_lives_in_the_user_message(builder: ContextBuilder) -> None:
    """Non-leading system messages are handled inconsistently across chat templates."""
    built = builder.build(
        system_prompt="core",
        user_message="where is f?",
        retrieved=[result("src/a.py")],
    )

    assert built.messages[-1].role == "user"
    assert CONTEXT_OPEN in built.messages[-1].content
    assert not any(CONTEXT_OPEN in m.content for m in built.messages if m.role == "system")


def test_context_block_is_framed_as_untrusted(builder: ContextBuilder) -> None:
    """First layer of prompt-injection defence: the model is told this is data."""
    built = builder.build(system_prompt="core", user_message="q", retrieved=[result("src/a.py")])
    content = built.messages[-1].content

    assert 'trust="untrusted-data"' in content
    assert content.count(CONTEXT_OPEN) == 1
    assert content.count(CONTEXT_CLOSE) == 1


def test_user_question_follows_the_context(builder: ContextBuilder) -> None:
    built = builder.build(system_prompt="core", user_message="MY QUESTION", retrieved=[result("a.py")])
    content = built.messages[-1].content

    assert content.index(CONTEXT_CLOSE) < content.index("MY QUESTION")


def test_no_context_block_when_nothing_retrieved(builder: ContextBuilder) -> None:
    built = builder.build(system_prompt="core", user_message="hello")
    assert built.messages[-1].content == "hello"


# --------------------------------------------------------------- citations


def test_citations_are_numbered_and_resolvable(builder: ContextBuilder) -> None:
    built = builder.build(
        system_prompt="core",
        user_message="q",
        retrieved=[result("src/a.py", start=10, end=20), result("src/b.py", start=1, end=5)],
    )
    content = built.messages[-1].content

    assert "[1] src/a.py:10-20" in content
    assert "[2] src/b.py:1-5" in content
    assert built.citations == ["src/a.py:10-20", "src/b.py:1-5"]


def test_symbol_scope_is_shown(builder: ContextBuilder) -> None:
    built = builder.build(system_prompt="core", user_message="q", retrieved=[result("a.py")])
    assert "(f)" in built.messages[-1].content


# ----------------------------------------------------------------- budgeting


def test_retrieval_is_capped(builder: ContextBuilder) -> None:
    """Never exceed num_ctx: Ollama would truncate silently rather than erroring."""
    huge = [result(f"src/file{i}.py", text="x" * 4_000) for i in range(50)]

    built = builder.build(system_prompt="core", user_message="q", retrieved=huge)

    assert built.dropped
    assert built.usage.used[Segment.RETRIEVAL] <= built.usage.budget.limit(Segment.RETRIEVAL)
    assert any("did not fit" in n for n in built.notices)


def test_large_pin_is_trimmed_rather_than_blowing_the_budget(builder: ContextBuilder) -> None:
    """M3 acceptance: a large @file pin must never exceed num_ctx."""
    built = builder.build(
        system_prompt="core",
        user_message="explain this",
        pinned=[("huge.py", "y" * 200_000)],
    )

    assert built.estimated_prompt_tokens < 12_288
    assert "huge.py" in built.citations


def test_pinned_files_come_before_retrieved(builder: ContextBuilder) -> None:
    """The user named them explicitly; they outrank whatever retrieval found."""
    built = builder.build(
        system_prompt="core",
        user_message="q",
        retrieved=[result("src/found.py")],
        pinned=[("pinned.py", "content")],
    )
    content = built.messages[-1].content

    assert content.index("pinned.py") < content.index("src/found.py")


def test_history_is_trimmed_from_the_front(builder: ContextBuilder) -> None:
    """Recent turns matter more, and dropping from the front keeps what remains append-only."""
    history = [Message(role="user", content="turn " + "x" * 5_000) for _ in range(20)]

    built = builder.build(system_prompt="core", user_message="q", history=history)
    kept = [m for m in built.messages if m.role in ("user", "assistant")][:-1]

    assert len(kept) < len(history)
    assert built.usage.used[Segment.HISTORY] <= built.usage.budget.limit(Segment.HISTORY)
    assert any("/compact" in n for n in built.notices)


def test_total_stays_within_the_window(builder: ContextBuilder) -> None:
    built = builder.build(
        system_prompt="core " * 500,
        project_instructions="conventions " * 500,
        repo_map="map " * 500,
        user_message="q",
        history=[Message(role="user", content="h " * 2_000)],
        retrieved=[result(f"f{i}.py", text="z" * 3_000) for i in range(30)],
    )

    assert built.estimated_prompt_tokens + built.usage.budget.output_reserve <= 12_288


def test_usage_reports_every_segment(builder: ContextBuilder) -> None:
    """`/context` renders these rows."""
    built = builder.build(
        system_prompt="core",
        project_instructions="conv",
        user_message="q",
        retrieved=[result("a.py")],
    )
    rows = dict((segment, used) for segment, used, _ in built.usage.rows())

    assert rows[Segment.SYSTEM] > 0
    assert rows[Segment.PROJECT] > 0
    assert rows[Segment.RETRIEVAL] > 0
