"""Compaction — docs/system-design.md §9.4.

The I2 acceptance criterion is a scripted 60-turn session that never exceeds `num_ctx`,
keeps pinned facts, and resumes correctly. That is the test at the bottom; the ones above
it cover the ways compaction can be subtly wrong while appearing to work:

* folding a user question away from the answer that belongs to it,
* nesting a summary of a summary until the history is nothing but summaries,
* replacing real turns with a summary that is *larger* than they were,
* losing a pinned fact because the model chose not to repeat it.

The summarizer is a stub. What compaction does with a summary is Hearth's logic and is
worth testing precisely; whether a 4B model writes a good one is a question for the eval.
"""

from __future__ import annotations

import pytest

from hearth.core.context.budget import budget_for
from hearth.core.context.compactor import (
    SUMMARY_SECTIONS,
    CompactionPolicy,
    build_summary_prompt,
    compact,
    is_summary,
)
from hearth.llm.types import Message


def policy(num_ctx: int = 12_288, threshold: float = 0.75) -> CompactionPolicy:
    return CompactionPolicy(budget=budget_for(num_ctx), threshold=threshold)


def conversation(turns: int, *, words: int = 40) -> list[Message]:
    """A user/assistant exchange per turn, each large enough to matter."""
    history: list[Message] = []
    for index in range(turns):
        history.append(Message(role="user", content=f"question {index} " + "detail " * words))
        history.append(Message(role="assistant", content=f"answer {index} " + "prose " * words))
    return history


async def fake_summary(prompt: str) -> str:
    return "\n".join(f"### {name}\nsomething about {name.lower()}" for name in SUMMARY_SECTIONS)


# ------------------------------------------------------------------- triggering


def test_a_short_session_does_not_compact() -> None:
    """Compaction costs a full prefill; spending it on a session with room is waste."""
    assert not policy().should_compact(conversation(2))


def test_a_long_session_triggers_before_the_window_is_full() -> None:
    """At 100% the turn being truncated is the one that needed the room."""
    rules = policy()
    history = conversation(60)

    assert rules.should_compact(history)
    assert rules.trigger_tokens() < rules.usable_tokens


def test_fixed_overhead_counts_toward_the_trigger() -> None:
    """A large repo map cannot shrink, so ignoring it lets a session sail past the
    trigger and hit the real ceiling instead."""
    rules = policy()
    history = conversation(12)

    assert not rules.should_compact(history)
    assert rules.should_compact(history, overhead_tokens=rules.trigger_tokens())


def test_the_output_reserve_is_never_counted_as_usable() -> None:
    rules = policy()

    assert rules.usable_tokens < rules.budget.num_ctx


# ---------------------------------------------------------------------- folding


async def test_compaction_replaces_old_turns_with_one_summary() -> None:
    result = await compact(conversation(30), policy=policy(), summarize=fake_summary)

    assert result.compacted
    assert is_summary(result.history[0])
    assert result.tokens_after < result.tokens_before
    assert result.replaced > 0


async def test_the_most_recent_turns_survive_verbatim() -> None:
    """The next message refers to them — "that file", "the error above"."""
    history = conversation(30)
    result = await compact(history, policy=policy(), summarize=fake_summary)

    assert history[-1] in result.history
    assert history[-2] in result.history


async def test_a_kept_turn_is_not_cut_from_its_answer() -> None:
    """Counting raw messages instead of turns would keep two tool results and call that
    two turns, orphaning the question they answered."""
    history = conversation(30)
    result = await compact(history, policy=policy(), summarize=fake_summary)

    kept = result.history[1:]
    assert kept[0].role == "user", "a kept window must start at a user turn"


async def test_tool_results_stay_with_the_call_they_answer() -> None:
    history = conversation(20)
    history.extend(
        [
            Message(role="user", content="run the tests"),
            Message(role="assistant", content="", tool_calls=[]),
            Message(role="tool", content="3 passed", tool_name="run_tests"),
            Message(role="assistant", content="They pass."),
        ]
    )

    result = await compact(history, policy=policy(), summarize=fake_summary)

    kept = result.history[1:]
    assert kept[0].role == "user"
    assert any(message.role == "tool" for message in kept)


async def test_compacting_twice_does_not_nest_summaries() -> None:
    """Otherwise history becomes a summary of a summary of a summary."""
    once = await compact(conversation(30), policy=policy(), summarize=fake_summary)
    twice = await compact(
        [*once.history, *conversation(20)], policy=policy(), summarize=fake_summary
    )

    summaries = [message for message in twice.history if is_summary(message)]
    assert len(summaries) == 1


# ----------------------------------------------------------------- pinned facts


async def test_pinned_facts_survive_verbatim() -> None:
    """A fact that survives only if the model repeats it is not pinned."""
    facts = ["the test command is `uv run pytest -q`", "never touch migrations/0001"]

    result = await compact(
        conversation(30), policy=policy(), summarize=fake_summary, pinned_facts=facts
    )

    for fact in facts:
        assert fact in result.history[0].content


async def test_pinned_facts_survive_a_summarizer_that_ignores_them() -> None:
    async def forgetful(prompt: str) -> str:
        return "### Goal\nnone"

    result = await compact(
        conversation(30),
        policy=policy(),
        summarize=forgetful,
        pinned_facts=["the deploy key lives in ops/keys.md"],
    )

    assert "ops/keys.md" in result.history[0].content


# -------------------------------------------------------------------- degrading


async def test_a_failing_summarizer_leaves_the_history_intact() -> None:
    """A session that cannot compact is still a working session."""

    async def broken(prompt: str) -> str:
        raise RuntimeError("model unreachable")

    history = conversation(30)
    result = await compact(history, policy=policy(), summarize=broken)

    assert not result.compacted
    assert result.history == history
    assert result.degraded is not None


async def test_an_empty_summary_is_refused() -> None:
    async def empty(prompt: str) -> str:
        return "   "

    result = await compact(conversation(30), policy=policy(), summarize=empty)

    assert not result.compacted
    assert result.degraded is not None


async def test_a_summary_larger_than_what_it_replaces_is_refused() -> None:
    """Swapping real turns for a longer, worse account of them is a strict loss."""

    async def verbose(prompt: str) -> str:
        return "padding " * 5000

    result = await compact(conversation(8), policy=policy(), summarize=verbose)

    assert not result.compacted
    assert result.degraded is not None


async def test_a_history_too_short_to_fold_is_returned_unchanged() -> None:
    history = conversation(1)

    result = await compact(history, policy=policy(), summarize=fake_summary)

    assert not result.compacted
    assert result.history == history


# ----------------------------------------------------------------- the prompt


def test_the_prompt_names_every_required_section() -> None:
    """Free-form prose about a coding session drops the paths and commands."""
    prompt = build_summary_prompt(conversation(4))

    for section in SUMMARY_SECTIONS:
        assert section in prompt


def test_the_prompt_forbids_omitting_a_heading() -> None:
    """A missing heading is indistinguishable from one the model judged empty."""
    assert "do not omit" in build_summary_prompt(conversation(2)).lower()


# ------------------------------------------------- the acceptance criterion


async def test_a_sixty_turn_session_never_exceeds_the_window() -> None:
    """The I2 criterion, run as a session rather than asserted about one call."""
    rules = policy()
    facts = ["the fixture repo is tests/fixtures/repos/py_small"]
    history: list[Message] = []
    compactions = 0

    for index in range(60):
        history.append(Message(role="user", content=f"question {index} " + "detail " * 40))
        history.append(Message(role="assistant", content=f"answer {index} " + "prose " * 40))

        if rules.should_compact(history):
            result = await compact(
                history, policy=rules, summarize=fake_summary, pinned_facts=facts
            )
            assert result.compacted, result.degraded
            history = result.history
            compactions += 1

        assert rules.measure(history) < rules.usable_tokens, f"over budget at turn {index}"

    assert compactions > 0, "a 60-turn session should have compacted at least once"
    assert facts[0] in history[0].content, "the pinned fact was lost"
    assert history[-1].content.startswith("answer 59"), "the session did not resume correctly"


@pytest.mark.parametrize("num_ctx", [8_192, 12_288, 16_384, 32_768])
async def test_the_window_is_respected_at_every_tier(num_ctx: int) -> None:
    rules = policy(num_ctx)
    history: list[Message] = []

    for index in range(60):
        history.append(Message(role="user", content=f"q{index} " + "word " * 60))
        history.append(Message(role="assistant", content=f"a{index} " + "word " * 60))
        if rules.should_compact(history):
            history = (
                await compact(history, policy=rules, summarize=fake_summary)
            ).history

    assert rules.measure(history) < rules.usable_tokens


# ------------------------------------------------------ wired into the runner


async def test_the_runner_compacts_and_starts_a_new_epoch(tmp_path) -> None:
    """Compaction rewrites the cached prefix, so the epoch must move with it.

    Not bumping would be worse than the prefill it costs: the server would be told a
    prefix it no longer holds is still valid.
    """
    from pathlib import Path

    from hearth.core.bus import EventBus
    from hearth.core.runner import ChatRunner
    from hearth.core.session import Session
    from hearth.llm.scripted_provider import ScriptedProvider, ScriptedResponse

    session = Session(
        id="s_compact",
        workspace=Path(tmp_path),
        model="scripted",
        num_ctx=12_288,
        history=conversation(60),
    )
    epoch_before = session.epoch
    provider = ScriptedProvider([ScriptedResponse("### Goal\nship it")])

    result = await ChatRunner(provider=provider, bus=EventBus()).maybe_compact(session)

    assert result.compacted
    assert session.epoch == epoch_before + 1
    assert len(session.history) < 120


async def test_the_runner_leaves_a_short_session_alone(tmp_path) -> None:
    """A full prefill is the cost of compaction; spending it with room to spare is waste."""
    from pathlib import Path

    from hearth.core.bus import EventBus
    from hearth.core.runner import ChatRunner
    from hearth.core.session import Session
    from hearth.llm.scripted_provider import ScriptedProvider

    session = Session(
        id="s_small",
        workspace=Path(tmp_path),
        model="scripted",
        num_ctx=12_288,
        history=conversation(2),
    )
    epoch_before = session.epoch

    result = await ChatRunner(provider=ScriptedProvider(), bus=EventBus()).maybe_compact(session)

    assert not result.compacted
    assert session.epoch == epoch_before
    assert len(session.history) == 4


async def test_pinned_paths_are_carried_through_compaction(tmp_path) -> None:
    from pathlib import Path

    from hearth.core.bus import EventBus
    from hearth.core.runner import ChatRunner
    from hearth.core.session import Session
    from hearth.llm.scripted_provider import ScriptedProvider, ScriptedResponse

    session = Session(
        id="s_pins",
        workspace=Path(tmp_path),
        model="scripted",
        num_ctx=12_288,
        history=conversation(60),
    )
    session.pin("src/billing/invoice_service.py")
    provider = ScriptedProvider([ScriptedResponse("### Goal\nfix rounding")])

    await ChatRunner(provider=provider, bus=EventBus()).maybe_compact(session)

    assert "src/billing/invoice_service.py" in session.history[0].content
