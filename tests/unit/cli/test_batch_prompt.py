"""The terminal's batch review screen — §6.4.

What the prompt is shaped by is approval fatigue: a batch exists to make many changes cheap
to answer, which is exactly where the reflex keypress becomes dangerous. So a bare Enter
lands on reviewing the files one by one, approve-all is a letter chosen deliberately, and
it is absent — not merely refused — when any item is flagged.
"""

from __future__ import annotations

from rich.console import Console

from hearth.cli.approval import ApprovalPrompt
from hearth.core.bus import EventBus
from hearth.core.events import ApprovalRequested, ApprovalResponse, BatchItemView


def view(name: str, *badges: str, added: int = 1, removed: int = 1) -> BatchItemView:
    return BatchItemView(
        call_id=f"call-{name}",
        tool="edit_file",
        path=f"{name}.py",
        summary=f"edit {name}.py",
        preview=f"--- a/{name}.py\n+++ b/{name}.py\n-old\n+new",
        badges=list(badges),
        added=added,
        removed=removed,
    )


def batch(*items: BatchItemView) -> ApprovalRequested:
    badges = sorted({badge for item in items for badge in item.badges})
    return ApprovalRequested(
        request_id="batch-1",
        call_id=items[0].call_id,
        tool="batch_review",
        risk="WRITE",
        preview="changes",
        badges=badges,
        options=["reject", "abort"] if badges else ["approve", "reject", "abort"],
        items=list(items),
    )


async def decide(request: ApprovalRequested, *answers: str):
    """Drive the prompt with scripted answers; returns (response, console text, asked)."""
    bus = EventBus()
    console = Console(record=True, force_terminal=False, width=120)
    remaining = list(answers)
    asked: list[str] = []

    async def ask(prompt: str) -> str:
        asked.append(prompt)
        if not remaining:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return remaining.pop(0)

    responses: list[ApprovalResponse] = []
    original = bus.resolve_approval

    def capture(response: ApprovalResponse) -> bool:
        responses.append(response)
        return original(response)

    bus.resolve_approval = capture  # type: ignore[method-assign]
    await ApprovalPrompt(bus=bus, console=console, ask=ask)(request)
    return responses[0], console.export_text(), asked


# ----------------------------------------------------------------- the screen


async def test_the_screen_lists_each_file_with_its_stats() -> None:
    _response, output, _asked = await decide(
        batch(view("a", added=3, removed=1), view("b", added=0, removed=7)), "n", ""
    )

    assert "2 file changes proposed in one step" in output
    assert "1. a.py" in output and "+3 -1" in output
    assert "2. b.py" in output and "+0 -7" in output


async def test_approve_all_is_offered_when_nothing_blocks_it() -> None:
    _response, output, _asked = await decide(batch(view("a"), view("b")), "n", "")

    assert "[a] approve all" in output


async def test_approve_all_approves_the_whole_batch() -> None:
    response, _output, _asked = await decide(batch(view("a"), view("b")), "a")

    assert response.decision == "approve"
    assert response.item_decisions is None


async def test_a_bare_enter_reviews_one_by_one_rather_than_approving_everything() -> None:
    """The reflex keypress has to land on the careful path."""
    response, _output, _asked = await decide(batch(view("a"), view("b")), "", "y", "y")

    assert response.item_decisions == {"call-a": "approve", "call-b": "approve"}


async def test_reject_all_asks_why_and_rejects() -> None:
    response, _output, asked = await decide(batch(view("a"), view("b")), "n", "not these")

    assert response.decision == "reject"
    assert response.feedback == "not these"
    assert any("why" in prompt for prompt in asked)


async def test_abort_ends_the_task() -> None:
    response, _output, _asked = await decide(batch(view("a"), view("b")), "q")

    assert response.decision == "abort"


# ------------------------------------------------------------------- blockers


async def test_a_blocking_badge_removes_approve_all_from_the_screen() -> None:
    _response, output, _asked = await decide(batch(view("a"), view("b", "SECRET?")), "n", "")

    assert "[a] approve all" not in output
    assert "approve-all is unavailable: SECRET?" in output


async def test_typing_the_removed_option_anyway_does_not_approve() -> None:
    """Absent from the menu *and* not honoured when typed: the unavailability is real, not
    cosmetic."""
    response, output, _asked = await decide(
        batch(view("a"), view("b", "PARSE-ERRORS-INTRODUCED")), "a", "n", ""
    )

    assert "Review the files one by one" in output
    assert response.decision == "reject"


async def test_a_flagged_file_has_no_default_answer_when_walking_the_files() -> None:
    """A bare Enter must not approve the one file that carries a warning."""
    request = batch(view("a"), view("b", "SECRET?"))

    response, output, asked = await decide(request, "i", "y", "", "n")

    assert "choice (no default)" in " ".join(asked)
    assert "unrecognised" in output
    assert response.item_decisions == {"call-a": "approve", "call-b": "reject"}


# ------------------------------------------------------------ walking the files


async def test_per_file_answers_are_recorded_by_call_id() -> None:
    response, _output, _asked = await decide(batch(view("a"), view("b"), view("c")), "i", "y", "n", "y")

    assert response.item_decisions == {
        "call-a": "approve",
        "call-b": "reject",
        "call-c": "approve",
    }


async def test_a_partial_approval_summarises_as_a_rejection() -> None:
    """The top-level decision is only for a frontend that ignores `item_decisions`, and for
    that reader "not everything was approved" must not read as approval."""
    response, _output, _asked = await decide(batch(view("a"), view("b")), "i", "y", "n")

    assert response.decision == "reject"


async def test_abort_part_way_through_the_files_ends_the_task() -> None:
    response, _output, _asked = await decide(batch(view("a"), view("b")), "i", "y", "q")

    assert response.decision == "abort"
    assert response.item_decisions is None


async def test_a_number_expands_that_files_diff_and_returns_to_the_prompt() -> None:
    _response, output, asked = await decide(batch(view("a"), view("b")), "2", "q")

    assert "+++ b/b.py" in output
    assert len(asked) == 2, "the expansion was not an answer"


async def test_the_full_diff_can_be_opened_while_walking() -> None:
    _response, output, _asked = await decide(batch(view("a"), view("b")), "i", "d", "y", "y")

    assert output.count("+++ b/a.py") >= 2, "shown in the panel and again on request"


async def test_unrecognised_input_reprompts_rather_than_guessing() -> None:
    response, output, _asked = await decide(batch(view("a"), view("b")), "maybe", "q")

    assert "unrecognised" in output
    assert response.decision == "abort"
