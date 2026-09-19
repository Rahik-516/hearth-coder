"""The approval prompt — docs/safety-and-tool-use.md §6.

Written before ``cli/approval.py`` (CLAUDE.md rule 3).

The prompt is the last thing between a proposal and a change to someone's files, and its
failure mode is not a crash — it is **approval fatigue** (T2). A prompt that offers
"always for this session" on something irreversible, or that accepts a bare keypress for a
destructive action, is training the user to stop reading. So most of these tests are about
what the prompt refuses to offer.

The decision logic is deliberately separated from the terminal I/O: ``choices_for`` and
``interpret`` are pure functions over an ``ApprovalRequested``, which is why they can be
tested exhaustively here without a tty.
"""

from __future__ import annotations

import pytest

from hearth.cli.approval import (
    ApprovalPrompt,
    batch_blockers,
    choices_for,
    interpret,
    needs_typed_confirmation,
)
from hearth.core.bus import EventBus
from hearth.core.events import ApprovalRequested

pytestmark = pytest.mark.anyio


def ask_for(**kwargs: object) -> ApprovalRequested:
    fields: dict[str, object] = {
        "request_id": "r1",
        "call_id": "c1",
        "tool": "edit_file",
        "risk": "WRITE",
        "preview": "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-old\n+new\n",
    }
    fields.update(kwargs)
    return ApprovalRequested(**fields)  # type: ignore[arg-type]


# ------------------------------------------------------------------- choices


def test_the_base_choices_are_always_offered() -> None:
    keys = [choice.key for choice in choices_for(ask_for())]

    assert keys[0] == "y", "approve is first, because it is the common answer"
    assert set(keys) >= {"y", "n", "e", "d", "q"}


def test_always_for_session_is_offered_when_policy_issued_a_grant_key() -> None:
    choices = choices_for(ask_for(grant_key="edit:src/a.py"))

    assert "s" in [choice.key for choice in choices]


def test_always_for_session_is_withheld_without_a_grant_key() -> None:
    """The engine withholds the key for DESTRUCTIVE, SHELL and NETWORK? calls (§6.1).

    The UI must not invent it back. Offering a grant policy would not honour is worse
    than not offering one: the user believes they have stopped being asked.
    """
    choices = choices_for(ask_for(grant_key=None))

    assert "s" not in [choice.key for choice in choices]


@pytest.mark.parametrize("badge", ["DESTRUCTIVE", "SHELL", "NETWORK?"])
def test_always_for_session_is_withheld_for_ungrantable_badges(badge: str) -> None:
    """Defence in depth: even if a grant key arrives, these badges veto the offer."""
    choices = choices_for(ask_for(grant_key="edit:src/a.py", badges=[badge]))

    assert "s" not in [choice.key for choice in choices]


def test_choices_map_to_real_approval_decisions() -> None:
    valid = {"approve", "reject", "edit", "always_session", "abort"}

    for choice in choices_for(ask_for(grant_key="edit:a.py")):
        assert choice.decision in valid or choice.decision == "show_diff"


# ---------------------------------------------------------------- interpreting


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("y", "approve"),
        ("Y", "approve"),
        (" y ", "approve"),
        ("", "approve"),  # bare Enter takes the default
        ("n", "reject"),
        ("e", "edit"),
        ("q", "abort"),
        ("s", "always_session"),
    ],
)
def test_keys_map_to_decisions(raw: str, expected: str) -> None:
    choices = choices_for(ask_for(grant_key="edit:a.py"))

    choice = interpret(raw, choices)

    assert choice is not None
    assert choice.decision == expected


def test_an_unrecognised_key_is_not_guessed() -> None:
    """Returning None re-prompts. Guessing at "j" could approve something."""
    assert interpret("j", choices_for(ask_for())) is None


def test_enter_does_not_default_to_approve_for_a_destructive_call() -> None:
    """A reflex keypress must not be able to authorise something irreversible (§6.3)."""
    choices = choices_for(ask_for(badges=["DESTRUCTIVE"], typed_confirmation="confirm"))

    assert interpret("", choices) is None


# ------------------------------------------------------- typed confirmation


def test_a_typed_confirmation_is_required_when_policy_asks_for_one() -> None:
    assert needs_typed_confirmation(ask_for(typed_confirmation="confirm")) is True
    assert needs_typed_confirmation(ask_for()) is False


# ------------------------------------------------------------- batch review


def test_approve_all_is_allowed_for_ordinary_edits() -> None:
    assert batch_blockers([ask_for(), ask_for(request_id="r2")]) == []


@pytest.mark.parametrize("badge", ["DESTRUCTIVE", "SECRET?", "PARSE-ERRORS-INTRODUCED"])
def test_approve_all_is_blocked_by_any_risky_item(badge: str) -> None:
    """§6.4. One item is enough: "approve all" over a list containing a secret or a
    syntax error is precisely the click nobody should be able to make quickly."""
    blockers = batch_blockers([ask_for(), ask_for(request_id="r2", badges=[badge])])

    assert badge in blockers


def test_batch_blockers_are_reported_once_each() -> None:
    requests = [ask_for(badges=["SECRET?"]), ask_for(request_id="r2", badges=["SECRET?"])]

    assert batch_blockers(requests) == ["SECRET?"]


# -------------------------------------------------------------- the prompt


class Script:
    """Canned answers, standing in for a terminal."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else "q"


async def resolve(bus: EventBus, request: ApprovalRequested):
    """Run the prompt as the bus would, and return the response it produced."""
    import asyncio

    task = asyncio.create_task(bus.request_approval(request))
    await asyncio.sleep(0)
    return await task


async def test_approving_resolves_the_request() -> None:
    bus = EventBus()
    script = Script("y")
    bus.subscribe(ApprovalPrompt(bus=bus, ask=script))

    response = await resolve(bus, ask_for())

    assert response.decision == "approve"


async def test_rejecting_collects_feedback() -> None:
    """The feedback becomes the tool result the model sees, so it is worth asking for."""
    bus = EventBus()
    script = Script("n", "do not touch the migrations")
    bus.subscribe(ApprovalPrompt(bus=bus, ask=script))

    response = await resolve(bus, ask_for())

    assert response.decision == "reject"
    assert response.feedback == "do not touch the migrations"


async def test_an_unrecognised_key_re_prompts_rather_than_deciding() -> None:
    bus = EventBus()
    script = Script("j", "?", "y")
    bus.subscribe(ApprovalPrompt(bus=bus, ask=script))

    response = await resolve(bus, ask_for())

    assert response.decision == "approve"
    assert len(script.prompts) == 3


async def test_a_destructive_call_needs_the_word_typed() -> None:
    bus = EventBus()
    script = Script("y", "confirm")
    bus.subscribe(ApprovalPrompt(bus=bus, ask=script))

    response = await resolve(bus, ask_for(badges=["DESTRUCTIVE"], typed_confirmation="confirm"))

    assert response.decision == "approve"
    assert any("confirm" in prompt for prompt in script.prompts)


async def test_a_wrong_typed_confirmation_rejects_rather_than_re_asking() -> None:
    """Failing closed. Re-asking would let someone brute-force their way through a
    prompt they have already shown they are not reading."""
    bus = EventBus()
    script = Script("y", "yes")
    bus.subscribe(ApprovalPrompt(bus=bus, ask=script))

    response = await resolve(bus, ask_for(badges=["DESTRUCTIVE"], typed_confirmation="confirm"))

    assert response.decision == "reject"


async def test_editing_opens_the_editor_and_returns_new_arguments() -> None:
    bus = EventBus()
    script = Script("e")
    edited: list[str] = []

    def editor(text: str) -> str:
        edited.append(text)
        return '{"path": "src/b.py", "old_string": "x", "new_string": "y"}'

    bus.subscribe(
        ApprovalPrompt(bus=bus, ask=script, editor=editor, arguments_for=lambda _: {"path": "src/a.py"})
    )

    response = await resolve(bus, ask_for())

    assert response.decision == "edit"
    assert response.edited_arguments == {"path": "src/b.py", "old_string": "x", "new_string": "y"}
    assert edited, "the editor was handed the current arguments"


async def test_unparseable_edited_arguments_reject_rather_than_run_something_odd() -> None:
    bus = EventBus()
    script = Script("e")
    bus.subscribe(
        ApprovalPrompt(bus=bus, ask=script, editor=lambda _: "not json at all", arguments_for=lambda _: {})
    )

    response = await resolve(bus, ask_for())

    assert response.decision == "reject"


async def test_showing_the_full_diff_returns_to_the_prompt() -> None:
    """`d` is not a decision. It must not fall through to one."""
    bus = EventBus()
    script = Script("d", "y")
    bus.subscribe(ApprovalPrompt(bus=bus, ask=script))

    response = await resolve(bus, ask_for())

    assert response.decision == "approve"
    assert len(script.prompts) == 2


async def test_the_prompt_ignores_events_that_are_not_approvals() -> None:
    from hearth.core.events import TurnStarted

    bus = EventBus()
    script = Script()
    prompt = ApprovalPrompt(bus=bus, ask=script)

    # Every event on the bus reaches every subscriber, so the prompt sees the whole
    # stream. Reacting to anything but an approval would put a prompt in the middle of a
    # streaming answer.
    await prompt(TurnStarted(session_id="s1", mode="agent", model="qwen3.5:4b"))

    assert script.prompts == []
    assert bus.pending_approvals == ()
