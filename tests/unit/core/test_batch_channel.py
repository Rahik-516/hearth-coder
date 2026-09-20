"""The bus adapter's half of batch review — §6.4.

"Approve all is disabled if any item carries DESTRUCTIVE, SECRET? or PARSE-ERRORS-
INTRODUCED" is a claim about the *system*, not about one prompt's rendering. So it is
enforced where the answer arrives, and these tests answer the request the way a buggy or
older frontend might: with an approve-all the screen should never have offered.
"""

from __future__ import annotations

from hearth.core.bus import EventBus
from hearth.core.events import ApprovalRequested, ApprovalResponse
from hearth.core.tool_channel import EventBusChannel
from hearth.tools.channel import BatchChannel, BatchItem


def item(call_id: str, *badges: str) -> BatchItem:
    return BatchItem(
        call_id=call_id,
        tool="edit_file",
        path=f"{call_id}.py",
        summary=f"edit {call_id}.py",
        preview="-a\n+b",
        badges=badges,
        added=1,
        removed=1,
    )


def answering(bus: EventBus, **response: object) -> list[ApprovalRequested]:
    """Subscribe a frontend that answers every approval with a fixed response."""
    seen: list[ApprovalRequested] = []

    def responder(event: object) -> None:
        if isinstance(event, ApprovalRequested):
            seen.append(event)
            bus.resolve_approval(ApprovalResponse(request_id=event.request_id, **response))  # type: ignore[arg-type]

    bus.subscribe(responder)
    return seen


def test_the_bus_channel_offers_the_batch_capability() -> None:
    assert isinstance(EventBusChannel(EventBus()), BatchChannel)


async def test_approve_all_is_honoured_when_nothing_blocks_it() -> None:
    bus = EventBus()
    answering(bus, decision="approve")

    answers = await EventBusChannel(bus).request_batch_approval([item("a"), item("b")])

    assert answers == {"a": "approve", "b": "approve"}


async def test_approve_all_is_withheld_from_the_options_when_an_item_is_blocked() -> None:
    bus = EventBus()
    seen = answering(bus, decision="reject")

    await EventBusChannel(bus).request_batch_approval([item("a"), item("b", "SECRET?")])

    (request,) = seen
    assert "approve" not in request.options
    assert "SECRET?" in "".join(request.reasons)
    assert len(request.items) == 2


async def test_an_approve_all_that_should_not_have_been_offered_is_a_rejection() -> None:
    """The defence in depth: a frontend that offers it anyway does not get it."""
    bus = EventBus()
    answering(bus, decision="approve")

    answers = await EventBusChannel(bus).request_batch_approval([item("a"), item("b", "SECRET?")])

    assert answers == {"a": "reject", "b": "reject"}


async def test_each_blocking_badge_disables_approve_all() -> None:
    for badge in ("DESTRUCTIVE", "SECRET?", "PARSE-ERRORS-INTRODUCED"):
        bus = EventBus()
        answering(bus, decision="approve")

        answers = await EventBusChannel(bus).request_batch_approval([item("a"), item("b", badge)])

        assert set(answers.values()) == {"reject"}, badge


async def test_per_item_decisions_are_honoured_even_with_a_blocker_present() -> None:
    """Only the *single* approve-all control is disabled. Someone who reviewed the flagged
    file and approved it on its own has made a specific decision."""
    bus = EventBus()
    answering(bus, decision="approve", item_decisions={"a": "approve", "b": "approve"})

    answers = await EventBusChannel(bus).request_batch_approval([item("a"), item("b", "SECRET?")])

    assert answers == {"a": "approve", "b": "approve"}


async def test_a_mixed_answer_is_passed_through() -> None:
    bus = EventBus()
    answering(bus, decision="reject", item_decisions={"a": "approve", "b": "reject"})

    answers = await EventBusChannel(bus).request_batch_approval([item("a"), item("b")])

    assert answers == {"a": "approve", "b": "reject"}


async def test_an_id_that_was_never_offered_is_ignored() -> None:
    """Not trusted: a response naming a call the screen never showed cannot approve it."""
    bus = EventBus()
    answering(bus, decision="approve", item_decisions={"a": "approve", "ghost": "approve"})

    answers = await EventBusChannel(bus).request_batch_approval([item("a"), item("b")])

    assert answers == {"a": "approve"}
    assert "ghost" not in answers


async def test_an_abort_rejects_everything() -> None:
    bus = EventBus()
    answering(bus, decision="abort")

    answers = await EventBusChannel(bus).request_batch_approval([item("a"), item("b")])

    assert set(answers.values()) == {"reject"}


async def test_a_closed_bus_gives_no_answer() -> None:
    """None, which the gateway treats as "ask each call individually", never as consent."""
    bus = EventBus()
    await bus.close()

    assert await EventBusChannel(bus).request_batch_approval([item("a"), item("b")]) is None


async def test_the_request_carries_every_items_stats() -> None:
    bus = EventBus()
    seen = answering(bus, decision="reject")

    await EventBusChannel(bus).request_batch_approval([item("a"), item("b")])

    (request,) = seen
    assert request.tool == "batch_review"
    assert [(v.path, v.added, v.removed) for v in request.items] == [("a.py", 1, 1), ("b.py", 1, 1)]
