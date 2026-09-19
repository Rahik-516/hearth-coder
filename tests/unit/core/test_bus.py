"""Event bus behaviour, with emphasis on the approval round trip.

The fail-closed tests are the point: an approval that cannot be answered must never
resolve into permission (docs/safety-and-tool-use.md §1.3).
"""

from __future__ import annotations

import asyncio

import pytest

from hearth.core.bus import BusClosedError, EventBus
from hearth.core.events import ApprovalRequested, ApprovalResponse, Notice, TextDelta


def approval(request_id: str = "r1") -> ApprovalRequested:
    return ApprovalRequested(
        request_id=request_id,
        call_id="c1",
        tool="edit_file",
        risk="WRITE",
        preview="--- a\n+++ b",
        options=["approve", "reject"],
    )


# ------------------------------------------------------------------- publishing


async def test_subscribers_receive_events_in_order() -> None:
    bus = EventBus()
    received: list[object] = []
    bus.subscribe(received.append)

    await bus.publish(TextDelta(text="one"))
    await bus.publish(TextDelta(text="two"))

    assert [e.text for e in received] == ["one", "two"]  # type: ignore[attr-defined]


async def test_async_subscribers_are_awaited() -> None:
    bus = EventBus()
    received: list[object] = []

    async def handler(event: object) -> None:
        await asyncio.sleep(0)
        received.append(event)

    bus.subscribe(handler)
    await bus.publish(Notice(message="hi"))

    assert len(received) == 1


async def test_all_subscribers_receive_each_event() -> None:
    bus = EventBus()
    a: list[object] = []
    b: list[object] = []
    bus.subscribe(a.append)
    bus.subscribe(b.append)

    await bus.publish(Notice(message="x"))

    assert len(a) == len(b) == 1


async def test_a_raising_subscriber_does_not_break_the_others() -> None:
    """A broken renderer must not take down the agent loop mid-turn."""
    bus = EventBus()
    received: list[object] = []

    def broken(event: object) -> None:
        raise RuntimeError("renderer exploded")

    bus.subscribe(broken)
    bus.subscribe(received.append)

    await bus.publish(Notice(message="still delivered"))

    assert len(received) == 1


async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    received: list[object] = []
    unsubscribe = bus.subscribe(received.append)

    await bus.publish(Notice(message="first"))
    unsubscribe()
    await bus.publish(Notice(message="second"))

    assert len(received) == 1


async def test_collect_context_manager_captures_events() -> None:
    bus = EventBus()
    async with bus.collect() as events:
        await bus.publish(TextDelta(text="inside"))
    await bus.publish(TextDelta(text="outside"))

    assert len(events) == 1


# --------------------------------------------------------------------- approvals


async def test_approval_request_blocks_until_answered() -> None:
    bus = EventBus()

    async def frontend(event: object) -> None:
        if isinstance(event, ApprovalRequested):
            bus.resolve_approval(ApprovalResponse(request_id=event.request_id, decision="approve"))

    bus.subscribe(frontend)
    response = await bus.request_approval(approval())

    assert response.decision == "approve"


async def test_request_is_pending_while_unanswered() -> None:
    bus = EventBus()
    task = asyncio.create_task(bus.request_approval(approval("pending-1")))
    await asyncio.sleep(0)

    assert bus.pending_approvals == ("pending-1",)
    assert not task.done()

    bus.resolve_approval(ApprovalResponse(request_id="pending-1", decision="reject"))
    assert (await task).decision == "reject"
    assert bus.pending_approvals == ()


async def test_responses_are_correlated_by_request_id() -> None:
    """Two concurrent approvals must not cross-resolve."""
    bus = EventBus()
    first = asyncio.create_task(bus.request_approval(approval("a")))
    second = asyncio.create_task(bus.request_approval(approval("b")))
    await asyncio.sleep(0)

    bus.resolve_approval(ApprovalResponse(request_id="b", decision="approve"))
    await asyncio.sleep(0)

    assert second.done()
    assert not first.done()
    assert (await second).decision == "approve"

    bus.resolve_approval(ApprovalResponse(request_id="a", decision="reject"))
    assert (await first).decision == "reject"


async def test_edited_arguments_survive_the_round_trip() -> None:
    """The runner re-prepares and re-evaluates these, so they must arrive intact."""
    bus = EventBus()

    async def frontend(event: object) -> None:
        if isinstance(event, ApprovalRequested):
            bus.resolve_approval(
                ApprovalResponse(
                    request_id=event.request_id,
                    decision="edit",
                    edited_arguments={"command": "pytest -q"},
                )
            )

    bus.subscribe(frontend)
    response = await bus.request_approval(approval())

    assert response.edited_arguments == {"command": "pytest -q"}


async def test_unknown_response_is_ignored() -> None:
    bus = EventBus()
    assert bus.resolve_approval(ApprovalResponse(request_id="nope", decision="approve")) is False


async def test_duplicate_response_is_ignored() -> None:
    """A frontend double-sending is a UI bug, not a reason to crash a turn."""
    bus = EventBus()
    task = asyncio.create_task(bus.request_approval(approval("dup")))
    await asyncio.sleep(0)

    assert bus.resolve_approval(ApprovalResponse(request_id="dup", decision="approve")) is True
    assert bus.resolve_approval(ApprovalResponse(request_id="dup", decision="reject")) is False
    assert (await task).decision == "approve"


# ------------------------------------------------------------------ fail closed


async def test_closing_the_bus_fails_pending_approvals() -> None:
    """A vanished frontend must not resolve into permission."""
    bus = EventBus()
    task = asyncio.create_task(bus.request_approval(approval("orphan")))
    await asyncio.sleep(0)

    await bus.close()

    with pytest.raises(BusClosedError):
        await task


async def test_closed_bus_refuses_new_approvals() -> None:
    bus = EventBus()
    await bus.close()

    with pytest.raises(BusClosedError):
        await bus.request_approval(approval())


async def test_closed_bus_refuses_publishing() -> None:
    bus = EventBus()
    await bus.close()

    with pytest.raises(BusClosedError):
        await bus.publish(Notice(message="too late"))


async def test_request_approval_has_no_timeout_parameter() -> None:
    """There is deliberately no timeout, because any timeout default would be a policy.

    A timeout that approves is catastrophic; a timeout that denies belongs to the caller,
    which knows the mode and can produce a proper denial reason for the model.
    """
    import inspect

    params = set(inspect.signature(EventBus.request_approval).parameters)
    assert params == {"self", "request"}
