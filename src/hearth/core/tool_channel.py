"""Bridges the tool gateway's channel onto the event bus.

``tools`` sits below ``core`` and cannot import the bus, so the gateway depends on the
``ToolChannel`` protocol it defines and this adapter supplies the implementation
(docs/project-structure.md §2). This is the only place the two vocabularies meet.
"""

from __future__ import annotations

import uuid
from typing import Any

from hearth.core.bus import BusClosedError, EventBus
from hearth.core.events import (
    ApprovalDecision,
    ApprovalRequested,
    BatchItemView,
    ToolCallProposed,
    ToolFinished,
    ToolStarted,
)
from hearth.tools.channel import (
    ApprovalAsk,
    ApprovalReply,
    BatchItem,
    batch_blockers,
)


class EventBusChannel:
    """Publishes tool lifecycle events and routes approvals through the bus."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def proposed(self, *, call_id: str, tool: str, arguments: dict[str, Any]) -> None:
        await self._bus.publish(ToolCallProposed(call_id=call_id, tool=tool, arguments=arguments))

    async def started(self, *, call_id: str, tool: str) -> None:
        await self._bus.publish(ToolStarted(call_id=call_id, tool=tool))

    async def finished(
        self,
        *,
        call_id: str,
        tool: str,
        ok: bool,
        summary: str,
        duration_ms: float,
    ) -> None:
        await self._bus.publish(
            ToolFinished(
                call_id=call_id,
                tool=tool,
                ok=ok,
                summary=summary,
                duration_ms=duration_ms,
            )
        )

    async def request_approval(self, ask: ApprovalAsk) -> ApprovalReply | None:
        """Publish an approval request and wait for the answer.

        A closed bus returns None, which the gateway treats as a denial. Approvals never
        resolve into permission by default (docs/safety-and-tool-use.md §1.3).
        """
        request = ApprovalRequested(
            request_id=ask.call_id,
            call_id=ask.call_id,
            tool=ask.tool,
            risk=ask.risk,  # type: ignore[arg-type]
            preview=ask.preview,
            badges=list(ask.badges),
            reasons=list(ask.reasons),
            options=_options_for(ask),
            grant_key=ask.grant_key,
            typed_confirmation=ask.typed_confirmation,
        )

        try:
            response = await self._bus.request_approval(request)
        except BusClosedError:
            return None

        return ApprovalReply(
            decision=response.decision,
            edited_arguments=response.edited_arguments,
            feedback=response.feedback,
        )


    async def request_batch_approval(self, items: list[BatchItem]) -> dict[str, str] | None:
        """One review screen for several writes (docs/safety-and-tool-use.md §6.4).

        The answer is validated here, not only by the frontend that produced it. A
        whole-batch "approve" is honoured only when no item carries a blocking badge; a
        frontend that offered it anyway — a bug, or one written before batches existed —
        gets the safe reading, which is a rejection of everything. "Approve all is
        disabled" is a property of the system rather than of one prompt's rendering.
        """
        blockers = batch_blockers([item.badges for item in items])
        options: list[ApprovalDecision] = ["reject", "abort"] if blockers else ["approve", "reject", "abort"]

        request = ApprovalRequested(
            request_id=f"batch-{uuid.uuid4().hex[:10]}",
            call_id=items[0].call_id,
            tool="batch_review",
            risk="WRITE",
            preview=f"{len(items)} file change(s) proposed in one step",
            badges=sorted({badge for item in items for badge in item.badges}),
            reasons=[f"approve-all is unavailable: {', '.join(blockers)}"] if blockers else [],
            options=options,
            items=[
                BatchItemView(
                    call_id=item.call_id,
                    tool=item.tool,
                    path=item.path,
                    summary=item.summary,
                    preview=item.preview,
                    badges=list(item.badges),
                    added=item.added,
                    removed=item.removed,
                )
                for item in items
            ],
        )

        try:
            response = await self._bus.request_approval(request)
        except BusClosedError:
            return None

        offered = {item.call_id for item in items}
        if response.item_decisions is not None:
            # Only ids that were offered, and only an explicit approve counts. An unknown
            # id is ignored rather than trusted, and a missing one is a rejection.
            return {
                call_id: ("approve" if decision in ("approve", "always_session") else "reject")
                for call_id, decision in response.item_decisions.items()
                if call_id in offered
            }

        if response.decision == "approve" and not blockers:
            return dict.fromkeys(offered, "approve")
        return dict.fromkeys(offered, "reject")


def _options_for(ask: ApprovalAsk) -> list[ApprovalDecision]:
    """Which answers the frontend should offer.

    "always for this session" appears only when policy issued a grant key. The engine
    withholds it for DESTRUCTIVE, SHELL and NETWORK? calls (§6.1), so the option cannot be
    offered for something a grant would not actually cover.
    """
    options: list[ApprovalDecision] = ["approve", "reject", "edit", "abort"]
    if ask.grant_key:
        options.insert(1, "always_session")
    return options
