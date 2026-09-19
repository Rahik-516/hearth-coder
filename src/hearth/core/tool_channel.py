"""Bridges the tool gateway's channel onto the event bus.

``tools`` sits below ``core`` and cannot import the bus, so the gateway depends on the
``ToolChannel`` protocol it defines and this adapter supplies the implementation
(docs/project-structure.md §2). This is the only place the two vocabularies meet.
"""

from __future__ import annotations

from typing import Any

from hearth.core.bus import BusClosedError, EventBus
from hearth.core.events import (
    ApprovalDecision,
    ApprovalRequested,
    ToolCallProposed,
    ToolFinished,
    ToolStarted,
)
from hearth.tools.channel import ApprovalAsk, ApprovalReply


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
