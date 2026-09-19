"""The async event bus, including request/response for approvals.

Two directions:

* **Publish** — the core emits events; every subscribed frontend receives them.
* **Request** — the core emits an ``ApprovalRequested`` and *waits* for the matching
  ``ApprovalResponse``. This is the mechanism that makes approvals blocking without the
  core knowing anything about terminals.

**Approvals never time out into approval.** ``request_approval`` takes no timeout that
resolves to "allow"; if a frontend disappears, the pending future is cancelled and the
caller must treat that as a denial. Failing closed is the rule
(docs/safety-and-tool-use.md §1.3).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from hearth.core.events import ApprovalRequested, ApprovalResponse

logger = logging.getLogger(__name__)

#: A subscriber callback. May be sync or async.
Subscriber = Callable[[Any], Any]


class BusClosedError(RuntimeError):
    """The bus was closed while an operation was pending."""


class EventBus:
    """Fan-out event delivery plus correlated approval request/response.

    Subscribers are invoked in registration order. A subscriber that raises does not
    prevent the others from receiving the event: one broken renderer must not take down
    the agent loop mid-turn.
    """

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._pending: dict[str, asyncio.Future[ApprovalResponse]] = {}
        self._closed = False

    # ------------------------------------------------------------- subscription

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        """Register a subscriber. Returns a function that unsubscribes it."""
        self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._subscribers.remove(subscriber)

        return unsubscribe

    @contextlib.asynccontextmanager
    async def collect(self) -> AsyncIterator[list[Any]]:
        """Collect every event published inside the block. For tests and transcripts."""
        received: list[Any] = []
        unsubscribe = self.subscribe(received.append)
        try:
            yield received
        finally:
            unsubscribe()

    # ---------------------------------------------------------------- publishing

    async def publish(self, event: Any) -> None:
        """Deliver an event to every subscriber, in order."""
        if self._closed:
            raise BusClosedError("cannot publish on a closed bus")

        for subscriber in list(self._subscribers):
            try:
                result = subscriber(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                # One broken renderer must not take down the agent loop mid-turn. But a
                # silently swallowed exception is its own bug, so it is logged rather
                # than discarded.
                logger.exception("event subscriber failed handling %s", type(event).__name__)

    # ------------------------------------------------------- request / response

    async def request_approval(self, request: ApprovalRequested) -> ApprovalResponse:
        """Publish an approval request and wait for its response.

        Raises:
            BusClosedError: if the bus closes while waiting. Callers must treat this as a
                denial — there is deliberately no timeout that resolves to "approve".
        """
        if self._closed:
            raise BusClosedError("cannot request approval on a closed bus")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[ApprovalResponse] = loop.create_future()
        self._pending[request.request_id] = future

        try:
            await self.publish(request)
            return await future
        finally:
            self._pending.pop(request.request_id, None)

    def resolve_approval(self, response: ApprovalResponse) -> bool:
        """Deliver a response to whoever is waiting. False if nobody was.

        A stale or duplicated response is ignored rather than raising: a frontend sending
        one twice is a UI bug, not a reason to crash a turn.
        """
        future = self._pending.get(response.request_id)
        if future is None or future.done():
            return False
        future.set_result(response)
        return True

    @property
    def pending_approvals(self) -> tuple[str, ...]:
        """Request ids currently awaiting a response."""
        return tuple(self._pending)

    # -------------------------------------------------------------- lifecycle

    async def close(self) -> None:
        """Close the bus, failing every pending approval closed."""
        self._closed = True
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(BusClosedError("bus closed while awaiting approval"))
        self._pending.clear()
        self._subscribers.clear()

    @property
    def closed(self) -> bool:
        return self._closed
