"""How the gateway talks to whoever is watching.

The gateway needs two things from the outside world: somewhere to report progress, and
someone to ask when a call needs approval. Both are ``core`` concerns — but ``tools`` sits
*below* ``core`` in the layering, so it cannot import the event bus
(docs/project-structure.md §2).

So the dependency is inverted. This module defines the interface in ``tools``; ``core``
provides an adapter that bridges it to the event bus. The gateway depends on the protocol
and never learns that events exist.

That is not bookkeeping. It means the gateway can be driven by a test harness, a script, or
a future non-event frontend without any of them constructing a bus — and it keeps the
"tools are deterministic" boundary honest, since a tool's surroundings cannot smuggle
session state in through the channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ApprovalAsk:
    """A request for a human decision about one tool call."""

    call_id: str
    tool: str
    risk: str
    preview: str
    badges: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    grant_key: str | None = None
    #: For DESTRUCTIVE calls: the word the user must type to confirm (§6.3). A plain
    #: keypress is too easy to reflex through when the action cannot be undone.
    typed_confirmation: str | None = None


@dataclass(frozen=True)
class ApprovalReply:
    """The answer. ``None`` is never a valid reply — absence means deny."""

    decision: str  # approve | reject | edit | always_session | abort
    edited_arguments: dict[str, Any] | None = None
    feedback: str | None = None

    @property
    def approved(self) -> bool:
        return self.decision in ("approve", "always_session")


class ToolChannel(Protocol):
    """What the gateway needs from its surroundings."""

    async def proposed(self, *, call_id: str, tool: str, arguments: dict[str, Any]) -> None:
        """A tool call has been requested by the model, before any checks."""
        ...

    async def started(self, *, call_id: str, tool: str) -> None:
        """A call passed every check and is about to run."""
        ...

    async def finished(
        self,
        *,
        call_id: str,
        tool: str,
        ok: bool,
        summary: str,
        duration_ms: float,
    ) -> None:
        """A call finished, successfully or not."""
        ...

    async def request_approval(self, ask: ApprovalAsk) -> ApprovalReply | None:
        """Ask for a decision, and wait.

        Returning ``None`` means no answer could be obtained — a closed channel, a
        headless run. Callers must treat that as a denial, never as consent
        (docs/safety-and-tool-use.md §1.3).
        """
        ...


#: Badges that disable "approve all" in a batch review (§6.4). One item carrying any of them
#: is enough. Defined here, beside the batch types, so that the terminal prompt, the bus
#: adapter and the gateway all read one list: "approve all is disabled" is only a guarantee
#: if every layer that could grant it agrees on when.
BATCH_BLOCKING_BADGES = ("DESTRUCTIVE", "SECRET?", "PARSE-ERRORS-INTRODUCED")


def batch_blockers(badge_groups: list[tuple[str, ...]] | list[list[str]]) -> list[str]:
    """The blocking badges present across a batch, in a stable order."""
    present = {badge for group in badge_groups for badge in group}
    return [badge for badge in BATCH_BLOCKING_BADGES if badge in present]


@dataclass(frozen=True)
class BatchItem:
    """One write offered for batch review: what would happen, and how to recognise it."""

    call_id: str
    tool: str
    path: str | None
    summary: str
    preview: str
    badges: tuple[str, ...] = ()
    added: int = 0
    removed: int = 0


@runtime_checkable
class BatchChannel(Protocol):
    """An optional channel capability: decide several writes on one screen (§6.4).

    Separate from :class:`ToolChannel` on purpose. A channel that does not implement it —
    the null channel, a headless run, a frontend that predates batches — simply gets no
    batching, and every write is asked about individually as before. Making it part of the
    base protocol would force every implementation to answer a question it may have no way
    to ask.
    """

    async def request_batch_approval(self, items: list[BatchItem]) -> dict[str, str] | None:
        """Decide each item. Returns ``call_id -> "approve" | "reject"``, or ``None`` when no
        answer could be obtained. An item missing from the answer is a rejection."""
        ...


class NullChannel:
    """A channel that reports nothing and denies every approval.

    The safe default. Used by scripts and tests that only exercise read tools: if such a
    caller ever reaches an approval, silently allowing it would be the wrong outcome, so
    this refuses instead.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def proposed(self, *, call_id: str, tool: str, arguments: dict[str, Any]) -> None:
        self.events.append(("proposed", {"call_id": call_id, "tool": tool}))

    async def started(self, *, call_id: str, tool: str) -> None:
        self.events.append(("started", {"call_id": call_id, "tool": tool}))

    async def finished(self, *, call_id: str, tool: str, ok: bool, summary: str, duration_ms: float) -> None:
        self.events.append(("finished", {"call_id": call_id, "tool": tool, "ok": ok}))

    async def request_approval(self, ask: ApprovalAsk) -> ApprovalReply | None:
        return None
