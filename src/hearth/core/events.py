"""The frontend contract: typed events (core -> frontend) and commands (frontend -> core).

Everything a user sees and every way they respond flows through these models. The CLI
renders them; a future IDE extension receives the same objects as JSON-RPC notifications;
a test frontend auto-responds to approvals. That is what makes frontends interchangeable
and the approval flow deterministically testable (docs/system-design.md §5.1).

The load-bearing one is ``ApprovalRequested``. Approvals are mid-loop, asynchronous and
blocking — modelling them as an event with a matching response is what lets the same core
serve a terminal, an editor and a test harness without special cases.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- shared

NoticeLevel = Literal["info", "warning", "error"]
TurnEndReason = Literal["answered", "step_limit", "aborted", "error", "denied"]
ApprovalDecision = Literal["approve", "reject", "edit", "always_session", "abort"]
RiskLevel = Literal["READ", "META", "WRITE", "EXEC", "VCS_WRITE"]


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class _Envelope(BaseModel):
    """Common base. Frozen: an event is a record of something that happened."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=_new_id)
    at: float = Field(default_factory=time.time)


# --------------------------------------------------------------------------- events


class TurnStarted(_Envelope):
    type: Literal["turn_started"] = "turn_started"
    session_id: str
    mode: str
    model: str


class TurnFinished(_Envelope):
    type: Literal["turn_finished"] = "turn_finished"
    session_id: str
    reason: TurnEndReason
    steps: int = 0
    duration_ms: float | None = None


class TextDelta(_Envelope):
    """A piece of streamed answer text."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(_Envelope):
    """A piece of streamed reasoning. Rendered collapsed — it is not the answer."""

    type: Literal["thinking_delta"] = "thinking_delta"
    text: str


class RetrievedSource(BaseModel):
    """One chunk injected into context, for transparency about what the model saw."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    start_line: int
    end_line: int
    score: float | None = None
    retriever: str | None = None


class RetrievalPerformed(_Envelope):
    type: Literal["retrieval_performed"] = "retrieval_performed"
    query: str
    sources: list[RetrievedSource] = Field(default_factory=list)
    duration_ms: float | None = None


class ToolCallProposed(_Envelope):
    type: Literal["tool_call_proposed"] = "tool_call_proposed"
    call_id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ApprovalRequested(_Envelope):
    """**Blocks the tool call until an ``ApprovalResponse`` arrives.**

    ``request_id`` correlates the response. ``badges`` carry the risk signals that make
    unusual operations visually distinct, so routine ones can be approved quickly and
    attention goes where it matters (docs/safety-and-tool-use.md §6.2).
    """

    type: Literal["approval_requested"] = "approval_requested"
    request_id: str
    call_id: str
    tool: str
    risk: RiskLevel
    preview: str
    badges: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    #: Which response options the frontend should offer, e.g. approve/reject/edit.
    options: list[ApprovalDecision] = Field(default_factory=list)
    #: When set, approving requires the user to type this word first.
    typed_confirmation: str | None = None
    grant_key: str | None = None


class ToolStarted(_Envelope):
    type: Literal["tool_started"] = "tool_started"
    call_id: str
    tool: str


class ToolOutputDelta(_Envelope):
    type: Literal["tool_output_delta"] = "tool_output_delta"
    call_id: str
    text: str


class ToolFinished(_Envelope):
    type: Literal["tool_finished"] = "tool_finished"
    call_id: str
    tool: str
    ok: bool
    summary: str = ""
    duration_ms: float | None = None
    exit_code: int | None = None


class ContextStats(_Envelope):
    """Budget and performance visibility — the per-turn stats line.

    ``cached_tokens`` is the prefix-cache payoff. A number well below ``used`` on a
    follow-up turn means the stable prefix was broken and prefill is being redone
    (docs/system-design.md §9.2).
    """

    type: Literal["context_stats"] = "context_stats"
    used: int
    budget: int
    cached_tokens: int | None = None
    prefill_ms: float | None = None
    generation_tps: float | None = None


class IndexProgress(_Envelope):
    type: Literal["index_progress"] = "index_progress"
    phase: str
    done: int
    total: int


class Notice(_Envelope):
    """A warning the user should see — stale index, truncated context, dropped rules."""

    type: Literal["notice"] = "notice"
    level: NoticeLevel = "info"
    message: str


class ErrorEvent(_Envelope):
    type: Literal["error"] = "error"
    message: str
    detail: str | None = None


Event = Annotated[
    TurnStarted
    | TurnFinished
    | TextDelta
    | ThinkingDelta
    | RetrievalPerformed
    | ToolCallProposed
    | ApprovalRequested
    | ToolStarted
    | ToolOutputDelta
    | ToolFinished
    | ContextStats
    | IndexProgress
    | Notice
    | ErrorEvent,
    Field(discriminator="type"),
]


# ------------------------------------------------------------------------- commands


class SendMessage(_Envelope):
    """A user turn. ``@path`` mentions arrive as ``pinned_paths``."""

    type: Literal["send_message"] = "send_message"
    text: str
    pinned_paths: list[str] = Field(default_factory=list)


class ApprovalResponse(_Envelope):
    """The answer to an ``ApprovalRequested``.

    ``edited_arguments`` are **re-prepared and re-evaluated by policy** before running, so
    editing a command cannot be used to slip past a deny rule
    (docs/system-design.md §5.9).
    """

    type: Literal["approval_response"] = "approval_response"
    request_id: str
    decision: ApprovalDecision
    edited_arguments: dict[str, Any] | None = None
    feedback: str | None = None


class Cancel(_Envelope):
    """Ctrl+C: stop the current generation or tool."""

    type: Literal["cancel"] = "cancel"


class SetMode(_Envelope):
    type: Literal["set_mode"] = "set_mode"
    mode: str


class SetModel(_Envelope):
    type: Literal["set_model"] = "set_model"
    model: str


class SlashCommand(_Envelope):
    type: Literal["slash_command"] = "slash_command"
    name: str
    args: str = ""


Command = Annotated[
    SendMessage | ApprovalResponse | Cancel | SetMode | SetModel | SlashCommand,
    Field(discriminator="type"),
]
