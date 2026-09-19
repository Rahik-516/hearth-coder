"""Tool results.

**Tool errors are returned, never raised** (docs/project-structure.md §4). A raised
exception unwinds past the runner and ends the turn; a returned error becomes a message
the model can read and act on. The difference decides whether a mistyped argument costs
one retry or the whole conversation.

Error text is written *for the model*, not for a log. "path not found: src/biling.py — did
you mean src/billing.py?" produces a corrected retry; "FileNotFoundError" produces another
identical call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ErrorCode(StrEnum):
    """Why a tool call failed. The model sees the message, not this code."""

    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    NOT_FOUND = "not_found"
    PATH_REFUSED = "path_refused"
    DENIED = "denied"
    REJECTED = "rejected"
    STALE_FILE = "stale_file"
    TOO_LARGE = "too_large"
    EXECUTION_FAILED = "execution_failed"
    NOT_IMPLEMENTED = "not_implemented"
    CANCELLED = "cancelled"


@dataclass
class ToolResult:
    """What a tool produced.

    ``content`` is what the model sees. ``display`` is what the user sees, when the two
    should differ — a diff rendered for a human versus the same edit described for a model.
    """

    ok: bool
    content: str = ""
    error: ErrorCode | None = None
    display: str | None = None
    #: Structured data for the frontend: line counts, match counts, paths touched.
    metadata: dict[str, object] = field(default_factory=dict)
    #: Full output when `content` was truncated for the context budget.
    output_id: str | None = None
    duration_ms: float | None = None

    @classmethod
    def success(
        cls,
        content: str,
        *,
        display: str | None = None,
        **metadata: object,
    ) -> ToolResult:
        return cls(ok=True, content=content, display=display, metadata=metadata)

    @classmethod
    def failure(cls, error: ErrorCode, message: str, **metadata: object) -> ToolResult:
        """A failure the model should be able to recover from.

        The message is phrased as guidance because it will be fed straight back as the
        tool's output, and the model's next action is a direct response to it.
        """
        return cls(ok=False, content=message, error=error, metadata=metadata)

    def render_for_model(self) -> str:
        """The string appended to the conversation as this tool's output."""
        if self.ok:
            return self.content
        return f"ERROR {self.error.value if self.error else 'failed'}: {self.content}"

    @property
    def is_retryable(self) -> bool:
        """Whether retrying with corrected arguments could plausibly work.

        A denial or a rejection is a decision, not a mistake — retrying it wastes a step
        and, worse, looks like the model arguing with the user.
        """
        return self.error in {
            ErrorCode.INVALID_ARGUMENTS,
            ErrorCode.UNKNOWN_TOOL,
            ErrorCode.NOT_FOUND,
            ErrorCode.STALE_FILE,
        }


def unknown_tool(name: str, available: list[str]) -> ToolResult:
    """Error for a tool the model invented.

    Lists the real tools, because a model that hallucinated `search_files` will otherwise
    hallucinate `file_search` next.
    """
    suggestions = _closest(name, available)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    return ToolResult.failure(
        ErrorCode.UNKNOWN_TOOL,
        f"No tool named {name!r}. Available tools: {', '.join(sorted(available))}.{hint}",
    )


def invalid_arguments(tool: str, detail: str, *, schema_hint: str | None = None) -> ToolResult:
    """Error for arguments that failed validation."""
    hint = f" Expected: {schema_hint}" if schema_hint else ""
    return ToolResult.failure(
        ErrorCode.INVALID_ARGUMENTS,
        f"Invalid arguments for {tool}: {detail}.{hint}",
    )


def _closest(name: str, candidates: list[str], *, limit: int = 3) -> list[str]:
    import difflib

    return difflib.get_close_matches(name, candidates, n=limit, cutoff=0.5)
