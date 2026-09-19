"""Parsing tool calls, tolerantly.

Native ``tool_calls`` first. When they are absent, common text formats are parsed as a
fallback, because local models emit tool calls as prose more often than hosted ones do
(docs/system-design.md §5.4).

Being tolerant here is not indulgence. A 4B model that writes a correct call in a fenced
JSON block has understood the task; refusing it because of the wrapper wastes a step and
teaches nothing. What is *not* tolerated is a call that fails validation — that returns a
corrective error, so the model learns the actual schema.

Every parsed call is validated against the registry before it is accepted, so a tolerant
parse can never widen what a tool will do.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from hearth.llm.types import ToolCall

#: <tool_call>{...}</tool_call>, used by several Qwen-family templates.
_TAGGED = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

#: ```json { ... } ``` or a bare fenced block.
_FENCED = re.compile(r"```(?:json|tool_code)?\s*(\{.*?\})\s*```", re.DOTALL)

#: A bare object at the start of a line that mentions a name/arguments pair.
_BARE_OBJECT = re.compile(r'^\s*(\{[^\n]*"(?:name|tool)"\s*:.*\})\s*$', re.MULTILINE | re.DOTALL)

#: Keys a model might use for the tool name and its arguments.
_NAME_KEYS = ("name", "tool", "tool_name", "function")
_ARG_KEYS = ("arguments", "args", "parameters", "input")


@dataclass
class ParseOutcome:
    """What a parse produced, and how."""

    calls: list[ToolCall] = field(default_factory=list)
    source: str = "none"  # native | tagged | fenced | bare | none
    #: Text that looked like a tool call but could not be parsed. Worth telling the model.
    malformed: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.calls)


def parse_tool_calls(
    *,
    native: list[ToolCall] | None = None,
    text: str = "",
    known_tools: set[str] | None = None,
) -> ParseOutcome:
    """Extract tool calls from a response.

    Args:
        native: Calls the provider already parsed. Always preferred.
        text: The assistant's text, searched only when there are no native calls.
        known_tools: When given, parsed names are checked against it. An unrecognised
            name in *text* is almost always prose about a tool rather than a call —
            "you could use read_file here" must not execute anything.
    """
    if native:
        return ParseOutcome(calls=list(native), source="native")

    if not text.strip():
        return ParseOutcome()

    seen_malformed: list[str] = []
    for source, pattern in (("tagged", _TAGGED), ("fenced", _FENCED), ("bare", _BARE_OBJECT)):
        calls, malformed = _extract(pattern, text, known_tools)
        if calls:
            return ParseOutcome(calls=calls, source=source, malformed=malformed)
        seen_malformed.extend(malformed)

    # Carried through even on total failure: this is precisely when the caller needs it,
    # because text that *looked* like a call and did not parse is what the corrective
    # hint is for. Dropping it here left the model with silence instead of a fix.
    return ParseOutcome(malformed=seen_malformed)


def _extract(
    pattern: re.Pattern[str], text: str, known_tools: set[str] | None
) -> tuple[list[ToolCall], list[str]]:
    calls: list[ToolCall] = []
    malformed: list[str] = []

    for match in pattern.finditer(text):
        blob = match.group(1)
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            malformed.append(blob[:200])
            continue

        call = _to_tool_call(payload, known_tools)
        if call is not None:
            calls.append(call)

    return calls, malformed


def _to_tool_call(payload: Any, known_tools: set[str] | None) -> ToolCall | None:
    if not isinstance(payload, dict):
        return None

    name = None
    for key in _NAME_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            name = value
            break
        # {"function": {"name": ..., "arguments": {...}}}
        if isinstance(value, dict):
            nested = value.get("name")
            if isinstance(nested, str) and nested:
                name = nested
                payload = {**payload, **value}
                break

    if not name:
        return None
    if known_tools is not None and name not in known_tools:
        # Prose mentioning a tool is not a call. Executing on a name we do not recognise
        # would also mean the corrective error says "unknown tool" for text that was never
        # meant as a call.
        return None

    arguments: dict[str, Any] = {}
    for key in _ARG_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            arguments = value
            break
        if isinstance(value, str):
            # Some templates emit arguments as a JSON *string*.
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                arguments = decoded
                break

    return ToolCall(call_id=uuid.uuid4().hex[:12], name=name, arguments=arguments)


def format_parse_hint(outcome: ParseOutcome, known_tools: set[str]) -> str | None:
    """A corrective message for text that looked like a call but was not usable.

    Returned to the model as the tool result, so the next attempt has something concrete
    to fix rather than a silent non-response.
    """
    if outcome.found or not outcome.malformed:
        return None

    return (
        "That looked like a tool call but could not be parsed as JSON. "
        "Emit a tool call using the provided function-calling format, with valid JSON "
        f"arguments. Available tools: {', '.join(sorted(known_tools))}."
    )
