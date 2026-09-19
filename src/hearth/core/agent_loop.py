"""The multi-step agent loop.

M3's runner answers in one shot. This adds the loop: the model may call tools, read their
results, and continue until it answers or hits a bound (docs/system-design.md §8.1).

The loop is deliberately small, and every exit is explicit:

* the model answers with no tool calls — done
* the step limit is reached — return what we have, labelled
* the retry budget is exhausted — the model cannot get the schema right; stop
* an identical call repeats — nudge once, then stop
* cancellation — propagate immediately

**Read tools run concurrently**, because they are the common case and they are
independent: three `read_file` calls should cost one round trip, not three. Only tools
marked ``concurrent_safe`` are batched, which today means only reads.

The loop never decides whether a tool is *allowed*. That is the gateway's job, and keeping
the two separate is what stops a loop-level shortcut from bypassing policy.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from hearth.core.bus import EventBus
from hearth.core.events import Notice
from hearth.core.limits import StepTracker, TurnLimits
from hearth.llm.tool_call_parser import parse_tool_calls
from hearth.llm.types import ChatRequest, Message, ToolCall
from hearth.tools.gateway import ToolGateway
from hearth.tools.results import ToolResult


@dataclass
class LoopOutcome:
    """What the loop produced."""

    answer: str
    thinking: str = ""
    reason: str = "answered"
    steps: int = 0
    tool_calls: int = 0
    messages: list[Message] = field(default_factory=list)
    usage: Any = None

    @property
    def ok(self) -> bool:
        return self.reason in ("answered", "stop", "")


class AgentLoop:
    """Runs a turn that may use tools."""

    def __init__(
        self,
        *,
        provider: Any,
        gateway: ToolGateway,
        bus: EventBus,
        limits: TurnLimits | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> None:
        self._provider = provider
        self._gateway = gateway
        self._bus = bus
        self._limits = limits or TurnLimits()
        self._schemas = tool_schemas or []
        self._known = {str(schema["function"]["name"]) for schema in self._schemas if "function" in schema}

    async def run(
        self,
        *,
        base_request: ChatRequest,
        on_text: Any = None,
        on_thinking: Any = None,
    ) -> LoopOutcome:
        """Drive the loop until the model answers or a bound is hit.

        ``base_request`` carries the cache-stable prefix built for this turn. Each
        iteration appends to its message list rather than rebuilding it, so the prefix
        stays byte-identical across steps and the KV cache survives the whole turn.
        """
        tracker = StepTracker(limits=self._limits)
        messages = list(base_request.messages)
        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        usage = None
        tool_calls_made = 0

        while True:
            tracker.begin_step()

            request = base_request.model_copy(update={"messages": messages, "tools": self._schemas})
            text, thinking, native_calls, chunk_usage, done_reason = await self._stream(
                request, on_text=on_text, on_thinking=on_thinking
            )
            usage = chunk_usage or usage

            outcome = parse_tool_calls(native=native_calls, text=text, known_tools=self._known or None)

            if not outcome.found:
                answer_parts.append(text)
                thinking_parts.append(thinking)
                return LoopOutcome(
                    answer="".join(answer_parts),
                    thinking="".join(thinking_parts),
                    reason=done_reason or "answered",
                    steps=tracker.steps,
                    tool_calls=tool_calls_made,
                    messages=messages,
                    usage=usage,
                )

            # The assistant turn that requested the tools has to be in history before
            # their results, or the transcript stops making sense to the model.
            messages.append(Message(role="assistant", content=text, tool_calls=outcome.calls))

            stop = tracker.stop_reason()
            if stop is not None:
                await self._notice(f"Stopping: {stop.replace('_', ' ')}. Answering with what is available.")
                return LoopOutcome(
                    answer="".join(answer_parts) or text,
                    thinking="".join(thinking_parts),
                    reason=stop,
                    steps=tracker.steps,
                    tool_calls=tool_calls_made,
                    messages=messages,
                    usage=usage,
                )

            results = await self._run_calls(outcome.calls, tracker, step=tracker.steps)
            tool_calls_made += len(results)

            for call, result, nudge in results:
                messages.append(
                    Message(
                        role="tool",
                        content=_with_nudge(result.render_for_model(), nudge),
                        tool_call_id=call.call_id,
                        tool_name=call.name,
                    )
                )

            if any(nudge and nudge.startswith("STOP:") for _, _, nudge in results):
                return LoopOutcome(
                    answer="".join(answer_parts) or text,
                    thinking="".join(thinking_parts),
                    reason="loop_detected",
                    steps=tracker.steps,
                    tool_calls=tool_calls_made,
                    messages=messages,
                    usage=usage,
                )

    # -------------------------------------------------------------- internals

    async def _run_calls(
        self, calls: list[ToolCall], tracker: StepTracker, *, step: int
    ) -> list[tuple[ToolCall, ToolResult, str | None]]:
        """Execute a batch of calls, concurrently when all of them are read-only.

        Concurrency is opt-in per tool rather than assumed: two reads are independent, but
        two writes are not, and the gateway's approval flow is sequential by design.
        """
        nudges = {call.call_id: tracker.observe_call(call.name, call.arguments) for call in calls}

        registry = self._gateway._registry
        all_concurrent = all(
            (tool := registry.get(call.name)) is not None and tool.concurrent_safe for call in calls
        )

        async def invoke(call: ToolCall) -> ToolResult:
            return await self._gateway.call(call.name, call.arguments, call_id=call.call_id, step=step)

        if all_concurrent and len(calls) > 1:
            results = await asyncio.gather(*(invoke(call) for call in calls))
        else:
            results = [await invoke(call) for call in calls]

        for result in results:
            tracker.record_result(ok=result.ok, retryable=result.is_retryable)

        return [(call, result, nudges[call.call_id]) for call, result in zip(calls, results, strict=True)]

    async def _final_answer(
        self,
        base_request: ChatRequest,
        messages: list[Message],
        *,
        on_text: Any,
    ) -> str:
        """One last completion with tools removed, so the turn ends with an answer.

        Failure here is swallowed deliberately: this runs *because* something already went
        wrong, and an exception would replace a partial answer with none at all.
        """
        request = base_request.model_copy(
            update={
                "messages": [
                    *messages,
                    Message(
                        role="user",
                        content=(
                            "You have run out of tool budget for this turn. "
                            "Answer now using only what you already gathered, with "
                            "path:line citations. If it is not enough, say precisely "
                            "what is still missing."
                        ),
                    ),
                ],
                "tools": [],
            }
        )
        try:
            text, _, _, _, _ = await self._stream(request, on_text=on_text, on_thinking=None)
        except Exception:
            return ""
        return text

    async def _stream(
        self, request: ChatRequest, *, on_text: Any, on_thinking: Any
    ) -> tuple[str, str, list[ToolCall], Any, str]:
        """One completion, accumulating text, thinking and native tool calls."""
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        calls: list[ToolCall] = []
        usage = None
        done_reason = ""

        async for chunk in self._provider.chat_stream(request):
            if chunk.content_delta:
                text_parts.append(chunk.content_delta)
                if on_text is not None:
                    await on_text(chunk.content_delta)
            if chunk.thinking_delta:
                thinking_parts.append(chunk.thinking_delta)
                if on_thinking is not None:
                    await on_thinking(chunk.thinking_delta)
            if chunk.tool_calls:
                calls.extend(chunk.tool_calls)
            if chunk.done:
                usage = chunk.usage
                done_reason = chunk.done_reason or ""

        return "".join(text_parts), "".join(thinking_parts), calls, usage, done_reason

    async def _notice(self, message: str) -> None:
        await self._bus.publish(Notice(level="warning", message=message))


def _with_nudge(rendered: str, nudge: str | None) -> str:
    """Append a loop nudge to a tool result.

    Delivered as part of the result rather than as a separate message so it arrives in the
    same place the model is already reading, and cannot be mistaken for user input.
    """
    return f"{rendered}\n\n{nudge}" if nudge else rendered
