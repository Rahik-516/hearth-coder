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
from hearth.llm.errors import MalformedOutputError
from hearth.llm.tool_call_parser import parse_tool_calls
from hearth.llm.types import ChatRequest, Message, ToolCall
from hearth.tools.gateway import ToolGateway
from hearth.tools.results import ToolResult

#: Extra attempts when the server rejects the model's own tool-call output. Two, because
#: a sample that fails to parse is independent of the last, and three consecutive broken
#: ones is the model being unable to do it rather than bad luck.
MAX_MALFORMED_RETRIES = 2


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
                # This step's tool calls are not going to run, so the assistant message
                # that requested them is dropped: a transcript with tool calls that never
                # reported back confuses some chat templates.
                messages.pop()
                return await self._wrap_up(
                    base_request,
                    messages,
                    reason=stop,
                    answer_parts=answer_parts,
                    thinking_parts=thinking_parts,
                    fallback=text,
                    steps=tracker.steps,
                    tool_calls=tool_calls_made,
                    usage=usage,
                    on_text=on_text,
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
                return await self._wrap_up(
                    base_request,
                    messages,
                    reason="loop_detected",
                    answer_parts=answer_parts,
                    thinking_parts=thinking_parts,
                    fallback=text,
                    steps=tracker.steps,
                    tool_calls=tool_calls_made,
                    usage=usage,
                    on_text=on_text,
                )

    # -------------------------------------------------------------- internals

    async def _wrap_up(
        self,
        base_request: ChatRequest,
        messages: list[Message],
        *,
        reason: str,
        answer_parts: list[str],
        thinking_parts: list[str],
        fallback: str,
        steps: int,
        tool_calls: int,
        usage: Any,
        on_text: Any,
    ) -> LoopOutcome:
        """End a stopped turn with an answer instead of silence.

        Every early stop — step limit, retry budget, repeated failures, a repeated call —
        used to return whatever text happened to accompany the last tool call, which is
        usually none. The observed case was the worst one: the change was made and the
        tests passed, then the model re-ran the same call until the loop cut it off, and
        the user was left with an empty ending and no way to tell it had worked.

        The wrap-up request has tools withdrawn, so the only thing the model can produce
        is a summary. If even that fails the caller still gets the reason and whatever text
        there was, rather than an exception replacing the partial answer.
        """
        summary = await self._final_answer(base_request, messages, on_text=on_text)
        answer = summary.strip() or "".join(answer_parts) or fallback
        return LoopOutcome(
            answer=answer,
            thinking="".join(thinking_parts),
            reason=reason,
            steps=steps,
            tool_calls=tool_calls,
            messages=messages,
            usage=usage,
        )

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
            if len(calls) > 1:
                # Several writes in one step: put them to the user on one screen rather
                # than as a run of separate prompts (docs/safety-and-tool-use.md §6.4).
                # A pre-pass only — each call still goes through the gateway in proposal
                # order, and the gateway decides what, if anything, the answers replace.
                await self._gateway.review_batch(
                    [(call.name, call.arguments, call.call_id) for call in calls]
                )
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
                            "You cannot call any more tools this turn. Answer now using only "
                            "what you already gathered. Say what you did and what you "
                            "verified, citing path:line where you can. If the task is not "
                            "finished, say precisely what is still missing."
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
        """One completion, retried when the model's own output was the problem.

        A small model sometimes emits a tool call the server cannot parse, and on the first
        live plan-mode eval that ended a whole task on one bad sample. It is a sampling
        failure, so the next attempt usually succeeds — unlike a server that is down, which
        is why only :class:`MalformedOutputError` is retried and every other provider error
        still ends the turn at once. Bounded, and announced, so a model that cannot produce
        a valid call is reported rather than looped on.
        """
        for attempt in range(MAX_MALFORMED_RETRIES + 1):
            try:
                return await self._stream_once(request, on_text=on_text, on_thinking=on_thinking)
            except MalformedOutputError:
                if attempt == MAX_MALFORMED_RETRIES:
                    raise
                await self._notice(
                    f"The model produced a malformed tool call; retrying "
                    f"({attempt + 1}/{MAX_MALFORMED_RETRIES})."
                )
        raise AssertionError("unreachable")  # pragma: no cover

    async def _stream_once(
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
