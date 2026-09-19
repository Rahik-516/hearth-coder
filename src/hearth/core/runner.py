"""The agent runner — M3's single-shot chat path.

One turn is: retrieve, build a cache-stable request, stream the answer, record what
happened. No tools yet; the full loop with tool calls and approvals arrives in M4-M6
(docs/system-design.md §8).

Everything the frontend sees is an event (docs/system-design.md §5.1). The runner never
prints, never blocks on input, and never knows whether a terminal or an editor is watching.
That is what makes the same code serve the REPL, a future IDE, and the tests.

Cancellation is first-class: Ctrl+C must stop a stream within a second, which means the
streaming loop has to be interruptible at every chunk rather than between turns.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hearth.core.agent_loop import AgentLoop
from hearth.core.bus import EventBus
from hearth.core.context.budget import Segment, budget_for
from hearth.core.context.builder import BuiltRequest, ContextBuilder
from hearth.core.context.compactor import CompactionPolicy, CompactionResult, compact
from hearth.core.context.tokens import observe
from hearth.core.events import (
    ContextStats,
    ErrorEvent,
    Notice,
    RetrievalPerformed,
    RetrievedSource,
    TextDelta,
    ThinkingDelta,
    TurnEndReason,
    TurnFinished,
    TurnStarted,
)
from hearth.core.limits import TurnLimits
from hearth.core.session import Session
from hearth.llm.errors import LLMError
from hearth.llm.provider import LLMProvider
from hearth.llm.types import ChatRequest, Message, Sampling, ThinkLevel
from hearth.prompts import system_prompt_for
from hearth.retrieval.engine import RetrievalEngine, RetrievalResult
from hearth.retrieval.repomap import RepoMapBuilder
from hearth.tools.gateway import ToolGateway

#: Produces a query vector for dense retrieval.
QueryEmbedder = Callable[[str], Awaitable[np.ndarray]]

#: Loop reasons that mean the turn finished on its own terms. Everything else — a step
#: limit, an exhausted retry budget, repeated failures — is a turn that stopped early.
_AGENT_SUCCESS_REASONS = frozenset({"answered", "stop", ""})


def _agent_reason(reason: str) -> str:
    return "answered" if reason in _AGENT_SUCCESS_REASONS else reason


@dataclass
class AgentTurnResult:
    """What one agent turn produced.

    Separate from :class:`TurnResult` because the interesting failures differ: a chat turn
    either answered or did not, while an agent turn can end because it ran out of steps,
    kept failing, or had everything it wanted to do refused — and "stopped at the step
    limit" needs to be distinguishable from "finished", or a caller reports a half-done
    change as a success.
    """

    answer: str
    thinking: str = ""
    reason: str = "answered"
    steps: int = 0
    tool_calls: int = 0
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether the turn actually finished.

        Deliberately **not** ``_turn_reason``, which normalises *provider* stop words. A
        loop reason like ``step_limit`` is Hearth's own, and passing it through that
        mapping reports a turn that ran out of steps as "answered" — the one error here
        that would make a half-done change look finished.
        """
        return _agent_reason(self.reason) == "answered"

    def summary(self) -> str:
        """The ``run_result`` line every exit path prints (§14.1).

        Printed on success *and* on every failure, because the common headless outcome is
        under-granting: a run that denied six calls and answered anyway is not a success,
        and without this the only way to find that out is to re-read the transcript.
        """
        parts = [
            f"reason={_agent_reason(self.reason)}",
            f"steps={self.steps}",
            f"tool_calls={self.tool_calls}",
            f"duration={self.duration_ms / 1000:.1f}s",
        ]
        return f"run_result: {' '.join(parts)}"


@dataclass
class TurnResult:
    """What one turn produced."""

    answer: str
    thinking: str = ""
    reason: str = "answered"
    citations: list[str] = field(default_factory=list)
    prompt_tokens: int | None = None
    estimated_prompt_tokens: int = 0
    duration_ms: float = 0.0
    #: The *prompt* was cut by the server.
    truncated: bool = False
    #: The *answer* stopped at the output limit rather than finishing.
    output_truncated: bool = False

    @property
    def ok(self) -> bool:
        """Whether the turn produced an answer.

        Normalized through the same mapping the event uses, so a provider saying "stop"
        and one saying "answered" are not treated differently.
        """
        return _turn_reason(self.reason) == "answered"


class ChatRunner:
    """Runs single-shot chat turns: retrieve, build, stream.

    Args:
        provider: Where completions come from.
        bus: Where events go.
        engine: Retrieval, or None to answer without repository context.
        embed_query: Optional callable producing a query vector for dense retrieval.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        bus: EventBus,
        engine: RetrievalEngine | None = None,
        embed_query: QueryEmbedder | None = None,
        retrieval_limit: int = 8,
        repo_map: RepoMapBuilder | None = None,
    ) -> None:
        self._provider = provider
        self._bus = bus
        self._engine = engine
        self._embed_query = embed_query
        self._retrieval_limit = retrieval_limit
        self._repo_map = repo_map
        #: (session id, epoch) -> rendered map. Keyed by epoch because the map sits in the
        #: cached prefix: rebuilding it mid-session would change those bytes and throw the
        #: KV cache away on every turn (docs/system-design.md §9.2).
        self._map_cache: dict[tuple[str, int], str] = {}

    async def run_turn(self, session: Session, user_text: str) -> TurnResult:
        """Run one turn end to end."""
        started = time.perf_counter()
        await self._bus.publish(
            TurnStarted(session_id=session.id, mode=session.mode.value, model=session.model)
        )

        try:
            # Before the turn, not after: the turn that would have overflowed is the one
            # that benefits, and the builder's truncation is a last resort, not the plan.
            await self.maybe_compact(session)
            retrieval = await self._retrieve(session, user_text)
            request = self._build(session, user_text, retrieval)
            result = await self._stream(session, request)
        except asyncio.CancelledError:
            # Cancellation is a normal outcome, not an error: the user pressed Ctrl+C.
            await self._bus.publish(TurnFinished(session_id=session.id, reason="aborted", steps=1))
            raise
        except LLMError as exc:
            await self._bus.publish(ErrorEvent(message=str(exc)))
            await self._bus.publish(TurnFinished(session_id=session.id, reason="error", steps=1))
            return TurnResult(answer="", reason="error", duration_ms=_ms_since(started))

        result.duration_ms = _ms_since(started)
        session.add_user(user_text)
        session.add_assistant(result.answer, thinking=result.thinking or None)

        if result.output_truncated:
            await self._bus.publish(
                Notice(
                    level="warning",
                    message=(
                        "The reply stopped at the output limit rather than finishing. "
                        "Raise the output reserve, or ask a narrower question."
                    ),
                )
            )

        await self._bus.publish(
            TurnFinished(
                session_id=session.id,
                reason=_turn_reason(result.reason),
                steps=1,
                duration_ms=result.duration_ms,
            )
        )
        return result

    async def run_agent_turn(
        self,
        session: Session,
        user_text: str,
        *,
        gateway: ToolGateway,
        tool_schemas: list[dict[str, Any]],
        limits: TurnLimits | None = None,
        think: ThinkLevel = "off",
    ) -> AgentTurnResult:
        """Run one agent turn: retrieve, build, then loop with tools until it answers.

        Deliberately a method on this class rather than a second runner. The prefix is
        cache-stable only if exactly one piece of code decides its layout — system prompt,
        project instructions, append-only history, then the user message carrying the
        retrieved context (docs/system-design.md §9.2). A separate agent runner would be a
        second implementation of that layout, and the two would drift by a word, which is
        all it takes: measured on the reference machine, a reused prefix prefills 28x
        faster than a cold one.

        Retrieval still happens once, up front, even though the loop can also search. The
        first search is nearly always worth it, and the loop's own `grep` calls are for
        following up on what the context showed rather than for starting from nothing.
        """
        started = time.perf_counter()
        await self._bus.publish(
            TurnStarted(session_id=session.id, mode=session.mode.value, model=session.model)
        )

        loop = AgentLoop(
            provider=self._provider,
            gateway=gateway,
            bus=self._bus,
            limits=limits,
            tool_schemas=tool_schemas,
        )

        try:
            await self.maybe_compact(session)
            retrieval = await self._retrieve(session, user_text)
            built = self._build(session, user_text, retrieval)
            base = ChatRequest(
                model=session.model,
                messages=built.messages,
                num_ctx=session.num_ctx,
                # Passed in rather than derived from the mode: whether a model should think
                # is a property of the model's profile, which this layer cannot see.
                think=think,
            )
            outcome = await loop.run(base_request=base, on_text=self._emit_text)
        except asyncio.CancelledError:
            await self._bus.publish(TurnFinished(session_id=session.id, reason="aborted", steps=1))
            raise
        except LLMError as exc:
            await self._bus.publish(ErrorEvent(message=str(exc)))
            await self._bus.publish(TurnFinished(session_id=session.id, reason="error", steps=1))
            return AgentTurnResult(
                answer="", reason="error", duration_ms=_ms_since(started), steps=1
            )

        session.add_user(user_text)
        session.add_assistant(outcome.answer, thinking=outcome.thinking or None)

        duration = _ms_since(started)
        await self._bus.publish(
            TurnFinished(
                session_id=session.id,
                reason=_turn_reason(outcome.reason),
                steps=outcome.steps,
                duration_ms=duration,
            )
        )
        return AgentTurnResult(
            answer=outcome.answer,
            thinking=outcome.thinking,
            reason=outcome.reason,
            steps=outcome.steps,
            tool_calls=outcome.tool_calls,
            duration_ms=duration,
        )

    async def _emit_text(self, delta: str) -> None:
        await self._bus.publish(TextDelta(text=delta))

    # -------------------------------------------------------------- internals

    async def _retrieve(self, session: Session, user_text: str) -> RetrievalResult | None:
        """Pre-retrieval. Failure degrades the turn rather than ending it."""
        if self._engine is None:
            return None

        query_vector = None
        if self._embed_query is not None:
            try:
                query_vector = await self._embed_query(user_text)
            except LLMError:
                # Dense is optional; the lexical retrievers still answer.
                query_vector = None

        result = await asyncio.to_thread(
            self._engine.retrieve,
            user_text,
            limit=self._retrieval_limit,
            query_vector=query_vector,
        )

        await self._bus.publish(
            RetrievalPerformed(
                query=user_text,
                duration_ms=result.duration_ms,
                sources=[
                    RetrievedSource(
                        path=found.path,
                        start_line=found.start_line,
                        end_line=found.end_line,
                        score=found.score,
                        retriever=",".join(r.value for r in found.retrievers),
                    )
                    for found in result.results
                ],
            )
        )

        if result.coverage[1] and not result.is_fully_embedded:
            embedded, total = result.coverage
            await self._bus.publish(
                Notice(
                    level="info",
                    message=(
                        f"Embeddings are {embedded}/{total} complete; "
                        f"semantic search is partial. Run `hearth embed` to finish."
                    ),
                )
            )
        return result

    def _build(self, session: Session, user_text: str, retrieval: RetrievalResult | None) -> BuiltRequest:
        budget = budget_for(session.num_ctx)
        builder = ContextBuilder(budget=budget, estimator=session.estimator)
        request = builder.build(
            system_prompt=system_prompt_for(session.mode.value),
            user_message=user_text,
            history=session.history,
            retrieved=retrieval.results if retrieval else None,
            repo_map=self._repo_map_for(session, budget.limit(Segment.REPO_MAP)),
        )
        return request

    async def maybe_compact(self, session: Session, *, force: bool = False) -> CompactionResult:
        """Compact this session's history when it is close to filling the window.

        Called before a turn is built, and by ``/compact``. Checked *before* rather than
        after, so the turn that would have overflowed is the one that benefits — the
        context builder's own truncation is a last resort, not the plan.
        """
        policy = CompactionPolicy(budget=budget_for(session.num_ctx), estimator=session.estimator)
        overhead = self._fixed_overhead(session)

        if not force and not policy.should_compact(session.history, overhead_tokens=overhead):
            return CompactionResult(history=list(session.history))

        result = await compact(
            session.history,
            policy=policy,
            summarize=lambda prompt: self._summarize(session, prompt),
            pinned_facts=list(session.pinned_paths),
        )

        if result.compacted:
            session.replace_history(result.history)
            await self._bus.publish(
                Notice(
                    level="info",
                    message=(
                        f"Compacted {result.replaced} message(s), freeing about "
                        f"{result.tokens_saved} tokens. The next turn re-prefills once."
                    ),
                )
            )
        elif result.degraded and force:
            # Only surfaced for an explicit `/compact`: an automatic attempt that declined
            # to shrink the history is ordinary, and reporting it every turn is noise.
            await self._bus.publish(Notice(level="warning", message=f"Not compacted: {result.degraded}"))

        return result

    async def _summarize(self, session: Session, prompt: str) -> str:
        """Ask the model for the summary. The only place compaction touches the LLM."""
        request = ChatRequest(
            model=session.model,
            messages=[Message(role="user", content=prompt)],
            num_ctx=session.num_ctx,
            think="off",
            sampling=Sampling(temperature=0.0),
        )

        parts: list[str] = []
        async for chunk in self._provider.chat_stream(request):
            if chunk.content_delta:
                parts.append(chunk.content_delta)
        return "".join(parts)

    def _fixed_overhead(self, session: Session) -> int:
        """Tokens compaction cannot reclaim: the system prompt and the repo map."""
        budget = budget_for(session.num_ctx)
        overhead = session.estimator.estimate(system_prompt_for(session.mode.value))
        repo_map = self._map_cache.get((session.id, session.epoch))
        if repo_map:
            overhead += session.estimator.estimate(repo_map)
        return min(overhead, budget.num_ctx)

    def _repo_map_for(self, session: Session, budget_tokens: int) -> str | None:
        """The repo map for this cache epoch, built once and reused.

        Personalised by what the session has touched or pinned, so the map describes the
        repository from where the user is standing rather than in the abstract.

        A failure here returns None rather than propagating: the map is an aid, and losing
        it should cost the model some context, not cost the user their turn.
        """
        if self._repo_map is None or budget_tokens <= 0:
            return None

        key = (session.id, session.epoch)
        if key in self._map_cache:
            return self._map_cache[key]

        personalization = {path: 1.0 for path in session.touched_paths}
        # Pins are a deliberate "look here" and outrank a file the session merely edited.
        personalization.update({path: 3.0 for path in session.pinned_paths})

        try:
            result = self._repo_map.build(
                budget_tokens=budget_tokens,
                personalization=personalization,
                estimator=session.estimator,
            )
            text = result.text
        except Exception:
            text = ""

        self._map_cache[key] = text
        return text or None

    async def _stream(self, session: Session, request: BuiltRequest) -> TurnResult:
        """Stream a completion, emitting deltas as they arrive."""
        for notice in request.notices:
            await self._bus.publish(Notice(level="warning", message=notice))

        chat_request = ChatRequest(
            model=session.model,
            messages=request.messages,
            num_ctx=session.num_ctx,
            # Thinking is off in chat mode on small windows: reasoning tokens compete with
            # the answer for an output reserve measured in hundreds of tokens
            # (docs/system-design.md §9.1).
            think="off",
            keep_alive="20m",
            sampling=Sampling(),
        )

        answer: list[str] = []
        thinking: list[str] = []
        prompt_tokens: int | None = None
        generation_tps: float | None = None
        prefill_ms: float | None = None
        reason = "answered"

        async for chunk in self._provider.chat_stream(chat_request):
            if chunk.content_delta:
                answer.append(chunk.content_delta)
                await self._bus.publish(TextDelta(text=chunk.content_delta))
            if chunk.thinking_delta:
                thinking.append(chunk.thinking_delta)
                await self._bus.publish(ThinkingDelta(text=chunk.thinking_delta))

            if chunk.done:
                reason = chunk.done_reason or "answered"
                if chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    generation_tps = chunk.usage.generation_tps
                    prefill_ms = chunk.usage.prefill_ms

        truncated = await self._reconcile_tokens(session, request, prompt_tokens)
        cached_tokens = session.observe_prefill(prompt_tokens=prompt_tokens, prefill_ms=prefill_ms)

        await self._bus.publish(
            ContextStats(
                used=request.estimated_prompt_tokens,
                budget=request.usage.budget.input_budget,
                cached_tokens=cached_tokens,
                prefill_ms=prefill_ms,
                generation_tps=generation_tps,
            )
        )

        return TurnResult(
            answer="".join(answer),
            thinking="".join(thinking),
            reason=reason,
            output_truncated=reason == "length",
            citations=request.citations,
            prompt_tokens=prompt_tokens,
            estimated_prompt_tokens=request.estimated_prompt_tokens,
            truncated=truncated,
        )

    async def _reconcile_tokens(
        self, session: Session, request: BuiltRequest, prompt_tokens: int | None
    ) -> bool:
        """Calibrate the estimator, or warn that the prompt was cut.

        This is the only reliable signal that silent truncation happened: the request
        succeeds, the answer looks plausible, and the system prompt was never seen
        (docs/system-design.md §16).
        """
        if prompt_tokens is None:
            return False

        report = observe(
            session.estimator,
            estimated=request.estimated_prompt_tokens,
            actual=prompt_tokens,
        )
        if report.truncated:
            await self._bus.publish(Notice(level="warning", message=report.message()))
        return report.truncated


def _turn_reason(raw: str) -> TurnEndReason:
    """Map a provider's stop reason onto the event vocabulary.

    Providers use their own words, and new ones appear over time. Anything that produced
    output counts as answered; the more specific cases (hitting the output limit) are
    surfaced as a notice instead, because they say something the caller can act on while
    "answered" does not.
    """
    if raw in ("", "stop", "answered", "length"):
        return "answered"
    if raw in ("aborted", "cancelled"):
        return "aborted"
    if raw == "error":
        return "error"
    return "answered"


def _ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000
