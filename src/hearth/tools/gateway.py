"""The tool gateway: validate → prepare → policy → approve → execute → record.

**Every tool call passes through here, in this order** (docs/system-design.md §5.8). The
order is the security model:

* **validate** before prepare, so malformed arguments never reach code that touches the
  filesystem.
* **prepare** before policy, because the policy engine judges the *resolved* action — the
  real argv, the real path — not the model's description of it.
* **policy** before approve, so a hard-denied call is refused without ever asking the user.
  Prompting for something that will be denied trains people to click through prompts.
* **approve** before execute, obviously — and edited arguments go back to the start, so a
  user edit cannot route around a deny rule (docs/system-design.md §5.9).
* **record** always, including for denials and failures. An audit log that only contains
  successes answers the wrong question.

The decision itself is not made here. The gateway is an **adapter**: it turns a prepared
call into the plain :class:`~hearth.safety.policy.PolicyRequest` the pure engine consumes,
and turns the engine's answer back into an approval round trip. The engine cannot import
``tools`` (the ``policy-pure`` contract), which is exactly what keeps it testable without
a filesystem — see ``safety/policy.py``.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any

from hearth.safety.audit import AuditLog, AuditRecord
from hearth.safety.errors import PathError
from hearth.safety.invariants import privilege_escalation_dirs
from hearth.safety.policy import ConfigView, Decision, PolicyRequest, SessionView, evaluate
from hearth.tools.base import Prepared, Tool, ToolContext
from hearth.tools.channel import ApprovalAsk, ApprovalReply, NullChannel, ToolChannel
from hearth.tools.registry import ToolAvailability, ToolRegistry
from hearth.tools.results import ErrorCode, ToolResult, unknown_tool

#: Decides one prepared call. A closure rather than a bound object, so ``core`` can supply
#: a *live* session view — a grant added mid-turn has to be visible to the next call.
PolicyFn = Callable[[PolicyRequest], Decision]

#: Called with a grant key when the user answers "always for this session".
GrantFn = Callable[[str], None]


def make_policy(session: SessionView, config: ConfigView) -> PolicyFn:
    """Bind a session and config view to the engine."""

    def policy(request: PolicyRequest) -> Decision:
        return evaluate(request, session, config)

    return policy


def default_policy() -> PolicyFn:
    """The policy a gateway gets when the caller supplies none: chat mode, supervised.

    Chat rather than agent, because a gateway built without an explicit session is a
    gateway nobody has decided the permissions for — and in chat mode every side-effecting
    risk is refused by the mode restriction. Defaulting to agent would mean a forgotten
    argument silently upgrades what a tool may do.
    """
    return make_policy(
        SessionView(mode="chat", level="supervised"),
        ConfigView(protected_dirs=privilege_escalation_dirs()),
    )


class ToolGateway:
    """Runs tool calls through the full lifecycle."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        context: ToolContext,
        channel: ToolChannel | None = None,
        audit: AuditLog | None = None,
        policy: PolicyFn | None = None,
        on_grant: GrantFn | None = None,
        session_id: str = "",
    ) -> None:
        self._registry = registry
        self._context = context
        self._channel = channel or NullChannel()
        self._audit = audit
        self._policy = policy or default_policy()
        self._on_grant = on_grant
        self._session_id = session_id

    def availability(
        self,
        mode: str,
        *,
        tool_reliability: str = "high",
        max_tools: int | None = None,
    ) -> ToolAvailability:
        """Which tools a mode exposes.

        Asked of the gateway rather than of a separately-held registry, so the schemas the
        model is shown and the tools this gateway will accept are one selection rather than
        two that agree. A frontend that switches mode — `/mode agent`, or `/plan` dropping
        into read-only tools — has to re-ask, and this is where it asks.
        """
        return self._registry.for_mode(
            mode, tool_reliability=tool_reliability, max_tools=max_tools
        )

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        call_id: str,
        step: int = 0,
    ) -> ToolResult:
        """Run one tool call. Never raises for tool-level problems."""
        started = time.perf_counter()

        # The context is per-session but the step is per-call, and a checkpoint has to be
        # attributed to the step `/undo` and `/rewind` will name. Without this every write
        # in a session lands on step 0, and the two commands collapse into "revert
        # everything". Safe to set here because reads never checkpoint, so the concurrent
        # read batch cannot race on it.
        self._context.step = step

        await self._channel.proposed(call_id=call_id, tool=name, arguments=arguments)

        tool = self._registry.get(name)
        if tool is None:
            result = unknown_tool(name, self._registry.names())
            await self._record(name, "unknown", "deny", arguments, result, step, started)
            return result

        # --- validate -----------------------------------------------------
        parsed = tool.validate(arguments)
        if isinstance(parsed, ToolResult):
            await self._record(name, tool.risk.value, "deny", arguments, parsed, step, started)
            return parsed

        # --- prepare ------------------------------------------------------
        try:
            prepared = tool.prepare(parsed, self._context)
        except PathError as exc:
            # The workspace jail is hard-invariant enforcement, so it is audited as such
            # rather than as a tool failure. No rule or grant could have permitted it.
            result = ToolResult.failure(ErrorCode.PATH_REFUSED, str(exc))
            await self._record(
                name,
                tool.risk.value,
                "deny",
                arguments,
                result,
                step,
                started,
                decided_by="invariant",
                rule_id="path-jail",
            )
            return result
        except Exception as exc:
            result = ToolResult.failure(ErrorCode.EXECUTION_FAILED, f"{name} could not prepare: {exc}")
            await self._record(name, tool.risk.value, "error", arguments, result, step, started)
            return result

        if prepared.failed and prepared.error is not None:
            # "unavailable", not "deny": the tool itself refused — a missing file, no index
            # built yet — which is a different fact from policy refusing the call. Reading
            # the audit log to answer "what was blocked?" should not surface these.
            await self._record(
                name,
                tool.risk.value,
                "unavailable",
                arguments,
                prepared.error,
                step,
                started,
                decided_by="tool",
            )
            return prepared.error

        # --- policy -------------------------------------------------------
        decision = self._policy(PolicyRequest(tool=name, risk=tool.risk, facts=prepared.facts))
        if decision.action == "deny":
            result = ToolResult.failure(ErrorCode.DENIED, _denial_text(decision))
            await self._record(
                name,
                tool.risk.value,
                "deny",
                arguments,
                result,
                step,
                started,
                decided_by=decision.decided_by,
                badges=list(decision.badges),
                rule_id=decision.rule_id,
            )
            return result

        # --- approve ------------------------------------------------------
        if decision.action == "ask":
            approved = await self._request_approval(
                tool=tool, prepared=prepared, decision=decision, call_id=call_id
            )
            if approved is None or approved.decision in ("reject", "abort"):
                feedback = (approved.feedback if approved else None) or "no reason given"
                result = ToolResult.failure(ErrorCode.REJECTED, f"REJECTED by user: {feedback}")
                await self._record(
                    name,
                    tool.risk.value,
                    "deny",
                    arguments,
                    result,
                    step,
                    started,
                    decided_by="user",
                    badges=list(decision.badges),
                )
                return result

            if approved.decision == "edit" and approved.edited_arguments is not None:
                # Edited arguments re-enter at the top: re-validated, re-prepared and
                # re-judged. Anything less would let an edit bypass a deny rule.
                return await self.call(name, approved.edited_arguments, call_id=call_id, step=step)

            if approved.decision == "always_session" and decision.grant_key and self._on_grant:
                # Recorded before executing, so a grant survives a failure in the call it
                # was granted for. The user approved the *action*, not its outcome.
                self._on_grant(decision.grant_key)

        # --- execute ------------------------------------------------------
        await self._channel.started(call_id=call_id, tool=name)
        try:
            outcome = tool.execute(parsed, self._context, prepared)
            # The exec tools are async; the filesystem tools are not. Awaiting only what
            # is awaitable keeps that a per-tool choice rather than a change every tool
            # has to absorb.
            result = await outcome if inspect.isawaitable(outcome) else outcome
        except PathError as exc:
            result = ToolResult.failure(ErrorCode.PATH_REFUSED, str(exc))
        except Exception as exc:
            result = ToolResult.failure(ErrorCode.EXECUTION_FAILED, f"{name} failed: {exc}")

        duration = (time.perf_counter() - started) * 1000
        result.duration_ms = duration

        await self._channel.finished(
            call_id=call_id,
            tool=name,
            ok=result.ok,
            summary=prepared.summary,
            duration_ms=duration,
        )
        await self._record(
            name,
            tool.risk.value,
            "allow",
            arguments,
            result,
            step,
            started,
            decided_by=decision.decided_by,
            badges=list(decision.badges),
            rule_id=decision.rule_id,
        )
        return result

    # -------------------------------------------------------------- internals

    async def _request_approval(
        self,
        *,
        tool: Tool[Any],
        prepared: Prepared,
        decision: Decision,
        call_id: str,
    ) -> ApprovalReply | None:
        """Ask, and wait. No answer means deny (docs/safety-and-tool-use.md §1.3)."""
        return await self._channel.request_approval(
            ApprovalAsk(
                call_id=call_id,
                tool=tool.name,
                risk=tool.risk.value,
                preview=prepared.preview or prepared.summary,
                badges=[*decision.badges, *prepared.badges],
                reasons=[decision.reason] if decision.reason else [],
                grant_key=decision.grant_key,
                typed_confirmation=decision.typed_confirmation,
            )
        )

    async def _record(
        self,
        tool: str,
        risk: str,
        decision: str,
        arguments: dict[str, Any],
        result: ToolResult,
        step: int,
        started: float,
        *,
        decided_by: str | None = None,
        badges: list[str] | None = None,
        rule_id: str | None = None,
    ) -> None:
        """Write the audit record. Every call, including refusals."""
        if self._audit is None:
            return

        self._audit.write(
            AuditRecord(
                tool=tool,
                risk=risk,
                decision=decision,
                session=self._session_id,
                step=step,
                args=arguments,
                decided_by=decided_by,
                duration_ms=(time.perf_counter() - started) * 1000,
                error=None if result.ok else result.content[:400],
                badges=badges or [],
                rule_id=rule_id,
            )
        )


def _denial_text(decision: Decision) -> str:
    """The refusal as the model sees it.

    The hint matters more than it looks: in headless runs a denial is the ordinary
    outcome of under-granting (§14.1), and naming the flag that would have permitted the
    call is what makes a failed run fixable without re-reading the transcript.
    """
    text = decision.reason or "denied by policy"
    if decision.hint:
        text = f"{text} ({decision.hint})"
    return text


#: A gateway call, as the runner invokes it.
ToolCaller = Callable[[str, dict[str, Any], str, int], Awaitable[ToolResult]]
