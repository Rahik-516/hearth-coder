"""The terminal approval prompt — docs/safety-and-tool-use.md §6.

A bus subscriber. It receives ``ApprovalRequested``, renders the panel, asks, and resolves
the request with an ``ApprovalResponse``. The core never learns that a terminal exists,
which is what makes an IDE or a web frontend a drop-in replacement
(docs/system-design.md §5.1).

The design is shaped by the failure mode, which is not a crash but **approval fatigue**
(T2). Someone who has answered forty similar prompts is not reading the forty-first. So:

* **"Always for this session" is only offered when policy issued a grant key**, and is
  withheld again for alarming badges. Offering a grant the engine would not honour is
  worse than not offering one, because the user believes they have stopped being asked.
* **A bare Enter approves an ordinary edit, and decides nothing for a destructive one.**
  The reflex keypress has to be safe.
* **A wrong typed confirmation rejects rather than re-asking.** Re-asking lets someone
  guess their way through a prompt they have already shown they are not reading.
* **Unrecognised input re-prompts.** Never guessed at, because a guess could approve.

The prompt runs *inside* the subscriber, awaiting the user, and that is deliberate: the
core is parked in ``bus.request_approval`` and the turn should be blocked. The ask is
awaited rather than blocking, so the event loop keeps running and Ctrl+C still works.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape

from hearth.cli.diff_view import render_approval, render_diff, render_options
from hearth.core.bus import EventBus
from hearth.core.events import (
    ApprovalDecision,
    ApprovalRequested,
    ApprovalResponse,
)
from hearth.tools.channel import BATCH_BLOCKING_BADGES

#: Badges that withdraw the "always for this session" offer even if a key was issued
#: (docs/safety-and-tool-use.md §6.1). Mirrors the engine's own list, as defence in depth:
#: if the two ever disagree, the UI must be the stricter one.
UNGRANTABLE_BADGES = frozenset({"DESTRUCTIVE", "SHELL", "NETWORK?", "WIN-INTEROP", "INLINE-CODE"})


#: A pseudo-decision: `d` shows the full diff and returns to the prompt.
SHOW_DIFF = "show_diff"

Ask = Callable[[str], Awaitable[str]]
Editor = Callable[[str], str]


@dataclass(frozen=True)
class Choice:
    """One answer the prompt offers."""

    key: str
    decision: str
    label: str
    #: Whether a bare Enter selects this.
    default: bool = False


def choices_for(request: ApprovalRequested) -> tuple[Choice, ...]:
    """The answers to offer for one request.

    Approve is first because it is the common answer, and last in the list is the one
    that ends the turn — so the destructive option is never adjacent to the default.
    """
    badges = set(request.badges)
    grantable = bool(request.grant_key) and not (UNGRANTABLE_BADGES & badges)
    # A destructive call has no default: the user has to choose it explicitly.
    default = not needs_typed_confirmation(request)

    choices = [Choice("y", "approve", "[y] approve", default=default)]
    if grantable:
        choices.append(Choice("s", "always_session", "[s] approve all for this session"))
    choices.extend(
        [
            Choice("e", "edit", "[e] edit"),
            Choice("n", "reject", "[n] reject + feedback"),
            Choice("d", SHOW_DIFF, "[d] full diff"),
            Choice("q", "abort", "[q] abort task"),
        ]
    )
    return tuple(choices)


def _batch_choices(blockers: list[str]) -> tuple[Choice, ...]:
    """The whole-batch answers. Approve-all is left out entirely when a blocker is present."""
    choices = [Choice("i", "individual", "[i] review one by one", default=True)]
    if not blockers:
        choices.append(Choice("a", "approve", "[a] approve all"))
    choices.extend(
        [
            Choice("n", "reject", "[n] reject all + feedback"),
            Choice("q", "abort", "[q] abort task"),
        ]
    )
    return tuple(choices)


def interpret(raw: str, choices: tuple[Choice, ...]) -> Choice | None:
    """Map a keypress to a choice. ``None`` means re-prompt.

    Never guesses: an unrecognised key on a prompt whose first option is "approve" must
    not be resolved by proximity.
    """
    key = raw.strip().lower()
    if not key:
        return next((choice for choice in choices if choice.default), None)
    return next((choice for choice in choices if choice.key == key), None)


def needs_typed_confirmation(request: ApprovalRequested) -> bool:
    return bool(request.typed_confirmation)


def batch_blockers(requests: list[ApprovalRequested]) -> list[str]:
    """Badges present in a batch that disable "approve all" (§6.4).

    One item is enough. "Approve all" over a list containing a secret or a newly
    introduced syntax error is exactly the click that should not be available quickly.
    """
    present = {badge for request in requests for badge in request.badges}
    return [badge for badge in BATCH_BLOCKING_BADGES if badge in present]


class ApprovalPrompt:
    """Renders approvals and answers them from the terminal."""

    def __init__(
        self,
        *,
        bus: EventBus,
        console: Console | None = None,
        ask: Ask | None = None,
        editor: Editor | None = None,
        arguments_for: Callable[[ApprovalRequested], dict[str, Any]] | None = None,
    ) -> None:
        self._bus = bus
        self._console = console or Console()
        self._ask = ask or _prompt_async
        self._editor = editor or _open_editor
        self._arguments_for = arguments_for or (lambda _: {})

    async def __call__(self, event: object) -> None:
        """Bus subscriber entry point."""
        if not isinstance(event, ApprovalRequested):
            return

        response = await (self._decide_batch(event) if event.items else self._decide(event))
        self._bus.resolve_approval(response)

    # ------------------------------------------------------------ batch review

    async def _decide_batch(self, request: ApprovalRequested) -> ApprovalResponse:
        """One screen for several writes (docs/safety-and-tool-use.md §6.4).

        A bare Enter goes through the files **one by one**, not to "approve all". A batch
        is exactly where approval fatigue is worst — the whole point is to make many
        changes cheap to answer — so the reflex keypress has to land on the careful path,
        and approving everything at once is a letter the person chooses deliberately.
        """
        blockers = [badge for badge in BATCH_BLOCKING_BADGES if badge in request.badges]
        self._render_batch(request, blockers)

        choices = _batch_choices(blockers)
        self._console.print(render_options([choice.label for choice in choices]))

        while True:
            answer = (await self._ask("choice [i]: ")).strip().lower()

            if answer.isdigit() and 1 <= int(answer) <= len(request.items):
                item = request.items[int(answer) - 1]
                self._console.print(render_diff(item.preview, limit=100_000))
                continue

            if answer == "a" and blockers:
                # Not offered, and not honoured when typed anyway.
                self._console.print(
                    f"[yellow]approve-all is unavailable: {', '.join(blockers)}. "
                    "Review the files one by one.[/yellow]"
                )
                continue

            choice = interpret(answer, choices)
            if choice is None:
                self._console.print("[dim]unrecognised — pick one of the options above[/dim]")
                continue

            if choice.decision == "individual":
                return await self._decide_each(request)
            if choice.decision == "reject":
                feedback = (await self._ask("why? (optional, sent to the model): ")).strip()
                return _reply(request, "reject", feedback=feedback or None)
            return _reply(request, choice.decision)

    async def _decide_each(self, request: ApprovalRequested) -> ApprovalResponse:
        """Walk the files in proposal order, asking about each."""
        decisions: dict[str, ApprovalDecision] = {}

        for number, item in enumerate(request.items, start=1):
            self._console.print()
            self._console.print(f"[dim]file {number} of {len(request.items)}[/dim]")
            self._console.print(
                render_approval(
                    tool=item.tool,
                    risk=request.risk,
                    preview=item.preview,
                    badges=list(item.badges),
                    reasons=[],
                )
            )
            alarming = bool(set(item.badges) & set(BATCH_BLOCKING_BADGES))
            choices = (
                Choice("y", "approve", "[y] approve", default=not alarming),
                Choice("n", "reject", "[n] reject"),
                Choice("d", SHOW_DIFF, "[d] full diff"),
                Choice("q", "abort", "[q] abort task"),
            )
            self._console.print(render_options([choice.label for choice in choices]))

            while True:
                answer = await self._ask("choice [y]: " if not alarming else "choice (no default): ")
                choice = interpret(answer, choices)
                if choice is None:
                    self._console.print("[dim]unrecognised — pick one of the options above[/dim]")
                    continue
                if choice.decision == SHOW_DIFF:
                    self._console.print(render_diff(item.preview, limit=100_000))
                    continue
                break

            if choice.decision == "abort":
                return _reply(request, "abort")
            decisions[item.call_id] = choice.decision  # type: ignore[assignment]

        # Anything not explicitly approved is rejected by the receiving side, so the
        # top-level decision is only a summary for a frontend that ignores item_decisions.
        summary: ApprovalDecision = "approve" if all(d == "approve" for d in decisions.values()) else "reject"
        return ApprovalResponse(
            request_id=request.request_id, decision=summary, item_decisions=decisions
        )

    def _render_batch(self, request: ApprovalRequested, blockers: list[str]) -> None:
        self._console.print()
        self._console.print(f"[bold]{len(request.items)} file changes proposed in one step[/bold]")
        for number, item in enumerate(request.items, start=1):
            stats = f"+{item.added} -{item.removed}"
            flags = f"  [red]{' '.join(item.badges)}[/red]" if item.badges else ""
            where = item.path or item.summary
            self._console.print(f"  {number}. {escape(where)}  [dim]{stats}[/dim]{flags}", markup=True)
        if blockers:
            self._console.print(
                f"[yellow]approve-all is unavailable: {', '.join(blockers)}[/yellow]"
            )

    # ----------------------------------------------------------- internals

    async def _decide(self, request: ApprovalRequested) -> ApprovalResponse:
        self._render(request)
        choices = choices_for(request)
        self._console.print(render_options([choice.label for choice in choices]))

        while True:
            answer = await self._ask(_prompt_text(request))
            choice = interpret(answer, choices)

            if choice is None:
                self._console.print("[dim]unrecognised — pick one of the options above[/dim]")
                continue

            if choice.decision == SHOW_DIFF:
                # Not a decision. Show everything and come back to the prompt.
                self._console.print(render_diff(request.preview, limit=100_000))
                continue

            return await self._respond(request, choice)

    async def _respond(self, request: ApprovalRequested, choice: Choice) -> ApprovalResponse:
        if choice.decision in ("approve", "always_session") and needs_typed_confirmation(request):
            typed = await self._ask(f"type {request.typed_confirmation!r} to confirm: ")
            if typed.strip() != request.typed_confirmation:
                # Rejecting rather than re-asking is the fail-closed reading: someone who
                # mistyped the confirmation word has not demonstrated they read the panel.
                self._console.print("[yellow]confirmation did not match — treated as a rejection[/yellow]")
                return _reply(request, "reject", feedback="typed confirmation did not match")

        if choice.decision == "reject":
            feedback = (await self._ask("why? (optional, sent to the model): ")).strip()
            return _reply(request, "reject", feedback=feedback or None)

        if choice.decision == "edit":
            return self._edited(request)

        return _reply(request, choice.decision)

    def _edited(self, request: ApprovalRequested) -> ApprovalResponse:
        """Open the arguments in `$EDITOR` and hand them back for re-evaluation.

        The edited arguments re-enter the lifecycle at the top — re-validated, re-prepared
        and re-judged (§5.9) — so this cannot be used to route around a deny rule. What it
        *can* do is produce invalid JSON, which is treated as a rejection rather than as a
        reason to run the original call.
        """
        current = json.dumps(self._arguments_for(request), indent=2, sort_keys=True)
        try:
            edited = self._editor(current)
        except Exception as exc:
            self._console.print(f"[yellow]editor failed ({exc}) — treated as a rejection[/yellow]")
            return _reply(request, "reject", feedback="the editor could not be opened")

        try:
            arguments = json.loads(edited)
        except json.JSONDecodeError as exc:
            self._console.print(f"[yellow]not valid JSON ({exc.msg}) — treated as a rejection[/yellow]")
            return _reply(request, "reject", feedback="edited arguments were not valid JSON")

        if not isinstance(arguments, dict):
            return _reply(request, "reject", feedback="edited arguments were not an object")

        return _reply(request, "edit", edited_arguments=arguments)

    def _render(self, request: ApprovalRequested) -> None:
        self._console.print()
        self._console.print(
            render_approval(
                tool=request.tool,
                risk=request.risk,
                preview=request.preview,
                badges=list(request.badges),
                reasons=list(request.reasons),
            )
        )


def _reply(
    request: ApprovalRequested,
    decision: str,
    *,
    feedback: str | None = None,
    edited_arguments: dict[str, Any] | None = None,
) -> ApprovalResponse:
    return ApprovalResponse(
        request_id=request.request_id,
        decision=decision,  # type: ignore[arg-type]
        feedback=feedback,
        edited_arguments=edited_arguments,
    )


def _prompt_text(request: ApprovalRequested) -> str:
    if needs_typed_confirmation(request):
        return "choice (no default — this cannot be undone): "
    return "choice [y]: "


async def _prompt_async(prompt: str) -> str:
    """Read one line without blocking the event loop.

    prompt_toolkit's async session is used when a terminal is attached; otherwise this
    falls back to a thread so a piped stdin still works. Blocking inline would freeze the
    loop, and with it Ctrl+C and every other subscriber.
    """
    import asyncio

    try:
        from prompt_toolkit import PromptSession

        session: PromptSession[str] = PromptSession()
        return await session.prompt_async(prompt)
    except Exception:
        return await asyncio.to_thread(input, prompt)


def _open_editor(text: str) -> str:
    """Round-trip text through `$EDITOR`."""
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
    handle, path = tempfile.mkstemp(suffix=".json", prefix="hearth-edit-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        subprocess.run([editor, path], check=False)  # noqa: S603 - argv, user's own $EDITOR
        return Path(path).read_text(encoding="utf-8")
    finally:
        Path(path).unlink(missing_ok=True)


#: Decisions the bus protocol accepts, re-exported so a frontend can validate its own.
VALID_DECISIONS: tuple[ApprovalDecision, ...] = (
    "approve",
    "reject",
    "edit",
    "always_session",
    "abort",
)
