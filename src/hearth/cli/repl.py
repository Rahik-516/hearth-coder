"""The interactive chat loop.

Inline streaming with prompt_toolkit for input and Rich for output
(docs/implementation-roadmap.md M3). Not a full-screen TUI: taking over the terminal would
cost the scrollback, and the output here is code that people copy.

**Ctrl+C cancels the current turn within a second, and does not exit.** That means the
streaming turn runs as a cancellable task rather than being awaited directly — there is no
other way to interrupt an async generator mid-stream. A second Ctrl+C at an empty prompt
exits.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.table import Table

from hearth.cli.checkpoint_commands import rewind, show_checkpoints, undo
from hearth.cli.completers import SLASH_COMMANDS, ChatCompleter, extract_pins
from hearth.cli.editor import edit_in_editor
from hearth.cli.render import ChatRenderer
from hearth.core.bus import EventBus
from hearth.core.context.budget import budget_for
from hearth.core.events import RetrievalPerformed
from hearth.core.plan import PlanError, PlanStore, build_execute_message, parse_plan
from hearth.core.runner import AgentTurnResult, ChatRunner, TurnResult
from hearth.core.session import Mode, Session, SessionStore
from hearth.llm.errors import LLMError
from hearth.llm.types import Message
from hearth.safety.checkpoints import CheckpointStore
from hearth.storage.index_repo import IndexRepository
from hearth.tools.gateway import ToolGateway
from hearth.workflows.commit import run_commit
from hearth.workflows.review import run_review
from hearth.workflows.targets import TargetError, resolve_target
from hearth.workflows.test_writer import run_test_workflow

_BANNER = """[bold]Hearth[/bold] — ask about this repository.
[dim]/help for commands · @path to pin a file · Ctrl+C cancels a reply · Ctrl+D exits[/dim]"""


@dataclass
class ChatREPL:
    """Drives an interactive session."""

    session: Session
    runner: ChatRunner
    bus: EventBus
    store: SessionStore
    console: Console = field(default_factory=Console)
    history_file: Path | None = None
    completer: ChatCompleter | None = None
    #: Workspace root, for the checkpoint commands. Defaults to the session's.
    workspace: Path | None = None
    #: None disables /undo, /rewind and /checkpoints rather than faking them.
    checkpoints: CheckpointStore | None = None
    #: None disables `/mode agent`, for the same reason: a mode that claims tools and has
    #: none would let the model propose edits that silently never happen.
    gateway: ToolGateway | None = None
    #: The session's grant set, shared with the gateway's policy closure. `/execute` adds
    #: the approved plan's edit grants here and `/plan` revokes them; mutating it is how
    #: a frontend widens permissions, which is why nothing else may hold a reference.
    grants: set[str] | None = None
    #: Index connection, for resolving `/test <symbol>` to a file. None means a target
    #: has to be a path.
    index_connection: sqlite3.Connection | None = None

    _renderer: ChatRenderer = field(init=False)
    _last_sources: list[RetrievalPerformed] = field(default_factory=list, init=False)
    _plans: PlanStore = field(default_factory=PlanStore, init=False)
    #: Monotonic within a session, so two plans never share an id — and so a revoked
    #: grant key can never be re-created by the next plan and quietly come back.
    _plans_made: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._renderer = ChatRenderer(console=self.console)
        self.bus.subscribe(self._renderer)
        self.bus.subscribe(self._capture_sources)

    async def run(self) -> None:
        """Read, answer, repeat, until EOF or /exit."""
        self.console.print(_BANNER)
        self.console.print(
            f"[dim]session {self.session.id} · {self.session.model} · num_ctx {self.session.num_ctx}[/dim]\n"
        )

        prompt_session: PromptSession[str] = PromptSession(
            history=FileHistory(str(self.history_file)) if self.history_file else None,
            completer=self.completer,
            complete_while_typing=False,
        )

        while True:
            try:
                with patch_stdout():
                    text = await prompt_session.prompt_async("> ")
            except KeyboardInterrupt:
                # Ctrl+C at an empty prompt: nothing to cancel, so just clear the line.
                continue
            except EOFError:
                break

            text = text.strip()
            if not text:
                continue

            if text.startswith("/"):
                if await self._handle_command(text):
                    break
                continue

            await self._ask(text)

        self.console.print("[dim]bye[/dim]")

    # ------------------------------------------------------------------ turns

    async def _ask(self, text: str) -> None:
        """Run one turn as a cancellable task, so Ctrl+C can interrupt the stream."""
        message, pins = extract_pins(text)
        for path in pins:
            self.session.pin(path)
        if not message:
            message = text

        self.store.set_title_from(self.session, message)

        task = asyncio.create_task(self._turn(message))
        try:
            result = await asyncio.shield(task)
        except KeyboardInterrupt:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self.console.print("\n[yellow]cancelled[/yellow]")
            return
        except asyncio.CancelledError:
            self.console.print("\n[yellow]cancelled[/yellow]")
            return

        if result.ok:
            # Persist only completed turns. A cancelled stream leaves a partial answer
            # that would poison the history and the cached prefix.
            self.store.save_message(self.session, Message(role="user", content=message))
            self.store.save_message(self.session, Message(role="assistant", content=result.answer))

        if isinstance(result, AgentTurnResult):
            # Printed whether or not the turn succeeded: a run that stopped at the step
            # limit, or had its edits refused, looks like a finished one otherwise (§14.1).
            self.console.print(f"[dim]{result.summary()}[/dim]")

        self.console.print()

    async def _turn(self, message: str) -> TurnResult | AgentTurnResult:
        """Route the turn by mode. Chat has no tools; agent drives the loop."""
        if self.session.mode is Mode.AGENT and self.gateway is not None:
            return await self.runner.run_agent_turn(
                self.session,
                message,
                gateway=self.gateway,
                tool_schemas=self._schemas_for(Mode.AGENT),
            )
        return await self.runner.run_turn(self.session, message)

    def _schemas_for(self, mode: Mode) -> list[dict[str, object]]:
        """The tool schemas for a mode, asked of the gateway each time.

        Not cached on this object: plan mode and agent mode expose different tools, and a
        list captured at construction would offer the model whichever set happened to be
        current when the REPL started. In plan mode that would mean handing a read-only
        turn the write tools.
        """
        if self.gateway is None:
            return []
        return list(self.gateway.availability(mode.value).schemas)

    def _capture_sources(self, event: object) -> None:
        if isinstance(event, RetrievalPerformed):
            self._last_sources = [event]

    # --------------------------------------------------------------- commands

    async def _handle_command(self, text: str) -> bool:
        """Run a slash command. Returns True to exit the REPL."""
        name, _, argument = text.partition(" ")
        argument = argument.strip()

        match name:
            case "/exit" | "/quit":
                return True
            case "/help":
                self._show_help()
            case "/context":
                self._show_context()
            case "/sources":
                self._show_sources()
            case "/clear":
                self.store.clear(self.session)
                # The plan and its grants are session state too. Leaving an approved
                # plan's edit grants live after the history it referred to is gone
                # would auto-approve writes for a task nothing on screen mentions.
                self._revoke_plan_grants()
                self.console.print("[dim]history cleared[/dim]")
            case "/model":
                self._show_or_set_model(argument)
            case "/mode":
                self._show_or_set_mode(argument)
            case "/test":
                await self._test(argument)
            case "/commit":
                await self._commit()
            case "/review":
                await self._review_diff(argument)
            case "/plan":
                await self._plan(argument)
            case "/execute":
                await self._execute()
            case "/compact":
                await self._compact()
            case "/thinking":
                self._renderer.show_thinking = not self._renderer.show_thinking
                state = "on" if self._renderer.show_thinking else "off"
                self.console.print(f"[dim]thinking {state}[/dim]")
            case "/sessions":
                self._show_sessions()
            case "/checkpoints":
                self._with_checkpoints(
                    lambda store, root: show_checkpoints(self.console, store, self.session.id)
                )
            case "/undo":
                self._with_checkpoints(
                    lambda store, root: undo(
                        self.console,
                        store,
                        self.session.id,
                        workspace=root,
                        confirm=self._confirm,
                        step=int(argument) if argument.isdigit() else None,
                    )
                )
            case "/rewind":
                if not argument.isdigit():
                    self.console.print("[yellow]usage:[/yellow] /rewind <step> — see /checkpoints")
                else:
                    self._with_checkpoints(
                        lambda store, root: rewind(
                            self.console,
                            store,
                            self.session.id,
                            int(argument),
                            workspace=root,
                            confirm=self._confirm,
                        )
                    )
            case _:
                self.console.print(f"[yellow]unknown command {name}[/yellow] — try /help")
        return False

    def _with_checkpoints(self, action: Callable[[CheckpointStore, Path], object]) -> None:
        """Run a checkpoint command, or explain why it is unavailable.

        Saying so is better than a no-op: a user who types `/undo` and sees nothing has no
        way to tell "there was nothing to undo" from "undo is not wired up".
        """
        root = self.workspace or (Path(self.session.workspace) if self.session.workspace else None)
        if self.checkpoints is None or root is None:
            self.console.print("[dim]checkpoints are not available in this session[/dim]")
            return
        action(self.checkpoints, root)

    def _confirm(self, question: str) -> bool:
        """Ask a yes/no question. Defaults to no — this is the conflict prompt."""
        answer = self.console.input(f"{question} [y/N] ").strip().lower()
        return answer in ("y", "yes")

    def _show_help(self) -> None:
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column("", style="bold cyan")
        table.add_column("")
        for command, description in SLASH_COMMANDS.items():
            table.add_row(command, description)
        self.console.print(table)
        self.console.print("[dim]@path pins a file into the next question's context[/dim]")

    def _show_context(self) -> None:
        """Show the budget table (an M3 acceptance criterion)."""
        budget = budget_for(self.session.num_ctx)
        estimator = self.session.estimator

        history_tokens = sum(estimator.estimate(m.content) for m in self.session.history)

        table = Table(title=f"context budget · num_ctx {self.session.num_ctx}", box=None)
        table.add_column("segment", style="bold")
        table.add_column("limit", justify="right")
        table.add_column("notes", style="dim")

        for segment, limit in budget.limits.items():
            note = ""
            if segment.value == "history":
                note = f"{history_tokens} used by {len(self.session.history)} message(s)"
            elif segment.value == "output":
                note = "reserved for the reply; never borrowed from"
            table.add_row(segment.value, str(limit), note)

        self.console.print(table)
        self.console.print(
            f"[dim]input budget {budget.input_budget} · compaction at "
            f"{budget.compaction_threshold} · "
            f"estimator {estimator.chars_per_token:.2f} chars/token "
            f"({'calibrated' if estimator.is_calibrated else 'uncalibrated'})[/dim]"
        )

    def _show_sources(self) -> None:
        if not self._last_sources:
            self.console.print("[dim]no retrieval yet[/dim]")
            return
        for source in self._last_sources[-1].sources:
            self.console.print(
                f"  [cyan]{source.path}:{source.start_line}-{source.end_line}[/cyan]"
                f"  [dim]{source.retriever or ''}[/dim]"
            )

    # -------------------------------------------------------------- workflows

    async def _cancellable[T](self, work: Coroutine[object, object, T]) -> T | None:
        """Run a long call as a task so Ctrl+C can interrupt it, as `_ask` does.

        Returns None when cancelled. The workflows are a single model call each, but that
        call is the slow part and a REPL that cannot be interrupted during it is the one
        the user kills with the window.
        """
        task = asyncio.ensure_future(work)
        try:
            return await asyncio.shield(task)
        except KeyboardInterrupt:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        except asyncio.CancelledError:
            pass
        self.console.print("\n[yellow]cancelled[/yellow]")
        return None

    def _target_repository(self) -> IndexRepository | None:
        """The index, for resolving a symbol name to a file. None when there is no index."""
        if self.index_connection is None:
            return None
        return IndexRepository(self.index_connection)

    def _persist_history_since(self, start: int) -> None:
        """Save the messages a workflow added, as `_ask` does for an ordinary turn.

        Workflows drive the runner directly rather than through `_ask`, so without this
        their turns would exist in memory only and `/resume` would find a conversation with
        a hole exactly where the work happened.
        """
        for message in self.session.history[start:]:
            self.store.save_message(self.session, message)

    async def _test(self, argument: str) -> None:
        """`/test <target>`: write tests, run them, and fix them until they pass."""
        if not argument:
            self.console.print("[yellow]usage:[/yellow] /test <file, symbol, or path::symbol>")
            return
        if self.gateway is None:
            self.console.print(
                "[yellow]/test is unavailable in this session[/yellow] — it needs a tool "
                "gateway, which requires an indexed workspace"
            )
            return

        root = self.workspace or Path(self.session.workspace)
        target = resolve_target(root, argument, repository=self._target_repository())
        if isinstance(target, TargetError):
            self.console.print(f"[yellow]{target.message}[/yellow]", markup=True)
            return

        if self.session.mode is not Mode.AGENT:
            self.session.switch_mode(Mode.AGENT)
            self.console.print("[dim]mode: agent — edits and test runs will ask[/dim]")

        self.console.print(f"[dim]writing tests for {target.describe()}[/dim]")
        start = len(self.session.history)
        try:
            outcome = await self._cancellable(
                run_test_workflow(
                    runner=self.runner,
                    session=self.session,
                    gateway=self.gateway,
                    root=root,
                    target=target,
                    tool_schemas=self._schemas_for(Mode.AGENT),
                )
            )
        except LLMError as exc:
            self.console.print(f"[red]{exc}[/red]")
            return
        finally:
            self._persist_history_since(start)
        if outcome is None:
            return

        self.console.print()
        if outcome.status == "passed":
            files = ", ".join(outcome.test_files)
            rounds = "" if outcome.iterations == 1 else f" after {outcome.iterations - 1} fix round(s)"
            self.console.print(f"[green]tests pass[/green]{rounds}: {files}")
        elif outcome.status == "failed":
            self.console.print(f"[red]tests still fail:[/red] {outcome.detail}")
            if outcome.summary:
                self.console.print(outcome.summary, markup=False)
        elif outcome.status == "refused":
            self.console.print(f"[yellow]{outcome.detail}[/yellow]")
        else:
            self.console.print(f"[yellow]{outcome.detail}[/yellow]")

        if outcome.source_modified:
            self.console.print(
                f"[yellow]note:[/yellow] {target.path} was changed during this run. The task was "
                "to test it as it is — check the change, or /undo it."
            )

    async def _commit(self) -> None:
        """`/commit`: message from the staged diff, then a normal approved commit."""
        if self.gateway is None:
            self.console.print(
                "[yellow]/commit is unavailable in this session[/yellow] — it needs a tool "
                "gateway, which requires an indexed workspace"
            )
            return

        # The commit is a `git_commit` tool call, which policy refuses outside agent mode.
        # Switching is announced rather than silent for the same reason /execute announces
        # it: it costs a re-prefill and changes what the model may do.
        if self.session.mode is not Mode.AGENT:
            self.session.switch_mode(Mode.AGENT)
            self.console.print("[dim]mode: agent — the commit will ask before it happens[/dim]")

        root = self.workspace or Path(self.session.workspace)
        try:
            outcome = await self._cancellable(
                run_commit(
                    runner=self.runner, session=self.session, gateway=self.gateway, root=root
                )
            )
        except LLMError as exc:
            self.console.print(f"[red]{exc}[/red]")
            return
        if outcome is None:
            return

        for warning in outcome.warnings:
            self.console.print(f"[yellow]{warning}[/yellow]")

        if outcome.status == "committed":
            self.console.print(f"[green]{outcome.detail}[/green]")
        elif outcome.status == "rejected":
            self.console.print("[dim]commit declined — nothing was committed[/dim]")
        else:
            self.console.print(f"[red]not committed:[/red] {outcome.detail}")
            if outcome.message and outcome.status == "failed":
                self.console.print("[dim]the message was:[/dim]")
                self.console.print(outcome.message, markup=False)

    async def _review_diff(self, argument: str) -> None:
        """`/review [--staged]`: read-only findings, with citations checked in code."""
        staged = "--staged" in argument.split()
        root = self.workspace or Path(self.session.workspace)

        try:
            outcome = await self._cancellable(
                run_review(runner=self.runner, session=self.session, root=root, staged=staged)
            )
        except LLMError as exc:
            self.console.print(f"[red]{exc}[/red]")
            return
        if outcome is None:
            return

        if outcome.refusal:
            self.console.print(f"[yellow]{outcome.refusal}[/yellow]")
            return

        self.console.print()
        # `markup=False`: the review quotes code, and code is full of square brackets.
        self.console.print(outcome.text, markup=False)

        if outcome.omitted:
            self.console.print(
                f"\n[yellow]not reviewed (did not fit):[/yellow] {', '.join(outcome.omitted)}"
            )
        checked = len(outcome.citations) - len(outcome.unverified)
        self.console.print(
            f"\n[dim]{checked}/{len(outcome.citations)} citation(s) verified against the diff[/dim]"
        )

    # ------------------------------------------------------------- plan mode

    async def _plan(self, task: str) -> None:
        """`/plan <task>`: investigate with read-only tools and propose a plan.

        Switches the session into plan mode and leaves it there. The switch costs one
        re-prefill, which is the documented price of changing modes, and paying it once
        per plan is better than the alternative — planning inside agent mode, where the
        write tools are on the table and a model that decides mid-thought to "just make
        the change" can.
        """
        if not task:
            self.console.print("[yellow]usage:[/yellow] /plan <what you want done>")
            return
        if self.gateway is None:
            self.console.print(
                "[yellow]plan mode is unavailable in this session[/yellow] — it needs a "
                "tool gateway, which requires an indexed workspace"
            )
            return

        if self.session.mode is not Mode.PLAN:
            self.session.switch_mode(Mode.PLAN)
            self.console.print("[dim]mode: plan (read-only; nothing can be edited here)[/dim]")

        # A new plan supersedes the old one, and its grants go with it. Dropping them here
        # rather than at `/execute` matters: between the two, the user is being shown a
        # different plan, and grants from one they have moved on from must not still be
        # live if they walk away mid-review.
        self._revoke_plan_grants()

        result = await self.runner.run_plan_turn(
            self.session,
            task,
            gateway=self.gateway,
            tool_schemas=self._schemas_for(Mode.PLAN),
        )

        if result.plan is None:
            self.console.print(f"\n[red]no plan:[/red] {result.error}")
            if result.raw:
                excerpt = result.raw.strip()[:400]
                self.console.print(f"[dim]the model returned:[/dim]\n{excerpt}")
            self.console.print("[dim]try /plan again, or narrow the task[/dim]")
            return

        self._plans_made += 1
        plan_id = f"{self.session.id[:8]}-{self._plans_made}"
        self._plans.propose(result.plan, plan_id=plan_id)

        self.console.print()
        # `markup=False`: the plan is model-generated text, and its own change-type labels
        # are `[add]`, `[modify]` — which Rich would read as markup tags and swallow. The
        # same escape hatch keeps a model that writes `[/bold]` from corrupting the panel.
        self.console.print(result.plan.render(), markup=False)
        self.console.print(
            f"\n[dim]{result.steps} step(s), {result.tool_calls} tool call(s), "
            f"{result.duration_ms / 1000:.1f}s[/dim]"
        )

        await self._review(plan_id)

    async def _review(self, plan_id: str) -> None:
        """Approve, edit or reject the pending plan (§8.3 step 3)."""
        while True:
            answer = self.console.input(
                "\n[bold]approve[/bold] / [bold]edit[/bold] / [bold]reject[/bold]? [a/e/r] "
            ).strip().lower()

            if answer in ("a", "approve", "y", "yes"):
                self._approve(plan_id)
                return
            if answer in ("r", "reject", "n", "no", ""):
                self._plans.pending.pop(plan_id, None)
                self.console.print("[dim]plan discarded — say what was wrong and /plan again[/dim]")
                return
            if answer in ("e", "edit"):
                if self._edit_pending(plan_id):
                    return
                continue

            self.console.print("[yellow]a, e or r[/yellow]")

    def _edit_pending(self, plan_id: str) -> bool:
        """Open the plan in $EDITOR and take the result. Returns True when review is over.

        The plan goes out as JSON rather than as the rendered text, because it has to come
        back through the same validation the model's answer did. Round-tripping prose would
        mean a second parser, and the user's edit is the version that gets executed — it is
        the last place to be lenient.
        """
        plan = self._plans.pending.get(plan_id)
        if plan is None:
            return True

        edited = edit_in_editor(plan.model_dump_json(indent=2), suffix=".json")
        if edited is None:
            self.console.print("[dim]no editor available — set $EDITOR[/dim]")
            return False

        try:
            revised = parse_plan(edited)
        except PlanError as exc:
            # Back to the prompt with the original still pending: discarding their edit
            # *and* the plan over a stray comma would be the worse of the two.
            self.console.print(f"[red]that did not parse:[/red] {exc}")
            return False

        self._plans.propose(revised, plan_id=plan_id)
        self.console.print()
        self.console.print(revised.render(), markup=False)
        return False

    def _approve(self, plan_id: str) -> None:
        """Approve the plan, and offer its edit grants as a separate, opt-in question.

        Two questions rather than one, because they are two decisions: "is this the right
        change?" and "may it happen without asking me again?". Folding the second into the
        first would turn every approved plan into a blanket edit permission, which is the
        default §8.3 explicitly does not want.
        """
        root = self.workspace or Path(self.session.workspace)
        approved = self._plans.approve(plan_id, workspace=root)

        self.console.print(f"[green]plan approved[/green] [dim]({plan_id})[/dim]")

        for path, why in approved.refused_files:
            self.console.print(f"  [yellow]{path}[/yellow] [dim]cannot be pre-approved: {why}[/dim]")

        if not approved.grant_files or self.grants is None:
            self.console.print("[dim]/execute to start — each edit will ask[/dim]")
            return

        listed = ", ".join(approved.grant_files)
        answer = self.console.input(
            f"Edit these {len(approved.grant_files)} file(s) without asking each time?\n"
            f"  [dim]{listed}[/dim]\n[y/N] "
        ).strip().lower()

        if answer in ("y", "yes"):
            self.grants.update(approved.edit_grants())
            self.console.print(
                f"[dim]granted for this session under {approved.grant_key} — "
                "/plan again or /clear revokes it[/dim]"
            )
        else:
            self.console.print("[dim]each edit will ask[/dim]")

        self.console.print("[dim]/execute to start[/dim]")

    async def _execute(self) -> None:
        """`/execute`: switch to agent mode and carry out the approved plan."""
        approved = self._plans.approved
        if approved is None:
            self.console.print("[yellow]no approved plan[/yellow] — /plan <task> first")
            return
        if self.gateway is None:
            self.console.print("[yellow]agent mode is unavailable in this session[/yellow]")
            return

        if self.session.mode is not Mode.AGENT:
            self.session.switch_mode(Mode.AGENT)
            self.console.print("[dim]mode: agent — edits, commands and commits will ask[/dim]")

        await self._ask(build_execute_message(approved))

    def _revoke_plan_grants(self) -> None:
        if self.grants is not None and self._plans.approved is not None:
            self.grants.difference_update(self._plans.approved.edit_grants())
        self._plans.clear()

    async def _compact(self) -> None:
        """Summarise the older history now, rather than waiting for the window to fill.

        Useful before a long task: compaction costs one full prefill, and paying it
        deliberately between turns is better than having it land in the middle of one.
        """
        result = await self.runner.maybe_compact(self.session, force=True)
        if result.compacted:
            self.console.print(
                f"[dim]compacted {result.replaced} message(s), "
                f"~{result.tokens_saved} tokens freed; the next turn re-prefills once[/dim]"
            )
        elif result.degraded is None:
            self.console.print("[dim]nothing to compact yet[/dim]")

    def _show_or_set_mode(self, argument: str) -> None:
        """Show the mode, or switch it.

        Switching to agent is refused when no gateway was supplied, rather than switching
        and quietly having no tools: a model told it can edit files will propose edits, and
        every one of them would vanish.
        """
        if not argument:
            self.console.print(f"[dim]mode: {self.session.mode.value}[/dim]")
            return

        try:
            mode = Mode(argument)
        except ValueError:
            allowed = ", ".join(item.value for item in Mode)
            self.console.print(f"[yellow]unknown mode {argument!r}[/yellow] — one of: {allowed}")
            return

        if mode is Mode.AGENT and self.gateway is None:
            self.console.print(
                "[yellow]agent mode is unavailable in this session[/yellow] — it needs a "
                "tool gateway, which requires an indexed workspace"
            )
            return
        if mode is Mode.PLAN and self.gateway is None:
            self.console.print(
                "[yellow]plan mode is unavailable in this session[/yellow] — it needs a "
                "tool gateway, which requires an indexed workspace"
            )
            return

        previous = self.session.mode
        self.session.switch_mode(mode)
        self.console.print(
            f"[dim]mode: {previous.value} → {mode.value} "
            "(new cache epoch; the next turn re-prefills)[/dim]"
        )
        if mode is Mode.AGENT:
            self.console.print(
                "[dim]edits, commands and commits will ask before they happen[/dim]"
            )
        elif mode is Mode.PLAN:
            self.console.print("[dim]read-only; /plan <task> to produce a plan[/dim]")

    def _show_or_set_model(self, argument: str) -> None:
        if not argument:
            self.console.print(f"[dim]model: {self.session.model}[/dim]")
            return

        previous = self.session.model
        self.session.switch_model(argument)
        self.console.print(
            f"[dim]model: {previous} → {argument} (new cache epoch; the next turn re-prefills)[/dim]"
        )

    def _show_sessions(self) -> None:
        table = Table(box=None)
        table.add_column("id", style="dim")
        table.add_column("title")
        table.add_column("turns", justify="right")

        for record in self.store.list_recent(limit=10):
            marker = "→ " if record.id == self.session.id else "  "
            table.add_row(f"{marker}{record.id}", record.display_title, str(record.message_count))
        self.console.print(table)
