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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.table import Table

from hearth.cli.checkpoint_commands import rewind, show_checkpoints, undo
from hearth.cli.completers import SLASH_COMMANDS, ChatCompleter, extract_pins
from hearth.cli.render import ChatRenderer
from hearth.core.bus import EventBus
from hearth.core.context.budget import budget_for
from hearth.core.events import RetrievalPerformed
from hearth.core.runner import AgentTurnResult, ChatRunner, TurnResult
from hearth.core.session import Mode, Session, SessionStore
from hearth.llm.types import Message
from hearth.safety.checkpoints import CheckpointStore
from hearth.tools.gateway import ToolGateway

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
    tool_schemas: list[dict[str, object]] = field(default_factory=list)

    _renderer: ChatRenderer = field(init=False)
    _last_sources: list[RetrievalPerformed] = field(default_factory=list, init=False)

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
                tool_schemas=list(self.tool_schemas),
            )
        return await self.runner.run_turn(self.session, message)

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
                self.console.print("[dim]history cleared[/dim]")
            case "/model":
                self._show_or_set_model(argument)
            case "/mode":
                self._show_or_set_mode(argument)
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
        if mode is Mode.PLAN:
            self.console.print("[yellow]plan mode arrives in I3[/yellow] — use chat or agent")
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
