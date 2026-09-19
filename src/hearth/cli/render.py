"""Rendering events to the terminal.

A subscriber on the event bus — it never calls the core, it only reacts to what the core
publishes (docs/system-design.md §5.1). That separation is what lets the same turn drive a
terminal, an editor, or a test frontend that asserts on the event sequence.

Output streams inline rather than taking over the screen (docs/implementation-roadmap.md
M3). It stays in the user's scrollback, next to whatever else they were doing, and is
copyable and greppable — which matters when the output is code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.markdown import Markdown

from hearth.core.events import (
    ContextStats,
    ErrorEvent,
    Notice,
    RetrievalPerformed,
    TextDelta,
    ThinkingDelta,
    TurnFinished,
    TurnStarted,
)

_LEVEL_STYLE = {"info": "dim", "warning": "yellow", "error": "red"}


@dataclass
class ChatRenderer:
    """Renders one turn's events."""

    console: Console = field(default_factory=Console)
    show_thinking: bool = False
    show_sources: bool = True

    _answer: list[str] = field(default_factory=list, init=False)
    _thinking_chars: int = field(default=0, init=False)
    _streaming: bool = field(default=False, init=False)
    _last_stats: ContextStats | None = field(default=None, init=False)

    def __call__(self, event: object) -> None:
        """Bus subscriber entry point."""
        match event:
            case TurnStarted():
                self._begin()
            case RetrievalPerformed():
                self._render_sources(event)
            case ThinkingDelta():
                self._render_thinking(event)
            case TextDelta():
                self._render_text(event)
            case Notice():
                self._render_notice(event)
            case ErrorEvent():
                self.console.print(f"[red]Error:[/red] {event.message}")
            case ContextStats():
                self._last_stats = event
            case TurnFinished():
                self._finish(event)
            case _:
                pass

    # ------------------------------------------------------------- internals

    def _begin(self) -> None:
        self._answer.clear()
        self._thinking_chars = 0
        self._streaming = False
        self._last_stats = None

    def _render_sources(self, event: RetrievalPerformed) -> None:
        if not self.show_sources or not event.sources:
            return
        shown = ", ".join(f"{s.path}:{s.start_line}-{s.end_line}" for s in event.sources[:3])
        extra = f" +{len(event.sources) - 3} more" if len(event.sources) > 3 else ""
        self.console.print(f"[dim]context: {shown}{extra}[/dim]")

    def _render_thinking(self, event: ThinkingDelta) -> None:
        """Thinking is collapsed to a progress hint unless explicitly requested.

        Reasoning tokens are not the answer, and on a small model there can be more of
        them than answer. Showing a counter keeps the user informed that work is happening
        without burying the reply.
        """
        self._thinking_chars += len(event.text)
        if self.show_thinking:
            self.console.print(f"[dim italic]{event.text}[/dim italic]", end="")
        elif self._thinking_chars and not self._streaming:
            self.console.print(f"[dim]thinking… ({self._thinking_chars} chars)[/dim]", end="\r")

    def _render_text(self, event: TextDelta) -> None:
        if not self._streaming:
            if self._thinking_chars and not self.show_thinking:
                self.console.print(" " * 40, end="\r")  # clear the thinking hint
            self._streaming = True
        self.console.print(event.text, end="", markup=False, highlight=False)
        self._answer.append(event.text)

    def _render_notice(self, event: Notice) -> None:
        style = _LEVEL_STYLE.get(event.level, "dim")
        prefix = "\n" if self._streaming else ""
        self.console.print(f"{prefix}[{style}]{event.message}[/{style}]")

    def _finish(self, event: TurnFinished) -> None:
        if self._streaming:
            self.console.print()

        if event.reason == "aborted":
            self.console.print("[yellow]cancelled[/yellow]")
        elif event.reason not in ("answered", "stop"):
            self.console.print(f"[yellow]turn ended: {event.reason}[/yellow]")

        if self._last_stats is not None:
            self.console.print(self._format_stats(self._last_stats, event))

    @staticmethod
    def _format_stats(stats: ContextStats, finished: TurnFinished) -> str:
        """The per-turn stats line (docs/system-design.md §15).

        Shows context use and throughput, because on local models those are the two
        numbers that explain why a turn felt the way it did.
        """
        parts = [f"{stats.used}/{stats.budget} ctx"]
        if stats.cached_tokens:
            parts.append(f"{stats.cached_tokens} cached")
        if stats.prefill_ms:
            parts.append(f"prefill {stats.prefill_ms:.0f}ms")
        if stats.generation_tps:
            parts.append(f"{stats.generation_tps:.1f} tok/s")
        if finished.duration_ms:
            parts.append(f"{finished.duration_ms / 1000:.1f}s")
        return f"[dim]{'  ·  '.join(parts)}[/dim]"

    @property
    def answer(self) -> str:
        return "".join(self._answer)

    def render_markdown(self) -> None:
        """Re-render the collected answer as formatted Markdown.

        Streaming prints raw text because Markdown cannot be rendered incrementally
        without redrawing; this offers the formatted version afterwards on request.
        """
        if self._answer:
            self.console.print(Markdown(self.answer))
