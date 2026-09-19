"""Rendering an approval for a terminal.

The panel is the entire basis on which someone decides whether to let a change happen, so
the rendering choices are safety choices (docs/safety-and-tool-use.md §6.1):

* **The diff is shown, not summarised.** "Updates the total calculation" is a description
  the model wrote; `-total = sum(...)` is what will happen.
* **Badges are prominent and few.** They are the fast path for "is this routine?", which
  only works if they are rare. A panel that always shows four badges has taught the reader
  to ignore all four (T2).
* **Long diffs are truncated with a visible marker**, never silently. A preview that
  quietly stops is a preview that misrepresents what is being approved; `[d]` opens the
  rest.
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

#: Diff lines shown inline. Beyond this the panel gets tall enough that people stop
#: reading it, which is worse than making them press `d`.
MAX_PREVIEW_LINES = 40

#: Badges that colour the panel red rather than yellow.
_ALARMING = frozenset({"DESTRUCTIVE", "SECRET?", "PARSE-ERRORS-INTRODUCED", "INJECTION?", "WIN-INTEROP"})

_BADGE_STYLE = {
    "WRITE": "yellow",
    "EXEC": "yellow",
    "VCS": "yellow",
    "VCS_WRITE": "yellow",
    "FUZZY-MATCH": "cyan",
    "LARGE": "cyan",
    "OUTSIDE-WORKSPACE": "magenta",
    "PROJECT-CONFIG": "magenta",
}


def render_diff(text: str, *, limit: int = MAX_PREVIEW_LINES) -> Text:
    """Colourise a unified diff, truncating with a visible marker."""
    rendered = Text()
    lines = text.splitlines()

    for line in lines[:limit]:
        rendered.append_text(_diff_line(line))
        rendered.append("\n")

    if len(lines) > limit:
        rendered.append(f"… {len(lines) - limit} more line(s) — press [d] for the full diff\n", style="dim")

    return rendered


def _diff_line(line: str) -> Text:
    if line.startswith(("+++", "---")):
        return Text(line, style="dim")
    if line.startswith("+"):
        return Text(line, style="green")
    if line.startswith("-"):
        return Text(line, style="red")
    if line.startswith("@@"):
        return Text(line, style="cyan")
    return Text(line)


def render_badges(badges: list[str]) -> Text:
    """The badge row. Alarming badges are red; the rest take their usual colour."""
    rendered = Text()
    for index, badge in enumerate(badges):
        if index:
            rendered.append(" ")
        style = "bold red" if badge in _ALARMING else _BADGE_STYLE.get(badge, "yellow")
        rendered.append(f"[{badge}]", style=style)
    return rendered


def render_approval(
    *,
    tool: str,
    risk: str,
    preview: str,
    badges: list[str],
    reasons: list[str],
    summary: str = "",
) -> Panel:
    """The approval panel, as §6.1 lays it out."""
    parts: list[RenderableType] = []

    if summary:
        parts.append(Text(summary, style="bold"))
    if badges:
        parts.append(render_badges(badges))
    if summary or badges:
        parts.append(Text(""))

    parts.append(render_diff(preview) if _looks_like_diff(preview) else Text(preview))

    if reasons:
        parts.append(Text(""))
        for reason in reasons:
            # escape(): reasons carry rule ids and file paths, and Rich would read
            # `[house-rule]` as markup and render nothing.
            parts.append(Text.from_markup(f"[dim]· {escape(reason)}[/dim]"))

    alarming = _ALARMING & set(badges)
    return Panel(
        Group(*parts),
        title=f"Approval required — {tool}",
        subtitle=f"[{risk}]",
        border_style="red" if alarming else "yellow",
        padding=(0, 1),
    )


def render_options(labels: list[str]) -> Text:
    """The key line under the panel."""
    return Text("  ".join(labels), style="dim")


def _looks_like_diff(text: str) -> bool:
    return text.startswith(("---", "diff ", "@@")) or "\n@@" in text
