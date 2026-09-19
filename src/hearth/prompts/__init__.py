"""Prompt text, stored as Markdown rather than inline strings.

Prompts are reviewed like code and guarded by snapshot tests, which is only practical if
they live in files (docs/project-structure.md §4). Loading is cached, since the same few
files are read on every turn.
"""

from __future__ import annotations

import contextlib
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


class PromptNotFoundError(FileNotFoundError):
    """A prompt file is missing from the package."""


@lru_cache(maxsize=32)
def load(name: str) -> str:
    """Load a prompt by name, without the .md suffix."""
    path = _PROMPTS_DIR / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise PromptNotFoundError(f"no prompt named {name!r} in {_PROMPTS_DIR}") from exc


def system_prompt_for(mode: str) -> str:
    """Core rules plus the mode's guidance, joined in a stable order.

    Stable because this string is the cached prefix: reordering it, or interpolating
    anything that varies per turn, throws away the KV cache on every request
    (docs/system-design.md §9.2).
    """
    parts = [load("system_core")]
    # Chat mode has no tools, so the tool protocol would be several hundred tokens of
    # instructions for capabilities that do not exist — and a model told it can edit
    # files, in a mode where it cannot, will try. Mode decides which tools exist
    # (`core.session.Mode`), so mode decides whether the protocol is included.
    if mode != "chat":
        parts.append(load("system_tools"))
    # A mode without its own prompt file falls back to the core rules alone.
    with contextlib.suppress(PromptNotFoundError):
        parts.append(load(f"mode_{mode}"))
    return "\n\n".join(parts)


def available() -> list[str]:
    return sorted(p.stem for p in _PROMPTS_DIR.glob("*.md"))
