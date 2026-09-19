"""REPL completion: ``/commands`` and ``@path`` mentions.

Paths come from the index rather than the filesystem, so completion offers exactly what
Hearth can actually retrieve. Suggesting a file that was filtered out — a secret, a
vendored bundle — would promise context that will never arrive.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

#: Slash commands offered in chat mode, with one-line help.
SLASH_COMMANDS: dict[str, str] = {
    "/help": "show available commands",
    "/context": "show context budget usage",
    "/model": "show or switch the chat model",
    "/clear": "clear this session's history",
    "/sources": "show the context retrieved for the last turn",
    "/thinking": "toggle showing the model's reasoning",
    "/sessions": "list recent sessions",
    "/checkpoints": "list this session's file changes",
    "/undo": "revert the most recent file change",
    "/rewind": "revert every file change after a step",
    "/exit": "leave the REPL",
}

_MAX_PATH_SUGGESTIONS = 20


class ChatCompleter(Completer):
    """Completes slash commands at the start of a line and ``@path`` anywhere in it."""

    def __init__(self, connection: sqlite3.Connection | None = None) -> None:
        self._connection = connection
        self._paths: list[str] | None = None

    def get_completions(self, document: Document, complete_event: object) -> Iterable[Completion]:
        text = document.text_before_cursor

        if text.startswith("/") and " " not in text:
            yield from self._complete_commands(text)
            return

        at = text.rfind("@")
        if at != -1 and (at == 0 or text[at - 1].isspace()):
            yield from self._complete_paths(text[at + 1 :])

    def _complete_commands(self, prefix: str) -> Iterable[Completion]:
        for command, help_text in SLASH_COMMANDS.items():
            if command.startswith(prefix):
                yield Completion(
                    command,
                    start_position=-len(prefix),
                    display=command,
                    display_meta=help_text,
                )

    def _complete_paths(self, fragment: str) -> Iterable[Completion]:
        lowered = fragment.lower()
        matches = 0

        for path in self._indexed_paths():
            if matches >= _MAX_PATH_SUGGESTIONS:
                return
            if lowered and lowered not in path.lower():
                continue
            yield Completion(path, start_position=-len(fragment), display=path)
            matches += 1

    def _indexed_paths(self) -> list[str]:
        """Paths from the index, loaded once per REPL session.

        Cached because completion fires on every keystroke and a repository has thousands
        of files; re-querying per character would make typing stutter.
        """
        if self._paths is not None:
            return self._paths
        if self._connection is None:
            self._paths = []
            return self._paths

        try:
            rows = self._connection.execute("SELECT path FROM files ORDER BY path").fetchall()
            self._paths = [str(row[0]) for row in rows]
        except sqlite3.Error:
            self._paths = []
        return self._paths

    def invalidate(self) -> None:
        """Drop the cached paths, after a reindex."""
        self._paths = None


def extract_pins(text: str) -> tuple[str, list[str]]:
    """Split ``@path`` mentions out of a message.

    Returns the text with mentions removed and the list of pinned paths. The mentions are
    stripped because they are addressed to Hearth, not to the model — the file arrives as
    context, so leaving "@src/a.py" in the question just adds noise.
    """
    pins: list[str] = []
    words: list[str] = []

    for word in text.split():
        if word.startswith("@") and len(word) > 1:
            pins.append(word[1:].rstrip(".,;:"))
        else:
            words.append(word)

    return " ".join(words), pins
