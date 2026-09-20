"""Handing text to the user's ``$EDITOR`` and taking back what they saved.

Used by plan review (§8.3 step 3) and, later, by approval-time argument editing. The point
in both cases is that correcting a mistake should cost a keystroke rather than a re-run.

Two things make this narrower than it looks:

* **The editor is the user's, not the model's.** The command comes from ``$VISUAL`` or
  ``$EDITOR`` — environment the person set — and never from anything a model produced. It
  is launched with ``shell=False`` and an explicit argv, so an ``EDITOR`` containing shell
  metacharacters fails to start rather than running a shell. That is not this module
  protecting the user from their own environment; it is making sure a value that could
  reach here from *somewhere else* cannot become a command line.
* **Cancelling means cancelling.** An unchanged buffer, an empty file, or a non-zero exit
  all return None. The caller keeps what it had, which is the answer that loses no work.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

#: Tried in order. ``VISUAL`` first by long convention: it names the full-screen editor,
#: and this is a full-screen editing task.
_EDITOR_VARS = ("VISUAL", "EDITOR")

#: What to launch when neither is set. Each is tried only if it is actually on PATH, so a
#: missing editor is reported as "set $EDITOR" rather than as a FileNotFoundError.
_FALLBACKS = ("nano", "vi")


def editor_command() -> list[str] | None:
    """The argv to launch, or None when no editor can be found.

    Split with ``shlex`` so ``EDITOR="code --wait"`` works — which is the common case for
    anyone not using a terminal editor, and refusing it would send them to nano.
    """
    for variable in _EDITOR_VARS:
        raw = os.environ.get(variable, "").strip()
        if not raw:
            continue
        try:
            argv = shlex.split(raw)
        except ValueError:
            # An unbalanced quote. Falling back to the next candidate is better than
            # guessing at what they meant and launching something else.
            continue
        if argv:
            return argv

    from shutil import which

    for fallback in _FALLBACKS:
        if which(fallback):
            return [fallback]
    return None


def edit_in_editor(content: str, *, suffix: str = ".txt") -> str | None:
    """Open ``content`` in the user's editor and return what they saved.

    Returns None when there is no editor, the editor exited non-zero, or the content came
    back unchanged or empty. Callers treat all four as "keep what you had": an empty buffer
    is how people cancel out of a commit message, and honouring it as "the user wants
    nothing" would discard their work on the strength of a habit.

    Args:
        content: The starting text.
        suffix: File extension, so the editor picks the right syntax highlighting.
    """
    argv = editor_command()
    if argv is None:
        return None

    with tempfile.TemporaryDirectory(prefix="hearth-edit-") as directory:
        path = Path(directory) / f"buffer{suffix}"
        path.write_text(content, encoding="utf-8")

        try:
            # S603: `argv` comes from the user's own $EDITOR and nowhere else, and
            # the list form means no shell interprets it — an EDITOR containing `;`
            # fails to exec rather than running anything. The one argument this call
            # adds is a path Hearth created in a temporary directory.
            completed = subprocess.run([*argv, str(path)], check=False)  # noqa: S603
        except OSError:
            # An `EDITOR` that does not exist, or is not executable.
            return None

        if completed.returncode != 0:
            return None

        edited = path.read_text(encoding="utf-8")

    if not edited.strip() or edited == content:
        return None
    return edited
