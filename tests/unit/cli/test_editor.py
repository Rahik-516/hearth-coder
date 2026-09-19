"""Handing text to `$EDITOR` — I3.

The editor is launched with an explicit argv and ``shell=False``. That is not defensive
theatre about the user's own environment: it is what keeps this from becoming a place where
a string that arrived from somewhere else turns into a command line. The test for it asserts
on the argv, because "we pass a list" is the whole guarantee and it is invisible otherwise.

The rest is about cancelling. Four different things mean "keep what you had" — no editor,
a non-zero exit, an empty buffer and an unchanged buffer — and each has to return None,
because the caller's fallback is the user's existing work.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hearth.cli.editor import edit_in_editor, editor_command


@pytest.fixture(autouse=True)
def clean_editor_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither variable set, so nothing here depends on the developer's shell."""
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)


# ------------------------------------------------------------ choosing one


def test_visual_wins_over_editor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Long convention: VISUAL names the full-screen editor, and this is a full-screen
    editing task."""
    monkeypatch.setenv("EDITOR", "ed")
    monkeypatch.setenv("VISUAL", "vim")

    assert editor_command() == ["vim"]


def test_an_editor_with_arguments_is_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """`EDITOR="code --wait"` is what anyone not using a terminal editor has set.
    Refusing it would silently send them to nano."""
    monkeypatch.setenv("EDITOR", "code --wait")

    assert editor_command() == ["code", "--wait"]


def test_an_unparseable_editor_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unbalanced quote is not a command. Guessing at what they meant and launching
    something else is worse than falling back."""
    monkeypatch.setenv("EDITOR", 'vim "--unterminated')

    assert editor_command() != ["vim", '"--unterminated']


def test_no_editor_anywhere_reports_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)

    assert editor_command() is None


# ---------------------------------------------------------------- launching


def test_the_editor_is_launched_as_argv_not_a_shell_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guarantee: a list, no `shell=True`.

    An EDITOR containing `;` then fails to exec rather than running whatever follows it.
    """
    seen: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        Path(argv[-1]).write_text("edited\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setenv("EDITOR", "myeditor --wait")
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert edit_in_editor("original\n") == "edited\n"
    argv = seen["argv"]
    assert isinstance(argv, list)
    assert argv[:2] == ["myeditor", "--wait"]
    assert "shell" not in seen["kwargs"], "no shell, not even shell=False by accident"


def test_the_suffix_reaches_the_editor(monkeypatch: pytest.MonkeyPatch) -> None:
    """So the plan opens with JSON highlighting rather than as plain text."""
    seen: list[str] = []

    def fake_run(argv, **kwargs):
        seen.append(argv[-1])
        Path(argv[-1]).write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setenv("EDITOR", "myeditor")
    monkeypatch.setattr(subprocess, "run", fake_run)

    edit_in_editor("original", suffix=".json")

    assert seen[0].endswith(".json")


# ---------------------------------------------------------------- cancelling


@pytest.mark.parametrize(
    ("returncode", "written", "why"),
    [
        (1, "edited\n", "a non-zero exit is the editor saying no"),
        (0, "", "an empty buffer is how people cancel a commit message"),
        (0, "original\n", "an unchanged buffer is nothing to apply"),
    ],
)
def test_cancelling_returns_none(
    monkeypatch: pytest.MonkeyPatch, returncode: int, written: str, why: str
) -> None:
    def fake_run(argv, **kwargs):
        Path(argv[-1]).write_text(written, encoding="utf-8")
        return subprocess.CompletedProcess(argv, returncode)

    monkeypatch.setenv("EDITOR", "myeditor")
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert edit_in_editor("original\n") is None, why


def test_a_missing_editor_binary_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """`EDITOR=nope` should not raise into a REPL turn."""

    def fake_run(argv, **kwargs):
        raise OSError("no such file")

    monkeypatch.setenv("EDITOR", "nope")
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert edit_in_editor("original\n") is None


def test_the_temporary_file_does_not_outlive_the_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The buffer can hold an unapproved plan or a commit message; it is not left behind."""
    seen: list[str] = []

    def fake_run(argv, **kwargs):
        seen.append(argv[-1])
        Path(argv[-1]).write_text("edited\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setenv("EDITOR", "myeditor")
    monkeypatch.setattr(subprocess, "run", fake_run)

    edit_in_editor("original\n")

    assert not Path(seen[0]).exists()
