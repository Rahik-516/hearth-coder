"""``/undo`` and ``/rewind`` at the CLI layer — docs/safety-and-tool-use.md §7.3.

The store is already tested in ``tests/unit/safety/test_checkpoints.py``; this covers the
one thing only the command layer has, which is M5's third acceptance criterion in full:
**the conflict prompt**. A file someone edited after Hearth wrote it must not be reverted
without a person saying so.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from hearth.cli.checkpoint_commands import rewind, show_checkpoints, undo
from hearth.safety.checkpoints import CheckpointStore
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository

ORIGINAL = b"def f():\r\n    return 1\r\n"
EDITED = b"def f():\r\n    return 2\r\n"


@pytest.fixture
def console() -> Console:
    # width fixed so assertions on the output are not at the mercy of the terminal
    return Console(record=True, width=100, no_color=True, legacy_windows=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return root


@pytest.fixture
def store(tmp_path: Path) -> CheckpointStore:
    connection = connect(tmp_path / "state.db")
    migrate(connection, database="state")
    return CheckpointStore(StateRepository(connection), BlobStore(tmp_path / "blobs"))


@pytest.fixture
def session(store: CheckpointStore, workspace: Path) -> str:
    return store.repo.create_session(workspace=str(workspace)).id


def never(question: str) -> bool:
    return False


def always(question: str) -> bool:
    return True


# ------------------------------------------------------------------- the easy path


def test_undo_restores_the_newest_step(
    console: Console, store: CheckpointStore, session: str, workspace: Path
) -> None:
    target = workspace / "a.py"
    target.write_bytes(EDITED)
    store.record(session_id=session, step=1, path="a.py", before=ORIGINAL, after=EDITED)

    changed = undo(console, store, session, workspace=workspace, confirm=never)

    assert changed
    assert target.read_bytes() == ORIGINAL
    assert "Reverted step 1" in console.export_text()


def test_undo_says_so_when_there_is_nothing_to_undo(
    console: Console, store: CheckpointStore, session: str, workspace: Path
) -> None:
    changed = undo(console, store, session, workspace=workspace, confirm=never)

    assert not changed
    assert "nothing to undo" in console.export_text()


def test_undo_reports_an_already_reverted_step(
    console: Console, store: CheckpointStore, session: str, workspace: Path
) -> None:
    target = workspace / "a.py"
    target.write_bytes(EDITED)
    store.record(session_id=session, step=1, path="a.py", before=ORIGINAL, after=EDITED)
    undo(console, store, session, workspace=workspace, confirm=never)

    changed = undo(console, store, session, workspace=workspace, step=1, confirm=never)

    assert not changed
    assert "already reverted" in console.export_text()


# --------------------------------------------------------------- the conflict prompt


def test_a_conflict_asks_before_overwriting(
    console: Console, store: CheckpointStore, session: str, workspace: Path
) -> None:
    """The acceptance criterion: "files changed afterwards (conflict prompt)"."""
    target = workspace / "a.py"
    store.record(session_id=session, step=1, path="a.py", before=ORIGINAL, after=EDITED)
    target.write_bytes(b"the user's own work\n")

    asked: list[str] = []

    def record_question(question: str) -> bool:
        asked.append(question)
        return False

    changed = undo(console, store, session, workspace=workspace, confirm=record_question)

    assert asked, "a conflict must ask"
    assert not changed
    assert target.read_bytes() == b"the user's own work\n", "declining leaves the file alone"
    output = console.export_text()
    assert "a.py" in output
    assert "edited since" in output


def test_declining_a_conflict_reverts_nothing_at_all(
    console: Console, store: CheckpointStore, session: str, workspace: Path
) -> None:
    """Not even the files that would have been clean.

    A step is reverted or it is not. Half-reverting would leave a state the user cannot
    reason about and `/checkpoints` cannot describe.
    """
    clean = workspace / "clean.py"
    dirty = workspace / "dirty.py"
    clean.write_bytes(EDITED)
    store.record(session_id=session, step=1, path="clean.py", before=ORIGINAL, after=EDITED)
    store.record(session_id=session, step=1, path="dirty.py", before=ORIGINAL, after=EDITED)
    dirty.write_bytes(b"mine\n")

    undo(console, store, session, workspace=workspace, confirm=never)

    assert clean.read_bytes() == EDITED, "untouched, because the step as a whole was refused"


def test_accepting_a_conflict_overwrites(
    console: Console, store: CheckpointStore, session: str, workspace: Path
) -> None:
    target = workspace / "a.py"
    store.record(session_id=session, step=1, path="a.py", before=ORIGINAL, after=EDITED)
    target.write_bytes(b"the user's own work\n")

    changed = undo(console, store, session, workspace=workspace, confirm=always)

    assert changed
    assert target.read_bytes() == ORIGINAL


def test_a_deleted_file_is_described_as_deleted(
    console: Console, store: CheckpointStore, session: str, workspace: Path
) -> None:
    """ "edited since" and "deleted since" call for different judgements from the user."""
    store.record(session_id=session, step=1, path="gone.py", before=ORIGINAL, after=EDITED)

    undo(console, store, session, workspace=workspace, confirm=never)

    assert "deleted since" in console.export_text()


# --------------------------------------------------------------------- rewind


def test_rewind_reverts_everything_after_a_step(
    console: Console, store: CheckpointStore, session: str, workspace: Path
) -> None:
    target = workspace / "a.py"
    store.record(session_id=session, step=1, path="a.py", before=b"v1", after=b"v2")
    store.record(session_id=session, step=2, path="a.py", before=b"v2", after=b"v3")
    target.write_bytes(b"v3")

    changed = rewind(console, store, session, 0, workspace=workspace, confirm=never)

    assert changed
    assert target.read_bytes() == b"v1"


def test_rewind_keeps_the_named_step(
    console: Console, store: CheckpointStore, session: str, workspace: Path
) -> None:
    target = workspace / "a.py"
    store.record(session_id=session, step=1, path="a.py", before=b"v1", after=b"v2")
    store.record(session_id=session, step=2, path="a.py", before=b"v2", after=b"v3")
    target.write_bytes(b"v3")

    rewind(console, store, session, 1, workspace=workspace, confirm=never)

    assert target.read_bytes() == b"v2"


# ---------------------------------------------------------------- the listing


def test_checkpoints_lists_steps_with_their_files(
    console: Console, store: CheckpointStore, session: str
) -> None:
    store.record(session_id=session, step=1, path="a.py", before=None, after=b"x")
    store.record(session_id=session, step=2, path="b.py", before=b"y", after=b"z")

    show_checkpoints(console, store, session)

    output = console.export_text()
    assert "a.py" in output
    assert "b.py" in output


def test_checkpoints_marks_a_reverted_step(
    console: Console, store: CheckpointStore, session: str, workspace: Path
) -> None:
    (workspace / "a.py").write_bytes(b"x")
    store.record(session_id=session, step=1, path="a.py", before=None, after=b"x")
    store.revert_step(session, 1, workspace=workspace)

    show_checkpoints(console, store, session)

    assert "reverted" in console.export_text()


def test_checkpoints_is_quiet_when_nothing_was_written(
    console: Console, store: CheckpointStore, session: str
) -> None:
    show_checkpoints(console, store, session)

    assert "no file changes" in console.export_text()
