"""Checkpoints, undo and rewind — docs/safety-and-tool-use.md §7.3.

Written before ``safety/checkpoints.py`` and ``storage/blobs.py`` (CLAUDE.md rule 3). This
file is M5's third acceptance criterion: "`/undo` restores exact bytes, including newly
created files (removed) and files changed afterwards (conflict prompt)".

"Exact bytes" is the whole point, and it is why the tests below are byte comparisons
rather than text ones. A checkpoint that restores a file's *content* while normalising its
BOM or line endings has not undone anything — it has made a second, silent edit at the
moment the user asked to be put back where they were.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hearth.safety.checkpoints import CheckpointStore
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository

ORIGINAL = b"def f():\r\n    return 1\r\n"
EDITED = b"def f():\r\n    return 2\r\n"
BOM_FILE = "﻿alpha\n".encode()


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
def session(store: CheckpointStore) -> str:
    return store.repo.create_session(workspace="/w").id


# --------------------------------------------------------------- the blob store


def test_a_blob_round_trips(tmp_path: Path) -> None:
    blobs = BlobStore(tmp_path / "blobs")

    digest = blobs.put(ORIGINAL)

    assert blobs.get(digest) == ORIGINAL


def test_identical_content_stores_once(tmp_path: Path) -> None:
    """Content-addressed, so re-editing the same file all session costs one copy."""
    blobs = BlobStore(tmp_path / "blobs")

    first = blobs.put(ORIGINAL)
    second = blobs.put(ORIGINAL)

    assert first == second
    assert blobs.count() == 1


def test_blobs_are_sharded(tmp_path: Path) -> None:
    """One flat directory with thousands of entries is slow to list on every filesystem."""
    blobs = BlobStore(tmp_path / "blobs")

    digest = blobs.put(ORIGINAL)

    assert blobs.path_for(digest).parent.name == digest[:2]


def test_an_absent_blob_is_reported_not_guessed(tmp_path: Path) -> None:
    blobs = BlobStore(tmp_path / "blobs")

    assert blobs.has("0" * 64) is False
    with pytest.raises(FileNotFoundError):
        blobs.get("0" * 64)


def test_empty_content_is_a_real_blob(tmp_path: Path) -> None:
    """An emptied file must be distinguishable from a file that never existed."""
    blobs = BlobStore(tmp_path / "blobs")

    digest = blobs.put(b"")

    assert blobs.has(digest)
    assert blobs.get(digest) == b""


# --------------------------------------------------------------- recording


def test_recording_a_modification_keeps_both_sides(
    store: CheckpointStore, session: str, workspace: Path
) -> None:
    target = workspace / "a.py"
    target.write_bytes(EDITED)

    store.record(session_id=session, step=1, path="a.py", before=ORIGINAL, after=EDITED)

    entries = store.entries_for_step(session, 1)
    assert len(entries) == 1
    assert store.blobs.get(entries[0].before_blob or "") == ORIGINAL
    assert store.blobs.get(entries[0].after_blob or "") == EDITED


def test_recording_a_creation_records_absence(store: CheckpointStore, session: str) -> None:
    """`before=None` means "did not exist", which undo turns into a removal.

    Storing an empty blob instead would make undo restore an empty file — leaving behind
    something the user never had.
    """
    store.record(session_id=session, step=1, path="new.py", before=None, after=b"x")

    assert store.entries_for_step(session, 1)[0].before_blob is None


# ------------------------------------------------------------------- undo


def test_undo_restores_exact_bytes_including_crlf(
    store: CheckpointStore, session: str, workspace: Path
) -> None:
    target = workspace / "a.py"
    target.write_bytes(ORIGINAL)
    store.record(session_id=session, step=1, path="a.py", before=ORIGINAL, after=EDITED)
    target.write_bytes(EDITED)

    report = store.revert_step(session, 1, workspace=workspace)

    assert report.restored == ["a.py"]
    assert target.read_bytes() == ORIGINAL


def test_undo_restores_a_bom(store: CheckpointStore, session: str, workspace: Path) -> None:
    target = workspace / "b.md"
    target.write_bytes(BOM_FILE)
    store.record(session_id=session, step=1, path="b.md", before=BOM_FILE, after=b"alpha\n")
    target.write_bytes(b"alpha\n")

    store.revert_step(session, 1, workspace=workspace)

    assert target.read_bytes() == BOM_FILE


def test_undo_removes_a_file_the_agent_created(store: CheckpointStore, session: str, workspace: Path) -> None:
    target = workspace / "new.py"
    target.write_bytes(b"x")
    store.record(session_id=session, step=1, path="new.py", before=None, after=b"x")

    report = store.revert_step(session, 1, workspace=workspace)

    assert report.removed == ["new.py"]
    assert not target.exists()


def test_a_removed_file_goes_to_trash_rather_than_away(
    store: CheckpointStore, session: str, workspace: Path, tmp_path: Path
) -> None:
    """§7.3: undo is itself reversible. Deleting outright would make it a one-way door."""
    trash = tmp_path / "trash"
    target = workspace / "new.py"
    target.write_bytes(b"keepme")
    store.record(session_id=session, step=1, path="new.py", before=None, after=b"keepme")

    store.revert_step(session, 1, workspace=workspace, trash=trash)

    assert not target.exists()
    assert any(path.read_bytes() == b"keepme" for path in trash.rglob("*") if path.is_file())


def test_undo_reverts_every_file_in_the_step(store: CheckpointStore, session: str, workspace: Path) -> None:
    """A step can touch several files; undo is per *step*, not per file."""
    for name in ("a.py", "b.py"):
        (workspace / name).write_bytes(b"after")
        store.record(session_id=session, step=1, path=name, before=b"before", after=b"after")

    report = store.revert_step(session, 1, workspace=workspace)

    assert sorted(report.restored) == ["a.py", "b.py"]
    assert (workspace / "a.py").read_bytes() == b"before"
    assert (workspace / "b.py").read_bytes() == b"before"


def test_a_reverted_step_is_marked_and_not_reverted_twice(
    store: CheckpointStore, session: str, workspace: Path
) -> None:
    target = workspace / "a.py"
    target.write_bytes(EDITED)
    store.record(session_id=session, step=1, path="a.py", before=ORIGINAL, after=EDITED)

    store.revert_step(session, 1, workspace=workspace)
    target.write_bytes(b"something the user wrote afterwards")
    second = store.revert_step(session, 1, workspace=workspace)

    assert second.already_reverted
    assert target.read_bytes() == b"something the user wrote afterwards", "must not stomp"


# --------------------------------------------------------------- conflicts


def test_a_file_changed_after_the_edit_is_a_conflict(
    store: CheckpointStore, session: str, workspace: Path
) -> None:
    """The user edited the file themselves after the agent did.

    Silently reverting would throw away their work. The bytes Hearth wrote are no longer
    what is on disk, so it has no basis for claiming the revert is a no-op.
    """
    target = workspace / "a.py"
    store.record(session_id=session, step=1, path="a.py", before=ORIGINAL, after=EDITED)
    target.write_bytes(b"the user's own change\n")

    report = store.revert_step(session, 1, workspace=workspace)

    assert report.restored == []
    assert [conflict.path for conflict in report.conflicts] == ["a.py"]
    assert target.read_bytes() == b"the user's own change\n", "nothing written without consent"


def test_a_conflict_can_be_overridden(store: CheckpointStore, session: str, workspace: Path) -> None:
    target = workspace / "a.py"
    store.record(session_id=session, step=1, path="a.py", before=ORIGINAL, after=EDITED)
    target.write_bytes(b"the user's own change\n")

    report = store.revert_step(session, 1, workspace=workspace, force=True)

    assert report.restored == ["a.py"]
    assert target.read_bytes() == ORIGINAL


def test_a_file_deleted_after_the_edit_is_a_conflict(
    store: CheckpointStore, session: str, workspace: Path
) -> None:
    store.record(session_id=session, step=1, path="a.py", before=ORIGINAL, after=EDITED)

    report = store.revert_step(session, 1, workspace=workspace)

    assert [conflict.path for conflict in report.conflicts] == ["a.py"]


def test_undoing_a_recorded_deletion_restores_the_file(
    store: CheckpointStore, session: str, workspace: Path
) -> None:
    """A deletion checkpoint is ``after=None``, and the file being absent is the state it
    describes — not a conflict.

    Treating a missing file as a conflict unconditionally made every delete un-undoable,
    which stayed invisible for as long as nothing recorded such a checkpoint. `delete_file`
    and the source end of `move_file` both do, so this is the test that keeps the previous
    test above from being read as the general rule.
    """
    target = workspace / "a.py"
    assert not target.exists(), "premise: the delete already happened"
    store.record(session_id=session, step=1, path="a.py", before=ORIGINAL, after=None)

    report = store.revert_step(session, 1, workspace=workspace)

    assert report.conflicts == []
    assert target.read_bytes() == ORIGINAL


def test_an_unchanged_file_is_not_a_conflict(store: CheckpointStore, session: str, workspace: Path) -> None:
    target = workspace / "a.py"
    target.write_bytes(EDITED)
    store.record(session_id=session, step=1, path="a.py", before=ORIGINAL, after=EDITED)

    report = store.revert_step(session, 1, workspace=workspace)

    assert report.conflicts == []
    assert target.read_bytes() == ORIGINAL


# ----------------------------------------------------------------- rewind


def test_rewind_reverts_later_steps_newest_first(
    store: CheckpointStore, session: str, workspace: Path
) -> None:
    """Newest first matters: two edits to one file only compose correctly in reverse."""
    target = workspace / "a.py"
    store.record(session_id=session, step=1, path="a.py", before=b"v1", after=b"v2")
    store.record(session_id=session, step=2, path="a.py", before=b"v2", after=b"v3")
    target.write_bytes(b"v3")

    report = store.revert_after(session, 0, workspace=workspace)

    assert target.read_bytes() == b"v1"
    assert report.restored.count("a.py") == 2


def test_rewind_keeps_steps_at_or_before_the_target(
    store: CheckpointStore, session: str, workspace: Path
) -> None:
    target = workspace / "a.py"
    store.record(session_id=session, step=1, path="a.py", before=b"v1", after=b"v2")
    store.record(session_id=session, step=2, path="a.py", before=b"v2", after=b"v3")
    target.write_bytes(b"v3")

    store.revert_after(session, 1, workspace=workspace)

    assert target.read_bytes() == b"v2", "step 1 stays applied"


# ------------------------------------------------------------------ listing


def test_steps_are_listed_newest_first_with_their_files(store: CheckpointStore, session: str) -> None:
    store.record(session_id=session, step=1, path="a.py", before=None, after=b"x")
    store.record(session_id=session, step=2, path="b.py", before=b"y", after=b"z")
    store.record(session_id=session, step=2, path="c.py", before=b"y", after=b"z")

    steps = store.steps(session)

    assert [step.step for step in steps] == [2, 1]
    assert steps[0].paths == ["b.py", "c.py"]
    assert steps[1].paths == ["a.py"]


def test_latest_step_finds_the_most_recent_unreverted_one(
    store: CheckpointStore, session: str, workspace: Path
) -> None:
    """What bare `/undo` acts on. A reverted step is not a candidate again."""
    (workspace / "a.py").write_bytes(b"x")
    store.record(session_id=session, step=1, path="a.py", before=None, after=b"x")
    store.record(session_id=session, step=2, path="b.py", before=None, after=b"x")

    assert store.latest_step(session) == 2
    store.revert_step(session, 2, workspace=workspace)
    assert store.latest_step(session) == 1


def test_latest_step_is_none_when_nothing_was_written(store: CheckpointStore, session: str) -> None:
    assert store.latest_step(session) is None


def test_checkpoints_are_scoped_to_one_session(store: CheckpointStore) -> None:
    first = store.repo.create_session(workspace="/w").id
    second = store.repo.create_session(workspace="/w").id
    store.record(session_id=first, step=1, path="a.py", before=None, after=b"x")

    assert store.steps(second) == []
