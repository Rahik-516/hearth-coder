"""``multi_edit``, ``move_file`` and ``delete_file`` — I3, §2.2.

Three tools, three different promises, and each one is easy to implement in a way that
passes a happy-path test and is wrong:

* ``multi_edit`` is **atomic**. Four edits that reach the third and fail must leave the
  file exactly as it was. The tempting implementation applies each edit to disk in turn,
  which passes every test that only checks the end state of a batch where all four worked.
* ``move_file`` checkpoints **both ends**. Recording only the destination makes undo leave
  a copy at each path — the file is restored without being removed, which reads as success.
* ``delete_file`` never unlinks. The file goes to Hearth's trash, so a wrong delete costs a
  `/undo` rather than the file.

Every test here drives the tools directly, which ``resolve_in_workspace`` makes safe: these
are path tools, and the jail is in ``prepare()``. Nothing in this file runs a command.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hearth.config import paths
from hearth.safety.checkpoints import CheckpointStore
from hearth.safety.errors import PathError
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository
from hearth.tools.base import ToolContext
from hearth.tools.results import ErrorCode
from hearth.tools.write_fs import (
    DeleteFileTool,
    MoveFileTool,
    MultiEditTool,
)
from hearth.util.hashing import content_hash

SOURCE = "def one():\n    return 1\n\n\ndef two():\n    return 2\n"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text(SOURCE, encoding="utf-8")
    return root


@pytest.fixture
def checkpoints(tmp_path: Path) -> CheckpointStore:
    connection = connect(tmp_path / "state.db")
    migrate(connection, database="state")
    return CheckpointStore(StateRepository(connection), BlobStore(tmp_path / "blobs"))


@pytest.fixture
def session_id(checkpoints: CheckpointStore, workspace: Path) -> str:
    return checkpoints.repo.create_session(workspace=str(workspace)).id


@pytest.fixture
def context(workspace: Path, checkpoints: CheckpointStore, session_id: str) -> ToolContext:
    return ToolContext(workspace=workspace, checkpoints=checkpoints.bind(session_id))


def mark_read(context: ToolContext, workspace: Path, relative: str) -> None:
    context.record_read(relative, content_hash((workspace / relative).read_bytes()))


def run(tool, context: ToolContext, **arguments):
    """Validate, prepare, execute — the same order the gateway uses."""
    args = tool.args_model.model_validate(arguments)
    prepared = tool.prepare(args, context)
    if prepared.failed:
        return prepared, prepared.error
    return prepared, tool.execute(args, context, prepared)


# ================================================================= multi_edit


def test_several_edits_land_as_one_change(context: ToolContext, workspace: Path) -> None:
    mark_read(context, workspace, "a.py")

    _prepared, result = run(
        MultiEditTool(),
        context,
        path="a.py",
        edits=[
            {"old_string": "return 1", "new_string": "return 10"},
            {"old_string": "return 2", "new_string": "return 20"},
        ],
    )

    assert result.ok
    text = (workspace / "a.py").read_text(encoding="utf-8")
    assert "return 10" in text
    assert "return 20" in text


def test_the_previewed_diff_is_what_gets_written(context: ToolContext, workspace: Path) -> None:
    """One diff for the batch, and the bytes on disk are the bytes it described."""
    mark_read(context, workspace, "a.py")

    prepared, result = run(
        MultiEditTool(),
        context,
        path="a.py",
        edits=[
            {"old_string": "return 1", "new_string": "return 10"},
            {"old_string": "return 2", "new_string": "return 20"},
        ],
    )

    assert result.ok
    assert (workspace / "a.py").read_text(encoding="utf-8") == prepared.payload["new_text"]
    assert prepared.preview.count("@@") >= 1, "one unified diff, not two summaries"


def test_a_failing_edit_leaves_the_file_untouched(context: ToolContext, workspace: Path) -> None:
    """The atomicity guarantee, and the reason this tool is not a loop over `edit_file`.

    An implementation that wrote each edit as it went would leave the first two applied —
    a file in a state neither the model nor the user asked for, halfway between two
    designs.
    """
    mark_read(context, workspace, "a.py")

    _prepared, result = run(
        MultiEditTool(),
        context,
        path="a.py",
        edits=[
            {"old_string": "return 1", "new_string": "return 10"},
            {"old_string": "return 2", "new_string": "return 20"},
            {"old_string": "def three():", "new_string": "def four():"},
        ],
    )

    assert not result.ok
    assert (workspace / "a.py").read_text(encoding="utf-8") == SOURCE


def test_a_failing_edit_leaves_no_checkpoint(
    context: ToolContext, checkpoints: CheckpointStore, session_id: str, workspace: Path
) -> None:
    """Nothing happened, so nothing is offered to undo. A checkpoint for a write that
    never occurred makes `/checkpoints` a list of things that may or may not be real."""
    mark_read(context, workspace, "a.py")

    run(
        MultiEditTool(),
        context,
        path="a.py",
        edits=[{"old_string": "nowhere", "new_string": "x"}],
    )

    assert checkpoints.steps(session_id) == []


def test_a_failure_says_which_edit_and_what_state_it_ran_against(
    context: ToolContext, workspace: Path
) -> None:
    """Edits compose, so edit 3 runs against the file as edits 1-2 left it.

    A model told only "edit 3 did not match" re-sends edit 3 verbatim — because against
    the file it can see, edit 3 is correct. The message has to name the composition.
    """
    mark_read(context, workspace, "a.py")

    _prepared, result = run(
        MultiEditTool(),
        context,
        path="a.py",
        edits=[
            {"old_string": "return 1", "new_string": "return 10"},
            {"old_string": "return 2", "new_string": "return 20"},
            # `return 1` on its own would still match, because edit 1 left `return 10`
            # behind and that contains it. This spans the line, so it cannot.
            {
                "old_string": "def one():\n    return 1\n",
                "new_string": "def zero():\n",
            },
        ],
    )

    assert not result.ok
    assert "edit 3 of 3" in result.content
    assert "1-2" in result.content, "it says the earlier edits had already been applied"


def test_a_later_edit_sees_an_earlier_one(context: ToolContext, workspace: Path) -> None:
    """The property that makes this worth having: edits chain within one call."""
    mark_read(context, workspace, "a.py")

    _prepared, result = run(
        MultiEditTool(),
        context,
        path="a.py",
        edits=[
            {"old_string": "return 1", "new_string": "return ONE"},
            {"old_string": "return ONE", "new_string": "return 1000"},
        ],
    )

    assert result.ok
    assert "return 1000" in (workspace / "a.py").read_text(encoding="utf-8")


def test_multi_edit_requires_a_prior_read(context: ToolContext) -> None:
    """Read-before-write applies exactly as it does to `edit_file`; batching is not a
    way around it."""
    _prepared, result = run(
        MultiEditTool(),
        context,
        path="a.py",
        edits=[{"old_string": "return 1", "new_string": "return 10"}],
    )

    assert result.error is ErrorCode.STALE_FILE


def test_multi_edit_shares_edit_files_grant_key(context: ToolContext, workspace: Path) -> None:
    """A grant is permission to change *this file*. Which tool does the changing is not
    something the user was asked about, so a second key would mean being asked twice for
    the identical write."""
    mark_read(context, workspace, "a.py")
    tool = MultiEditTool()
    args = tool.args_model.model_validate(
        {"path": "a.py", "edits": [{"old_string": "return 1", "new_string": "return 10"}]}
    )

    prepared = tool.prepare(args, context)

    assert prepared.facts.grant_key == "edit:a.py"


def test_multi_edit_cannot_escape_the_workspace(context: ToolContext) -> None:
    _tool = MultiEditTool()
    args = _tool.args_model.model_validate(
        {"path": "../escape.py", "edits": [{"old_string": "a", "new_string": "b"}]}
    )

    with pytest.raises(PathError):
        _tool.prepare(args, context)


def test_edits_that_cancel_out_are_refused(context: ToolContext, workspace: Path) -> None:
    """Every edit matched, and the file is unchanged. Writing it would produce an empty
    diff, a checkpoint with nothing in it, and a reported success with no effect."""
    mark_read(context, workspace, "a.py")

    _prepared, result = run(
        MultiEditTool(),
        context,
        path="a.py",
        edits=[
            {"old_string": "return 1", "new_string": "return ONE"},
            {"old_string": "return ONE", "new_string": "return 1"},
        ],
    )

    assert not result.ok
    assert result.error is ErrorCode.INVALID_ARGUMENTS


# ================================================================== move_file


def test_a_move_relocates_the_file(context: ToolContext, workspace: Path) -> None:
    _prepared, result = run(MoveFileTool(), context, src="a.py", dst="pkg/b.py")

    assert result.ok
    assert not (workspace / "a.py").exists()
    assert (workspace / "pkg" / "b.py").read_text(encoding="utf-8") == SOURCE


def test_a_move_does_not_require_a_prior_read(context: ToolContext, workspace: Path) -> None:
    """Read-before-write exists so a diff is computed against bytes someone has seen. A
    move changes no bytes, so requiring it would only make the model read a file in order
    to rename it."""
    _prepared, result = run(MoveFileTool(), context, src="a.py", dst="b.py")

    assert result.ok


def test_a_move_checkpoints_both_ends(
    context: ToolContext, checkpoints: CheckpointStore, session_id: str, workspace: Path
) -> None:
    """Recording only the destination makes undo restore the file without removing it,
    leaving a copy at each path — which reads as a successful undo."""
    run(MoveFileTool(), context, src="a.py", dst="b.py")

    entries = checkpoints.entries_for_step(session_id, 0)
    by_path = {entry.path: entry for entry in entries}
    assert set(by_path) == {"a.py", "b.py"}
    assert by_path["a.py"].after_blob is None, "the source is gone"
    assert by_path["b.py"].before_blob is None, "the destination is new"


def test_undoing_a_move_leaves_no_copy(
    context: ToolContext, checkpoints: CheckpointStore, session_id: str, workspace: Path
) -> None:
    """The end-to-end version of the claim above, through the real revert path."""
    run(MoveFileTool(), context, src="a.py", dst="b.py")

    checkpoints.revert_step(session_id, 0, workspace=workspace)

    assert (workspace / "a.py").read_text(encoding="utf-8") == SOURCE
    assert not (workspace / "b.py").exists()


def test_moving_onto_an_existing_file_is_refused(context: ToolContext, workspace: Path) -> None:
    """Silently clobbering the destination would destroy a file the user never saw
    mentioned in the approval."""
    (workspace / "b.py").write_text("keep me\n", encoding="utf-8")

    _prepared, result = run(MoveFileTool(), context, src="a.py", dst="b.py")

    assert not result.ok
    assert (workspace / "b.py").read_text(encoding="utf-8") == "keep me\n"


def test_moving_a_missing_file_is_refused(context: ToolContext) -> None:
    _prepared, result = run(MoveFileTool(), context, src="nope.py", dst="b.py")

    assert result.error is ErrorCode.NOT_FOUND


def test_moving_a_file_onto_itself_is_refused(context: ToolContext) -> None:
    _prepared, result = run(MoveFileTool(), context, src="a.py", dst="a.py")

    assert result.error is ErrorCode.INVALID_ARGUMENTS


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        ("a.py", "../escape.py"),
        ("../../etc/passwd", "a.py"),
        ("a.py", ".git/config"),
        (".git/config", "a.py"),
    ],
)
def test_neither_end_of_a_move_can_escape_the_jail(
    context: ToolContext, src: str, dst: str
) -> None:
    """The source resolves for *write* too, because a move deletes it. Resolving it
    read-only would let a file be moved out of `.git/` by a call the jail never refused.
    """
    tool = MoveFileTool()
    args = tool.args_model.model_validate({"src": src, "dst": dst})

    with pytest.raises(PathError):
        tool.prepare(args, context)


def test_a_moved_files_read_hash_follows_it(context: ToolContext, workspace: Path) -> None:
    """Otherwise a model that moves a file it has read and then edits it at the new path
    is told to read it again — for a rename it performed itself."""
    mark_read(context, workspace, "a.py")
    before = context.hash_at_last_read("a.py")

    run(MoveFileTool(), context, src="a.py", dst="b.py")

    assert context.hash_at_last_read("b.py") == before


# ================================================================ delete_file


def test_a_delete_removes_the_file_from_the_workspace(
    context: ToolContext, workspace: Path
) -> None:
    _prepared, result = run(DeleteFileTool(), context, path="a.py")

    assert result.ok
    assert not (workspace / "a.py").exists()


def test_a_deleted_file_is_in_the_trash_not_gone(context: ToolContext, workspace: Path) -> None:
    """An agent that can permanently remove a file is an agent whose worst mistake is
    unbounded."""
    run(DeleteFileTool(), context, path="a.py")

    trashed = list(paths.trash_dir(workspace).rglob("a.py"))
    assert len(trashed) == 1
    assert trashed[0].read_text(encoding="utf-8") == SOURCE


def test_undo_restores_a_deleted_file(
    context: ToolContext, checkpoints: CheckpointStore, session_id: str, workspace: Path
) -> None:
    run(DeleteFileTool(), context, path="a.py")

    checkpoints.revert_step(session_id, 0, workspace=workspace)

    assert (workspace / "a.py").read_text(encoding="utf-8") == SOURCE


def test_deleting_an_untracked_file_is_destructive(context: ToolContext) -> None:
    """No git copy, so Hearth's trash is the only one. That earns a typed confirmation."""
    tool = DeleteFileTool()
    args = tool.args_model.model_validate({"path": "a.py"})

    prepared = tool.prepare(args, context)

    assert "DESTRUCTIVE" in prepared.badges
    assert prepared.facts.destructive


def test_the_preview_states_size_and_git_status(context: ToolContext) -> None:
    """The two facts that decide how bad a wrong delete would be."""
    tool = DeleteFileTool()
    args = tool.args_model.model_validate({"path": "a.py"})

    prepared = tool.prepare(args, context)

    assert "line(s)" in prepared.preview
    assert "git:" in prepared.preview
    assert "trash" in prepared.preview


def test_delete_is_never_grantable(context: ToolContext) -> None:
    """"Always delete files matching this" is not a permission anybody means to give."""
    tool = DeleteFileTool()
    args = tool.args_model.model_validate({"path": "a.py"})

    prepared = tool.prepare(args, context)

    assert prepared.facts.grant_key is None


def test_deleting_a_directory_is_refused(context: ToolContext, workspace: Path) -> None:
    """One file at a time. A recursive delete behind one approval is the shape of the
    mistake this whole tool is built to bound."""
    (workspace / "pkg").mkdir()

    _prepared, result = run(DeleteFileTool(), context, path="pkg")

    assert result.error is ErrorCode.INVALID_ARGUMENTS


def test_deleting_a_missing_file_is_refused(context: ToolContext) -> None:
    _prepared, result = run(DeleteFileTool(), context, path="nope.py")

    assert result.error is ErrorCode.NOT_FOUND


@pytest.mark.parametrize("hostile", ["../outside.py", ".git/config", ".hearth/config.toml"])
def test_delete_cannot_escape_the_jail(context: ToolContext, hostile: str) -> None:
    tool = DeleteFileTool()
    args = tool.args_model.model_validate({"path": hostile})

    with pytest.raises(PathError):
        tool.prepare(args, context)


def test_a_file_that_changed_during_approval_is_not_deleted(
    context: ToolContext, workspace: Path
) -> None:
    """The gap between prepare and execute is an approval prompt a person may take a
    minute over. What they agreed to delete is the file they were shown."""
    tool = DeleteFileTool()
    args = tool.args_model.model_validate({"path": "a.py"})
    prepared = tool.prepare(args, context)

    (workspace / "a.py").write_text("someone else wrote this\n", encoding="utf-8")
    result = tool.execute(args, context, prepared)

    assert result.error is ErrorCode.STALE_FILE
    assert (workspace / "a.py").exists()


def test_a_delete_without_checkpoints_is_refused(workspace: Path) -> None:
    """An irreversible delete is not a degraded delete; it is a different operation."""
    tool = DeleteFileTool()
    context = ToolContext(workspace=workspace)
    args = tool.args_model.model_validate({"path": "a.py"})
    prepared = tool.prepare(args, context)

    result = tool.execute(args, context, prepared)

    assert not result.ok
    assert (workspace / "a.py").exists()


def test_deleting_the_same_path_twice_keeps_both_copies(
    context: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trash is the copy that exists so nothing is quietly destroyed. A second delete
    of the same path must not overwrite the first file in it."""
    stamps = iter(["20260920-120000", "20260920-120001"])
    monkeypatch.setattr("time.strftime", lambda _fmt: next(stamps))

    run(DeleteFileTool(), context, path="a.py")
    (workspace / "a.py").write_text("a different file\n", encoding="utf-8")
    run(DeleteFileTool(), context, path="a.py")

    trashed = sorted(paths.trash_dir(workspace).rglob("a.py"))
    assert len(trashed) == 2
    assert {path.read_text(encoding="utf-8") for path in trashed} == {
        SOURCE,
        "a different file\n",
    }
