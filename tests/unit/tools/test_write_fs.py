"""``edit_file`` and ``write_file`` — docs/safety-and-tool-use.md §7.2.

Written before ``tools/write_fs.py`` (CLAUDE.md rule 3). These are the first tools in
Hearth that can change a file, so the tests are organised around the guarantees rather
than the happy path:

* **what you see is what runs** — the bytes on disk equal the previewed bytes, exactly;
* **read before write** — an edit to a file the model has not read is refused;
* **TOCTOU** — a file that changes between prepare and execute is not written;
* **reversibility** — every write leaves a checkpoint with the real before-bytes.

The happy path gets one test. The refusals get most of the file, because a write tool that
is merely usually right is not usable at all.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from hearth.safety.checkpoints import CheckpointStore
from hearth.safety.errors import PathError
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository
from hearth.tools.base import ToolContext
from hearth.tools.results import ErrorCode
from hearth.tools.write_fs import EditFileTool, WriteFileArgs, WriteFileTool
from hearth.util.hashing import content_hash

SOURCE = "def finalize(self):\n    total = compute()\n    return total\n"
CRLF_SOURCE = b"def f():\r\n    return 1\r\n"
BOM_SOURCE = "\ufeff# notes\nalpha\n".encode()


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
def context(workspace: Path, checkpoints: CheckpointStore) -> ToolContext:
    session = checkpoints.repo.create_session(workspace=str(workspace)).id
    return ToolContext(
        workspace=workspace,
        checkpoints=checkpoints.bind(session),
    )


def mark_read(context: ToolContext, workspace: Path, relative: str) -> None:
    """Pretend `read_file` was called, which is what the real flow requires."""
    context.record_read(relative, content_hash((workspace / relative).read_bytes()))


def run_edit(context: ToolContext, *, path: str = "a.py", old: str, new: str, **kwargs: object):
    tool = EditFileTool()
    args = tool.args_model.model_validate({"path": path, "old_string": old, "new_string": new, **kwargs})
    prepared = tool.prepare(args, context)  # type: ignore[arg-type]
    if prepared.failed:
        return prepared, prepared.error
    return prepared, tool.execute(args, context, prepared)  # type: ignore[arg-type]


# -------------------------------------------------------------- the happy path


def test_an_approved_edit_writes_exactly_the_previewed_bytes(context: ToolContext, workspace: Path) -> None:
    """The core promise (§1.3.3). The preview and the write share one computation."""
    mark_read(context, workspace, "a.py")

    prepared, result = run_edit(context, old="    return total", new="    return round(total)")

    assert result.ok
    expected: str = prepared.payload["new_text"]
    assert (workspace / "a.py").read_text(encoding="utf-8") == expected
    assert "return round(total)" in expected


def test_the_preview_is_a_unified_diff(context: ToolContext, workspace: Path) -> None:
    mark_read(context, workspace, "a.py")
    tool = EditFileTool()
    args = tool.args_model.model_validate(
        {"path": "a.py", "old_string": "    return total", "new_string": "    return 0"}
    )

    prepared = tool.prepare(args, context)  # type: ignore[arg-type]

    assert "-    return total" in prepared.preview
    assert "+    return 0" in prepared.preview
    assert prepared.payload["added"] == 1
    assert prepared.payload["removed"] == 1


def test_prepare_does_not_touch_the_file(context: ToolContext, workspace: Path) -> None:
    """``prepare()`` must not mutate anything — that is what makes a preview safe to show
    for a call the user then rejects."""
    mark_read(context, workspace, "a.py")
    before = (workspace / "a.py").read_bytes()
    tool = EditFileTool()
    args = tool.args_model.model_validate(
        {"path": "a.py", "old_string": "    return total", "new_string": "    return 0"}
    )

    tool.prepare(args, context)  # type: ignore[arg-type]

    assert (workspace / "a.py").read_bytes() == before


# ----------------------------------------------------------- read before write


def test_editing_an_unread_file_is_refused(context: ToolContext) -> None:
    """§7.2 item 1. The model must have seen the file it is changing.

    Without this, a model that guessed a file's contents from its name could land an edit
    that happens to match — and the user would approve a diff against text the model never
    actually read.
    """
    _, result = run_edit(context, old="    return total", new="    return 0")

    assert not result.ok
    assert result.error is ErrorCode.STALE_FILE
    assert "read_file" in result.content


def test_editing_a_file_changed_since_it_was_read_is_refused(context: ToolContext, workspace: Path) -> None:
    mark_read(context, workspace, "a.py")
    (workspace / "a.py").write_text(SOURCE + "# someone else edited this\n", encoding="utf-8")

    _, result = run_edit(context, old="    return total", new="    return 0")

    assert not result.ok
    assert result.error is ErrorCode.STALE_FILE


def test_a_stale_file_error_is_retryable(context: ToolContext) -> None:
    """The model can fix this itself by reading the file, so it must not burn the budget."""
    _, result = run_edit(context, old="    return total", new="    return 0")

    assert result.is_retryable


# ------------------------------------------------------------------- TOCTOU


def test_a_file_changed_between_prepare_and_execute_is_not_written(
    context: ToolContext, workspace: Path
) -> None:
    """The approval gap is the dangerous window (§1.3.3, §7.2 item 7.1).

    The user may take a minute over the prompt. If the file moves in that time, the diff
    they approved describes a file that no longer exists, so execute must abort rather
    than apply the edit to whatever is there now.
    """
    mark_read(context, workspace, "a.py")
    tool = EditFileTool()
    args = tool.args_model.model_validate(
        {"path": "a.py", "old_string": "    return total", "new_string": "    return 0"}
    )
    prepared = tool.prepare(args, context)  # type: ignore[arg-type]

    intervening = SOURCE.replace("compute()", "compute_total()")
    (workspace / "a.py").write_text(intervening, encoding="utf-8")

    result = tool.execute(args, context, prepared)  # type: ignore[arg-type]

    assert not result.ok
    assert result.error is ErrorCode.STALE_FILE
    assert (workspace / "a.py").read_text(encoding="utf-8") == intervening, "no write happened"


# ----------------------------------------------------------------- the jail


@pytest.mark.parametrize(
    "path",
    [
        "../escape.py",
        "/etc/hosts",
        ".git/config",
        ".git/hooks/pre-commit",
        ".hearth/config.toml",  # the agent cannot edit its own permissions (§5.2)
    ],
)
def test_the_jail_refuses_a_write_by_raising(context: ToolContext, path: str) -> None:
    """``PathError`` propagates out of ``prepare()`` rather than becoming a tool error.

    That distinction is what the audit log is read for. The gateway has a dedicated branch
    that records a jail refusal as a *denial by the hard invariants*; if the tool swallowed
    it into a `Prepared` error, the same event would be logged as "the tool could not run"
    — and "what did policy block?" would come back missing its most important entries.
    """
    tool = EditFileTool()
    args = tool.args_model.model_validate({"path": path, "old_string": "x", "new_string": "y"})

    with pytest.raises(PathError):
        tool.prepare(args, context)  # type: ignore[arg-type]


# ------------------------------------------------------------- form preserved


def test_a_crlf_file_stays_crlf(context: ToolContext, workspace: Path) -> None:
    (workspace / "w.py").write_bytes(CRLF_SOURCE)
    mark_read(context, workspace, "w.py")

    _, result = run_edit(context, path="w.py", old="    return 1", new="    return 2")

    assert result.ok
    assert (workspace / "w.py").read_bytes() == b"def f():\r\n    return 2\r\n"


def test_a_bom_survives(context: ToolContext, workspace: Path) -> None:
    (workspace / "n.md").write_bytes(BOM_SOURCE)
    mark_read(context, workspace, "n.md")

    _, result = run_edit(context, path="n.md", old="alpha", new="beta")

    assert result.ok
    assert (workspace / "n.md").read_bytes() == "\ufeff# notes\nbeta\n".encode()


def test_mode_bits_are_preserved(context: ToolContext, workspace: Path) -> None:
    """An edit to a script must not make it non-executable."""
    script = workspace / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    script.chmod(0o755)
    mark_read(context, workspace, "run.sh")

    _, result = run_edit(context, path="run.sh", old="echo hi", new="echo bye")

    assert result.ok
    assert stat.S_IMODE(script.stat().st_mode) == 0o755


def test_a_binary_file_is_refused(context: ToolContext, workspace: Path) -> None:
    target = workspace / "blob.bin"
    target.write_bytes(b"\x00\x01\x02")
    mark_read(context, workspace, "blob.bin")

    _, result = run_edit(context, path="blob.bin", old="x", new="y")

    assert not result.ok


# --------------------------------------------------------------- checkpoints


def test_an_edit_leaves_a_checkpoint_with_the_real_before_bytes(
    context: ToolContext, workspace: Path, checkpoints: CheckpointStore
) -> None:
    mark_read(context, workspace, "a.py")
    original = (workspace / "a.py").read_bytes()
    context.step = 3

    _, result = run_edit(context, old="    return total", new="    return 0")

    assert result.ok
    entries = checkpoints.entries_for_step(context.checkpoints.session_id, 3)
    assert len(entries) == 1
    assert checkpoints.blobs.get(entries[0].before_blob or "") == original
    assert checkpoints.blobs.get(entries[0].after_blob or "") == (workspace / "a.py").read_bytes()


def test_the_edit_can_be_undone_to_the_exact_bytes(
    context: ToolContext, workspace: Path, checkpoints: CheckpointStore
) -> None:
    """The end-to-end reversibility claim, through the real tool."""
    mark_read(context, workspace, "a.py")
    original = (workspace / "a.py").read_bytes()
    context.step = 1

    run_edit(context, old="    return total", new="    return 0")
    report = checkpoints.revert_step(context.checkpoints.session_id, 1, workspace=workspace)

    assert report.restored == ["a.py"]
    assert (workspace / "a.py").read_bytes() == original


def test_a_refused_edit_leaves_no_checkpoint(context: ToolContext, checkpoints: CheckpointStore) -> None:
    """Nothing happened, so there must be nothing to undo."""
    run_edit(context, old="    return total", new="    return 0")

    assert checkpoints.steps(context.checkpoints.session_id) == []


# ------------------------------------------------------------------- badges


def test_a_fuzzy_match_is_badged(context: ToolContext, workspace: Path) -> None:
    (workspace / "f.py").write_text("class A:\n    def f(self):\n        return 1\n", encoding="utf-8")
    mark_read(context, workspace, "f.py")
    tool = EditFileTool()
    args = tool.args_model.model_validate(
        {
            "path": "f.py",
            "old_string": "def f(self):\n    return 1",
            "new_string": "def f(self):\n    return 2",
        }
    )

    prepared = tool.prepare(args, context)  # type: ignore[arg-type]

    assert "FUZZY-MATCH" in prepared.badges


def test_an_edit_introducing_a_syntax_error_is_badged(context: ToolContext, workspace: Path) -> None:
    """M5 acceptance criterion 4, through the tool rather than the engine."""
    mark_read(context, workspace, "a.py")
    tool = EditFileTool()
    args = tool.args_model.model_validate(
        {"path": "a.py", "old_string": "    return total", "new_string": "    return total((("}
    )

    prepared = tool.prepare(args, context)  # type: ignore[arg-type]

    assert "PARSE-ERRORS-INTRODUCED" in prepared.badges


def test_writing_a_credential_is_badged(context: ToolContext, workspace: Path) -> None:
    """§10: the scan warns; it does not silently refuse.

    A legitimate test fixture may well contain a fake-looking key, so this is information
    for the approval prompt rather than a veto.
    """
    mark_read(context, workspace, "a.py")
    tool = EditFileTool()
    args = tool.args_model.model_validate(
        {
            "path": "a.py",
            "old_string": "    return total",
            "new_string": '    token = "ghp_0123456789abcdefghijklmnopqrstuvwxyz"',
        }
    )

    prepared = tool.prepare(args, context)  # type: ignore[arg-type]

    assert "SECRET?" in prepared.badges


def test_the_policy_facts_name_the_resolved_path(context: ToolContext, workspace: Path) -> None:
    """What the gateway hands the policy engine. A wrong path here is a wrong decision."""
    mark_read(context, workspace, "a.py")
    tool = EditFileTool()
    args = tool.args_model.model_validate(
        {"path": "a.py", "old_string": "    return total", "new_string": "    return 0"}
    )

    prepared = tool.prepare(args, context)  # type: ignore[arg-type]

    assert prepared.facts.path == "a.py"
    assert prepared.facts.inside_workspace
    assert prepared.facts.grant_key == "edit:a.py"


# ----------------------------------------------------------------- write_file


def test_write_file_creates_a_new_file(context: ToolContext, workspace: Path) -> None:
    tool = WriteFileTool()
    args = WriteFileArgs(path="new/deep/mod.py", content="print('hi')\n")

    prepared = tool.prepare(args, context)
    result = tool.execute(args, context, prepared)

    assert result.ok
    assert (workspace / "new/deep/mod.py").read_text(encoding="utf-8") == "print('hi')\n"


def test_creating_a_file_needs_no_prior_read(context: ToolContext) -> None:
    """There is nothing to have read. Requiring it would make creation impossible."""
    tool = WriteFileTool()
    args = WriteFileArgs(path="brand_new.py", content="x = 1\n")

    prepared = tool.prepare(args, context)

    assert not prepared.failed


def test_creating_a_file_checkpoints_its_absence(context: ToolContext, checkpoints: CheckpointStore) -> None:
    """So undo removes it rather than restoring an empty file."""
    context.step = 2
    tool = WriteFileTool()
    args = WriteFileArgs(path="brand_new.py", content="x = 1\n")
    tool.execute(args, context, tool.prepare(args, context))

    entries = checkpoints.entries_for_step(context.checkpoints.session_id, 2)

    assert entries[0].before_blob is None
    assert entries[0].was_created


def test_overwriting_an_existing_file_needs_a_prior_read(context: ToolContext) -> None:
    """Overwriting is the most destructive thing `write_file` does (§7.2 item 1)."""
    tool = WriteFileTool()
    args = WriteFileArgs(path="a.py", content="replaced\n")

    prepared = tool.prepare(args, context)

    assert prepared.failed
    assert prepared.error is not None
    assert prepared.error.error is ErrorCode.STALE_FILE


def test_overwriting_shows_a_diff_not_just_the_content(context: ToolContext, workspace: Path) -> None:
    mark_read(context, workspace, "a.py")
    tool = WriteFileTool()
    args = WriteFileArgs(path="a.py", content="def finalize(self):\n    return 0\n")

    prepared = tool.prepare(args, context)

    assert "-    return total" in prepared.preview


def test_content_over_two_megabytes_is_refused(context: ToolContext) -> None:
    tool = WriteFileTool()
    args = WriteFileArgs(path="big.txt", content="x" * (2 * 1024 * 1024 + 1))

    prepared = tool.prepare(args, context)

    assert prepared.failed
    assert prepared.error is not None
    assert prepared.error.error is ErrorCode.TOO_LARGE


def test_discarding_a_large_file_is_destructive(context: ToolContext, workspace: Path) -> None:
    """§6.2: overwriting many lines is a classification overlay, so it asks with a typed
    confirmation even at auto-edit."""
    big = workspace / "big.py"
    big.write_text("".join(f"line_{n} = {n}\n" for n in range(300)), encoding="utf-8")
    mark_read(context, workspace, "big.py")
    tool = WriteFileTool()
    args = WriteFileArgs(path="big.py", content="x = 1\n")

    prepared = tool.prepare(args, context)

    assert prepared.facts.destructive


def test_a_small_overwrite_is_not_destructive(context: ToolContext, workspace: Path) -> None:
    mark_read(context, workspace, "a.py")
    tool = WriteFileTool()
    args = WriteFileArgs(path="a.py", content="def finalize(self):\n    return 0\n")

    prepared = tool.prepare(args, context)

    assert not prepared.facts.destructive


def test_write_file_refuses_a_directory(context: ToolContext, workspace: Path) -> None:
    (workspace / "adir").mkdir()
    tool = WriteFileTool()
    args = WriteFileArgs(path="adir", content="x")

    prepared = tool.prepare(args, context)

    assert prepared.failed


# -------------------------------------------------------------------- reindex


def test_a_write_reindexes_the_file_synchronously(context: ToolContext, workspace: Path) -> None:
    """The model's next `find_symbol` has to see the edit it just made (§7.2 item 7.6)."""
    reindexed: list[str] = []
    context.reindex = reindexed.append
    mark_read(context, workspace, "a.py")

    _, result = run_edit(context, old="    return total", new="    return 0")

    assert result.ok
    assert reindexed == ["a.py"]


def test_a_failing_reindex_does_not_fail_the_write(context: ToolContext, workspace: Path) -> None:
    """The bytes are already on disk. Reporting failure would tell the model to retry an
    edit that has in fact been applied — which is how a file gets edited twice."""

    def explode(path: str) -> None:
        raise RuntimeError("index is locked")

    context.reindex = explode
    mark_read(context, workspace, "a.py")

    _, result = run_edit(context, old="    return total", new="    return 0")

    assert result.ok
    assert (workspace / "a.py").read_text(encoding="utf-8").count("return 0") == 1


# ---------------------------------------------- a new file's preview names its path


def test_creating_a_file_shows_where_it_will_be_created(context: ToolContext) -> None:
    """The approval preview for a new file used to be only the numbered content.

    An edit's diff names its file in the header; a new file's preview did not, so the
    person approving saw *what* would be written and was never told *where*. The
    destination is half of "you saw exactly what will happen".
    """
    tool = WriteFileTool()
    args = tool.args_model.model_validate({"path": "pkg/new_module.py", "content": "x = 1\n"})

    prepared = tool.prepare(args, context)

    assert "pkg/new_module.py" in prepared.preview
    assert prepared.preview.startswith("--- /dev/null\n+++ b/pkg/new_module.py")
    assert "x = 1" in prepared.preview


def test_the_new_file_preview_reports_its_size(context: ToolContext) -> None:
    tool = WriteFileTool()
    body = "".join(f"line {n}\n" for n in range(100))
    args = tool.args_model.model_validate({"path": "big.txt", "content": body})

    prepared = tool.prepare(args, context)

    assert "100 line(s)" in prepared.preview
    assert "40 more line(s)" in prepared.preview
