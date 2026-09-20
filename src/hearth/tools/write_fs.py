"""The first tools that can change a file: ``edit_file`` and ``write_file``.

Everything else in Hearth exists so that these two can be trusted. The write protocol is
docs/safety-and-tool-use.md §7.2, and four of its steps are load-bearing in ways that are
easy to shave off by accident:

**Read before write.** An edit to a file the model has not read in this session is
refused. Without it, a model that guessed a file's contents from its name could land an
edit that happens to match — and the user would approve a diff computed against text
nobody ever read.

**The hash is checked twice: in ``prepare()`` and again in ``execute()``.** The gap
between them is an approval prompt, which a person may take a minute over. If the file
moves in that window, the diff they approved describes a file that no longer exists, so
the write is abandoned rather than applied to whatever is there now.

**The checkpoint is taken inside ``execute()``, after the re-verification and immediately
before the mutation.** Snapshotting in ``prepare()`` would store the bytes that were there
when the preview was computed, which are not necessarily the bytes being replaced.

**The reindex is synchronous.** The model's next step is often a `find_symbol` over code it
just changed, and an index that lags one step behind a write teaches the model that its own
edits do not take effect.

Every path goes through ``safety.paths.resolve_in_workspace(for_write=True)``. These tools
never call ``open()`` on a model-supplied string, and a CI grep guard enforces it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hearth.indexing.languages import detect_language
from hearth.safety.policy import PolicyFacts
from hearth.safety.secrets import find_secrets
from hearth.tools.base import Prepared, Risk, Tool, ToolContext
from hearth.tools.edit_engine import (
    MAX_WRITE_BYTES,
    FileForm,
    apply_edit,
    diff_stats,
    encode_with_form,
    read_form,
    unified_diff,
)
from hearth.tools.results import ErrorCode, ToolResult
from hearth.util.hashing import content_hash

#: Discarding more existing lines than this makes an overwrite DESTRUCTIVE (§6.2), which
#: means it asks with a typed confirmation even at `auto-edit`. Matches the threshold §2.2
#: uses for `delete_file`: at this size you are replacing a file, not editing it.
DESTRUCTIVE_OVERWRITE_LINES = 200

#: Edits one ``multi_edit`` call may carry. Beyond this the model has stopped editing a
#: file and started rewriting it, and one diff of that size is not reviewable — which
#: defeats the point of putting it behind a single approval.
MAX_EDITS_PER_CALL = 20


# ----------------------------------------------------------------- edit_file


class EditFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Workspace-relative path of an existing file you have read")
    old_string: str = Field(
        min_length=1,
        description="Exact text to replace, including enough context to be unique",
    )
    new_string: str = Field(description="Replacement text")
    replace_all: bool = Field(default=False, description="Replace every occurrence")


class EditFileTool(Tool[EditFileArgs]):
    name = "edit_file"
    description = (
        "Replace text in a file. Read the file first. old_string must match exactly once "
        "unless replace_all is true."
    )
    risk = Risk.WRITE
    args_model = EditFileArgs

    def prepare(self, args: EditFileArgs, context: ToolContext) -> Prepared:
        """Resolve, verify freshness, compute the edit and its diff. Mutates nothing."""
        # PathError is deliberately *not* caught. The gateway audits it as a denial by
        # the hard invariants, which is what a jail refusal is; turning it into a
        # `Prepared` error here would record it as "the tool could not run" instead.
        target = _resolve(context, args.path)
        relative = target.relative
        summary = f"edit {relative}"

        if not target.path.is_file():
            return _refused(
                relative,
                ErrorCode.NOT_FOUND,
                f"{relative} does not exist. Use write_file to create it.",
            )

        try:
            raw = target.path.read_bytes()
        except OSError as exc:
            return _refused(relative, ErrorCode.NOT_FOUND, f"could not read {relative}: {exc}")

        form = read_form(raw)
        if not form.usable:
            return _refused(relative, ErrorCode.INVALID_ARGUMENTS, f"{relative}: {form.reason}")

        stale = _staleness(context, relative, raw)
        if stale is not None:
            return _refused(relative, ErrorCode.STALE_FILE, stale)

        text = form.decode(raw)
        outcome = apply_edit(
            text,
            args.old_string,
            args.new_string,
            replace_all=args.replace_all,
            language=detect_language(relative),
        )
        if not outcome.ok or outcome.new_text is None:
            message = outcome.error or "the edit could not be applied"
            if outcome.hint:
                message = f"{message}\n\n{outcome.hint}"
            return _refused(relative, ErrorCode.NOT_FOUND, message)

        added, removed = diff_stats(text, outcome.new_text)
        badges = [*outcome.badges, *_secret_badges(text, outcome.new_text)]

        return Prepared(
            summary=f"{summary}  +{added} -{removed}",
            preview=unified_diff(text, outcome.new_text, path=relative),
            badges=badges,
            payload={
                "resolved": target.path,
                "relative": relative,
                "new_text": outcome.new_text,
                "form": form,
                "before_hash": content_hash(raw),
                "added": added,
                "removed": removed,
                "strategy": outcome.strategy,
                "replacements": outcome.replacements,
            },
            facts=PolicyFacts(
                path=relative,
                absolute_path=target.path.as_posix(),
                inside_workspace=True,
                badges=tuple(badges),
                grant_key=f"edit:{relative}",
            ),
        )

    def execute(self, args: EditFileArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        return _commit(context, prepared, verb="Edited")


# ---------------------------------------------------------------- multi_edit


class EditSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    old_string: str = Field(min_length=1, description="Exact text to replace")
    new_string: str = Field(description="Replacement text")
    replace_all: bool = Field(default=False, description="Replace every occurrence")


class MultiEditArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Workspace-relative path of an existing file you have read")
    edits: list[EditSpec] = Field(
        min_length=1,
        max_length=MAX_EDITS_PER_CALL,
        description="Edits to apply in order, as one atomic change",
    )


class MultiEditTool(Tool[MultiEditArgs]):
    """Several replacements in one file, as one approval and one diff (§2.2).

    **All or nothing.** The edits are applied in order to an in-memory copy, and if any of
    them fails the whole call fails with nothing written. That is not a convenience: a
    partial application would leave the file in a state neither the model nor the user
    asked for, halfway between two designs, and the model's next step would be an edit
    computed against the shape it *expected*.

    Applying in order also means a later edit sees the earlier ones. This is what makes
    the tool worth having over repeated ``edit_file`` calls — a rename that touches six
    call sites is one diff to read and one checkpoint to undo, rather than six of each —
    but it has a sharp edge: an edit whose ``old_string`` an earlier edit has already
    rewritten will not match. The error says which index failed and what the file looked
    like by then, because "edit 4 of 6 did not match" is otherwise indistinguishable from
    "edit 4 was wrong".
    """

    name = "multi_edit"
    description = (
        "Apply several replacements to one file as a single change. Read the file first. "
        "Edits apply in order; if any fails, none are applied."
    )
    risk = Risk.WRITE
    args_model = MultiEditArgs

    def prepare(self, args: MultiEditArgs, context: ToolContext) -> Prepared:
        target = _resolve(context, args.path)  # PathError propagates; see edit_file.
        relative = target.relative

        if not target.path.is_file():
            return _refused(
                relative,
                ErrorCode.NOT_FOUND,
                f"{relative} does not exist. Use write_file to create it.",
            )

        try:
            raw = target.path.read_bytes()
        except OSError as exc:
            return _refused(relative, ErrorCode.NOT_FOUND, f"could not read {relative}: {exc}")

        form = read_form(raw)
        if not form.usable:
            return _refused(relative, ErrorCode.INVALID_ARGUMENTS, f"{relative}: {form.reason}")

        stale = _staleness(context, relative, raw)
        if stale is not None:
            return _refused(relative, ErrorCode.STALE_FILE, stale)

        original = form.decode(raw)
        language = detect_language(relative)
        text = original
        badges: list[str] = []
        replacements = 0

        for index, edit in enumerate(args.edits, start=1):
            outcome = apply_edit(
                text,
                edit.old_string,
                edit.new_string,
                replace_all=edit.replace_all,
                language=language,
            )
            if not outcome.ok or outcome.new_text is None:
                return _refused(
                    relative,
                    ErrorCode.NOT_FOUND,
                    _multi_edit_failure(index, len(args.edits), outcome, applied=index - 1),
                )
            text = outcome.new_text
            replacements += outcome.replacements
            badges.extend(badge for badge in outcome.badges if badge not in badges)

        if text == original:
            return _refused(
                relative,
                ErrorCode.INVALID_ARGUMENTS,
                f"every edit matched but left {relative} unchanged, so there is nothing to write.",
            )

        added, removed = diff_stats(original, text)
        badges.extend(badge for badge in _secret_badges(original, text) if badge not in badges)

        return Prepared(
            summary=f"edit {relative} ({len(args.edits)} edits)  +{added} -{removed}",
            preview=unified_diff(original, text, path=relative),
            badges=badges,
            payload={
                "resolved": target.path,
                "relative": relative,
                "new_text": text,
                "form": form,
                "before_hash": content_hash(raw),
                "added": added,
                "removed": removed,
                "replacements": replacements,
            },
            facts=PolicyFacts(
                path=relative,
                absolute_path=target.path.as_posix(),
                inside_workspace=True,
                badges=tuple(badges),
                # The same key `edit_file` uses. A grant is permission to change *this
                # file*, and which tool does the changing is not something the user was
                # asked about — offering a second key would mean granting one tool and
                # being asked again by the other for the identical write.
                grant_key=f"edit:{relative}",
            ),
        )

    def execute(self, args: MultiEditArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        return _commit(context, prepared, verb="Edited")


# ---------------------------------------------------------------- write_file


class WriteFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Workspace-relative path to write")
    content: str = Field(description="The file's full new contents")


class WriteFileTool(Tool[WriteFileArgs]):
    name = "write_file"
    description = (
        "Create a file, or replace an existing file's entire contents. Read the file first "
        "if it already exists. Prefer edit_file for changes to part of a file."
    )
    risk = Risk.WRITE
    args_model = WriteFileArgs

    def prepare(self, args: WriteFileArgs, context: ToolContext) -> Prepared:
        target = _resolve(context, args.path)  # PathError propagates; see edit_file.
        relative = target.relative

        if len(args.content.encode("utf-8")) > MAX_WRITE_BYTES:
            return _refused(
                relative,
                ErrorCode.TOO_LARGE,
                f"content is larger than the {MAX_WRITE_BYTES // (1024 * 1024)} MB write limit.",
            )
        if target.path.is_dir():
            return _refused(relative, ErrorCode.INVALID_ARGUMENTS, f"{relative} is a directory.")

        existing = target.path.is_file()
        form = FileForm()
        before_text = ""
        before_hash: str | None = None

        if existing:
            try:
                raw = target.path.read_bytes()
            except OSError as exc:
                return _refused(relative, ErrorCode.NOT_FOUND, f"could not read {relative}: {exc}")

            form = read_form(raw)
            if not form.usable:
                return _refused(relative, ErrorCode.INVALID_ARGUMENTS, f"{relative}: {form.reason}")

            # Overwriting is the most destructive thing this tool does, so it is held to
            # the same read-before-write rule as an edit. Creating a file is not: there is
            # nothing to have read, and requiring it would make creation impossible.
            stale = _staleness(context, relative, raw)
            if stale is not None:
                return _refused(relative, ErrorCode.STALE_FILE, stale)

            before_text = form.decode(raw)
            before_hash = content_hash(raw)

        new_text = args.content
        if new_text == before_text:
            return _refused(
                relative,
                ErrorCode.INVALID_ARGUMENTS,
                f"{relative} already has exactly this content, so there is nothing to write.",
            )

        added, removed = diff_stats(before_text, new_text)
        badges = list(_secret_badges(before_text, new_text))
        destructive = removed > DESTRUCTIVE_OVERWRITE_LINES
        if destructive:
            badges.append("DESTRUCTIVE")

        verb = "overwrite" if existing else "create"
        preview = (
            unified_diff(before_text, new_text, path=relative)
            if existing
            else _new_file_preview(new_text, path=relative)
        )

        return Prepared(
            summary=f"{verb} {relative}  +{added} -{removed}",
            preview=preview,
            badges=badges,
            payload={
                "resolved": target.path,
                "relative": relative,
                "new_text": new_text,
                "form": form,
                "before_hash": before_hash,
                "added": added,
                "removed": removed,
            },
            facts=PolicyFacts(
                path=relative,
                absolute_path=target.path.as_posix(),
                inside_workspace=True,
                destructive=destructive,
                badges=tuple(badges),
                grant_key=f"edit:{relative}",
            ),
        )

    def execute(self, args: WriteFileArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        return _commit(context, prepared, verb="Wrote")


# ----------------------------------------------------------------- move_file


class MoveFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: str = Field(description="Workspace-relative path to move")
    dst: str = Field(description="Workspace-relative destination path")


class MoveFileTool(Tool[MoveFileArgs]):
    """Rename or move a file within the workspace (§2.2).

    Read-before-write does **not** apply here, and that is deliberate rather than an
    oversight. The rule exists so a diff is computed against bytes someone has seen; a move
    changes no bytes, so there is nothing to have read and requiring it would only make the
    model read a file in order to rename it.

    What the preview carries instead is a **reference count**: how many other files mention
    the old path. Moving a module is rarely the whole change, and "14 files mention
    src/billing/models.py" is the fact that tells a reviewer this rename is either the
    start of a larger edit or a mistake. It is a hint, not a check — Hearth does not
    rewrite imports, and pretending otherwise by refusing the move would be worse.

    Both ends are checkpointed, so undo puts the file back where it was rather than leaving
    a copy at each path.
    """

    name = "move_file"
    description = "Move or rename a file within the workspace. Does not update references to it."
    risk = Risk.WRITE
    args_model = MoveFileArgs

    def prepare(self, args: MoveFileArgs, context: ToolContext) -> Prepared:
        # Both ends resolve for write. The destination obviously must; so must the source,
        # because a move deletes it — resolving it read-only would let a file be moved out
        # of `.git/` by a call the jail had no reason to refuse.
        source = _resolve(context, args.src)
        destination = _resolve(context, args.dst)

        if not source.path.is_file():
            return _refused(
                source.relative, ErrorCode.NOT_FOUND, f"{source.relative} does not exist."
            )
        if source.path == destination.path:
            return _refused(
                source.relative,
                ErrorCode.INVALID_ARGUMENTS,
                f"{source.relative} and {destination.relative} are the same path.",
            )
        if destination.path.exists():
            return _refused(
                destination.relative,
                ErrorCode.INVALID_ARGUMENTS,
                f"{destination.relative} already exists. Delete it first if that is intended.",
            )

        try:
            raw = source.path.read_bytes()
        except OSError as exc:
            return _refused(
                source.relative, ErrorCode.NOT_FOUND, f"could not read {source.relative}: {exc}"
            )

        references = _reference_count(context, source.relative)
        preview = [f"{source.relative}  →  {destination.relative}"]
        if references:
            preview.append(
                f"\n{references} other file(s) mention {source.relative}. "
                "Moving it does not update them."
            )

        return Prepared(
            summary=f"move {source.relative} → {destination.relative}",
            preview="\n".join(preview),
            payload={
                "source": source.path,
                "destination": destination.path,
                "src_relative": source.relative,
                "dst_relative": destination.relative,
                "before_hash": content_hash(raw),
                "bytes": raw,
            },
            facts=PolicyFacts(
                # The destination is the path policy judges: it is what comes into
                # existence, and it is the one a `path = "src/**"` rule should be read as
                # being about. The source is checked by the jail either way.
                path=destination.relative,
                absolute_path=destination.path.as_posix(),
                inside_workspace=True,
                grant_key=f"edit:{destination.relative}",
            ),
        )

    def execute(self, args: MoveFileArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        source: Path = prepared.payload["source"]
        destination: Path = prepared.payload["destination"]
        src_relative: str = prepared.payload["src_relative"]
        dst_relative: str = prepared.payload["dst_relative"]
        expected: str = prepared.payload["before_hash"]

        if context.checkpoints is None:
            return ToolResult.failure(
                ErrorCode.EXECUTION_FAILED,
                "no checkpoint store is available, so this move cannot be made revertible.",
            )

        # --- re-verify (TOCTOU) ---------------------------------------------
        try:
            raw = source.read_bytes()
        except OSError:
            return ToolResult.failure(
                ErrorCode.STALE_FILE,
                f"{src_relative} is gone or unreadable, so it was not moved.",
            )
        if content_hash(raw) != expected:
            return ToolResult.failure(
                ErrorCode.STALE_FILE,
                f"{src_relative} changed while this move was awaiting approval, so it was "
                "not moved.",
            )
        if destination.exists():
            return ToolResult.failure(
                ErrorCode.STALE_FILE,
                f"{dst_relative} appeared while this move was awaiting approval, so it was "
                "not moved.",
            )

        # Both ends, before the mutation. One entry saying the source is gone and one
        # saying the destination arrived; undo replays them together and the file ends up
        # where it started, with no copy left behind.
        context.checkpoints.snapshot(step=context.step, path=src_relative, before=raw, after=None)
        context.checkpoints.snapshot(step=context.step, path=dst_relative, before=None, after=raw)

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
        except OSError as exc:
            return ToolResult.failure(
                ErrorCode.EXECUTION_FAILED, f"could not move {src_relative}: {exc}"
            )

        # The hash follows the file. Without this, a model that moves a file it has read
        # and then edits it at the new path is told to read it again — for a rename it
        # performed itself.
        context.record_read(dst_relative, expected)
        _reindex(context, src_relative)
        _reindex(context, dst_relative)

        return ToolResult.success(
            f"Moved {src_relative} to {dst_relative}.",
            display=prepared.preview,
            path=dst_relative,
        )


# --------------------------------------------------------------- delete_file


class DeleteFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Workspace-relative path to delete")


class DeleteFileTool(Tool[DeleteFileArgs]):
    """Delete a file by moving it to Hearth's trash (§2.2).

    **Nothing is unlinked.** The file goes to ``.hearth/<project>/trash/`` and a checkpoint
    records it, so a delete is revertible twice over — by `/undo`, and by hand from the
    trash directory afterwards. An agent that can permanently remove a file is an agent
    whose worst mistake is unbounded, and the cost of avoiding that is a directory the user
    can empty whenever they like.

    The preview carries the file's size and its git status, because those decide how bad a
    wrong delete would be. A tracked file with no uncommitted changes is recoverable from
    git whatever Hearth does; an untracked file exists only here. The second case gets the
    ``DESTRUCTIVE`` badge — a typed confirmation (§6.3) — as does any file large enough
    that losing it would be a real loss.
    """

    name = "delete_file"
    description = "Delete a file. It is moved to Hearth's trash, not removed permanently."
    risk = Risk.WRITE
    args_model = DeleteFileArgs

    def prepare(self, args: DeleteFileArgs, context: ToolContext) -> Prepared:
        target = _resolve(context, args.path)  # PathError propagates; see edit_file.
        relative = target.relative

        if target.path.is_dir():
            return _refused(
                relative,
                ErrorCode.INVALID_ARGUMENTS,
                f"{relative} is a directory. This tool deletes one file at a time.",
            )
        if not target.path.is_file():
            return _refused(relative, ErrorCode.NOT_FOUND, f"{relative} does not exist.")

        try:
            raw = target.path.read_bytes()
        except OSError as exc:
            return _refused(relative, ErrorCode.NOT_FOUND, f"could not read {relative}: {exc}")

        lines = raw.count(b"\n") + (0 if raw.endswith(b"\n") or not raw else 1)
        status = _git_status_for(context.workspace, relative)

        # Recoverable from git means a wrong delete costs a `git checkout`. Unrecoverable
        # means it costs the file, and the trash is the only copy — that is the case worth
        # a typed confirmation, along with anything big enough to be a real loss.
        recoverable = status == "tracked, committed"
        destructive = not recoverable or lines > DESTRUCTIVE_OVERWRITE_LINES
        badges = ["DESTRUCTIVE"] if destructive else []

        preview = "\n".join(
            [
                f"delete {relative}",
                f"  {lines} line(s), {len(raw)} byte(s)",
                f"  git: {status}",
                "  moved to Hearth's trash; /undo restores it",
            ]
        )

        return Prepared(
            summary=f"delete {relative} ({lines} lines)",
            preview=preview,
            badges=badges,
            payload={
                "resolved": target.path,
                "relative": relative,
                "before_hash": content_hash(raw),
                "lines": lines,
            },
            facts=PolicyFacts(
                path=relative,
                absolute_path=target.path.as_posix(),
                inside_workspace=True,
                destructive=destructive,
                badges=tuple(badges),
                # Deliberately ungrantable. "Always delete files matching this" is not a
                # permission anybody means to give, and the policy engine refuses to offer
                # a grant for a DESTRUCTIVE call anyway (§6.1) — stating None here means
                # the non-destructive case does not quietly become grantable either.
                grant_key=None,
            ),
        )

    def execute(self, args: DeleteFileArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        resolved: Path = prepared.payload["resolved"]
        relative: str = prepared.payload["relative"]
        expected: str = prepared.payload["before_hash"]

        if context.checkpoints is None:
            return ToolResult.failure(
                ErrorCode.EXECUTION_FAILED,
                "no checkpoint store is available, so this delete cannot be made revertible.",
            )

        try:
            raw = resolved.read_bytes()
        except OSError:
            return ToolResult.failure(
                ErrorCode.STALE_FILE, f"{relative} is gone or unreadable, so it was not deleted."
            )
        if content_hash(raw) != expected:
            return ToolResult.failure(
                ErrorCode.STALE_FILE,
                f"{relative} changed while this delete was awaiting approval, so it was not "
                "deleted. Read it again before deciding.",
            )

        context.checkpoints.snapshot(step=context.step, path=relative, before=raw, after=None)

        try:
            destination = _trash(context.workspace, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            resolved.replace(destination)
        except OSError:
            # `os.replace` fails across filesystems, which the trash can be on — the
            # project data directory is not necessarily on the same mount as the
            # workspace. Copy-then-unlink, in that order, so a failure mid-way leaves the
            # file present rather than gone.
            try:
                destination.write_bytes(raw)
                resolved.unlink()
            except OSError as exc:
                return ToolResult.failure(
                    ErrorCode.EXECUTION_FAILED, f"could not delete {relative}: {exc}"
                )

        context.read_hashes.pop(relative, None)
        _reindex(context, relative)

        return ToolResult.success(
            f"Deleted {relative} ({prepared.payload['lines']} lines). "
            "It is in Hearth's trash; /undo restores it.",
            display=prepared.preview,
            path=relative,
        )


# ------------------------------------------------------------------ internals


def _multi_edit_failure(index: int, total: int, outcome: object, *, applied: int) -> str:
    """Why one edit in a batch did not apply, and what state the file was in by then.

    The second half is what makes this actionable. Edits compose, so edit 4 runs against
    the file as edits 1-3 left it — and a model told only "edit 4 did not match" will
    re-send edit 4 verbatim, because against the file it can see, edit 4 is correct.
    """
    error = getattr(outcome, "error", None) or "the edit could not be applied"
    hint = getattr(outcome, "hint", None)

    parts = [f"edit {index} of {total} did not apply: {error}"]
    if applied:
        parts.append(
            f"Edits 1-{applied} were applied to a working copy first, so edit {index} ran "
            "against the file as they left it — not against the file you read. Nothing was "
            "written."
        )
    else:
        parts.append("Nothing was written.")
    if hint:
        parts.append(str(hint))
    return "\n\n".join(parts)


def _reference_count(context: ToolContext, relative: str) -> int:
    """How many *other* indexed files mention this path.

    A hint for the preview, never a check. It uses the index rather than reading the tree
    because this runs inside ``prepare()``, which must be cheap and must not touch the
    filesystem more than it has to — and an unindexed workspace returning 0 is the right
    answer to give when there is nothing to count from.
    """
    if context.index_connection is None:
        return 0

    stem = relative.rsplit("/", 1)[-1].removesuffix(".py")
    if not stem:
        return 0

    try:
        rows = context.index_connection.execute(
            "SELECT COUNT(DISTINCT f.path) FROM chunks c JOIN files f ON f.id = c.file_id "
            "WHERE c.content LIKE ? AND f.path <> ?",
            (f"%{stem}%", relative),
        ).fetchone()
    except Exception:
        # The count is decoration. A schema that has moved on should not fail a move.
        return 0
    return int(rows[0]) if rows else 0


def _git_status_for(workspace: Path, relative: str) -> str:
    """One phrase describing what git knows about this file.

    ``tracked, committed`` is the only value that means a wrong delete is free, so it is
    the only one ``delete_file`` treats as recoverable. Everything else — untracked,
    modified, ignored, or no repository at all — leaves Hearth's trash as the only copy.
    """
    from hearth.git.runner import GitError, is_git_repository, run_git

    if not is_git_repository(workspace):
        return "not a git repository"

    try:
        tracked = run_git(workspace, ["ls-files", "--error-unmatch", "--", relative])
        if not tracked.ok:
            return "untracked — Hearth's trash will be the only copy"

        status = run_git(workspace, ["status", "--porcelain", "--", relative])
    except GitError:
        return "unknown (git could not be run)"

    return "tracked, committed" if not status.text().strip() else "tracked, with uncommitted changes"


def _trash(workspace: Path, relative: str) -> Path:
    """Where a deleted file goes, keeping its workspace-relative shape.

    Timestamped, so deleting and recreating the same path twice does not have the second
    delete overwrite the first file in the trash — which would quietly destroy the copy
    that exists precisely so nothing is quietly destroyed.
    """
    import time

    from hearth.config import paths

    stamp = time.strftime("%Y%m%d-%H%M%S")
    return paths.trash_dir(workspace) / stamp / relative


class _Target:
    __slots__ = ("path", "relative")

    def __init__(self, path: Path, relative: str) -> None:
        self.path = path
        self.relative = relative


def _resolve(context: ToolContext, user_path: str) -> _Target:
    """Resolve for writing, or raise. The jail decides, not this module."""
    from hearth.safety.paths import relative_to_workspace, resolve_in_workspace

    resolved = resolve_in_workspace(context.workspace, user_path, for_write=True)
    return _Target(resolved, relative_to_workspace(resolved, context.workspace))


def _staleness(context: ToolContext, relative: str, raw: bytes) -> str | None:
    """Whether the model may write this file, given what it has read.

    Returns the message for the model, or None when the write may proceed. One message for
    both cases — never read, and read-then-changed — because the model's corrective action
    is identical: read the file again.
    """
    known = context.hash_at_last_read(relative)
    if known is None:
        return (
            f"{relative} was never read in this session. Call read_file on it first, so the "
            "edit is based on its actual contents."
        )
    if known != content_hash(raw):
        return (
            f"{relative} changed since you read it. Call read_file again and redo the edit "
            "against the current contents."
        )
    return None


def _secret_badges(before: str, after: str) -> tuple[str, ...]:
    """`SECRET?` when the edit *adds* a credential-looking string.

    Scoped to added lines on purpose. Badging an edit because the file already contained
    something secret-shaped would fire on every change to a test fixture, and a badge that
    is always on is a badge nobody reads (T2).
    """
    existing = set(before.splitlines())
    added = [line for line in after.splitlines() if line not in existing]
    return ("SECRET?",) if any(find_secrets(line) for line in added) else ()


def _new_file_preview(text: str, *, path: str, limit: int = 60) -> str:
    """The approval preview for a file that does not exist yet.

    **Headed with the path**, in the same ``---``/``+++`` shape a diff of an edit has. It did
    not use to be, and the omission mattered: an edit's diff names its file in the header,
    but a new file's preview was only the numbered content, so the person approving was
    shown *what* would be written without being told *where*. For a tool whose whole
    approval is "you saw exactly what will happen", the destination is half of it.
    """
    lines = text.splitlines()
    shown = "\n".join(f"{number:>5}| {line}" for number, line in enumerate(lines[:limit], start=1))
    if len(lines) > limit:
        shown += f"\n      … {len(lines) - limit} more line(s)"
    return f"--- /dev/null\n+++ b/{path}  (new file, {len(lines)} line(s))\n{shown}"


def _commit(context: ToolContext, prepared: Prepared, *, verb: str) -> ToolResult:
    """Re-verify, checkpoint, write atomically, then reindex.

    The order is the contract. Re-verify before anything is touched; checkpoint before the
    mutation; reindex only after the bytes are safely on disk.
    """
    resolved: Path = prepared.payload["resolved"]
    relative: str = prepared.payload["relative"]
    new_text: str = prepared.payload["new_text"]
    form: FileForm = prepared.payload["form"]
    expected: str | None = prepared.payload["before_hash"]

    if context.checkpoints is None:
        # Not a degraded write — a different and worse operation. Refuse rather than
        # perform an irreversible edit the user believes is revertible.
        return ToolResult.failure(
            ErrorCode.EXECUTION_FAILED,
            "no checkpoint store is available, so this write cannot be made revertible.",
        )

    # --- re-verify (TOCTOU) ---------------------------------------------
    before: bytes | None
    try:
        before = resolved.read_bytes()
    except FileNotFoundError:
        before = None
    except OSError as exc:
        return ToolResult.failure(ErrorCode.EXECUTION_FAILED, f"could not read {relative}: {exc}")

    actual = None if before is None else content_hash(before)
    if actual != expected:
        return ToolResult.failure(
            ErrorCode.STALE_FILE,
            f"{relative} changed while this edit was awaiting approval, so it was not "
            "applied. Read the file again and redo the edit.",
        )

    payload = encode_with_form(new_text, form)

    # --- checkpoint, then write -----------------------------------------
    context.checkpoints.snapshot(step=context.step, path=relative, before=before, after=payload)

    try:
        _atomic_write(resolved, payload)
    except OSError as exc:
        return ToolResult.failure(ErrorCode.EXECUTION_FAILED, f"could not write {relative}: {exc}")

    # The registry has to move forward, or the model's next edit to this file is refused
    # as stale against the content it just wrote itself.
    context.record_read(relative, content_hash(payload))
    _reindex(context, relative)

    added = prepared.payload.get("added", 0)
    removed = prepared.payload.get("removed", 0)
    return ToolResult.success(
        f"{verb} {relative} (+{added} -{removed}).",
        display=prepared.preview,
        path=relative,
        added=added,
        removed=removed,
    )


def _reindex(context: ToolContext, relative: str) -> None:
    """Refresh the index for one file, never failing the write.

    The bytes are already on disk by this point. Reporting failure would tell the model to
    retry an edit that has in fact been applied, which is how a file gets edited twice.
    """
    if context.reindex is None:
        return
    try:
        context.reindex(relative)
    except Exception:
        return


def _atomic_write(target: Path, data: bytes) -> None:
    """Write via a temp file in the same directory, then rename.

    Same-directory matters: ``os.replace`` is only atomic within a filesystem, and a temp
    file elsewhere could land on a different one. Mode bits are carried over so editing a
    script does not make it non-executable.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = target.stat().st_mode if target.exists() else None

    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".hearth-tmp-")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            Path(temporary).chmod(mode)
        Path(temporary).replace(target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _refused(relative: str, code: ErrorCode, message: str) -> Prepared:
    """A prepare that already knows the call cannot succeed."""
    return Prepared(
        summary=f"refused {relative}",
        error=ToolResult.failure(code, message),
        facts=PolicyFacts(path=relative),
    )
