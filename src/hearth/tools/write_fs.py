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
            unified_diff(before_text, new_text, path=relative) if existing else _new_file_preview(new_text)
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


# ------------------------------------------------------------------ internals


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


def _new_file_preview(text: str, *, limit: int = 60) -> str:
    lines = text.splitlines()
    shown = "\n".join(f"{number:>5}| {line}" for number, line in enumerate(lines[:limit], start=1))
    if len(lines) > limit:
        shown += f"\n      … {len(lines) - limit} more line(s)"
    return shown


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
