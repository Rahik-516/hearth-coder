"""Checkpoints: making every file write Hearth performs reversible.

The promise is narrow and worth stating precisely (docs/safety-and-tool-use.md §7.3):
**file writes made by Hearth's own tools can be undone, exactly.** Side effects of
`run_command`, `run_tests` and git operations cannot, and the approval UI says so rather
than implying otherwise.

"Exactly" is the demanding word. A revert that restored a file's *content* while
normalising its BOM or line endings would not have undone anything — it would have made a
second, silent edit at the moment the user asked to be put back. So checkpoints store raw
bytes in the blob store, never decoded text.

The other load-bearing behaviour is **conflict detection**. Before restoring, the file's
current bytes are compared against the ``after`` blob — what Hearth believes it last
wrote. If they differ, someone edited the file since, and reverting would throw their work
away. Hearth reports the conflict and writes nothing unless told to.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from hearth.storage.blobs import BlobStore
from hearth.storage.state_repo import StateRepository
from hearth.util.hashing import blob_hash


@dataclass(frozen=True)
class CheckpointEntry:
    """One file's before/after within one write step."""

    id: int
    step: int
    path: str
    #: None means the file did not exist before the write.
    before_blob: str | None
    after_blob: str | None
    created_at: int
    reverted: bool

    @property
    def was_created(self) -> bool:
        return self.before_blob is None


@dataclass(frozen=True)
class CheckpointStep:
    """One write step, as `/checkpoints` lists it."""

    step: int
    paths: list[str]
    created_at: int
    reverted: bool


@dataclass(frozen=True)
class Conflict:
    """A file that changed after Hearth wrote it."""

    path: str
    #: What Hearth last wrote, and what it expected to still find.
    expected: str | None
    #: What is actually on disk now. None when the file has been deleted.
    actual: str | None

    @property
    def deleted(self) -> bool:
        return self.actual is None


@dataclass
class RevertReport:
    """What a revert did, and what it refused to do."""

    restored: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    already_reverted: bool = False

    @property
    def touched(self) -> int:
        return len(self.restored) + len(self.removed)

    @property
    def clean(self) -> bool:
        return not self.conflicts


class CheckpointStore:
    """Records and reverses Hearth's file writes."""

    def __init__(self, repo: StateRepository, blobs: BlobStore) -> None:
        self.repo = repo
        self.blobs = blobs

    def bind(self, session_id: str) -> BoundCheckpointer:
        """A writer scoped to one session, for a tool to hold.

        Tools get this rather than the store itself: a tool has no business listing or
        reverting checkpoints, and cannot be handed the ability to write one against a
        different session.
        """
        return BoundCheckpointer(self, session_id)

    # ------------------------------------------------------------- recording

    def record(
        self,
        *,
        session_id: str,
        step: int,
        path: str,
        before: bytes | None,
        after: bytes | None,
    ) -> int:
        """Snapshot one file around a write.

        Called from inside ``execute()``, after the TOCTOU re-verification and immediately
        before the mutation — so the ``before`` blob is the bytes that were actually
        replaced, not the bytes that were there when the preview was computed.

        Args:
            before: The file's bytes before the write, or None if it did not exist.
            after: The bytes written, or None if the file was removed.
        """
        return self.repo.add_checkpoint(
            session_id=session_id,
            step=step,
            path=path,
            before_blob=None if before is None else self.blobs.put(before),
            after_blob=None if after is None else self.blobs.put(after),
        )

    # -------------------------------------------------------------- listing

    def entries_for_step(self, session_id: str, step: int) -> list[CheckpointEntry]:
        return [_entry(row) for row in self.repo.checkpoints_for_step(session_id, step)]

    def steps(self, session_id: str) -> list[CheckpointStep]:
        """Write steps, newest first, each with the files it touched."""
        grouped: dict[int, list[CheckpointEntry]] = {}
        for row in self.repo.checkpoint_rows(session_id):
            entry = _entry(row)
            grouped.setdefault(entry.step, []).append(entry)

        return [
            CheckpointStep(
                step=number,
                paths=[entry.path for entry in entries],
                created_at=min(entry.created_at for entry in entries),
                reverted=all(entry.reverted for entry in entries),
            )
            for number, entries in sorted(grouped.items(), reverse=True)
        ]

    def latest_step(self, session_id: str) -> int | None:
        """The most recent step that has not been reverted — what bare `/undo` acts on."""
        for step in self.steps(session_id):
            if not step.reverted:
                return step.step
        return None

    # --------------------------------------------------------------- undo

    def revert_step(
        self,
        session_id: str,
        step: int,
        *,
        workspace: Path,
        force: bool = False,
        trash: Path | None = None,
    ) -> RevertReport:
        """Put every file in one step back the way it was.

        Nothing is written until every file has been checked, so a step that conflicts on
        its second file does not leave the first one reverted. A partial undo is worse
        than a refused one: the user cannot tell which half happened.
        """
        entries = self.entries_for_step(session_id, step)
        if not entries:
            return RevertReport()
        if all(entry.reverted for entry in entries):
            return RevertReport(already_reverted=True)

        report = RevertReport()
        planned: list[tuple[CheckpointEntry, Path]] = []

        for entry in entries:
            target = workspace / entry.path
            conflict = self._conflict_for(entry, target)
            if conflict is not None and not force:
                report.conflicts.append(conflict)
                continue
            planned.append((entry, target))

        if report.conflicts and not force:
            # Report and stop. Reverting the non-conflicting files would leave the step
            # half-applied, which is a state nothing else in Hearth knows how to describe.
            return report

        for entry, target in planned:
            if entry.was_created:
                self._remove(target, trash=trash, relative=entry.path)
                report.removed.append(entry.path)
            else:
                _atomic_write(target, self.blobs.get(entry.before_blob or ""))
                report.restored.append(entry.path)

        self.repo.mark_step_reverted(session_id, step)
        return report

    def revert_after(
        self,
        session_id: str,
        step: int,
        *,
        workspace: Path,
        force: bool = False,
        trash: Path | None = None,
    ) -> RevertReport:
        """Revert every step *after* ``step``, newest first — `/rewind`.

        Newest first is not cosmetic. Two edits to one file only compose back correctly in
        reverse: replaying them oldest-first would restore v1 and then immediately be told
        that v2's checkpoint expected v2 on disk.
        """
        combined = RevertReport()
        for candidate in self.steps(session_id):
            if candidate.step <= step or candidate.reverted:
                continue
            report = self.revert_step(
                session_id, candidate.step, workspace=workspace, force=force, trash=trash
            )
            combined.restored.extend(report.restored)
            combined.removed.extend(report.removed)
            combined.conflicts.extend(report.conflicts)
        return combined

    # ----------------------------------------------------------- internals

    def _conflict_for(self, entry: CheckpointEntry, target: Path) -> Conflict | None:
        """Whether the file on disk still holds the bytes Hearth last wrote."""
        try:
            current = blob_hash(target.read_bytes())
        except FileNotFoundError:
            if entry.was_created:
                # Undo wanted to remove it and it is already gone. Nothing to reconcile.
                return None
            return Conflict(path=entry.path, expected=entry.after_blob, actual=None)
        except OSError:
            return Conflict(path=entry.path, expected=entry.after_blob, actual=None)

        if entry.after_blob is not None and current != entry.after_blob:
            return Conflict(path=entry.path, expected=entry.after_blob, actual=current)
        return None

    def _remove(self, target: Path, *, trash: Path | None, relative: str) -> None:
        """Move a created file aside, so undo is itself reversible (§7.3).

        Deleting outright would make `/undo` a one-way door: a user who undoes a file
        creation and then changes their mind has no way back.
        """
        if not target.exists():
            return
        if trash is None:
            target.unlink()
            return

        destination = trash / f"{int(time.time() * 1000)}-{relative.replace('/', '_')}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        target.replace(destination)


class BoundCheckpointer:
    """The write side of the checkpoint store, scoped to one session.

    Deliberately tiny. It is the whole of what a write tool is allowed to do with
    checkpoints: record a snapshot. Reading and reverting belong to the REPL commands and
    to `hearth undo`, which act on the user's instruction rather than the model's.
    """

    def __init__(self, store: CheckpointStore, session_id: str) -> None:
        self._store = store
        self.session_id = session_id

    def snapshot(self, *, step: int, path: str, before: bytes | None, after: bytes | None) -> int:
        return self._store.record(
            session_id=self.session_id, step=step, path=path, before=before, after=after
        )


def _entry(row: object) -> CheckpointEntry:
    mapping = dict(row)  # type: ignore[call-overload]
    return CheckpointEntry(
        id=int(mapping["id"]),
        step=int(mapping["step"]),
        path=str(mapping["path"]),
        before_blob=mapping["before_blob"],
        after_blob=mapping["after_blob"],
        created_at=int(mapping["created_at"]),
        reverted=bool(mapping["reverted"]),
    )


def _atomic_write(target: Path, data: bytes) -> None:
    """Replace a file's bytes without a window where it is truncated.

    Undo is the operation a user reaches for when something already went wrong; it must
    not be able to leave a half-written file behind.
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
