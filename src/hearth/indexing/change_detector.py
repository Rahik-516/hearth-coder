"""Deciding what actually needs reindexing.

Two tiers (docs/system-design.md §6.8):

1. **Stat fast path** — size and mtime_ns match what the index recorded, so the file is
   assumed unchanged and is never opened.
2. **Content hash** — for anything whose stat differs, blake2b-128 over the bytes. mtime
   changes for reasons content does not (``touch``, a checkout that rewrites an identical
   file, a clone), so hashing is what prevents a branch switch from reindexing the world.

The M1 acceptance criterion depends on this: re-running ``hearth index`` with no changes
must perform **zero parses**, and the counters here are how that is verified.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from hearth.indexing.scanner import ScannedFile
from hearth.util.hashing import file_hash


@dataclass
class ChangeSet:
    """What a scan found relative to the current index."""

    added: list[ScannedFile] = field(default_factory=list)
    modified: list[ScannedFile] = field(default_factory=list)
    unchanged: list[ScannedFile] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    #: Files whose stat differed but whose content hash matched. Counted separately
    #: because a high number here means something is touching files without changing
    #: them, which is worth knowing when indexing feels slower than it should.
    stat_changed_content_same: int = 0

    #: How many files had to be opened and hashed.
    hashed: int = 0

    @property
    def needs_indexing(self) -> list[ScannedFile]:
        return [*self.added, *self.modified]

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    def summary(self) -> str:
        return (
            f"{len(self.added)} added, {len(self.modified)} modified, "
            f"{len(self.deleted)} deleted, {len(self.unchanged)} unchanged"
        )


def detect_changes(
    scanned: Sequence[ScannedFile],
    indexed: dict[str, tuple[int, int, str]],
    *,
    force: bool = False,
) -> ChangeSet:
    """Compare a scan against the index.

    Args:
        scanned: Files found on disk.
        indexed: ``path -> (size_bytes, mtime_ns, content_hash)`` from the index.
        force: Treat everything as modified, for ``--rebuild``.
    """
    changes = ChangeSet()
    seen: set[str] = set()

    for file in scanned:
        seen.add(file.relative_path)
        previous = indexed.get(file.relative_path)

        if previous is None:
            changes.added.append(file)
            continue

        if force:
            changes.modified.append(file)
            continue

        previous_size, previous_mtime, previous_hash = previous

        # Fast path: nothing about the file's stat suggests a change, so don't open it.
        if file.size_bytes == previous_size and file.mtime_ns == previous_mtime:
            changes.unchanged.append(file)
            continue

        # Slow path: stat differs, but the content may not.
        try:
            current_hash = file_hash(file.absolute_path)
        except OSError:
            changes.modified.append(file)
            continue

        changes.hashed += 1
        if current_hash == previous_hash:
            changes.unchanged.append(file)
            changes.stat_changed_content_same += 1
        else:
            changes.modified.append(file)

    changes.deleted = sorted(set(indexed) - seen)
    return changes


def stale_paths(indexed_paths: Iterable[str], present_paths: Iterable[str]) -> list[str]:
    """Indexed paths that no longer exist on disk."""
    return sorted(set(indexed_paths) - set(present_paths))
