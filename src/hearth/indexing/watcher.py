"""Keeping the index current while the user works.

An index built once goes stale the moment someone saves a file, and a stale index is worse
than an obviously absent one: retrieval still answers, with citations pointing at lines
that have moved. So the workspace is watched and changed files are re-indexed as they
settle.

The hard case is not a single save — it is a **git branch switch**, which rewrites hundreds
of files in well under a second. Handled naively that is hundreds of re-index passes,
each opening the database, while the user waits. Two things prevent it:

* **Debounce.** Events are collected until the filesystem goes quiet, so a burst arrives as
  one batch. ``watchfiles`` does this natively; the default window is deliberately wide
  enough to cover a checkout rather than tuned for keystroke latency.
* **Re-index only what changed.** Each path in the batch goes through the same change
  detection a full run uses, so a branch switch that touched 400 files but left 380
  byte-identical parses 20 of them. That is the I2 acceptance criterion.

The decision of *what* a batch means is separated from the watching (:func:`plan_batch`),
because a test that races a real filesystem watcher is a test that fails on a slow machine
and passes on a fast one. The loop is thin; the judgement is pure.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from hearth.indexing.filters import PathFilter
from hearth.indexing.pipeline import Indexer

#: How long the filesystem must be quiet before a batch is considered settled. Wide enough
#: to swallow a `git checkout` (hundreds of writes over a few hundred milliseconds) rather
#: than reacting to each one.
DEFAULT_DEBOUNCE_MS = 1_600

#: Polling interval within the debounce window.
DEFAULT_STEP_MS = 50

#: Directories whose churn is never worth re-indexing for. `.git` matters most: a branch
#: switch rewrites its contents, so watching it would make every checkout trigger a batch
#: describing Hearth's own trigger.
_ALWAYS_IGNORED = ("/.git/", "/.hearth/", "/node_modules/", "/__pycache__/", "/.venv/")


@dataclass(frozen=True)
class WatchBatch:
    """What one settled burst of filesystem events means for the index."""

    #: Workspace-relative paths that exist and should be re-indexed.
    changed: tuple[str, ...] = ()
    #: Workspace-relative paths that are gone and should be removed.
    deleted: tuple[str, ...] = ()
    #: Events discarded by the ignore rules, for the "why did nothing happen" question.
    ignored: int = 0

    @property
    def empty(self) -> bool:
        return not self.changed and not self.deleted

    @property
    def total(self) -> int:
        return len(self.changed) + len(self.deleted)


@dataclass
class WatchStats:
    """Cumulative work done by a watcher."""

    batches: int = 0
    reindexed: int = 0
    removed: int = 0
    #: Files in a batch that turned out to be byte-identical. The number that makes a
    #: branch switch cheap, and the one worth watching if it ever stops being large.
    unchanged: int = 0
    #: Files actually handed to tree-sitter. The honest measure of work done.
    parsed: int = 0
    failed: int = 0
    paths_seen: set[str] = field(default_factory=set)


def plan_batch(
    events: Iterable[tuple[object, str]],
    *,
    root: Path,
    path_filter: PathFilter | None = None,
) -> WatchBatch:
    """Turn raw watcher events into index work.

    Args:
        events: ``(change, absolute path)`` pairs, as ``watchfiles`` yields them. The change
            kind is deliberately ignored in favour of checking the filesystem: an editor
            saving atomically emits delete-then-add for what the user experienced as one
            edit, and trusting the event kind would delete the file from the index and
            re-add it, losing nothing but doing twice the work. What is on disk *now* is
            the only reliable answer.
        root: The workspace, for making paths relative.
        path_filter: Ignore rules. Without one only the built-in directory list applies.
    """
    changed: list[str] = []
    deleted: list[str] = []
    ignored = 0
    seen: set[str] = set()

    for _change, raw in events:
        absolute = Path(raw)
        relative = _relative_to(absolute, root)
        if relative is None or relative in seen:
            continue
        seen.add(relative)

        if _is_always_ignored(relative) or not _is_indexable(relative, path_filter):
            ignored += 1
            continue

        if absolute.is_file():
            changed.append(relative)
        else:
            deleted.append(relative)

    return WatchBatch(
        changed=tuple(sorted(changed)),
        deleted=tuple(sorted(deleted)),
        ignored=ignored,
    )


class IndexWatcher:
    """Watches a workspace and keeps its index in step."""

    def __init__(
        self,
        *,
        root: Path,
        indexer: Indexer,
        path_filter: PathFilter | None = None,
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
        on_batch: Callable[[WatchBatch], None] | None = None,
    ) -> None:
        self._root = root.resolve()
        self._indexer = indexer
        self._filter = path_filter
        self._debounce_ms = debounce_ms
        self._on_batch = on_batch
        self.stats = WatchStats()

    def apply(self, batch: WatchBatch) -> WatchStats:
        """Re-index one settled batch.

        Goes through ``index_paths`` rather than ``index_one`` per file, and the
        difference is the acceptance criterion. ``index_one`` re-indexes
        unconditionally — right for the write tools, which call it knowing the file just
        changed — but a branch switch hands over the whole working tree. Scoped change
        detection means the files git left byte-identical are not parsed again.

        A failure does not abandon the batch. The usual cause is a file rewritten again
        while being read, which the next batch covers; dropping the other 399 files of a
        checkout over one racing write would be a poor trade.
        """
        if self._on_batch is not None:
            self._on_batch(batch)

        self.stats.batches += 1
        touched = [*batch.changed, *batch.deleted]

        try:
            stats = self._indexer.index_paths(touched)
        except Exception:
            self.stats.failed += len(touched)
            return self.stats

        self.stats.reindexed += stats.files_written
        self.stats.removed += stats.deleted
        self.stats.unchanged += stats.unchanged
        self.stats.parsed += stats.parsed
        self.stats.paths_seen.update(touched)
        return self.stats

    async def watch(self, *, stop_event: object | None = None) -> WatchStats:
        """Watch until ``stop_event`` is set. Long-running.

        ``watchfiles`` is imported here rather than at module scope: it carries a compiled
        extension, and the CLI's startup budget should not pay for it on paths that never
        watch anything.
        """
        from watchfiles import awatch

        async for events in awatch(
            self._root,
            debounce=self._debounce_ms,
            step=DEFAULT_STEP_MS,
            stop_event=stop_event,
            # Hearth's own ignore rules decide what matters; the library's defaults would
            # additionally hide things a repository legitimately tracks.
            watch_filter=None,
            recursive=True,
        ):
            batch = plan_batch(events, root=self._root, path_filter=self._filter)
            if not batch.empty:
                self.apply(batch)

        return self.stats


def _relative_to(absolute: Path, root: Path) -> str | None:
    """Workspace-relative POSIX path, or None when the event is outside the workspace."""
    with contextlib.suppress(ValueError):
        return absolute.resolve().relative_to(root.resolve()).as_posix()
    return None


def _is_always_ignored(relative: str) -> bool:
    probe = f"/{relative}"
    return any(marker in probe for marker in _ALWAYS_IGNORED)


def _is_indexable(relative: str, path_filter: PathFilter | None) -> bool:
    """Whether the indexer would accept this path. ``FilterDecision`` is truthy on include."""
    return True if path_filter is None else bool(path_filter.decide(relative))
