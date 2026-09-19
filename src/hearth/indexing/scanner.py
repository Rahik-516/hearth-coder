"""File discovery.

Inside a git repository, ``git ls-files -co --exclude-standard -z`` is the fastest correct
answer: it covers tracked files plus untracked-but-not-ignored ones, and it applies the
user's full ignore configuration — global excludes and ``.git/info/exclude`` included —
which a hand-rolled walk would miss (docs/system-design.md §6.2).

Outside git, or when git fails, the walk fallback applies ``.gitignore`` semantics via
pathspec.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from hearth.git import runner as git_runner
from hearth.indexing.filters import DEFAULT_IGNORED_DIRS, PathFilter, load_ignore_file


@dataclass(frozen=True)
class ScannedFile:
    """A candidate file, with the stat data change detection needs."""

    relative_path: str
    absolute_path: Path
    size_bytes: int
    mtime_ns: int


@dataclass
class ScanResult:
    files: list[ScannedFile]
    used_git: bool
    skipped: dict[str, int]


def is_git_repository(root: Path) -> bool:
    return git_runner.is_git_repository(root)


def scan(root: Path, *, path_filter: PathFilter | None = None) -> ScanResult:
    """Discover indexable files under ``root``."""
    resolved = root.resolve()
    active_filter = path_filter or build_default_filter(resolved)

    candidates, used_git = _discover(resolved)

    files: list[ScannedFile] = []
    skipped: dict[str, int] = {}

    for relative in candidates:
        decision = active_filter.decide(relative)
        if not decision:
            skipped[decision.reason] = skipped.get(decision.reason, 0) + 1
            continue

        absolute = resolved / relative
        try:
            stat = absolute.stat()
        except OSError:
            # Raced with a delete, or a broken symlink. Not an error worth failing on.
            skipped["unreadable"] = skipped.get("unreadable", 0) + 1
            continue

        if not absolute.is_file():
            continue

        files.append(
            ScannedFile(
                relative_path=relative,
                absolute_path=absolute,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )

    files.sort(key=lambda f: f.relative_path)
    return ScanResult(files=files, used_git=used_git, skipped=skipped)


def build_default_filter(root: Path) -> PathFilter:
    """A filter seeded from the repository's own ignore files."""
    return PathFilter(
        gitignore=load_ignore_file(_read_optional(root / ".gitignore")),
        hearthignore=load_ignore_file(_read_optional(root / ".hearthignore")),
    )


def _discover(root: Path) -> tuple[list[str], bool]:
    """Prefer git's own listing; fall back to a walk when git can't answer.

    The git path goes through ``hearth.git.runner`` rather than calling git here, because
    even ``ls-files`` honours ``core.fsmonitor`` — which a hostile repository could point
    at an executable (docs/safety-and-tool-use.md §9.1).
    """
    if is_git_repository(root):
        listed = git_runner.list_files(root)
        if listed is not None:
            return listed, True
    return list(_walk(root)), False


def _walk(root: Path) -> Iterator[str]:
    """Directory walk, pruning ignored directories as it goes.

    Pruning during the walk rather than filtering afterwards is what keeps this from
    descending into ``node_modules``, which on a large repo dominates the runtime.
    """
    for dirpath, dirnames, filenames in root.walk():
        dirnames[:] = sorted(d for d in dirnames if d not in DEFAULT_IGNORED_DIRS)

        for filename in sorted(filenames):
            absolute = dirpath / filename
            try:
                relative = absolute.relative_to(root)
            except ValueError:  # pragma: no cover - defensive
                continue
            yield relative.as_posix()


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
