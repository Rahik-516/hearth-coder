"""Hardened git invocation.

**Every git call in Hearth goes through here.** The flags are a security control, not
tidiness (docs/safety-and-tool-use.md §9.1).

Repository-local git configuration can make an *ordinary read* execute programs, via
fsmonitor hooks, external diff drivers and textconv filters. That config is not normally
cloned, but a repository obtained as an archive can carry its own ``.git`` directory — and
Hearth's whole job is to point itself at repositories it did not write. So read operations
disable those mechanisms explicitly.

``core.fsmonitor`` matters even for ``ls-files``, which is why file discovery uses this
module rather than calling git directly.

Commits deliberately keep hooks **enabled** and never pass ``--no-verify``: a project's
pre-commit checks are the user's, and silently skipping them would be Hearth overriding a
decision that isn't its to make.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

#: Config overrides applied to every invocation. `-c` flags win over repo config.
_BASE_HARDENING: tuple[str, ...] = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "diff.external=",
    "-c",
    "core.pager=cat",
    "-c",
    "color.ui=false",
    "-c",
    "advice.detachedHead=false",
)

#: Additional flags for read-only subcommands, where hooks have no legitimate role.
_READ_HARDENING: tuple[str, ...] = ("-c", "core.hooksPath=/dev/null")

#: Environment for git subprocesses: non-interactive, no credential prompts, no pager.
_GIT_ENV: dict[str, str] = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_PAGER": "cat",
    "GIT_OPTIONAL_LOCKS": "0",
    "GCM_INTERACTIVE": "never",
    "NO_COLOR": "1",
}

DEFAULT_TIMEOUT_S = 60


class GitError(Exception):
    """A git invocation failed or could not be run."""


@dataclass(frozen=True)
class GitResult:
    returncode: int
    stdout: bytes
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")


def run_git(
    root: Path,
    args: list[str],
    *,
    read_only: bool = True,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    check: bool = False,
) -> GitResult:
    """Run a git subcommand in ``root`` with hardening applied.

    Args:
        root: Repository directory.
        args: Subcommand and its arguments, e.g. ``["ls-files", "-z"]``.
        read_only: Apply the read-only hardening (hooks disabled). Set False for
            operations such as ``commit``, where the project's hooks must run.
        check: Raise ``GitError`` on a non-zero exit instead of returning it.
    """
    hardening = _BASE_HARDENING + (_READ_HARDENING if read_only else ())
    argv = ["git", "-C", str(root), *hardening, "--no-pager", *args]

    import os

    env = {**os.environ, **_GIT_ENV}

    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        raise GitError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {' '.join(args)} timed out after {timeout_s}s") from exc
    except OSError as exc:
        raise GitError(f"could not run git: {exc}") from exc

    result = GitResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr.decode("utf-8", errors="replace").strip(),
    )
    if check and not result.ok:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr}")
    return result


def is_git_repository(root: Path) -> bool:
    """Whether ``root`` is inside a git working tree."""
    if not (root / ".git").exists():
        return False
    try:
        result = run_git(root, ["rev-parse", "--is-inside-work-tree"], timeout_s=10)
    except GitError:
        return False
    return result.ok and result.text().strip() == "true"


def list_files(root: Path, *, timeout_s: int = DEFAULT_TIMEOUT_S) -> list[str] | None:
    """Tracked plus untracked-but-not-ignored files, NUL-separated.

    Returns None when git is unavailable or the command fails, so callers can fall back
    to a directory walk rather than treating it as fatal.
    """
    try:
        result = run_git(
            root,
            ["ls-files", "-co", "--exclude-standard", "-z"],
            timeout_s=timeout_s,
        )
    except GitError:
        return None

    if not result.ok:
        return None
    return [entry for entry in result.text().split("\0") if entry]
