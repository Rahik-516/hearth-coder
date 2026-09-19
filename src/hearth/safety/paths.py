"""The workspace jail.

**Every filesystem access made by a tool goes through :func:`resolve_in_workspace`**
(docs/project-structure.md §4; a CI grep guard enforces it). This is the boundary that
makes "the agent edited `src/a.py`" mean what it says. If a path can be talked into
resolving somewhere else, the approval the user gave described something other than what
happened.

Three rules, each with a different rationale (docs/safety-and-tool-use.md §7.1):

* **Writes must land inside the workspace**, after symlink resolution. This is a hard
  invariant — no configuration relaxes it.
* **Reads outside the workspace resolve successfully.** Whether they are *allowed* is the
  policy engine's decision, not the jail's. Conflating the two would make it impossible to
  ever permit reading a sibling checkout, even when the user explicitly wants that.
* **Sensitive paths are refused outright, for reads too.** For `~/.ssh` and friends, the
  read *is* the harm: once a key reaches the model's context it may be echoed, summarized,
  or embedded.

Resolution happens before any of those checks, so a symlink cannot smuggle a path past a
string comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from hearth.safety.errors import PathError

#: Directories whose contents must never be read. Absolute, user-relative.
#:
#: Reading is the harm here, which is why these are refused rather than escalated to the
#: policy engine: an approval prompt that says "the model wants to read your SSH key" is
#: a prompt that should never have been generated.
_SENSITIVE_HOME_RELATIVE: tuple[str, ...] = (
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".kube",
    ".docker",
    ".config/gcloud",
    ".config/gh",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    ".ollama",
)

#: Absolute sensitive locations that are not under a home directory.
_SENSITIVE_ABSOLUTE: tuple[str, ...] = (
    "/etc/shadow",
    "/etc/sudoers",
    "/root",
    "/proc/self/environ",
)

#: Under WSL2, the Windows user's credentials are mounted and just as sensitive
#: (docs/safety-and-tool-use.md §5.2, invariant 3). Matched by shape, since the drive
#: letter and username vary.
_WSL_SENSITIVE_SUFFIXES: tuple[str, ...] = (
    ".ssh",
    ".aws",
    ".azure",
    ".kube",
    ".docker",
    ".gnupg",
    ".ollama",
    "appdata/roaming/microsoft/credentials",
)

#: Workspace-relative paths the agent may never write. Matched case-folded, because macOS
#: and Windows are case-insensitive and `.GIT/config` is the same file as `.git/config`.
_PROTECTED_WRITE_PREFIXES: tuple[str, ...] = (
    ".git",
    ".hearth",
    ".hg",
    ".svn",
)


def resolve_in_workspace(
    root: Path,
    user_path: str,
    *,
    for_write: bool = False,
    sensitive_roots: Sequence[Path] | None = None,
) -> Path:
    """Resolve a tool-supplied path against the workspace, or refuse it.

    Args:
        root: The workspace root. Must exist.
        user_path: The path as the model supplied it. Untrusted.
        for_write: Apply the write rules — inside the workspace, not protected.
        sensitive_roots: Override the sensitive set. Tests use this; production does not.

    Returns:
        The resolved absolute path. For reads this may be outside the workspace, which the
        policy engine then judges.

    Raises:
        PathError: for anything refused. This is the only exception callers must handle.
    """
    if not user_path:
        raise PathError("empty path")
    if "\x00" in user_path:
        raise PathError("NUL byte in path", requested=user_path)

    try:
        root_real = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PathError(f"workspace root is unusable: {exc}") from exc

    try:
        raw = Path(user_path).expanduser() if user_path.startswith("~") else Path(user_path)
        candidate = raw if raw.is_absolute() else root_real / raw
        # strict=False: the target may not exist yet, which is legitimate for a write.
        # Symlinks among the *existing* components are still followed, which is what stops
        # a link from smuggling a path past the checks below.
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        # RuntimeError covers symlink loops; ValueError covers embedded-NUL variants that
        # slip past the check above on some platforms.
        raise PathError(f"unresolvable path: {exc}", requested=user_path) from exc

    if is_sensitive_read(resolved, sensitive_roots=sensitive_roots):
        raise PathError("sensitive path", requested=user_path)

    if not for_write:
        # Outside-the-workspace reads are the policy engine's call, not the jail's.
        return resolved

    if not _is_inside(resolved, root_real):
        raise PathError("write outside workspace", requested=user_path)

    if is_protected_write(resolved.relative_to(root_real)):
        raise PathError("protected path", requested=user_path)

    return resolved


def is_sensitive_read(path: Path, *, sensitive_roots: Sequence[Path] | None = None) -> bool:
    """Whether a resolved path is in the never-read set."""
    if sensitive_roots is not None:
        return any(_is_inside(path, root) or path == root for root in sensitive_roots)

    text = path.as_posix()
    lowered = text.lower()

    for absolute in _SENSITIVE_ABSOLUTE:
        if lowered == absolute or lowered.startswith(absolute + "/"):
            return True

    home = Path.home()
    for relative in _SENSITIVE_HOME_RELATIVE:
        target = home / relative
        if path == target or _is_inside(path, target):
            return True

    # WSL: /mnt/<drive>/Users/<name>/.ssh/... and friends.
    if lowered.startswith("/mnt/"):
        for suffix in _WSL_SENSITIVE_SUFFIXES:
            if f"/{suffix}/" in lowered or lowered.endswith(f"/{suffix}"):
                return True

    return False


def is_protected_write(relative_path: Path | PurePosixPath) -> bool:
    """Whether a workspace-relative path is write-protected.

    Case-folded, because the same file is reachable as `.git/config` and `.GIT/config` on
    a case-insensitive filesystem and only one spelling would otherwise be caught.
    """
    parts = [part.lower() for part in PurePosixPath(str(relative_path)).parts]

    # Every component, including the last. A nested `.git` protects vendored subrepos,
    # and the *final* component matters just as much: writing a file literally named
    # `.git` creates a gitlink, which points git at a directory of the writer's choosing.
    return any(part in _PROTECTED_WRITE_PREFIXES for part in parts)


def is_inside_workspace(path: Path, root: Path) -> bool:
    """Whether a resolved path lies within a workspace root."""
    try:
        return _is_inside(path, root.resolve(strict=False))
    except (OSError, RuntimeError):
        return False


def relative_to_workspace(path: Path, root: Path) -> str:
    """Workspace-relative POSIX path for display, falling back to the absolute one."""
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except (OSError, ValueError, RuntimeError):
        return path.as_posix()


def _is_inside(path: Path, root: Path) -> bool:
    if path == root:
        return True
    try:
        return path.is_relative_to(root)
    except (OSError, ValueError):
        return False


def is_symlink_to_outside(path: Path, root: Path) -> bool:
    """Whether ``path`` is a symlink whose target leaves the workspace.

    Checked separately at write time: :func:`resolve_in_workspace` resolves links, but a
    write re-examines the final component because the filesystem can change between the
    check and the write (docs/safety-and-tool-use.md §7.1, TOCTOU).
    """
    try:
        if not path.is_symlink():
            return False
        return not _is_inside(path.resolve(strict=False), root.resolve(strict=False))
    except (OSError, RuntimeError):
        return True  # unreadable link: refuse rather than guess


def has_hard_links(path: Path) -> bool:
    """Whether a file has more than one name.

    A hard link can point at a file outside the workspace while appearing to live inside
    it, so writes to such a file are escalated rather than assumed safe.
    """
    try:
        return path.is_file() and path.stat().st_nlink > 1
    except OSError:
        return False
