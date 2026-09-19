"""Workspace jail — written before the implementation, adversarially.

This is the boundary every file-touching tool goes through. If it can be talked into
returning a path outside the workspace, every later control is decoration: the policy
engine approves an edit to what it believes is `src/a.py`, and something else changes.

So the tests come first and try to break it (docs/safety-and-tool-use.md §7.1, and the
security checklist in §15). Hypothesis generates the traversal cases, because the
interesting failures are the ones nobody thought to enumerate.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from hearth.safety.errors import PathError
from hearth.safety.paths import (
    is_protected_write,
    is_sensitive_read,
    resolve_in_workspace,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (root / ".hearth").mkdir()
    (root / ".hearth" / "config.toml").write_text("[project]\n", encoding="utf-8")
    return root


# ------------------------------------------------------------------- allowed


def test_relative_path_resolves_inside(workspace: Path) -> None:
    assert resolve_in_workspace(workspace, "src/a.py") == (workspace / "src" / "a.py").resolve()


def test_nested_relative_path(workspace: Path) -> None:
    resolved = resolve_in_workspace(workspace, "./src/../src/a.py")
    assert resolved == (workspace / "src" / "a.py").resolve()


def test_absolute_path_inside_workspace(workspace: Path) -> None:
    target = workspace / "src" / "a.py"
    assert resolve_in_workspace(workspace, str(target)) == target.resolve()


def test_path_that_does_not_exist_yet_is_allowed_for_write(workspace: Path) -> None:
    """Creating a new file is legitimate; only its *location* is constrained."""
    resolved = resolve_in_workspace(workspace, "src/new_file.py", for_write=True)
    assert resolved.parent == (workspace / "src").resolve()


def test_workspace_root_itself(workspace: Path) -> None:
    assert resolve_in_workspace(workspace, ".") == workspace.resolve()


# ---------------------------------------------------------------- traversal


@pytest.mark.parametrize(
    "attempt",
    [
        "../outside.txt",
        "../../outside.txt",
        "src/../../outside.txt",
        "src/../../../etc/passwd",
        "./../../outside.txt",
        "src/./../../outside.txt",
        "a/b/c/../../../../outside.txt",
    ],
)
def test_traversal_is_refused_for_write(workspace: Path, attempt: str) -> None:
    with pytest.raises(PathError):
        resolve_in_workspace(workspace, attempt, for_write=True)


def test_absolute_path_outside_is_refused_for_write(workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.txt"
    with pytest.raises(PathError, match="outside"):
        resolve_in_workspace(workspace, str(outside), for_write=True)


def test_reads_outside_are_returned_for_policy_to_judge(workspace: Path, tmp_path: Path) -> None:
    """Reads outside the workspace are not a jail error.

    The jail's job is to resolve honestly; whether an outside read is allowed is the
    policy engine's decision (Ask in supervised mode, Deny in headless). Conflating the
    two would make `read_file ../sibling/README.md` impossible to ever permit.
    """
    outside = tmp_path / "sibling.txt"
    outside.write_text("hello", encoding="utf-8")

    resolved = resolve_in_workspace(workspace, str(outside), for_write=False)

    assert resolved == outside.resolve()
    assert not resolved.is_relative_to(workspace.resolve())


# ------------------------------------------------------------------ symlinks


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges on Windows")
def test_symlink_escaping_the_workspace_is_refused_for_write(workspace: Path, tmp_path: Path) -> None:
    """The classic escape: a link inside the tree pointing out of it."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "escape").symlink_to(outside)

    with pytest.raises(PathError, match="outside"):
        resolve_in_workspace(workspace, "escape/evil.txt", for_write=True)


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges on Windows")
def test_symlink_to_sensitive_path_is_refused_even_for_read(workspace: Path, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    (fake_home / ".ssh").mkdir(parents=True)
    key = fake_home / ".ssh" / "id_rsa"
    key.write_text("PRIVATE", encoding="utf-8")
    (workspace / "innocent.txt").symlink_to(key)

    with pytest.raises(PathError, match="sensitive"):
        resolve_in_workspace(
            workspace, "innocent.txt", for_write=False, sensitive_roots=(fake_home / ".ssh",)
        )


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges on Windows")
def test_symlink_staying_inside_is_allowed(workspace: Path) -> None:
    """A link is not suspicious on its own — only where it lands."""
    (workspace / "link").symlink_to(workspace / "src")

    resolved = resolve_in_workspace(workspace, "link/a.py", for_write=True)

    assert resolved == (workspace / "src" / "a.py").resolve()


# ----------------------------------------------------------------- protected


@pytest.mark.parametrize(
    "attempt",
    [
        ".git/config",
        ".git/HEAD",
        ".git/hooks/pre-commit",
        ".hearth/config.toml",
        ".hearth/trust.json",
    ],
)
def test_protected_paths_refuse_writes(workspace: Path, attempt: str) -> None:
    """The agent may never edit git internals or its own permissions."""
    with pytest.raises(PathError, match="protected"):
        resolve_in_workspace(workspace, attempt, for_write=True)


def test_protected_paths_are_readable(workspace: Path) -> None:
    """Reading .git/config is fine; writing it is not."""
    assert resolve_in_workspace(workspace, ".git/config", for_write=False)


def test_protected_matching_is_case_folded(workspace: Path) -> None:
    """macOS and Windows are case-insensitive; `.GIT/config` must not slip through."""
    with pytest.raises(PathError, match="protected"):
        resolve_in_workspace(workspace, ".GIT/config", for_write=True)


def test_is_protected_write_reports_the_same_set(workspace: Path) -> None:
    assert is_protected_write(Path(".git/config"))
    assert is_protected_write(Path(".hearth/config.toml"))
    assert not is_protected_write(Path("src/a.py"))


# ----------------------------------------------------------------- sensitive


@pytest.mark.parametrize(
    "relative",
    [".ssh/id_rsa", ".aws/credentials", ".gnupg/secring.gpg", ".config/gcloud/creds.json"],
)
def test_sensitive_home_paths_are_refused(tmp_path: Path, relative: str) -> None:
    """These are refused for reads too — the one case where reading is itself the harm."""
    home = tmp_path / "home"
    target = home / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("secret", encoding="utf-8")
    workspace = tmp_path / "repo"
    workspace.mkdir()

    with pytest.raises(PathError, match="sensitive"):
        resolve_in_workspace(
            workspace,
            str(target),
            for_write=False,
            sensitive_roots=(home / ".ssh", home / ".aws", home / ".gnupg", home / ".config/gcloud"),
        )


def test_is_sensitive_read_uses_the_documented_set() -> None:
    home = Path.home()
    assert is_sensitive_read(home / ".ssh" / "id_rsa")
    assert is_sensitive_read(home / ".aws" / "credentials")
    assert not is_sensitive_read(Path("/tmp/ordinary.txt"))


@pytest.mark.skipif(sys.platform == "win32", reason="WSL-specific mount layout")
def test_wsl_windows_side_credentials_are_sensitive() -> None:
    """Under WSL the Windows user's keys are reachable at /mnt/c/Users/... too."""
    assert is_sensitive_read(Path("/mnt/c/Users/dev/.ssh/id_rsa"))
    assert is_sensitive_read(Path("/mnt/d/Users/dev/.aws/credentials"))


# -------------------------------------------------------------- malformed


def test_nul_byte_is_refused(workspace: Path) -> None:
    """A NUL truncates the path in any C-level call underneath us."""
    with pytest.raises(PathError, match="NUL"):
        resolve_in_workspace(workspace, "src/a\x00.py")


def test_empty_path_is_refused(workspace: Path) -> None:
    with pytest.raises(PathError):
        resolve_in_workspace(workspace, "")


def test_tilde_expands_to_home_and_is_then_judged(workspace: Path) -> None:
    """`~` must not silently become a literal directory named '~' inside the workspace."""
    resolved = resolve_in_workspace(workspace, "~/notes.txt", for_write=False)
    assert resolved.is_relative_to(Path.home())


def test_tilde_write_outside_workspace_is_refused(workspace: Path) -> None:
    with pytest.raises(PathError, match="outside"):
        resolve_in_workspace(workspace, "~/notes.txt", for_write=True)


# ------------------------------------------------------------------ property


_SEGMENTS = st.sampled_from(["..", ".", "src", "a", "b", "..", "…", "sub dir", ".git"])


@settings(max_examples=300, deadline=None)
@given(st.lists(_SEGMENTS, min_size=1, max_size=8))
def test_write_never_escapes_the_workspace(tmp_path_factory, segments: list[str]) -> None:
    """Whatever the path, a permitted write resolves inside the root.

    The property that matters: not "traversal is rejected" but "nothing that is accepted
    lands outside". Those differ whenever a new syntax slips past a denylist.
    """
    root = tmp_path_factory.mktemp("prop")
    candidate = "/".join(segments)

    try:
        resolved = resolve_in_workspace(root, candidate, for_write=True)
    except PathError:
        return  # refusing is always a valid outcome

    assert resolved.is_relative_to(root.resolve())


@settings(max_examples=200, deadline=None)
@given(st.text(min_size=1, max_size=60))
def test_never_raises_anything_but_path_error(tmp_path_factory, candidate: str) -> None:
    """Arbitrary input must produce a PathError, never a crash.

    A tool that raises OSError instead of PathError escapes the gateway's error handling
    and surfaces as a traceback rather than a corrective message to the model.
    """
    assume("\x00" not in candidate or True)  # NUL is explicitly covered above
    root = tmp_path_factory.mktemp("prop2")

    with contextlib.suppress(PathError):
        resolve_in_workspace(root, candidate, for_write=True)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.sampled_from([".git", "config", "hooks", "src", "a.py"]), min_size=1, max_size=5))
def test_git_internals_are_never_writable(tmp_path_factory, segments: list[str]) -> None:
    root = tmp_path_factory.mktemp("prop3")
    candidate = "/".join(segments)

    try:
        resolved = resolve_in_workspace(root, candidate, for_write=True)
    except PathError:
        return

    relative = resolved.relative_to(root.resolve())
    assert ".git" not in {part.lower() for part in relative.parts}


def test_root_must_exist(tmp_path: Path) -> None:
    """A workspace that is not there cannot be reasoned about; fail rather than guess."""
    with pytest.raises(PathError):
        resolve_in_workspace(tmp_path / "missing", "a.py")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_resolution_does_not_follow_into_unreadable_dirs(workspace: Path) -> None:
    """Resolution must not raise on a directory it cannot stat."""
    locked = workspace / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        with contextlib.suppress(PathError):
            resolve_in_workspace(workspace, "locked/inner.txt", for_write=True)
    finally:
        locked.chmod(0o755)
