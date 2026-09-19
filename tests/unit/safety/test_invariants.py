"""Hard invariants — docs/safety-and-tool-use.md §5.2.

Written before ``safety/invariants.py`` (CLAUDE.md rule 3).

These are the checks no configuration can relax, which is the whole point of the module:
``safety/paths.py`` already refuses most of these at resolution time, but that refusal
depends on the *tool* having asked for a write resolution. Invariants run as step 1 of the
policy engine so a permissive rule — `path = "**"` with effect allow — still cannot
authorise a write to `.git/config` or to Hearth's own config.
"""

from __future__ import annotations

import pytest

from hearth.safety.invariants import (
    check_read_path,
    check_write_path,
    privilege_escalation_dirs,
)

HEARTH_DIRS = ("/home/u/.config/hearth", "/home/u/.local/share/hearth")


# --------------------------------------------------------------- write targets


def test_write_inside_the_workspace_is_fine() -> None:
    assert (
        check_write_path(
            relative_path="src/billing/models.py",
            absolute_path="/w/src/billing/models.py",
            inside_workspace=True,
            protected_dirs=HEARTH_DIRS,
        )
        is None
    )


def test_write_outside_the_workspace_is_refused() -> None:
    violation = check_write_path(
        relative_path=None,
        absolute_path="/etc/hosts",
        inside_workspace=False,
        protected_dirs=HEARTH_DIRS,
    )

    assert violation is not None
    assert violation.invariant == "write-outside-workspace"


@pytest.mark.parametrize(
    "relative",
    [
        ".git/config",
        ".git/hooks/pre-commit",
        ".GIT/config",  # case-insensitive filesystems reach the same file
        "vendor/sub/.git/config",  # a vendored subrepo is still git internals
        ".git",  # a *file* named .git is a gitlink: it repoints the repository
        ".hearth/config.toml",
        ".hearth/trust.json",
    ],
)
def test_protected_workspace_paths_are_refused(relative: str) -> None:
    violation = check_write_path(
        relative_path=relative,
        absolute_path=f"/w/{relative}",
        inside_workspace=True,
        protected_dirs=HEARTH_DIRS,
    )

    assert violation is not None, f"{relative} must be protected"
    assert violation.invariant == "protected-write-path"


def test_hearth_own_config_is_refused_even_when_it_is_inside_the_workspace() -> None:
    """The workspace-relative check is not enough on its own.

    Open a session with the workspace set to the home directory and
    `.config/hearth/config.toml` becomes an ordinary workspace-relative path. Writing it
    would let the agent grant itself permissions, which invariant 2 exists to stop, so the
    check has to look at the resolved absolute path too.
    """
    violation = check_write_path(
        relative_path=".config/hearth/config.toml",
        absolute_path="/home/u/.config/hearth/config.toml",
        inside_workspace=True,
        protected_dirs=HEARTH_DIRS,
    )

    assert violation is not None
    assert violation.invariant == "privilege-escalation"


def test_hearth_data_dir_is_refused() -> None:
    violation = check_write_path(
        relative_path=".local/share/hearth/projects/x/state.db",
        absolute_path="/home/u/.local/share/hearth/projects/x/state.db",
        inside_workspace=True,
        protected_dirs=HEARTH_DIRS,
    )

    assert violation is not None
    assert violation.invariant == "privilege-escalation"


def test_a_sibling_named_like_a_protected_dir_is_not_refused() -> None:
    """`/home/u/.config/hearth-notes` is not inside `/home/u/.config/hearth`."""
    assert (
        check_write_path(
            relative_path="notes.md",
            absolute_path="/home/u/.config/hearth-notes/notes.md",
            inside_workspace=True,
            protected_dirs=HEARTH_DIRS,
        )
        is None
    )


# ---------------------------------------------------------------- read targets


def test_sensitive_read_is_refused() -> None:
    violation = check_read_path(absolute_path="/home/u/.ssh/id_rsa", sensitive=True)

    assert violation is not None
    assert violation.invariant == "sensitive-read-path"


def test_ordinary_read_outside_the_workspace_is_not_an_invariant_violation() -> None:
    """Whether it is *allowed* is the permission level's call, not an invariant.

    Conflating the two would make it impossible to ever read a sibling checkout, even
    when the user explicitly asks for it (docs/safety-and-tool-use.md §7.1).
    """
    assert check_read_path(absolute_path="/srv/other-repo/main.py", sensitive=False) is None


# ------------------------------------------------------------------- the set


def test_privilege_escalation_dirs_names_real_locations() -> None:
    dirs = privilege_escalation_dirs()

    assert dirs, "there is always at least a global config directory"
    assert all(path.startswith("/") or ":" in path for path in dirs)
