"""The hard invariants: step 1 of the policy engine, and never configurable.

``safety/paths.py`` already refuses most of these when a tool resolves a path. This module
exists because that refusal is only as good as the tool remembering to ask for a *write*
resolution, and because the invariants have to sit **above** the rule engine: a user rule
saying ``path = "**"`` with effect ``allow`` must still not authorise a write to
``.git/config``. Evaluation order is what enforces that (docs/safety-and-tool-use.md §5.1,
step 1 before step 6), so these checks are deliberately duplicated rather than trusted to
happen earlier.

Every function here is pure and takes plain values, not request objects. That keeps the
module free of import cycles with ``policy.py`` and makes each invariant a named predicate
that can be tested on its own.

Command invariants (§5.2 items 4-7 — privilege escalation, remote git, pipe-to-shell,
system destruction) need the argv classifier, which lands with the EXEC tools in M6. The
seam is :func:`check_command`, which currently has nothing to classify.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from hearth.config.paths import data_dir, global_config_dir
from hearth.safety.paths import is_protected_write


@dataclass(frozen=True)
class Violation:
    """A refused action, with the invariant that refused it.

    ``invariant`` is a stable identifier so the audit log and the tests can name the rule
    that fired; ``reason`` is the sentence the user and the model see.
    """

    invariant: str
    reason: str


def check_write_path(
    *,
    relative_path: str | None,
    absolute_path: str,
    inside_workspace: bool,
    protected_dirs: tuple[str, ...] | list[str],
) -> Violation | None:
    """Invariants 1 and 2: where a write may land.

    Args:
        relative_path: Workspace-relative POSIX path, or None when the target is outside.
        absolute_path: The resolved absolute path, POSIX-style.
        inside_workspace: Whether the resolved path is within the workspace root.
        protected_dirs: Absolute directories the agent may never write into — Hearth's own
            config and data locations. Passed in rather than looked up, so this stays pure.
    """
    if not inside_workspace:
        return Violation(
            "write-outside-workspace",
            "writes must stay inside the workspace; this path resolves outside it",
        )

    if relative_path is not None and is_protected_write(PurePosixPath(relative_path)):
        return Violation(
            "protected-write-path",
            "this path is repository or Hearth metadata, which no rule can make writable",
        )

    if _within_any(absolute_path, protected_dirs):
        return Violation(
            "privilege-escalation",
            "this path is Hearth's own configuration or data; the agent cannot change its "
            "own permissions",
        )

    return None


def check_read_path(*, absolute_path: str, sensitive: bool) -> Violation | None:
    """Invariant 3: credentials are never read, whatever the permission level.

    A read outside the workspace is *not* an invariant violation — the permission level
    decides that (Ask in agent mode, Deny in chat). Only the sensitive set is absolute,
    because for `~/.ssh` the read itself is the harm.
    """
    if sensitive:
        return Violation(
            "sensitive-read-path",
            "this path holds credentials; reading it is refused regardless of permissions",
        )
    _ = absolute_path
    return None


def check_command(*, argv: tuple[str, ...] | None, hard_denied: str | None) -> Violation | None:
    """Invariants 4-7, as far as the classifier's verdict lets us check them.

    The classification itself lives in ``safety/command_classifier.py`` (M6). This only
    honours a verdict already attached to the call, so ``policy.py`` never has to import
    the classifier and stays a pure evaluator of facts it is handed.
    """
    _ = argv
    if hard_denied:
        return Violation("hard-denied-command", hard_denied)
    return None


def privilege_escalation_dirs() -> tuple[str, ...]:
    """Hearth's own config and data directories, as absolute POSIX strings.

    Not pure — it reads platformdirs — which is why the policy engine receives the result
    in its config view instead of calling this itself.
    """
    return (
        global_config_dir().resolve().as_posix(),
        data_dir().resolve().as_posix(),
    )


def _within_any(candidate: str, roots: tuple[str, ...] | list[str]) -> bool:
    """Path-component-aware containment, so `hearth-notes` is not inside `hearth`."""
    target = PurePosixPath(candidate)
    for root in roots:
        root_path = PurePosixPath(root)
        if target == root_path or target.is_relative_to(root_path):
            return True
    return False
