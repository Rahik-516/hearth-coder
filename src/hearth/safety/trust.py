"""Project trust — whether a repository's own config may relax permissions.

The threat this answers is T4: you clone a repository and it ships a
`.hearth/config.toml` allowlisting `run_command` for something it should not have. The
answer is that **allow rules from a project config do nothing until the user runs
`hearth trust`**, and that trust is recorded against `sha256(config file bytes)` rather
than against the path (docs/safety-and-tool-use.md §5.5).

Hashing bytes rather than parsed content is deliberate. Parse-and-hash would let an
attacker append a rule in a form that re-parses to something the user never reviewed, and
it would make "trusted" depend on the parser version. Bytes have neither problem, and they
give a useful property for free: checking out a branch whose config you already approved
restores trust without re-asking, while any edit — even whitespace — drops it.

Deny and ask rules are not gated, because they can only tighten. A repository is allowed
to ask for *more* caution than your defaults without your permission.
"""

from __future__ import annotations

from pathlib import Path

from hearth.safety.rules import Rule
from hearth.storage.state_repo import StateRepository
from hearth.util.hashing import blob_hash


def config_fingerprint(path: Path) -> str | None:
    """SHA-256 of the config file's bytes, or None when there is no such file.

    None and "untrusted" are different facts and callers need both: a project with no
    config has nothing to trust, while a project with an untrusted config has rules being
    actively ignored and deserves a notice.
    """
    try:
        return blob_hash(path.read_bytes())
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None
    except OSError:
        # Unreadable is treated as absent rather than raising: a permissions problem on
        # the project config must not stop the session from starting untrusted.
        return None


def is_project_trusted(repo: StateRepository, config_path: Path) -> bool:
    """Whether this project's config, *at its current bytes*, has been trusted."""
    fingerprint = config_fingerprint(config_path)
    if fingerprint is None:
        return False
    return repo.is_trusted(fingerprint)


def trust_project(repo: StateRepository, config_path: Path) -> str:
    """Record trust for the config's current bytes, returning the fingerprint.

    Raises:
        FileNotFoundError: if there is no config to trust. Recording trust for a file
            that does not exist would silently pre-approve whatever appears there later.
    """
    fingerprint = config_fingerprint(config_path)
    if fingerprint is None:
        raise FileNotFoundError(f"no project config to trust at {config_path}")
    repo.record_trust(fingerprint)
    return fingerprint


def revoke_project_trust(repo: StateRepository, config_path: Path) -> bool:
    """Drop trust for the config's current bytes. False if it was not trusted."""
    fingerprint = config_fingerprint(config_path)
    if fingerprint is None:
        return False
    return repo.revoke_trust(fingerprint)


def relaxing_rules(rules: tuple[Rule, ...]) -> tuple[Rule, ...]:
    """The project rules that `hearth trust` is about to start honouring.

    Only project allow rules qualify. Showing deny and ask rules too would pad the prompt
    with rules the user's answer cannot affect, and a long prompt full of irrelevant lines
    is how approval fatigue (T2) gets trained.
    """
    return tuple(rule for rule in rules if rule.relaxing and rule.source == "project")
