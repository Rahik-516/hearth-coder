"""Project trust — docs/safety-and-tool-use.md §5.5.

Written before ``safety/trust.py`` (CLAUDE.md rule 3). This is the control for T4 in the
threat model: a cloned repository shipping a `.hearth/config.toml` that allowlists
something dangerous. Trust is pinned to the file's bytes, so the interesting tests are the
ones about *losing* trust.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hearth.config.schema import PermissionRule, PermissionsConfig
from hearth.safety.rules import compile_rules
from hearth.safety.trust import (
    config_fingerprint,
    is_project_trusted,
    relaxing_rules,
    revoke_project_trust,
    trust_project,
)
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository


@pytest.fixture
def repo(tmp_path: Path) -> StateRepository:
    connection = connect(tmp_path / "state.db")
    migrate(connection, database="state")
    return StateRepository(connection)


@pytest.fixture
def project_config(tmp_path: Path) -> Path:
    path = tmp_path / "project" / ".hearth" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text('[[permissions.allow]]\nid = "tests"\ntool = "run_tests"\n', encoding="utf-8")
    return path


# ----------------------------------------------------------------- fingerprint


def test_fingerprint_is_stable_for_identical_bytes(tmp_path: Path) -> None:
    first = tmp_path / "a.toml"
    second = tmp_path / "b.toml"
    first.write_bytes(b"tool = 'x'\n")
    second.write_bytes(b"tool = 'x'\n")

    assert config_fingerprint(first) == config_fingerprint(second)


def test_fingerprint_is_none_for_a_missing_file(tmp_path: Path) -> None:
    """No config is not the same as an untrusted config; callers must tell them apart."""
    assert config_fingerprint(tmp_path / "absent.toml") is None


def test_fingerprint_changes_with_whitespace(tmp_path: Path) -> None:
    """*Any* byte change invalidates trust (§5.5) — not just a semantic one.

    Hashing parsed content instead would let an attacker append a rule in a way that
    re-parses to something trust never covered.
    """
    path = tmp_path / "c.toml"
    path.write_bytes(b"tool = 'x'\n")
    before = config_fingerprint(path)
    path.write_bytes(b"tool = 'x'\n\n")

    assert config_fingerprint(path) != before


# --------------------------------------------------------------- trust records


def test_a_project_starts_untrusted(repo: StateRepository, project_config: Path) -> None:
    assert is_project_trusted(repo, project_config) is False


def test_trusting_a_project_records_it(repo: StateRepository, project_config: Path) -> None:
    trust_project(repo, project_config)

    assert is_project_trusted(repo, project_config) is True


def test_editing_the_config_loses_trust(repo: StateRepository, project_config: Path) -> None:
    """The heart of §5.5: trust is for bytes you saw, not for a path."""
    trust_project(repo, project_config)

    project_config.write_text(
        '[[permissions.allow]]\nid = "evil"\ntool = "run_command"\nargv = ["sh", "**"]\n',
        encoding="utf-8",
    )

    assert is_project_trusted(repo, project_config) is False


def test_restoring_the_original_bytes_restores_trust(repo: StateRepository, project_config: Path) -> None:
    """A consequence of hashing bytes, and the right one.

    Reverting a branch switch back to the config you already reviewed should not require
    re-approving it — the record is of content, and that content is trusted.
    """
    original = project_config.read_bytes()
    trust_project(repo, project_config)
    project_config.write_bytes(b"tool = 'other'\n")
    assert is_project_trusted(repo, project_config) is False

    project_config.write_bytes(original)

    assert is_project_trusted(repo, project_config) is True


def test_revoking_trust_takes_effect(repo: StateRepository, project_config: Path) -> None:
    trust_project(repo, project_config)

    assert revoke_project_trust(repo, project_config) is True
    assert is_project_trusted(repo, project_config) is False


def test_revoking_an_untrusted_project_reports_that_nothing_changed(
    repo: StateRepository, project_config: Path
) -> None:
    assert revoke_project_trust(repo, project_config) is False


def test_a_missing_config_is_never_trusted(repo: StateRepository, tmp_path: Path) -> None:
    assert is_project_trusted(repo, tmp_path / "absent.toml") is False


def test_trusting_a_missing_config_is_refused(repo: StateRepository, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        trust_project(repo, tmp_path / "absent.toml")


# ------------------------------------------------- what `hearth trust` displays


def test_relaxing_rules_lists_only_the_allow_rules() -> None:
    """`hearth trust` must show what it is about to permit, and nothing else.

    Deny and ask rules already apply untrusted, so listing them would pad the prompt with
    rules the answer does not affect — which is how approval fatigue starts (T2).
    """
    compiled = compile_rules(
        PermissionsConfig(
            allow=[PermissionRule(id="a", tool="run_tests")],
            deny=[PermissionRule(id="d", tool="run_command", argv=["docker", "**"])],
            ask=[PermissionRule(id="k", tool="edit_file", path="**/migrations/**")],
        ),
        source="project",
    )

    relaxing = relaxing_rules(compiled)

    assert [rule.id for rule in relaxing] == ["a"]


def test_relaxing_rules_ignores_global_rules() -> None:
    """Trust governs project rules only; the user's own config needs no approval."""
    compiled = compile_rules(
        PermissionsConfig(allow=[PermissionRule(id="mine", tool="run_tests")]), source="global"
    )

    assert relaxing_rules(compiled) == ()


def test_rules_describe_themselves_for_the_prompt() -> None:
    compiled = compile_rules(
        PermissionsConfig(
            allow=[PermissionRule(id="ruff", tool="run_command", argv=["ruff", "check", "**"])]
        ),
        source="project",
    )

    text = compiled[0].describe()

    assert "ruff" in text
    assert "run_command" in text


# --------------------------------------------------------------- session grants


def test_grants_are_recorded_and_read_back(repo: StateRepository) -> None:
    session = repo.create_session(workspace="/w")

    repo.add_grant(session.id, "edit:src/a.py")

    assert repo.grants_for(session.id) == frozenset({"edit:src/a.py"})


def test_grants_are_scoped_to_one_session(repo: StateRepository) -> None:
    """A grant is "for this session". Leaking across sessions would outlive the context
    in which the user granted it."""
    first = repo.create_session(workspace="/w")
    second = repo.create_session(workspace="/w")

    repo.add_grant(first.id, "edit:src/a.py")

    assert repo.grants_for(second.id) == frozenset()


def test_adding_the_same_grant_twice_is_harmless(repo: StateRepository) -> None:
    session = repo.create_session(workspace="/w")

    repo.add_grant(session.id, "edit:src/a.py")
    repo.add_grant(session.id, "edit:src/a.py")

    assert repo.grants_for(session.id) == frozenset({"edit:src/a.py"})


def test_grants_vanish_with_the_session(repo: StateRepository) -> None:
    """ON DELETE CASCADE. A grant must not outlive the session that authorised it."""
    session = repo.create_session(workspace="/w")
    repo.add_grant(session.id, "edit:src/a.py")

    repo.delete_session(session.id)

    assert repo.grants_for(session.id) == frozenset()
