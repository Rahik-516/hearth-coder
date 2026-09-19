"""The policy engine — docs/safety-and-tool-use.md §5.

Written before ``safety/policy.py`` (CLAUDE.md rule 3). Two kinds of test, because the
engine makes two kinds of promise:

* **Tables** pin the §5.1 precedence order and the §4.1 defaults. Precedence is the part
  that is easy to get subtly wrong — a grant checked before a deny rule would be a real
  privilege escalation with no visible symptom.
* **Properties** pin what must hold for *every* input, including rule sets a table would
  never think to write. "No configuration can authorise a write outside the workspace" is
  only meaningful if it survives an adversarial config.
"""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hearth.config.schema import PermissionRule, PermissionsConfig
from hearth.safety.policy import (
    ConfigView,
    PolicyFacts,
    PolicyRequest,
    SessionView,
    evaluate,
)
from hearth.safety.risk import Risk
from hearth.safety.rules import Rule, compile_rules

PROTECTED = ("/home/u/.config/hearth", "/home/u/.local/share/hearth")


def write(path: str = "src/a.py", **overrides: object) -> PolicyRequest:
    """An ``edit_file`` call on a workspace path, unless overridden."""
    facts: dict[str, object] = {
        "path": path,
        "absolute_path": f"/w/{path}",
        "inside_workspace": True,
        "grant_key": f"edit:{path}",
    }
    facts.update(overrides)
    return PolicyRequest(tool="edit_file", risk=Risk.WRITE, facts=PolicyFacts(**facts))  # type: ignore[arg-type]


def read(path: str = "src/a.py", **overrides: object) -> PolicyRequest:
    facts: dict[str, object] = {
        "path": path,
        "absolute_path": f"/w/{path}",
        "inside_workspace": True,
    }
    facts.update(overrides)
    return PolicyRequest(tool="read_file", risk=Risk.READ, facts=PolicyFacts(**facts))  # type: ignore[arg-type]


def rules(**sections: list[PermissionRule]) -> tuple[Rule, ...]:
    return compile_rules(PermissionsConfig(**sections), source="global")  # type: ignore[arg-type]


def config(**overrides: object) -> ConfigView:
    fields: dict[str, object] = {"protected_dirs": PROTECTED}
    fields.update(overrides)
    return ConfigView(**fields)  # type: ignore[arg-type]


AGENT = SessionView(mode="agent", level="supervised")


# ----------------------------------------------------------- 1. invariants win


def test_an_allow_rule_cannot_authorise_a_write_to_git_internals() -> None:
    """Step 1 beats step 6. The single most important ordering in the engine."""
    decision = evaluate(
        write(".git/config"),
        AGENT,
        config(rules=rules(allow=[PermissionRule(id="everything", tool="edit_file", path="**")])),
    )

    assert decision.action == "deny"
    assert decision.decided_by == "invariant"


def test_an_allow_rule_cannot_authorise_a_write_outside_the_workspace() -> None:
    decision = evaluate(
        write("../../etc/hosts", absolute_path="/etc/hosts", inside_workspace=False),
        AGENT,
        config(rules=rules(allow=[PermissionRule(id="everything", tool="edit_file", path="**")])),
    )

    assert decision.action == "deny"
    assert decision.decided_by == "invariant"


def test_a_sensitive_read_is_denied_even_with_an_allow_rule() -> None:
    decision = evaluate(
        read(".ssh/id_rsa", absolute_path="/home/u/.ssh/id_rsa", sensitive_read=True),
        AGENT,
        config(rules=rules(allow=[PermissionRule(id="r", tool="read_file", path="**")])),
    )

    assert decision.action == "deny"
    assert decision.decided_by == "invariant"


def test_a_write_with_no_resolved_path_is_refused() -> None:
    """Fail closed. Every write has a path, so a call without one is malformed.

    The invariant check needs a resolved path to judge; skipping it when the path is
    missing would let a malformed call fall through to the rule engine, where a
    `path = "**"` allow rule would happily cover it.
    """
    request = PolicyRequest(tool="edit_file", risk=Risk.WRITE, facts=PolicyFacts())

    decision = evaluate(
        request,
        AGENT,
        config(rules=rules(allow=[PermissionRule(id="all", tool="edit_file", path="**")])),
    )

    assert decision.action == "deny"
    assert decision.decided_by == "invariant"


# --------------------------------------------------------- 2. mode restrictions


@pytest.mark.parametrize("mode", ["chat", "plan"])
@pytest.mark.parametrize("risk", [Risk.WRITE, Risk.EXEC, Risk.VCS_WRITE])
def test_side_effects_are_refused_outside_agent_mode(mode: str, risk: Risk) -> None:
    request = PolicyRequest(
        tool="edit_file",
        risk=risk,
        facts=PolicyFacts(path="src/a.py", absolute_path="/w/src/a.py"),
    )

    decision = evaluate(request, SessionView(mode=mode), config())

    assert decision.action == "deny"
    assert decision.decided_by == "mode"


def test_reads_are_allowed_in_chat_mode() -> None:
    assert evaluate(read(), SessionView(mode="chat"), config()).action == "allow"


# -------------------------------------------------------------- 3. deny rules


def test_a_deny_rule_beats_a_grant_and_an_allow_rule() -> None:
    decision = evaluate(
        write("src/db/migrations/0001.py"),
        SessionView(mode="agent", grants=frozenset({"edit:src/db/migrations/0001.py"})),
        config(
            rules=rules(
                deny=[PermissionRule(id="no-migrations", tool="edit_file", path="**/migrations/**")],
                allow=[PermissionRule(id="all", tool="edit_file", path="**")],
            )
        ),
    )

    assert decision.action == "deny"
    assert decision.decided_by == "rule"
    assert decision.rule_id == "no-migrations"


def test_a_deny_rule_reason_reaches_the_decision() -> None:
    deny_rule = PermissionRule(id="r", tool="edit_file", path="**", reason="ask me first")
    decision = evaluate(write(), AGENT, config(rules=rules(deny=[deny_rule])))

    assert "ask me first" in decision.reason


def test_project_deny_rules_apply_without_trust() -> None:
    """Deny and ask rules only tighten, so an untrusted repo may still say no (§5.5)."""
    project = compile_rules(
        PermissionsConfig(deny=[PermissionRule(id="p", tool="edit_file", path="src/**")]),
        source="project",
    )

    decision = evaluate(write(), AGENT, config(rules=project, project_trusted=False))

    assert decision.action == "deny"
    assert decision.rule_id == "p"


# ---------------------------------------------------- 4. classification overlays


def test_destructive_asks_with_a_typed_confirmation() -> None:
    decision = evaluate(write(destructive=True), AGENT, config())

    assert decision.action == "ask"
    assert decision.typed_confirmation
    assert "DESTRUCTIVE" in decision.badges


def test_destructive_beats_an_allow_rule() -> None:
    decision = evaluate(
        write(destructive=True),
        AGENT,
        config(rules=rules(allow=[PermissionRule(id="all", tool="edit_file", path="**")])),
    )

    assert decision.action == "ask"


def test_destructive_is_denied_in_headless_mode() -> None:
    decision = evaluate(write(destructive=True), SessionView(mode="agent", headless=True), config())

    assert decision.action == "deny"


def test_a_grant_never_covers_a_destructive_call() -> None:
    """§5.4: grants never cover DESTRUCTIVE classifications."""
    decision = evaluate(
        write(destructive=True),
        SessionView(mode="agent", grants=frozenset({"edit:src/a.py"})),
        config(),
    )

    assert decision.action == "ask"


def test_a_hard_denied_command_is_refused() -> None:
    request = PolicyRequest(
        tool="run_command",
        risk=Risk.EXEC,
        facts=PolicyFacts(hard_denied="sudo is never permitted"),
    )

    decision = evaluate(request, AGENT, config())

    assert decision.action == "deny"
    assert decision.decided_by == "invariant"


# -------------------------------------------------------------- 5. session grants


def test_a_matching_grant_allows() -> None:
    decision = evaluate(write(), SessionView(mode="agent", grants=frozenset({"edit:src/a.py"})), config())

    assert decision.action == "allow"
    assert decision.decided_by == "grant"


def test_grants_are_exact() -> None:
    """A grant for one file must not cover its neighbour (§5.4)."""
    decision = evaluate(
        write("src/b.py"), SessionView(mode="agent", grants=frozenset({"edit:src/a.py"})), config()
    )

    assert decision.action == "ask"


# --------------------------------------------------------------- 6. allow rules


def test_a_global_allow_rule_allows() -> None:
    decision = evaluate(
        write(),
        AGENT,
        config(rules=rules(allow=[PermissionRule(id="src", tool="edit_file", path="src/**")])),
    )

    assert decision.action == "allow"
    assert decision.decided_by == "rule"
    assert decision.rule_id == "src"


def test_a_project_allow_rule_is_ignored_until_the_project_is_trusted() -> None:
    project = compile_rules(
        PermissionsConfig(allow=[PermissionRule(id="p", tool="edit_file", path="**")]),
        source="project",
    )

    untrusted = evaluate(write(), AGENT, config(rules=project, project_trusted=False))
    trusted = evaluate(write(), AGENT, config(rules=project, project_trusted=True))

    assert untrusted.action == "ask", "an untrusted repo cannot grant itself write access"
    assert trusted.action == "allow"
    assert trusted.rule_id == "p"


# ----------------------------------------------------------------- 7. ask rules


def test_an_ask_rule_forces_a_prompt_at_auto_edit() -> None:
    """The point of ask rules: re-tighten a level that would otherwise allow (step 7)."""
    decision = evaluate(
        write("src/handlers/api.py"),
        SessionView(mode="agent", level="auto-edit"),
        config(rules=rules(ask=[PermissionRule(id="api", tool="edit_file", path="src/handlers/**")])),
    )

    assert decision.action == "ask"
    assert decision.rule_id == "api"


# ------------------------------------------------- 8. permission-level defaults


def test_supervised_asks_for_workspace_writes() -> None:
    decision = evaluate(write(), AGENT, config())

    assert decision.action == "ask"
    assert decision.decided_by == "default"
    assert decision.grant_key == "edit:src/a.py"


def test_auto_edit_allows_workspace_writes() -> None:
    decision = evaluate(write(), SessionView(mode="agent", level="auto-edit"), config())

    assert decision.action == "allow"


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/ci.yml",
        ".gitlab-ci.yml",
        "uv.lock",
        "package-lock.json",
        "Dockerfile",
        "Dockerfile.dev",
        "src/db/migrations/0001_init.py",
        "infra/main.tf",
        "AGENTS.md",
        ".env.production",
    ],
)
def test_auto_edit_still_asks_for_the_excluded_set(path: str) -> None:
    """§4.1: auto-edit is for source files, not CI, lockfiles or infrastructure."""
    decision = evaluate(write(path), SessionView(mode="agent", level="auto-edit"), config())

    assert decision.action == "ask", f"{path} must still ask at auto-edit"


def test_read_outside_the_workspace_asks_in_agent_mode() -> None:
    decision = evaluate(
        read("/srv/other/main.py", absolute_path="/srv/other/main.py", inside_workspace=False),
        AGENT,
        config(),
    )

    assert decision.action == "ask"


def test_read_outside_the_workspace_is_denied_in_chat_mode() -> None:
    decision = evaluate(
        read("/srv/other/main.py", absolute_path="/srv/other/main.py", inside_workspace=False),
        SessionView(mode="chat"),
        config(),
    )

    assert decision.action == "deny"


def test_vcs_write_asks_when_supervised() -> None:
    request = PolicyRequest(tool="git_commit", risk=Risk.VCS_WRITE, facts=PolicyFacts())

    assert evaluate(request, AGENT, config()).action == "ask"


def test_meta_is_always_allowed() -> None:
    request = PolicyRequest(tool="todo_write", risk=Risk.META, facts=PolicyFacts())

    assert evaluate(request, SessionView(mode="chat"), config()).action == "allow"


# ------------------------------------------------------------------- headless


def test_headless_turns_every_ask_into_a_deny() -> None:
    decision = evaluate(write(), SessionView(mode="agent", headless=True), config())

    assert decision.action == "deny"
    assert decision.decided_by == "headless"


def test_headless_denial_names_the_flag_that_would_permit_it() -> None:
    """§14.1: exit code 2 is the ordinary outcome, so it has to be actionable."""
    decision = evaluate(write(), SessionView(mode="agent", headless=True), config())

    assert decision.hint is not None
    assert "--allow-edits" in decision.hint


def test_headless_allow_edits_permits_workspace_writes() -> None:
    decision = evaluate(write(), SessionView(mode="agent", headless=True), config(headless_allow_edits=True))

    assert decision.action == "allow"


def test_headless_allow_edits_does_not_permit_commits() -> None:
    request = PolicyRequest(tool="git_commit", risk=Risk.VCS_WRITE, facts=PolicyFacts())

    decision = evaluate(
        request, SessionView(mode="agent", headless=True), config(headless_allow_edits=True)
    )

    assert decision.action == "deny"


def test_headless_allow_commit_permits_a_commit() -> None:
    request = PolicyRequest(tool="git_commit", risk=Risk.VCS_WRITE, facts=PolicyFacts())

    decision = evaluate(
        request, SessionView(mode="agent", headless=True), config(headless_allow_commit=True)
    )

    assert decision.action == "allow"


def test_headless_still_allows_reads() -> None:
    assert evaluate(read(), SessionView(mode="agent", headless=True), config()).action == "allow"


# ------------------------------------------------------------------ properties

_EFFECTS = st.sampled_from(["allow", "deny", "ask"])
_GLOBS = st.sampled_from(["**", "*", "src/**", ".git/**", "**/*.py", ".hearth/**", "?*"])
_TOOLS = st.sampled_from(["edit_file", "write_file", "*", "read_file"])


@st.composite
def _rule_sets(draw: st.DrawFn) -> tuple[Rule, ...]:
    entries = draw(st.lists(st.tuples(_EFFECTS, _TOOLS, _GLOBS), max_size=6))
    sections: dict[str, list[PermissionRule]] = {"allow": [], "deny": [], "ask": []}
    for index, (effect, tool, glob) in enumerate(entries):
        sections[effect].append(PermissionRule(id=f"r{index}", tool=tool, path=glob))
    return compile_rules(PermissionsConfig(**sections), source="global")  # type: ignore[arg-type]


@settings(max_examples=150, deadline=None)
@given(
    compiled=_rule_sets(),
    trusted=st.booleans(),
    level=st.sampled_from(["supervised", "auto-edit"]),
    grants=st.sets(st.sampled_from(["edit:.git/config", "edit:src/a.py", "edit:x"]), max_size=3),
)
def test_no_configuration_allows_a_write_to_git_internals(
    compiled: tuple[Rule, ...], trusted: bool, level: str, grants: set[str]
) -> None:
    decision = evaluate(
        write(".git/config"),
        SessionView(mode="agent", level=level, grants=frozenset(grants)),
        config(rules=compiled, project_trusted=trusted),
    )

    assert decision.action == "deny"


@settings(max_examples=150, deadline=None)
@given(compiled=_rule_sets(), trusted=st.booleans(), level=st.sampled_from(["supervised", "auto-edit"]))
def test_no_configuration_allows_a_write_outside_the_workspace(
    compiled: tuple[Rule, ...], trusted: bool, level: str
) -> None:
    decision = evaluate(
        write("../escape", absolute_path="/etc/escape", inside_workspace=False),
        SessionView(mode="agent", level=level),
        config(rules=compiled, project_trusted=trusted),
    )

    assert decision.action == "deny"


@settings(max_examples=150, deadline=None)
@given(
    compiled=_rule_sets(),
    level=st.sampled_from(["supervised", "auto-edit"]),
    risk=st.sampled_from(list(Risk)),
)
def test_headless_never_asks(compiled: tuple[Rule, ...], level: str, risk: Risk) -> None:
    """There is no channel to ask on, so an Ask that survived would hang or be read as consent."""
    request = PolicyRequest(
        tool="edit_file",
        risk=risk,
        facts=PolicyFacts(path="src/a.py", absolute_path="/w/src/a.py", inside_workspace=True),
    )

    decision = evaluate(
        request, SessionView(mode="agent", level=level, headless=True), config(rules=compiled)
    )

    assert decision.action in ("allow", "deny")


@settings(max_examples=150, deadline=None)
@given(compiled=_rule_sets(), trusted=st.booleans())
def test_a_sensitive_read_is_never_allowed(compiled: tuple[Rule, ...], trusted: bool) -> None:
    decision = evaluate(
        read(".ssh/id_rsa", absolute_path="/home/u/.ssh/id_rsa", sensitive_read=True),
        AGENT,
        config(rules=compiled, project_trusted=trusted),
    )

    assert decision.action == "deny"


@settings(max_examples=150, deadline=None)
@given(compiled=_rule_sets(), level=st.sampled_from(["supervised", "auto-edit"]))
def test_an_untrusted_project_can_never_widen_permissions(
    compiled: tuple[Rule, ...], level: str
) -> None:
    """An untrusted project config must never turn a non-allow outcome into an allow one.

    Re-sourced as ``project`` and left untrusted, the same rules that would grant write
    access from the global config must be inert (§5.5, T4 in the threat model).
    """
    project = tuple(dataclasses.replace(rule, source="project") for rule in compiled)
    session = SessionView(mode="agent", level=level)

    baseline = evaluate(write(), session, config(rules=()))
    with_project = evaluate(write(), session, config(rules=project, project_trusted=False))

    if with_project.action == "allow":
        assert baseline.action == "allow", "an untrusted project config granted write access"
