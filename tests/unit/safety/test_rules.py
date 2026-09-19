"""Rule matching — docs/safety-and-tool-use.md §5.4.

Written before ``safety/rules.py`` (CLAUDE.md rule 3). Matching is where a permission
rule turns into a permission, so a rule that matches more than the user meant is a
privilege escalation with a config file as the exploit.
"""

from __future__ import annotations

import pytest

from hearth.config.schema import PermissionRule, PermissionsConfig
from hearth.safety.rules import Rule, argv_matches, compile_rules, path_matches

# ------------------------------------------------------------------ path globs


@pytest.mark.parametrize(
    ("glob", "path", "expected"),
    [
        ("src/**", "src/billing/models.py", True),
        ("src/**", "tests/test_models.py", False),
        ("**/migrations/**", "src/db/migrations/0003_add.py", True),
        ("**/migrations/**", "src/db/migrations/sub/x.sql", True),
        ("**/migrations/**", "src/db/migration.py", False),
        ("*.tf", "main.tf", True),
        ("*.tf", "infra/main.tf", True),  # gitignore semantics: bare name matches at any depth
        ("Dockerfile*", "Dockerfile", True),
        ("Dockerfile*", "Dockerfile.dev", True),
        (".github/workflows/**", ".github/workflows/ci.yml", True),
        (".github/workflows/**", ".github/dependabot.yml", False),
        ("AGENTS.md", "AGENTS.md", True),
        ("AGENTS.md", "docs/AGENTS.md", True),
    ],
)
def test_path_glob_semantics(glob: str, path: str, expected: bool) -> None:
    assert path_matches(glob, path) is expected


def test_path_glob_does_not_match_when_path_is_absent() -> None:
    """A path rule cannot match a call that has no path — e.g. a command."""
    assert path_matches("src/**", None) is False


# ------------------------------------------------------------------- argv rules


@pytest.mark.parametrize(
    ("pattern", "argv", "expected"),
    [
        (("ruff", "check", "**"), ("ruff", "check", "."), True),
        (("ruff", "check", "**"), ("ruff", "check"), True),  # ** matches zero remaining
        (("ruff", "check", "**"), ("ruff", "format", "."), False),
        (("uv", "run", "mypy", "**"), ("uv", "run", "mypy", "src"), True),
        (("uv", "run", "mypy", "**"), ("uv", "run", "pytest"), False),
        (("pytest",), ("pytest",), True),
        (("pytest",), ("pytest", "-q"), False),  # no ** means exact arity
        (("pytest", "**"), ("pytest", "-q", "tests/"), True),
        (("npm", "run", "test"), ("npm", "run", "build"), False),
    ],
)
def test_argv_prefix_matching(pattern: tuple[str, ...], argv: tuple[str, ...], expected: bool) -> None:
    assert argv_matches(pattern, argv) is expected


def test_argv_rule_never_matches_a_missing_argv() -> None:
    """Fail closed: an argv rule cannot cover a call whose argv could not be parsed.

    A command with shell metacharacters yields ``argv=None`` from the classifier
    (docs/safety-and-tool-use.md §5.4). That is exactly the `pytest; rm -rf ~` case, and
    it must fall through to Ask rather than match an allow rule for `pytest`.
    """
    assert argv_matches(("pytest", "**"), None) is False


def test_glob_element_matches_one_argument_only() -> None:
    assert argv_matches(("git", "log", "-*"), ("git", "log", "-5")) is True
    assert argv_matches(("git", "log", "-*"), ("git", "log", "-5", "--oneline")) is False


# --------------------------------------------------------------- compile_rules


def test_compile_expands_a_list_of_tools() -> None:
    """§5.3 shows `tool = ["edit_file", "write_file"]`; one rule covers several tools."""
    config = PermissionsConfig(
        ask=[PermissionRule(id="migrations", tool=["edit_file", "write_file"], path="**/migrations/**")]
    )
    rules = compile_rules(config, source="global")

    assert len(rules) == 1
    assert rules[0].tools == frozenset({"edit_file", "write_file"})


def test_compile_records_effect_and_source() -> None:
    config = PermissionsConfig(
        allow=[PermissionRule(id="tests", tool="run_tests")],
        deny=[PermissionRule(id="no-docker", tool="run_command", argv=["docker", "**"])],
    )
    rules = compile_rules(config, source="project")

    by_id = {rule.id: rule for rule in rules}
    assert by_id["tests"].effect == "allow"
    assert by_id["no-docker"].effect == "deny"
    assert all(rule.source == "project" for rule in rules)


def test_compile_generates_an_id_when_the_rule_omits_one() -> None:
    """Audit records cite `rule_id`; an unnamed rule still has to be identifiable."""
    config = PermissionsConfig(allow=[PermissionRule(tool="run_tests")])

    rules = compile_rules(config, source="global")

    assert rules[0].id


def test_wildcard_tool_matches_any_tool() -> None:
    rules = compile_rules(PermissionsConfig(deny=[PermissionRule(id="all", tool="*")]), source="global")

    assert rules[0].matches_tool("edit_file")
    assert rules[0].matches_tool("run_command")


def test_rule_requires_every_stated_condition() -> None:
    """Conditions are ANDed. A rule with both a path and an argv needs both to match."""
    rule = Rule(
        id="r",
        effect="allow",
        tools=frozenset({"edit_file"}),
        path_glob="src/**",
        argv=None,
        env=False,
        reason=None,
        source="global",
    )

    assert rule.matches_tool("edit_file")
    assert not rule.matches_tool("write_file")
