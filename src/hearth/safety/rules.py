"""Compiling and matching user-authored permission rules.

Rule matching is where a line in a TOML file becomes a permission, so the bias throughout
is toward matching *less* than the user might have meant. Two consequences are worth
naming because they look like bugs until you see the attack they stop
(docs/safety-and-tool-use.md §5.4):

* **An ``argv`` rule can never match a command that did not parse into a single simple
  command.** The classifier hands over ``argv=None`` for anything containing shell
  metacharacters, and :func:`argv_matches` refuses ``None`` outright. That is what makes
  an allow rule for ``pytest`` useless to ``pytest; rm -rf ~``.
* **Every stated condition must match.** A rule naming both a tool and a path covers only
  calls satisfying both. Conditions narrow; they never widen.

Path globs use gitignore syntax through ``pathspec``, the same engine
``indexing/filters.py`` uses for ignore files, so "what does this glob match" has one
answer in Hearth rather than one per module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import pathspec

from hearth.config.schema import PermissionRule, PermissionsConfig

Effect = Literal["allow", "deny", "ask"]
RuleSource = Literal["global", "project", "flag"]

#: Matches any tool.
ANY_TOOL = "*"

#: Glob element meaning "and everything after this".
REST = "**"


@dataclass(frozen=True)
class Rule:
    """One compiled permission rule.

    Frozen and free of I/O so the policy engine that consumes it stays pure.
    """

    id: str
    effect: Effect
    tools: frozenset[str]
    path_glob: str | None
    argv: tuple[str, ...] | None
    env: bool
    reason: str | None
    source: RuleSource

    @property
    def relaxing(self) -> bool:
        """Whether this rule grants something. Only these are gated on project trust.

        Deny and ask rules from a project config always apply, because they can only
        tighten — an untrusted repository is allowed to ask for *more* caution than the
        user's defaults (docs/safety-and-tool-use.md §5.5).
        """
        return self.effect == "allow"

    def matches_tool(self, tool: str) -> bool:
        return ANY_TOOL in self.tools or tool in self.tools

    def conditions(self) -> str:
        """The rule's matching conditions, without its effect or id.

        Separate from :meth:`describe` because a table that already has an Effect column
        should not repeat it in the text — and because the id needs escaping when it
        reaches Rich, which would otherwise read ``[project:allow:0]`` as markup and
        render nothing at all.
        """
        parts = ["/".join(sorted(self.tools))]
        if self.argv:
            parts.append(" ".join(self.argv))
        if self.path_glob:
            parts.append(self.path_glob)
        if self.env:
            parts.append("env=true")
        return "  ".join(parts)

    def describe(self) -> str:
        """One line for logs and approval reasons."""
        return f"[{self.id}] {self.effect} {self.conditions()}"


def compile_rules(config: PermissionsConfig, *, source: RuleSource) -> tuple[Rule, ...]:
    """Turn a config section into compiled rules, in declaration order."""
    compiled: list[Rule] = []
    for effect, entries in (("deny", config.deny), ("allow", config.allow), ("ask", config.ask)):
        for position, entry in enumerate(entries):
            compiled.append(_compile_one(entry, effect, source, position))
    return tuple(compiled)


def _compile_one(entry: PermissionRule, effect: str, source: RuleSource, position: int) -> Rule:
    names = [entry.tool] if isinstance(entry.tool, str) else list(entry.tool)
    return Rule(
        id=entry.id or f"{source}:{effect}:{position}",
        effect=effect,  # type: ignore[arg-type]
        tools=frozenset(name.strip() for name in names),
        path_glob=entry.path,
        argv=tuple(entry.argv) if entry.argv else None,
        env=entry.env,
        reason=entry.reason,
        source=source,
    )


def path_matches(glob: str, path: str | None) -> bool:
    """Gitignore-style match of one glob against a workspace-relative POSIX path.

    ``None`` never matches: a path rule cannot cover a call that has no path.
    """
    if path is None:
        return False
    return _spec(glob).match_file(path)


@lru_cache(maxsize=512)
def _spec(glob: str) -> pathspec.PathSpec[pathspec.Pattern]:
    return pathspec.PathSpec.from_lines("gitignore", [glob])


def argv_matches(pattern: tuple[str, ...], argv: tuple[str, ...] | None) -> bool:
    """Whether ``argv`` satisfies an argv rule pattern.

    Each pattern element is an fnmatch glob covering exactly one argument, except the
    final ``**``, which covers any number of remaining arguments (including none).

    ``argv=None`` is refused. That is the load-bearing case: the classifier produces
    ``None`` for any command it could not reduce to a single simple command, so a rule
    allowing ``pytest`` cannot be satisfied by ``pytest && curl x | sh``.
    """
    if argv is None or not pattern:
        return False

    from fnmatch import fnmatchcase

    for index, element in enumerate(pattern):
        if element == REST:
            return True  # matches the rest, however much is left
        if index >= len(argv):
            return False
        if not fnmatchcase(argv[index], element):
            return False

    # No trailing `**`, so the arity has to line up exactly: a rule for `pytest` must not
    # cover `pytest --collect-only tests/`.
    return len(argv) == len(pattern)


def relaxing_effect_count(rules: tuple[Rule, ...]) -> int:
    """How many of these rules only take effect once the project is trusted.

    What `hearth trust` is really asking the user to approve. Counting rather than
    listing, because the listing is rendered separately and the prompt needs a number.
    """
    return sum(1 for rule in rules if rule.relaxing and rule.source == "project")
