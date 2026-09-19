"""The policy engine — a pure decision function.

**Invariant: this module performs no I/O.** No filesystem, no network, no clock, no
database. It takes (request, session view, config view) and returns a decision. That is
what makes it exhaustively testable with tables and Hypothesis, and it is enforced by the
``policy-pure`` contract in ``.importlinter``: importing ``hearth.storage``,
``hearth.llm``, ``hearth.tools`` or ``hearth.git`` from here fails CI.

That contract is also why the engine never sees a ``Tool`` object. Callers describe a
prepared call as :class:`PolicyFacts` — the resolved path, the parsed argv, the
classification flags — and the gateway is the adapter that builds it. The indirection pays
for itself twice: policy cannot accidentally reach into a tool to re-derive a fact, and a
test can express "a destructive write to a protected path" in one line without a
filesystem.

Evaluation order, first decisive result wins (docs/safety-and-tool-use.md §5.1):

1. Hard invariants          -> Deny (never configurable)
2. Mode restrictions        -> Deny
3. User DENY rules          -> Deny
4. Classification overlays  -> Destructive: Ask+typed, or Deny when headless
5. Session grants           -> Allow (exact grant keys only)
6. User ALLOW rules         -> Allow (project rules only when the project is trusted)
7. User ASK rules           -> Ask
8. Permission-level default -> per the table in docs/safety-and-tool-use.md §4.1

Two rules govern every change here, and both are checked by property tests rather than
trusted to review:

* **It fails closed.** Every path out of :func:`evaluate` that is not an explicit Allow is
  an Ask or a Deny, and in headless mode an Ask becomes a Deny — because there is no
  channel to ask on, and an Ask that survived would either hang or be read as consent.
* **The agent can never raise its own privileges.** Invariants run first, so no rule,
  grant or permission level can authorise a write to git internals, to Hearth's own
  config, or to anywhere outside the workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from hearth.safety.invariants import check_command, check_read_path, check_write_path
from hearth.safety.risk import AGENT_ONLY, SIDE_EFFECT_FREE, Risk
from hearth.safety.rules import Rule, argv_matches, path_matches

Action = Literal["allow", "ask", "deny"]

#: Paths `auto-edit` still asks about (docs/safety-and-tool-use.md §4.1).
#:
#: The common thread is blast radius beyond the file: CI config decides what runs on every
#: push, a lockfile decides what code is installed, a migration is usually irreversible,
#: and `AGENTS.md` is an instruction file the agent would be editing for its own future
#: self.
AUTO_EDIT_ASK_GLOBS: tuple[str, ...] = (
    ".github/workflows/**",
    ".gitlab-ci.yml",
    ".gitlab-ci.yaml",
    "**/migrations/**",
    "Dockerfile*",
    "docker-compose*.y*ml",
    "*.tf",
    "*.tfvars",
    "AGENTS.md",
    "HEARTH.md",
    "CLAUDE.md",
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "Cargo.lock",
    "go.sum",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "*.p12",
    "*.pfx",
)

#: The word a user types to confirm a destructive action (§6.3).
DESTRUCTIVE_CONFIRMATION = "confirm"

#: Badges that make a call ineligible for "always for this session" (§6.1).
UNGRANTABLE_BADGES = frozenset({"DESTRUCTIVE", "SHELL", "NETWORK?", "WIN-INTEROP", "INLINE-CODE"})


@dataclass(frozen=True)
class PolicyFacts:
    """What a prepared call tells the engine about itself.

    Produced by ``prepare()`` and passed through unchanged. Nothing here is re-derived by
    the engine: if the diff the user approved was computed from one resolved path, the
    decision has to be made about that same path, not a second resolution of the same
    string.
    """

    #: Resolved, workspace-relative POSIX path. None when the call has no path.
    path: str | None = None
    #: Resolved absolute path, POSIX-style. Needed for the Hearth-own-config invariant.
    absolute_path: str | None = None
    inside_workspace: bool = True
    sensitive_read: bool = False
    #: Parsed argv, or None when the command did not reduce to one simple command.
    argv: tuple[str, ...] | None = None
    #: Leading ``VAR=value`` assignments. Present so a rule can refuse to match on them:
    #: `FOO=1 pytest` must not satisfy a rule for `pytest` unless the rule says `env=true`
    #: (docs/safety-and-tool-use.md §5.4). The environment is part of what a command *is*.
    env_prefix: tuple[tuple[str, str], ...] = ()
    #: Classification flags, filled in by the command classifier in M6.
    shell: bool = False
    destructive: bool = False
    network_likely: bool = False
    hard_denied: str | None = None
    badges: tuple[str, ...] = ()
    #: The exact session-grant key this call would match or create (§5.4).
    grant_key: str | None = None


@dataclass(frozen=True)
class PolicyRequest:
    """One prepared tool call, as the engine sees it."""

    tool: str
    risk: Risk
    facts: PolicyFacts = field(default_factory=PolicyFacts)


@dataclass(frozen=True)
class SessionView:
    """The part of the session that affects decisions.

    A view rather than the session itself, so the engine cannot read history, touch the
    provider, or mutate anything.
    """

    mode: str = "agent"
    level: str = "supervised"
    headless: bool = False
    grants: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ConfigView:
    """The part of configuration that affects decisions."""

    rules: tuple[Rule, ...] = ()
    #: Whether `<repo>/.hearth/config.toml` has been trusted at its current bytes (§5.5).
    project_trusted: bool = False
    #: Absolute directories that would let the agent change its own permissions.
    protected_dirs: tuple[str, ...] = ()
    auto_edit_ask: tuple[str, ...] = AUTO_EDIT_ASK_GLOBS
    #: Headless widening flags (§14). Narrow and explicit, never implied.
    headless_allow_edits: bool = False
    headless_allow_tests: bool = False
    headless_allow_commit: bool = False


@dataclass(frozen=True)
class Decision:
    """A policy outcome, carrying why — not just what."""

    action: Action
    reason: str = ""
    badges: tuple[str, ...] = ()
    #: Offered to the frontend as "always for this session". None means not grantable.
    grant_key: str | None = None
    rule_id: str | None = None
    #: invariant | mode | rule | overlay | grant | default | headless
    decided_by: str = "default"
    #: Set for DESTRUCTIVE: the word the user must type (§6.3).
    typed_confirmation: str | None = None
    #: For headless denials: the flag that would have permitted this (§14.1).
    hint: str | None = None

    @property
    def allowed(self) -> bool:
        return self.action == "allow"


def evaluate(request: PolicyRequest, session: SessionView, config: ConfigView) -> Decision:
    """Decide whether one prepared call may run. Pure."""
    facts = request.facts
    badges = tuple(facts.badges)

    # --- 1. hard invariants, which nothing overrides ----------------------
    violation = check_command(argv=facts.argv, hard_denied=facts.hard_denied)
    if violation is None and request.risk is Risk.READ:
        violation = check_read_path(
            absolute_path=facts.absolute_path or "",
            sensitive=facts.sensitive_read,
        )
    if violation is None and _is_write(request.risk):
        if facts.absolute_path is None:
            # Fail closed. Every write has a resolved path, so a call without one is
            # malformed — and skipping the invariant check for it would let the call fall
            # through to the rule engine, where a `path = "**"` allow rule covers it.
            return Decision(
                action="deny",
                reason=f"{request.tool} did not resolve a target path",
                badges=badges,
                decided_by="invariant",
                rule_id="unresolved-write-path",
            )
        violation = check_write_path(
            relative_path=facts.path,
            absolute_path=facts.absolute_path,
            inside_workspace=facts.inside_workspace,
            protected_dirs=config.protected_dirs,
        )
    if violation is not None:
        return Decision(
            action="deny",
            reason=violation.reason,
            badges=badges,
            decided_by="invariant",
            rule_id=violation.invariant,
        )

    # --- 2. mode restrictions ---------------------------------------------
    if request.risk in AGENT_ONLY and session.mode != "agent":
        return Decision(
            action="deny",
            reason=f"{request.tool} is not available in {session.mode} mode",
            badges=badges,
            decided_by="mode",
        )

    # --- 3. user DENY rules -----------------------------------------------
    # Project deny rules apply whether or not the project is trusted: they can only
    # tighten, and a repository is allowed to ask for more caution than the user's
    # defaults (§5.5).
    denied = _first_match(config.rules, request, effect="deny", trusted=True)
    if denied is not None:
        return Decision(
            action="deny",
            reason=denied.reason or f"denied by rule {denied.id}",
            badges=badges,
            decided_by="rule",
            rule_id=denied.id,
        )

    # --- 4. classification overlays ---------------------------------------
    if facts.destructive:
        badges = _with_badge(badges, "DESTRUCTIVE")
        if session.headless:
            return Decision(
                action="deny",
                reason="destructive operations are always denied in headless mode",
                badges=badges,
                decided_by="overlay",
            )
        return Decision(
            action="ask",
            reason="this operation is destructive or irreversible",
            badges=badges,
            decided_by="overlay",
            typed_confirmation=DESTRUCTIVE_CONFIRMATION,
        )

    # --- 5. session grants ------------------------------------------------
    if facts.grant_key and facts.grant_key in session.grants:
        return Decision(
            action="allow",
            reason=f"granted for this session: {facts.grant_key}",
            badges=badges,
            decided_by="grant",
            grant_key=facts.grant_key,
        )

    # --- 6. user ALLOW rules ----------------------------------------------
    allowed = _first_match(config.rules, request, effect="allow", trusted=config.project_trusted)
    if allowed is not None:
        return Decision(
            action="allow",
            reason=allowed.reason or f"allowed by rule {allowed.id}",
            badges=badges,
            decided_by="rule",
            rule_id=allowed.id,
        )

    # --- 7. user ASK rules ------------------------------------------------
    asked = _first_match(config.rules, request, effect="ask", trusted=True)
    if asked is not None:
        return _maybe_headless(
            Decision(
                action="ask",
                reason=asked.reason or f"rule {asked.id} requires confirmation",
                badges=badges,
                decided_by="rule",
                rule_id=asked.id,
                grant_key=_grantable(facts, badges),
            ),
            request,
            session,
            config,
        )

    # --- 8. permission-level default --------------------------------------
    return _maybe_headless(_default_decision(request, session, config, badges), request, session, config)


# ---------------------------------------------------------------- internals


def _default_decision(
    request: PolicyRequest, session: SessionView, config: ConfigView, badges: tuple[str, ...]
) -> Decision:
    """The §4.1 table."""
    facts = request.facts

    if request.risk in SIDE_EFFECT_FREE:
        if request.risk is Risk.META or facts.inside_workspace:
            return Decision(action="allow", reason="reads and session state are allowed", badges=badges)
        # Outside the workspace and not sensitive: the user may genuinely want a sibling
        # checkout read, so agent mode asks rather than refusing outright.
        if session.mode == "agent":
            return Decision(
                action="ask",
                reason="this path is outside the workspace",
                badges=_with_badge(badges, "OUTSIDE-WORKSPACE"),
            )
        return Decision(
            action="deny",
            reason=f"reading outside the workspace is not allowed in {session.mode} mode",
            badges=_with_badge(badges, "OUTSIDE-WORKSPACE"),
        )

    if request.risk is Risk.WRITE:
        if session.level == "auto-edit" and not _in_ask_set(facts.path, config.auto_edit_ask):
            return Decision(action="allow", reason="auto-edit allows workspace writes", badges=badges)
        return Decision(
            action="ask",
            reason="this writes to a file in your workspace",
            badges=badges,
            grant_key=_grantable(facts, badges),
        )

    # EXEC and VCS_WRITE both ask at every interactive level. They gain their allow paths
    # from user rules and grants, never from a level.
    return Decision(
        action="ask",
        reason="this has side effects outside the workspace files",
        badges=badges,
        grant_key=_grantable(facts, badges),
    )


def _maybe_headless(
    decision: Decision, request: PolicyRequest, session: SessionView, config: ConfigView
) -> Decision:
    """Map Ask to Deny when there is nobody to ask (§14).

    Applied as a final pass rather than inside each branch, so a new Ask added anywhere in
    the engine inherits the behaviour instead of quietly becoming a hang.
    """
    if not session.headless or decision.action != "ask":
        return decision

    flag = _headless_flag(request.risk)
    if flag is not None and _headless_permits(request.risk, config):
        return Decision(
            action="allow",
            reason=f"permitted by {flag}",
            badges=decision.badges,
            decided_by="headless",
        )

    return Decision(
        action="deny",
        reason="headless mode cannot ask for approval, so this is denied",
        badges=decision.badges,
        decided_by="headless",
        rule_id=decision.rule_id,
        hint=f"re-run with {flag}" if flag else None,
    )


def _headless_flag(risk: Risk) -> str | None:
    match risk:
        case Risk.WRITE:
            return "--allow-edits"
        case Risk.EXEC:
            return "--allow-tests"
        case Risk.VCS_WRITE:
            return "--allow-commit"
        case _:
            return None


def _headless_permits(risk: Risk, config: ConfigView) -> bool:
    match risk:
        case Risk.WRITE:
            return config.headless_allow_edits
        case Risk.EXEC:
            return config.headless_allow_tests
        case Risk.VCS_WRITE:
            return config.headless_allow_commit
        case _:
            return False


def _first_match(
    rules: tuple[Rule, ...], request: PolicyRequest, *, effect: str, trusted: bool
) -> Rule | None:
    """First rule of ``effect`` that covers this call, in declaration order.

    ``trusted`` gates *relaxing* rules from a project config. Passing ``trusted=True`` for
    deny and ask lookups is not a shortcut: those rules only tighten, so trust does not
    apply to them (§5.5).
    """
    for rule in rules:
        if rule.effect != effect:
            continue
        if rule.relaxing and rule.source == "project" and not trusted:
            continue
        if _rule_covers(rule, request):
            return rule
    return None


def _rule_covers(rule: Rule, request: PolicyRequest) -> bool:
    """Whether every condition the rule states is satisfied. Conditions narrow."""
    if not rule.matches_tool(request.tool):
        return False
    if rule.path_glob is not None and not path_matches(rule.path_glob, request.facts.path):
        return False
    if rule.argv is None:
        return True
    if not argv_matches(rule.argv, request.facts.argv):
        return False
    # A leading `FOO=bar` never satisfies an allow rule unless the rule opts in (§5.4).
    return rule.env or not _has_env_prefix(request.facts)


def _has_env_prefix(facts: PolicyFacts) -> bool:
    """Whether the command carried leading environment assignments.

    Reported by the classifier. A rule matches such a command only with ``env = true``,
    because `FOO=bar pytest` is not the same execution as `pytest` — the variable can
    change what the command does, and the rule's author did not approve it.
    """
    return bool(facts.env_prefix)


def _in_ask_set(path: str | None, globs: tuple[str, ...]) -> bool:
    return any(path_matches(glob, path) for glob in globs)


def _is_write(risk: Risk) -> bool:
    return risk is Risk.WRITE


def _with_badge(badges: tuple[str, ...], badge: str) -> tuple[str, ...]:
    return badges if badge in badges else (*badges, badge)


def _grantable(facts: PolicyFacts, badges: tuple[str, ...]) -> str | None:
    """The grant key to offer, or None when this call may never be granted.

    §6.1: "always for session" is not offered for SHELL, NETWORK? or DESTRUCTIVE. The
    engine decides that rather than the UI, so a future frontend cannot offer a grant the
    policy would not honour.
    """
    if facts.grant_key is None:
        return None
    if UNGRANTABLE_BADGES & set(badges):
        return None
    if facts.destructive or facts.shell or facts.network_likely:
        return None
    return facts.grant_key
