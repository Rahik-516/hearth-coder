"""Building the environment an agent-run command gets.

An **allowlist, not a denylist** (docs/safety-and-tool-use.md §8.2). That direction is the
whole point: a denylist has to anticipate every credential variable anyone will ever
invent, and the one it misses is the one that leaks. An allowlist fails the other way —
a build that needs `JAVA_HOME` breaks until someone adds it to config, which is visible,
fixable, and not a disclosure.

Two rules that look redundant and are not:

* **The scrub patterns are applied last, after every allowance.** So a project config
  cannot allowlist `AWS_SECRET_ACCESS_KEY` back in, and neither can a `VAR=value` prefix on
  the command. Config supplies values, never permissions (§5.6).
* **The non-interactive variables are forced, not defaulted.** `PAGER=cat` and
  `GIT_TERMINAL_PROMPT=0` are the difference between a command that fails in two seconds
  and one that hangs for the full timeout while the user wonders what is happening.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

#: Variables a build or test run legitimately needs.
BASE_ALLOWED: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "TZ",
        "TMPDIR",
        "PWD",
        # Toolchain locations that are locations, not credentials.
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PYENV_ROOT",
        "NVM_DIR",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "GOPATH",
        "GOROOT",
        "GOCACHE",
        "GOMODCACHE",
        "UV_CACHE_DIR",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_CONFIG_HOME",
        "SYSTEMROOT",
        "WSLENV",
        "WSL_DISTRO_NAME",
    }
)

#: Also allowed: anything matching these prefixes. `LC_*` is a family, not a name.
_ALLOWED_PREFIXES: tuple[str, ...] = ("LC_",)

#: Credential-shaped names, scrubbed even when otherwise allowed (§8.2).
#:
#: Overlapping with the allowlist on purpose. These run *after* every allowance, so no
#: later decision can re-admit one.
SCRUB_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i).*(?:^|_)TOKEN(?:$|_).*"),
    re.compile(r"(?i).*(?:^|_)SECRET(?:$|_).*"),
    re.compile(r"(?i).*(?:^|_)PASSWORD(?:$|_).*"),
    re.compile(r"(?i).*PASSWD.*"),
    re.compile(r"(?i).*API_?KEY.*"),
    re.compile(r"(?i).*(?:^|_)KEY(?:$|_).*"),
    re.compile(r"(?i).*CREDENTIAL.*"),
    re.compile(r"(?i).*(?:^|_)AUTH(?:$|_).*"),
    re.compile(r"(?i)^AWS_.*"),
    re.compile(r"(?i)^GITHUB_.*"),
    re.compile(r"(?i)^GITLAB_.*"),
    re.compile(r"(?i)^OPENAI_.*"),
    re.compile(r"(?i)^ANTHROPIC_.*"),
    re.compile(r"(?i)^AZURE_.*"),
    re.compile(r"(?i)^GOOGLE_.*"),
    re.compile(r"(?i)^GCP_.*"),
    re.compile(r"(?i)^NPM_.*"),
    re.compile(r"(?i)^PYPI_.*"),
    re.compile(r"(?i)^TWINE_.*"),
    re.compile(r"(?i)^DOCKER_.*"),
    re.compile(r"(?i)^SLACK_.*"),
    re.compile(r"(?i)^STRIPE_.*"),
    re.compile(r"(?i)^SSH_AUTH_SOCK$"),
    re.compile(r"(?i)^GPG_AGENT_INFO$"),
    re.compile(r"(?i)^GNUPGHOME$"),
)

#: Forced into every child, so nothing waits for a human that is not there (§8.2).
NON_INTERACTIVE: dict[str, str] = {
    "CI": "1",
    "TERM": "dumb",
    "NO_COLOR": "1",
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "GCM_INTERACTIVE": "never",
    "PIP_NO_INPUT": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "DEBIAN_FRONTEND": "noninteractive",
    "HOMEBREW_NO_AUTO_UPDATE": "1",
    "NPM_CONFIG_FUND": "false",
    "NPM_CONFIG_AUDIT": "false",
    "NPM_CONFIG_UPDATE_NOTIFIER": "false",
}


def build_environment(
    base: Mapping[str, str],
    *,
    extra_allowed: Iterable[str] = (),
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment for one command.

    Args:
        base: Where to draw from — normally ``os.environ``.
        extra_allowed: Additional names to keep, from the project's toolchain config.
        overrides: Values to set, e.g. a ``VAR=value`` prefix parsed off the command.

    Returns:
        A new mapping. Scrubbing is applied after everything else, so neither an extra
        allowance nor an override can reintroduce a credential.
    """
    allowed = BASE_ALLOWED | {name for name in extra_allowed}

    env = {
        name: value for name, value in base.items() if name in allowed or name.startswith(_ALLOWED_PREFIXES)
    }
    env.update(overrides or {})
    env.update(NON_INTERACTIVE)

    # Last, and deliberately unconditional.
    return {name: value for name, value in env.items() if not is_sensitive_name(name)}


def is_sensitive_name(name: str) -> bool:
    """Whether a variable name looks like a credential."""
    return any(pattern.fullmatch(name) or pattern.match(name) for pattern in SCRUB_PATTERNS)


def scrubbed_names(base: Mapping[str, str]) -> tuple[str, ...]:
    """Which variables were removed, so the approval panel can say "3 secrets removed"."""
    return tuple(sorted(name for name in base if is_sensitive_name(name)))
