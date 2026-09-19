"""Turning a command string into facts a human and the policy engine can both judge.

Pure: no filesystem, no process, no clock. It takes a string and returns a
:class:`Classification`, which a tool's ``prepare()`` copies into ``PolicyFacts``.

**What this module is for.** Until OS sandboxing arrives in Phase 3, subprocess execution
is *not* a security boundary — the approval is (docs/safety-and-tool-use.md §1.2).
Approving `npm test` runs whatever the test scripts do, and no amount of parsing changes
that. So the classifier is not trying to make arbitrary commands safe. It has two narrower
jobs, and both are achievable:

1. **Stop an allow rule being satisfied by something the user did not mean.** Anything that
   does not reduce to a single simple command yields ``argv=None``, and
   ``rules.argv_matches`` refuses ``None`` outright. That is the entire mechanism behind
   "a rule for `pytest` is useless to `pytest; rm -rf ~`" (§5.4).
2. **Make the preview honest.** A command needing a shell gets `SHELL`, because `sh -c` and
   `exec` are different executions and the difference is the user's to see.

**Why it is deliberately over-cautious.** Every judgement call here resolves toward
refusing to parse. A classifier that is clever about which compound commands are "probably
fine" is a classifier that can be argued into something; one that refuses all of them costs
the user an extra approval and nothing else. Concretely: unquoted globs force a shell, an
unbalanced quote is refused rather than guessed, and the hard-deny list is applied to every
segment *and* to the raw string, because `pytest && sudo rm -rf /` must not present itself
as an ordinary `pytest` ask.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

#: §8.1: reject anything larger. A command this long is not something a person is reading
#: in an approval panel, so there is nothing safe to show them.
MAX_COMMAND_BYTES = 8 * 1024

#: More than this many paths passed to a removal command makes it DESTRUCTIVE.
_BULK_REMOVE_PATHS = 3

#: Directories whose executables are "normal". Anything else earns `OUTSIDE-PATH`, because
#: `/tmp/pytest` is not `pytest` (§5.4, path hijack).
_SYSTEM_BIN_PREFIXES = ("/usr/bin", "/usr/local/bin", "/bin", "/sbin", "/usr/sbin", "/opt")

#: Metacharacters that mean a shell is required. Detected outside quotes only.
_SHELL_METACHARACTERS = frozenset(";&|<>\n(){}")

#: Glob characters. Unquoted, they need a shell to mean what the reader thinks.
_GLOB_CHARACTERS = frozenset("*?[")

#: Flags that make an interpreter run a string instead of a file (§5.4).
_INLINE_CODE_FLAGS: dict[str, tuple[str, ...]] = {
    "python": ("-c",),
    "python2": ("-c",),
    "python3": ("-c",),
    "node": ("-e", "--eval", "-p", "--print"),
    "deno": ("eval",),
    "bun": ("-e", "--eval"),
    "bash": ("-c",),
    "sh": ("-c",),
    "zsh": ("-c",),
    "ksh": ("-c",),
    "dash": ("-c",),
    "perl": ("-e", "-E"),
    "ruby": ("-e",),
    "php": ("-r",),
    "powershell.exe": ("-command", "-c", "-encodedcommand", "-file"),
    "pwsh": ("-command", "-c", "-encodedcommand", "-file"),
    "cmd.exe": ("/c", "/k"),
    "wsl.exe": ("-e", "--exec"),
}

_READ_ONLY = frozenset(
    {"ls", "cat", "head", "tail", "wc", "pwd", "echo", "tree", "rg", "grep", "file", "stat", "du", "df"}
)
_BUILD_TEST = frozenset(
    {
        "pytest",
        "tox",
        "nox",
        "jest",
        "vitest",
        "mocha",
        "tsc",
        "ruff",
        "mypy",
        "pyright",
        "eslint",
        "prettier",
        "black",
        "isort",
        "flake8",
        "gofmt",
        "gradle",
        "mvn",
        "dotnet",
    }
)
_PACKAGE_NETWORK = frozenset(
    {
        "pip",
        "pip3",
        "pipx",
        "npm",
        "pnpm",
        "yarn",
        "bun",
        "npx",
        "curl",
        "wget",
        "ssh",
        "scp",
        "rsync",
        "docker",
        "podman",
    }
)
_INTERPRETERS = frozenset({"python", "python2", "python3", "node", "deno", "bun", "ruby", "perl", "php"})
_OPAQUE_RUNNERS = frozenset({"make", "just", "task", "rake", "invoke", "mage"})

#: Tools whose family depends on the subcommand. `cargo build` is local; `cargo install`
#: fetches from the network. Treating the whole executable as one family would either badge
#: every build as NETWORK? or let an install through unbadged.
_SUBCOMMAND_NETWORK: dict[str, frozenset[str]] = {
    "uv": frozenset({"add", "remove", "sync", "lock", "pip", "tool", "python", "publish", "venv"}),
    "cargo": frozenset({"install", "add", "publish", "update", "fetch", "search", "login"}),
    "go": frozenset({"get", "install", "mod", "download"}),
    "poetry": frozenset({"add", "install", "update", "lock", "publish"}),
    "gem": frozenset({"install", "update", "push"}),
    "brew": frozenset({"install", "update", "upgrade", "tap"}),
    "apt": frozenset({"install", "update", "upgrade"}),
    "apt-get": frozenset({"install", "update", "upgrade"}),
}

_SUBCOMMAND_BUILD: dict[str, frozenset[str]] = {
    "uv": frozenset({"run"}),
    "cargo": frozenset({"build", "test", "check", "clippy", "fmt", "run", "bench"}),
    "go": frozenset({"build", "test", "vet", "fmt", "run"}),
    "poetry": frozenset({"run"}),
}

#: Windows executables that change system state. Hard denied (§8.1).
_WINDOWS_DENIED = frozenset(
    {
        "reg.exe",
        "schtasks.exe",
        "sc.exe",
        "netsh.exe",
        "bcdedit.exe",
        "vssadmin.exe",
        "wmic.exe",
        "diskpart.exe",
    }
)

#: Windows executables reachable from WSL2 without a `.exe` suffix or an `/mnt/` path.
#: `pwsh` is the one that matters: it is a real Windows shell on PATH inside WSL.
_WINDOWS_INTEROP_NAMES = frozenset({"pwsh", "powershell", "cmd", "wslview", "explorer"})

#: Privilege escalation (§5.2 invariant 4).
_PRIVILEGE = frozenset({"sudo", "su", "doas", "pkexec", "runas", "runas.exe"})

#: Whole-system commands (§5.2 invariant 7).
_SYSTEM_DESTRUCTION = frozenset(
    {
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init",
        "crontab",
        "launchctl",
        "systemctl",
        "service",
        "chpasswd",
        "useradd",
        "usermod",
    }
)

#: git subcommands that reach the network or rewrite history (§5.2 invariant 5).
_GIT_DENIED_SUBCOMMANDS = frozenset(
    {
        "push",
        "pull",
        "fetch",
        "clone",
        "rebase",
        "filter-branch",
        "filter-repo",
        "update-ref",
        "reflog",
        "daemon",
        "request-pull",
        "send-email",
        "submodule",
    }
)

#: `git remote` is only denied for the subcommands that add a destination. Reading remotes
#: is harmless, and denying `git remote -v` would be noise.
_GIT_REMOTE_DENIED = frozenset({"add", "set-url", "set-branches", "rename"})

#: git subcommands that discard work without touching the network.
_GIT_DESTRUCTIVE = frozenset({"checkout", "restore", "stash", "branch", "reset", "clean"})

#: Regexes over the *raw* string, for shapes that survive segmentation.
_RAW_HARD_DENY: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "a download piped into an interpreter runs code nobody reviewed",
        re.compile(
            r"\b(?:curl|wget|iwr|invoke-webrequest)\b[^|\n]{0,400}\|\s*(?:sudo\s+)?"
            r"(?:(?:ba|z|k|da)?sh|python3?|perl|ruby|node)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "this would delete the filesystem root or your home directory",
        re.compile(
            r"\brm\b[^\n]{0,80}?\s(?:-[a-zA-Z]*[rf][a-zA-Z]*\s+)+"
            r"(?:/|~|/\*|\"?\$\{?HOME\}?\"?|'\$HOME')\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "writing to a shell startup file would run this on every future shell",
        re.compile(
            r">>?\s*~?/?(?:\.bashrc|\.zshrc|\.profile|\.bash_profile|\.zprofile|\.kshrc)\b", re.IGNORECASE
        ),
    ),
    (
        "writing directly to a block device destroys the filesystem on it",
        re.compile(r"\bdd\b[^\n]{0,200}\bof=/dev/", re.IGNORECASE),
    ),
    (
        "an encoded PowerShell command cannot be reviewed before it runs",
        re.compile(r"\b(?:powershell(?:\.exe)?|pwsh)\b[^\n]{0,200}-encodedcommand\b", re.IGNORECASE),
    ),
    (
        "this would unregister or shut down the WSL distribution",
        re.compile(r"\bwsl(?:\.exe)?\b[^\n]{0,80}--(?:unregister|shutdown|terminate)\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class Classification:
    """What a command is, as far as static inspection can tell."""

    raw: str
    #: The parsed argv, or None when the string did not reduce to one simple command.
    #: None is the value that makes an allow rule unsatisfiable (§5.4).
    argv: tuple[str, ...] | None
    #: Leading ``VAR=value`` assignments, separated out so a rule cannot match on them.
    env_prefix: tuple[tuple[str, str], ...] = ()
    #: Best-effort per-segment argv, for display and deny checks only — never for matching.
    segments: tuple[tuple[str, ...], ...] = ()
    shell: bool = False
    families: frozenset[str] = frozenset()
    badges: tuple[str, ...] = ()
    destructive: bool = False
    network_likely: bool = False
    inline_code: bool = False
    opaque: bool = False
    #: Set when a hard invariant refuses this outright. The message is shown to the user.
    hard_denied: str | None = None
    #: Set when the string could not be parsed at all: NUL, oversize, unbalanced quotes.
    refused: str | None = None

    @property
    def executable(self) -> str | None:
        """The command's basename, normalised. ``/tmp/pytest`` -> ``pytest``."""
        if self.argv:
            return PurePosixPath(self.argv[0]).name
        if self.segments:
            return PurePosixPath(self.segments[0][0]).name
        return None


def classify(command: str) -> Classification:
    """Inspect a command string. Never raises."""
    if "\x00" in command:
        return _refused(command, "the command contains a NUL byte")
    if len(command.encode("utf-8", errors="replace")) > MAX_COMMAND_BYTES:
        return _refused(command, f"the command is longer than {MAX_COMMAND_BYTES} bytes")
    if not command.strip():
        return _refused(command, "the command is empty")

    scan = _scan(command)
    segments = _split_segments(command, shell=scan.shell)

    # Hard denies are checked on the raw string and on every segment. Checking only the
    # first segment would let `pytest && sudo rm -rf /` present itself as a pytest ask.
    hard_denied = _raw_hard_deny(command) or _segment_hard_deny(segments)

    if scan.refused is not None:
        return _refused(command, scan.refused, segments=segments, hard_denied=hard_denied)

    badges: list[str] = ["EXEC"]
    families: set[str] = set()
    destructive = False
    network = False
    inline = False
    opaque = False
    outside_path = False
    win_interop = False

    for segment in segments:
        facts = _classify_segment(segment)
        families |= facts.families
        destructive = destructive or facts.destructive
        network = network or facts.network
        inline = inline or facts.inline_code
        opaque = opaque or facts.opaque
        outside_path = outside_path or facts.outside_path
        win_interop = win_interop or facts.win_interop

    if scan.shell:
        badges.append("SHELL")
    if destructive:
        badges.append("DESTRUCTIVE")
    if network:
        badges.append("NETWORK?")
    if inline:
        badges.append("INLINE-CODE")
    if opaque:
        badges.append("OPAQUE")
    if outside_path:
        badges.append("OUTSIDE-PATH")
    if win_interop:
        badges.append("WIN-INTEROP")

    # argv is offered only for a single simple command with nothing that needs a shell.
    # Inline code parses fine but must never satisfy a rule (§5.4), so it is withheld too.
    simple = not scan.shell and len(segments) == 1 and not inline
    argv = segments[0] if simple and segments else None

    return Classification(
        raw=command,
        argv=argv,
        env_prefix=scan.env_prefix,
        segments=segments,
        shell=scan.shell,
        families=frozenset(families),
        badges=tuple(badges),
        destructive=destructive,
        network_likely=network,
        inline_code=inline,
        opaque=opaque,
        hard_denied=hard_denied,
    )


# ------------------------------------------------------------------- scanning


@dataclass(frozen=True)
class _Scan:
    shell: bool
    env_prefix: tuple[tuple[str, str], ...]
    refused: str | None


def _scan(command: str) -> _Scan:
    """Walk the string tracking quote state, looking for anything needing a shell.

    Quote state matters for correctness in both directions. `grep 'a;b' src` must stay
    allowlistable, so a quoted `;` is not a metacharacter — but `"$HOME"` *does* expand
    inside double quotes, so `$` is special there. Getting this backwards either makes
    rules useless or lets an expansion through.
    """
    state: str | None = None
    needs_shell = False
    index = 0

    while index < len(command):
        char = command[index]

        if state == "'":
            state = None if char == "'" else state
        elif state == '"':
            if char == '"':
                state = None
            elif char in "$`":
                needs_shell = True  # expansion happens inside double quotes
            elif char == "\\":
                index += 1
        elif char in ("'", '"'):
            state = char
        elif char == "\\":
            index += 1
        elif char in _SHELL_METACHARACTERS or char in "$`":
            needs_shell = True
        elif char in _GLOB_CHARACTERS:
            # A glob only means what the reader thinks if a shell expands it; passing
            # `*.py` literally to exec would run something different from the preview.
            needs_shell = True

        index += 1

    if state is not None:
        return _Scan(shell=needs_shell, env_prefix=(), refused="the command has an unbalanced quote")

    return _Scan(shell=needs_shell, env_prefix=_env_prefix(command), refused=None)


_ENV_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _env_prefix(command: str) -> tuple[tuple[str, str], ...]:
    """Leading ``VAR=value`` assignments, which never satisfy an allow rule (§5.4)."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ()

    found: list[tuple[str, str]] = []
    for token in tokens:
        match = _ENV_ASSIGNMENT.match(token)
        if match is None:
            break
        found.append((match.group(1), match.group(2)))
    return tuple(found)


def _split_segments(command: str, *, shell: bool) -> tuple[tuple[str, ...], ...]:
    """Best-effort split into per-command argvs, for display and deny checks only.

    Never used for rule matching: that only ever consults ``argv``, which is None whenever
    this produced more than one segment or needed a shell at all.
    """
    pieces = re.split(r"\|\||&&|[;|&\n]", command) if shell else [command]
    segments: list[tuple[str, ...]] = []

    for piece in pieces:
        text = piece.strip()
        if not text:
            continue
        try:
            tokens = shlex.split(text)
        except ValueError:
            # Unbalanced quotes in one segment: fall back to whitespace so the deny
            # checks still see the words. Refusing to look would be worse.
            tokens = text.split()
        tokens = [token for token in tokens if _ENV_ASSIGNMENT.match(token) is None] or tokens
        if tokens:
            segments.append(tuple(tokens))

    return tuple(segments)


# ---------------------------------------------------------------- per segment


@dataclass(frozen=True)
class _SegmentFacts:
    families: set[str]
    destructive: bool = False
    network: bool = False
    inline_code: bool = False
    opaque: bool = False
    outside_path: bool = False
    win_interop: bool = False


def _classify_segment(argv: tuple[str, ...]) -> _SegmentFacts:
    executable = PurePosixPath(argv[0]).name.lower()
    rest = [token.lower() for token in argv[1:]]
    families: set[str] = set()

    win_interop = (
        executable.endswith(".exe") or argv[0].startswith("/mnt/") or executable in _WINDOWS_INTEROP_NAMES
    )
    outside_path = _is_outside_path(argv[0])

    if executable in _PRIVILEGE:
        families.add("privilege")
    if executable in _SYSTEM_DESTRUCTION or executable.startswith("mkfs"):
        families.add("system")
    if executable in _READ_ONLY:
        families.add("read-only")
    if executable in _BUILD_TEST:
        families.add("build/test")
    if executable in _OPAQUE_RUNNERS or (executable in {"npm", "pnpm", "yarn"} and "run" in rest):
        families.add("opaque")
    if executable in _INTERPRETERS:
        families.add("interpreter")
    if executable in _PACKAGE_NETWORK:
        families.add("package/network")

    subcommand = rest[0] if rest else ""
    if subcommand in _SUBCOMMAND_NETWORK.get(executable, frozenset()):
        families.add("package/network")
    if subcommand in _SUBCOMMAND_BUILD.get(executable, frozenset()):
        families.add("build/test")

    network = (
        executable in _PACKAGE_NETWORK and not _is_local_only(executable, rest)
    ) or subcommand in _SUBCOMMAND_NETWORK.get(executable, frozenset())
    inline = _is_inline_code(executable, argv[1:])
    destructive = _is_destructive(executable, argv[1:])

    if executable == "git":
        if subcommand in {"fetch", "pull", "push", "clone"}:
            families.add("package/network")
            network = True
        elif subcommand in {"status", "diff", "log", "show", "blame"}:
            families.add("read-only")
        if subcommand in _GIT_DESTRUCTIVE:
            destructive = destructive or _git_is_destructive(subcommand, rest[1:])

    return _SegmentFacts(
        families=families,
        destructive=destructive,
        network=network,
        inline_code=inline,
        opaque="opaque" in families,
        outside_path=outside_path,
        win_interop=win_interop,
    )


def _is_outside_path(executable: str) -> bool:
    """Whether an absolute executable lives outside the usual system directories."""
    if not executable.startswith("/"):
        return False
    if executable.startswith("/mnt/"):
        return False  # reported as WIN-INTEROP instead, which is the stronger signal
    return not any(executable.startswith(prefix + "/") for prefix in _SYSTEM_BIN_PREFIXES)


def _is_local_only(executable: str, rest: list[str]) -> bool:
    """Whether a package-manager invocation is plainly local."""
    if executable == "docker":
        return not any(word in rest for word in ("pull", "push", "run", "build"))
    if executable in {"npm", "pnpm", "yarn", "bun"}:
        return bool(rest) and rest[0] in {"run", "test", "ls", "list", "why"}
    return False


def _is_inline_code(executable: str, rest: tuple[str, ...]) -> bool:
    flags = _INLINE_CODE_FLAGS.get(executable)
    if not flags:
        return False
    return any(token.lower() in flags for token in rest)


_REMOVE_COMMANDS = frozenset({"rm", "rmdir", "shred", "truncate", "unlink"})


def _is_destructive(executable: str, rest: tuple[str, ...]) -> bool:
    """§8.1's destructive family.

    A single `rm build/stale.o` is not destructive. Badging every removal would make the
    badge meaningless, and DESTRUCTIVE carries a typed confirmation — the most expensive
    prompt Hearth has (§6.3).
    """
    if executable in {"shred", "truncate"}:
        return True
    if executable in _REMOVE_COMMANDS:
        flags = [token for token in rest if token.startswith("-")]
        paths = [token for token in rest if not token.startswith("-")]
        recursive = any("r" in flag.lower() or "f" in flag.lower() for flag in flags if flag != "-")
        return recursive or len(paths) > _BULK_REMOVE_PATHS
    if executable == "find":
        lowered = [token.lower() for token in rest]
        return "-delete" in lowered or "-exec" in lowered
    if executable == "mv":
        return "-f" in rest
    return False


def _git_is_destructive(subcommand: str, rest: list[str]) -> bool:
    match subcommand:
        case "checkout":
            return "--" in rest or "." in rest
        case "restore":
            return True
        case "stash":
            return bool(rest) and rest[0] in {"drop", "clear"}
        case "branch":
            return any(flag in rest for flag in ("-d", "-D", "--delete"))
        case "reset" | "clean":
            return True
        case _:
            return False


# ------------------------------------------------------------- hard denials


def _raw_hard_deny(command: str) -> str | None:
    for reason, pattern in _RAW_HARD_DENY:
        if pattern.search(command):
            return reason
    return None


def _segment_hard_deny(segments: tuple[tuple[str, ...], ...]) -> str | None:
    for argv in segments:
        executable = PurePosixPath(argv[0]).name.lower()
        rest = [token.lower() for token in argv[1:]]

        if executable in _PRIVILEGE:
            return f"{executable} is never permitted; run it yourself if you mean to"
        if executable in _WINDOWS_DENIED:
            return f"{executable} modifies Windows system state and is never permitted"
        if executable in _SYSTEM_DESTRUCTION or executable.startswith("mkfs"):
            return f"{executable} changes the whole system and is never permitted"
        if executable == "git" and rest:
            if rest[0] == "remote" and len(rest) > 1 and rest[1] in _GIT_REMOTE_DENIED:
                return (
                    "`git remote add` points the repository at a new destination, which is "
                    "how code leaves the machine. Run it yourself if you mean to"
                )
            if rest[0] in _GIT_DENIED_SUBCOMMANDS:
                return (
                    f"`git {rest[0]}` is deliberately unavailable — it reaches the network or "
                    "rewrites history. Run it yourself if you mean to"
                )
            if rest[0] == "reset" and "--hard" in rest:
                return "`git reset --hard` discards work irreversibly and is never permitted"
            if rest[0] == "clean" and any(flag.startswith("-f") for flag in rest):
                return "`git clean -f` deletes untracked files irreversibly and is never permitted"
            if rest[0] == "commit" and "--amend" in rest:
                return "`git commit --amend` rewrites history; Hearth never amends"
    return None


def _refused(
    command: str,
    reason: str,
    *,
    segments: tuple[tuple[str, ...], ...] = (),
    hard_denied: str | None = None,
) -> Classification:
    """A command that could not be parsed. ``argv`` is None, so no rule can match it."""
    return Classification(  # noqa: S604 - `shell` is a Classification field, not a subprocess call
        raw=command,
        argv=None,
        segments=segments,
        shell=True,
        badges=("EXEC", "SHELL"),
        refused=reason,
        hard_denied=hard_denied,
    )
