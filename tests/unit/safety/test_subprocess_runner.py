"""Environment scrubbing and process control — docs/safety-and-tool-use.md §8.2.

Written before ``safety/env.py`` and ``safety/sandbox/subprocess_runner.py``
(CLAUDE.md rule 3). This file is M6's second acceptance criterion: *"a command that waits
for stdin terminates at timeout, and the process group is killed, with no orphan
processes"*.

None of this is a security boundary — that arrives with bubblewrap in Phase 3. What it
buys is the L0 guarantees from §12: an approved command cannot hang the session, cannot
read credentials out of the environment, and cannot leave children running after it is
killed. Those are reliability properties, and they are the ones that actually bite during
a long agent run.

These tests spawn real processes. They are POSIX-specific by design: Hearth runs in WSL2,
and native Windows process-tree handling is explicitly deferred to Phase 3.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

from hearth.safety.env import SCRUB_PATTERNS, build_environment
from hearth.safety.sandbox.subprocess_runner import (
    HEAD_LINES,
    TAIL_LINES,
    CommandOutcome,
    SubprocessRunner,
    truncate_output,
)
from hearth.storage.blobs import BlobStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def runner(tmp_path: Path) -> SubprocessRunner:
    return SubprocessRunner(blobs=BlobStore(tmp_path / "blobs"))


# ------------------------------------------------------------------ environment


def test_the_allowlist_keeps_what_a_build_needs() -> None:
    env = build_environment({"PATH": "/usr/bin", "HOME": "/home/u", "LANG": "C.UTF-8"})

    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/u"
    assert env["LANG"] == "C.UTF-8"


@pytest.mark.parametrize(
    "name",
    [
        "GITHUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OLLAMA_API_KEY",
        "MY_DB_PASSWORD",
        "SOME_SECRET",
        "NPM_TOKEN",
        "SSH_AUTH_SOCK",
        "GPG_AGENT_INFO",
    ],
)
def test_credentials_are_scrubbed(name: str) -> None:
    """§8.2. A test suite has no business reading the user's tokens, and a compromised or
    confused one would exfiltrate them without any of Hearth's other controls noticing."""
    env = build_environment({"PATH": "/usr/bin", name: "sensitive-value"})

    assert name not in env


def test_an_unknown_variable_is_dropped_rather_than_passed() -> None:
    """Allowlist, not denylist. A new credential variable named something Hearth has never
    heard of must not reach the child by default."""
    env = build_environment({"PATH": "/usr/bin", "SOME_RANDOM_THING": "x"})

    assert "SOME_RANDOM_THING" not in env


def test_non_interactive_variables_are_forced() -> None:
    """§8.2: the difference between a command that fails in 2s and one that hangs for 120s.

    A pager waiting for a keypress, or git asking for a password, looks identical to a
    hung build from the outside.
    """
    env = build_environment({"PATH": "/usr/bin"})

    assert env["CI"] == "1"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["PAGER"] == "cat"
    assert env["GIT_PAGER"] == "cat"
    assert env["NO_COLOR"] == "1"
    assert env["TERM"] == "dumb"
    assert env["PIP_NO_INPUT"] == "1"
    assert env["DEBIAN_FRONTEND"] == "noninteractive"


def test_toolchain_variables_can_be_added_explicitly() -> None:
    """Projects legitimately need e.g. a JAVA_HOME. It has to be opt-in, from config."""
    env = build_environment({"PATH": "/usr/bin", "JAVA_HOME": "/opt/jdk"}, extra_allowed=("JAVA_HOME",))

    assert env["JAVA_HOME"] == "/opt/jdk"


def test_an_extra_allowance_cannot_re_admit_a_scrubbed_name() -> None:
    """Otherwise a project config could allowlist `AWS_SECRET_ACCESS_KEY` and the scrub
    would be advisory. Config supplies values, never permissions (§5.6)."""
    env = build_environment(
        {"PATH": "/usr/bin", "AWS_SECRET_ACCESS_KEY": "x"},
        extra_allowed=("AWS_SECRET_ACCESS_KEY",),
    )

    assert "AWS_SECRET_ACCESS_KEY" not in env


def test_the_scrub_patterns_are_not_empty() -> None:
    assert SCRUB_PATTERNS


def test_env_prefix_from_the_command_is_applied_on_top() -> None:
    """`FOO=bar pytest` has to actually set FOO, having been shown in the preview."""
    env = build_environment({"PATH": "/usr/bin"}, overrides={"FOO": "bar"})

    assert env["FOO"] == "bar"


def test_an_override_cannot_smuggle_a_credential_back_in() -> None:
    env = build_environment({"PATH": "/usr/bin"}, overrides={"GITHUB_TOKEN": "x"})

    assert "GITHUB_TOKEN" not in env


# ----------------------------------------------------------------- running


async def test_a_simple_command_runs_and_reports_its_output(runner: SubprocessRunner) -> None:
    outcome = await runner.run(argv=("echo", "hello"), cwd=Path.cwd())

    assert outcome.exit_code == 0
    assert "hello" in outcome.text
    assert outcome.timed_out is False


async def test_a_failing_command_reports_its_exit_code(runner: SubprocessRunner) -> None:
    outcome = await runner.run(argv=("false",), cwd=Path.cwd())

    assert outcome.exit_code == 1
    assert outcome.ok is False


async def test_stderr_is_captured_with_stdout(runner: SubprocessRunner) -> None:
    """A build failure usually speaks on stderr. Losing it would return an empty error."""
    outcome = await runner.run(
        argv=(sys.executable, "-c", "import sys; sys.stderr.write('boom')"), cwd=Path.cwd()
    )

    assert "boom" in outcome.text


async def test_a_missing_executable_is_reported_not_raised(runner: SubprocessRunner) -> None:
    outcome = await runner.run(argv=("hearth-does-not-exist",), cwd=Path.cwd())

    assert outcome.ok is False
    assert outcome.error is not None


# -------------------------------------------------------------------- stdin


async def test_a_command_reading_stdin_fails_fast_rather_than_hanging(
    runner: SubprocessRunner,
) -> None:
    """stdin is DEVNULL, so a prompt reads EOF immediately (§8.2).

    Without this the command waits the full timeout and the user stares at a stalled
    session for two minutes with no idea why.
    """
    outcome = await runner.run(argv=(sys.executable, "-c", "input()"), cwd=Path.cwd(), timeout_s=30)

    assert outcome.timed_out is False, "it should hit EOF, not the timeout"
    assert outcome.ok is False


# ------------------------------------------------------------------ timeouts


async def test_a_hanging_command_is_killed_at_the_timeout(runner: SubprocessRunner) -> None:
    outcome = await runner.run(argv=("sleep", "30"), cwd=Path.cwd(), timeout_s=1)

    assert outcome.timed_out is True
    assert outcome.ok is False
    assert outcome.duration_s < 20, "the escalation must not wait out the full ladder"


async def test_the_timeout_message_says_what_happened(runner: SubprocessRunner) -> None:
    outcome = await runner.run(argv=("sleep", "30"), cwd=Path.cwd(), timeout_s=1)

    assert "timed out" in outcome.text.lower() or "timed out" in (outcome.error or "").lower()


async def test_children_of_a_killed_command_do_not_survive(runner: SubprocessRunner) -> None:
    """The acceptance criterion. A killed shell whose `sleep` keeps running is an orphan
    holding a file lock or a port, and nothing will ever clean it up.

    `start_new_session=True` puts the command in its own process group, and the kill goes
    to the group rather than the direct child.
    """
    script = (
        "import subprocess, sys, time;"
        "p = subprocess.Popen(['sleep', '300']);"
        "sys.stdout.write(str(p.pid) + '\\n'); sys.stdout.flush();"
        "time.sleep(300)"
    )
    outcome = await runner.run(argv=(sys.executable, "-u", "-c", script), cwd=Path.cwd(), timeout_s=2)

    assert outcome.timed_out is True

    child_pid = int(outcome.text.strip().splitlines()[0])
    with pytest.raises(ProcessLookupError):
        # Signal 0 only checks existence. If the grandchild survived, this succeeds.
        os.kill(child_pid, 0)


async def test_a_command_that_ignores_sigint_is_still_killed(runner: SubprocessRunner) -> None:
    """The escalation ladder exists for exactly this: SIGINT, then SIGTERM, then SIGKILL."""
    script = (
        "import signal, time;"
        "signal.signal(signal.SIGINT, signal.SIG_IGN);"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(300)"
    )
    outcome = await runner.run(
        argv=(sys.executable, "-c", script),
        cwd=Path.cwd(),
        timeout_s=1,
        sigint_grace_s=0.3,
        sigterm_grace_s=0.3,
    )

    assert outcome.timed_out is True


# ------------------------------------------------------------- the environment


async def test_the_child_cannot_see_scrubbed_secrets(runner: SubprocessRunner) -> None:
    """§15's "scrubbed environment verified inside the child"."""
    outcome = await runner.run(
        argv=(sys.executable, "-c", "import os; print(os.environ.get('GITHUB_TOKEN', 'ABSENT'))"),
        cwd=Path.cwd(),
        base_environment={"PATH": os.environ["PATH"], "GITHUB_TOKEN": "leaked-value"},
    )

    assert "ABSENT" in outcome.text
    assert "leaked-value" not in outcome.text


async def test_the_child_runs_in_the_given_directory(runner: SubprocessRunner, tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()

    outcome = await runner.run(argv=("pwd",), cwd=workdir)

    assert str(workdir.resolve()) in outcome.text


# ------------------------------------------------------------------ truncation


def test_short_output_is_untouched() -> None:
    text = "line one\nline two\n"

    assert truncate_output(text) == text


def test_long_output_keeps_head_and_tail() -> None:
    """§8.2: the model gets the first 40 and last 120 lines.

    Both ends matter and for different reasons: a build failure's cause is usually at the
    top, and a test summary is always at the bottom. Keeping one end would lose half of
    every diagnosis.
    """
    lines = [f"line {number}" for number in range(1000)]

    result = truncate_output("\n".join(lines))

    assert "line 0" in result
    assert "line 999" in result
    assert "line 500" not in result
    assert "omitted" in result


def test_truncation_says_how_much_it_removed() -> None:
    result = truncate_output("\n".join(f"line {n}" for n in range(1000)))

    assert "840" in result, "1000 - 40 head - 120 tail"


def test_the_truncation_window_matches_the_spec() -> None:
    assert (HEAD_LINES, TAIL_LINES) == (40, 120)


async def test_full_output_is_kept_in_a_blob_for_paging(runner: SubprocessRunner, tmp_path: Path) -> None:
    """The model sees a window; `output_id` is how the untruncated text stays reachable."""
    script = "for n in range(1000): print('line', n)"
    outcome = await runner.run(argv=(sys.executable, "-c", script), cwd=Path.cwd())

    assert outcome.output_id is not None
    stored = runner.blobs.get(outcome.output_id).decode()
    assert "line 500" in stored, "the blob holds everything, unlike the model's view"


async def test_an_outcome_is_a_plain_value(runner: SubprocessRunner) -> None:
    """``CommandOutcome`` crosses into ``prepare``/``execute`` and the audit record, so it
    must not hold a live process handle."""
    outcome = await runner.run(argv=("echo", "x"), cwd=Path.cwd())

    assert isinstance(outcome, CommandOutcome)
    assert isinstance(outcome.exit_code, int)


# -------------------------------------------------------------------- shell


async def test_a_shell_command_runs_through_sh_when_asked(runner: SubprocessRunner) -> None:
    """Only ever after approval (§8.2). The preview carries the SHELL badge precisely so
    the user knows this is the execution they are authorising."""
    outcome = await runner.run(shell_command="echo one && echo two", cwd=Path.cwd())

    assert outcome.exit_code == 0
    assert "one" in outcome.text
    assert "two" in outcome.text


async def test_running_with_neither_argv_nor_a_shell_command_is_refused(
    runner: SubprocessRunner,
) -> None:
    outcome = await runner.run(cwd=Path.cwd())

    assert outcome.ok is False
    assert outcome.error is not None


async def test_signals_are_reported_as_such(runner: SubprocessRunner) -> None:
    """A command killed by a signal has no meaningful exit code; saying "exit 143" to the
    model invites it to debug the wrong thing."""
    outcome = await runner.run(
        argv=(sys.executable, "-c", f"import os, signal; os.kill(os.getpid(), {signal.SIGTERM})"),
        cwd=Path.cwd(),
    )

    assert outcome.ok is False
    assert outcome.signal_name == "SIGTERM"
