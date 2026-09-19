"""Running an approved command without letting it hang, leak, or leave children behind.

This is sandbox level **L0** (docs/safety-and-tool-use.md §12), and it is worth being
blunt about what that means: **this is not a security boundary.** An approved command runs
with the user's privileges and can do anything they can. Real containment is bubblewrap in
Phase 3. What L0 provides is the set of guarantees that stop an agent run going wrong in
boring ways, which in practice is what actually happens:

* **Nothing waits for a human.** stdin is ``DEVNULL`` and the environment forces
  ``PAGER=cat`` and ``GIT_TERMINAL_PROMPT=0``, so a prompt reads EOF and fails in
  milliseconds instead of consuming the whole timeout.
* **Nothing survives the kill.** The command gets its own process group
  (``start_new_session=True``) and signals go to the *group*. A build script whose `sleep`
  or dev server outlives the kill is an orphan holding a port or a lock, and nothing will
  ever clean it up.
* **Nothing reads the user's credentials.** The environment is an allowlist
  (``safety/env.py``).
* **Nothing floods the context.** Full output goes to a blob; the model sees a head/tail
  window and an ``output_id`` to page with.

The kill ladder is ``SIGINT → SIGTERM → SIGKILL``, in that order, because they mean
different things to a build tool. SIGINT is what Ctrl+C sends, so well-behaved programs
clean up temporary files and release locks on it. Going straight to SIGKILL would leave
exactly the mess this module exists to prevent — but a process that ignores the first two
still dies, because the ladder does not depend on cooperation.

POSIX only. Native Windows process-tree handling is deferred to Phase 3 with the rest of
the native-Windows work; Hearth runs in WSL2.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

from hearth.safety.env import build_environment
from hearth.storage.blobs import BlobStore

#: Lines of output the model sees from the start, and from the end (§8.2).
#:
#: Asymmetric on purpose: a build failure's cause is usually near the top, and a test
#: summary is always at the bottom, so the tail gets the larger share.
HEAD_LINES = 40
TAIL_LINES = 120

#: Captured output beyond this is dropped rather than stored (§8.2).
MAX_OUTPUT_BYTES = 20 * 1024 * 1024

#: Default and maximum timeouts, in seconds.
DEFAULT_TIMEOUT_S = 120
MAX_TIMEOUT_S = 600

#: How long each rung of the kill ladder waits before escalating.
SIGINT_GRACE_S = 5.0
SIGTERM_GRACE_S = 3.0


@dataclass(frozen=True)
class CommandOutcome:
    """What running a command produced.

    A plain value with no live handles: it crosses from ``execute()`` into the audit record
    and the model's context, both of which outlive the process.
    """

    exit_code: int
    #: Output as the model should see it — already truncated.
    text: str
    duration_s: float
    timed_out: bool = False
    #: SHA-256 of the full captured output in the blob store, for paging.
    output_id: str | None = None
    output_bytes: int = 0
    #: Set when the process died from a signal rather than exiting.
    signal_name: str | None = None
    #: Set when the command could not be run at all.
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None


class SubprocessRunner:
    """Runs one command at a time, with the L0 guarantees applied."""

    def __init__(self, *, blobs: BlobStore | None = None) -> None:
        self.blobs = blobs

    async def run(
        self,
        *,
        argv: tuple[str, ...] | None = None,
        shell_command: str | None = None,
        cwd: Path,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        base_environment: dict[str, str] | None = None,
        extra_allowed: tuple[str, ...] = (),
        env_overrides: dict[str, str] | None = None,
        sigint_grace_s: float = SIGINT_GRACE_S,
        sigterm_grace_s: float = SIGTERM_GRACE_S,
    ) -> CommandOutcome:
        """Run a command. Never raises for command-level problems.

        Args:
            argv: A parsed command, run without a shell. Preferred.
            shell_command: A raw string, run via ``/bin/sh -c``. Only ever reached after
                an approval that showed the ``SHELL`` badge (§8.2).
            cwd: Working directory. The caller has already confirmed it is in the jail.
            timeout_s: Clamped to :data:`MAX_TIMEOUT_S`.
        """
        if argv is None and shell_command is None:
            return CommandOutcome(
                exit_code=-1,
                text="",
                duration_s=0.0,
                error="nothing to run: neither argv nor a shell command was given",
            )

        environment = build_environment(
            base_environment if base_environment is not None else dict(os.environ),
            extra_allowed=extra_allowed,
            overrides=env_overrides,
        )
        environment.setdefault("PWD", str(cwd))

        started = time.monotonic()
        try:
            process = await self._spawn(argv, shell_command, cwd=cwd, environment=environment)
        except FileNotFoundError:
            name = argv[0] if argv else "sh"
            return CommandOutcome(
                exit_code=-1,
                text="",
                duration_s=time.monotonic() - started,
                error=f"{name} is not installed or not on PATH",
            )
        except OSError as exc:
            return CommandOutcome(
                exit_code=-1, text="", duration_s=time.monotonic() - started, error=str(exc)
            )

        captured, timed_out = await self._collect(
            process,
            timeout_s=min(timeout_s, MAX_TIMEOUT_S),
            sigint_grace_s=sigint_grace_s,
            sigterm_grace_s=sigterm_grace_s,
        )
        duration = time.monotonic() - started

        return self._outcome(process, captured, timed_out=timed_out, duration_s=duration)

    # ----------------------------------------------------------- internals

    async def _spawn(
        self,
        argv: tuple[str, ...] | None,
        shell_command: str | None,
        *,
        cwd: Path,
        environment: dict[str, str],
    ) -> asyncio.subprocess.Process:
        """Start the process in its own session, with stdin closed.

        ``start_new_session=True`` is the load-bearing argument: it makes the child a
        process-group leader, which is what lets the timeout path signal the whole tree
        rather than just the process Hearth can see.
        """
        common = {
            "cwd": str(cwd),
            "env": environment,
            "stdin": asyncio.subprocess.DEVNULL,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.STDOUT,
            "start_new_session": True,
        }

        if argv is not None:
            return await asyncio.create_subprocess_exec(*argv, **common)  # type: ignore[arg-type]
        return await asyncio.create_subprocess_shell(shell_command or "", **common)  # type: ignore[arg-type]

    async def _collect(
        self,
        process: asyncio.subprocess.Process,
        *,
        timeout_s: int,
        sigint_grace_s: float,
        sigterm_grace_s: float,
    ) -> tuple[bytes, bool]:
        """Read the output, killing the process group if it overruns."""
        if process.stdout is None:  # pragma: no cover - stdout is always a pipe here
            with contextlib.suppress(Exception):
                await process.wait()
            return b"", False

        reader = asyncio.create_task(self._read_capped(process.stdout))
        try:
            await asyncio.wait_for(asyncio.shield(reader), timeout=timeout_s)
        except TimeoutError:
            await self._terminate_group(process, sigint_grace_s, sigterm_grace_s)
            with contextlib.suppress(Exception):
                captured = await reader
            return captured if isinstance(captured, bytes) else b"", True
        except asyncio.CancelledError:
            # The turn was cancelled (Ctrl+C). Same treatment: the command does not get to
            # outlive the request that started it.
            await self._terminate_group(process, sigint_grace_s, sigterm_grace_s)
            reader.cancel()
            raise

        with contextlib.suppress(Exception):
            await process.wait()
        return await reader, False

    async def _read_capped(self, stream: asyncio.StreamReader) -> bytes:
        """Read until EOF, stopping at the byte cap.

        Capped while reading rather than afterwards: a command producing gigabytes would
        otherwise be held in memory in full before anyone decided it was too much.
        """
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            if total < MAX_OUTPUT_BYTES:
                chunks.append(chunk[: MAX_OUTPUT_BYTES - total])
                total += len(chunk)
        return b"".join(chunks)

    async def _terminate_group(
        self, process: asyncio.subprocess.Process, sigint_grace_s: float, sigterm_grace_s: float
    ) -> None:
        """SIGINT, then SIGTERM, then SIGKILL — to the process *group*."""
        for sig, grace in (
            (signal.SIGINT, sigint_grace_s),
            (signal.SIGTERM, sigterm_grace_s),
            (signal.SIGKILL, 0.0),
        ):
            if process.returncode is not None:
                return
            self._signal_group(process, sig)
            if grace <= 0:
                break
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(_wait(process)), timeout=grace)
                return

        with contextlib.suppress(Exception):
            await asyncio.wait_for(_wait(process), timeout=2.0)

    def _signal_group(self, process: asyncio.subprocess.Process, sig: int) -> None:
        """Signal the whole group, falling back to the single process.

        The fallback matters on the race where the child has already exited: `killpg`
        raises rather than doing nothing, and a failed cleanup must not mask the original
        timeout.
        """
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(ProcessLookupError, OSError):
                process.send_signal(sig)

    def _outcome(
        self,
        process: asyncio.subprocess.Process,
        captured: bytes,
        *,
        timed_out: bool,
        duration_s: float,
    ) -> CommandOutcome:
        text = captured.decode("utf-8", errors="replace")
        output_id = self.blobs.put(captured) if self.blobs is not None and captured else None

        returncode = process.returncode if process.returncode is not None else -1
        signal_name = None
        if returncode < 0:
            with contextlib.suppress(ValueError):
                signal_name = signal.Signals(-returncode).name

        shown = truncate_output(text)
        if timed_out:
            shown = (
                f"{shown}\n\n[timed out after {duration_s:.1f}s; the process group was killed]"
            ).lstrip()

        return CommandOutcome(
            exit_code=returncode,
            text=shown,
            duration_s=duration_s,
            timed_out=timed_out,
            output_id=output_id,
            output_bytes=len(captured),
            signal_name=signal_name,
            error=f"timed out after {duration_s:.1f}s" if timed_out else None,
        )


async def _wait(process: asyncio.subprocess.Process) -> int:
    return await process.wait()


def truncate_output(text: str, *, head: int = HEAD_LINES, tail: int = TAIL_LINES) -> str:
    """Keep the first ``head`` and last ``tail`` lines, with a marker between them.

    Both ends, because they answer different questions: the cause of a build failure is
    usually near the top, and the summary of a test run is always at the bottom. Keeping
    one end would lose half of every diagnosis.
    """
    lines = text.splitlines()
    if len(lines) <= head + tail:
        return text

    omitted = len(lines) - head - tail
    return "\n".join(
        [
            *lines[:head],
            f"\n… {omitted} line(s) omitted; use output_id to page through the rest …\n",
            *lines[-tail:],
        ]
    )
