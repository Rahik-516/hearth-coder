"""``run_command``: the tool that runs whatever the user approves.

Worth stating plainly, because every design choice here follows from it: until OS
sandboxing lands in Phase 3, **execution is not the security boundary — the approval is**
(docs/safety-and-tool-use.md §1.2). Approving ``npm test`` runs whatever the test scripts
do. So this tool is not trying to make arbitrary commands safe. It is trying to make sure
the user sees *accurately* what is about to run, and that the payload shapes that can never
be legitimate never reach a prompt at all.

Three consequences:

**Classification happens in ``prepare()``, not in the policy engine.** Turning a command
string into argv, families and badges *is* resolving the action, exactly as resolving a
path is. So the classification the user is shown and the classification policy judges are
one computation, not two that merely agree. ``policy.py`` stays pure and never imports the
classifier.

**A command that does not reduce to a single simple command carries ``argv=None``**, which
no allow rule can satisfy (§5.4). That is the whole mechanism behind `pytest; rm -rf ~`:
an allow rule for ``pytest`` is not *narrowly* unable to cover it — it is structurally
unable to, because there is no argv to match against.

**A hard-denied command is refused by policy, not by this tool.** It would be easy to
return an error from ``prepare()`` and save a round trip, but then `sudo rm -rf /` would be
audited as "the tool declined" rather than as an invariant refusing it. The audit log has
to be able to answer "what was blocked, and by what".
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hearth.safety.command_classifier import Classification, classify
from hearth.safety.errors import PathError
from hearth.safety.paths import (
    is_inside_workspace,
    relative_to_workspace,
    resolve_in_workspace,
)
from hearth.safety.policy import PolicyFacts
from hearth.safety.sandbox.subprocess_runner import (
    DEFAULT_TIMEOUT_S,
    MAX_TIMEOUT_S,
    CommandOutcome,
    SubprocessRunner,
)
from hearth.tools.base import Prepared, Risk, Tool, ToolContext
from hearth.tools.results import ErrorCode, ToolResult


class RunCommandArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(
        min_length=1,
        description="The command line to run, exactly as you would type it in a terminal",
    )
    cwd: str | None = Field(
        default=None,
        description="Workspace-relative directory to run in. Defaults to the workspace root.",
    )
    timeout_s: int = Field(
        default=DEFAULT_TIMEOUT_S,
        ge=1,
        le=MAX_TIMEOUT_S,
        description=f"Seconds before the process group is killed (max {MAX_TIMEOUT_S})",
    )


class RunCommandTool(Tool[RunCommandArgs]):
    name = "run_command"
    description = (
        "Run a command in the workspace and return its output. Nothing is interactive: "
        "there is no stdin, and prompts fail immediately rather than hanging."
    )
    risk = Risk.EXEC
    args_model = RunCommandArgs

    def __init__(self, runner: SubprocessRunner | None = None) -> None:
        self._runner = runner or SubprocessRunner()

    def prepare(self, args: RunCommandArgs, context: ToolContext) -> Prepared:
        """Classify the command and resolve where it would run. Mutates nothing."""
        classification = classify(args.command)
        if classification.refused is not None:
            return _refused(ErrorCode.INVALID_ARGUMENTS, classification.refused)

        # PathError is deliberately not caught: a cwd outside the jail is an invariant
        # refusal, and the gateway audits it as one.
        cwd = _resolve_cwd(context.workspace, args.cwd)
        if not cwd.is_dir():
            target = args.cwd or "."
            return _refused(ErrorCode.NOT_FOUND, f"{target} is not a directory in the workspace")

        relative_cwd = relative_to_workspace(cwd, context.workspace)
        return Prepared(
            summary=f"run {_display(classification)}",
            preview=_preview(classification, cwd=relative_cwd, timeout_s=args.timeout_s),
            badges=list(classification.badges),
            payload={
                "argv": classification.argv,
                "shell_command": args.command if classification.argv is None else None,
                "cwd": cwd,
                "relative_cwd": relative_cwd,
                "timeout_s": args.timeout_s,
                "display": _display(classification),
            },
            facts=_facts(classification),
        )

    async def execute(
        self, args: RunCommandArgs, context: ToolContext, prepared: Prepared
    ) -> ToolResult:
        payload = prepared.payload

        drift = _reverify(args, payload)
        if drift is not None:
            return drift

        outcome = await self._runner.run(
            argv=payload["argv"],
            shell_command=payload["shell_command"],
            cwd=payload["cwd"],
            timeout_s=payload["timeout_s"],
        )
        return command_result(outcome, display=payload["display"])


# ---------------------------------------------------------------- shared helpers


def command_result(outcome: CommandOutcome, *, display: str) -> ToolResult:
    """Render a finished command for the model.

    **A non-zero exit is a successful tool call.** Failing tests are the normal middle of
    an agent run — the loop is edit → test → fix — so reporting them as a tool *failure*
    would spend the retry budget on the agent doing exactly what it was asked to do, and
    three failing test runs would end the turn through `failing_repeatedly`. Only "could
    not run at all" and "was killed at the timeout" are failures of the call itself.
    """
    if outcome.error is not None and not outcome.timed_out:
        return ToolResult.failure(ErrorCode.EXECUTION_FAILED, f"{display}: {outcome.error}")

    header = f"$ {display}\nexit {outcome.exit_code} in {outcome.duration_s:.1f}s"
    body = outcome.text.strip()
    content = f"{header}\n\n{body}" if body else header

    result = ToolResult(
        ok=not outcome.timed_out,
        content=content,
        error=ErrorCode.EXECUTION_FAILED if outcome.timed_out else None,
        metadata={
            "exit_code": outcome.exit_code,
            "timed_out": outcome.timed_out,
            "duration_s": round(outcome.duration_s, 3),
            "output_bytes": outcome.output_bytes,
            "signal": outcome.signal_name,
        },
    )
    result.output_id = outcome.output_id
    return result


def _resolve_cwd(workspace: Path, user_path: str | None) -> Path:
    """Resolve the working directory, which must be inside the workspace (§8.2).

    Containment is checked here rather than by asking for a *write* resolution, for two
    reasons. A cwd is not a write, so ``for_write=True`` would refuse running anything
    with a cwd of ``.git`` as a "protected path" and would report an escape as "write
    outside workspace" — a misleading sentence for what actually happened. And the read
    resolution deliberately *permits* paths outside the workspace, leaving them for the
    policy engine to judge, which is right for reading a file and wrong for a cwd: §8.2
    lists the working directory as part of the jail, not as something a rule may widen.
    """
    if user_path is None:
        return workspace

    resolved = resolve_in_workspace(workspace, user_path)
    if not is_inside_workspace(resolved, workspace):
        raise PathError("working directory outside workspace", requested=user_path)
    return resolved


def _facts(classification: Classification) -> PolicyFacts:
    """Hand the engine the classification verdict, unchanged.

    ``grant_key`` is offered only for a command that parsed into one simple command:
    "always for this session" has to name something exact (§5.4), and a string the
    classifier could not reduce to argv has no exact form to name. The engine applies the
    further §6.1 filters — no grants for SHELL, NETWORK? or DESTRUCTIVE.
    """
    grant_key = f"run_command:{' '.join(classification.argv)}" if classification.argv else None
    return PolicyFacts(
        argv=classification.argv,
        env_prefix=classification.env_prefix,
        shell=classification.shell,
        destructive=classification.destructive,
        network_likely=classification.network_likely,
        hard_denied=classification.hard_denied,
        badges=classification.badges,
        grant_key=grant_key,
    )


def _reverify(args: RunCommandArgs, payload: dict[str, object]) -> ToolResult | None:
    """Re-derive the command and refuse if it is not what was approved.

    The gap between ``prepare()`` and here is an approval prompt a person may take a
    minute over. This re-classifies the *same* argument string and compares the result
    with what the preview was built from, so that the thing that runs is the thing that
    was shown (§1.3.3) rather than a second resolution that merely tends to agree.
    """
    again = classify(args.command)
    if again.refused is not None or again.argv != payload["argv"]:
        return ToolResult.failure(
            ErrorCode.EXECUTION_FAILED,
            "the command changed between approval and execution; nothing was run",
        )

    cwd = payload["cwd"]
    if not isinstance(cwd, Path) or not cwd.is_dir():
        return ToolResult.failure(
            ErrorCode.NOT_FOUND, "the working directory no longer exists; nothing was run"
        )
    return None


def _display(classification: Classification) -> str:
    """The command as the approval panel and the audit log name it."""
    if classification.argv is None:
        return classification.raw.strip()
    prefix = "".join(f"{name}={value} " for name, value in classification.env_prefix)
    return f"{prefix}{' '.join(classification.argv)}"


def _preview(classification: Classification, *, cwd: str, timeout_s: int) -> str:
    """What the user approves.

    Shows the resolved argv rather than the raw string, because the difference between the
    two is exactly where a bypass hides. For a command that did not parse into one simple
    command, the per-segment split is shown instead, labelled as what it is: a best-effort
    reading of a string that will be handed to a shell.
    """
    lines = [f"$ {_display(classification)}"]

    if classification.argv is None and classification.segments:
        lines.append("")
        lines.append("runs through a shell; best-effort reading of the segments:")
        lines.extend(
            f"  {index}. {' '.join(segment)}"
            for index, segment in enumerate(classification.segments, 1)
        )

    if classification.env_prefix:
        assignments = ", ".join(f"{name}={value}" for name, value in classification.env_prefix)
        lines.append(f"environment: {assignments}")

    lines.append("")
    lines.append(f"cwd:     {cwd or '.'}")
    lines.append(f"timeout: {timeout_s}s")
    if classification.families:
        lines.append(f"family:  {', '.join(sorted(classification.families))}")
    return "\n".join(lines)


def _refused(code: ErrorCode, message: str) -> Prepared:
    """A refusal by the tool itself, before policy sees the call."""
    return Prepared(summary="run_command refused", error=ToolResult.failure(code, message))
