"""Read-only git tools: ``git_status``, ``git_diff``, ``git_log``.

All three go through ``hearth.git.runner``, which disables fsmonitor, external diff drivers
and textconv (docs/safety-and-tool-use.md §9.1). That matters here more than anywhere:
these are the tools an agent reaches for on a repository nobody has inspected yet, and
those mechanisms turn an ordinary read into arbitrary code execution.

Output is truncated at insertion time, not afterwards: a large diff would otherwise consume
the whole context budget before anyone could object (docs/system-design.md §9.3).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hearth.git.runner import GitError, is_git_repository, run_git
from hearth.tools.base import Prepared, Risk, Tool, ToolContext
from hearth.tools.results import ErrorCode, ToolResult

#: Diff budget, roughly 3K tokens (docs/system-design.md §9.3).
MAX_DIFF_LINES = 400
MAX_LOG_ENTRIES = 30


class GitStatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GitStatusTool(Tool[GitStatusArgs]):
    name = "git_status"
    description = "Show which files are modified, staged or untracked."
    risk = Risk.READ
    args_model = GitStatusArgs
    concurrent_safe = True

    def prepare(self, args: GitStatusArgs, context: ToolContext) -> Prepared:
        return _require_repo(context, "git_status")

    def execute(self, args: GitStatusArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        try:
            result = run_git(context.workspace, ["status", "--porcelain=v2", "--branch"])
        except GitError as exc:
            return ToolResult.failure(ErrorCode.EXECUTION_FAILED, str(exc))

        if not result.ok:
            return ToolResult.failure(ErrorCode.EXECUTION_FAILED, result.stderr or "git status failed")

        text = result.text().strip()
        if not text:
            return ToolResult.success("The working tree is clean.")

        return ToolResult.success(_summarize_status(text))


class GitDiffArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    staged: bool = Field(default=False, description="Show staged changes instead of unstaged")
    path: str | None = Field(default=None, description="Restrict to one path")


class GitDiffTool(Tool[GitDiffArgs]):
    name = "git_diff"
    description = "Show uncommitted changes."
    risk = Risk.READ
    args_model = GitDiffArgs
    concurrent_safe = True

    def prepare(self, args: GitDiffArgs, context: ToolContext) -> Prepared:
        return _require_repo(context, "git_diff")

    def execute(self, args: GitDiffArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        command = ["diff", "--no-ext-diff", "--no-textconv"]
        if args.staged:
            command.append("--staged")
        if args.path:
            command.extend(["--", args.path])

        try:
            result = run_git(context.workspace, command)
        except GitError as exc:
            return ToolResult.failure(ErrorCode.EXECUTION_FAILED, str(exc))

        text = result.text()
        if not text.strip():
            scope = "staged" if args.staged else "unstaged"
            return ToolResult.success(f"No {scope} changes.")

        lines = text.splitlines()
        if len(lines) <= MAX_DIFF_LINES:
            return ToolResult.success(text, lines=len(lines))

        shown = "\n".join(lines[:MAX_DIFF_LINES])
        return ToolResult.success(
            f"{shown}\n\n… diff truncated at {MAX_DIFF_LINES} of {len(lines)} lines. "
            f"Use the path argument to narrow it.",
            lines=len(lines),
            truncated=True,
        )


class GitLogArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=10, ge=1, le=MAX_LOG_ENTRIES)
    path: str | None = Field(default=None, description="Only commits touching this path")


class GitLogTool(Tool[GitLogArgs]):
    name = "git_log"
    description = "Show recent commits."
    risk = Risk.READ
    args_model = GitLogArgs
    concurrent_safe = True

    def prepare(self, args: GitLogArgs, context: ToolContext) -> Prepared:
        return _require_repo(context, "git_log")

    def execute(self, args: GitLogArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        command = [
            "log",
            f"--max-count={args.limit}",
            "--format=%h %ad %an — %s",
            "--date=short",
        ]
        if args.path:
            command.extend(["--", args.path])

        try:
            result = run_git(context.workspace, command)
        except GitError as exc:
            return ToolResult.failure(ErrorCode.EXECUTION_FAILED, str(exc))

        text = result.text().strip()
        if not text:
            return ToolResult.success("No commits found.")
        return ToolResult.success(text, count=len(text.splitlines()))


def _require_repo(context: ToolContext, tool: str) -> Prepared:
    if not is_git_repository(context.workspace):
        return Prepared(
            summary=tool,
            error=ToolResult.failure(ErrorCode.NOT_FOUND, f"{context.workspace} is not a git repository."),
        )
    return Prepared(summary=tool)


def _summarize_status(porcelain: str) -> str:
    """Turn porcelain v2 into something a model can read.

    The raw format is designed for machines and wastes tokens on a model that only needs
    to know which files changed.
    """
    branch = ""
    changed: list[str] = []
    untracked: list[str] = []

    for line in porcelain.splitlines():
        if line.startswith("# branch.head"):
            branch = line.split(maxsplit=2)[-1]
        elif line.startswith(("1 ", "2 ")):
            parts = line.split(maxsplit=8)
            if len(parts) >= 9:
                changed.append(f"  {parts[1]}  {parts[8]}")
        elif line.startswith("? "):
            untracked.append(f"  {line[2:]}")

    sections = [f"branch: {branch}" if branch else "detached HEAD"]
    if changed:
        sections.append(f"changed ({len(changed)}):\n" + "\n".join(changed[:50]))
    if untracked:
        sections.append(f"untracked ({len(untracked)}):\n" + "\n".join(untracked[:20]))
    if not changed and not untracked:
        sections.append("working tree clean")
    return "\n".join(sections)
