"""The git operations that change something: staging, committing, branching, switching.

docs/safety-and-tool-use.md §9.2. Four decisions here are load-bearing:

**Hooks stay enabled, and ``--no-verify`` is never passed.** A project's pre-commit checks
belong to the user. Skipping them would be Hearth quietly overruling a decision that is not
its to make, and it is the kind of shortcut that is only ever noticed after it has let
something through. When a hook rejects a commit, its output goes back to the model, which
is usually enough for it to fix the problem and try again.

**The secret scan reads the staged added lines**, not the working tree and not the whole
file. Those are the bytes that are about to enter history, where a leaked key is
permanent — `git revert` does not remove it, and the rewrite that would is exactly the
operation Hearth refuses to perform.

**What is shown is what is committed.** ``prepare()`` records the staged tree's object id;
``execute()`` re-reads it and abandons the commit if it moved. Otherwise the diff a person
spent a minute reading and the snapshot that lands are two computations that merely tend to
agree.

**Nothing here rewrites history or reaches the network.** No amend, no rebase, no reset,
no push — those are refused by the classifier before they reach a tool, and no tool exists
to route around it. The approval text says how to undo a commit by hand, because the
honest answer is that the user does it in their own terminal.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hearth.git.runner import GitError, GitResult, is_git_repository, run_git
from hearth.safety.paths import resolve_in_workspace
from hearth.safety.policy import PolicyFacts
from hearth.safety.secrets import find_secrets
from hearth.tools.base import Prepared, Risk, Tool, ToolContext
from hearth.tools.results import ErrorCode, ToolResult
from hearth.util.hashing import content_hash

#: Diff lines shown in the approval panel. The full diff stays reachable with `git_diff`.
PREVIEW_DIFF_LINES = 200

#: How a commit is undone. Shown on every commit approval, because Hearth will not do it.
UNDO_HINT = "Undo with `git revert <sha>` or `git reset --soft HEAD~1` in your terminal."


# ------------------------------------------------------------------- git_add


class GitAddArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(
        min_length=1,
        description="Workspace-relative paths to stage. Directories stage their contents.",
    )


class GitAddTool(Tool[GitAddArgs]):
    name = "git_add"
    description = "Stage files for the next commit."
    risk = Risk.VCS_WRITE
    args_model = GitAddArgs

    def prepare(self, args: GitAddArgs, context: ToolContext) -> Prepared:
        repo = _require_repo(context, self.name)
        if repo is not None:
            return repo

        # ``for_write=True`` for the same reason an edit uses it: staging mutates the
        # index, `git add ../../etc` must not reach outside the workspace, and the
        # protected set (`.git/**`, `.hearth/**`) is exactly what the agent may not put
        # into a commit. PathError is left to propagate so the gateway audits it as a
        # jail refusal rather than as a tool that declined.
        relatives = [
            _relative(
                context.workspace,
                resolve_in_workspace(context.workspace, path, for_write=True),
            )
            for path in args.paths
        ]

        return Prepared(
            summary=f"stage {len(relatives)} path(s)",
            preview="\n".join(f"  + {relative}" for relative in relatives),
            payload={"relatives": relatives},
            facts=PolicyFacts(grant_key="git_add"),
        )

    def execute(self, args: GitAddArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        relatives = prepared.payload["relatives"]
        result = _git(context.workspace, ["add", "--", *relatives])
        if isinstance(result, ToolResult):
            return result

        staged = _staged_names(context.workspace)
        return ToolResult.success(
            f"Staged {len(relatives)} path(s). {len(staged)} file(s) now staged.",
            staged=staged,
        )


# ---------------------------------------------------------------- git_commit


class GitCommitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, description="The commit message")


class GitCommitTool(Tool[GitCommitArgs]):
    name = "git_commit"
    description = (
        "Commit the staged changes. Runs the project's hooks. Stage files with git_add first."
    )
    risk = Risk.VCS_WRITE
    args_model = GitCommitArgs

    def prepare(self, args: GitCommitArgs, context: ToolContext) -> Prepared:
        repo = _require_repo(context, self.name)
        if repo is not None:
            return repo

        staged = _staged_names(context.workspace)
        if not staged:
            return _refused(
                ErrorCode.NOT_FOUND,
                "nothing is staged. Use git_add to stage the files you want to commit.",
            )

        diff = _git(context.workspace, ["diff", "--staged", "--no-ext-diff", "--no-textconv"])
        if isinstance(diff, ToolResult):
            return Prepared(summary="git_commit failed", error=diff)

        stat = _git(context.workspace, ["diff", "--staged", "--stat"])
        stat_text = stat.text().strip() if isinstance(stat, GitResult) else ""

        diff_text = diff.text()
        badges = _secret_badges(diff_text)

        # Identity of what is about to be committed, taken from the bytes actually shown.
        # `git write-tree` would give a stronger id but writes objects into the database,
        # and `prepare()` mutating anything is the invariant that lets a preview be
        # generated for a call that is then rejected, with nothing to undo.
        staged_hash = content_hash(diff_text.encode("utf-8"))

        return Prepared(
            summary=f"commit {len(staged)} file(s)",
            preview=_commit_preview(args.message, stat_text, diff_text),
            badges=badges,
            payload={"message": args.message, "staged_hash": staged_hash, "staged": staged},
            facts=PolicyFacts(badges=tuple(badges), grant_key=None),
        )

    def execute(self, args: GitCommitArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        payload = prepared.payload

        current = _git(
            context.workspace, ["diff", "--staged", "--no-ext-diff", "--no-textconv"]
        )
        if isinstance(current, ToolResult):
            return current
        if content_hash(current.text().encode("utf-8")) != payload["staged_hash"]:
            return ToolResult.failure(
                ErrorCode.STALE_FILE,
                "the staged changes moved between approval and commit; nothing was committed. "
                "Re-stage and ask again.",
            )

        # read_only=False: the project's hooks must run. `--no-verify` is never passed.
        try:
            result = run_git(
                context.workspace,
                ["commit", "-m", str(payload["message"])],
                read_only=False,
                timeout_s=120,
            )
        except GitError as exc:
            return ToolResult.failure(ErrorCode.EXECUTION_FAILED, str(exc))

        if not result.ok:
            # Hook output is the actionable part: it names what to fix.
            detail = (result.text() + "\n" + result.stderr).strip()
            return ToolResult.failure(
                ErrorCode.EXECUTION_FAILED,
                f"the commit was rejected, most likely by a hook:\n{detail}",
            )

        sha = _head_sha(context.workspace)
        return ToolResult.success(
            f"Committed {len(payload['staged'])} file(s) as {sha}. {UNDO_HINT}",
            sha=sha,
        )


# --------------------------------------------------------- branch and switch


class GitBranchCreateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Branch name, e.g. hearth/fix-rounding")


class GitBranchCreateTool(Tool[GitBranchCreateArgs]):
    name = "git_branch_create"
    description = "Create a branch at the current commit without switching to it."
    risk = Risk.VCS_WRITE
    args_model = GitBranchCreateArgs

    def prepare(self, args: GitBranchCreateArgs, context: ToolContext) -> Prepared:
        repo = _require_repo(context, self.name)
        if repo is not None:
            return repo

        invalid = _invalid_branch_name(args.name)
        if invalid is not None:
            return _refused(ErrorCode.INVALID_ARGUMENTS, invalid)

        return Prepared(
            summary=f"create branch {args.name}",
            preview=f"create branch {args.name} at {_head_sha(context.workspace)}",
            payload={"name": args.name},
            facts=PolicyFacts(grant_key="git_branch_create"),
        )

    def execute(
        self, args: GitBranchCreateArgs, context: ToolContext, prepared: Prepared
    ) -> ToolResult:
        result = _git(context.workspace, ["branch", "--", str(prepared.payload["name"])])
        if isinstance(result, ToolResult):
            return result
        return ToolResult.success(f"Created branch {prepared.payload['name']}.")


class GitSwitchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Existing branch to switch to")


class GitSwitchTool(Tool[GitSwitchArgs]):
    name = "git_switch"
    description = "Switch to an existing branch. Refused when the working tree has changes."
    risk = Risk.VCS_WRITE
    args_model = GitSwitchArgs

    def prepare(self, args: GitSwitchArgs, context: ToolContext) -> Prepared:
        repo = _require_repo(context, self.name)
        if repo is not None:
            return repo

        invalid = _invalid_branch_name(args.name)
        if invalid is not None:
            return _refused(ErrorCode.INVALID_ARGUMENTS, invalid)

        # A dirty tree is refused rather than stashed. Switching with local changes either
        # carries them onto the new branch or fails halfway, and "where did my edits go"
        # is not a question an agent should be able to create.
        dirty = _git(context.workspace, ["status", "--porcelain"])
        if isinstance(dirty, ToolResult):
            return Prepared(summary="git_switch failed", error=dirty)
        if dirty.text().strip():
            return _refused(
                ErrorCode.EXECUTION_FAILED,
                "the working tree has uncommitted changes. Commit them first; Hearth will "
                "not stash or discard your work to switch branches.",
            )

        return Prepared(
            summary=f"switch to {args.name}",
            preview=f"switch to branch {args.name} (working tree is clean)",
            payload={"name": args.name},
            facts=PolicyFacts(grant_key="git_switch"),
        )

    def execute(self, args: GitSwitchArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        result = _git(context.workspace, ["switch", "--", str(prepared.payload["name"])])
        if isinstance(result, ToolResult):
            return result
        return ToolResult.success(f"Switched to {prepared.payload['name']}.")


# ------------------------------------------------------------------ helpers


def _git(root: Path, args: list[str]) -> GitResult | ToolResult:
    """Run a read-only git command, returning a ToolResult when it failed."""
    try:
        result = run_git(root, args)
    except GitError as exc:
        return ToolResult.failure(ErrorCode.EXECUTION_FAILED, str(exc))
    if not result.ok:
        return ToolResult.failure(
            ErrorCode.EXECUTION_FAILED, result.stderr or f"git {args[0]} failed"
        )
    return result


def _require_repo(context: ToolContext, tool: str) -> Prepared | None:
    if is_git_repository(context.workspace):
        return None
    return _refused(ErrorCode.NOT_FOUND, f"{tool} needs a git repository; this workspace is not one")


def _staged_names(root: Path) -> list[str]:
    result = _git(root, ["diff", "--staged", "--name-only"])
    if isinstance(result, ToolResult):
        return []
    return [line for line in result.text().splitlines() if line.strip()]


def _head_sha(root: Path) -> str:
    result = _git(root, ["rev-parse", "--short", "HEAD"])
    if isinstance(result, ToolResult):
        return "(no commits yet)"
    return result.text().strip() or "(no commits yet)"


def _secret_badges(diff_text: str) -> list[str]:
    """Scan only the added lines of the staged diff.

    A key already present in the file is not what this commit is about; a key on a ``+``
    line is about to enter history permanently.
    """
    added = "\n".join(
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    labels = find_secrets(added)
    return [f"SECRET? {', '.join(sorted(set(labels)))}"] if labels else []


def _commit_preview(message: str, stat_text: str, diff_text: str) -> str:
    lines = [message.strip(), ""]
    if stat_text:
        lines.extend([stat_text, ""])

    diff_lines = diff_text.splitlines()
    if len(diff_lines) > PREVIEW_DIFF_LINES:
        omitted = len(diff_lines) - PREVIEW_DIFF_LINES
        shown = [*diff_lines[:PREVIEW_DIFF_LINES], f"… {omitted} more line(s) of diff …"]
    else:
        shown = diff_lines
    lines.extend(shown)

    lines.extend(["", UNDO_HINT])
    return "\n".join(lines)


def _invalid_branch_name(name: str) -> str | None:
    """Refuse names git would reject or that could be read as a flag.

    A leading ``-`` is the interesting one: without ``--`` in the argv it would be parsed
    as an option rather than a branch. The separator is passed anyway; this refuses the
    name outright so the error says what is wrong instead of failing obscurely.
    """
    if name.startswith("-"):
        return "a branch name cannot start with '-'"
    if name.strip() != name or not name.strip():
        return "a branch name cannot begin or end with whitespace"
    forbidden = {"..", "~", "^", ":", "?", "*", "[", "\\", " ", "\t"}
    if any(token in name for token in forbidden):
        return f"a branch name cannot contain any of: {' '.join(sorted(forbidden))}"
    if name.endswith((".lock", "/", ".")):
        return "a branch name cannot end with '/', '.' or '.lock'"
    return None


def _relative(root: Path, resolved: Path) -> str:
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:  # pragma: no cover - resolve_in_workspace already refused these
        return resolved.as_posix()


def _refused(code: ErrorCode, message: str) -> Prepared:
    return Prepared(summary="git write refused", error=ToolResult.failure(code, message))
