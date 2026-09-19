"""``git_add``, ``git_commit``, ``git_branch_create``, ``git_switch`` — §9.2.

The M6 acceptance criterion is here: *"Commit approval shows the staged diff and message.
Staged content containing a fake AWS key is flagged."*

Every test runs against a throwaway repository under ``tmp_path``. None of them touches
the repository this file lives in — CLAUDE.md rule 8, which matters more than usual for
the one tool suite whose whole job is to write git history.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hearth.safety.audit import AuditLog
from hearth.safety.errors import PathError
from hearth.safety.policy import ConfigView, SessionView
from hearth.tools.base import ToolContext
from hearth.tools.channel import ApprovalAsk, ApprovalReply
from hearth.tools.gateway import ToolGateway, make_policy
from hearth.tools.git_write import (
    GitAddTool,
    GitBranchCreateTool,
    GitCommitTool,
    GitSwitchTool,
)
from hearth.tools.registry import ToolRegistry
from hearth.tools.results import ErrorCode

FAKE_AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Hearth Test")
    git(root, "config", "commit.gpgsign", "false")
    (root / "a.py").write_text("value = 1\n", encoding="utf-8")
    git(root, "add", "a.py")
    git(root, "commit", "-qm", "initial")
    return root


@pytest.fixture
def context(repo: Path) -> ToolContext:
    return ToolContext(workspace=repo)


class RecordingChannel:
    def __init__(self, *, approve: bool = True) -> None:
        self._approve = approve
        self.asks: list[ApprovalAsk] = []

    async def proposed(self, **_: object) -> None:
        return None

    async def started(self, **_: object) -> None:
        return None

    async def finished(self, **_: object) -> None:
        return None

    async def request_approval(self, ask: ApprovalAsk) -> ApprovalReply:
        self.asks.append(ask)
        return ApprovalReply(decision="approve" if self._approve else "reject")


@pytest.fixture
def gateway_and_channel(context: ToolContext, tmp_path: Path):
    channel = RecordingChannel()
    gateway = ToolGateway(
        registry=ToolRegistry(
            [GitAddTool(), GitCommitTool(), GitBranchCreateTool(), GitSwitchTool()]
        ),
        context=context,
        channel=channel,
        audit=AuditLog(tmp_path / "audit", fsync=False),
        policy=make_policy(SessionView(mode="agent", level="supervised"), ConfigView()),
        session_id="s_git",
    )
    return gateway, channel


# --------------------------------------------------------------- the commit flow


async def test_the_commit_approval_shows_the_message_and_the_staged_diff(
    repo: Path, gateway_and_channel
) -> None:
    gateway, channel = gateway_and_channel
    (repo / "a.py").write_text("value = 2\n", encoding="utf-8")

    await gateway.call("git_add", {"paths": ["a.py"]}, call_id="c1")
    result = await gateway.call("git_commit", {"message": "Bump value"}, call_id="c2")

    assert result.ok
    commit_ask = channel.asks[-1]
    assert "Bump value" in commit_ask.preview
    assert "-value = 1" in commit_ask.preview
    assert "+value = 2" in commit_ask.preview
    assert "git revert" in commit_ask.preview, "the panel must say how to undo it"


async def test_a_fake_aws_key_in_staged_content_is_flagged(
    repo: Path, gateway_and_channel
) -> None:
    """The M6 criterion. The scan reads added lines, which are what enters history."""
    gateway, channel = gateway_and_channel
    (repo / "config.py").write_text(f'AWS_ACCESS_KEY_ID = "{FAKE_AWS_KEY}"\n', encoding="utf-8")

    await gateway.call("git_add", {"paths": ["config.py"]}, call_id="c1")
    await gateway.call("git_commit", {"message": "Add config"}, call_id="c2")

    badges = channel.asks[-1].badges
    assert any(badge.startswith("SECRET?") for badge in badges), badges


async def test_a_key_already_in_history_is_not_re_flagged(
    repo: Path, gateway_and_channel
) -> None:
    """Only added lines are scanned: an existing key is not what this commit is about."""
    gateway, channel = gateway_and_channel
    (repo / "config.py").write_text(f'KEY = "{FAKE_AWS_KEY}"\n', encoding="utf-8")
    git(repo, "add", "config.py")
    git(repo, "commit", "-qm", "pre-existing")

    (repo / "config.py").write_text(
        f'KEY = "{FAKE_AWS_KEY}"\nTIMEOUT = 30\n', encoding="utf-8"
    )
    await gateway.call("git_add", {"paths": ["config.py"]}, call_id="c1")
    await gateway.call("git_commit", {"message": "Add timeout"}, call_id="c2")

    assert not any(badge.startswith("SECRET?") for badge in channel.asks[-1].badges)


async def test_committing_nothing_is_refused_with_a_usable_message(
    gateway_and_channel,
) -> None:
    gateway, _ = gateway_and_channel

    result = await gateway.call("git_commit", {"message": "Empty"}, call_id="c1")

    assert not result.ok
    assert "git_add" in result.content


def test_a_stage_that_moved_after_approval_is_not_committed(
    repo: Path, context: ToolContext
) -> None:
    """The gap between the diff a person read and the commit is an approval prompt."""
    tool = GitCommitTool()
    (repo / "a.py").write_text("value = 2\n", encoding="utf-8")
    git(repo, "add", "a.py")

    args = tool.args_model.model_validate({"message": "Bump"})
    prepared = tool.prepare(args, context)  # type: ignore[arg-type]

    # The user is reading the diff; something else stages more.
    (repo / "b.py").write_text("extra = True\n", encoding="utf-8")
    git(repo, "add", "b.py")

    result = tool.execute(args, context, prepared)  # type: ignore[arg-type]

    assert not result.ok
    assert result.error is ErrorCode.STALE_FILE
    log = subprocess.run(
        ["git", "-C", str(repo), "log", "--oneline"], capture_output=True, text=True, check=True
    ).stdout
    assert log.count("\n") == 1, "nothing new was committed"


async def test_a_failing_hook_returns_its_output_to_the_model(
    repo: Path, gateway_and_channel
) -> None:
    """Hooks stay enabled and `--no-verify` is never passed, so this must surface."""
    gateway, _ = gateway_and_channel
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho 'lint: trailing whitespace in a.py'\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)

    (repo / "a.py").write_text("value = 2\n", encoding="utf-8")
    await gateway.call("git_add", {"paths": ["a.py"]}, call_id="c1")
    result = await gateway.call("git_commit", {"message": "Bump"}, call_id="c2")

    assert not result.ok
    assert "trailing whitespace" in result.content


# ------------------------------------------------------------------- staging


async def test_staging_a_path_outside_the_workspace_is_refused(
    context: ToolContext,
) -> None:
    tool = GitAddTool()
    args = tool.args_model.model_validate({"paths": ["../../etc/hosts"]})

    with pytest.raises(PathError):
        tool.prepare(args, context)  # type: ignore[arg-type]


async def test_staging_repository_metadata_is_refused(context: ToolContext) -> None:
    """`.git/**` is a protected path; no rule can make it stageable by the agent."""
    tool = GitAddTool()
    args = tool.args_model.model_validate({"paths": [".git/config"]})

    with pytest.raises(PathError):
        tool.prepare(args, context)  # type: ignore[arg-type]


# ------------------------------------------------------- branching and switching


async def test_switching_with_a_dirty_tree_is_refused(
    repo: Path, gateway_and_channel
) -> None:
    """Hearth will not stash or discard work to switch branches."""
    gateway, _ = gateway_and_channel
    git(repo, "branch", "other")
    (repo / "a.py").write_text("uncommitted = True\n", encoding="utf-8")

    result = await gateway.call("git_switch", {"name": "other"}, call_id="c1")

    assert not result.ok
    assert "uncommitted changes" in result.content


async def test_branch_create_and_switch_on_a_clean_tree(
    repo: Path, gateway_and_channel
) -> None:
    gateway, _ = gateway_and_channel

    created = await gateway.call(
        "git_branch_create", {"name": "hearth/fix-rounding"}, call_id="c1"
    )
    switched = await gateway.call("git_switch", {"name": "hearth/fix-rounding"}, call_id="c2")

    assert created.ok and switched.ok
    # Read HEAD directly rather than shelling out: a blocking subprocess inside an async
    # test is what ASYNC221 is about, and the ref file is the state being asserted anyway.
    head = (repo / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    assert head == "ref: refs/heads/hearth/fix-rounding"


async def test_a_branch_name_that_looks_like_a_flag_is_refused(
    gateway_and_channel,
) -> None:
    gateway, _ = gateway_and_channel

    result = await gateway.call("git_branch_create", {"name": "--force"}, call_id="c1")

    assert not result.ok
    assert result.error is ErrorCode.INVALID_ARGUMENTS
