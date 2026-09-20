"""`/commit` — I3.

The workflow's claim is that the model is asked for *wording only*. Everything that decides
what happens — what is staged, whether anything is, whether the user agrees, whether hooks
pass — is code and the existing `git_commit` tool. So these tests are mostly about what the
model's reply is **not** allowed to do:

* it cannot make the workflow stage anything (`test_the_workflow_never_stages`),
* it cannot make a commit happen without the approval prompt,
* a reply that is not a message (empty, fenced, quoted) is cleaned or refused rather than
  committed as-is.

Real git repositories in tmp dirs, driven through the real `git_commit` tool and gateway;
nothing here touches a real repository (CLAUDE.md rule 8).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hearth.core.bus import EventBus
from hearth.core.context.tokens import TokenEstimator
from hearth.core.runner import ChatRunner
from hearth.core.session import Mode, Session
from hearth.llm.scripted_provider import ScriptedProvider, ScriptedResponse
from hearth.safety.policy import ConfigView, SessionView
from hearth.tools.base import ToolContext
from hearth.tools.channel import ApprovalAsk, ApprovalReply
from hearth.tools.gateway import ToolGateway, make_policy
from hearth.tools.git_write import GitCommitTool
from hearth.tools.registry import ToolRegistry
from hearth.workflows.commit import (
    SUBJECT_LIMIT,
    clean_commit_message,
    prepare_commit,
    run_commit,
)


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Hearth Test")
    (root / "a.py").write_text("value = 1\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "Add the value")
    return root


class ScriptedChannel:
    """Answers approvals from a queue and remembers what it was shown."""

    def __init__(self, *replies: ApprovalReply) -> None:
        self.replies = list(replies)
        self.asks: list[ApprovalAsk] = []

    async def proposed(self, *, call_id: str, tool: str, arguments: dict) -> None: ...

    async def started(self, *, call_id: str, tool: str) -> None: ...

    async def finished(
        self, *, call_id: str, tool: str, ok: bool, summary: str, duration_ms: float
    ) -> None: ...

    async def request_approval(self, ask: ApprovalAsk) -> ApprovalReply | None:
        self.asks.append(ask)
        return self.replies.pop(0) if self.replies else None


def build(repo: Path, reply: str, *approvals: ApprovalReply):
    session = Session(id="s_commit", workspace=repo, model="scripted", num_ctx=8192, mode=Mode.AGENT)
    channel = ScriptedChannel(*approvals)
    gateway = ToolGateway(
        registry=ToolRegistry([GitCommitTool()]),
        context=ToolContext(workspace=repo),
        channel=channel,
        policy=make_policy(SessionView(mode="agent", level="supervised"), ConfigView()),
        session_id=session.id,
    )
    provider = ScriptedProvider([ScriptedResponse(reply)])
    runner = ChatRunner(provider=provider, bus=EventBus())
    return session, gateway, runner, provider, channel


def stage_change(repo: Path) -> None:
    (repo / "a.py").write_text("value = 2\n", encoding="utf-8")
    git(repo, "add", "a.py")


# --------------------------------------------------------------- preparation


def test_nothing_staged_is_refused_with_a_way_forward(repo: Path) -> None:
    prep = prepare_commit(repo, estimator=TokenEstimator())

    assert prep.prompt is None
    assert prep.refusal is not None
    assert "nothing is staged" in prep.refusal
    assert "git add" in prep.refusal


def test_unstaged_changes_do_not_count(repo: Path) -> None:
    """Deciding what goes in a commit is the user's call. A modified but unstaged file is
    not something this workflow may quietly include."""
    (repo / "a.py").write_text("value = 99\n", encoding="utf-8")

    assert prepare_commit(repo, estimator=TokenEstimator()).prompt is None


def test_the_prompt_carries_the_staged_diff_and_recent_subjects(repo: Path) -> None:
    stage_change(repo)

    prep = prepare_commit(repo, estimator=TokenEstimator())

    assert prep.prompt is not None
    assert "+value = 2" in prep.prompt
    assert "Add the value" in prep.prompt, "the recent log, so the style can be matched"
    assert prep.files == ("a.py",)
    assert "{{" not in prep.prompt, "every placeholder was filled"


def test_a_brace_heavy_diff_does_not_break_the_template(repo: Path) -> None:
    """Placeholders are replaced, not `str.format`ted: diffs are full of braces."""
    (repo / "a.py").write_text("x = {'a': {1, 2}}  # {0} {name}\n", encoding="utf-8")
    git(repo, "add", "a.py")

    prep = prepare_commit(repo, estimator=TokenEstimator())

    assert prep.prompt is not None
    assert "{name}" in prep.prompt


def test_an_incomplete_diff_says_so_in_the_prompt(repo: Path) -> None:
    """A message that reads as covering a change the model only half saw is a wrong
    message. The note tells it to describe what it can see."""
    for n in range(40):
        (repo / f"f{n}.py").write_text("x = 1\n" * 300, encoding="utf-8")
    git(repo, "add", "-A")

    prep = prepare_commit(repo, estimator=TokenEstimator())

    assert prep.prompt is not None
    assert prep.omitted, "premise: not everything fit"
    assert "incomplete" in prep.prompt


# ------------------------------------------------------------------- cleaning


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Fix the total\n", "Fix the total\n"),
        ("```\nFix the total\n```", "Fix the total\n"),
        ("```text\nFix the total\n```", "Fix the total\n"),
        ('"Fix the total"', "Fix the total\n"),
        ("Here is the commit message:\nFix the total", "Fix the total\n"),
        ("Commit message: Fix the total", "Fix the total\n"),
    ],
)
def test_wrapping_around_the_message_is_stripped(raw: str, expected: str) -> None:
    """A commit whose subject is a code fence is a commit somebody amends later."""
    assert clean_commit_message(raw).text == expected


def test_a_body_is_separated_from_the_subject() -> None:
    """Without the blank line git folds the body into the subject."""
    message = clean_commit_message("Fix the total\nIt was off by one.")

    assert message.text == "Fix the total\n\nIt was off by one.\n"


def test_a_long_subject_warns_but_is_not_rewritten() -> None:
    """Truncating a subject the model wrote would change what it says; the user can edit
    it at the approval prompt."""
    subject = "x" * (SUBJECT_LIMIT + 10)

    message = clean_commit_message(subject)

    assert message.text.strip() == subject
    assert message.warnings


def test_an_empty_reply_is_not_a_message() -> None:
    assert clean_commit_message("   \n").text == ""
    assert clean_commit_message("```\n\n```").text == ""


# ----------------------------------------------------------------- the workflow


async def test_an_approved_commit_lands_with_the_models_message(repo: Path) -> None:
    stage_change(repo)
    session, gateway, runner, _provider, _channel = build(
        repo, "Bump the value to two", ApprovalReply(decision="approve")
    )

    outcome = await run_commit(runner=runner, session=session, gateway=gateway, root=repo)

    assert outcome.status == "committed"
    assert git(repo, "log", "-1", "--format=%s").strip() == "Bump the value to two"


async def test_the_approval_prompt_shows_the_diff_and_the_message(repo: Path) -> None:
    """What the user approves is the staged diff *beside* the message, which is what makes
    approving the model's wording an informed act."""
    stage_change(repo)
    session, gateway, runner, _provider, channel = build(
        repo, "Bump the value to two", ApprovalReply(decision="reject")
    )

    await run_commit(runner=runner, session=session, gateway=gateway, root=repo)

    (ask,) = channel.asks
    assert "Bump the value to two" in ask.preview
    assert "+value = 2" in ask.preview


async def test_a_rejected_commit_commits_nothing(repo: Path) -> None:
    stage_change(repo)
    before = git(repo, "rev-parse", "HEAD")
    session, gateway, runner, _provider, _channel = build(
        repo, "Bump the value to two", ApprovalReply(decision="reject")
    )

    outcome = await run_commit(runner=runner, session=session, gateway=gateway, root=repo)

    assert outcome.status == "rejected"
    assert git(repo, "rev-parse", "HEAD") == before


async def test_no_answer_at_the_prompt_commits_nothing(repo: Path) -> None:
    """Fail closed: a channel with nobody there returns None, and that is a deny."""
    stage_change(repo)
    before = git(repo, "rev-parse", "HEAD")
    session, gateway, runner, _provider, _channel = build(repo, "Bump the value")

    outcome = await run_commit(runner=runner, session=session, gateway=gateway, root=repo)

    assert outcome.status != "committed"
    assert git(repo, "rev-parse", "HEAD") == before


async def test_the_workflow_never_stages(repo: Path) -> None:
    """The model's reply cannot widen what is committed, because the workflow has no path
    by which to stage: an unstaged edit stays unstaged and out of the commit."""
    stage_change(repo)
    (repo / "b.py").write_text("other = 1\n", encoding="utf-8")
    session, gateway, runner, _provider, _channel = build(
        repo, "Bump the value", ApprovalReply(decision="approve")
    )

    await run_commit(runner=runner, session=session, gateway=gateway, root=repo)

    assert git(repo, "show", "--name-only", "--format=", "HEAD").split() == ["a.py"]
    assert "b.py" in git(repo, "status", "--porcelain")


async def test_nothing_staged_never_reaches_the_model(repo: Path) -> None:
    """Refused in code before a token is spent."""
    session, gateway, runner, provider, _channel = build(repo, "unused")

    outcome = await run_commit(runner=runner, session=session, gateway=gateway, root=repo)

    assert outcome.status == "refused"
    assert provider.requests == []


async def test_an_empty_reply_is_reported_not_committed(repo: Path) -> None:
    stage_change(repo)
    before = git(repo, "rev-parse", "HEAD")
    session, gateway, runner, _provider, channel = build(repo, "```\n```")

    outcome = await run_commit(runner=runner, session=session, gateway=gateway, root=repo)

    assert outcome.status == "failed"
    assert channel.asks == [], "no prompt for a message that is not one"
    assert git(repo, "rev-parse", "HEAD") == before


async def test_the_model_call_is_standalone(repo: Path) -> None:
    """One request, no history, no tools: the message is a transformation of the diff, and
    the session's conversation would only be noise in it."""
    stage_change(repo)
    session, gateway, runner, provider, _channel = build(
        repo, "Bump the value", ApprovalReply(decision="reject")
    )
    session.add_user("earlier question")
    session.add_assistant("earlier answer")

    await run_commit(runner=runner, session=session, gateway=gateway, root=repo)

    (request,) = provider.requests
    assert len(request.messages) == 1
    assert request.tools == []
    assert "earlier question" not in request.messages[0].content
    assert len(session.history) == 2, "the workflow added nothing to the history"


async def test_a_hook_that_rejects_the_commit_is_reported(repo: Path) -> None:
    """Hooks run, and never with `--no-verify`. A failing hook is an outcome the user has
    to see, with its output, not a silent non-commit."""
    stage_change(repo)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho 'lint failed' >&2\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    session, gateway, runner, _provider, _channel = build(
        repo, "Bump the value", ApprovalReply(decision="approve")
    )

    outcome = await run_commit(runner=runner, session=session, gateway=gateway, root=repo)

    assert outcome.status == "failed"
    assert "lint failed" in outcome.detail
