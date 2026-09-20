"""`/commit` and `/review` at the REPL — I3.

The workflows have their own tests; this file is about what the *command* adds: the mode
switch `/commit` needs, the refusals reaching the user as words, and both commands staying
out of the conversation history.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from rich.console import Console

from hearth.cli.repl import ChatREPL
from hearth.core.bus import EventBus
from hearth.core.runner import ChatRunner
from hearth.core.session import Mode, SessionStore
from hearth.llm.scripted_provider import ScriptedProvider, ScriptedResponse
from hearth.safety.policy import ConfigView, SessionView
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository
from hearth.tools.base import ToolContext
from hearth.tools.channel import ApprovalAsk, ApprovalReply
from hearth.tools.gateway import ToolGateway, make_policy
from hearth.tools.git_write import GitCommitTool
from hearth.tools.registry import ToolRegistry


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


class Approving:
    """A channel that approves everything, for tests where the prompt is not the subject."""

    async def proposed(self, *, call_id: str, tool: str, arguments: dict) -> None: ...

    async def started(self, *, call_id: str, tool: str) -> None: ...

    async def finished(
        self, *, call_id: str, tool: str, ok: bool, summary: str, duration_ms: float
    ) -> None: ...

    async def request_approval(self, ask: ApprovalAsk) -> ApprovalReply | None:
        return ApprovalReply(decision="approve")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Hearth Test")
    (root / "a.py").write_text("value = 1\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "baseline")
    return root


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    connection = connect(tmp_path / "state.db")
    migrate(connection, database="state")
    return SessionStore(StateRepository(connection))


def build_repl(repo: Path, store: SessionStore, reply: str):
    session = store.create(workspace=repo, model="scripted", num_ctx=8192, mode=Mode.CHAT)
    bus = EventBus()
    session_ref = session

    def policy(request):
        # Live view of the session's mode, as the real gateway's closure has: /commit has
        # to *actually* be in agent mode for `git_commit` to be allowed to ask.
        return make_policy(
            SessionView(mode=session_ref.mode.value, level="supervised"), ConfigView()
        )(request)

    gateway = ToolGateway(
        registry=ToolRegistry([GitCommitTool()]),
        context=ToolContext(workspace=repo),
        channel=Approving(),
        policy=policy,
        session_id=session.id,
    )
    console = Console(record=True, force_terminal=False, width=120)
    provider = ScriptedProvider([ScriptedResponse(reply)])
    repl = ChatREPL(
        session=session,
        runner=ChatRunner(provider=provider, bus=bus),
        bus=bus,
        store=store,
        console=console,
        workspace=repo,
        gateway=gateway,
    )
    return repl, console, provider


# ------------------------------------------------------------------- /commit


async def test_commit_switches_to_agent_mode_and_commits(repo: Path, store: SessionStore) -> None:
    """Without the switch, policy refuses `git_commit` in chat mode and the command would
    report a mode restriction the user never asked about."""
    (repo / "a.py").write_text("value = 2\n", encoding="utf-8")
    git(repo, "add", "a.py")
    repl, console, _provider = build_repl(repo, store, "Bump the value")

    await repl._handle_command("/commit")

    assert repl.session.mode is Mode.AGENT
    assert git(repo, "log", "-1", "--format=%s").strip() == "Bump the value"
    assert "mode: agent" in console.export_text()


async def test_commit_with_nothing_staged_explains_and_does_not_switch_mode_needlessly(
    repo: Path, store: SessionStore
) -> None:
    repl, console, provider = build_repl(repo, store, "unused")

    await repl._handle_command("/commit")

    assert "nothing is staged" in console.export_text()
    assert provider.requests == []


async def test_commit_leaves_the_conversation_alone(repo: Path, store: SessionStore) -> None:
    """A commit message is a transformation of a diff, not a turn of the conversation."""
    (repo / "a.py").write_text("value = 2\n", encoding="utf-8")
    git(repo, "add", "a.py")
    repl, _console, _provider = build_repl(repo, store, "Bump the value")

    await repl._handle_command("/commit")

    assert repl.session.history == []


async def test_commit_is_unavailable_without_a_gateway(repo: Path, store: SessionStore) -> None:
    repl, console, _provider = build_repl(repo, store, "x")
    repl.gateway = None

    await repl._handle_command("/commit")

    assert "unavailable" in console.export_text()


# ------------------------------------------------------------------- /review


async def test_review_prints_findings_and_the_citation_count(
    repo: Path, store: SessionStore
) -> None:
    (repo / "a.py").write_text("value = 2\n", encoding="utf-8")
    repl, console, _provider = build_repl(repo, store, "**Off by one** — `a.py:1`\nWrong value.")

    await repl._handle_command("/review")

    output = console.export_text()
    assert "Off by one" in output
    assert "1/1 citation(s) verified" in output


async def test_review_flags_an_invented_citation(repo: Path, store: SessionStore) -> None:
    (repo / "a.py").write_text("value = 2\n", encoding="utf-8")
    repl, console, _provider = build_repl(repo, store, "**Crash** — `a.py:400`\nBad.")

    await repl._handle_command("/review")

    output = console.export_text()
    assert "could not be verified" in output
    assert "0/1 citation(s) verified" in output


async def test_review_does_not_change_mode_or_history(repo: Path, store: SessionStore) -> None:
    """Read-only, and it must not cost a re-prefill by switching anything."""
    (repo / "a.py").write_text("value = 2\n", encoding="utf-8")
    repl, _console, _provider = build_repl(repo, store, "Nothing of note.")

    await repl._handle_command("/review")

    assert repl.session.mode is Mode.CHAT
    assert repl.session.history == []


async def test_review_staged_flag_is_honoured(repo: Path, store: SessionStore) -> None:
    (repo / "a.py").write_text("value = 2\n", encoding="utf-8")
    repl, console, provider = build_repl(repo, store, "unused")

    await repl._handle_command("/review --staged")

    assert "no staged changes" in console.export_text()
    assert provider.requests == []


async def test_review_with_no_changes_says_so(repo: Path, store: SessionStore) -> None:
    repl, console, _provider = build_repl(repo, store, "unused")

    await repl._handle_command("/review")

    assert "no uncommitted changes" in console.export_text()


async def test_review_prints_code_in_square_brackets_verbatim(
    repo: Path, store: SessionStore
) -> None:
    """A review quotes code, and code is full of `[0]` that Rich would read as markup."""
    (repo / "a.py").write_text("value = items[0]\n", encoding="utf-8")
    repl, console, _provider = build_repl(
        repo, store, "**Index** — `a.py:1`\n`items[0]` fails on an empty list [/bold]."
    )

    await repl._handle_command("/review")

    assert "items[0]" in console.export_text()


# --------------------------------------------------------------------- /test


def stub_workflow(monkeypatch: pytest.MonkeyPatch, outcome, *, adds_history: bool = False):
    """Replace the workflow with a canned outcome, recording how it was called.

    The workflow itself is tested in `test_workflow_test_writer.py` against a real pytest
    run. What is under test here is what the *command* adds around it.
    """
    from hearth.workflows.test_writer import TestWorkflowOutcome  # noqa: F401

    calls: list[dict] = []

    async def fake(**kwargs):
        calls.append(kwargs)
        if adds_history:
            kwargs["session"].add_user("write tests")
            kwargs["session"].add_assistant("wrote tests")
        return outcome

    monkeypatch.setattr("hearth.cli.repl.run_test_workflow", fake)
    return calls


async def test_test_without_a_target_prints_usage(repo: Path, store: SessionStore) -> None:
    repl, console, _provider = build_repl(repo, store, "x")

    await repl._handle_command("/test")

    assert "usage" in console.export_text()


async def test_test_with_an_unresolvable_target_says_why_and_runs_nothing(
    repo: Path, store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hearth.workflows.test_writer import TestWorkflowOutcome

    calls = stub_workflow(monkeypatch, TestWorkflowOutcome("passed"))
    repl, console, _provider = build_repl(repo, store, "x")

    await repl._handle_command("/test nope.py")

    assert "not a file" in console.export_text()
    assert calls == []
    assert repl.session.mode is Mode.CHAT, "no mode switch for a command that did not run"


async def test_test_switches_to_agent_and_reports_a_pass(
    repo: Path, store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hearth.workflows.test_writer import TestWorkflowOutcome

    calls = stub_workflow(
        monkeypatch,
        TestWorkflowOutcome("passed", test_files=["tests/test_a.py"], iterations=2),
    )
    repl, console, _provider = build_repl(repo, store, "x")

    await repl._handle_command("/test a.py")

    output = console.export_text()
    assert repl.session.mode is Mode.AGENT
    assert "tests pass" in output
    assert "after 1 fix round" in output
    assert calls[0]["target"].path == "a.py"


async def test_test_reports_a_failure_with_the_last_summary(
    repo: Path, store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hearth.workflows.test_writer import TestWorkflowOutcome

    stub_workflow(
        monkeypatch,
        TestWorkflowOutcome(
            "failed", detail="still failing after 3 fix round(s).", summary="1 failed: test_x [tests]"
        ),
    )
    repl, console, _provider = build_repl(repo, store, "x")

    await repl._handle_command("/test a.py")

    output = console.export_text()
    assert "still fail" in output
    assert "1 failed: test_x [tests]" in output, "runner output is printed verbatim"


async def test_test_warns_when_the_source_was_edited(
    repo: Path, store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hearth.workflows.test_writer import TestWorkflowOutcome

    stub_workflow(monkeypatch, TestWorkflowOutcome("passed", test_files=["t.py"], source_modified=True))
    repl, console, _provider = build_repl(repo, store, "x")

    await repl._handle_command("/test a.py")

    output = console.export_text()
    assert "was changed during this run" in output
    assert "/undo" in output


async def test_test_persists_the_turns_the_workflow_added(
    repo: Path, store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workflows drive the runner directly, not through `_ask`, so without this their
    turns would live in memory only and `/resume` would find a hole where the work was."""
    from hearth.workflows.test_writer import TestWorkflowOutcome

    stub_workflow(monkeypatch, TestWorkflowOutcome("passed", test_files=["t.py"]), adds_history=True)
    repl, _console, _provider = build_repl(repo, store, "x")

    await repl._handle_command("/test a.py")

    resumed = store.resume(repl.session.id)
    assert resumed is not None
    assert [m.content for m in resumed.history] == ["write tests", "wrote tests"]


async def test_test_is_unavailable_without_a_gateway(repo: Path, store: SessionStore) -> None:
    repl, console, _provider = build_repl(repo, store, "x")
    repl.gateway = None

    await repl._handle_command("/test a.py")

    assert "unavailable" in console.export_text()
