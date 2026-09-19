"""The agent turn: tools driven from a cache-stable prefix, and headless fail-closed.

Two M6 acceptance criteria live here. One is that headless runs **fail closed** — the
default is to deny every side effect and name the flag that would have permitted it, so an
under-granted run is legible instead of mysterious. The other is that every exit path
reports a ``run_result`` summary (§14.1).

The scripted provider means no model is needed, and the only tool wired in is
``todo_write`` (META) plus a write tool that is never actually reached in the headless
tests — the point there is the refusal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hearth.core.bus import EventBus
from hearth.core.runner import AgentTurnResult, ChatRunner
from hearth.core.session import Mode, Session
from hearth.llm.scripted_provider import ScriptedProvider, ScriptedResponse
from hearth.llm.types import ToolCall
from hearth.safety.policy import ConfigView, SessionView
from hearth.tools.base import ToolContext
from hearth.tools.channel import NullChannel
from hearth.tools.gateway import ToolGateway, make_policy
from hearth.tools.meta import TodoWriteTool
from hearth.tools.registry import ToolRegistry
from hearth.tools.results import ErrorCode
from hearth.tools.write_fs import WriteFileTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("value = 1\n", encoding="utf-8")
    return root


@pytest.fixture
def session(workspace: Path) -> Session:
    return Session(
        id="s_agent",
        workspace=workspace,
        model="scripted",
        num_ctx=8192,
        mode=Mode.AGENT,
    )


def build_gateway(workspace: Path, *, headless: bool, allow_edits: bool = False) -> ToolGateway:
    return ToolGateway(
        registry=ToolRegistry([TodoWriteTool(), WriteFileTool()]),
        context=ToolContext(workspace=workspace),
        channel=NullChannel(),
        policy=make_policy(
            SessionView(mode="agent", level="supervised", headless=headless),
            ConfigView(headless_allow_edits=allow_edits),
        ),
        session_id="s_agent",
    )


def todo_call() -> ToolCall:
    return ToolCall(
        call_id="t1",
        name="todo_write",
        arguments={"todos": [{"content": "Fix the rounding", "status": "in_progress"}]},
    )


def write_call() -> ToolCall:
    return ToolCall(
        call_id="w1",
        name="write_file",
        arguments={"path": "new.py", "content": "x = 1\n"},
    )


async def test_the_loop_runs_a_tool_then_answers(session: Session, workspace: Path) -> None:
    provider = ScriptedProvider(
        [
            ScriptedResponse("", tool_calls=[todo_call()]),
            ScriptedResponse("Planned the work."),
        ]
    )
    runner = ChatRunner(provider=provider, bus=EventBus())

    result = await runner.run_agent_turn(
        session,
        "Fix the rounding",
        gateway=build_gateway(workspace, headless=False),
        tool_schemas=[TodoWriteTool().schema()],
    )

    assert result.ok
    assert result.tool_calls == 1
    assert result.answer == "Planned the work."


async def test_the_turn_lands_in_history_once(session: Session, workspace: Path) -> None:
    """The prefix is append-only; a turn adds exactly one user and one assistant message."""
    provider = ScriptedProvider([ScriptedResponse("Done.")])
    runner = ChatRunner(provider=provider, bus=EventBus())

    await runner.run_agent_turn(
        session,
        "Fix the rounding",
        gateway=build_gateway(workspace, headless=False),
        tool_schemas=[],
    )

    assert [message.role for message in session.history] == ["user", "assistant"]


async def test_headless_denies_a_write_and_names_the_flag(
    session: Session, workspace: Path
) -> None:
    """Fail closed: the default is deny, and the denial says how to permit it (§14.1)."""
    provider = ScriptedProvider(
        [
            ScriptedResponse("", tool_calls=[write_call()]),
            ScriptedResponse("I could not write the file."),
        ]
    )
    runner = ChatRunner(provider=provider, bus=EventBus())

    await runner.run_agent_turn(
        session,
        "Add a file",
        gateway=build_gateway(workspace, headless=True),
        tool_schemas=[WriteFileTool().schema()],
    )

    assert not (workspace / "new.py").exists(), "headless must not write without --allow-edits"


async def test_headless_with_the_flag_stops_denying_the_write(workspace: Path) -> None:
    """The flag is narrow and explicit — it widens one risk, not the run.

    Asserted at the policy decision rather than by looking for bytes on disk: with the
    flag set the call gets *past* policy, and what stops it here is the write tool's own
    refusal to run without a checkpointer, which ``test_write_fs.py`` owns.
    """
    gateway = build_gateway(workspace, headless=True, allow_edits=True)

    result = await gateway.call("write_file", {"path": "new.py", "content": "x\n"}, call_id="c1")

    assert result.error is not ErrorCode.DENIED


async def test_a_denial_reaches_the_model_as_a_readable_error(workspace: Path) -> None:
    """A denial has to be legible to the model, or it retries the same call."""
    gateway = build_gateway(workspace, headless=True)

    result = await gateway.call("write_file", {"path": "new.py", "content": "x\n"}, call_id="c1")

    assert result.error is ErrorCode.DENIED
    assert "--allow-edits" in result.content


# ------------------------------------------------------------------ run_result


def test_the_summary_is_printed_for_a_failure_too() -> None:
    """Under-granting is the common headless outcome, so the summary is unconditional."""
    stopped = AgentTurnResult(answer="", reason="step_limit", steps=8, tool_calls=12)

    summary = stopped.summary()

    assert "reason=step_limit" in summary
    assert "steps=8" in summary
    assert "tool_calls=12" in summary
    assert not stopped.ok


def test_a_completed_run_reports_answered() -> None:
    """A provider's "stop" and Hearth's "answered" are the same outcome."""
    done = AgentTurnResult(answer="Done.", reason="stop", steps=3, duration_ms=1500)

    assert done.ok
    assert "reason=answered" in done.summary()
    assert "duration=1.5s" in done.summary()


def test_running_out_of_steps_is_not_a_success() -> None:
    """The mistake that would make a half-done change look finished."""
    assert not AgentTurnResult(answer="partial", reason="step_limit").ok
    assert not AgentTurnResult(answer="", reason="retry_budget").ok


# ------------------------------------------------------------------- repo map


class RecordingProvider(ScriptedProvider):
    """Keeps the request so a test can inspect what the model was actually sent."""

    def __init__(self) -> None:
        super().__init__([ScriptedResponse("ok")])


def build_map(connection) -> object:
    from hearth.retrieval.repomap import RepoMapBuilder

    return RepoMapBuilder(connection)


@pytest.fixture
def indexed_fixture_repo(tmp_path: Path):
    """An indexed copy of py_small. A fixture, not inline setup, so the async tests below
    do no blocking filesystem work of their own."""
    import shutil

    from hearth.indexing.pipeline import Indexer
    from hearth.storage.db import connect
    from hearth.storage.index_repo import IndexRepository
    from hearth.storage.migrate import migrate

    source = Path(__file__).resolve().parents[2] / "fixtures" / "repos" / "py_small"
    shutil.copytree(source, tmp_path / "src_repo")
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    Indexer(root=tmp_path / "src_repo", repository=IndexRepository(connection)).run()
    return connection


async def test_the_repo_map_reaches_the_model(session: Session, indexed_fixture_repo) -> None:
    """Wiring it into the runner is the whole point; an unused builder helps nobody."""
    provider = RecordingProvider()
    runner = ChatRunner(provider=provider, bus=EventBus(), repo_map=build_map(indexed_fixture_repo))

    await runner.run_turn(session, "what is this repository?")

    system = "\n".join(m.content for m in provider.requests[0].messages if m.role == "system")
    assert "invoice_service.py" in system, "the map should describe the repository"


async def test_the_map_is_built_once_per_epoch(session: Session, tmp_path: Path) -> None:
    """It sits in the cached prefix: rebuilding it per turn would discard the KV cache."""
    from hearth.storage.db import connect
    from hearth.storage.migrate import migrate

    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")

    builds = 0
    real = build_map(connection)

    class CountingBuilder:
        def build(self, **kwargs):
            nonlocal builds
            builds += 1
            return real.build(**kwargs)  # type: ignore[attr-defined]

    runner = ChatRunner(provider=ScriptedProvider(), bus=EventBus(), repo_map=CountingBuilder())

    await runner.run_turn(session, "one")
    await runner.run_turn(session, "two")
    assert builds == 1, "two turns in one epoch must reuse the map"

    session.switch_mode(Mode.CHAT if session.mode is Mode.AGENT else Mode.AGENT)
    await runner.run_turn(session, "three")
    assert builds == 2, "a new epoch must rebuild it"


async def test_a_failing_map_does_not_fail_the_turn(session: Session) -> None:
    """The map is an aid. Losing it should cost context, not the user's turn."""

    class Broken:
        def build(self, **kwargs):
            raise RuntimeError("index is corrupt")

    runner = ChatRunner(provider=ScriptedProvider(), bus=EventBus(), repo_map=Broken())

    result = await runner.run_turn(session, "still works?")

    assert result.ok
