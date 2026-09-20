"""Batch review through the agent loop — the whole path, §6.4.

A scripted model proposes two edits in one step. The real loop, gateway, bus adapter and
terminal prompt handle them, and the observable is what the *user* was asked: one screen
rather than two prompts, and a per-file answer that actually reaches the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from hearth.cli.approval import ApprovalPrompt
from hearth.core.bus import EventBus
from hearth.core.runner import ChatRunner
from hearth.core.session import Mode, Session
from hearth.core.tool_channel import EventBusChannel
from hearth.llm.scripted_provider import ScriptedProvider, ScriptedResponse
from hearth.llm.types import ToolCall
from hearth.safety.checkpoints import CheckpointStore
from hearth.safety.policy import ConfigView, SessionView
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository
from hearth.tools.base import ToolContext
from hearth.tools.gateway import ToolGateway, make_policy
from hearth.tools.registry import build_default_registry

#: AWS's documented example key: inert, and shaped enough to raise the SECRET? badge.
FAKE_KEY = "AKIAIOSFODNN7EXAMPLE"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("a", "b"):
        (root / f"{name}.py").write_text(f"value_{name} = 1\n", encoding="utf-8")
    return root


def edit(name: str, call_id: str, new: str) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        name="edit_file",
        arguments={
            "path": f"{name}.py",
            "old_string": f"value_{name} = 1",
            "new_string": f"value_{name} = {new}",
        },
    )


def read(name: str, call_id: str) -> ToolCall:
    return ToolCall(call_id=call_id, name="read_file", arguments={"path": f"{name}.py"})


async def run_turn(workspace: Path, tmp_path: Path, answers: list[str], second_value: str = "2"):
    connection = connect(tmp_path / "state.db")
    migrate(connection, database="state")
    state = StateRepository(connection)
    checkpoints = CheckpointStore(state, BlobStore(tmp_path / "blobs"))
    session_row = state.create_session(workspace=str(workspace))

    bus = EventBus()
    gateway = ToolGateway(
        registry=build_default_registry(),
        context=ToolContext(workspace=workspace, checkpoints=checkpoints.bind(session_row.id)),
        channel=EventBusChannel(bus),
        policy=make_policy(SessionView(mode="agent", level="supervised"), ConfigView()),
        session_id=session_row.id,
    )

    asked: list[str] = []
    remaining = list(answers)

    async def ask(prompt: str) -> str:
        asked.append(prompt)
        if not remaining:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return remaining.pop(0)

    console = Console(record=True, force_terminal=False, width=120)
    bus.subscribe(ApprovalPrompt(bus=bus, console=console, ask=ask))

    provider = ScriptedProvider(
        [
            # Read both files first (read-before-write), then propose both edits together.
            ScriptedResponse("", tool_calls=[read("a", "r1"), read("b", "r2")]),
            ScriptedResponse("", tool_calls=[edit("a", "e1", "2"), edit("b", "e2", second_value)]),
            ScriptedResponse("Done."),
        ]
    )
    session = Session(
        id=session_row.id, workspace=workspace, model="scripted", num_ctx=8192, mode=Mode.AGENT
    )
    runner = ChatRunner(provider=provider, bus=bus)
    schemas = gateway.availability("agent").schemas

    result = await runner.run_agent_turn(session, "change both", gateway=gateway, tool_schemas=schemas)
    return result, asked, console.export_text()


def text(workspace: Path, name: str) -> str:
    return (workspace / f"{name}.py").read_text(encoding="utf-8")


async def test_two_edits_in_one_step_are_one_prompt(workspace: Path, tmp_path: Path) -> None:
    result, asked, output = await run_turn(workspace, tmp_path, ["a"])

    assert result.ok
    assert len(asked) == 1, "one screen, not one prompt per file"
    assert "2 file changes proposed in one step" in output
    assert "value_a = 2" in text(workspace, "a")
    assert "value_b = 2" in text(workspace, "b")


async def test_a_per_file_answer_reaches_the_filesystem(workspace: Path, tmp_path: Path) -> None:
    """Review one by one: approve a.py, reject b.py. Only a.py changes."""
    result, _asked, _output = await run_turn(workspace, tmp_path, ["i", "y", "n"])

    assert result.ok
    assert "value_a = 2" in text(workspace, "a")
    assert text(workspace, "b") == "value_b = 1\n"


async def test_rejecting_the_batch_changes_nothing(workspace: Path, tmp_path: Path) -> None:
    _result, _asked, _output = await run_turn(workspace, tmp_path, ["n", ""])

    assert text(workspace, "a") == "value_a = 1\n"
    assert text(workspace, "b") == "value_b = 1\n"


async def test_a_secret_in_one_file_removes_approve_all_for_the_whole_step(
    workspace: Path, tmp_path: Path
) -> None:
    """The end-to-end form of §6.4's rule: one flagged file and the person cannot approve
    everything with one keypress."""
    _result, _asked, output = await run_turn(
        workspace, tmp_path, ["a", "i", "y", "n"], second_value=f"'{FAKE_KEY}'"
    )

    assert "approve-all is unavailable: SECRET?" in output
    assert "[a] approve all" not in output
    assert "value_a = 2" in text(workspace, "a")
    assert FAKE_KEY not in text(workspace, "b"), "the flagged file was rejected when walked"
