"""`/plan`'s two-phase turn — I3.

The design claim being tested is that planning is **two requests, not one**: an ordinary
agent loop over read-only tools, then a separate extraction that asks for the schema with
the tools taken away. Both halves have a failure mode that a single-request design walks
straight into, and neither is visible in the returned plan — you have to look at what was
actually sent.

* If the extraction request still carried tools, a model that decides to search once more
  answers with a tool call instead of a plan, and the turn costs a round trip to find out.
* If `format` were set during exploration, the schema would pull the model towards filling
  it in from the question rather than from the code — which is exactly what plan mode is
  for preventing.

So the assertions here are mostly about ``provider.requests``, not about the result. The
result being right is necessary but not sufficient: it would also be right, sometimes, for
a design that is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hearth.core.bus import EventBus
from hearth.core.runner import ChatRunner
from hearth.core.session import Mode, Session
from hearth.llm.scripted_provider import ScriptedProvider, ScriptedResponse
from hearth.llm.types import ToolCall
from hearth.safety.policy import ConfigView, SessionView
from hearth.tools.base import ToolContext
from hearth.tools.channel import NullChannel
from hearth.tools.gateway import ToolGateway, make_policy
from hearth.tools.read_fs import ReadFileTool
from hearth.tools.registry import ToolRegistry
from hearth.tools.write_fs import WriteFileTool

PLAN_JSON = json.dumps(
    {
        "goal": "Add a tax rate to invoices",
        "steps": [
            {
                "description": "add the rate constant",
                "files": ["billing.py"],
                "change_type": "modify",
            }
        ],
        "tests": ["tests/test_billing.py"],
        "risks": [],
        "open_questions": [],
    }
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "billing.py").write_text("TOTAL = 0\n", encoding="utf-8")
    return root


@pytest.fixture
def session(workspace: Path) -> Session:
    return Session(
        id="s_plan", workspace=workspace, model="scripted", num_ctx=8192, mode=Mode.PLAN
    )


def build_gateway(workspace: Path) -> ToolGateway:
    return ToolGateway(
        registry=ToolRegistry([ReadFileTool(), WriteFileTool()]),
        context=ToolContext(workspace=workspace),
        channel=NullChannel(),
        policy=make_policy(SessionView(mode="plan", level="supervised"), ConfigView()),
        session_id="s_plan",
    )


async def run_plan(
    provider: ScriptedProvider, session: Session, workspace: Path, *, schemas=None
):
    runner = ChatRunner(provider=provider, bus=EventBus())
    return await runner.run_plan_turn(
        session,
        "add tax to invoices",
        gateway=build_gateway(workspace),
        tool_schemas=schemas if schemas is not None else [],
    )


# ------------------------------------------------------------------ the shape


async def test_a_plan_comes_back_parsed(session: Session, workspace: Path) -> None:
    provider = ScriptedProvider(
        [ScriptedResponse("I have read billing.py."), ScriptedResponse(PLAN_JSON)]
    )

    result = await run_plan(provider, session, workspace)

    assert result.ok
    assert result.plan is not None
    assert result.plan.goal == "Add a tax rate to invoices"
    assert result.plan.files == ["billing.py"]


async def test_exploration_and_extraction_are_separate_requests(
    session: Session, workspace: Path
) -> None:
    provider = ScriptedProvider(
        [ScriptedResponse("looked around"), ScriptedResponse(PLAN_JSON)]
    )

    await run_plan(provider, session, workspace)

    assert len(provider.requests) == 2, "one loop turn plus one extraction"


async def test_only_the_extraction_request_carries_the_schema(
    session: Session, workspace: Path
) -> None:
    """`format` during exploration would make the model answer before it had looked."""
    provider = ScriptedProvider(
        [ScriptedResponse("looked around"), ScriptedResponse(PLAN_JSON)]
    )

    await run_plan(provider, session, workspace)

    exploration, extraction = provider.requests
    assert exploration.format is None
    assert extraction.format is not None
    assert "goal" in extraction.format["properties"]


async def test_the_extraction_request_has_no_tools(session: Session, workspace: Path) -> None:
    """Otherwise a model inclined to search once more answers with a tool call, and the
    only way to discover that is to spend the round trip."""
    schemas = [ReadFileTool().schema()]
    provider = ScriptedProvider(
        [ScriptedResponse("looked around"), ScriptedResponse(PLAN_JSON)]
    )

    await run_plan(provider, session, workspace, schemas=schemas)

    exploration, extraction = provider.requests
    assert exploration.tools, "the exploration phase is where looking happens"
    assert extraction.tools == []


async def test_the_extraction_reuses_the_exploration_transcript(
    session: Session, workspace: Path
) -> None:
    """The plan has to be written from what was read, not from the question again.

    Driven through a real tool call so the transcript contains a tool result — the thing
    that would be missing if extraction started from a fresh message list.
    """
    call = ToolCall(call_id="c1", name="read_file", arguments={"path": "billing.py"})
    provider = ScriptedProvider(
        [
            ScriptedResponse("", tool_calls=[call]),
            ScriptedResponse("billing.py has TOTAL"),
            ScriptedResponse(PLAN_JSON),
        ]
    )

    await run_plan(provider, session, workspace, schemas=[ReadFileTool().schema()])

    extraction = provider.requests[-1]
    roles = [message.role for message in extraction.messages]
    assert "tool" in roles, "the file the model read is still in the transcript"
    assert any("TOTAL" in (message.content or "") for message in extraction.messages)


# ---------------------------------------------------------------- failure paths


async def test_prose_instead_of_a_plan_is_reported_not_swallowed(
    session: Session, workspace: Path
) -> None:
    """The one failure that must never be quiet.

    An empty plan here would reach `/execute`, which would seed no todos, run nothing, and
    report a finished task. A visible error costs the user one retry; a silent one costs
    them their trust in every later run.
    """
    provider = ScriptedProvider(
        [ScriptedResponse("looked around"), ScriptedResponse("I'd start with billing.py.")]
    )

    result = await run_plan(provider, session, workspace)

    assert not result.ok
    assert result.plan is None
    assert "JSON" in result.error
    assert "billing.py" in result.raw, "the user needs to see what the model actually said"


async def test_a_failed_plan_does_not_enter_the_history(
    session: Session, workspace: Path
) -> None:
    """A rejected non-plan in the history would be quoted back on the next attempt as if
    it had been accepted, making the second `/plan` more likely to fail than the first."""
    provider = ScriptedProvider(
        [ScriptedResponse("looked around"), ScriptedResponse("not json")]
    )

    await run_plan(provider, session, workspace)

    assert session.history == []


async def test_a_successful_plan_enters_the_history_as_the_rendered_plan(
    session: Session, workspace: Path
) -> None:
    """The rendered plan, not the raw JSON and not the exploration prose.

    The exploration text is a draft written before the model had read anything; keeping
    both would put two descriptions of the same intent in the window, and a small model
    will sometimes follow the draft.
    """
    provider = ScriptedProvider(
        [ScriptedResponse("my first thought is to edit everything"), ScriptedResponse(PLAN_JSON)]
    )

    await run_plan(provider, session, workspace)

    assistant = [message for message in session.history if message.role == "assistant"]
    assert len(assistant) == 1
    assert "Add a tax rate to invoices" in (assistant[0].content or "")
    assert "my first thought" not in (assistant[0].content or "")
