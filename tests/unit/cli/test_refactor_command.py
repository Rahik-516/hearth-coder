"""`/refactor` end to end at the REPL — I3.

One scripted model, a real index of a `py_small` copy, real edits through the gateway, and a
real pytest run. The command is composition — `/plan`, review, `/execute`, then checks — so
the value of testing it whole is that the seams are exercised: the task the plan turn
receives carries the index's references, the approval questions come in the right order,
and what is reported afterwards comes from the tree and a test run, not from the model.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from rich.console import Console

from hearth.cli.repl import ChatREPL
from hearth.core.bus import EventBus
from hearth.core.runner import ChatRunner
from hearth.core.session import Mode, SessionStore
from hearth.indexing.pipeline import Indexer
from hearth.llm.scripted_provider import ScriptedProvider, ScriptedResponse
from hearth.llm.types import ToolCall
from hearth.safety.checkpoints import CheckpointStore
from hearth.safety.policy import ConfigView, SessionView
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository
from hearth.tools.base import ToolContext
from hearth.tools.channel import ApprovalAsk, ApprovalReply
from hearth.tools.gateway import ToolGateway, make_policy
from hearth.tools.read_fs import ReadFileTool
from hearth.tools.registry import ToolRegistry
from hearth.tools.tests import RunTestsTool
from hearth.tools.write_fs import EditFileTool

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "repos"

OLD_DOC = '"""The invoice\'s line items are unusable."""'
NEW_DOC = '"""The invoice\'s line items are unusable, or missing."""'


class ScriptedConsole(Console):
    """Answers `input()` from a queue; an unexpected prompt raises rather than blocking."""

    def __init__(self, answers: list[str]) -> None:
        super().__init__(record=True, force_terminal=False, width=140)
        self.answers = list(answers)

    def input(self, prompt="", **kwargs) -> str:  # type: ignore[override]
        if not self.answers:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return self.answers.pop(0)


class Approving:
    async def proposed(self, **_: object) -> None: ...

    async def started(self, **_: object) -> None: ...

    async def finished(self, **_: object) -> None: ...

    async def request_approval(self, ask: ApprovalAsk) -> ApprovalReply:
        return ApprovalReply(decision="approve")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(FIXTURES / "py_small", root)
    return root


def build(workspace: Path, tmp_path: Path, script: list[ScriptedResponse], answers: list[str]):
    state_connection = connect(tmp_path / "state.db")
    migrate(state_connection, database="state")
    state = StateRepository(state_connection)
    store = SessionStore(state)

    index_connection = connect(tmp_path / "index.db")
    migrate(index_connection, database="index")
    Indexer(root=workspace, repository=IndexRepository(index_connection)).run()

    session = store.create(workspace=workspace, model="scripted", num_ctx=8192, mode=Mode.CHAT)
    checkpoints = CheckpointStore(state, BlobStore(tmp_path / "blobs"))
    context = ToolContext(workspace=workspace, checkpoints=checkpoints.bind(session.id))

    def policy(request):
        return make_policy(SessionView(mode=session.mode.value, level="supervised"), ConfigView())(
            request
        )

    gateway = ToolGateway(
        registry=ToolRegistry(
            [
                ReadFileTool(),
                EditFileTool(),
                RunTestsTool(test_command=f"{sys.executable} -m pytest -q -p no:cacheprovider"),
            ]
        ),
        context=context,
        channel=Approving(),
        policy=policy,
        session_id=session.id,
    )
    bus = EventBus()
    provider = ScriptedProvider(script)
    console = ScriptedConsole(answers)
    repl = ChatREPL(
        session=session,
        runner=ChatRunner(provider=provider, bus=bus),
        bus=bus,
        store=store,
        console=console,
        workspace=workspace,
        gateway=gateway,
        grants=set(),
        index_connection=index_connection,
    )
    return repl, console, provider


PLAN = json.dumps(
    {
        "goal": "Clarify the LineItemError docstring",
        "steps": [
            {
                "description": "reword the docstring",
                "files": ["src/billing/errors.py"],
                "change_type": "modify",
            }
        ],
    }
)


def happy_script() -> list[ScriptedResponse]:
    return [
        ScriptedResponse("I read the references."),  # plan exploration
        ScriptedResponse(PLAN),  # plan extraction
        ScriptedResponse(
            "",
            tool_calls=[
                ToolCall(call_id="r1", name="read_file", arguments={"path": "src/billing/errors.py"})
            ],
        ),
        ScriptedResponse(
            "",
            tool_calls=[
                ToolCall(
                    call_id="e1",
                    name="edit_file",
                    arguments={
                        "path": "src/billing/errors.py",
                        "old_string": OLD_DOC,
                        "new_string": NEW_DOC,
                    },
                )
            ],
        ),
        ScriptedResponse("Reworded the docstring."),
    ]


async def test_the_whole_flow_plans_edits_reports_and_runs_the_suite(
    workspace: Path, tmp_path: Path
) -> None:
    # Answers: approve the plan; decline the edit grants (each edit still asks).
    repl, console, provider = build(workspace, tmp_path, happy_script(), ["a", "n"])

    await repl._handle_command("/refactor LineItemError reword its docstring")

    output = console.export_text()
    assert NEW_DOC in (workspace / "src" / "billing" / "errors.py").read_text(encoding="utf-8")
    assert "file(s) reference it" in output
    assert "plan approved" in output
    assert "were modified" in output
    assert "changed    src/billing/errors.py" in output
    assert "tests pass" in output

    plan_request = provider.requests[0]
    task = next(m.content for m in plan_request.messages if m.role == "user")
    assert "LineItemError" in task
    assert "reword its docstring" in task
    assert "src/billing/invoice_service.py" in task, "the index's callers were in the task"


async def test_untouched_callers_are_called_out(workspace: Path, tmp_path: Path) -> None:
    """The failure this command exists to make visible: a refactor that edited the
    definition and none of the places that use it."""
    repl, console, _provider = build(workspace, tmp_path, happy_script(), ["a", "n"])

    await repl._handle_command("/refactor LineItemError reword its docstring")

    output = console.export_text()
    assert "not changed" in output
    assert "check these are still correct" in output
    assert "src/billing/invoice_service.py" in output


async def test_a_rejected_plan_edits_nothing_and_runs_no_tests(
    workspace: Path, tmp_path: Path
) -> None:
    before = (workspace / "src" / "billing" / "errors.py").read_bytes()
    repl, console, provider = build(
        workspace,
        tmp_path,
        [ScriptedResponse("looked"), ScriptedResponse(PLAN)],
        ["r"],
    )

    await repl._handle_command("/refactor LineItemError reword its docstring")

    assert (workspace / "src" / "billing" / "errors.py").read_bytes() == before
    output = console.export_text()
    assert "were modified" not in output
    assert "tests pass" not in output
    assert len(provider.requests) == 2, "no execute turn was started"


async def test_a_bad_target_never_starts_a_plan(workspace: Path, tmp_path: Path) -> None:
    repl, console, provider = build(workspace, tmp_path, [ScriptedResponse("unused")], [])

    await repl._handle_command("/refactor NoSuchThing do something")

    assert "no file or symbol" in console.export_text()
    assert provider.requests == []
    assert repl.session.mode is Mode.CHAT


async def test_a_missing_goal_prints_usage(workspace: Path, tmp_path: Path) -> None:
    repl, console, provider = build(workspace, tmp_path, [ScriptedResponse("unused")], [])

    await repl._handle_command("/refactor LineItemError")

    assert "usage" in console.export_text()
    assert provider.requests == []


async def test_a_second_refactor_does_not_inherit_the_first_plans_approval(
    workspace: Path, tmp_path: Path
) -> None:
    """`_plan` clears the previous approval up front, so a *rejected* second plan cannot be
    mistaken for the first one's approval and executed."""
    script = [*happy_script(), ScriptedResponse("looked"), ScriptedResponse(PLAN)]
    repl, _console, provider = build(workspace, tmp_path, script, ["a", "n", "r"])

    await repl._handle_command("/refactor LineItemError reword its docstring")
    requests_after_first = len(provider.requests)
    await repl._handle_command("/refactor LineItemError something else entirely")

    assert len(provider.requests) == requests_after_first + 2, "plan turn only; no execute"
    assert repl._plans.approved is None
