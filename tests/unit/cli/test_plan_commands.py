"""`/plan` and `/execute` in the REPL — I3.

These tests are about **grants**, not about rendering. An approved plan can pre-authorise
edits to the files it names (docs/system-design.md §8.3 step 5), which makes this the one
place in the CLI where a frontend widens what the policy engine will allow. Everything
else here — the review prompt, the mode switch — exists to get that decision in front of a
person, so the assertions are on the grant set before and after each answer.

The lifecycle matters as much as the grant itself. A permission that is easy to add and
hard to remove is not a permission, it is a mode change, and the cases below pin down the
three ways it must come back off: a superseding plan, `/clear`, and never having been
granted in the first place.

Console input is answered by a scripted queue rather than a terminal. An empty queue is a
test asking a question it did not mean to, so it raises instead of blocking.
"""

from __future__ import annotations

import json
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
from hearth.tools.channel import NullChannel
from hearth.tools.gateway import ToolGateway, make_policy
from hearth.tools.read_fs import ReadFileTool
from hearth.tools.registry import ToolRegistry
from hearth.tools.write_fs import WriteFileTool


def plan_json(*files: str, goal: str = "Add tax") -> str:
    return json.dumps(
        {
            "goal": goal,
            "steps": [{"description": "do it", "files": list(files), "change_type": "modify"}],
        }
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "billing.py").write_text("TOTAL = 0\n", encoding="utf-8")
    (root / "rates.py").write_text("RATE = 0\n", encoding="utf-8")
    return root


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    connection = connect(tmp_path / "state.db")
    migrate(connection, database="state")
    return SessionStore(StateRepository(connection))


class ScriptedConsole(Console):
    """A console whose `input` comes from a queue.

    Raises on an empty queue rather than returning a default: a test that reaches an
    unanswered prompt has discovered a question the code asks and the test did not know
    about, and silently answering it would hide exactly that.
    """

    def __init__(self, answers: list[str]) -> None:
        super().__init__(force_terminal=False, no_color=True, width=100)
        self.answers = list(answers)
        self.asked: list[str] = []

    def input(self, prompt="", **kwargs) -> str:  # type: ignore[override]
        self.asked.append(str(prompt))
        if not self.answers:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return self.answers.pop(0)


def build_repl(
    workspace: Path,
    store: SessionStore,
    *,
    answers: list[str],
    script: list[ScriptedResponse],
) -> tuple[ChatREPL, set[str], ScriptedConsole]:
    # Created through the store rather than constructed: `/execute` runs a real turn,
    # and persisting its messages needs a session row the foreign key can point at.
    session = store.create(
        workspace=workspace, model="scripted", num_ctx=8192, mode=Mode.CHAT
    )
    bus = EventBus()
    grants: set[str] = set()
    gateway = ToolGateway(
        registry=ToolRegistry([ReadFileTool(), WriteFileTool()]),
        context=ToolContext(workspace=workspace),
        channel=NullChannel(),
        policy=make_policy(SessionView(mode="plan", level="supervised"), ConfigView()),
        on_grant=grants.add,
        session_id=session.id,
    )
    console = ScriptedConsole(answers)
    repl = ChatREPL(
        session=session,
        runner=ChatRunner(provider=ScriptedProvider(script), bus=bus),
        bus=bus,
        store=store,
        console=console,
        workspace=workspace,
        gateway=gateway,
        grants=grants,
    )
    return repl, grants, console


def script_for(*plans: str) -> list[ScriptedResponse]:
    """Exploration prose then a plan, for each plan in turn."""
    entries: list[ScriptedResponse] = []
    for plan in plans:
        entries.append(ScriptedResponse("I looked at billing.py."))
        entries.append(ScriptedResponse(plan))
    return entries


# ---------------------------------------------------------------- the basics


async def test_plan_switches_into_plan_mode(workspace: Path, store: SessionStore) -> None:
    """Read-only for the duration, so a model that decides mid-thought to just make the
    change has nothing to make it with."""
    repl, _grants, _console = build_repl(
        workspace, store, answers=["r"], script=script_for(plan_json("billing.py"))
    )

    await repl._handle_command("/plan add tax")

    assert repl.session.mode is Mode.PLAN


async def test_plan_without_a_task_explains_itself(workspace: Path, store: SessionStore) -> None:
    repl, _grants, _console = build_repl(workspace, store, answers=[], script=script_for(""))

    await repl._handle_command("/plan")

    assert repl.session.mode is Mode.CHAT, "no turn was run, so no mode change"


async def test_rejecting_a_plan_leaves_nothing_approved(
    workspace: Path, store: SessionStore
) -> None:
    repl, grants, _console = build_repl(
        workspace, store, answers=["r"], script=script_for(plan_json("billing.py"))
    )

    await repl._handle_command("/plan add tax")

    assert repl._plans.approved is None
    assert grants == set()


async def test_a_plan_the_model_botched_asks_nothing(
    workspace: Path, store: SessionStore
) -> None:
    """No review prompt for a plan that does not exist. `ScriptedConsole` enforces this
    by raising, so reaching the prompt fails the test rather than hanging it."""
    repl, _grants, _console = build_repl(
        workspace,
        store,
        answers=[],
        script=[ScriptedResponse("looked"), ScriptedResponse("not json at all")],
    )

    await repl._handle_command("/plan add tax")

    assert repl._plans.approved is None


# --------------------------------------------------------------- the grants


async def test_approval_alone_grants_nothing(workspace: Path, store: SessionStore) -> None:
    """The two questions are separate on purpose.

    "Is this the right change?" and "may it happen without asking me again?" are different
    decisions, and folding the second into the first would make every approved plan a
    blanket edit permission.
    """
    repl, grants, console = build_repl(
        workspace, store, answers=["a", "n"], script=script_for(plan_json("billing.py"))
    )

    await repl._handle_command("/plan add tax")

    assert repl._plans.approved is not None, "the plan itself was approved"
    assert grants == set(), "and it still asks before each edit"
    assert len(console.asked) == 2, "approval, then grants — two questions"


async def test_opting_in_grants_exactly_the_planned_files(
    workspace: Path, store: SessionStore
) -> None:
    repl, grants, _console = build_repl(
        workspace,
        store,
        answers=["a", "y"],
        script=script_for(plan_json("billing.py", "rates.py")),
    )

    await repl._handle_command("/plan add tax")

    assert grants == {"edit:billing.py", "edit:rates.py"}


async def test_a_refused_path_is_never_granted(workspace: Path, store: SessionStore) -> None:
    """A plan naming `.git/hooks/pre-commit` must not be able to pre-authorise it, however
    enthusiastically the user approved the plan. The legitimate file still is."""
    repl, grants, _console = build_repl(
        workspace,
        store,
        answers=["a", "y"],
        script=script_for(plan_json("billing.py", ".git/hooks/pre-commit")),
    )

    await repl._handle_command("/plan add tax")

    assert grants == {"edit:billing.py"}


async def test_a_second_plan_revokes_the_first_plans_grants(
    workspace: Path, store: SessionStore
) -> None:
    """The moment that matters is the *start* of the second `/plan`, not the end of it.

    The user is now looking at a different task. If they walk away mid-review, the grants
    from a plan they have moved on from must not still be live.
    """
    repl, grants, _console = build_repl(
        workspace,
        store,
        answers=["a", "y", "r"],
        script=script_for(plan_json("billing.py"), plan_json("rates.py")),
    )

    await repl._handle_command("/plan add tax")
    assert grants == {"edit:billing.py"}

    await repl._handle_command("/plan something else")

    assert grants == set(), "the first plan's grants did not survive the second /plan"


async def test_clear_revokes_the_grants(workspace: Path, store: SessionStore) -> None:
    """Otherwise an approved plan's edit grants outlive the history that explained them,
    and later writes are auto-approved for a task nothing on screen mentions."""
    repl, grants, _console = build_repl(
        workspace, store, answers=["a", "y"], script=script_for(plan_json("billing.py"))
    )
    await repl._handle_command("/plan add tax")
    assert grants

    await repl._handle_command("/clear")

    assert grants == set()
    assert repl._plans.approved is None


async def test_two_plans_never_share_a_grant_key(workspace: Path, store: SessionStore) -> None:
    """Ids are monotonic, so a revoked key can never be re-created by a later plan and
    quietly come back into force."""
    repl, _grants, _console = build_repl(
        workspace,
        store,
        answers=["a", "n", "a", "n"],
        script=script_for(plan_json("billing.py"), plan_json("rates.py")),
    )

    await repl._handle_command("/plan one")
    first = repl._plans.approved
    assert first is not None
    first_key = first.grant_key

    await repl._handle_command("/plan two")
    second = repl._plans.approved

    assert second is not None
    assert second.grant_key != first_key


# --------------------------------------------------------------- /execute


async def test_execute_without_a_plan_does_nothing(workspace: Path, store: SessionStore) -> None:
    repl, _grants, _console = build_repl(workspace, store, answers=[], script=script_for(""))

    await repl._handle_command("/execute")

    assert repl.session.mode is Mode.CHAT


async def test_execute_switches_to_agent_and_restates_the_plan(
    workspace: Path, store: SessionStore
) -> None:
    """Restated verbatim, not referred to.

    The plan was produced under a different system prompt and possibly before a
    compaction, so "follow the plan above" can point at nothing at all.
    """
    repl, _grants, _console = build_repl(
        workspace,
        store,
        answers=["a", "n"],
        script=[
            ScriptedResponse("I looked at billing.py."),
            ScriptedResponse(plan_json("billing.py", goal="Add tax to invoices")),
            ScriptedResponse("Done."),
        ],
    )
    await repl._handle_command("/plan add tax")

    await repl._handle_command("/execute")

    assert repl.session.mode is Mode.AGENT
    users = [m for m in repl.session.history if m.role == "user"]
    assert any("Add tax to invoices" in (m.content or "") for m in users)
    assert any("todo_write" in (m.content or "") for m in users)


async def test_the_agent_turn_gets_agent_tools_not_plan_tools(
    workspace: Path, store: SessionStore
) -> None:
    """The regression this guards: schemas captured once at construction.

    A REPL holding one fixed list would hand the agent turn whatever set was current when
    it started — in practice plan mode's read-only tools, so every edit the model proposed
    would be a call to a tool it had never been shown.
    """
    repl, _grants, _console = build_repl(workspace, store, answers=[], script=script_for(""))

    plan_tools = {schema["function"]["name"] for schema in repl._schemas_for(Mode.PLAN)}
    agent_tools = {schema["function"]["name"] for schema in repl._schemas_for(Mode.AGENT)}

    assert "write_file" not in plan_tools
    assert "write_file" in agent_tools
