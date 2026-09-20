"""Structured plans — I3.

Two things are being tested, and only one of them is about plans.

The first is parsing: a 4B model asked for JSON returns JSON *most* of the time, wrapped in
a fence some of the time, and prose occasionally. Each of those has to end somewhere a user
can act on, which for the third case means a clear error rather than an empty plan.

The second is the part that matters. An approved plan **auto-approves edits to the files it
names** (docs/system-design.md §8.3), which makes the file list an input to the permission
system written by the model. Every path in it is treated as hostile here: traversal,
absolute paths, `.git`, Windows separators, and a plan that names a protected path
alongside legitimate ones. The rule throughout is that a refused path costs an approval
prompt, never a widened jail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hearth.core.plan import (
    MAX_STEPS,
    ApprovedPlan,
    Plan,
    PlanError,
    PlanStep,
    PlanStore,
    parse_plan,
    plan_schema,
    scope_grants,
)


def make_plan(*files: str, goal: str = "Add tax handling") -> Plan:
    """A plan whose single step names the given files."""
    return Plan(goal=goal, steps=[PlanStep(description="do the thing", files=list(files))])


# ---------------------------------------------------------------------- parsing


def test_a_well_formed_plan_round_trips() -> None:
    payload = {
        "goal": "Add VAT to invoices",
        "steps": [
            {
                "description": "add a rate table",
                "files": ["src/billing/rates.py"],
                "change_type": "add",
            },
            {"description": "apply it in the total", "files": ["src/billing/models.py"]},
        ],
        "tests": ["tests/test_rates.py"],
        "risks": ["existing invoices assume no VAT"],
        "open_questions": [],
    }

    plan = parse_plan(json.dumps(payload))

    assert plan.goal == "Add VAT to invoices"
    assert len(plan.steps) == 2
    assert plan.steps[1].change_type == "modify", "the default, not an error"
    assert plan.files == ["src/billing/rates.py", "src/billing/models.py"]


def test_a_fenced_plan_is_accepted() -> None:
    """Small models wrap JSON in a fence despite being asked for raw JSON, and the plan
    inside it is perfectly good. Refusing it would fail the user over punctuation."""
    fenced = '```json\n{"goal": "g", "steps": [{"description": "s"}]}\n```'

    assert parse_plan(fenced).goal == "g"


def test_prose_is_a_clear_error_not_an_empty_plan() -> None:
    """This error reaches a person at a `/plan` prompt, so it has to say what happened.

    An empty plan would be worse than the error: `/execute` would accept it and run a
    zero-step task that looks like success.
    """
    with pytest.raises(PlanError, match="valid JSON"):
        parse_plan("Sure! First I would look at the billing module, then...")


def test_an_empty_response_is_an_error() -> None:
    with pytest.raises(PlanError, match="nothing"):
        parse_plan("   ")


def test_a_json_array_is_not_a_plan() -> None:
    with pytest.raises(PlanError, match="not a plan object"):
        parse_plan('[{"description": "s"}]')


def test_a_plan_with_no_steps_is_refused() -> None:
    """A plan is a commitment to do something. Zero steps is a refusal dressed as one,
    and `/execute` would seed an empty todo list and then report success."""
    with pytest.raises(PlanError, match="expected shape"):
        parse_plan('{"goal": "g", "steps": []}')


def test_a_runaway_step_count_is_refused() -> None:
    steps = [{"description": f"step {n}"} for n in range(MAX_STEPS + 1)]

    with pytest.raises(PlanError, match="expected shape"):
        parse_plan(json.dumps({"goal": "g", "steps": steps}))


def test_unknown_fields_are_refused() -> None:
    """`extra="forbid"`, so a model inventing a `command` field cannot smuggle it into the
    plan object on the chance that something downstream reads it."""
    with pytest.raises(PlanError, match="expected shape"):
        parse_plan('{"goal": "g", "steps": [{"description": "s"}], "command": "rm -rf /"}')


def test_the_schema_describes_a_plan() -> None:
    """The schema is sent as `format`, so it is what constrains the model's output."""
    schema = plan_schema()

    assert schema["type"] == "object"
    assert set(schema["required"]) >= {"goal", "steps"}


# ------------------------------------------------------------------- rendering


def test_the_rendered_plan_shows_steps_files_and_questions() -> None:
    plan = Plan(
        goal="Add VAT",
        steps=[PlanStep(description="add a rate table", files=["src/rates.py"], change_type="add")],
        open_questions=["which rate applies to exports?"],
    )

    rendered = plan.render()

    assert "Add VAT" in rendered
    assert "1. [add] add a rate table  (src/rates.py)" in rendered
    assert "which rate applies to exports?" in rendered


def test_empty_sections_are_not_rendered() -> None:
    """A plan with no risks should not print an empty "Risks" heading; the user reads
    this to decide whether to approve, and empty headings train them to skim."""
    assert "Risks" not in make_plan().render()


# -------------------------------------------------- grant scoping (the security part)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return root


def test_ordinary_paths_are_granted(workspace: Path) -> None:
    granted, refused = scope_grants(make_plan("src/billing/models.py"), workspace=workspace)

    assert granted == ("src/billing/models.py",)
    assert refused == ()


@pytest.mark.parametrize(
    "hostile",
    ["../../etc/passwd", "../sibling/secrets.py", "src/../../escape.py"],
)
def test_traversal_is_refused(workspace: Path, hostile: str) -> None:
    """A plan cannot widen the jail. The model writes this list."""
    granted, refused = scope_grants(make_plan(hostile), workspace=workspace)

    assert granted == ()
    assert refused[0][0] == hostile


@pytest.mark.parametrize("absolute", ["/etc/passwd", "/home/hearth/.ssh/id_rsa", "~/.bashrc"])
def test_absolute_and_home_paths_are_refused(workspace: Path, absolute: str) -> None:
    granted, _refused = scope_grants(make_plan(absolute), workspace=workspace)

    assert granted == ()


@pytest.mark.parametrize(
    "protected",
    [".git/config", ".git/hooks/pre-commit", ".hearth/config.toml", "vendor/lib/.git/config"],
)
def test_protected_paths_are_refused(workspace: Path, protected: str) -> None:
    """`.git/hooks/pre-commit` is the one that matters: a writable hook turns the next
    approved commit into arbitrary execution. No plan may pre-authorise it, however
    enthusiastically the user approved the plan."""
    granted, refused = scope_grants(make_plan(protected), workspace=workspace)

    assert granted == ()
    assert "protected" in refused[0][1]


def test_windows_separators_are_normalised(workspace: Path) -> None:
    """Hearth runs in WSL2 against repositories a Windows editor also writes to, so a
    model that has read a Windows traceback will produce backslash paths. They have to
    normalise, or `covers()` compares two spellings of one file and always says no."""
    granted, _refused = scope_grants(make_plan("src\\billing\\models.py"), workspace=workspace)

    assert granted == ("src/billing/models.py",)


def test_a_refused_path_does_not_poison_the_rest(workspace: Path) -> None:
    """The step still runs; it just asks for that one file.

    Refusing the whole plan over one bad path would teach users that plan mode is broken,
    and the pressure would then be to loosen the check rather than to fix the plan.
    """
    granted, refused = scope_grants(make_plan("src/a.py", ".git/config", "src/b.py"), workspace=workspace)

    assert granted == ("src/a.py", "src/b.py")
    assert len(refused) == 1


def test_a_path_named_twice_is_granted_once(workspace: Path) -> None:
    plan = Plan(
        goal="g",
        steps=[
            PlanStep(description="one", files=["src/a.py"]),
            PlanStep(description="two", files=["src/a.py"]),
        ],
    )

    granted, _refused = scope_grants(plan, workspace=workspace)

    assert granted == ("src/a.py",)


# ------------------------------------------------------------------- the store


def test_approving_a_pending_plan_produces_a_grant_key(workspace: Path) -> None:
    store = PlanStore()
    store.propose(make_plan("src/a.py"), plan_id="p1")

    approved = store.approve("p1", workspace=workspace)

    assert approved.grant_key == "plan-edits:p1"
    assert approved.covers("src/a.py")
    assert not approved.covers("src/b.py")


def test_approving_an_unknown_plan_is_an_error(workspace: Path) -> None:
    with pytest.raises(PlanError, match="no pending plan"):
        PlanStore().approve("nope", workspace=workspace)


def test_approving_consumes_the_pending_plan(workspace: Path) -> None:
    """Otherwise `/execute` after a second `/plan` could approve the older one by id."""
    store = PlanStore()
    store.propose(make_plan("src/a.py"), plan_id="p1")
    store.approve("p1", workspace=workspace)

    with pytest.raises(PlanError):
        store.approve("p1", workspace=workspace)


def test_a_second_approved_plan_replaces_the_first(workspace: Path) -> None:
    """Two live plans would mean a file granted by a plan the user has moved on from.

    The grant set must always be exactly what the user last looked at and said yes to.
    """
    store = PlanStore()
    store.propose(make_plan("src/old.py"), plan_id="p1")
    store.approve("p1", workspace=workspace)
    store.propose(make_plan("src/new.py"), plan_id="p2")
    store.approve("p2", workspace=workspace)

    assert store.approved is not None
    assert store.approved.covers("src/new.py")
    assert not store.approved.covers("src/old.py"), "the old grant must not survive"


def test_clearing_revokes_the_grant(workspace: Path) -> None:
    store = PlanStore()
    store.propose(make_plan("src/a.py"), plan_id="p1")
    store.approve("p1", workspace=workspace)

    store.clear()

    assert store.approved is None


def test_todos_seed_one_in_progress_item(workspace: Path) -> None:
    """`todo_write` validates exactly one active item, so the seed has to match its
    contract — a list of all-pending todos is rejected at the tool boundary."""
    store = PlanStore()
    store.propose(
        Plan(goal="g", steps=[PlanStep(description="first"), PlanStep(description="second")]),
        plan_id="p1",
    )
    store.approve("p1", workspace=workspace)

    todos = store.todos()

    assert [todo["status"] for todo in todos] == ["in_progress", "pending"]
    assert todos[0]["content"] == "first"


def test_todos_without_an_approved_plan_are_empty() -> None:
    assert PlanStore().todos() == []


def test_an_unscoped_approved_plan_covers_nothing() -> None:
    """The default is the safe one: constructed directly, with no grant files, it
    auto-approves nothing rather than everything."""
    assert not ApprovedPlan(plan=make_plan("src/a.py"), plan_id="p1").covers("src/a.py")
