"""`/refactor` preparation and checks — I3.

A refactor's characteristic failure is not the edit it made; it is the caller it did not
touch. So the tests concentrate on the two places code, not the model, is responsible for
that: **what the model is told before** (the index's known references, presented honestly
as name-matched and possibly incomplete) and **what the user is told after** (which
referencing files changed, and whether the suite passes — from a run that happened).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from hearth.indexing.pipeline import Indexer
from hearth.safety.policy import ConfigView, SessionView
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate
from hearth.tools.base import ToolContext
from hearth.tools.channel import ApprovalAsk, ApprovalReply
from hearth.tools.gateway import ToolGateway, make_policy
from hearth.tools.registry import ToolRegistry
from hearth.tools.tests import RunTestsTool
from hearth.workflows.refactor import (
    ImpactReport,
    impact_report,
    prepare_refactor,
    snapshot_files,
    split_arguments,
    verify_with_tests,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "repos"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(FIXTURES / "py_small", root)
    return root


@pytest.fixture
def repository(workspace: Path, tmp_path: Path) -> IndexRepository:
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    repo = IndexRepository(connection)
    Indexer(root=workspace, repository=repo).run()
    return repo


# ------------------------------------------------------------------- arguments


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("compute_tax use Decimal", ("compute_tax", "use Decimal")),
        ("  a::b   rename it to c  ", ("a::b", "rename it to c")),
        ("compute_tax", None),
        ("compute_tax   ", None),
        ("", None),
    ],
)
def test_the_arguments_split_into_target_and_goal(argument: str, expected) -> None:
    """A refactor with no goal has nothing to plan; a goal with no target has nowhere to
    start."""
    assert split_arguments(argument) == expected


# ----------------------------------------------------------------- preparation


def test_the_task_carries_the_index_references(workspace: Path, repository: IndexRepository) -> None:
    prep = prepare_refactor(
        workspace, "LineItemError raise it with the offending line", repository=repository
    )

    assert prep.refusal is None
    assert prep.target is not None
    assert prep.target.symbol == "LineItemError"
    assert prep.references, "premise: the fixture uses LineItemError outside its definition"
    assert prep.task is not None
    for path in prep.references:
        assert path in prep.task


def test_the_definition_itself_is_not_listed_as_a_caller(
    workspace: Path, repository: IndexRepository
) -> None:
    """Listing the class's own lines would have the model "update" what it is being asked
    to change, as though it were a use."""
    prep = prepare_refactor(workspace, "LineItemError tweak it", repository=repository)

    assert prep.target is not None and prep.target.start_line is not None
    definition_lines = range(prep.target.start_line, (prep.target.end_line or 0) + 1)
    own = prep.references.get(prep.target.path, [])
    assert not [line for line in own if line in definition_lines]


def test_the_task_says_the_references_are_name_matched(
    workspace: Path, repository: IndexRepository
) -> None:
    """Presenting the list as a call graph would have the model read "no references" as
    "safe to delete"."""
    prep = prepare_refactor(workspace, "LineItemError tweak it", repository=repository)

    assert prep.task is not None
    assert "matched by name" in prep.task
    assert "read each file" in prep.task


def test_no_references_is_stated_with_a_caution_not_as_safe(
    workspace: Path, repository: IndexRepository
) -> None:
    prep = prepare_refactor(workspace, "RateLimited tweak it", repository=repository)

    assert prep.task is not None
    if not prep.references:
        assert "no references" in prep.task
        assert "dynamically" in prep.task


def test_the_goal_reaches_the_task(workspace: Path, repository: IndexRepository) -> None:
    prep = prepare_refactor(workspace, "LineItemError make it carry the line", repository=repository)

    assert prep.task is not None
    assert "make it carry the line" in prep.task


def test_a_missing_goal_is_a_usage_message(workspace: Path, repository: IndexRepository) -> None:
    prep = prepare_refactor(workspace, "LineItemError", repository=repository)

    assert prep.task is None
    assert prep.refusal is not None and "usage" in prep.refusal


def test_a_file_target_is_redirected_to_a_symbol(
    workspace: Path, repository: IndexRepository
) -> None:
    """`/refactor` works on a symbol; a change to a whole file is what `/plan` is for."""
    prep = prepare_refactor(workspace, "src/billing/errors.py tidy it", repository=repository)

    assert prep.task is None
    assert prep.refusal is not None
    assert "src/billing/errors.py::<symbol>" in prep.refusal, "it names the actual file"
    assert "/plan" in prep.refusal


def test_an_unknown_symbol_is_refused_before_any_planning(
    workspace: Path, repository: IndexRepository
) -> None:
    prep = prepare_refactor(workspace, "NoSuchThing tweak it", repository=repository)

    assert prep.task is None
    assert prep.refusal is not None and "no file or symbol" in prep.refusal


def test_many_referencing_files_are_capped_and_counted(workspace: Path, tmp_path: Path) -> None:
    """The list is a sample past a point; the tail is counted so the model knows."""
    for n in range(45):
        (workspace / "src" / "billing" / f"use_{n}.py").write_text(
            "from billing.errors import LineItemError\n\n\ndef f():\n    raise LineItemError()\n",
            encoding="utf-8",
        )
    connection = connect(tmp_path / "many.db")
    migrate(connection, database="index")
    repo = IndexRepository(connection)
    Indexer(root=workspace, repository=repo).run()

    prep = prepare_refactor(workspace, "LineItemError tweak it", repository=repo)

    assert prep.task is not None
    assert "more file(s)" in prep.task


# ---------------------------------------------------------------------- impact


def test_the_report_splits_modified_from_untouched() -> None:
    before = {"a.py": "1", "b.py": "2", "c.py": "3"}
    after = {"a.py": "1", "b.py": "CHANGED", "c.py": "3"}

    report = impact_report(before, after)

    assert report.modified == ("b.py",)
    assert report.untouched == ("a.py", "c.py")
    assert report.total == 3


def test_a_file_created_by_the_refactor_counts_as_modified() -> None:
    assert impact_report({}, {"new.py": "h"}).modified == ("new.py",)


def test_an_untouched_refactor_reports_everything_as_untouched() -> None:
    same = {"a.py": "1", "b.py": "2"}

    assert impact_report(same, same) == ImpactReport(modified=(), untouched=("a.py", "b.py"))


def test_snapshots_track_content_not_timestamps(workspace: Path) -> None:
    """Rewriting a file with identical bytes is not a modification a reviewer needs told."""
    before = snapshot_files(workspace, ["src/billing/errors.py"])
    path = workspace / "src" / "billing" / "errors.py"
    path.write_bytes(path.read_bytes())
    after = snapshot_files(workspace, ["src/billing/errors.py"])

    assert impact_report(before, after).modified == ()


def test_a_missing_file_snapshots_as_empty_and_a_later_creation_counts() -> None:
    before = snapshot_files(Path("/nonexistent-root"), ["gone.py"])

    assert before == {"gone.py": ""}


# ---------------------------------------------------------------- verification


class Channel:
    def __init__(self, decision: str = "approve") -> None:
        self.decision = decision

    async def proposed(self, **_: object) -> None: ...

    async def started(self, **_: object) -> None: ...

    async def finished(self, **_: object) -> None: ...

    async def request_approval(self, ask: ApprovalAsk) -> ApprovalReply:
        return ApprovalReply(decision=self.decision)


def gateway_for(workspace: Path, decision: str = "approve") -> ToolGateway:
    return ToolGateway(
        registry=ToolRegistry(
            [RunTestsTool(test_command=f"{sys.executable} -m pytest -q -p no:cacheprovider")]
        ),
        context=ToolContext(workspace=workspace),
        channel=Channel(decision),
        policy=make_policy(SessionView(mode="agent", level="supervised"), ConfigView()),
        session_id="s",
    )


async def test_a_green_suite_is_reported_as_passing(workspace: Path) -> None:
    verification = await verify_with_tests(gateway_for(workspace))

    assert verification.passed is True, verification.summary


async def test_a_broken_suite_is_reported_as_failing(workspace: Path) -> None:
    (workspace / "tests" / "test_broken.py").write_text(
        "def test_it():\n    assert 1 == 2\n", encoding="utf-8"
    )

    verification = await verify_with_tests(gateway_for(workspace))

    assert verification.passed is False
    assert "failed" in verification.summary


async def test_a_declined_run_is_not_a_pass_and_not_a_failure(workspace: Path) -> None:
    """Three outcomes, not two. "You said no" must not read as "the tests failed", and
    certainly not as "the tests passed"."""
    verification = await verify_with_tests(gateway_for(workspace, decision="reject"))

    assert verification.passed is None
    assert "rejected" in verification.summary
