"""The task eval's scaffolding: disposable copies and structural scoring.

Tests the eval harness, not the model. The scoring predicates are the part that has to be
trustworthy — a scorer that says "passed" when the suite is red would make every later
benchmark number meaningless, and nobody re-checks a green eval.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hearth.evals.task_eval import TASKS, EvalReport, TaskOutcome, TaskSpec, prepare_workspace

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_every_task_names_a_fixture_that_exists() -> None:
    for spec in TASKS:
        assert (REPO_ROOT / "tests" / "fixtures" / "repos" / spec.fixture).is_dir(), spec.name


def test_preparing_a_workspace_copies_rather_than_using_the_fixture(tmp_path: Path) -> None:
    """CLAUDE.md rule 8: the agent edits a copy, never the fixture itself."""
    spec = TASKS[0]
    source = REPO_ROOT / "tests" / "fixtures" / "repos" / spec.fixture
    before = sorted(path.name for path in source.rglob("*.py"))

    workspace = prepare_workspace(spec, repo_root=REPO_ROOT, destination=tmp_path / "copy")
    (workspace / "scribble.py").write_text("x = 1\n", encoding="utf-8")

    assert workspace != source
    assert sorted(path.name for path in source.rglob("*.py")) == before


def test_the_failing_test_task_starts_red(tmp_path: Path) -> None:
    """Its setup has to actually break something, or the task scores free points."""
    spec = next(task for task in TASKS if task.name == "fix-failing-test")

    workspace = prepare_workspace(spec, repo_root=REPO_ROOT, destination=tmp_path / "copy")

    assert (workspace / "tests" / "test_eval_rounding.py").exists()
    passed, _ = spec.check(workspace)
    assert not passed, "the task must begin failing, or passing it proves nothing"


def test_the_rename_task_starts_unsatisfied(tmp_path: Path) -> None:
    spec = next(task for task in TASKS if task.name == "rename-symbol")

    workspace = prepare_workspace(spec, repo_root=REPO_ROOT, destination=tmp_path / "copy")

    passed, detail = spec.check(workspace)
    assert not passed
    # "old name still in ..." is the signal that the symbol is genuinely present and
    # genuinely un-renamed. The earlier form of this assertion accepted "appears nowhere"
    # too, which is what an *impossible* task looks like — so it passed while the task
    # named a function the fixture did not have.
    assert "old name still in" in detail


def test_deleting_the_failing_test_does_not_count_as_fixing_it(tmp_path: Path) -> None:
    """The obvious way to game this task: a green suite with the test removed."""
    spec = next(task for task in TASKS if task.name == "fix-failing-test")
    workspace = prepare_workspace(spec, repo_root=REPO_ROOT, destination=tmp_path / "copy")

    (workspace / "tests" / "test_eval_rounding.py").unlink()

    passed, detail = spec.check(workspace)
    assert not passed
    assert "deleted" in detail


def test_editing_the_failing_test_does_not_count_either(tmp_path: Path) -> None:
    spec = next(task for task in TASKS if task.name == "fix-failing-test")
    workspace = prepare_workspace(spec, repo_root=REPO_ROOT, destination=tmp_path / "copy")

    (workspace / "tests" / "test_eval_rounding.py").write_text(
        "def test_rounds_half_up() -> None:\n    assert True\n", encoding="utf-8"
    )

    passed, detail = spec.check(workspace)
    assert not passed
    assert "edited" in detail


def test_a_fixture_that_is_missing_is_an_error(tmp_path: Path) -> None:
    spec = TaskSpec(name="nope", fixture="does_not_exist", prompt="", check=lambda _: (True, ""))

    with pytest.raises(FileNotFoundError):
        prepare_workspace(spec, repo_root=REPO_ROOT, destination=tmp_path / "copy")


def test_the_report_counts_and_renders() -> None:
    report = EvalReport(
        outcomes=[
            TaskOutcome(name="a", passed=True, steps=4, tool_calls=6, duration_s=12.5),
            TaskOutcome(name="b", passed=False, detail="suite still failing", steps=8),
        ]
    )

    rendered = report.render()

    assert report.passed == 1
    assert report.total == 2
    assert "1/2 tasks passed" in rendered
    assert "PASS  a" in rendered
    assert "FAIL  b" in rendered


def test_every_task_names_symbols_the_fixture_actually_has(tmp_path: Path) -> None:
    """The bug that made the first live eval meaningless.

    Two of the three tasks named functions (`compute_total`, `apply_discount`) that
    py_small does not contain. The model searched, correctly reported the symbol was
    missing, and the scorer recorded a failure — so the run measured the tasks, not the
    model. prepare_workspace now refuses an unanswerable task outright.
    """
    for index, spec in enumerate(TASKS):
        prepare_workspace(spec, repo_root=REPO_ROOT, destination=tmp_path / f"copy{index}")


def test_a_task_naming_a_missing_symbol_is_refused(tmp_path: Path) -> None:
    spec = TaskSpec(
        name="impossible",
        fixture="py_small",
        prompt="rename nonexistent_function",
        check=lambda _: (True, ""),
        required_symbols=("nonexistent_function",),
    )

    with pytest.raises(ValueError, match="unanswerable"):
        prepare_workspace(spec, repo_root=REPO_ROOT, destination=tmp_path / "copy")
