"""The agent task eval: can the loop actually finish a small, real change?

Three tasks, each run in a **disposable copy** of a fixture repository — add a unit test,
rename a symbol, fix a failing test. They are deliberately small. The question is not
whether a 4B model can do interesting work; it is whether the edit → test → fix loop closes
at all on the reference machine, which is the MVP's defining claim.

**Scored structurally, never by a model.** Each task states a predicate over the resulting
working tree: the renamed symbol appears nowhere, the new test exists and passes, the suite
goes from red to green. A model grading another model's diff would make the eval's own
numbers depend on the thing being measured, and a 4B judge is not a reliable grader of a 4B
worker. Structural checks are boring, cheap, and mean the score is reproducible.

A task that ends with the suite failing scores zero regardless of how good the diff looks.
That is the point: an unverified change is not a completed task.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

#: Where the fixture repositories live, relative to the repository root.
FIXTURES = Path("tests/fixtures/repos")


@dataclass(frozen=True)
class TaskSpec:
    """One eval task: what to ask for, and how to tell whether it worked."""

    name: str
    fixture: str
    prompt: str
    #: Structural predicate over the finished working tree. Returns (passed, detail).
    check: Callable[[Path], tuple[bool, str]]
    #: Applied to the disposable copy before the run, for tasks that need a broken state.
    setup: Callable[[Path], None] | None = None


@dataclass
class TaskOutcome:
    """What one task run produced."""

    name: str
    passed: bool
    detail: str = ""
    steps: int = 0
    tool_calls: int = 0
    duration_s: float = 0.0
    reason: str = ""


@dataclass
class EvalReport:
    outcomes: list[TaskOutcome] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.passed)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    def render(self) -> str:
        lines = [f"{self.passed}/{self.total} tasks passed", ""]
        for outcome in self.outcomes:
            mark = "PASS" if outcome.passed else "FAIL"
            lines.append(
                f"  {mark}  {outcome.name}  "
                f"steps={outcome.steps} tools={outcome.tool_calls} "
                f"{outcome.duration_s:.1f}s  {outcome.detail}".rstrip()
            )
        return "\n".join(lines)


# ----------------------------------------------------------------- the tasks


def _run_pytest(root: Path) -> tuple[int, str]:
    """Run the fixture's own tests. Returns (exit code, output)."""
    # sys.executable, not "python": the eval must run the fixture's suite under the same
    # interpreter as Hearth, and a partial name would resolve against PATH — which is the
    # kind of ambiguity S607 exists to flag.
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return completed.returncode, completed.stdout + completed.stderr


def _check_suite_green(root: Path) -> tuple[bool, str]:
    code, output = _run_pytest(root)
    if code == 0:
        return True, "suite green"
    return False, f"suite still failing ({output.strip().splitlines()[-1] if output.strip() else code})"


def _check_added_test(root: Path) -> tuple[bool, str]:
    """A new test exists for `apply_discount`, and the suite still passes."""
    tests = list((root / "tests").rglob("test_*.py"))
    mentions = [path for path in tests if "apply_discount" in path.read_text(encoding="utf-8")]
    if not mentions:
        return False, "no test mentions apply_discount"
    return _check_suite_green(root)


def _check_rename(root: Path) -> tuple[bool, str]:
    """The old name is gone everywhere, the new one is used, and the suite passes."""
    stale = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "compute_total" in path.read_text(encoding="utf-8")
    ]
    if stale:
        return False, f"old name still in {', '.join(sorted(stale)[:3])}"

    renamed = any("calculate_total" in path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    if not renamed:
        return False, "new name appears nowhere"
    return _check_suite_green(root)


def _break_rounding(root: Path) -> None:
    """Introduce the failing test the fix-a-failing-test task has to repair."""
    target = root / "tests" / "test_eval_rounding.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "from decimal import Decimal\n\n"
        "from billing.invoice_service import round_half_up\n\n\n"
        "def test_rounds_half_up() -> None:\n"
        "    assert round_half_up(Decimal('1.005'), 2) == Decimal('1.01')\n",
        encoding="utf-8",
    )


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        name="add-unit-test",
        fixture="py_small",
        prompt=(
            "Add a unit test for apply_discount in src/billing/invoice_service.py. "
            "Put it with the existing tests and run the suite to check it passes."
        ),
        check=_check_added_test,
    ),
    TaskSpec(
        name="rename-symbol",
        fixture="py_small",
        prompt=(
            "Rename the function compute_total to calculate_total everywhere it appears, "
            "including its callers and tests. Run the tests afterwards."
        ),
        check=_check_rename,
    ),
    TaskSpec(
        name="fix-failing-test",
        fixture="py_small",
        prompt=(
            "tests/test_eval_rounding.py is failing. Find out why and fix the source so "
            "it passes. Do not change the test."
        ),
        check=_check_suite_green,
        setup=_break_rounding,
    ),
)


def prepare_workspace(spec: TaskSpec, *, repo_root: Path, destination: Path) -> Path:
    """Copy the fixture to a disposable location and apply the task's setup.

    A copy, never the fixture itself: the agent is about to edit whatever it is pointed
    at, and CLAUDE.md rule 8 exists because a test that mutates its own fixtures is one
    that passes once (docs/safety-and-tool-use.md, and the incident in CLAUDE.md).
    """
    source = repo_root / FIXTURES / spec.fixture
    if not source.is_dir():
        raise FileNotFoundError(f"fixture repo not found: {source}")

    shutil.copytree(source, destination, dirs_exist_ok=False)
    if spec.setup is not None:
        spec.setup(destination)
    return destination


def score(spec: TaskSpec, workspace: Path, *, steps: int, tool_calls: int, reason: str,
          started: float) -> TaskOutcome:
    """Apply the task's structural predicate to the finished tree."""
    try:
        passed, detail = spec.check(workspace)
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
        passed, detail = False, f"check could not run: {exc}"

    return TaskOutcome(
        name=spec.name,
        passed=passed,
        detail=detail,
        steps=steps,
        tool_calls=tool_calls,
        duration_s=time.monotonic() - started,
        reason=reason,
    )
