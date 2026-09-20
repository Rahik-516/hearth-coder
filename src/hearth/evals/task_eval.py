"""The agent task eval: can the loop actually finish a small, real change?

Ten tasks, each run in a **disposable copy** of a fixture repository. The first three ask
whether the loop closes at all — add a unit test, rename a symbol, fix a failing test — and
the rest cover the kinds of work I3's workflows exist for: a small feature, test-writing, a
two-file refactor, generated documentation, a second bug-fix shape, validation and a docs
edit. They are deliberately small. The question is not
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

import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

#: Where the fixture repositories live, relative to the repository root.
FIXTURES = Path("tests/fixtures/repos")

#: The rename task's symbols. Both must exist in the fixture — the first run of this eval
#: used invented names, and the model correctly reported that the symbol was not there,
#: which the scorer then counted as a failure. `required_symbols` exists so that can never
#: happen silently again.
RENAME_FROM = "compute_subtotal"
RENAME_TO = "calculate_subtotal"


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
    #: Identifiers the fixture must already contain for the task to be answerable at all.
    #: A task naming something that is not there is not a hard task, it is a broken one:
    #: the only correct response is the model saying so, which the scorer would mark as a
    #: failure. Checked by :func:`prepare_workspace` before the run.
    required_symbols: tuple[str, ...] = ()


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
    """A new test exercises `compute_tax`, and the suite still passes."""
    tests = list((root / "tests").rglob("test_*.py"))
    mentions = [path for path in tests if "compute_tax" in path.read_text(encoding="utf-8")]
    if not mentions:
        return False, "no test mentions compute_tax"
    return _check_suite_green(root)


def _check_rename(root: Path) -> tuple[bool, str]:
    """The old name is gone everywhere, the new one is used, and the suite passes."""
    stale = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if RENAME_FROM in path.read_text(encoding="utf-8")
    ]
    if stale:
        return False, f"old name still in {', '.join(sorted(stale)[:3])}"

    renamed = any(RENAME_TO in path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    if not renamed:
        return False, "new name appears nowhere"
    return _check_suite_green(root)


#: The test seeded by `fix-failing-test`, kept as a constant so the scorer can prove it
#: came back unchanged.
SEEDED_TEST_PATH = "tests/test_eval_rounding.py"
SEEDED_TEST = (
    "from decimal import Decimal\n\n"
    "from billing.invoice_service import round_half_up\n\n\n"
    "def test_rounds_half_up() -> None:\n"
    "    assert round_half_up(Decimal('1.005'), 2) == Decimal('1.01')\n"
)


def _break_rounding(root: Path) -> None:
    """Introduce the failing test the fix-a-failing-test task has to repair."""
    target = root / SEEDED_TEST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SEEDED_TEST, encoding="utf-8")


def _check_fixed_without_touching_the_test(root: Path) -> tuple[bool, str]:
    """Green suite **and** the seeded test still byte-identical.

    Without the second half the task is trivially gamed: deleting or rewriting the failing
    test also turns the suite green, and the scorer would call that a pass. The prompt says
    not to touch the test, but an eval that relies on the model obeying the prompt is
    measuring the prompt, not the model — the check has to enforce it.
    """
    target = root / SEEDED_TEST_PATH
    if not target.is_file():
        return False, "the failing test was deleted rather than fixed"
    if target.read_text(encoding="utf-8") != SEEDED_TEST:
        return False, "the failing test was edited instead of the source"
    return _check_suite_green(root)


# ------------------------------------------------- tasks 4-10 (I3): breadth, not depth
#
# The first three tasks ask whether the loop closes at all. These seven ask whether it
# closes across the *kinds* of work I3 added workflows for: a small feature, test-writing,
# a two-file refactor, generated documentation, a second bug-fix shape, input validation
# and a docs edit. Same rules as before — structural predicates, no model grading, and a
# check that an untouched workspace fails (proved in tests/unit/test_task_eval.py).


def _run_python(root: Path, code: str) -> tuple[bool, str]:
    """Run a snippet against the workspace's ``src`` and report whether it exited cleanly.

    A behavioural check rather than a text match: "the property exists" is satisfied by a
    property that returns the wrong thing, and a text match cannot tell.
    """
    import os

    # S603: the interpreter is `sys.executable` and the snippet is one of this module's own
    # constants, run against a disposable copy of a fixture — neither comes from a model.
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    if completed.returncode == 0:
        return True, ""
    lines = (completed.stderr or completed.stdout).strip().splitlines()
    return False, lines[-1] if lines else f"exit {completed.returncode}"


def _read(root: Path, relative: str) -> str:
    path = root / relative
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _new_test_files(root: Path) -> list[Path]:
    """Test files other than the one the fixture ships with."""
    return [
        path
        for path in (root / "tests").rglob("test_*.py")
        if path.name not in {"test_invoice_service.py", Path(SEEDED_TEST_PATH).name}
    ]


# -- 4. add-model-property -------------------------------------------------------------

_PROPERTY_PROBE = """
from datetime import date
from decimal import Decimal
from uuid import uuid4
from billing.models import Customer, Invoice

customer = Customer(id=uuid4(), name="a", email="a@example.test")
draft = Invoice(id=uuid4(), customer=customer, issued_on=date(2026, 1, 1))
assert draft.is_finalized is False, "a draft reported as finalized"
done = draft.with_totals(subtotal=Decimal("1"), tax=Decimal("0"), total=Decimal("1"))
assert done.is_finalized is True, "a finalized invoice reported as not finalized"
"""


def _check_model_property(root: Path) -> tuple[bool, str]:
    ok, detail = _run_python(root, _PROPERTY_PROBE)
    if not ok:
        return False, f"Invoice.is_finalized is wrong or missing ({detail})"
    if not any("is_finalized" in path.read_text(encoding="utf-8") for path in _new_test_files(root)) and (
        "is_finalized" not in _read(root, "tests/test_invoice_service.py")
    ):
        return False, "no test mentions is_finalized"
    return _check_suite_green(root)


# -- 5. write-error-tests --------------------------------------------------------------


def _check_error_tests(root: Path) -> tuple[bool, str]:
    """A new test file imports the error types and has at least three tests."""
    candidates = [
        path for path in _new_test_files(root) if "billing.errors" in path.read_text(encoding="utf-8")
    ]
    if not candidates:
        return False, "no new test file imports billing.errors"
    written = max(path.read_text(encoding="utf-8").count("def test_") for path in candidates)
    if written < 3:
        return False, f"only {written} test(s) written; expected at least 3"
    return _check_suite_green(root)


# -- 6. extract-quantize (two-file refactor) -------------------------------------------

_QUANTIZE_PROBE = """
from decimal import Decimal
from billing.money import quantize

assert quantize(Decimal("2.675")) == Decimal("2.68"), "half-up rounding changed"
assert quantize(Decimal("0.125")) == Decimal("0.13"), "half-up rounding changed"
assert quantize(Decimal("1")) == Decimal("1.00"), "cents no longer applied"
"""


def _check_extract_quantize(root: Path) -> tuple[bool, str]:
    if not (root / "src/billing/money.py").is_file():
        return False, "src/billing/money.py was not created"
    ok, detail = _run_python(root, _QUANTIZE_PROBE)
    if not ok:
        return False, f"billing.money.quantize is wrong or missing ({detail})"

    service = _read(root, "src/billing/invoice_service.py")
    if not re.search(r"from\s+(?:billing|\.)\.?money\s+import|import\s+billing\.money", service):
        return False, "invoice_service.py does not import from money"
    if "value.quantize(" in service:
        return False, "the rounding logic is still inlined in invoice_service.py"
    return _check_suite_green(root)


# -- 7. generate-architecture-doc (doc generation, structural) -------------------------

_MODULES = ("models", "errors", "invoice_service", "payments", "reporting")


def _check_architecture_doc(root: Path) -> tuple[bool, str]:
    """Well-formed, covers the real components, and names no file that does not exist.

    The third condition is the one that matters. A generated document's characteristic
    failure is describing something that is not there; the check is exactly "every file the
    document mentions is a file".
    """
    text = _read(root, "docs/ARCHITECTURE.md")
    if not text.strip():
        return False, "docs/ARCHITECTURE.md was not written"

    headings = re.findall(r"^#{1,3}\s+\S", text, flags=re.MULTILINE)
    if len(headings) < 3:
        return False, f"only {len(headings)} heading(s); expected at least 3"
    if text.count("```") % 2:
        return False, "a code fence is not closed"

    covered = [module for module in _MODULES if module in text]
    if len(covered) < 4:
        return False, f"mentions only {len(covered)} of the 5 components ({', '.join(covered) or 'none'})"

    existing = {path.name for path in root.rglob("*") if path.is_file()}
    invented = sorted({name for name in re.findall(r"[\w.-]+\.py\b", text) if name not in existing})
    if invented:
        return False, f"mentions files that do not exist: {', '.join(invented[:3])}"
    return True, "well-formed and grounded"


# -- 8. fix-injected-bug ---------------------------------------------------------------

SEEDED_SUBTOTAL_TEST_PATH = "tests/test_eval_subtotal.py"
SEEDED_SUBTOTAL_TEST = (
    "from decimal import Decimal\n\n"
    "from billing.invoice_service import InvoiceService\n"
    "from billing.models import LineItem\n\n\n"
    "def test_subtotal_is_quantity_times_price() -> None:\n"
    "    service = InvoiceService(repository=None, tax_policy=None)  # type: ignore[arg-type]\n"
    "    lines = [LineItem('a', 2, Decimal('1.50')), LineItem('b', 1, Decimal('4.00'))]\n"
    "    assert service.compute_subtotal(lines) == Decimal('7.00')\n"
)

_BUG_BEFORE = 'total = sum((line.amount for line in lines), Decimal("0"))'
_BUG_AFTER = 'total = sum((line.unit_price for line in lines), Decimal("0"))'


def _inject_subtotal_bug(root: Path) -> None:
    """Seed a test that describes correct behaviour, and break the code it exercises."""
    target = root / "src/billing/invoice_service.py"
    source = target.read_text(encoding="utf-8")
    if _BUG_BEFORE not in source:
        raise ValueError("the fixture no longer contains the line this task mutates")
    target.write_text(source.replace(_BUG_BEFORE, _BUG_AFTER), encoding="utf-8")

    seeded = root / SEEDED_SUBTOTAL_TEST_PATH
    seeded.parent.mkdir(parents=True, exist_ok=True)
    seeded.write_text(SEEDED_SUBTOTAL_TEST, encoding="utf-8")


def _check_subtotal_fixed(root: Path) -> tuple[bool, str]:
    seeded = root / SEEDED_SUBTOTAL_TEST_PATH
    if not seeded.is_file():
        return False, "the failing test was deleted rather than fixed"
    if seeded.read_text(encoding="utf-8") != SEEDED_SUBTOTAL_TEST:
        return False, "the failing test was edited instead of the source"
    if _BUG_AFTER in _read(root, "src/billing/invoice_service.py"):
        return False, "the injected bug is still in the source"
    return _check_suite_green(root)


# -- 9. add-validation -----------------------------------------------------------------

_VALIDATION_PROBE = """
from decimal import Decimal
from billing.models import LineItem

LineItem("ok", 1, Decimal("1.00"))
for bad in (0, -1):
    try:
        LineItem("bad", bad, Decimal("1.00"))
    except ValueError:
        continue
    raise AssertionError(f"quantity {bad} was accepted")
"""


def _check_validation(root: Path) -> tuple[bool, str]:
    ok, detail = _run_python(root, _VALIDATION_PROBE)
    if not ok:
        return False, f"LineItem validation is wrong or missing ({detail})"
    tests = "\n".join(path.read_text(encoding="utf-8") for path in (root / "tests").rglob("test_*.py"))
    if "ValueError" not in tests or "LineItem" not in tests:
        return False, "no test exercises the ValueError on LineItem"
    return _check_suite_green(root)


# -- 10. readme-usage-section ----------------------------------------------------------


def _check_readme_usage(root: Path) -> tuple[bool, str]:
    text = _read(root, "README.md")
    if "## Layout" not in text:
        return False, "the existing Layout section was removed"
    if not re.search(r"^#{1,3}\s+Usage\b", text, flags=re.MULTILINE):
        return False, "no Usage heading"

    blocks = re.findall(r"```(?:python|py)\n(.*?)```", text, flags=re.DOTALL)
    examples = [block for block in blocks if "InvoiceService" in block]
    if not examples:
        return False, "no python example that uses InvoiceService"

    import ast

    for block in examples:
        try:
            ast.parse(block)
        except SyntaxError as exc:
            return False, f"the example is not valid Python ({exc.msg})"
        if "billing" not in block:
            return False, "the example does not import from the billing package"
    return True, "usage section with a valid example"


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        name="add-unit-test",
        fixture="py_small",
        prompt=(
            "Add a unit test for InvoiceService.compute_tax in "
            "src/billing/invoice_service.py. Put it with the existing tests and run the "
            "suite to check it passes."
        ),
        check=_check_added_test,
        required_symbols=("compute_tax",),
    ),
    TaskSpec(
        name="rename-symbol",
        fixture="py_small",
        prompt=(
            f"Rename the method {RENAME_FROM} to {RENAME_TO} everywhere it appears, "
            "including its callers and tests. Run the tests afterwards."
        ),
        check=_check_rename,
        required_symbols=(RENAME_FROM,),
    ),
    TaskSpec(
        name="fix-failing-test",
        fixture="py_small",
        prompt=(
            "tests/test_eval_rounding.py is failing. Find out why and fix the source so "
            "it passes. Do not change the test."
        ),
        check=_check_fixed_without_touching_the_test,
        setup=_break_rounding,
    ),
    TaskSpec(
        name="add-model-property",
        fixture="py_small",
        prompt=(
            "Add a property `is_finalized` to the `Invoice` dataclass in "
            "src/billing/models.py that returns True when the invoice's status is "
            "FINALIZED. Add a test for it under tests/ and run the suite."
        ),
        check=_check_model_property,
        required_symbols=("with_totals", "is_editable"),
    ),
    TaskSpec(
        name="write-error-tests",
        fixture="py_small",
        prompt=(
            "Write tests for src/billing/errors.py. Put them in a new file under tests/ "
            "and run the suite to check they pass."
        ),
        check=_check_error_tests,
        required_symbols=("PaymentDeclined", "InvoiceNotFound"),
    ),
    TaskSpec(
        name="extract-quantize",
        fixture="py_small",
        prompt=(
            "Move the rounding logic in InvoiceService._quantize into a new module "
            "src/billing/money.py as a function `quantize(value)`, and make "
            "InvoiceService call it instead. Behaviour must not change. Run the tests."
        ),
        check=_check_extract_quantize,
        required_symbols=("_quantize",),
    ),
    TaskSpec(
        name="generate-architecture-doc",
        fixture="py_small",
        prompt=(
            "Write docs/ARCHITECTURE.md describing the components of src/billing and "
            "how they relate. Only mention files that actually exist."
        ),
        check=_check_architecture_doc,
    ),
    TaskSpec(
        name="fix-injected-bug",
        fixture="py_small",
        prompt=(
            "tests/test_eval_subtotal.py is failing. Find the cause in the source and "
            "fix it so the test passes. Do not change the test."
        ),
        check=_check_subtotal_fixed,
        setup=_inject_subtotal_bug,
    ),
    TaskSpec(
        name="add-validation",
        fixture="py_small",
        prompt=(
            "Make `LineItem` in src/billing/models.py raise ValueError when its quantity "
            "is not positive. Add a test for it under tests/ and run the suite."
        ),
        check=_check_validation,
        required_symbols=("LineItem",),
    ),
    TaskSpec(
        name="readme-usage-section",
        fixture="py_small",
        prompt=(
            "Add a Usage section to README.md with a short Python example that builds "
            "an InvoiceService. Keep the existing sections."
        ),
        check=_check_readme_usage,
        required_symbols=("InvoiceService",),
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

    missing = [
        symbol
        for symbol in spec.required_symbols
        if not any(symbol in path.read_text(encoding="utf-8") for path in source.rglob("*.py"))
    ]
    if missing:
        raise ValueError(
            f"task {spec.name!r} names {', '.join(missing)}, which {spec.fixture} does not "
            "contain — the task is unanswerable, and a model that says so would be scored "
            "as failing it"
        )

    shutil.copytree(source, destination, dirs_exist_ok=False)
    if spec.setup is not None:
        spec.setup(destination)
    return destination


def score(
    spec: TaskSpec, workspace: Path, *, steps: int, tool_calls: int, reason: str, started: float
) -> TaskOutcome:
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
