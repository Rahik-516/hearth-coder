"""The I3 eval tasks' scorers — proved against hand-written solutions and known cheats.

An eval's numbers are only as good as its scorers, and nobody re-checks a green eval. This
project has already been bitten twice: a scorer that marked a correct refusal as a failure,
and a harness that supplied the passing signal the model was supposed to earn. So each new
task carries three kinds of evidence here:

* **Doing nothing scores zero.** The workspace as prepared must fail its own check, and for
  a reason that names the missing work — not "the task is impossible".
* **A correct solution passes.** Written by hand, applied to a real disposable copy, and
  scored by the real predicate, including the real pytest run.
* **The obvious shortcuts fail.** Deleting the failing test, satisfying a text match with
  a property that returns the wrong thing, a document that names a file that is not there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hearth.evals.task_eval import (
    SEEDED_SUBTOTAL_TEST_PATH,
    TASKS,
    TaskSpec,
    prepare_workspace,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

NEW_TASKS = (
    "add-model-property",
    "write-error-tests",
    "extract-quantize",
    "generate-architecture-doc",
    "fix-injected-bug",
    "add-validation",
    "readme-usage-section",
)


def spec_named(name: str) -> TaskSpec:
    return next(task for task in TASKS if task.name == name)


def workspace_for(name: str, tmp_path: Path) -> tuple[TaskSpec, Path]:
    spec = spec_named(name)
    return spec, prepare_workspace(spec, repo_root=REPO_ROOT, destination=tmp_path / "copy")


def replace_in(root: Path, relative: str, old: str, new: str) -> None:
    path = root / relative
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{relative} does not contain {old!r}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ the whole set


def test_the_eval_has_ten_tasks_with_unique_names() -> None:
    names = [task.name for task in TASKS]

    assert len(names) == 10
    assert len(set(names)) == 10


@pytest.mark.parametrize("name", NEW_TASKS)
def test_doing_nothing_scores_zero(name: str, tmp_path: Path) -> None:
    """The workspace as prepared has to fail its own check, or passing proves nothing."""
    spec, workspace = workspace_for(name, tmp_path)

    passed, detail = spec.check(workspace)

    assert not passed, f"{name} passes on an untouched workspace"
    assert detail, "and it says why"
    assert "impossible" not in detail


# ------------------------------------------------------------ add-model-property


PROPERTY = (
    "\n    @property\n    def is_finalized(self) -> bool:\n"
    "        return self.status is InvoiceStatus.FINALIZED\n"
)
EDITABLE = (
    "    @property\n    def is_editable(self) -> bool:\n"
    "        return self.status is InvoiceStatus.DRAFT\n"
)

PROPERTY_TEST = (
    "from datetime import date\nfrom uuid import uuid4\n\n"
    "from billing.models import Customer, Invoice\n\n\n"
    "def test_a_draft_is_not_finalized() -> None:\n"
    "    invoice = Invoice(id=uuid4(), customer=Customer(id=uuid4(), name='a', email='e'),"
    " issued_on=date(2026, 1, 1))\n"
    "    assert invoice.is_finalized is False\n"
)


def test_a_correct_property_and_test_pass(tmp_path: Path) -> None:
    spec, workspace = workspace_for("add-model-property", tmp_path)
    replace_in(
        workspace,
        "src/billing/models.py",
        EDITABLE,
        EDITABLE + PROPERTY,
    )
    write(workspace, "tests/test_is_finalized.py", PROPERTY_TEST)

    assert spec.check(workspace) == (True, "suite green")


def test_a_property_that_always_says_yes_fails_the_behavioural_probe(tmp_path: Path) -> None:
    """A text match for "is_finalized" would accept this; the probe does not."""
    spec, workspace = workspace_for("add-model-property", tmp_path)
    replace_in(
        workspace,
        "src/billing/models.py",
        "    @property\n    def is_editable(self) -> bool:",
        "    @property\n    def is_finalized(self) -> bool:\n        return True\n\n"
        "    @property\n    def is_editable(self) -> bool:",
    )
    write(workspace, "tests/test_is_finalized.py", PROPERTY_TEST)

    passed, detail = spec.check(workspace)

    assert not passed
    assert "wrong or missing" in detail


def test_a_correct_property_with_no_test_fails(tmp_path: Path) -> None:
    spec, workspace = workspace_for("add-model-property", tmp_path)
    replace_in(
        workspace,
        "src/billing/models.py",
        "    @property\n    def is_editable(self) -> bool:",
        "    @property\n    def is_finalized(self) -> bool:\n"
        "        return self.status is InvoiceStatus.FINALIZED\n\n"
        "    @property\n    def is_editable(self) -> bool:",
    )

    passed, detail = spec.check(workspace)

    assert not passed
    assert "no test mentions" in detail


# --------------------------------------------------------------- write-error-tests

ERROR_TESTS = (
    "from billing.errors import BillingError, InvoiceNotFound, PaymentDeclined\n\n\n"
    "def test_errors_share_a_base() -> None:\n    assert issubclass(InvoiceNotFound, BillingError)\n\n\n"
    "def test_a_declined_payment_carries_its_reason() -> None:\n"
    "    assert PaymentDeclined('x', 'expired').reason == 'expired'\n\n\n"
    "def test_not_found_names_the_invoice() -> None:\n    assert 'x' in str(InvoiceNotFound('x'))\n"
)


def test_three_real_error_tests_pass(tmp_path: Path) -> None:
    spec, workspace = workspace_for("write-error-tests", tmp_path)
    write(workspace, "tests/test_errors.py", ERROR_TESTS)

    assert spec.check(workspace)[0]


def test_two_tests_are_not_enough(tmp_path: Path) -> None:
    spec, workspace = workspace_for("write-error-tests", tmp_path)
    write(workspace, "tests/test_errors.py", ERROR_TESTS.rsplit("def test_not_found", 1)[0])

    passed, detail = spec.check(workspace)

    assert not passed
    assert "only 2 test(s)" in detail


def test_tests_that_never_touch_the_errors_module_do_not_count(tmp_path: Path) -> None:
    spec, workspace = workspace_for("write-error-tests", tmp_path)
    write(
        workspace,
        "tests/test_other.py",
        "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n\n\ndef test_c():\n    pass\n",
    )

    passed, detail = spec.check(workspace)

    assert not passed
    assert "imports billing.errors" in detail


# ---------------------------------------------------------------- extract-quantize

MONEY = (
    "from decimal import ROUND_HALF_UP, Decimal\n\nCENTS = Decimal('0.01')\n\n\n"
    "def quantize(value: Decimal) -> Decimal:\n"
    "    return value.quantize(CENTS, rounding=ROUND_HALF_UP)\n"
)


def extract_quantize(workspace: Path) -> None:
    write(workspace, "src/billing/money.py", MONEY)
    replace_in(
        workspace,
        "src/billing/invoice_service.py",
        "        return value.quantize(CENTS, rounding=ROUND_HALF_UP)",
        "        return quantize(value)",
    )
    replace_in(
        workspace,
        "src/billing/invoice_service.py",
        "from decimal import ROUND_HALF_UP, Decimal",
        "from decimal import Decimal\n\nfrom billing.money import quantize",
    )


def test_a_correct_extraction_passes(tmp_path: Path) -> None:
    spec, workspace = workspace_for("extract-quantize", tmp_path)
    extract_quantize(workspace)

    assert spec.check(workspace)[0]


def test_a_new_module_that_nothing_uses_fails(tmp_path: Path) -> None:
    """Copying the function out and leaving the original in place is not a refactor."""
    spec, workspace = workspace_for("extract-quantize", tmp_path)
    write(workspace, "src/billing/money.py", MONEY)

    passed, detail = spec.check(workspace)

    assert not passed
    assert "does not import from money" in detail


def test_an_extraction_that_changes_the_rounding_fails(tmp_path: Path) -> None:
    """ "Behaviour must not change" is enforced, not requested."""
    spec, workspace = workspace_for("extract-quantize", tmp_path)
    extract_quantize(workspace)
    replace_in(workspace, "src/billing/money.py", "ROUND_HALF_UP)", "ROUND_DOWN)")
    replace_in(workspace, "src/billing/money.py", "import ROUND_HALF_UP,", "import ROUND_DOWN,")

    passed, detail = spec.check(workspace)

    assert not passed
    assert "wrong or missing" in detail


# ------------------------------------------------------- generate-architecture-doc

GOOD_DOC = (
    "# Architecture\n\n## Components\n\n"
    "- `models.py` holds the dataclasses.\n- `errors.py` is the exception hierarchy.\n"
    "- `invoice_service.py` orchestrates finalization.\n- `payments.py` charges cards.\n\n"
    "## Relations\n\nThe service uses the models and raises the errors.\n"
)


def test_a_grounded_document_passes(tmp_path: Path) -> None:
    spec, workspace = workspace_for("generate-architecture-doc", tmp_path)
    write(workspace, "docs/ARCHITECTURE.md", GOOD_DOC)

    assert spec.check(workspace) == (True, "well-formed and grounded")


def test_a_document_naming_an_invented_file_fails(tmp_path: Path) -> None:
    """The characteristic failure of generated documentation, made a check."""
    spec, workspace = workspace_for("generate-architecture-doc", tmp_path)
    write(workspace, "docs/ARCHITECTURE.md", GOOD_DOC + "\nSee also `services.py`.\n")

    passed, detail = spec.check(workspace)

    assert not passed
    assert "services.py" in detail


def test_a_document_covering_too_few_components_fails(tmp_path: Path) -> None:
    spec, workspace = workspace_for("generate-architecture-doc", tmp_path)
    write(workspace, "docs/ARCHITECTURE.md", "# A\n\n## B\n\n## C\n\nOnly `models.py` here.\n")

    passed, detail = spec.check(workspace)

    assert not passed
    assert "of the 5 components" in detail


def test_an_unclosed_fence_fails(tmp_path: Path) -> None:
    spec, workspace = workspace_for("generate-architecture-doc", tmp_path)
    write(workspace, "docs/ARCHITECTURE.md", GOOD_DOC + "\n```python\nx = 1\n")

    passed, detail = spec.check(workspace)

    assert not passed
    assert "fence" in detail


# -------------------------------------------------------------- fix-injected-bug


def test_the_injected_bug_makes_the_seeded_test_fail_at_the_start(tmp_path: Path) -> None:
    """Both halves of the setup matter: the test exists, and the code is actually wrong."""
    _spec, workspace = workspace_for("fix-injected-bug", tmp_path)

    assert (workspace / SEEDED_SUBTOTAL_TEST_PATH).is_file()
    source = (workspace / "src/billing/invoice_service.py").read_text(encoding="utf-8")
    assert "line.unit_price for line in lines" in source


def test_reverting_the_bug_passes(tmp_path: Path) -> None:
    spec, workspace = workspace_for("fix-injected-bug", tmp_path)
    replace_in(
        workspace, "src/billing/invoice_service.py", "line.unit_price for line", "line.amount for line"
    )

    assert spec.check(workspace)[0]


def test_deleting_the_seeded_test_is_not_a_fix(tmp_path: Path) -> None:
    spec, workspace = workspace_for("fix-injected-bug", tmp_path)
    (workspace / SEEDED_SUBTOTAL_TEST_PATH).unlink()

    passed, detail = spec.check(workspace)

    assert not passed
    assert "deleted" in detail


def test_rewriting_the_seeded_test_is_not_a_fix(tmp_path: Path) -> None:
    spec, workspace = workspace_for("fix-injected-bug", tmp_path)
    write(
        workspace,
        SEEDED_SUBTOTAL_TEST_PATH,
        "def test_subtotal_is_quantity_times_price():\n    assert True\n",
    )

    passed, detail = spec.check(workspace)

    assert not passed
    assert "edited" in detail


# ---------------------------------------------------------------- add-validation

VALIDATION_TEST = (
    "from decimal import Decimal\n\nimport pytest\n\nfrom billing.models import LineItem\n\n\n"
    "def test_zero_quantity_is_rejected() -> None:\n"
    "    with pytest.raises(ValueError):\n        LineItem('x', 0, Decimal('1'))\n"
)


def add_validation(workspace: Path) -> None:
    replace_in(
        workspace,
        "src/billing/models.py",
        "    unit_price: Decimal\n\n    @property\n    def amount",
        "    unit_price: Decimal\n\n    def __post_init__(self) -> None:\n"
        "        if self.quantity <= 0:\n            raise ValueError('quantity must be positive')\n\n"
        "    @property\n    def amount",
    )


def test_correct_validation_with_a_test_passes(tmp_path: Path) -> None:
    spec, workspace = workspace_for("add-validation", tmp_path)
    add_validation(workspace)
    write(workspace, "tests/test_line_item.py", VALIDATION_TEST)

    assert spec.check(workspace)[0]


def test_validation_without_a_test_fails(tmp_path: Path) -> None:
    spec, workspace = workspace_for("add-validation", tmp_path)
    add_validation(workspace)

    passed, detail = spec.check(workspace)

    assert not passed
    assert "no test exercises" in detail


def test_a_test_without_the_validation_fails(tmp_path: Path) -> None:
    spec, workspace = workspace_for("add-validation", tmp_path)
    write(workspace, "tests/test_line_item.py", VALIDATION_TEST)

    passed, detail = spec.check(workspace)

    assert not passed
    assert "wrong or missing" in detail


# ---------------------------------------------------------- readme-usage-section

USAGE = (
    "\n## Usage\n\n```python\nfrom billing.invoice_service import InvoiceService\n\n"
    "service = InvoiceService(repository=None, tax_policy=None)\n```\n"
)


def test_a_usage_section_with_a_valid_example_passes(tmp_path: Path) -> None:
    spec, workspace = workspace_for("readme-usage-section", tmp_path)
    with (workspace / "README.md").open("a", encoding="utf-8") as stream:
        stream.write(USAGE)

    assert spec.check(workspace)[0]


def test_replacing_the_readme_fails(tmp_path: Path) -> None:
    """ "Keep the existing sections" is enforced: overwriting the file is the cheap route."""
    spec, workspace = workspace_for("readme-usage-section", tmp_path)
    write(workspace, "README.md", USAGE)

    passed, detail = spec.check(workspace)

    assert not passed
    assert "Layout" in detail


def test_an_example_that_is_not_python_fails(tmp_path: Path) -> None:
    spec, workspace = workspace_for("readme-usage-section", tmp_path)
    with (workspace / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Usage\n\n```python\nfrom billing import InvoiceService(\n```\n")

    passed, detail = spec.check(workspace)

    assert not passed
    assert "not valid Python" in detail


def test_a_usage_section_without_an_example_fails(tmp_path: Path) -> None:
    spec, workspace = workspace_for("readme-usage-section", tmp_path)
    with (workspace / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Usage\n\nCall the service.\n")

    passed, detail = spec.check(workspace)

    assert not passed
    assert "no python example" in detail
