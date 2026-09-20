"""Test-runner output parsers — the fix-loop's eyes.

These had no direct tests, and it showed. The pytest parser only read lines containing
``=``, so it never once parsed ``pytest -q`` — which is this project's own fallback test
command. A passing run and a failing run both came back as "could not parse the test
output", and the model was handed raw output instead of a one-line result. Nothing failed
loudly: `run_tests` degraded exactly as designed, which is why the gap stayed invisible
until a workflow needed `tests_ok` to be true for a run that passed.

The samples below are real runner output in each shape, banner and quiet.
"""

from __future__ import annotations

import pytest

from hearth.tools.test_parsers import detect_framework, parse_test_output

BANNER_FAILURE = """\
============================= test session starts ==============================
collected 3 items

tests/test_invoice.py ..F                                                [100%]

=================================== FAILURES ===================================
____________________ test_finalize_rounds_half_up ____________________
E       AssertionError: assert Decimal('1.24') == Decimal('1.25')
tests/test_invoice.py:31: AssertionError
=========================== short test summary info ============================
FAILED tests/test_invoice.py::test_finalize_rounds_half_up - AssertionError
========================= 1 failed, 2 passed in 0.41s ==========================
"""

QUIET_GREEN = """\
..                                                                       [100%]
2 passed in 0.11s
"""

QUIET_FAILURE = """\
.F                                                                       [100%]
=================================== FAILURES ===================================
__________________________ test_the_message_names_it ___________________________
E       AssertionError: assert 'invoice inv-1 not found' == 'wrong message'
tests/test_errors.py:5: AssertionError
=========================== short test summary info ============================
FAILED tests/test_errors.py::test_the_message_names_it - AssertionError: assert ...
1 failed, 1 passed in 0.12s
"""

NO_TESTS = """\


no tests ran in 0.03s
"""


# ---------------------------------------------------------------------- pytest


def test_banner_output_is_parsed() -> None:
    summary = parse_test_output(BANNER_FAILURE)

    assert (summary.passed, summary.failed) == (2, 1)
    assert summary.parsed
    assert not summary.ok
    assert summary.duration_s == pytest.approx(0.41)


def test_a_quiet_pass_is_parsed() -> None:
    """The regression: `pytest -q` prints no banner, and this used to be unparsed."""
    summary = parse_test_output(QUIET_GREEN)

    assert summary.parsed
    assert summary.passed == 2
    assert summary.ok, "a passing quiet run has to read as passing"
    assert summary.duration_s == pytest.approx(0.11)


def test_a_quiet_failure_is_parsed_with_the_failing_test_named() -> None:
    summary = parse_test_output(QUIET_FAILURE)

    assert (summary.passed, summary.failed) == (1, 1)
    assert not summary.ok
    assert [failure.name for failure in summary.failures] == ["test_the_message_names_it"]
    assert summary.failures[0].location == "tests/test_errors.py"


def test_the_one_line_description_names_the_failure() -> None:
    described = parse_test_output(QUIET_FAILURE).describe()

    assert "1 failed" in described
    assert "test_the_message_names_it" in described


def test_no_tests_ran_is_a_readable_result_and_not_a_pass() -> None:
    """Different from output the parser cannot read — and different from success.

    Zero tests "pass" vacuously, which is exactly what a file of never-collected tests
    looks like. It is parsed (so the model is told *why* nothing happened) but not ok.
    """
    summary = parse_test_output(NO_TESTS)

    assert summary.parsed
    assert summary.total == 0
    assert not summary.ok
    assert summary.describe().startswith("no tests ran")


def test_an_all_skipped_run_counts_as_having_run() -> None:
    """Collected and skipped is a result the suite reported, not an empty run."""
    summary = parse_test_output("ss\n2 skipped in 0.01s\n")

    assert summary.parsed
    assert summary.total == 2
    assert summary.ok


def test_numbers_in_a_traceback_are_not_counts() -> None:
    """Only summary lines carry counts. An assertion message that happens to contain
    "3 passed" must not be read as the result."""
    output = (
        "E       AssertionError: expected 3 passed but got 0\n"
        "1 failed in 0.05s\n"
    )

    summary = parse_test_output(output, framework="pytest")

    assert (summary.passed, summary.failed) == (0, 1)


def test_an_error_summary_is_not_ok() -> None:
    summary = parse_test_output("==== 1 error in 0.10s ====\n")

    assert summary.errors == 1
    assert not summary.ok


def test_unreadable_output_is_never_ok() -> None:
    """The mistake that would mislead the model into stopping: "I could not read this"
    reported as "nothing failed"."""
    summary = parse_test_output("Segmentation fault (core dumped)\n", framework="pytest")

    assert not summary.parsed
    assert not summary.ok
    assert summary.describe() == "could not parse the test output"


def test_empty_output_is_unparsed() -> None:
    assert not parse_test_output("   \n").parsed


# ------------------------------------------------------------------ detection


@pytest.mark.parametrize(
    ("text", "framework"),
    [
        (QUIET_GREEN, "pytest"),
        (BANNER_FAILURE, "pytest"),
        (NO_TESTS, "pytest"),
        ("Tests:       1 failed, 3 passed, 4 total\n", "jest"),
        (" Test Files  1 passed (1)\n", "vitest"),
        ("some unrelated output\n", None),
    ],
)
def test_the_framework_is_recognised(text: str, framework: str | None) -> None:
    assert detect_framework(text) == framework


# ---------------------------------------------------------------------- jest


def test_a_jest_summary_is_parsed() -> None:
    # Jest separates the describe block and the test name with U+203A, so the sample has to.
    name = "InvoiceService › rounds half up"  # noqa: RUF001
    output = (
        f"  ● {name}\n\n"
        "Tests:       1 failed, 3 passed, 4 total\n"
        "Time:        1.234 s\n"
    )

    summary = parse_test_output(output)

    assert (summary.passed, summary.failed) == (3, 1)
    assert summary.framework == "jest"
    assert [failure.name for failure in summary.failures] == [name]
