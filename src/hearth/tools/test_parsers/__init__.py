"""Turning test-runner output into something a small model can act on.

A failing `pytest` run is several hundred lines. The daily model here has a 12K window, so
handing over the raw text costs most of it and buries the one fact that matters. One line —
``2 failed: test_finalize_rounds_half_up (tests/test_invoice.py) AssertionError`` — carries
the same information and is what makes an edit → test → fix loop work on a 4B model rather
than merely look plausible.

**Placed in ``tools/`` rather than ``workflows/``.** docs/project-structure.md §1 lists
``workflows/test_parsers/`` next to ``test_writer.py``, for cohesion with the loop that
will use them in I3. But ``workflows`` sits *above* ``core`` which sits above ``tools``
(§2), so ``tools/tests.py`` could not import them there — and the parse belongs to the tool
that ran the tests, the same way interpreting a diff belongs to ``edit_file``. Here, both
consumers work: ``run_tests`` now, and I3's ``test_writer`` importing downward later.

**The parsers are forgiving on purpose.** Runners change their output between versions, and
a parser that returned nothing on an unfamiliar format would silently disable the fix-loop.
So an unreadable run reports ``parsed=False`` — never ``0 failed``, which would tell the
model the tests passed — and the caller falls back to truncated raw output. Degraded, not
broken.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Failures named in the one-line summary before it says "and N more".
MAX_LISTED_FAILURES = 8


@dataclass(frozen=True)
class TestFailure:
    """One failing test, as much as the output revealed."""

    name: str
    #: ``path`` or ``path:line`` when the runner said. The model turns this into a read.
    location: str | None = None
    message: str = ""

    def describe(self) -> str:
        where = f" ({self.location})" if self.location else ""
        detail = f" {self.message}" if self.message else ""
        return f"{self.name}{where}{detail}".rstrip()


@dataclass(frozen=True)
class TestSummary:
    """What a test run produced."""

    framework: str | None = None
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    xfailed: int = 0
    failures: tuple[TestFailure, ...] = ()
    duration_s: float | None = None
    #: False when the output could not be read. Distinct from "nothing failed".
    parsed: bool = False

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped + self.errors + self.xfailed

    @property
    def ok(self) -> bool:
        """Whether the run passed.

        Two things are never ok. An unparsed run: treating "could not read the output" as
        success is the one mistake here that would actively mislead the model into
        stopping. And a run in which nothing executed: zero tests "pass" vacuously, and a
        file of tests that were never collected is exactly what a confused model writes.
        Skipped tests count as having run, since they were collected and the suite said so.
        """
        return self.parsed and self.total > 0 and self.failed == 0 and self.errors == 0

    def describe(self) -> str:
        """The line (or few) the model sees instead of the full output."""
        if not self.parsed:
            return "could not parse the test output"

        counts = [
            f"{self.failed} failed" if self.failed else "",
            f"{self.errors} error{'s' if self.errors != 1 else ''}" if self.errors else "",
            f"{self.passed} passed" if self.passed else "",
            f"{self.skipped} skipped" if self.skipped else "",
            f"{self.xfailed} xfailed" if self.xfailed else "",
        ]
        headline = ", ".join(part for part in counts if part) or "no tests ran"
        if self.duration_s is not None:
            headline = f"{headline} in {self.duration_s:.2f}s"

        if not self.failures:
            return headline

        shown = self.failures[:MAX_LISTED_FAILURES]
        lines = [headline, *(f"  - {failure.describe()}" for failure in shown)]
        remaining = len(self.failures) - len(shown)
        if remaining > 0:
            lines.append(f"  … and {remaining} more")
        return "\n".join(lines)


def detect_framework(text: str) -> str | None:
    """Which runner produced this output, or None.

    None rather than a default: a wrong framework yields confidently wrong counts, which
    is worse than falling back to raw output.
    """
    if "Test Files" in text or re.search(r"^\s*[×✓❯]", text, re.MULTILINE):
        return "vitest"
    if re.search(r"^Tests:\s", text, re.MULTILINE) or " PASS " in text or " FAIL " in text:
        return "jest"
    if (
        "test session starts" in text
        or "no tests ran" in text
        or re.search(r"\b\d+ (?:passed|failed|error|skipped)\b", text)
    ):
        return "pytest"
    return None


def parse_test_output(text: str, *, framework: str | None = None) -> TestSummary:
    """Summarise a test run.

    Args:
        text: The runner's combined stdout and stderr.
        framework: Forced framework, e.g. from the project config. Beats sniffing.
    """
    if not text.strip():
        return TestSummary(parsed=False)

    chosen = framework or detect_framework(text)
    match chosen:
        case "pytest":
            return _parse_pytest(text)
        case "jest":
            return _parse_jest(text, framework="jest")
        case "vitest":
            return _parse_vitest(text)
        case _:
            return TestSummary(parsed=False)


# -------------------------------------------------------------------- pytest

#: pytest's final line: `===== 2 failed, 42 passed, 1 skipped in 3.21s =====`
_PYTEST_COUNT = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected)")
_PYTEST_DURATION = re.compile(r"\bin\s+([\d.]+)s")
_PYTEST_FAILED_LINE = re.compile(r"^(FAILED|ERROR)\s+(\S+?)(?:::(\S+))?\s*(?:-\s*(.*))?$", re.MULTILINE)


#: The closing line under ``-q``, which has no ``=====`` banner:
#: ``1 failed, 2 passed in 0.41s`` — optionally with a ``(0:00:00)`` clock after the time.
_PYTEST_QUIET_SUMMARY = re.compile(
    r"^\s*(?:\d+\s+\w+(?:,\s*)?)+\s+in\s+[\d.]+s\b", re.MULTILINE
)
_PYTEST_NO_TESTS = re.compile(r"\bno tests ran\b")


def _parse_pytest(text: str) -> TestSummary:
    counts: dict[str, int] = {}
    # Only the summary lines carry counts; scanning the whole output would pick up numbers
    # out of tracebacks and assertion messages. A summary line is either the banner form
    # (``===== 1 failed, 2 passed in 0.4s =====``) or, under ``-q``, the same words with no
    # banner. ``-q`` is this project's own default test command, so a parser that only knew
    # the banner never once parsed a run of it — passing or failing — and the model was
    # handed raw output instead of a one-line result.
    quiet_lines = {match.group(0).strip() for match in _PYTEST_QUIET_SUMMARY.finditer(text)}

    for line in text.splitlines():
        if not re.search(r"\b(?:passed|failed|error|skipped|xfailed)\b", line):
            continue
        is_banner = "=" in line or "short test summary" in line
        if not (is_banner or any(line.strip().startswith(quiet) for quiet in quiet_lines)):
            continue
        for amount, word in _PYTEST_COUNT.findall(line):
            counts[word] = max(counts.get(word, 0), int(amount))

    if not counts:
        # `no tests ran` is a *readable* result — zero tests, exit code 5 — and different
        # from output the parser cannot read. Reporting it as unparsed would tell the model
        # nothing about why nothing happened.
        if _PYTEST_NO_TESTS.search(text):
            duration = _PYTEST_DURATION.search(text)
            return TestSummary(
                framework="pytest",
                duration_s=float(duration.group(1)) if duration else None,
                parsed=True,
            )
        return TestSummary(framework="pytest", parsed=False)

    duration = _PYTEST_DURATION.search(text)
    failures = tuple(
        TestFailure(
            name=name or path,
            location=path,
            message=(message or "").strip(),
        )
        for _kind, path, name, message in _PYTEST_FAILED_LINE.findall(text)
    )

    return TestSummary(
        framework="pytest",
        passed=counts.get("passed", 0),
        failed=counts.get("failed", 0),
        skipped=counts.get("skipped", 0),
        errors=counts.get("error", 0) + counts.get("errors", 0),
        xfailed=counts.get("xfailed", 0),
        failures=failures,
        duration_s=float(duration.group(1)) if duration else None,
        parsed=True,
    )


# ---------------------------------------------------------------------- jest

#: `Tests:       2 failed, 42 passed, 44 total`
_JEST_SUMMARY = re.compile(r"^Tests:\s+(.*)$", re.MULTILINE)
_JEST_COUNT = re.compile(r"(\d+)\s+(passed|failed|skipped|todo|pending)")
_JEST_DURATION = re.compile(r"^Time:\s+([\d.]+)\s*s", re.MULTILINE)
#: `  ● InvoiceService › rounds half up`
_JEST_FAILURE = re.compile(r"^\s*●\s+(.+?)\s*$", re.MULTILINE)


def _parse_jest(text: str, *, framework: str) -> TestSummary:
    summary = _JEST_SUMMARY.search(text)
    if summary is None:
        return TestSummary(framework=framework, parsed=False)

    counts = {word: int(amount) for amount, word in _JEST_COUNT.findall(summary.group(1))}
    duration = _JEST_DURATION.search(text)
    failures = tuple(
        TestFailure(name=name.strip())
        for name in _JEST_FAILURE.findall(text)
        if not name.startswith("Console")
    )

    return TestSummary(
        framework=framework,
        passed=counts.get("passed", 0),
        failed=counts.get("failed", 0),
        skipped=counts.get("skipped", 0) + counts.get("pending", 0) + counts.get("todo", 0),
        failures=failures,
        duration_s=float(duration.group(1)) if duration else None,
        parsed=True,
    )


# -------------------------------------------------------------------- vitest

#: `      Tests  2 failed | 42 passed (44)`
_VITEST_SUMMARY = re.compile(r"^\s*Tests\s+(.*)$", re.MULTILINE)
_VITEST_COUNT = re.compile(r"(\d+)\s+(passed|failed|skipped|todo)")
_VITEST_DURATION = re.compile(r"^\s*Duration\s+([\d.]+)s", re.MULTILINE)
#: `   × InvoiceService > rounds half up 3ms`
_VITEST_FAILURE = re.compile(r"^\s*[×✗]\s+(.+?)(?:\s+\d+ms)?\s*$", re.MULTILINE)


def _parse_vitest(text: str) -> TestSummary:
    summary = _VITEST_SUMMARY.search(text)
    if summary is None:
        # vitest and jest share enough shape that its summary line is worth trying.
        return _parse_jest(text, framework="vitest")

    counts = {word: int(amount) for amount, word in _VITEST_COUNT.findall(summary.group(1))}
    duration = _VITEST_DURATION.search(text)
    failures = tuple(TestFailure(name=name.strip()) for name in _VITEST_FAILURE.findall(text))

    return TestSummary(
        framework="vitest",
        passed=counts.get("passed", 0),
        failed=counts.get("failed", 0),
        skipped=counts.get("skipped", 0) + counts.get("todo", 0),
        failures=failures,
        duration_s=float(duration.group(1)) if duration else None,
        parsed=True,
    )
