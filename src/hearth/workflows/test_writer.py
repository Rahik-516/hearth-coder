"""``/test <target>``: write tests, run them, and fix them until they pass (§5.11).

The loop is **write → run → parse → fix**, and the important design decision is *who runs
the tests*. Not the model. An agent asked to "write tests and run them" will report that it
ran them; a model that was never offered the tool, or ran out of steps first, reports it
just as fluently. This project has measured that: an earlier task-eval run recorded a
transcript in which the model "ran the tests" with no tool by which it could have. So the
workflow runs them itself, through ``run_tests`` and the gateway, and what the model is told
about the result is the parsed outcome of a run that demonstrably happened.

Everything that can be decided in code is:

* **What the project's conventions are** — which framework, where tests live, how they are
  named — by looking at the tree, not by asking a model to guess. The model is handed a
  path to write to and an existing test as a style sample.
* **Whether tests were written at all**, by comparing the test files before and after a
  turn. A model that answers "I've added tests" and wrote nothing is caught here.
* **Whether the source was touched.** The task is to test the code, not to change it, and a
  model whose test fails is tempted to make the failure go away in the source. That is
  reported, not prevented — the write tools are the model's — so the user can `/undo` it.
* **Whether the tests pass**, from a run that happened.

The model is used only where judgement is needed: what to assert.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from hearth.core.limits import TurnLimits
from hearth.core.runner import AgentTurnResult, ChatRunner
from hearth.core.session import Session
from hearth.indexing.filters import PathFilter
from hearth.prompts import load
from hearth.tools.gateway import ToolGateway
from hearth.workflows.targets import ResolvedTarget

#: Rounds of "run, show the failures, let the model fix" before giving up. Three because a
#: 4B model that has not fixed a test in three attempts with the failure in front of it is
#: repeating itself — the same conclusion the agent loop's own repeat detector reaches.
MAX_FIX_ITERATIONS = 3

#: Files examined when looking for conventions. A bound, not a target: past a few tens of
#: thousands the answer no longer changes and the scan is only latency.
MAX_FILES_WALKED = 30_000

#: Raw runner output kept when the parser could not read a run. The tail, because that
#: is where pytest and jest print the failure that matters.
MAX_RAW_TAIL_CHARS = 1_500

#: Lines of an existing test shown to the model as a style sample.
SAMPLE_LINES = 60

#: Test files the workflow will run individually after a turn. More than this and the model
#: has scattered tests across the tree, which is worth reporting rather than chasing.
MAX_FILES_RUN = 3

Status = Literal["passed", "failed", "no_tests_written", "tests_not_run", "refused"]

_SKIPPED_DIRS = frozenset(
    {".git", ".hearth", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", "target"}
)


@dataclass(frozen=True)
class TestConventions:
    """What the tree says about how tests are written here."""

    __test__ = False  # pytest: this is a data class, not a test class

    language: str
    framework: str
    #: Where the model is told to write. Workspace-relative POSIX.
    test_path: str
    #: True when ``test_path`` already exists, so the model extends it instead of
    #: overwriting somebody's tests.
    exists: bool = False
    sample_path: str | None = None
    sample_text: str = ""
    notes: tuple[str, ...] = ()


@dataclass
class TestWorkflowOutcome:
    __test__ = False

    status: Status
    detail: str = ""
    test_files: list[str] = field(default_factory=list)
    iterations: int = 0
    #: The last parsed test summary, as the model saw it.
    summary: str = ""
    #: The source file changed during the run. Reported, because the job was to test it.
    source_modified: bool = False
    conventions: TestConventions | None = None


# ------------------------------------------------------------------ conventions

_PY_TEST = re.compile(r"^(?:test_.+|.+_test)\.py$")
_JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs")


def detect_conventions(root: Path, target: ResolvedTarget) -> TestConventions | str:
    """Work out where tests go and what they look like. Returns a reason on refusal.

    A string return means "this workflow does not know how to test that", and the string
    says why in words the user can act on.
    """
    suffix = PurePosixPath(target.path).suffix.lower()

    if suffix == ".py":
        return _python_conventions(root, target)
    if suffix in _JS_EXTENSIONS:
        return _js_conventions(root, target)
    if suffix == ".go":
        return _go_conventions(root, target)

    return (
        f"/test does not know the test conventions for {suffix or 'this kind of file'}. "
        "Describe the tests you want and use /plan, or ask in agent mode."
    )


def is_test_file(relative: str) -> bool:
    """Whether a path looks like a test, by any supported convention."""
    name = PurePosixPath(relative).name
    if _PY_TEST.match(name):
        return True
    if re.search(r"\.(?:test|spec)\.(?:tsx?|jsx?|mjs)$", name):
        return True
    return name.endswith("_test.go")


def snapshot_test_files(root: Path) -> dict[str, str]:
    """Content hashes of every test file, for telling what a turn wrote."""
    hashes: dict[str, str] = {}
    for relative in _walk(root):
        if is_test_file(relative):
            hashes[relative] = _hash_file(root / relative)
    return hashes


def changed_test_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Test files that are new or different since ``before``. Sorted, for determinism."""
    return sorted(path for path, digest in after.items() if before.get(path) != digest)


def _python_conventions(root: Path, target: ResolvedTarget) -> TestConventions:
    source = PurePosixPath(target.path)
    existing = [path for path in _walk(root) if _PY_TEST.match(PurePosixPath(path).name)]
    notes: list[str] = []

    same_name = [path for path in existing if PurePosixPath(path).name == f"test_{source.stem}.py"]
    if same_name:
        chosen = same_name[0]
        notes.append(f"{chosen} already exists; extend it rather than replacing it.")
    else:
        directory = _best_test_dir(existing, source.parent) or "tests"
        # Joined as a path, not formatted: a test directory of "." (tests that sit at the
        # repository root) would otherwise give "./test_x.py", which is a different string
        # for the same file and would never match `covers()` or the changed-file check.
        chosen = str(PurePosixPath(directory) / f"test_{source.stem}.py")

    sample = _pick_sample(root, existing, PurePosixPath(chosen).parent, exclude=chosen)
    return TestConventions(
        language="python",
        framework=_python_framework(root),
        test_path=chosen,
        exists=(root / chosen).is_file(),
        sample_path=sample[0] if sample else None,
        sample_text=sample[1] if sample else "",
        notes=tuple(notes),
    )


def _python_framework(root: Path) -> str:
    """pytest unless the tree says unittest. pytest runs unittest tests too, so the only
    reason to say otherwise is a project that has clearly chosen it."""
    pyproject = _read(root / "pyproject.toml")
    if "pytest" in pyproject or (root / "pytest.ini").is_file() or (root / "conftest.py").is_file():
        return "pytest"
    sample = [path for path in _walk(root) if _PY_TEST.match(PurePosixPath(path).name)][:20]
    texts = [_read(root / path) for path in sample]
    if texts and all("import pytest" not in text for text in texts) and any(
        "unittest" in text for text in texts
    ):
        return "unittest"
    return "pytest"


def _best_test_dir(existing: list[str], source_dir: PurePosixPath) -> str | None:
    """The directory whose existing tests sit closest to the source's own location.

    A tree with ``tests/unit/billing/`` mirroring ``src/billing/`` wants a new test in the
    mirror, not in whatever directory happens to hold the most files. So directories are
    scored by how many trailing path components they share with the source's directory,
    with file count only breaking ties.
    """
    if not existing:
        return None

    counts = Counter(str(PurePosixPath(path).parent) for path in existing)
    source_parts = list(source_dir.parts)[::-1]

    def score(directory: str) -> tuple[int, int]:
        shared = 0
        for a, b in zip(source_parts, list(PurePosixPath(directory).parts)[::-1], strict=False):
            if a != b:
                break
            shared += 1
        return shared, counts[directory]

    return max(counts, key=score)


def _js_conventions(root: Path, target: ResolvedTarget) -> TestConventions:
    source = PurePosixPath(target.path)
    package = _read(root / "package.json")
    # Jest is also the answer when package.json names neither: it is the common default,
    # and a wrong guess here costs a failing import the fix loop will show the model.
    framework = "vitest" if "vitest" in package else "jest"

    existing = [
        path for path in _walk(root) if re.search(r"\.(?:test|spec)\.(?:tsx?|jsx?|mjs)$", path)
    ]
    style = Counter("spec" if ".spec." in path else "test" for path in existing)
    word = style.most_common(1)[0][0] if style else "test"

    chosen = str(source.with_name(f"{source.stem}.{word}{source.suffix}"))
    sample = _pick_sample(root, existing, PurePosixPath(chosen).parent, exclude=chosen)
    return TestConventions(
        language="javascript",
        framework=framework,
        test_path=chosen,
        exists=(root / chosen).is_file(),
        sample_path=sample[0] if sample else None,
        sample_text=sample[1] if sample else "",
    )


def _go_conventions(root: Path, target: ResolvedTarget) -> TestConventions:
    source = PurePosixPath(target.path)
    chosen = str(source.with_name(f"{source.stem}_test.go"))
    existing = [path for path in _walk(root) if path.endswith("_test.go")]
    sample = _pick_sample(root, existing, source.parent, exclude=chosen)
    return TestConventions(
        language="go",
        framework="go test",
        test_path=chosen,
        exists=(root / chosen).is_file(),
        sample_path=sample[0] if sample else None,
        sample_text=sample[1] if sample else "",
    )


def _pick_sample(
    root: Path, existing: list[str], near: PurePosixPath, *, exclude: str
) -> tuple[str, str] | None:
    """An existing test to imitate, preferring one in the same directory.

    Shown to the model as a style sample. Not too short (nothing to learn from) and not
    too long (a 60-line window of a 600-line file is representative of its imports and
    first tests, which is what carries the conventions).
    """
    candidates = sorted((p for p in existing if p != exclude), key=lambda p: (
        0 if str(PurePosixPath(p).parent) == str(near) else 1, p
    ))
    for path in candidates:
        text = _read(root / path)
        lines = text.splitlines()
        if len(lines) >= 8:
            return path, "\n".join(lines[:SAMPLE_LINES])
    return None


# ------------------------------------------------------------------- the prompt


def build_prompt(target: ResolvedTarget, conventions: TestConventions) -> str:
    """The instruction for the first turn, from the template and the facts."""
    where = target.describe()
    notes = "\n".join(f"- {note}" for note in conventions.notes) or "- (none)"

    if conventions.sample_path:
        sample = (
            f"An existing test in this project ({conventions.sample_path}), for style — "
            f"imports, naming, structure:\n\n```\n{conventions.sample_text}\n```"
        )
    else:
        sample = "There are no existing tests to copy the style from."

    return (
        load("workflows/test")
        .replace("{{target}}", where)
        .replace("{{source_path}}", target.path)
        .replace("{{framework}}", conventions.framework)
        .replace("{{test_path}}", conventions.test_path)
        .replace("{{notes}}", notes)
        .replace("{{sample}}", sample)
    )


def build_fix_prompt(summary: str, files: list[str], iteration: int) -> str:
    """The instruction for a fix round, carrying the parsed failure of a real run."""
    return (
        load("workflows/test_fix")
        .replace("{{summary}}", summary)
        .replace("{{files}}", ", ".join(files))
        .replace("{{iteration}}", str(iteration))
        .replace("{{max}}", str(MAX_FIX_ITERATIONS))
    )


# ------------------------------------------------------------------ the workflow


async def run_test_workflow(
    *,
    runner: ChatRunner,
    session: Session,
    gateway: ToolGateway,
    root: Path,
    target: ResolvedTarget,
    tool_schemas: list[dict[str, Any]],
    limits: TurnLimits | None = None,
) -> TestWorkflowOutcome:
    """Write tests for ``target`` and drive them to green, or say precisely why not."""
    conventions = detect_conventions(root, target)
    if isinstance(conventions, str):
        return TestWorkflowOutcome("refused", detail=conventions)

    source_before = _hash_file(root / target.path)
    tests_before = snapshot_test_files(root)
    outcome = TestWorkflowOutcome("failed", conventions=conventions)

    prompt = build_prompt(target, conventions)
    written: list[str] = []

    for iteration in range(MAX_FIX_ITERATIONS + 1):
        turn = await runner.run_agent_turn(
            session, prompt, gateway=gateway, tool_schemas=tool_schemas, limits=limits
        )
        outcome.iterations = iteration + 1

        written = changed_test_files(tests_before, snapshot_test_files(root))
        outcome.test_files = written
        outcome.source_modified = _hash_file(root / target.path) != source_before

        if not written:
            outcome.status = "no_tests_written"
            outcome.detail = _no_tests_detail(turn)
            return outcome

        run = await _run_written_tests(gateway, written)
        outcome.summary = run.summary

        if run.status == "not_run":
            outcome.status = "tests_not_run"
            outcome.detail = run.summary
            return outcome
        if run.status == "passed":
            outcome.status = "passed"
            return outcome

        if iteration == MAX_FIX_ITERATIONS:
            break
        prompt = build_fix_prompt(run.summary, written, iteration + 1)

    outcome.status = "failed"
    outcome.detail = f"still failing after {MAX_FIX_ITERATIONS} fix round(s)."
    return outcome


@dataclass
class _Run:
    status: Literal["passed", "failed", "not_run"]
    summary: str


async def _run_written_tests(gateway: ToolGateway, files: list[str]) -> _Run:
    """Run the test files the model wrote, through the gateway, and read the result.

    ``tests_ok`` comes from the parsed run and is False for "could not parse", so an
    unreadable result is never mistaken for a pass. A run in which zero tests executed is
    also not a pass: a file of tests that were never collected passes vacuously, and that
    is exactly the file a confused model produces.
    """
    lines: list[str] = []
    all_ok = True

    for path in files[:MAX_FILES_RUN]:
        result = await gateway.call(
            "run_tests", {"target": path}, call_id=f"wf-test-{uuid.uuid4().hex[:8]}"
        )
        if result.error is not None and result.error.value in ("denied", "rejected"):
            return _Run("not_run", f"the test run was {result.error.value}: {result.content}")

        executed = sum(_count(result.metadata, key) for key in ("passed", "failed", "skipped"))
        parsed_ok = bool(result.metadata.get("tests_ok"))
        readable = bool(result.metadata.get("parsed"))
        errors = _count(result.metadata, "errors")

        if readable and executed == 0 and errors == 0:
            lines.append(f"{path}: the file ran but no tests were collected from it.")
            all_ok = False
        elif parsed_ok:
            lines.append(f"{path}: {_headline(result.content)}")
        else:
            all_ok = False
            lines.append(f"{path}:\n{_failure_text(result.content)}")

    return _Run("passed" if all_ok else "failed", "\n".join(lines))


def _count(metadata: dict[str, object], key: str) -> int:
    """A non-negative integer from tool metadata, which is typed ``object``."""
    value = metadata.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _headline(content: str) -> str:
    """The parsed one-liner. ``run_tests`` puts it first, before the raw output."""
    return content.strip().split("\n\n", 1)[0]


def _failure_text(content: str) -> str:
    """What the model is shown about a failing run: the parsed summary, then enough raw
    output to act on if the parser could not read it. Bounded, because the fix round has to
    fit in the window beside the file it is fixing."""
    head, _, raw = content.strip().partition("\n\n")
    if "could not parse" in head and raw:
        return f"{head}\n{raw[-MAX_RAW_TAIL_CHARS:]}"
    return head


def _no_tests_detail(turn: AgentTurnResult) -> str:
    if turn.reason not in ("answered", "stop", ""):
        return f"the turn ended early ({turn.reason.replace('_', ' ')}) before any test file was written."
    return (
        "the model finished without writing a test file. Its answer was not a test — "
        "try again, or narrow the target to one function."
    )


# ------------------------------------------------------------------ small helpers


def _walk(root: Path) -> list[str]:
    """Workspace-relative POSIX paths of files, skipping the usual noise, bounded."""
    path_filter = PathFilter()
    found: list[str] = []
    for path in sorted(root.rglob("*")):
        if len(found) >= MAX_FILES_WALKED:
            break
        parts = path.relative_to(root).parts
        if any(part in _SKIPPED_DIRS for part in parts) or not path.is_file():
            continue
        relative = "/".join(parts)
        if path_filter.decide(relative).include:
            found.append(relative)
    return found


def _hash_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
