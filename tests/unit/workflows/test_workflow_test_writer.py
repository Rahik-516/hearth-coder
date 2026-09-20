"""`/test` — I3.

Two halves. The first is convention detection, which is pure tree-reading and is tested on
small synthetic trees. The second is the loop, which is driven with a scripted model against
a **real pytest run** on a copy of the `py_small` fixture — because the claim worth testing
is not "the workflow calls a tool", it is "the workflow's verdict comes from tests that
actually ran".

That last point is the reason for the shape of these tests. An earlier version of this
project's task eval reported passes that came from the harness running the suite, while a
transcript claimed the model had; the fix was to make the *workflow* the one that runs the
tests. So the loop tests below have a model that writes tests and says whatever it likes,
and assert on what the run actually found.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from hearth.core.bus import EventBus
from hearth.core.runner import ChatRunner
from hearth.core.session import Mode, Session
from hearth.llm.scripted_provider import ScriptedProvider, ScriptedResponse
from hearth.llm.types import ToolCall
from hearth.safety.checkpoints import CheckpointStore
from hearth.safety.policy import ConfigView, SessionView
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository
from hearth.tools.base import ToolContext
from hearth.tools.channel import ApprovalAsk, ApprovalReply
from hearth.tools.gateway import ToolGateway, make_policy
from hearth.tools.read_fs import ReadFileTool
from hearth.tools.registry import ToolRegistry
from hearth.tools.tests import RunTestsTool
from hearth.tools.write_fs import EditFileTool, WriteFileTool
from hearth.workflows.targets import ResolvedTarget
from hearth.workflows.test_writer import (
    build_prompt,
    changed_test_files,
    detect_conventions,
    is_test_file,
    run_test_workflow,
    snapshot_test_files,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "repos"

GOOD_TESTS = '''\
from billing.errors import BillingError, InvoiceNotFound, PaymentDeclined


def test_errors_share_a_base() -> None:
    assert issubclass(InvoiceNotFound, BillingError)


def test_a_declined_payment_carries_its_reason() -> None:
    error = PaymentDeclined("inv-1", "card expired")

    assert error.reason == "card expired"
    assert "card expired" in str(error)
'''

FAILING_TESTS = '''\
from billing.errors import InvoiceNotFound


def test_the_message_names_the_invoice() -> None:
    assert str(InvoiceNotFound("inv-1")) == "wrong message"
'''


# ===================================================================== conventions


def make_tree(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def target(path: str) -> ResolvedTarget:
    return ResolvedTarget(path=path)


def test_python_tests_go_in_the_existing_tests_directory(tmp_path: Path) -> None:
    root = make_tree(
        tmp_path,
        {
            "src/pkg/mod.py": "def f(): ...\n",
            "tests/test_other.py": "import pytest\n" + "x = 1\n" * 10,
            "pyproject.toml": "[tool.pytest.ini_options]\n",
        },
    )

    conventions = detect_conventions(root, target("src/pkg/mod.py"))

    assert not isinstance(conventions, str)
    assert conventions.test_path == "tests/test_mod.py"
    assert conventions.framework == "pytest"


def test_a_mirrored_test_tree_gets_a_mirrored_location(tmp_path: Path) -> None:
    """`tests/unit/billing/` mirrors `src/billing/`, so a new test belongs there — not in
    whichever directory happens to hold the most files."""
    files = {"src/billing/tax.py": "def f(): ...\n"}
    for n in range(5):
        files[f"tests/unit/other/test_o{n}.py"] = "import pytest\n" + "x = 1\n" * 10
    files["tests/unit/billing/test_invoice.py"] = "import pytest\n" + "x = 1\n" * 10
    root = make_tree(tmp_path, files)

    conventions = detect_conventions(root, target("src/billing/tax.py"))

    assert not isinstance(conventions, str)
    assert conventions.test_path == "tests/unit/billing/test_tax.py"


def test_an_existing_test_file_is_extended_not_replaced(tmp_path: Path) -> None:
    """Overwriting somebody's tests to add to them would be the worst outcome here."""
    root = make_tree(
        tmp_path,
        {
            "src/mod.py": "def f(): ...\n",
            "tests/test_mod.py": "def test_existing(): ...\n",
        },
    )

    conventions = detect_conventions(root, target("src/mod.py"))

    assert not isinstance(conventions, str)
    assert conventions.exists
    assert any("extend" in note for note in conventions.notes)


def test_with_no_tests_at_all_the_default_is_a_tests_directory(tmp_path: Path) -> None:
    root = make_tree(tmp_path, {"src/mod.py": "def f(): ...\n"})

    conventions = detect_conventions(root, target("src/mod.py"))

    assert not isinstance(conventions, str)
    assert conventions.test_path == "tests/test_mod.py"
    assert conventions.sample_path is None


def test_an_existing_test_is_offered_as_a_style_sample(tmp_path: Path) -> None:
    root = make_tree(
        tmp_path,
        {
            "src/mod.py": "def f(): ...\n",
            "tests/test_other.py": "import pytest\n\n\ndef test_a():\n" + "    assert True\n" * 10,
        },
    )

    conventions = detect_conventions(root, target("src/mod.py"))

    assert not isinstance(conventions, str)
    assert conventions.sample_path == "tests/test_other.py"
    assert "import pytest" in conventions.sample_text


def test_a_javascript_project_follows_its_own_test_word(tmp_path: Path) -> None:
    root = make_tree(
        tmp_path,
        {
            "package.json": '{"devDependencies": {"vitest": "^1"}}',
            "src/util.ts": "export const f = 1\n",
            "src/other.spec.ts": "import { it } from 'vitest'\n" + "it('a', () => {})\n" * 10,
        },
    )

    conventions = detect_conventions(root, target("src/util.ts"))

    assert not isinstance(conventions, str)
    assert conventions.test_path == "src/util.spec.ts"
    assert conventions.framework == "vitest"


def test_go_tests_sit_beside_the_source(tmp_path: Path) -> None:
    root = make_tree(tmp_path, {"pkg/calc.go": "package pkg\n"})

    conventions = detect_conventions(root, target("pkg/calc.go"))

    assert not isinstance(conventions, str)
    assert conventions.test_path == "pkg/calc_test.go"


def test_an_unsupported_language_is_refused_with_a_way_forward(tmp_path: Path) -> None:
    root = make_tree(tmp_path, {"lib/thing.rb": "def f; end\n"})

    reason = detect_conventions(root, target("lib/thing.rb"))

    assert isinstance(reason, str)
    assert "/plan" in reason


def test_noise_directories_do_not_count_as_existing_tests(tmp_path: Path) -> None:
    """A vendored package's tests in `.venv` or `node_modules` say nothing about this
    project's conventions."""
    root = make_tree(
        tmp_path,
        {
            "src/mod.py": "def f(): ...\n",
            ".venv/lib/site-packages/pkg/test_vendor.py": "def test_v(): ...\n",
            "node_modules/x/y.test.js": "it()\n",
        },
    )

    assert snapshot_test_files(root) == {}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("test_a.py", True),
        ("a_test.py", True),
        ("a.py", False),
        ("a.test.ts", True),
        ("a.spec.tsx", True),
        ("a.ts", False),
        ("a_test.go", True),
        ("a.go", False),
    ],
)
def test_what_counts_as_a_test_file(name: str, expected: bool) -> None:
    assert is_test_file(f"some/dir/{name}") is expected


def test_only_new_or_changed_test_files_are_reported() -> None:
    before = {"tests/test_a.py": "h1", "tests/test_b.py": "h2"}
    after = {"tests/test_a.py": "h1", "tests/test_b.py": "CHANGED", "tests/test_c.py": "h3"}

    assert changed_test_files(before, after) == ["tests/test_b.py", "tests/test_c.py"]


def test_the_prompt_names_the_path_and_leaves_no_placeholder(tmp_path: Path) -> None:
    root = make_tree(tmp_path, {"src/mod.py": "def f(): ...\n"})
    conventions = detect_conventions(root, target("src/mod.py"))
    assert not isinstance(conventions, str)

    prompt = build_prompt(target("src/mod.py"), conventions)

    assert "tests/test_mod.py" in prompt
    assert "{{" not in prompt
    assert "Do not run the tests yourself" in prompt


# ========================================================================= the loop


class Approving:
    """Approves everything and records what it was asked."""

    def __init__(self) -> None:
        self.asks: list[ApprovalAsk] = []

    async def proposed(self, **_: object) -> None: ...

    async def started(self, **_: object) -> None: ...

    async def finished(self, **_: object) -> None: ...

    async def request_approval(self, ask: ApprovalAsk) -> ApprovalReply:
        self.asks.append(ask)
        return ApprovalReply(decision="approve")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(FIXTURES / "py_small", root)
    return root


def write_call(path: str, content: str, call_id: str = "w1") -> ToolCall:
    return ToolCall(call_id=call_id, name="write_file", arguments={"path": path, "content": content})


def edit_call(path: str, old: str, new: str, call_id: str = "e1") -> ToolCall:
    return ToolCall(
        call_id=call_id,
        name="edit_file",
        arguments={"path": path, "old_string": old, "new_string": new},
    )


def read_call(path: str, call_id: str = "r1") -> ToolCall:
    return ToolCall(call_id=call_id, name="read_file", arguments={"path": path})


def build(workspace: Path, tmp_path: Path, script: list[ScriptedResponse]):
    connection = connect(tmp_path / "state.db")
    migrate(connection, database="state")
    repo = StateRepository(connection)
    checkpoints = CheckpointStore(repo, BlobStore(tmp_path / "blobs"))
    session_row = repo.create_session(workspace=str(workspace))

    session = Session(
        id=session_row.id, workspace=workspace, model="scripted", num_ctx=8192, mode=Mode.AGENT
    )
    channel = Approving()
    context = ToolContext(workspace=workspace, checkpoints=checkpoints.bind(session.id))
    gateway = ToolGateway(
        registry=ToolRegistry(
            [
                ReadFileTool(),
                WriteFileTool(),
                EditFileTool(),
                # An absolute interpreter path, so the suite under test runs in the
                # environment this suite runs in regardless of what is on PATH.
                RunTestsTool(test_command=f"{sys.executable} -m pytest -q -p no:cacheprovider"),
            ]
        ),
        context=context,
        channel=channel,
        policy=make_policy(SessionView(mode="agent", level="supervised"), ConfigView()),
        session_id=session.id,
    )
    provider = ScriptedProvider(script)
    runner = ChatRunner(provider=provider, bus=EventBus())
    schemas = [tool.schema() for tool in (ReadFileTool(), WriteFileTool(), EditFileTool())]
    return session, gateway, runner, provider, channel, schemas


async def run(workspace: Path, tmp_path: Path, script: list[ScriptedResponse]):
    session, gateway, runner, provider, channel, schemas = build(workspace, tmp_path, script)
    outcome = await run_test_workflow(
        runner=runner,
        session=session,
        gateway=gateway,
        root=workspace,
        target=ResolvedTarget(path="src/billing/errors.py"),
        tool_schemas=schemas,
    )
    return outcome, provider, channel


async def test_good_tests_are_written_run_and_pass(workspace: Path, tmp_path: Path) -> None:
    outcome, _provider, _channel = await run(
        workspace,
        tmp_path,
        [
            ScriptedResponse("", tool_calls=[write_call("tests/test_errors.py", GOOD_TESTS)]),
            ScriptedResponse("I added tests for the error types."),
        ],
    )

    assert outcome.status == "passed", outcome.summary or outcome.detail
    assert outcome.test_files == ["tests/test_errors.py"]
    assert outcome.iterations == 1
    assert (workspace / "tests" / "test_errors.py").read_text(encoding="utf-8") == GOOD_TESTS


async def test_the_verdict_comes_from_a_run_that_happened(workspace: Path, tmp_path: Path) -> None:
    """The reason the workflow runs the tests itself.

    The model here *claims* success and wrote a failing test. If the verdict were taken
    from what the model said, this would pass.
    """
    outcome, _provider, _channel = await run(
        workspace,
        tmp_path,
        [
            ScriptedResponse("", tool_calls=[write_call("tests/test_errors.py", FAILING_TESTS)]),
            ScriptedResponse("All tests pass, I ran them and they are green."),
        ]
        + [ScriptedResponse("Still fine.")] * 6,
    )

    assert outcome.status == "failed"
    assert "wrong message" in outcome.summary or "failed" in outcome.summary


async def test_the_test_run_is_asked_about_like_any_other_command(
    workspace: Path, tmp_path: Path
) -> None:
    """The workflow's `run_tests` goes through the gateway, so the user sees and approves
    it — running tests is EXEC, and a workflow does not get a shortcut around that."""
    _outcome, _provider, channel = await run(
        workspace,
        tmp_path,
        [
            ScriptedResponse("", tool_calls=[write_call("tests/test_errors.py", GOOD_TESTS)]),
            ScriptedResponse("Done."),
        ],
    )

    assert any("pytest" in ask.preview for ask in channel.asks)


async def test_a_declined_test_run_stops_the_workflow(workspace: Path, tmp_path: Path) -> None:
    """Declining is an answer. The workflow reports that the tests were not run rather than
    asking again or claiming a result."""
    session, gateway, runner, _provider, channel, schemas = build(
        workspace,
        tmp_path,
        [
            ScriptedResponse("", tool_calls=[write_call("tests/test_errors.py", GOOD_TESTS)]),
            ScriptedResponse("Done."),
        ],
    )

    async def decline_run_tests(ask: ApprovalAsk) -> ApprovalReply:
        return ApprovalReply(decision="reject" if "pytest" in ask.preview else "approve")

    channel.request_approval = decline_run_tests  # type: ignore[method-assign]

    outcome = await run_test_workflow(
        runner=runner,
        session=session,
        gateway=gateway,
        root=workspace,
        target=ResolvedTarget(path="src/billing/errors.py"),
        tool_schemas=schemas,
    )

    assert outcome.status == "tests_not_run"


async def test_a_model_that_writes_nothing_is_caught(workspace: Path, tmp_path: Path) -> None:
    """"I've added tests" with no test file is the model reporting an intention."""
    outcome, _provider, _channel = await run(
        workspace, tmp_path, [ScriptedResponse("I have added thorough tests for the errors.")]
    )

    assert outcome.status == "no_tests_written"
    assert outcome.test_files == []


async def test_a_failing_run_is_fed_back_and_a_fix_lands(workspace: Path, tmp_path: Path) -> None:
    """Write → run (fail) → fix prompt with the failure → fix → run (pass)."""
    outcome, provider, _channel = await run(
        workspace,
        tmp_path,
        [
            ScriptedResponse("", tool_calls=[write_call("tests/test_errors.py", FAILING_TESTS)]),
            ScriptedResponse("Wrote a test."),
            ScriptedResponse("", tool_calls=[write_call("tests/test_errors.py", GOOD_TESTS, "w2")]),
            ScriptedResponse("Fixed the expectation."),
        ],
    )

    assert outcome.status == "passed"
    assert outcome.iterations == 2

    # The last *user* message, not the last message: the scripted provider records the loop's
    # live message list, which keeps growing after the request was made.
    fix_request = provider.requests[2]
    fix_prompt = next(m.content for m in reversed(fix_request.messages) if m.role == "user")
    assert "did not pass" in fix_prompt
    assert "wrong message" in fix_prompt, "the failure the model sees is from the real run"


async def test_the_fix_loop_is_bounded(workspace: Path, tmp_path: Path) -> None:
    """A model that cannot fix a test in three attempts with the failure in front of it is
    repeating itself; the workflow stops and says so."""
    outcome, _provider, _channel = await run(
        workspace,
        tmp_path,
        [ScriptedResponse("", tool_calls=[write_call("tests/test_errors.py", FAILING_TESTS)])]
        + [ScriptedResponse("Tried again.")] * 12,
    )

    assert outcome.status == "failed"
    assert outcome.iterations == 4, "one write round plus three fix rounds"
    assert "still failing" in outcome.detail


async def test_editing_the_source_is_reported(workspace: Path, tmp_path: Path) -> None:
    """The model's failing test tempts it to change the code until the test passes. That is
    not prevented — the write tools are the model's — but it is never silent."""
    outcome, _provider, _channel = await run(
        workspace,
        tmp_path,
        [
            ScriptedResponse("", tool_calls=[read_call("src/billing/errors.py")]),
            ScriptedResponse(
                "",
                tool_calls=[
                    write_call("tests/test_errors.py", GOOD_TESTS, "w1"),
                    edit_call("src/billing/errors.py", "Outbound gateway", "Outbound HTTP gateway"),
                ],
            ),
            ScriptedResponse("Done."),
        ],
    )

    assert outcome.source_modified


async def test_untouched_source_is_not_reported_as_modified(
    workspace: Path, tmp_path: Path
) -> None:
    outcome, _provider, _channel = await run(
        workspace,
        tmp_path,
        [
            ScriptedResponse("", tool_calls=[write_call("tests/test_errors.py", GOOD_TESTS)]),
            ScriptedResponse("Done."),
        ],
    )

    assert not outcome.source_modified


async def test_a_test_file_with_no_tests_is_not_a_pass(workspace: Path, tmp_path: Path) -> None:
    """A file that ran but collected nothing passes vacuously — exactly what a confused
    model produces."""
    outcome, _provider, _channel = await run(
        workspace,
        tmp_path,
        [
            ScriptedResponse(
                "",
                tool_calls=[
                    write_call("tests/test_errors.py", "from billing.errors import BillingError\n")
                ],
            ),
            ScriptedResponse("Done."),
        ]
        + [ScriptedResponse("Still nothing.")] * 6,
    )

    assert outcome.status == "failed"
    assert "no tests were collected" in outcome.summary


async def test_an_unsupported_language_never_reaches_the_model(
    workspace: Path, tmp_path: Path
) -> None:
    (workspace / "lib").mkdir()
    (workspace / "lib" / "thing.rb").write_text("def f; end\n", encoding="utf-8")
    session, gateway, runner, provider, _channel, schemas = build(workspace, tmp_path, [])

    outcome = await run_test_workflow(
        runner=runner,
        session=session,
        gateway=gateway,
        root=workspace,
        target=ResolvedTarget(path="lib/thing.rb"),
        tool_schemas=schemas,
    )

    assert outcome.status == "refused"
    assert provider.requests == []
