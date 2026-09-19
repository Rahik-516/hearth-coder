"""Tool gateway lifecycle, read tools, and the agent loop's bounds.

The scripted-transcript tests are an M4 acceptance criterion: invalid arguments, an unknown
tool and a repeated identical call must produce corrective errors, exhaust the retry
budget, and trigger loop detection.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hearth.core.agent_loop import AgentLoop
from hearth.core.bus import EventBus
from hearth.core.limits import StepTracker, TurnLimits
from hearth.indexing.pipeline import Indexer
from hearth.llm.tool_call_parser import parse_tool_calls
from hearth.llm.types import ChatChunk, ChatRequest, Message, ToolCall
from hearth.safety.audit import AuditLog
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate
from hearth.tools.base import ToolContext
from hearth.tools.channel import NullChannel
from hearth.tools.gateway import ToolGateway
from hearth.tools.registry import ToolRegistry, build_default_registry
from hearth.tools.results import ErrorCode

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "repos"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    import shutil

    destination = tmp_path / "py_small"
    shutil.copytree(FIXTURES / "py_small", destination)
    return destination


@pytest.fixture
def indexed(workspace: Path, tmp_path: Path):
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    Indexer(root=workspace, repository=IndexRepository(connection)).run()
    return connection


@pytest.fixture
def context(workspace: Path, indexed) -> ToolContext:
    return ToolContext(workspace=workspace, index_connection=indexed)


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_registry()


@pytest.fixture
def gateway(registry: ToolRegistry, context: ToolContext, tmp_path: Path) -> ToolGateway:
    return ToolGateway(
        registry=registry,
        context=context,
        channel=NullChannel(),
        audit=AuditLog(tmp_path / "audit", fsync=False),
        session_id="s_test",
    )


async def call(gateway: ToolGateway, tool_name: str, /, **arguments):
    """Positional-only tool name, so a tool argument called `name` does not collide."""
    return await gateway.call(tool_name, arguments, call_id="c1")


# --------------------------------------------------------------- read tools


async def test_read_file_returns_numbered_lines(gateway: ToolGateway) -> None:
    result = await call(gateway, "read_file", path="src/billing/payments.py")

    assert result.ok
    assert "TokenBucket" in result.content
    assert "1\t" in result.content


async def test_read_file_registers_a_hash(gateway: ToolGateway, context: ToolContext) -> None:
    """Read-before-write in M5 depends on this registry existing from M4."""
    await call(gateway, "read_file", path="src/billing/payments.py")

    assert context.hash_at_last_read("src/billing/payments.py")


async def test_read_file_paginates(gateway: ToolGateway) -> None:
    result = await call(gateway, "read_file", path="src/billing/payments.py", offset=1, limit=5)

    assert result.ok
    assert "more line(s)" in result.content
    assert result.metadata["shown_lines"] == 5


async def test_read_file_refuses_missing_file(gateway: ToolGateway) -> None:
    result = await call(gateway, "read_file", path="src/nope.py")

    assert not result.ok
    assert result.error is ErrorCode.NOT_FOUND
    assert "find_files" in result.content, "the error should suggest a next step"


async def test_read_file_refuses_traversal(gateway: ToolGateway) -> None:
    result = await call(gateway, "read_file", path="../../etc/passwd")

    assert not result.ok
    # A read outside the workspace resolves, then fails on absence or policy — either way
    # it must not return /etc/passwd's contents.
    assert "root:" not in result.content


async def test_read_file_refuses_secret_files(gateway: ToolGateway, workspace: Path) -> None:
    (workspace / ".env").write_text("SECRET=hunter2\n", encoding="utf-8")

    result = await call(gateway, "read_file", path=".env")

    assert not result.ok
    assert result.error is ErrorCode.PATH_REFUSED
    assert "hunter2" not in result.content


async def test_list_dir(gateway: ToolGateway) -> None:
    result = await call(gateway, "list_dir", path="src")

    assert result.ok
    assert "src/billing/payments.py" in result.content


async def test_find_files(gateway: ToolGateway) -> None:
    result = await call(gateway, "find_files", glob="src/**/*.py")

    assert result.ok
    assert "invoice_service.py" in result.content


async def test_find_files_refuses_escaping_globs(gateway: ToolGateway) -> None:
    result = await call(gateway, "find_files", glob="../*")

    assert not result.ok
    assert result.error is ErrorCode.PATH_REFUSED


async def test_grep_finds_matches(gateway: ToolGateway) -> None:
    result = await call(gateway, "grep", pattern="TokenBucket")

    assert result.ok
    assert "payments.py" in result.content


async def test_grep_reports_no_matches_clearly(gateway: ToolGateway) -> None:
    result = await call(gateway, "grep", pattern="zzz_not_present_anywhere")

    assert result.ok, "no matches is a valid answer, not a failure"
    assert "No matches" in result.content


async def test_grep_rejects_a_bad_regex(gateway: ToolGateway) -> None:
    result = await call(gateway, "grep", pattern="([unclosed", regex=True)

    assert not result.ok
    assert result.error is ErrorCode.INVALID_ARGUMENTS
    assert "regex=false" in result.content


async def test_find_symbol(gateway: ToolGateway) -> None:
    result = await call(gateway, "find_symbol", name="InvoiceService")

    assert result.ok
    assert "invoice_service.py" in result.content


async def test_find_references_states_its_limitation(gateway: ToolGateway) -> None:
    """The graph is name-based; the model must not report counts as certain."""
    result = await call(gateway, "find_references", name="finalize")

    assert result.ok
    assert "matched by name" in result.content


async def test_index_backed_tools_degrade_without_an_index(
    registry: ToolRegistry, workspace: Path, tmp_path: Path
) -> None:
    audit = AuditLog(tmp_path / "audit", fsync=False)
    bare = ToolGateway(
        registry=registry,
        context=ToolContext(workspace=workspace),
        channel=NullChannel(),
        audit=audit,
    )

    result = await bare.call("find_symbol", {"name": "InvoiceService"}, call_id="c1")

    assert not result.ok
    assert "not indexed" in result.content
    assert "grep" in result.content, "should point at a tool that does work"

    # A tool that cannot run is not a tool that was refused. Recording this as "deny"
    # would make the audit log answer "what did policy block?" wrongly, which is the
    # question the log exists to answer.
    record = audit.read_records()[-1]
    assert record["decision"] == "unavailable"
    assert record["decided_by"] == "tool"


# ----------------------------------------------------------------- lifecycle


async def test_unknown_tool_lists_the_real_ones(gateway: ToolGateway) -> None:
    result = await gateway.call("search_files", {}, call_id="c1")

    assert not result.ok
    assert result.error is ErrorCode.UNKNOWN_TOOL
    assert "find_files" in result.content


async def test_invalid_arguments_produce_a_schema_hint(gateway: ToolGateway) -> None:
    result = await gateway.call("read_file", {"filepath": "a.py"}, call_id="c1")

    assert not result.ok
    assert result.error is ErrorCode.INVALID_ARGUMENTS
    assert "read_file(" in result.content


async def test_extra_arguments_are_rejected(gateway: ToolGateway) -> None:
    """extra='forbid' turns an invented parameter into a correctable error."""
    result = await gateway.call(
        "read_file", {"path": "src/billing/models.py", "encoding": "utf-8"}, call_id="c1"
    )

    assert not result.ok
    assert result.error is ErrorCode.INVALID_ARGUMENTS


async def test_a_gateway_with_no_policy_refuses_side_effects(
    registry: ToolRegistry, context: ToolContext
) -> None:
    """The default policy is chat mode, so every side-effecting risk is refused.

    A gateway constructed without an explicit session is one nobody has decided the
    permissions for. Defaulting to agent mode would mean a forgotten argument silently
    upgrades what a tool is allowed to do.
    """
    from hearth.safety.policy import PolicyFacts, PolicyRequest
    from hearth.safety.risk import Risk
    from hearth.tools.gateway import default_policy

    policy = default_policy()

    for risk in (Risk.WRITE, Risk.EXEC, Risk.VCS_WRITE):
        request = PolicyRequest(
            tool="whatever",
            risk=risk,
            facts=PolicyFacts(path="src/a.py", absolute_path="/w/src/a.py"),
        )
        assert policy(request).action == "deny", risk

    reads = PolicyRequest(tool="read_file", risk=Risk.READ, facts=PolicyFacts(path="src/a.py"))
    assert policy(reads).action == "allow"


async def test_every_call_is_audited(gateway: ToolGateway, tmp_path: Path) -> None:
    """M4 acceptance: every tool call produces an audit record."""
    audit = AuditLog(tmp_path / "audit", fsync=False)

    await call(gateway, "read_file", path="src/billing/models.py")
    await gateway.call("nonexistent_tool", {}, call_id="c2")
    await gateway.call("read_file", {"bad": 1}, call_id="c3")

    records = audit.read_records()
    assert len(records) >= 3
    assert {r["tool"] for r in records} >= {"read_file", "nonexistent_tool"}
    assert any(r["decision"] == "deny" for r in records)


async def test_audit_redacts_secrets(tmp_path: Path) -> None:
    """The audit log is permanent; a credential in it would be too."""
    from hearth.safety.audit import AuditRecord

    audit = AuditLog(tmp_path / "audit", fsync=False)
    audit.write(
        AuditRecord(
            tool="grep",
            risk="READ",
            decision="allow",
            session="s",
            args={"pattern": "AWS_SECRET_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE"},
        )
    )

    raw = (tmp_path / "audit").glob("*.jsonl")
    text = next(raw).read_text(encoding="utf-8")

    assert "AKIAIOSFODNN7EXAMPLE" not in text
    assert "redacted" in text


# ------------------------------------------------------------- tool parsing


def test_native_calls_are_preferred() -> None:
    native = [ToolCall(call_id="1", name="read_file", arguments={"path": "a.py"})]
    outcome = parse_tool_calls(native=native, text='{"name": "grep"}')

    assert outcome.source == "native"
    assert outcome.calls == native


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>',
        '```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```',
        '{"name": "read_file", "arguments": {"path": "a.py"}}',
        '{"tool": "read_file", "args": {"path": "a.py"}}',
        '{"name": "read_file", "arguments": "{\\"path\\": \\"a.py\\"}"}',
    ],
)
def test_text_fallbacks_are_parsed(text: str) -> None:
    """Local models emit calls as prose; refusing them wastes a step and teaches nothing."""
    outcome = parse_tool_calls(text=text, known_tools={"read_file"})

    assert outcome.found
    assert outcome.calls[0].name == "read_file"
    assert outcome.calls[0].arguments == {"path": "a.py"}


def test_prose_mentioning_a_tool_is_not_a_call() -> None:
    """ "You could use read_file here" must not execute anything."""
    outcome = parse_tool_calls(text="You could use read_file to inspect it.", known_tools={"read_file"})
    assert not outcome.found


def test_unknown_tool_names_in_text_are_ignored() -> None:
    outcome = parse_tool_calls(text='{"name": "totally_made_up", "arguments": {}}', known_tools={"read_file"})
    assert not outcome.found


def test_malformed_json_is_reported_not_executed() -> None:
    outcome = parse_tool_calls(
        text='<tool_call>{"name": "read_file", broken}</tool_call>', known_tools={"read_file"}
    )

    assert not outcome.found
    assert outcome.malformed


# ------------------------------------------------------------------ limits


def test_step_limit_stops_the_turn() -> None:
    tracker = StepTracker(limits=TurnLimits(max_steps=3))
    for _ in range(3):
        tracker.begin_step()

    assert tracker.stop_reason() == "step_limit"


def test_retry_budget_counts_only_retryable_failures() -> None:
    """A denial is a decision, not a mistake; spending retries on it would end turns early."""
    tracker = StepTracker(limits=TurnLimits(max_retries=2))

    tracker.record_result(ok=False, retryable=False)
    tracker.record_result(ok=False, retryable=False)
    assert not tracker.retries_exhausted

    tracker.record_result(ok=False, retryable=True)
    tracker.record_result(ok=False, retryable=True)
    assert tracker.retries_exhausted


def test_success_resets_consecutive_failures() -> None:
    tracker = StepTracker(limits=TurnLimits(max_consecutive_failures=2))
    tracker.record_result(ok=False, retryable=True)
    tracker.record_result(ok=True, retryable=False)
    tracker.record_result(ok=False, retryable=True)

    assert not tracker.failing_repeatedly


def test_loop_detection_nudges_then_stops() -> None:
    tracker = StepTracker()
    args = {"path": "a.py"}

    assert tracker.observe_call("read_file", args) is None
    nudge = tracker.observe_call("read_file", args)
    assert nudge and "already called" in nudge
    stop = tracker.observe_call("read_file", args)
    assert stop and stop.startswith("STOP:")


def test_loop_detection_ignores_key_order() -> None:
    """A model re-emitting a call rarely reproduces key order exactly."""
    tracker = StepTracker()
    tracker.observe_call("grep", {"pattern": "x", "regex": False})
    nudge = tracker.observe_call("grep", {"regex": False, "pattern": "x"})

    assert nudge is not None


def test_different_arguments_are_not_a_loop() -> None:
    tracker = StepTracker()
    assert tracker.observe_call("read_file", {"path": "a.py"}) is None
    assert tracker.observe_call("read_file", {"path": "b.py"}) is None


def test_limits_scale_with_profile_reliability() -> None:
    from hearth.llm.profiles import ProfileRegistry

    registry = ProfileRegistry.load()
    small = TurnLimits.from_profile(registry.for_model("qwen3.5:4b", parameter_count_b=4.0), "agent")
    large = TurnLimits.from_profile(registry.for_model("qwen3.8:27b", parameter_count_b=27.0), "agent")

    assert small.max_steps < large.max_steps
    assert small.max_retries <= large.max_retries


# ------------------------------------------------------------------- loop


class ScriptedLoopProvider:
    """Emits a fixed sequence of responses, each optionally requesting tools."""

    def __init__(self, script: list[tuple[str, list[ToolCall]]]) -> None:
        self._script = script
        self.requests: list[ChatRequest] = []

    async def chat_stream(self, request: ChatRequest):
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._script) - 1)
        text, calls = self._script[index]
        if text:
            yield ChatChunk(content_delta=text)
        yield ChatChunk(tool_calls=calls, done=True, done_reason="stop")


def make_loop(provider, gateway: ToolGateway, **kwargs):
    return AgentLoop(provider=provider, gateway=gateway, bus=EventBus(), **kwargs)


def base_request() -> ChatRequest:
    return ChatRequest(
        model="scripted",
        messages=[Message(role="system", content="rules"), Message(role="user", content="q")],
        num_ctx=12_288,
    )


async def test_loop_answers_without_tools(gateway: ToolGateway) -> None:
    provider = ScriptedLoopProvider([("the answer", [])])
    outcome = await make_loop(provider, gateway).run(base_request=base_request())

    assert outcome.ok
    assert outcome.answer == "the answer"
    assert outcome.tool_calls == 0


async def test_loop_calls_a_tool_then_answers(gateway: ToolGateway) -> None:
    provider = ScriptedLoopProvider(
        [
            ("", [ToolCall(call_id="t1", name="find_symbol", arguments={"name": "TokenBucket"})]),
            ("It is in payments.py.", []),
        ]
    )
    outcome = await make_loop(provider, gateway).run(base_request=base_request())

    assert outcome.ok
    assert outcome.tool_calls == 1
    assert "payments.py" in outcome.answer


async def test_tool_results_reach_the_model(gateway: ToolGateway) -> None:
    provider = ScriptedLoopProvider(
        [
            ("", [ToolCall(call_id="t1", name="find_symbol", arguments={"name": "TokenBucket"})]),
            ("done", []),
        ]
    )
    await make_loop(provider, gateway).run(base_request=base_request())

    second = provider.requests[1]
    tool_messages = [m for m in second.messages if m.role == "tool"]
    assert tool_messages
    assert "payments.py" in tool_messages[0].content


async def test_step_limit_ends_the_loop(gateway: ToolGateway) -> None:
    """A model that keeps calling tools must not run forever."""
    provider = ScriptedLoopProvider(
        [("", [ToolCall(call_id="t", name="find_symbol", arguments={"name": f"S{i}"})]) for i in range(20)]
    )
    loop = make_loop(provider, gateway, limits=TurnLimits(max_steps=3))

    outcome = await loop.run(base_request=base_request())

    assert outcome.reason == "step_limit"
    assert outcome.steps <= 3


async def test_repeated_identical_calls_are_stopped(gateway: ToolGateway) -> None:
    """M4 acceptance: loop detection."""
    call_spec = ToolCall(call_id="t", name="find_symbol", arguments={"name": "Same"})
    provider = ScriptedLoopProvider([("", [call_spec])] * 10)

    outcome = await make_loop(provider, gateway, limits=TurnLimits(max_steps=10)).run(
        base_request=base_request()
    )

    assert outcome.reason == "loop_detected"


async def test_retry_budget_exhaustion_ends_the_loop(gateway: ToolGateway) -> None:
    """M4 acceptance: a model that cannot get the schema right stops burning steps."""
    provider = ScriptedLoopProvider(
        [("", [ToolCall(call_id=f"t{i}", name="read_file", arguments={"wrong": i})]) for i in range(10)]
    )
    loop = make_loop(provider, gateway, limits=TurnLimits(max_steps=10, max_retries=2))

    outcome = await loop.run(base_request=base_request())

    assert outcome.reason in ("retry_budget", "repeated_failures")
    assert outcome.steps < 10


async def test_corrective_errors_are_fed_back(gateway: ToolGateway) -> None:
    provider = ScriptedLoopProvider(
        [
            ("", [ToolCall(call_id="t1", name="read_file", arguments={"wrong": 1})]),
            ("sorry", []),
        ]
    )
    await make_loop(provider, gateway).run(base_request=base_request())

    tool_messages = [m for m in provider.requests[1].messages if m.role == "tool"]
    assert tool_messages
    assert "ERROR invalid_arguments" in tool_messages[0].content


async def test_reads_run_concurrently(gateway: ToolGateway) -> None:
    """Three independent reads should cost one round trip, not three."""
    calls = [
        ToolCall(call_id=f"t{i}", name="find_symbol", arguments={"name": n})
        for i, n in enumerate(["InvoiceService", "TokenBucket", "LineItem"])
    ]
    provider = ScriptedLoopProvider([("", calls), ("done", [])])

    outcome = await make_loop(provider, gateway).run(base_request=base_request())

    assert outcome.tool_calls == 3
    assert len([m for m in provider.requests[1].messages if m.role == "tool"]) == 3


async def test_prefix_stays_stable_across_steps(gateway: ToolGateway) -> None:
    """The KV cache must survive a multi-step turn, not just a single-shot one."""
    provider = ScriptedLoopProvider(
        [
            ("", [ToolCall(call_id="t1", name="find_symbol", arguments={"name": "X"})]),
            ("done", []),
        ]
    )
    await make_loop(provider, gateway).run(base_request=base_request())

    first, second = provider.requests
    assert first.messages[0].content == second.messages[0].content
    assert second.messages[: len(first.messages)] == first.messages


async def test_cancellation_propagates(gateway: ToolGateway) -> None:
    class HangingProvider:
        async def chat_stream(self, request):
            while True:
                await asyncio.sleep(0.02)
                yield ChatChunk(content_delta=".")

    loop = make_loop(HangingProvider(), gateway)
    task = asyncio.create_task(loop.run(base_request=base_request()))
    await asyncio.sleep(0.1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------- registry


def test_chat_mode_exposes_read_tools(registry: ToolRegistry) -> None:
    availability = registry.for_mode("chat", tool_reliability="high")

    assert "read_file" in availability.names
    assert len(availability.tools) <= 6


def test_chat_mode_does_not_expose_write_tools(registry: ToolRegistry) -> None:
    """A tool that is not exposed cannot be called by mistake, which is stronger than one
    that would be denied."""
    availability = registry.for_mode("chat", tool_reliability="high")

    assert "edit_file" not in availability.names
    assert "write_file" not in availability.names


def test_agent_mode_exposes_writes(registry: ToolRegistry) -> None:
    availability = registry.for_mode("agent", tool_reliability="high")

    assert "edit_file" in availability.names
    assert "write_file" in availability.names


def test_a_tool_cap_never_drops_the_write_tools(registry: ToolRegistry) -> None:
    """Found by a live run: `for_mode("agent", max_tools=8)` exposed ten read tools and
    no writes, because reads are listed first and the cap truncated from the end.

    An agent session that silently loses `edit_file` is a read-only session still calling
    itself agent mode — the model tries to answer by describing the change it cannot make.
    """
    availability = registry.for_mode("agent", tool_reliability="high", max_tools=6)

    assert len(availability.tools) == 6
    assert "edit_file" in availability.names
    assert "write_file" in availability.names
    assert "read_file" in availability.names, "reads are trimmed, not eliminated"


def test_a_cap_preserves_schema_order(registry: ToolRegistry) -> None:
    """The schema list is part of the cached prompt prefix, so order must stay stable."""
    full = registry.for_mode("agent", tool_reliability="high").names
    capped = registry.for_mode("agent", tool_reliability="high", max_tools=6).names

    assert list(capped) == [name for name in full if name in set(capped)]


def test_low_reliability_models_get_no_tools(registry: ToolRegistry) -> None:
    """A model that invents arguments does more harm with tools than without."""
    availability = registry.for_mode("chat", tool_reliability="low")

    assert availability.tools == ()
    assert "low tool reliability" in availability.reason


def test_schemas_are_generated_from_the_args_model(registry: ToolRegistry) -> None:
    """The schema the model sees and the one that validates its call cannot drift."""
    schema = registry.get("read_file").schema()

    assert schema["function"]["name"] == "read_file"
    assert "path" in schema["function"]["parameters"]["properties"]


def test_registry_refuses_duplicate_names() -> None:
    from hearth.tools.read_fs import ReadFileTool

    registry = ToolRegistry([ReadFileTool()])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ReadFileTool())


# ------------------------------------------------------- a stopped turn still answers


class ToolsWithdrawnProvider:
    """Keeps requesting the same tool until tools are taken away, then answers in words.

    That is what a real model does when the loop ends its turn: the wrap-up request has no
    tools attached, so the only thing left to produce is text.
    """

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    async def chat_stream(self, request: ChatRequest):
        self.requests.append(request)
        if request.tools:
            call = ToolCall(call_id="t", name="find_symbol", arguments={"name": "Same"})
            yield ChatChunk(tool_calls=[call], done=True, done_reason="stop")
            return
        yield ChatChunk(content_delta="I added the tests and they pass.")
        yield ChatChunk(done=True, done_reason="stop")


async def test_a_loop_stop_still_ends_with_an_answer(gateway: ToolGateway) -> None:
    """The real `hearth run` bug: the work was done, the model kept re-running the same
    call, the loop cut it off — and the user got an empty ending with no summary.

    `_final_answer` existed for exactly this and was never called.
    """
    provider = ToolsWithdrawnProvider()
    loop = make_loop(
        provider,
        gateway,
        limits=TurnLimits(max_steps=10),
        tool_schemas=[{"type": "function", "function": {"name": "find_symbol"}}],
    )

    outcome = await loop.run(base_request=base_request())

    assert outcome.reason == "loop_detected"
    assert "tests and they pass" in outcome.answer


async def test_a_step_limit_stop_still_ends_with_an_answer(gateway: ToolGateway) -> None:
    provider = ToolsWithdrawnProvider()
    loop = make_loop(
        provider,
        gateway,
        limits=TurnLimits(max_steps=1),
        tool_schemas=[{"type": "function", "function": {"name": "find_symbol"}}],
    )

    outcome = await loop.run(base_request=base_request())

    assert outcome.reason == "step_limit"
    assert outcome.answer.strip(), "a stopped turn must not end silently"


async def test_the_wrap_up_request_carries_no_tools_and_no_dangling_calls(
    gateway: ToolGateway,
) -> None:
    """Tools are withdrawn so the model can only answer, and a step that stopped before its
    calls ran must not leave an assistant message asking for tools that never reported back —
    a transcript with unanswered tool calls confuses some chat templates."""
    provider = ToolsWithdrawnProvider()
    loop = make_loop(
        provider,
        gateway,
        limits=TurnLimits(max_steps=1),
        tool_schemas=[{"type": "function", "function": {"name": "find_symbol"}}],
    )

    await loop.run(base_request=base_request())

    wrap_up = provider.requests[-1]
    assert wrap_up.tools == []
    assert not any(m.role == "assistant" and m.tool_calls for m in wrap_up.messages[2:])
