"""Chat turns, session state and persistence.

Runs entirely against ``ScriptedProvider``: the point is the turn mechanics — event order,
citation handling, cancellation, persistence — not whether a model writes good prose.
Model quality is the eval's job.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from hearth.core.bus import EventBus
from hearth.core.events import (
    ContextStats,
    Notice,
    RetrievalPerformed,
    TextDelta,
    TurnFinished,
)
from hearth.core.runner import ChatRunner
from hearth.core.session import Mode, Session, SessionStore
from hearth.indexing.pipeline import Indexer
from hearth.llm.scripted_provider import ScriptedProvider, ScriptedResponse
from hearth.llm.types import ChatChunk
from hearth.retrieval.engine import RetrievalEngine
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository
from hearth.storage.vector_index import NumpyVectorIndex

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "repos"
CITATION = re.compile(r"([\w./-]+\.\w+):(\d+)-(\d+)")


@pytest.fixture
def indexed(tmp_path: Path):
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    Indexer(root=FIXTURES / "py_small", repository=IndexRepository(connection)).run()
    return connection


@pytest.fixture
def engine(indexed) -> RetrievalEngine:
    return RetrievalEngine(
        connection=indexed, vector_index=NumpyVectorIndex(indexed, model_id="fake", dims=8)
    )


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    connection = connect(tmp_path / "state.db")
    migrate(connection, database="state")
    return SessionStore(StateRepository(connection))


def make_session(**overrides) -> Session:
    defaults = {
        "id": "s_test",
        "workspace": FIXTURES / "py_small",
        "model": "scripted",
        "num_ctx": 12_288,
    }
    defaults.update(overrides)
    return Session(**defaults)  # type: ignore[arg-type]


async def run_turn(provider, engine, session, text):
    bus = EventBus()
    events: list[object] = []
    bus.subscribe(events.append)
    runner = ChatRunner(provider=provider, bus=bus, engine=engine)
    result = await runner.run_turn(session, text)
    return result, events


# ------------------------------------------------------------------- a turn


async def test_turn_produces_an_answer(engine: RetrievalEngine) -> None:
    provider = ScriptedProvider([ScriptedResponse("Totals are rounded half-up.")])

    result, _ = await run_turn(provider, engine, make_session(), "how are totals rounded?")

    assert result.ok
    assert result.answer == "Totals are rounded half-up."


async def test_event_order(engine: RetrievalEngine) -> None:
    """Frontends depend on this sequence; a test frontend asserts on it."""
    provider = ScriptedProvider([ScriptedResponse("answer")])

    _, events = await run_turn(provider, engine, make_session(), "question")
    kinds = [type(e).__name__ for e in events]

    assert kinds[0] == "TurnStarted"
    assert kinds[-1] == "TurnFinished"
    assert "RetrievalPerformed" in kinds
    assert "TextDelta" in kinds
    assert kinds.index("RetrievalPerformed") < kinds.index("TextDelta")


async def test_text_is_streamed_not_delivered_whole(engine: RetrievalEngine) -> None:
    provider = ScriptedProvider([ScriptedResponse("a fairly long answer here", chunk_size=4)])

    _, events = await run_turn(provider, engine, make_session(), "q")
    deltas = [e for e in events if isinstance(e, TextDelta)]

    assert len(deltas) > 1
    assert "".join(d.text for d in deltas) == "a fairly long answer here"


async def test_history_accumulates(engine: RetrievalEngine) -> None:
    provider = ScriptedProvider([ScriptedResponse("one"), ScriptedResponse("two")])
    session = make_session()

    await run_turn(provider, engine, session, "first")
    await run_turn(provider, engine, session, "second")

    assert [m.role for m in session.history] == ["user", "assistant", "user", "assistant"]
    assert session.turn_count == 2


async def test_stats_are_reported(engine: RetrievalEngine) -> None:
    provider = ScriptedProvider([ScriptedResponse("answer")])

    _, events = await run_turn(provider, engine, make_session(), "q")
    stats = next(e for e in events if isinstance(e, ContextStats))

    assert stats.used > 0
    assert stats.budget > stats.used


# ---------------------------------------------------------------- citations


async def test_context_offers_resolvable_citations(engine: RetrievalEngine) -> None:
    """M3 acceptance: citations must point at ranges that actually exist.

    Checks the *offer* rather than the model's output: a scripted model cannot be made to
    cite correctly, but the context it is given must be citable.
    """
    provider = ScriptedProvider([ScriptedResponse("see the code")])
    session = make_session()
    bus = EventBus()
    events: list[object] = []
    bus.subscribe(events.append)

    runner = ChatRunner(provider=provider, bus=bus, engine=engine)
    await runner.run_turn(session, "where is TokenBucket?")

    retrieval = next(e for e in events if isinstance(e, RetrievalPerformed))
    assert retrieval.sources

    for source in retrieval.sources:
        path = FIXTURES / "py_small" / source.path
        assert path.is_file(), f"cited {source.path} does not exist"

        lines = path.read_text(encoding="utf-8").splitlines()
        assert 1 <= source.start_line <= len(lines)
        assert source.start_line <= source.end_line <= len(lines)


async def test_citation_format_is_parseable(engine: RetrievalEngine) -> None:
    """The header format the model is asked to echo must be machine-checkable."""
    provider = ScriptedProvider([ScriptedResponse("ok")])
    session = make_session()

    result, _ = await run_turn(provider, engine, session, "where is TokenBucket?")

    assert result.citations
    for citation in result.citations:
        assert CITATION.fullmatch(citation), f"unparseable citation {citation!r}"


async def test_context_block_reaches_the_model(engine: RetrievalEngine) -> None:
    provider = ScriptedProvider([ScriptedResponse("ok")])

    await run_turn(provider, engine, make_session(), "where is TokenBucket?")
    sent = provider.requests[0]

    assert 'trust="untrusted-data"' in sent.messages[-1].content
    assert "TokenBucket" in sent.messages[-1].content


async def test_num_ctx_is_always_sent(engine: RetrievalEngine) -> None:
    provider = ScriptedProvider([ScriptedResponse("ok")])

    await run_turn(provider, engine, make_session(num_ctx=8_192), "q")

    assert provider.requests[0].num_ctx == 8_192


async def test_prefix_is_stable_across_turns(engine: RetrievalEngine) -> None:
    """The 28x prefill difference measured on the reference machine rides on this."""
    provider = ScriptedProvider([ScriptedResponse("one"), ScriptedResponse("two")])
    session = make_session()

    await run_turn(provider, engine, session, "first question")
    await run_turn(provider, engine, session, "second question")

    first, second = provider.requests
    assert first.messages[0].content == second.messages[0].content
    assert first.messages[0].role == "system"


# --------------------------------------------------------------- degradation


async def test_runs_without_retrieval() -> None:
    """An unindexed repository still answers, just without context."""
    provider = ScriptedProvider([ScriptedResponse("no context available")])

    result, events = await run_turn(provider, None, make_session(), "q")

    assert result.ok
    assert not any(isinstance(e, RetrievalPerformed) for e in events)


async def test_partial_embeddings_are_announced(engine: RetrievalEngine) -> None:
    """The user should know dense results are incomplete rather than silently worse."""
    provider = ScriptedProvider([ScriptedResponse("ok")])

    _, events = await run_turn(provider, engine, make_session(), "q")
    notices = [e for e in events if isinstance(e, Notice)]

    assert any("Embeddings are" in n.message for n in notices)


async def test_truncation_is_surfaced(engine: RetrievalEngine) -> None:
    """Silent truncation is the failure that removes the system prompt unnoticed."""
    provider = ScriptedProvider([ScriptedResponse("ok")])
    session = make_session()
    bus = EventBus()
    events: list[object] = []
    bus.subscribe(events.append)

    runner = ChatRunner(provider=provider, bus=bus, engine=engine)
    await runner.run_turn(session, "q")

    # ScriptedProvider reports a plausible count, so no warning should fire here.
    warnings = [e for e in events if isinstance(e, Notice) and e.level == "warning"]
    assert not any("truncated" in w.message for w in warnings)

    # The detector itself is exercised directly.
    assert session.estimator.looks_truncated(estimated=10_000, actual=2_000)


# -------------------------------------------------------------- cancellation


async def test_cancellation_ends_the_turn_promptly(engine: RetrievalEngine) -> None:
    """M3 acceptance: Ctrl+C must stop a stream within a second."""

    class SlowProvider(ScriptedProvider):
        """Streams forever, so the turn can only end by being cancelled."""

        async def chat_stream(self, request):
            while True:
                await asyncio.sleep(0.02)
                yield ChatChunk(content_delta=".")

    bus = EventBus()
    events: list[object] = []
    bus.subscribe(events.append)
    runner = ChatRunner(provider=SlowProvider(), bus=bus, engine=engine)

    task = asyncio.create_task(runner.run_turn(make_session(), "q"))
    await asyncio.sleep(0.15)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    finished = [e for e in events if isinstance(e, TurnFinished)]
    assert finished and finished[-1].reason == "aborted"


# ------------------------------------------------------------------ session


def test_switching_model_starts_a_new_epoch() -> None:
    """The KV cache belongs to the previously loaded model."""
    session = make_session()
    session.estimator.calibrate(estimated=100, actual=110)
    epoch = session.epoch

    session.switch_model("other-model")

    assert session.epoch == epoch + 1
    assert session.estimator.is_calibrated is False


def test_switching_mode_starts_a_new_epoch() -> None:
    """Tool schemas are part of the cached prefix (docs/system-design.md §9.2 rule 4)."""
    session = make_session()
    epoch = session.epoch

    session.switch_mode(Mode.AGENT)

    assert session.epoch == epoch + 1


def test_clear_resets_history_and_epoch() -> None:
    session = make_session()
    session.add_user("q")
    session.add_assistant("a")

    session.clear()

    assert session.history == []
    assert session.epoch == 1


def test_pins_apply_once() -> None:
    session = make_session()
    session.pin("src/a.py")

    assert session.take_pins() == ["src/a.py"]
    assert session.take_pins() == []


def test_prefill_baseline_detects_reuse() -> None:
    """Cache reuse is inferred from prefill rate, since Ollama reports no hit count."""
    session = make_session()

    # Cold: 750 tokens in 15s = 50 tok/s.
    assert session.observe_prefill(prompt_tokens=750, prefill_ms=15_000) is None
    # Warm: same size in 0.5s = 1500 tok/s.
    cached = session.observe_prefill(prompt_tokens=750, prefill_ms=500)

    assert cached is not None
    assert cached > 700


def test_similar_prefill_rate_means_no_reuse() -> None:
    session = make_session()
    session.observe_prefill(prompt_tokens=750, prefill_ms=15_000)

    assert session.observe_prefill(prompt_tokens=750, prefill_ms=14_000) == 0


# ------------------------------------------------------------- persistence


def test_session_round_trips(store: SessionStore, tmp_path: Path) -> None:
    session = store.create(workspace=tmp_path, model="m", num_ctx=12_288)
    store.save_message(session, session.add_user("first question"))
    store.save_message(session, session.add_assistant("an answer"))

    resumed = store.resume(session.id)

    assert resumed is not None
    assert [m.content for m in resumed.history] == ["first question", "an answer"]
    assert resumed.num_ctx == 12_288


def test_resume_latest_picks_the_newest(store: SessionStore, tmp_path: Path) -> None:
    older = store.create(workspace=tmp_path, model="m", num_ctx=12_288)
    store.save_message(older, older.add_user("old"))
    newer = store.create(workspace=tmp_path, model="m", num_ctx=12_288)
    store.save_message(newer, newer.add_user("new"))

    resumed = store.resume_latest(workspace=tmp_path)

    assert resumed is not None
    assert resumed.id == newer.id


def test_title_comes_from_the_first_question(store: SessionStore, tmp_path: Path) -> None:
    """A list of timestamps is unusable; the opening question is what people remember."""
    session = store.create(workspace=tmp_path, model="m", num_ctx=12_288)

    store.set_title_from(session, "how does invoice finalization work?")
    store.set_title_from(session, "a later question")

    assert session.title == "how does invoice finalization work?"


def test_clear_removes_stored_history(store: SessionStore, tmp_path: Path) -> None:
    session = store.create(workspace=tmp_path, model="m", num_ctx=12_288)
    store.save_message(session, session.add_user("q"))

    store.clear(session)

    resumed = store.resume(session.id)
    assert resumed is not None
    assert resumed.history == []


def test_resuming_an_unknown_session_returns_none(store: SessionStore) -> None:
    assert store.resume("s_nonexistent") is None
