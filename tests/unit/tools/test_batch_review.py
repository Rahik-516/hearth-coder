"""Batch review at the gateway — docs/safety-and-tool-use.md §6.4.

A batch answer replaces exactly one thing: the human's yes or no at the approval step. Every
other part of the lifecycle still happens per call. So these tests are organised around what
the substitution must **not** be able to do:

* approve something policy would have denied, or something that needed its own prompt,
* approve a diff other than the one the user was shown,
* be replayed, or outlive the step it was given for,
* turn a channel that cannot answer into consent.

They drive the real gateway with real edits, and a channel that records both the batch
screen and any individual prompt — because "was the user asked once, or five times, and
about what" is the observable behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hearth.config.schema import PermissionRule, PermissionsConfig
from hearth.safety.checkpoints import CheckpointStore
from hearth.safety.policy import ConfigView, SessionView
from hearth.safety.rules import compile_rules
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository
from hearth.tools.base import ToolContext
from hearth.tools.channel import ApprovalAsk, ApprovalReply, BatchItem
from hearth.tools.gateway import ToolGateway, make_policy
from hearth.tools.registry import build_default_registry
from hearth.tools.results import ErrorCode
from hearth.util.hashing import content_hash


class Channel:
    """Records the batch screen and every individual prompt, and answers from a script."""

    def __init__(
        self,
        batch: dict[str, str] | None = None,
        *,
        individual: str = "approve",
        supports_batch: bool = True,
    ) -> None:
        self.batch_answer = batch
        self.individual = individual
        self.batches: list[list[BatchItem]] = []
        self.asks: list[ApprovalAsk] = []
        self.on_batch = None
        if supports_batch:
            self.request_batch_approval = self._request_batch_approval  # type: ignore[method-assign]

    async def proposed(self, **_: object) -> None: ...

    async def started(self, **_: object) -> None: ...

    async def finished(self, **_: object) -> None: ...

    async def request_approval(self, ask: ApprovalAsk) -> ApprovalReply:
        self.asks.append(ask)
        return ApprovalReply(decision=self.individual)

    async def _request_batch_approval(self, items: list[BatchItem]) -> dict[str, str] | None:
        self.batches.append(items)
        if self.on_batch is not None:
            self.on_batch()
        if callable(self.batch_answer):
            return self.batch_answer(items)
        return self.batch_answer


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("a", "b", "c"):
        (root / f"{name}.py").write_text(f"value_{name} = 1\n", encoding="utf-8")
    return root


def build(
    workspace: Path,
    tmp_path: Path,
    channel: Channel,
    *,
    config: ConfigView | None = None,
    grants: frozenset[str] = frozenset(),
) -> tuple[ToolGateway, ToolContext]:
    connection = connect(tmp_path / "state.db")
    migrate(connection, database="state")
    state = StateRepository(connection)
    checkpoints = CheckpointStore(state, BlobStore(tmp_path / "blobs"))
    session_id = state.create_session(workspace=str(workspace)).id
    context = ToolContext(workspace=workspace, checkpoints=checkpoints.bind(session_id))
    gateway = ToolGateway(
        registry=build_default_registry(),
        context=context,
        channel=channel,
        policy=make_policy(
            SessionView(mode="agent", level="supervised", grants=grants), config or ConfigView()
        ),
        session_id=session_id,
    )
    return gateway, context


def read(context: ToolContext, workspace: Path, name: str) -> None:
    context.record_read(name, content_hash((workspace / name).read_bytes()))


def edit_call(name: str, call_id: str, *, new: str = "changed") -> tuple[str, dict, str]:
    var = f"value_{name[0]}"
    return (
        "edit_file",
        {"path": name, "old_string": f"{var} = 1", "new_string": f"{var} = {new!r}"},
        call_id,
    )


async def run_step(gateway: ToolGateway, calls: list[tuple[str, dict, str]]):
    """What the agent loop does for a multi-write step: the pre-pass, then each call."""
    await gateway.review_batch(calls)
    return [await gateway.call(name, args, call_id=call_id) for name, args, call_id in calls]


def text(workspace: Path, name: str) -> str:
    return (workspace / name).read_text(encoding="utf-8")


# ------------------------------------------------------------- the happy path


async def test_several_writes_are_put_to_the_user_once(workspace: Path, tmp_path: Path) -> None:
    channel = Channel({"c1": "approve", "c2": "approve"})
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")
    read(context, workspace, "b.py")

    results = await run_step(gateway, [edit_call("a.py", "c1"), edit_call("b.py", "c2")])

    assert all(result.ok for result in results)
    assert len(channel.batches) == 1
    assert [item.path for item in channel.batches[0]] == ["a.py", "b.py"]
    assert channel.asks == [], "no separate prompt for either file"
    assert "changed" in text(workspace, "a.py") and "changed" in text(workspace, "b.py")


async def test_the_screen_carries_stats_and_the_diff(workspace: Path, tmp_path: Path) -> None:
    channel = Channel({"c1": "approve", "c2": "approve"})
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")
    read(context, workspace, "b.py")

    await run_step(gateway, [edit_call("a.py", "c1"), edit_call("b.py", "c2")])

    item = channel.batches[0][0]
    assert (item.added, item.removed) == (1, 1)
    assert "value_a = 'changed'" in item.preview


async def test_a_per_file_rejection_leaves_that_file_alone(workspace: Path, tmp_path: Path) -> None:
    channel = Channel({"c1": "approve", "c2": "reject"})
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")
    read(context, workspace, "b.py")
    before = (workspace / "b.py").read_bytes()

    first, second = await run_step(gateway, [edit_call("a.py", "c1"), edit_call("b.py", "c2")])

    assert first.ok
    assert second.error is ErrorCode.REJECTED
    assert "batch review" in second.content
    assert (workspace / "b.py").read_bytes() == before


async def test_approved_items_still_checkpoint_individually(
    workspace: Path, tmp_path: Path
) -> None:
    """The substitution replaces the human's answer and nothing else, so each approved
    write is still its own revertible entry."""
    channel = Channel({"c1": "approve", "c2": "approve"})
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")
    read(context, workspace, "b.py")

    await run_step(gateway, [edit_call("a.py", "c1"), edit_call("b.py", "c2")])

    connection = connect(tmp_path / "state.db")
    rows = connection.execute("SELECT path FROM checkpoints ORDER BY id").fetchall()
    assert [row["path"] for row in rows] == ["a.py", "b.py"]


# ------------------------------------------------------ what is never batched


async def test_a_single_write_is_not_a_batch(workspace: Path, tmp_path: Path) -> None:
    channel = Channel({"c1": "approve"})
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")

    (result,) = await run_step(gateway, [edit_call("a.py", "c1")])

    assert result.ok
    assert channel.batches == []
    assert len(channel.asks) == 1, "asked the ordinary way"


async def test_two_writes_to_one_file_are_asked_about_individually(
    workspace: Path, tmp_path: Path
) -> None:
    """The second diff depends on the first having been applied, so it cannot be shown
    correctly in advance."""
    channel = Channel({"c1": "approve", "c2": "approve"})
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")

    calls = [
        edit_call("a.py", "c1"),
        ("edit_file", {"path": "a.py", "old_string": "value_a", "new_string": "renamed"}, "c2"),
    ]
    await gateway.review_batch(calls)

    assert channel.batches == [], "neither was offered for batching"


async def test_a_destructive_write_keeps_its_own_typed_confirmation(
    workspace: Path, tmp_path: Path
) -> None:
    """§6.3: destructive calls need a typed word, which a batch screen cannot collect."""
    (workspace / "big.py").write_text("".join(f"x{n} = {n}\n" for n in range(300)), encoding="utf-8")
    channel = Channel({"c1": "approve", "c2": "approve", "c3": "approve"})
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")
    read(context, workspace, "b.py")
    read(context, workspace, "big.py")

    calls = [
        edit_call("a.py", "c1"),
        edit_call("b.py", "c2"),
        ("write_file", {"path": "big.py", "content": "small = 1\n"}, "c3"),
    ]
    await gateway.review_batch(calls)

    assert [item.call_id for item in channel.batches[0]] == ["c1", "c2"]


async def test_a_denied_write_never_reaches_the_screen_and_is_still_denied(
    workspace: Path, tmp_path: Path
) -> None:
    config = ConfigView(
        rules=compile_rules(
            PermissionsConfig(deny=[PermissionRule(id="no-b", tool="edit_file", path="b.py")]),
            source="global",
        )
    )
    channel = Channel({"c1": "approve", "c3": "approve"})
    gateway, context = build(workspace, tmp_path, channel, config=config)
    for name in ("a.py", "b.py", "c.py"):
        read(context, workspace, name)

    results = await run_step(
        gateway, [edit_call("a.py", "c1"), edit_call("b.py", "c2"), edit_call("c.py", "c3")]
    )

    assert [item.path for item in channel.batches[0]] == ["a.py", "c.py"]
    assert results[1].error is ErrorCode.DENIED
    assert "changed" not in text(workspace, "b.py")


async def test_a_granted_write_runs_without_a_screen(workspace: Path, tmp_path: Path) -> None:
    """Policy already allowed it, so there is nothing to ask about."""
    channel = Channel({"c2": "approve", "c3": "approve"})
    gateway, context = build(workspace, tmp_path, channel, grants=frozenset({"edit:a.py"}))
    for name in ("a.py", "b.py", "c.py"):
        read(context, workspace, name)

    results = await run_step(
        gateway, [edit_call("a.py", "c1"), edit_call("b.py", "c2"), edit_call("c.py", "c3")]
    )

    assert all(result.ok for result in results)
    assert [item.path for item in channel.batches[0]] == ["b.py", "c.py"]


async def test_a_channel_without_batch_support_gets_no_batching(
    workspace: Path, tmp_path: Path
) -> None:
    """The null channel, a headless run, a frontend that predates batches: everything is
    asked about individually, exactly as before."""
    channel = Channel(supports_batch=False)
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")
    read(context, workspace, "b.py")

    results = await run_step(gateway, [edit_call("a.py", "c1"), edit_call("b.py", "c2")])

    assert all(result.ok for result in results)
    assert len(channel.asks) == 2


# ---------------------------------------------------- the substitution's limits


async def test_a_diff_that_moved_since_review_is_asked_about_again(
    workspace: Path, tmp_path: Path
) -> None:
    """An answer binds to the preview it was given for. If the file changes while the
    screen is open, "approved" applied to the new diff would be approving something the
    person never saw — so it is discarded and they are asked about what is now true."""
    channel = Channel({"c1": "approve", "c2": "approve"}, individual="reject")
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")
    read(context, workspace, "b.py")

    def someone_edits_b() -> None:
        (workspace / "b.py").write_text("value_b = 1\n# added by someone else\n", encoding="utf-8")
        context.record_read("b.py", content_hash((workspace / "b.py").read_bytes()))

    channel.on_batch = someone_edits_b

    first, second = await run_step(gateway, [edit_call("a.py", "c1"), edit_call("b.py", "c2")])

    assert first.ok, "a.py was untouched, so its answer still applies"
    assert not second.ok, "b.py's approval did not carry over to a diff that changed"
    assert [ask.call_id for ask in channel.asks] == ["c2"], "and it was asked about individually"
    assert "changed" not in text(workspace, "b.py")


async def test_an_answer_is_used_once(workspace: Path, tmp_path: Path) -> None:
    """Consumed on use, so it cannot be replayed against a second call with the same id."""
    channel = Channel({"c1": "approve", "c2": "approve"}, individual="reject")
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")
    read(context, workspace, "b.py")
    calls = [edit_call("a.py", "c1"), edit_call("b.py", "c2")]
    await run_step(gateway, calls)

    read(context, workspace, "a.py")
    (workspace / "a.py").write_text("value_a = 1\n", encoding="utf-8")
    read(context, workspace, "a.py")
    replay = await gateway.call(*calls[0][:2], call_id="c1")

    assert not replay.ok, "the first answer was spent; this call had to ask, and was refused"
    assert any(ask.call_id == "c1" for ask in channel.asks)


async def test_no_answer_is_never_consent(workspace: Path, tmp_path: Path) -> None:
    """A closed bus or a headless run returns None. Nothing is stored, so each call asks
    for itself and fails closed if there is still nobody to ask."""
    channel = Channel(None, individual="reject")
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")
    read(context, workspace, "b.py")
    before = (workspace / "a.py").read_bytes(), (workspace / "b.py").read_bytes()

    results = await run_step(gateway, [edit_call("a.py", "c1"), edit_call("b.py", "c2")])

    assert not any(result.ok for result in results)
    assert ((workspace / "a.py").read_bytes(), (workspace / "b.py").read_bytes()) == before


async def test_an_item_the_screen_did_not_answer_is_rejected(
    workspace: Path, tmp_path: Path
) -> None:
    """Absence never approves."""
    channel = Channel({"c1": "approve"})  # says nothing about c2
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")
    read(context, workspace, "b.py")

    first, second = await run_step(gateway, [edit_call("a.py", "c1"), edit_call("b.py", "c2")])

    assert first.ok
    assert second.error is ErrorCode.REJECTED


async def test_a_previewing_pass_changes_nothing(workspace: Path, tmp_path: Path) -> None:
    """`prepare()` must not mutate, and the pre-pass is nothing but `prepare()` calls."""
    channel = Channel({"c1": "reject", "c2": "reject"})
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")
    read(context, workspace, "b.py")
    before = {name: (workspace / name).read_bytes() for name in ("a.py", "b.py")}

    await gateway.review_batch([edit_call("a.py", "c1"), edit_call("b.py", "c2")])

    assert {name: (workspace / name).read_bytes() for name in ("a.py", "b.py")} == before


async def test_a_path_outside_the_jail_is_skipped_by_the_preview_not_reported_by_it(
    workspace: Path, tmp_path: Path
) -> None:
    """The real call hits the same refusal and audits it properly; the preview pass is not
    the place to record or report it."""
    channel = Channel({"c1": "approve", "c3": "approve"})
    gateway, context = build(workspace, tmp_path, channel)
    read(context, workspace, "a.py")
    read(context, workspace, "c.py")

    calls = [
        edit_call("a.py", "c1"),
        ("write_file", {"path": "../escape.py", "content": "x"}, "c2"),
        edit_call("c.py", "c3"),
    ]
    results = await run_step(gateway, calls)

    assert [item.path for item in channel.batches[0]] == ["a.py", "c.py"]
    assert results[1].error is ErrorCode.PATH_REFUSED
