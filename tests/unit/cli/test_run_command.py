"""``hearth run`` — the headless contract, tested at the gateway it assembles.

The M6 acceptance criterion is that **headless fails closed**: it does not skip approvals,
it refuses everything that would have needed one and names the flag that would have
permitted it. That is worth testing at the seam where the CLI builds the gateway rather
than only against the pure policy engine, because the bug this prevents is a wiring
mistake — a flag read from the wrong variable, or a `SessionView` built without
``headless=True`` — which a policy-level test cannot see.

No model is involved: these exercise the assembled gateway directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hearth.cli.chat_commands import _build_gateway
from hearth.config.loader import LoadedConfig
from hearth.config.schema import HearthConfig
from hearth.core.bus import EventBus
from hearth.core.events import ApprovalRequested, ApprovalResponse
from hearth.core.session import Mode, Session
from hearth.safety.checkpoints import CheckpointStore
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository
from hearth.tools.results import ErrorCode


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("value = 1\n", encoding="utf-8")
    return root


@pytest.fixture
def checkpoints(tmp_path: Path) -> CheckpointStore:
    connection = connect(tmp_path / "state.db")
    migrate(connection, database="state")
    return CheckpointStore(StateRepository(connection), BlobStore(tmp_path / "blobs"))


@pytest.fixture
def session(workspace: Path, checkpoints: CheckpointStore) -> Session:
    created = checkpoints.repo.create_session(workspace=str(workspace))
    return Session(
        id=created.id,
        workspace=workspace,
        model="qwen3.5:4b",
        num_ctx=8192,
        mode=Mode.AGENT,
    )


def build(
    workspace: Path,
    session: Session,
    checkpoints: CheckpointStore,
    **flags: bool,
):
    return _build_gateway(
        workspace,
        LoadedConfig(config=HearthConfig()),
        session=session,
        bus=EventBus(),
        checkpoints=checkpoints,
        index_connection=None,
        engine=None,
        headless=flags.get("headless", True),
        allow_edits=flags.get("allow_edits", False),
        allow_tests=flags.get("allow_tests", False),
        allow_commit=flags.get("allow_commit", False),
    )


async def test_headless_denies_an_edit_and_names_the_flag(
    workspace: Path, session: Session, checkpoints: CheckpointStore
) -> None:
    gateway = build(workspace, session, checkpoints)

    result = await gateway.call(
        "write_file", {"path": "new.py", "content": "x = 1\n"}, call_id="c1"
    )

    assert result.error is ErrorCode.DENIED
    assert "--allow-edits" in result.content
    assert not (workspace / "new.py").exists()


async def test_headless_denies_a_command_and_names_the_flag(
    workspace: Path, session: Session, checkpoints: CheckpointStore
) -> None:
    gateway = build(workspace, session, checkpoints)

    result = await gateway.call("run_command", {"command": "echo one"}, call_id="c1")

    assert result.error is ErrorCode.DENIED
    assert "--allow-tests" in result.content


async def test_allow_tests_does_not_also_allow_edits(
    workspace: Path, session: Session, checkpoints: CheckpointStore
) -> None:
    """Each flag widens exactly one risk class. There is no --allow-all."""
    gateway = build(workspace, session, checkpoints, allow_tests=True)

    command = await gateway.call("run_command", {"command": "echo one"}, call_id="c1")
    edit = await gateway.call("write_file", {"path": "new.py", "content": "x\n"}, call_id="c2")

    assert command.ok
    assert edit.error is ErrorCode.DENIED


async def test_allow_edits_permits_a_write_and_leaves_a_checkpoint(
    workspace: Path, session: Session, checkpoints: CheckpointStore
) -> None:
    """A headless write is still revertable: the run is unattended, not unaccountable."""
    gateway = build(workspace, session, checkpoints, allow_edits=True)

    result = await gateway.call("write_file", {"path": "new.py", "content": "x = 1\n"}, call_id="c1")

    assert result.ok, result.content
    assert (workspace / "new.py").read_text(encoding="utf-8") == "x = 1\n"
    assert checkpoints.steps(session.id), "an unattended write still needs an undo"


async def test_a_hard_denied_command_is_refused_even_with_every_flag(
    workspace: Path, session: Session, checkpoints: CheckpointStore
) -> None:
    """The flags widen permissions; they do not reach the hard invariants.

    Inert payload: `sudo true` would at worst prompt for a password against a closed stdin.
    """
    gateway = build(
        workspace, session, checkpoints, allow_edits=True, allow_tests=True, allow_commit=True
    )

    result = await gateway.call("run_command", {"command": "sudo true"}, call_id="c1")

    assert result.error is ErrorCode.DENIED


async def test_interactive_mode_asks_rather_than_denying(
    workspace: Path, session: Session, checkpoints: CheckpointStore
) -> None:
    """Without --headless the same command reaches a human instead of being refused.

    A responder has to be subscribed for this: ``EventBusChannel.request_approval``
    publishes and then *waits*, so with nothing listening the call blocks rather than
    failing closed. That is the right behaviour for a terminal, where the prompt is always
    subscribed, and it is why the headless path uses its own channel instead of relying on
    the policy engine never asking.
    """
    bus = EventBus()
    gateway = _build_gateway(
        workspace,
        LoadedConfig(config=HearthConfig()),
        session=session,
        bus=bus,
        checkpoints=checkpoints,
        index_connection=None,
        engine=None,
        headless=False,
        allow_edits=False,
        allow_tests=False,
        allow_commit=False,
    )

    asked: list[ApprovalRequested] = []

    def responder(event: object) -> None:
        # Resolved through `resolve_approval`, not by publishing: the waiter is a future
        # keyed by request_id, and a published event would never reach it.
        if isinstance(event, ApprovalRequested):
            asked.append(event)
            bus.resolve_approval(
                ApprovalResponse(request_id=event.request_id, decision="reject")
            )

    bus.subscribe(responder)

    result = await gateway.call("run_command", {"command": "echo one"}, call_id="c1")

    assert asked, "an interactive run must reach a human, not a denial"
    assert result.error is ErrorCode.REJECTED
