"""The security suite — M5's first acceptance criterion.

From docs/implementation-roadmap.md M5:

* no file is modified without an Allow decision,
* rejected edits leave files byte-identical,
* an edited-args approval is re-evaluated by policy,
* headless mode performs zero writes, and
* project allow-rules are ignored until trusted and again after the config changes.

Every test here goes through the **real gateway** with the **real policy engine**. That is
the point of the file: the unit tests prove each part behaves, and this proves they are
actually wired to each other. A policy engine that denies correctly while the gateway
forgets to consult it would pass every test in `test_policy.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from hearth.config.schema import PermissionRule, PermissionsConfig
from hearth.safety.audit import AuditLog
from hearth.safety.checkpoints import CheckpointStore
from hearth.safety.invariants import privilege_escalation_dirs
from hearth.safety.policy import ConfigView, SessionView
from hearth.safety.rules import compile_rules
from hearth.safety.trust import is_project_trusted, trust_project
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository
from hearth.tools.base import ToolContext
from hearth.tools.channel import ApprovalAsk, ApprovalReply
from hearth.tools.gateway import ToolGateway, make_policy
from hearth.tools.registry import build_default_registry
from hearth.tools.results import ErrorCode
from hearth.util.hashing import content_hash

SOURCE = "def finalize(self):\n    total = compute()\n    return total\n"

pytestmark = pytest.mark.anyio


# ------------------------------------------------------------------- harness


@dataclass
class Recorder:
    """A channel that answers approvals from a script, and remembers what it was asked."""

    replies: list[ApprovalReply | None]
    asks: list[ApprovalAsk]

    def __init__(self, *replies: ApprovalReply | None) -> None:
        self.replies = list(replies)
        self.asks = []

    async def proposed(self, *, call_id: str, tool: str, arguments: dict) -> None: ...

    async def started(self, *, call_id: str, tool: str) -> None: ...

    async def finished(
        self, *, call_id: str, tool: str, ok: bool, summary: str, duration_ms: float
    ) -> None: ...

    async def request_approval(self, ask: ApprovalAsk) -> ApprovalReply | None:
        self.asks.append(ask)
        return self.replies.pop(0) if self.replies else None


@dataclass
class Harness:
    gateway: ToolGateway
    workspace: Path
    context: ToolContext
    channel: Recorder
    checkpoints: CheckpointStore
    audit: AuditLog
    repo: StateRepository
    session_id: str

    def file_bytes(self, name: str = "a.py") -> bytes:
        return (self.workspace / name).read_bytes()

    def mark_read(self, name: str = "a.py") -> None:
        self.context.record_read(name, content_hash(self.file_bytes(name)))

    async def edit(self, *, old: str = "    return total", new: str = "    return 0", **kwargs):
        arguments = {"path": kwargs.pop("path", "a.py"), "old_string": old, "new_string": new}
        return await self.gateway.call("edit_file", arguments, call_id="c1", **kwargs)


def build(
    tmp_path: Path,
    *,
    replies: tuple[ApprovalReply | None, ...] = (),
    session: SessionView | None = None,
    config: ConfigView | None = None,
) -> Harness:
    workspace = tmp_path / "repo"
    workspace.mkdir(parents=True)
    (workspace / "a.py").write_text(SOURCE, encoding="utf-8")

    connection = connect(tmp_path / "state.db")
    migrate(connection, database="state")
    repo = StateRepository(connection)
    checkpoints = CheckpointStore(repo, BlobStore(tmp_path / "blobs"))
    session_id = repo.create_session(workspace=str(workspace)).id

    context = ToolContext(workspace=workspace, checkpoints=checkpoints.bind(session_id))
    channel = Recorder(*replies)
    audit = AuditLog(tmp_path / "audit", fsync=False)

    view = session or SessionView(mode="agent", level="supervised")
    settings = config or ConfigView(protected_dirs=privilege_escalation_dirs())

    gateway = ToolGateway(
        registry=build_default_registry(),
        context=context,
        channel=channel,
        audit=audit,
        policy=make_policy(view, settings),
        on_grant=lambda key: repo.add_grant(session_id, key),
        session_id=session_id,
    )
    return Harness(gateway, workspace, context, channel, checkpoints, audit, repo, session_id)


# --------------------------------------- 1. no write without an Allow decision


async def test_a_denied_edit_does_not_touch_the_file(tmp_path: Path) -> None:
    harness = build(
        tmp_path,
        config=ConfigView(
            protected_dirs=privilege_escalation_dirs(),
            rules=compile_rules(
                PermissionsConfig(deny=[PermissionRule(id="no-src", tool="edit_file", path="*.py")]),
                source="global",
            ),
        ),
    )
    harness.mark_read()
    before = harness.file_bytes()

    result = await harness.edit()

    assert not result.ok
    assert result.error is ErrorCode.DENIED
    assert harness.file_bytes() == before
    assert harness.channel.asks == [], "a denied call must never reach an approval prompt"


async def test_an_edit_in_chat_mode_is_refused(tmp_path: Path) -> None:
    """Mode restriction, through the gateway. `edit_file` is not even exposed in chat,
    but a call that arrives anyway must still be refused rather than run."""
    harness = build(tmp_path, session=SessionView(mode="chat"))
    harness.mark_read()
    before = harness.file_bytes()

    result = await harness.edit()

    assert not result.ok
    assert harness.file_bytes() == before


async def test_no_rule_can_authorise_writing_git_internals(tmp_path: Path) -> None:
    """The invariant, end to end: an allow-everything rule does not help."""
    harness = build(
        tmp_path,
        config=ConfigView(
            protected_dirs=privilege_escalation_dirs(),
            rules=compile_rules(
                PermissionsConfig(allow=[PermissionRule(id="all", tool="*", path="**")]),
                source="global",
            ),
        ),
    )
    (harness.workspace / ".git").mkdir()
    (harness.workspace / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    result = await harness.gateway.call(
        "write_file", {"path": ".git/config", "content": "[core]\n\thooksPath = /tmp/x\n"}, call_id="c1"
    )

    assert not result.ok
    assert result.error is ErrorCode.PATH_REFUSED
    assert (harness.workspace / ".git" / "config").read_text(encoding="utf-8") == "[core]\n"


async def test_a_jail_refusal_is_audited_as_an_invariant_denial(tmp_path: Path) -> None:
    """ "What did policy block?" has to be answerable from the log.

    A path-jail refusal recorded as "the tool could not run" would leave the audit log
    silent about the single most important class of refusal it has.
    """
    harness = build(tmp_path)

    await harness.gateway.call("write_file", {"path": "../escape.py", "content": "x"}, call_id="c1")

    record = harness.audit.read_records()[-1]
    assert record["decision"] == "deny"
    assert record["decided_by"] == "invariant"
    assert record["rule_id"] == "path-jail"


async def test_an_allowed_edit_does_write(tmp_path: Path) -> None:
    """The other half of the criterion: an Allow decision really does let the write land."""
    harness = build(
        tmp_path,
        config=ConfigView(
            protected_dirs=privilege_escalation_dirs(),
            rules=compile_rules(
                PermissionsConfig(allow=[PermissionRule(id="py", tool="edit_file", path="*.py")]),
                source="global",
            ),
        ),
    )
    harness.mark_read()

    result = await harness.edit()

    assert result.ok
    assert "return 0" in harness.file_bytes().decode()
    assert harness.channel.asks == [], "an allow rule should not prompt"


# ------------------------------------------ 2. rejected edits change nothing


async def test_a_rejected_edit_leaves_the_file_byte_identical(tmp_path: Path) -> None:
    harness = build(tmp_path, replies=(ApprovalReply(decision="reject", feedback="not the migrations"),))
    harness.mark_read()
    before = harness.file_bytes()

    result = await harness.edit()

    assert not result.ok
    assert result.error is ErrorCode.REJECTED
    assert harness.file_bytes() == before
    assert "not the migrations" in result.content


async def test_a_rejected_edit_leaves_nothing_to_undo(tmp_path: Path) -> None:
    harness = build(tmp_path, replies=(ApprovalReply(decision="reject"),))
    harness.mark_read()

    await harness.edit()

    assert harness.checkpoints.steps(harness.session_id) == []


async def test_an_unanswered_approval_is_a_denial(tmp_path: Path) -> None:
    """No answer means deny (§1.3). Approvals never time out into approval."""
    harness = build(tmp_path, replies=(None,))
    harness.mark_read()
    before = harness.file_bytes()

    result = await harness.edit()

    assert not result.ok
    assert harness.file_bytes() == before


async def test_aborting_leaves_the_file_alone(tmp_path: Path) -> None:
    harness = build(tmp_path, replies=(ApprovalReply(decision="abort"),))
    harness.mark_read()
    before = harness.file_bytes()

    await harness.edit()

    assert harness.file_bytes() == before


async def test_an_approved_edit_writes_what_the_preview_showed(tmp_path: Path) -> None:
    harness = build(tmp_path, replies=(ApprovalReply(decision="approve"),))
    harness.mark_read()

    result = await harness.edit()

    assert result.ok
    preview = harness.channel.asks[0].preview
    assert "+    return 0" in preview
    assert harness.file_bytes().decode() == SOURCE.replace("    return total", "    return 0")


# ------------------------------------- 3. edited arguments are re-evaluated


async def test_edited_arguments_are_re_evaluated_by_policy(tmp_path: Path) -> None:
    """§5.9. The user edits the call into something a deny rule forbids.

    The edit must go back through prepare *and* policy. Executing the edited arguments
    directly would make the approval prompt a way around the rules — which is exactly
    backwards, since the prompt exists to enforce them.
    """
    harness = build(
        tmp_path,
        replies=(
            ApprovalReply(
                decision="edit",
                edited_arguments={
                    "path": "secrets.py",
                    "old_string": "x",
                    "new_string": "y",
                },
            ),
        ),
        config=ConfigView(
            protected_dirs=privilege_escalation_dirs(),
            rules=compile_rules(
                PermissionsConfig(
                    deny=[PermissionRule(id="no-secrets", tool="edit_file", path="secrets.py")]
                ),
                source="global",
            ),
        ),
    )
    (harness.workspace / "secrets.py").write_text("x = 1\n", encoding="utf-8")
    harness.mark_read()
    harness.mark_read("secrets.py")

    result = await harness.edit()

    assert not result.ok
    assert result.error is ErrorCode.DENIED
    assert (harness.workspace / "secrets.py").read_text(encoding="utf-8") == "x = 1\n"


async def test_edited_arguments_are_re_checked_against_the_jail(tmp_path: Path) -> None:
    """The same guarantee for the invariants rather than the rules."""
    harness = build(
        tmp_path,
        replies=(
            ApprovalReply(
                decision="edit",
                edited_arguments={"path": "../escape.py", "old_string": "x", "new_string": "y"},
            ),
        ),
    )
    harness.mark_read()

    result = await harness.edit()

    assert not result.ok
    assert result.error is ErrorCode.PATH_REFUSED
    assert not (tmp_path / "escape.py").exists()


# ------------------------------------------------- 4. headless writes nothing


async def test_headless_performs_zero_writes(tmp_path: Path) -> None:
    harness = build(tmp_path, session=SessionView(mode="agent", headless=True))
    harness.mark_read()
    before = harness.file_bytes()

    result = await harness.edit()

    assert not result.ok
    assert harness.file_bytes() == before
    assert harness.channel.asks == [], "there is nobody to ask"


async def test_a_headless_denial_names_the_flag_that_would_allow_it(tmp_path: Path) -> None:
    harness = build(tmp_path, session=SessionView(mode="agent", headless=True))
    harness.mark_read()

    result = await harness.edit()

    assert "--allow-edits" in result.content


async def test_headless_with_allow_edits_does_write(tmp_path: Path) -> None:
    """The flag is narrow and explicit, and it works — otherwise it is not a real escape
    hatch and users will reach for something worse."""
    harness = build(
        tmp_path,
        session=SessionView(mode="agent", headless=True),
        config=ConfigView(protected_dirs=privilege_escalation_dirs(), headless_allow_edits=True),
    )
    harness.mark_read()

    result = await harness.edit()

    assert result.ok
    assert "return 0" in harness.file_bytes().decode()


async def test_headless_destructive_is_denied_even_with_allow_edits(tmp_path: Path) -> None:
    """§14: DESTRUCTIVE is always denied headless, whatever was granted."""
    harness = build(
        tmp_path,
        session=SessionView(mode="agent", headless=True),
        config=ConfigView(protected_dirs=privilege_escalation_dirs(), headless_allow_edits=True),
    )
    big = harness.workspace / "big.py"
    big.write_text("".join(f"line_{n} = {n}\n" for n in range(300)), encoding="utf-8")
    harness.mark_read("big.py")

    result = await harness.gateway.call("write_file", {"path": "big.py", "content": "x = 1\n"}, call_id="c1")

    assert not result.ok
    assert big.read_text(encoding="utf-8").count("\n") == 300


# ----------------------------------------------------- 5. project trust gating


async def test_a_project_allow_rule_does_nothing_until_trusted(tmp_path: Path) -> None:
    """T4, end to end: a cloned repo ships a config that allowlists editing everything."""
    project_rules = compile_rules(
        PermissionsConfig(allow=[PermissionRule(id="anything", tool="edit_file", path="**")]),
        source="project",
    )

    def harness_for(trusted: bool) -> Harness:
        harness = build(
            tmp_path / ("trusted" if trusted else "untrusted"),
            replies=(ApprovalReply(decision="reject"),),
            config=ConfigView(
                protected_dirs=privilege_escalation_dirs(),
                rules=project_rules,
                project_trusted=trusted,
            ),
        )
        harness.mark_read()
        return harness

    untrusted = harness_for(False)
    result = await untrusted.edit()

    assert not result.ok, "an untrusted project cannot grant itself write access"
    assert untrusted.channel.asks, "it falls through to Ask, which the script rejects"

    trusted = harness_for(True)
    allowed = await trusted.edit()

    assert allowed.ok
    assert trusted.channel.asks == [], "once trusted, the allow rule applies without asking"


async def test_trust_is_lost_when_the_config_changes(tmp_path: Path) -> None:
    """The second half of the criterion: "and again after the config changes"."""
    connection = connect(tmp_path / "state.db")
    migrate(connection, database="state")
    repo = StateRepository(connection)

    config_file = tmp_path / ".hearth" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('[[permissions.allow]]\ntool = "edit_file"\n', encoding="utf-8")

    trust_project(repo, config_file)
    assert is_project_trusted(repo, config_file)

    config_file.write_text(
        '[[permissions.allow]]\ntool = "edit_file"\n[[permissions.allow]]\ntool = "write_file"\n',
        encoding="utf-8",
    )

    assert not is_project_trusted(repo, config_file)


# ------------------------------------------------------ grants and auditing


async def test_always_for_session_records_a_grant_and_stops_asking(tmp_path: Path) -> None:
    harness = build(
        tmp_path,
        replies=(ApprovalReply(decision="always_session"),),
    )
    harness.mark_read()

    first = await harness.edit()

    assert first.ok
    assert harness.repo.grants_for(harness.session_id) == frozenset({"edit:a.py"})


async def test_a_checkpoint_is_attributed_to_the_step_that_made_it(tmp_path: Path) -> None:
    """`/undo` and `/rewind <step>` are only meaningful if steps are distinguishable.

    The gateway owns the step number and the tool context is per-session, so the gateway
    has to stamp it on each call. Without that every write in a session records against
    step 0, and both commands degrade into "revert everything at once" — which is exactly
    the behaviour the conflict-detection design exists to avoid.
    """
    harness = build(tmp_path, replies=(ApprovalReply(decision="approve"),))
    harness.mark_read()

    await harness.gateway.call(
        "edit_file",
        {"path": "a.py", "old_string": "    return total", "new_string": "    return 0"},
        call_id="c1",
        step=4,
    )

    assert [step.step for step in harness.checkpoints.steps(harness.session_id)] == [4]


async def test_two_writes_in_one_turn_are_separate_steps(tmp_path: Path) -> None:
    harness = build(
        tmp_path,
        replies=(ApprovalReply(decision="approve"), ApprovalReply(decision="approve")),
    )
    (harness.workspace / "b.py").write_text("x = 1\n", encoding="utf-8")
    harness.mark_read()
    harness.mark_read("b.py")

    await harness.gateway.call(
        "edit_file",
        {"path": "a.py", "old_string": "    return total", "new_string": "    return 0"},
        call_id="c1",
        step=1,
    )
    await harness.gateway.call(
        "edit_file",
        {"path": "b.py", "old_string": "x = 1", "new_string": "x = 2"},
        call_id="c2",
        step=2,
    )

    steps = harness.checkpoints.steps(harness.session_id)

    assert [step.step for step in steps] == [2, 1]
    assert harness.checkpoints.latest_step(harness.session_id) == 2


async def test_every_write_call_is_audited_with_its_decision(tmp_path: Path) -> None:
    harness = build(tmp_path, replies=(ApprovalReply(decision="approve"),))
    harness.mark_read()

    await harness.edit()

    record = harness.audit.read_records()[-1]
    assert record["tool"] == "edit_file"
    assert record["risk"] == "WRITE"
    assert record["decision"] == "allow"


async def test_the_audit_log_cites_the_rule_that_denied(tmp_path: Path) -> None:
    harness = build(
        tmp_path,
        config=ConfigView(
            protected_dirs=privilege_escalation_dirs(),
            rules=compile_rules(
                PermissionsConfig(deny=[PermissionRule(id="house-rule", tool="edit_file", path="**")]),
                source="global",
            ),
        ),
    )
    harness.mark_read()

    await harness.edit()

    record = harness.audit.read_records()[-1]
    assert record["rule_id"] == "house-rule"
    assert record["decided_by"] == "rule"
