"""Structured plans: what ``/plan`` produces and ``/execute`` consumes.

docs/system-design.md §8.3. Plan-then-execute is the recommended shape for multi-file work
on a local model, and the reason is not ceremony. A 4B model asked to "refactor the billing
module" will start editing within one step and discover the shape of the problem by
changing things. Asking it to *say what it intends first*, in read-only mode, produces a
cheap artefact a human can correct before any file is touched — and correcting a plan costs
a sentence, while correcting six applied edits costs an undo and a re-run.

The plan is a **JSON schema the model fills in**, not prose it writes. Prose reads well and
cannot be executed against: `/execute` needs the file list to scope edit grants, and the
steps to seed todos. A paragraph mentioning `invoice_service.py` in passing is not a file
list, and parsing one out of prose is guesswork the user pays for.

**The file list is a security boundary**, which is why it is normalised here rather than
trusted as written. Plan-scoped grants auto-approve edits to files the plan names (§8.3
step 5), so a plan naming ``../../etc/passwd`` or ``.git/config`` must not be able to
pre-authorise them. Paths are resolved against the workspace and anything that escapes or
is protected is dropped from the grant set — the step survives, it simply asks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hearth.safety.paths import is_protected_write

#: Steps beyond this and the model has written a transcript rather than a plan. A local
#: model producing thirty steps has almost always decomposed into keystrokes.
MAX_STEPS = 12

#: Files one step may name. A step touching more than this is not one step.
MAX_FILES_PER_STEP = 8

ChangeType = Literal["add", "modify", "delete", "rename", "test", "investigate"]


class PlanStep(BaseModel):
    """One unit of work in a plan."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, description="What this step does, in one sentence")
    files: list[str] = Field(
        default_factory=list,
        max_length=MAX_FILES_PER_STEP,
        description="Workspace-relative paths this step touches",
    )
    change_type: ChangeType = Field(
        default="modify", description="add, modify, delete, rename, test or investigate"
    )

    @field_validator("files")
    @classmethod
    def _strip_paths(cls, value: list[str]) -> list[str]:
        return [path.strip() for path in value if path.strip()]


class Plan(BaseModel):
    """A structured plan, as the model returns it and the user reviews it."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, description="What the change accomplishes, in one sentence")
    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_STEPS)
    tests: list[str] = Field(
        default_factory=list, description="Tests to add or run to verify the change"
    )
    risks: list[str] = Field(default_factory=list, description="What could go wrong")
    open_questions: list[str] = Field(
        default_factory=list,
        description="Anything that needs a human answer before this is safe to run",
    )

    @property
    def files(self) -> list[str]:
        """Every path any step names, de-duplicated, in first-mention order."""
        seen: dict[str, None] = {}
        for step in self.steps:
            for path in step.files:
                seen.setdefault(path, None)
        return list(seen)

    def render(self) -> str:
        """The plan as the user reviews it, and as `/execute` pins into the prompt."""
        lines = [f"# {self.goal}", ""]

        for number, step in enumerate(self.steps, start=1):
            files = f"  ({', '.join(step.files)})" if step.files else ""
            lines.append(f"{number}. [{step.change_type}] {step.description}{files}")

        for heading, items in (
            ("Tests", self.tests),
            ("Risks", self.risks),
            ("Open questions", self.open_questions),
        ):
            if items:
                lines.extend(["", f"## {heading}", *(f"- {item}" for item in items)])

        return "\n".join(lines)


def plan_schema() -> dict[str, Any]:
    """The JSON schema sent as ``format``, so the model returns a plan and not prose.

    Generated from the model rather than hand-written, so the schema the model is given
    and the schema that validates its answer cannot drift — the same reasoning as
    ``Tool.schema``.
    """
    schema = Plan.model_json_schema()
    schema.pop("title", None)
    return schema


def parse_plan(raw: str) -> Plan:
    """Parse a model's response into a plan.

    Raises:
        PlanError: when the response is not a usable plan. The message is written for the
            user, because this failure surfaces at a `/plan` prompt rather than to the
            model — there is no retry loop here, and "the model returned something that
            is not a plan" is the whole of what a person can act on.
    """
    text = _strip_fences(raw.strip())
    if not text:
        raise PlanError("the model returned nothing")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanError(f"the model did not return valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise PlanError("the model returned JSON that is not a plan object")

    try:
        return Plan.model_validate(payload)
    except Exception as exc:
        raise PlanError(f"the plan did not match the expected shape: {exc}") from exc


class PlanError(ValueError):
    """A model response that could not be read as a plan."""


@dataclass
class ApprovedPlan:
    """A plan the user accepted, with the grant scope it earns.

    ``grant_files`` is deliberately not ``plan.files``: it is the subset that survived
    path resolution. A plan may legitimately mention ``../sibling-repo/notes.md`` in a
    step description, and the step still runs — it simply asks for that edit rather than
    being pre-approved for it.
    """

    plan: Plan
    plan_id: str
    grant_files: tuple[str, ...] = ()
    #: Paths named by the plan that cannot be granted, with why. Shown at approval, so
    #: "I approved this plan and it still asked" has an answer.
    refused_files: tuple[tuple[str, str], ...] = ()

    @property
    def grant_key(self) -> str:
        """The label the plan's grants were added under, for revoking them together."""
        return f"plan-edits:{self.plan_id}"

    def covers(self, relative_path: str) -> bool:
        return relative_path in self.grant_files

    def edit_grants(self) -> tuple[str, ...]:
        """The session-grant keys this plan pre-authorises.

        Spelled exactly as ``write_fs`` spells them, because that is what makes this a
        reuse of the existing grant mechanism rather than a second one beside it. The
        policy engine needs no knowledge of plans: an approved plan is a user who answered
        "always for this session" to a set of edits in advance, which is precisely what
        these keys already mean.

        The consequence worth being deliberate about: a plan grant is **per file, not per
        edit**. It covers every edit to that file for as long as the plan is live, not the
        one the plan described. Making it narrower would mean predicting the diff, which
        cannot be done before the model has written it — so the honest boundary is the
        file, and the user approves the file list knowing that.
        """
        return tuple(f"edit:{path}" for path in self.grant_files)


def scope_grants(plan: Plan, *, workspace: Path) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Split a plan's files into those a grant may cover and those it may not.

    The grant auto-approves edits, so this is the check that decides what a plan is
    allowed to pre-authorise. Three things are refused:

    * paths that resolve outside the workspace — a plan cannot widen the jail,
    * protected paths (``.git/**``, ``.hearth/**``) — no rule may make those writable, and
      a plan is a rule the *model* proposed,
    * absolute paths, which are not workspace-relative and so mean something the plan
      author did not intend.
    """
    granted: list[str] = []
    refused: list[tuple[str, str]] = []

    for raw in plan.files:
        path = raw.replace("\\", "/").strip()
        if not path:
            continue

        if PurePosixPath(path).is_absolute() or path.startswith("~"):
            refused.append((raw, "not a workspace-relative path"))
            continue

        normalised = PurePosixPath(path)
        if ".." in normalised.parts:
            refused.append((raw, "escapes the workspace"))
            continue

        if is_protected_write(normalised):
            refused.append((raw, "protected path; no plan can make it writable"))
            continue

        resolved = (workspace / normalised).resolve()
        if not _inside(resolved, workspace):
            refused.append((raw, "escapes the workspace"))
            continue

        granted.append(normalised.as_posix())

    return tuple(dict.fromkeys(granted)), tuple(refused)


@dataclass
class PlanStore:
    """The approved plan for a session, if any.

    One at a time: a second approved plan replaces the first rather than adding to it, so
    the grant set is always exactly what the user last looked at. Two live plans would
    mean a file granted by a plan the user has since moved on from.
    """

    approved: ApprovedPlan | None = None
    #: Plans proposed but not yet approved, by id.
    pending: dict[str, Plan] = field(default_factory=dict)

    def propose(self, plan: Plan, *, plan_id: str) -> None:
        self.pending[plan_id] = plan

    def approve(self, plan_id: str, *, workspace: Path) -> ApprovedPlan:
        plan = self.pending.get(plan_id)
        if plan is None:
            raise PlanError(f"no pending plan {plan_id!r} to approve")

        granted, refused = scope_grants(plan, workspace=workspace)
        self.approved = ApprovedPlan(
            plan=plan, plan_id=plan_id, grant_files=granted, refused_files=refused
        )
        self.pending.pop(plan_id, None)
        return self.approved

    def clear(self) -> None:
        self.approved = None
        self.pending.clear()

    def todos(self) -> list[dict[str, str]]:
        """The approved plan's steps as ``todo_write`` arguments.

        The first step starts in progress and the rest pending, which is the shape
        ``todo_write`` validates: exactly one active item.
        """
        if self.approved is None:
            return []
        return [
            {"content": step.description, "status": "in_progress" if index == 0 else "pending"}
            for index, step in enumerate(self.approved.plan.steps)
        ]


def build_execute_message(approved: ApprovedPlan) -> str:
    """The user message `/execute` sends to start the work.

    The plan is restated verbatim rather than referred to. It was produced in plan mode,
    under a different system prompt and possibly before a compaction, so "follow the plan
    above" can point at nothing at all. Restating it costs a few hundred tokens once and
    removes a whole class of silent failure.

    It goes in a *user* message, which is the volatile position in the cached prefix
    (docs/system-design.md §9.2) — putting it any earlier would move it into the bytes the
    KV cache depends on and cost a full re-prefill on every subsequent turn.

    Recording the todos is requested rather than done: ``todo_write`` holds no state, so
    there is nothing for a frontend to seed. The list only exists once the model has said
    it, and asking for that as step one is what puts it on screen while redirecting is
    still cheap.
    """
    lines = [
        "Carry out this plan. It has been reviewed and approved — follow it rather than "
        "re-deciding the approach.",
        "",
        approved.plan.render(),
        "",
        "Start by calling `todo_write` with one item per step above, the first marked "
        "in_progress. Update it as you go.",
    ]

    if approved.plan.open_questions:
        lines.extend(
            [
                "",
                "The open questions were left open on purpose. If one blocks a step, stop "
                "and ask rather than choosing an answer.",
            ]
        )

    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    """Remove a ```json fence, which small models add despite being asked for raw JSON."""
    if not text.startswith("```"):
        return text
    body = text.split("\n", 1)[1] if "\n" in text else ""
    return body.rsplit("```", 1)[0].strip() if "```" in body else body.strip()


def _inside(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return False
    return True
