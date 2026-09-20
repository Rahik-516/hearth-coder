"""``/refactor <symbol> <goal>``: analyse, plan, edit, verify (system-design §5.11).

This workflow is mostly **composition**. Everything a refactor needs already exists as a
piece: the index knows where a symbol is defined and used, `/plan` produces a reviewed plan,
`/execute` carries it out under approval, and `run_tests` checks the result. What the
workflow adds is the two things that make those pieces a refactor rather than a chat:

* **Before**: the facts. The model is not asked to find the callers of `compute_tax` — that
  is a database query, and a 4B model doing it by grep misses the ones in files it never
  opened. The plan is written with the known references already in front of it.
* **After**: the checks. A refactor's failure mode is not the edit it made; it is the caller
  it did not touch. So the workflow snapshots every file that references the symbol, and
  afterwards reports which were modified and which were not, and runs the test suite itself.
  Neither claim comes from the model.

The reference list is presented as what it is — by simple name, so namesakes are included
and dynamic uses are missed — rather than as a call graph. A prompt that overstated it would
have the model treating "no references found" as "safe to delete".
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from hearth.storage.index_repo import IndexRepository
from hearth.tools.gateway import ToolGateway
from hearth.workflows.targets import ResolvedTarget, TargetError, resolve_target

#: References fetched from the index. Bounded because a name like `get` has thousands, and
#: past this the list stops being information and starts being noise in a small window.
MAX_REFERENCES = 200

#: Files listed in the task text. The rest are counted, so the model knows the list is a
#: sample, without spending the window on it.
MAX_FILES_LISTED = 30


@dataclass(frozen=True)
class RefactorPrep:
    """What deterministic preparation found."""

    target: ResolvedTarget | None = None
    goal: str = ""
    #: The text handed to `/plan`. None when the workflow cannot proceed.
    task: str | None = None
    #: Workspace-relative path -> lines on which the symbol is referenced.
    references: dict[str, list[int]] = field(default_factory=dict)
    refusal: str | None = None


@dataclass(frozen=True)
class ImpactReport:
    """Which of the files that reference a symbol a refactor actually touched."""

    modified: tuple[str, ...] = ()
    #: Referencing files left byte-identical. Not necessarily wrong — a rename that keeps
    #: the old name as an alias needs no caller changes — but the one list a reviewer
    #: should read, because a missed caller is this workflow's characteristic failure.
    untouched: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return len(self.modified) + len(self.untouched)


@dataclass(frozen=True)
class Verification:
    """The result of running the test suite after the refactor."""

    #: None when the run did not happen (declined, or the tool refused).
    passed: bool | None
    summary: str


def split_arguments(argument: str) -> tuple[str, str] | None:
    """``"compute_tax use Decimal throughout"`` -> ``("compute_tax", "use Decimal ...")``.

    None when either half is missing: a refactor without a goal has nothing to plan, and a
    goal without a target has nowhere to start.
    """
    parts = argument.strip().split(None, 1)
    if len(parts) < 2 or not parts[1].strip():
        return None
    return parts[0], parts[1].strip()


def prepare_refactor(
    root: Path, argument: str, *, repository: IndexRepository | None
) -> RefactorPrep:
    """Resolve the target, look up its references, and write the task for `/plan`."""
    split = split_arguments(argument)
    if split is None:
        return RefactorPrep(refusal="usage: /refactor <symbol or path::symbol> <what to change>")
    target_text, goal = split

    target = resolve_target(root, target_text, repository=repository)
    if isinstance(target, TargetError):
        return RefactorPrep(goal=goal, refusal=target.message)
    if target.symbol is None:
        return RefactorPrep(
            goal=goal,
            refusal=(
                f"{target.path} is a file, and /refactor works on a symbol. Name one "
                f"(`{target.path}::<symbol>`), or use /plan for a change to the whole file."
            ),
        )

    references = _references_by_file(target, repository)
    return RefactorPrep(
        target=target,
        goal=goal,
        task=build_task(target, goal, references),
        references=references,
    )


def build_task(
    target: ResolvedTarget, goal: str, references: dict[str, list[int]]
) -> str:
    """The task text `/plan` receives, with the facts the index already holds."""
    lines = [
        f"Refactor `{target.symbol}` — {target.describe()}.",
        "",
        f"Goal: {goal}",
        "",
    ]

    if references:
        lines.append(
            "Known references, from the index. They are matched by name, so they may include "
            "an unrelated symbol with the same name and will miss dynamic uses — treat the "
            "list as a starting point and read each file before deciding:"
        )
        for path in list(references)[:MAX_FILES_LISTED]:
            shown = ", ".join(str(line) for line in references[path][:8])
            more = len(references[path]) - 8
            lines.append(f"- {path}: line {shown}" + (f" (+{more} more)" if more > 0 else ""))
        hidden = len(references) - MAX_FILES_LISTED
        if hidden > 0:
            lines.append(f"- … and {hidden} more file(s)")
    else:
        lines.append(
            "The index found no references to it outside its definition. That may be true, "
            "or it may be used dynamically or from outside this repository; check before "
            "assuming it is safe to change."
        )

    lines.extend(
        [
            "",
            "Keep behaviour the same unless the goal says otherwise. Update every caller the "
            "change affects, name the tests that cover it, and change nothing unrelated.",
        ]
    )
    return "\n".join(lines)


def snapshot_files(root: Path, paths: list[str]) -> dict[str, str]:
    """Content hashes of the given files, for telling afterwards which were touched."""
    hashes: dict[str, str] = {}
    for relative in paths:
        try:
            hashes[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        except OSError:
            hashes[relative] = ""
    return hashes


def impact_report(before: dict[str, str], after: dict[str, str]) -> ImpactReport:
    """Split the snapshotted files into those a refactor changed and those it did not."""
    modified = sorted(path for path, digest in after.items() if before.get(path) != digest)
    untouched = sorted(path for path in after if path not in set(modified))
    return ImpactReport(modified=tuple(modified), untouched=tuple(untouched))


async def verify_with_tests(gateway: ToolGateway) -> Verification:
    """Run the whole suite through the gateway and report what happened.

    The full suite rather than a target: a refactor's damage is in the files nobody named,
    and narrowing to the definition's own tests would check the part most likely to be fine.
    Goes through the gateway like any command — the user sees and approves it.
    """
    result = await gateway.call("run_tests", {}, call_id=f"wf-refactor-{uuid.uuid4().hex[:8]}")

    if result.error is not None and result.error.value in ("denied", "rejected"):
        return Verification(None, f"the test run was {result.error.value}.")

    headline = result.content.strip().split("\n\n", 1)[0] if result.content.strip() else "(no output)"
    if result.metadata.get("tests_ok") is True:
        return Verification(True, headline)
    return Verification(False, headline)


def _references_by_file(
    target: ResolvedTarget, repository: IndexRepository | None
) -> dict[str, list[int]]:
    if repository is None or target.symbol is None:
        return {}

    grouped: dict[str, list[int]] = {}
    for row in repository.find_references(target.symbol, limit=MAX_REFERENCES):
        path, line = str(row["path"]), int(row["line"])  # type: ignore[call-overload]
        # The definition's own lines are not callers; listing them would have the model
        # "update" the function it is being asked to change, as though it were a use.
        if path == target.path and target.start_line and target.start_line <= line <= (
            target.end_line or target.start_line
        ):
            continue
        grouped.setdefault(path, []).append(line)
    return grouped
