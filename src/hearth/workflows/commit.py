"""``/commit``: the staged diff in, a reviewed commit out (system-design §5.11, §12.2).

**Code does what code can do.** Gathering the staged diff, deciding it is not empty,
fitting it to a budget, cleaning the message and performing the commit are all
deterministic. The model is asked for exactly one thing — the wording — and nothing it says
can widen what happens: the commit goes through ``git_commit``, so the user sees the staged
diff and the message together, the secret scan runs, the project's hooks run, and a change
to what is staged between approval and execution abandons the commit.

That last point is why this workflow does not stage anything itself. Deciding *what* goes in
a commit is the user's call, and a helper that ran ``git add -A`` first would make the
approval prompt describe a commit the user never assembled.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from hearth.core.context.tokens import TokenEstimator
from hearth.core.runner import ChatRunner
from hearth.core.session import Session
from hearth.git.runner import GitError, run_git
from hearth.prompts import load
from hearth.tools.gateway import ToolGateway
from hearth.workflows.diffs import parse_diff, render

#: What the diff may occupy. Well under the window, because the prompt, the recent log and
#: the reply share it — and a commit message needs the *shape* of a change more than all
#: of it.
DIFF_BUDGET_TOKENS = 4_000

#: Recent subjects shown so the message can match the repository's own style.
LOG_SUBJECTS = 8

#: The conventional subject limit. A soft warning, not a rewrite: truncating a subject the
#: model wrote would change its meaning, and the user can edit it at the approval prompt.
SUBJECT_LIMIT = 72

Status = Literal["committed", "refused", "rejected", "failed"]


@dataclass(frozen=True)
class CommitPrep:
    """What deterministic preparation found."""

    prompt: str | None = None
    #: Why the workflow cannot proceed, worded for the user. Set iff ``prompt`` is None.
    refusal: str | None = None
    files: tuple[str, ...] = ()
    #: Files too large to fit, named so the message is not presented as covering them.
    omitted: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommitMessage:
    text: str
    warnings: tuple[str, ...] = ()


@dataclass
class CommitOutcome:
    status: Status
    message: str = ""
    #: What the tool said — the commit id on success, the reason otherwise.
    detail: str = ""
    warnings: list[str] = field(default_factory=list)


def prepare_commit(root: Path, *, estimator: TokenEstimator) -> CommitPrep:
    """Read the staged changes and build the prompt. Touches nothing."""
    try:
        staged = run_git(
            root,
            ["-c", "core.quotepath=off", "diff", "--staged", "--no-ext-diff", "--no-textconv"],
        )
    except GitError as exc:
        return CommitPrep(refusal=f"could not read the staged changes: {exc}")

    if not staged.ok:
        return CommitPrep(refusal=f"git could not read the staged changes: {staged.stderr}")

    files = parse_diff(staged.text())
    if not files:
        return CommitPrep(
            refusal=(
                "nothing is staged. Stage what you want committed (`git add <files>`, or ask "
                "the agent to use git_add), then run /commit again."
            )
        )

    rendered = render(files, budget_tokens=DIFF_BUDGET_TOKENS, estimator=estimator, numbered=False)

    omitted_note = ""
    if rendered.omitted or rendered.truncated:
        parts = []
        if rendered.omitted:
            parts.append("not shown: " + ", ".join(rendered.omitted))
        if rendered.truncated:
            parts.append("one file was cut to fit")
        omitted_note = (
            "Note: the diff below is incomplete (" + "; ".join(parts) + "). Describe what you "
            "can see and do not guess at the rest.\n"
        )

    prompt = (
        load("workflows/commit")
        .replace("{{omitted}}", omitted_note)
        .replace("{{log}}", _recent_subjects(root))
        .replace("{{diff}}", rendered.text)
    )
    return CommitPrep(
        prompt=prompt,
        files=tuple(file.path for file in files),
        omitted=tuple(rendered.omitted),
    )


def clean_commit_message(raw: str) -> CommitMessage:
    """Turn a model's reply into a commit message, or say why it cannot be one.

    Small models wrap the message in a fence, put it in quotes, or lead with "Here is the
    commit message:". Each is stripped, because a commit whose subject is
    ```` ```text ```` is a commit somebody amends later.
    """
    text = raw.strip()

    fenced = re.match(r"^```[\w-]*\n(?P<body>.*?)\n?```\s*$", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group("body").strip()

    text = re.sub(r"^(?:here(?:'s| is)[^\n:]*:|commit message:)\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()

    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()

    if not text:
        return CommitMessage("", ("the model returned an empty message",))

    lines = [line.rstrip() for line in text.splitlines()]
    subject = lines[0]
    warnings: list[str] = []

    if len(subject) > SUBJECT_LIMIT:
        warnings.append(f"the subject is {len(subject)} characters (conventional limit {SUBJECT_LIMIT})")

    # A body must be separated from the subject by a blank line or git folds it into it.
    if len(lines) > 1 and lines[1].strip():
        lines.insert(1, "")

    return CommitMessage("\n".join(lines).strip() + "\n", tuple(warnings))


async def run_commit(
    *,
    runner: ChatRunner,
    session: Session,
    gateway: ToolGateway,
    root: Path,
) -> CommitOutcome:
    """Prepare, ask the model for wording, then commit through the gateway.

    The commit is a tool call like any other: it needs the session to be in a mode that
    permits ``git_commit``, and it asks the user, showing the staged diff beside the
    message. Declining there is an ordinary outcome, reported as ``rejected``.
    """
    prep = prepare_commit(root, estimator=session.estimator)
    if prep.prompt is None:
        return CommitOutcome("refused", detail=prep.refusal or "nothing to commit")

    raw = await runner.complete(session, prep.prompt, num_predict=400)
    message = clean_commit_message(raw)
    if not message.text:
        return CommitOutcome("failed", detail="; ".join(message.warnings), warnings=list(message.warnings))

    result = await gateway.call(
        "git_commit",
        {"message": message.text},
        call_id=f"wf-commit-{uuid.uuid4().hex[:8]}",
    )

    if result.ok:
        return CommitOutcome(
            "committed", message.text, result.content, list(message.warnings)
        )

    status: Status = "rejected" if result.error is not None and result.error.value == "rejected" else "failed"
    return CommitOutcome(status, message.text, result.content, list(message.warnings))


def _recent_subjects(root: Path) -> str:
    """The last few commit subjects, or a placeholder when there is no history yet."""
    try:
        result = run_git(root, ["log", f"-n{LOG_SUBJECTS}", "--format=%s"])
    except GitError:
        return "(no history available)"
    subjects = result.text().strip() if result.ok else ""
    return subjects or "(no commits yet)"
