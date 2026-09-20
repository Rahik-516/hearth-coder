"""``/review [--staged]``: findings about a diff, each one checkable (system-design §5.11).

Read-only by construction — this workflow calls no tool and edits nothing. It is one
completion over a diff the code has already gathered, so there is no way for a review to
change the tree it is reviewing.

The interesting part is what happens *after* the model answers. A review is a list of
claims about specific lines, and a small model will make claims about lines that are not
there: a plausible-looking ``billing.py:88`` in a diff that stops at line 60. So the answer
is not trusted as returned. Every ``path:line`` in it is checked against the numbered view
the model was given (``diffs.verify_citations``), and anything that does not check out is
listed under the review rather than left for the reader to discover by looking. The model
is not asked to be right about line numbers; code checks them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hearth.core.context.tokens import TokenEstimator
from hearth.core.runner import ChatRunner
from hearth.core.session import Session
from hearth.git.runner import GitError, run_git
from hearth.prompts import load
from hearth.workflows.diffs import (
    Citation,
    FileDiff,
    parse_diff,
    render,
    unverified_footer,
    verify_citations,
)

#: A review needs more of the diff than a commit message does — it has to see the lines it
#: is criticising — so it gets a larger share of the window. Still bounded: the reply
#: shares the rest.
DIFF_BUDGET_TOKENS = 6_000


@dataclass(frozen=True)
class ReviewPrep:
    prompt: str | None = None
    refusal: str | None = None
    files: list[FileDiff] = field(default_factory=list)
    omitted: tuple[str, ...] = ()
    truncated: bool = False


@dataclass
class ReviewOutcome:
    text: str = ""
    refusal: str | None = None
    citations: list[Citation] = field(default_factory=list)
    #: Files the reviewer did not see. Surfaced so "no findings" is not read as "all clear".
    omitted: tuple[str, ...] = ()

    @property
    def unverified(self) -> list[Citation]:
        return [citation for citation in self.citations if not citation.verified]


def prepare_review(root: Path, *, staged: bool, estimator: TokenEstimator) -> ReviewPrep:
    """Gather the diff to review. ``staged`` selects the index; otherwise it is the whole
    working tree against ``HEAD``, so a change that is partly staged is reviewed whole."""
    args = ["-c", "core.quotepath=off", "diff", "--no-ext-diff", "--no-textconv"]
    args += ["--staged"] if staged else ["HEAD"]

    try:
        result = run_git(root, args)
    except GitError as exc:
        return ReviewPrep(refusal=f"could not read the diff: {exc}")
    if not result.ok:
        return ReviewPrep(refusal=f"git could not produce a diff: {result.stderr}")

    files = parse_diff(result.text())
    if not files:
        where = "staged" if staged else "uncommitted"
        hint = "" if staged else (" Untracked files are not included; `git add -N <file>` "
                                  "makes one reviewable.")
        return ReviewPrep(refusal=f"there are no {where} changes to review." + hint)

    rendered = render(files, budget_tokens=DIFF_BUDGET_TOKENS, estimator=estimator, numbered=True)

    note = ""
    if rendered.omitted or rendered.truncated:
        parts = []
        if rendered.omitted:
            parts.append("not shown: " + ", ".join(rendered.omitted))
        if rendered.truncated:
            parts.append("one file was cut to fit")
        note = (
            "Note: this diff is incomplete (" + "; ".join(parts) + "). Say which parts you "
            "could not review rather than implying they are fine.\n"
        )

    prompt = load("workflows/review").replace("{{omitted}}", note).replace("{{diff}}", rendered.text)
    return ReviewPrep(
        prompt=prompt,
        files=files,
        omitted=tuple(rendered.omitted),
        truncated=rendered.truncated,
    )


async def run_review(
    *,
    runner: ChatRunner,
    session: Session,
    root: Path,
    staged: bool = False,
) -> ReviewOutcome:
    """Review the diff and verify the citations in the answer."""
    prep = prepare_review(root, staged=staged, estimator=session.estimator)
    if prep.prompt is None:
        return ReviewOutcome(refusal=prep.refusal)

    answer = (await runner.complete(session, prep.prompt, num_predict=1200)).strip()
    citations = verify_citations(answer, prep.files)

    text = answer + unverified_footer(citations)
    return ReviewOutcome(text=text, citations=citations, omitted=prep.omitted)
