"""`/review` — I3.

The workflow is read-only and one model call, so what is worth testing is the two things
around that call: **what the model is shown** and **what happens to what it says**.

What it is shown must be numbered with new-side line numbers (so it can copy rather than
compute), and must say when it has not seen everything. What it says is checked in code:
a citation of a line that was never shown is reported under the review, not passed along.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hearth.core.bus import EventBus
from hearth.core.context.tokens import TokenEstimator
from hearth.core.runner import ChatRunner
from hearth.core.session import Mode, Session
from hearth.llm.scripted_provider import ScriptedProvider, ScriptedResponse
from hearth.workflows.review import prepare_review, run_review


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Hearth Test")
    (root / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "baseline")
    return root


def build(repo: Path, reply: str):
    session = Session(id="s_review", workspace=repo, model="scripted", num_ctx=8192, mode=Mode.CHAT)
    provider = ScriptedProvider([ScriptedResponse(reply)])
    return session, ChatRunner(provider=provider, bus=EventBus()), provider


def edit(repo: Path) -> None:
    (repo / "a.py").write_text("def f():\n    return 2\n\n\ndef g():\n    return 3\n", encoding="utf-8")


# ---------------------------------------------------------------- preparation


def test_no_changes_is_refused(repo: Path) -> None:
    prep = prepare_review(repo, staged=False, estimator=TokenEstimator())

    assert prep.prompt is None
    assert "no uncommitted changes" in (prep.refusal or "")


def test_the_untracked_file_hint_is_given(repo: Path) -> None:
    """`git diff HEAD` cannot see a new file, so "no changes" alone would be misleading."""
    (repo / "new.py").write_text("x = 1\n", encoding="utf-8")

    prep = prepare_review(repo, staged=False, estimator=TokenEstimator())

    assert "Untracked" in (prep.refusal or "")


def test_the_default_is_the_whole_working_tree_against_head(repo: Path) -> None:
    """A change that is partly staged is reviewed whole, not as the two halves."""
    edit(repo)
    git(repo, "add", "a.py")
    (repo / "a.py").write_text("def f():\n    return 2\n# unstaged\n", encoding="utf-8")

    prep = prepare_review(repo, staged=False, estimator=TokenEstimator())

    assert prep.prompt is not None
    assert "unstaged" in prep.prompt


def test_staged_reviews_only_the_index(repo: Path) -> None:
    edit(repo)
    git(repo, "add", "a.py")
    (repo / "a.py").write_text("def f():\n    return 2\n# not staged yet\n", encoding="utf-8")

    prep = prepare_review(repo, staged=True, estimator=TokenEstimator())

    assert prep.prompt is not None
    assert "not staged yet" not in prep.prompt


def test_nothing_staged_is_refused_for_staged_review(repo: Path) -> None:
    edit(repo)

    prep = prepare_review(repo, staged=True, estimator=TokenEstimator())

    assert prep.prompt is None
    assert "staged" in (prep.refusal or "")


def test_the_model_is_shown_new_side_line_numbers(repo: Path) -> None:
    edit(repo)

    prep = prepare_review(repo, staged=False, estimator=TokenEstimator())

    assert prep.prompt is not None
    assert "    2 +    return 2" in prep.prompt
    assert "{{" not in prep.prompt


# ------------------------------------------------------------------ the workflow


async def test_a_correct_citation_passes_through_clean(repo: Path) -> None:
    edit(repo)
    session, runner, _provider = build(repo, "**Bad return** — `a.py:2`\nReturns the wrong value.")

    outcome = await run_review(runner=runner, session=session, root=repo)

    assert outcome.unverified == []
    assert "could not be verified" not in outcome.text


async def test_an_invented_line_is_reported_under_the_review(repo: Path) -> None:
    """The reason the workflow exists in this shape: the model cites `a.py:88` in a diff
    that stops at line 6, and the reader is told so instead of going to look for it."""
    edit(repo)
    session, runner, _provider = build(repo, "**Crash** — `a.py:88`\nDivides by zero.")

    outcome = await run_review(runner=runner, session=session, root=repo)

    assert [c.line for c in outcome.unverified] == [88]
    assert "a.py:88" in outcome.text.split("could not be verified")[-1]


async def test_the_review_is_one_standalone_call_with_no_tools(repo: Path) -> None:
    """Read-only by construction: with no tools offered there is nothing a review could
    change in the tree it is reviewing."""
    edit(repo)
    session, runner, provider = build(repo, "Nothing of note.")

    await run_review(runner=runner, session=session, root=repo)

    (request,) = provider.requests
    assert request.tools == []
    assert len(request.messages) == 1


async def test_reviewing_changes_nothing_on_disk(repo: Path) -> None:
    edit(repo)
    before = git(repo, "status", "--porcelain"), (repo / "a.py").read_bytes()
    session, runner, _provider = build(repo, "Nothing of note.")

    await run_review(runner=runner, session=session, root=repo)

    assert (git(repo, "status", "--porcelain"), (repo / "a.py").read_bytes()) == before


async def test_no_changes_never_reaches_the_model(repo: Path) -> None:
    session, runner, provider = build(repo, "unused")

    outcome = await run_review(runner=runner, session=session, root=repo)

    assert outcome.refusal
    assert provider.requests == []


async def test_files_that_did_not_fit_are_reported(repo: Path) -> None:
    """"No findings" about a diff that was half-shown must not read as all clear."""
    for n in range(40):
        (repo / f"f{n}.py").write_text("x = 1\n" * 400, encoding="utf-8")
    git(repo, "add", "-A")
    session, runner, _provider = build(repo, "Nothing of note.")

    outcome = await run_review(runner=runner, session=session, root=repo, staged=True)

    assert outcome.omitted
