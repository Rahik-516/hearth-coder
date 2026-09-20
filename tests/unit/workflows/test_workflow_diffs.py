"""Diff parsing, rendering and citation checking — I3 workflows.

`/review` is only as trustworthy as its citations, and citations are only checkable if the
diff was parsed correctly, so most of this file is about the parser being right in the
awkward cases: a deleted line that looks like a file header, a rename, a binary file, a hunk
whose header omits its counts.

The verification tests are the point of the module. The model is shown a numbered view and
asked to copy line numbers from it; what these check is that a number it did *not* copy —
invented, or miscounted — is caught rather than passed on to the reader as a finding.
"""

from __future__ import annotations

import pytest

from hearth.core.context.tokens import TokenEstimator
from hearth.workflows.diffs import (
    parse_diff,
    render,
    unverified_footer,
    verify_citations,
)

MODIFIED = """\
diff --git a/src/billing.py b/src/billing.py
index 111..222 100644
--- a/src/billing.py
+++ b/src/billing.py
@@ -10,4 +10,5 @@ def total():
     a = 1
-    b = 2
+    b = 3
+    c = 4
     return a
"""


def test_a_modified_file_is_parsed_with_new_side_line_numbers() -> None:
    (file,) = parse_diff(MODIFIED)

    assert file.path == "src/billing.py"
    assert file.status == "modified"
    assert (file.added, file.removed) == (2, 1)
    numbered = [(line.kind, line.new_line) for line in file.lines]
    assert numbered == [("ctx", 10), ("del", None), ("add", 11), ("add", 12), ("ctx", 13)]


def test_a_deleted_line_has_no_new_position() -> None:
    """It does not exist in the new file, so it cannot be cited — and the numbered view
    must not give the model a number to cite it by."""
    (file,) = parse_diff(MODIFIED)

    assert 11 in file.citable_lines()
    assert len([line for line in file.lines if line.new_line is None]) == 1


def test_a_removed_line_that_looks_like_a_file_header_is_still_a_line() -> None:
    """The reason the parser counts instead of pattern-matching.

    Deleting the SQL comment `-- old` produces the diff line `--- old`, which is
    indistinguishable from a `--- a/file` header by its first characters.
    """
    diff = (
        "diff --git a/q.sql b/q.sql\n"
        "--- a/q.sql\n"
        "+++ b/q.sql\n"
        "@@ -1,2 +1,1 @@\n"
        "--- old comment\n"
        " SELECT 1;\n"
    )

    (file,) = parse_diff(diff)

    assert [line.kind for line in file.lines] == ["del", "ctx"]
    assert file.lines[0].text == "-- old comment"
    assert file.path == "q.sql", "the deleted line did not overwrite the path"


def test_an_added_line_that_looks_like_a_file_header_is_still_a_line() -> None:
    diff = (
        "diff --git a/q.sql b/q.sql\n"
        "--- a/q.sql\n"
        "+++ b/q.sql\n"
        "@@ -1,1 +1,2 @@\n"
        " SELECT 1;\n"
        "+++ new comment\n"
    )

    (file,) = parse_diff(diff)

    assert file.lines[-1].kind == "add"
    assert file.path == "q.sql"


def test_several_files_are_separated() -> None:
    second = MODIFIED.replace("billing", "ledger")

    files = parse_diff(MODIFIED + second)

    assert [file.path for file in files] == ["src/billing.py", "src/ledger.py"]


def test_a_new_file_is_marked_added() -> None:
    diff = (
        "diff --git a/n.py b/n.py\nnew file mode 100644\n--- /dev/null\n+++ b/n.py\n"
        "@@ -0,0 +1,2 @@\n+one\n+two\n"
    )

    (file,) = parse_diff(diff)

    assert file.status == "added"
    assert [line.new_line for line in file.lines] == [1, 2]


def test_a_deleted_file_keeps_its_path() -> None:
    diff = (
        "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n-x = 1\n"
    )

    (file,) = parse_diff(diff)

    assert file.path == "gone.py"
    assert file.status == "deleted"
    assert file.citable_lines() == frozenset()


def test_a_rename_records_both_names() -> None:
    diff = (
        "diff --git a/old.py b/new.py\nsimilarity index 100%\nrename from old.py\n"
        "rename to new.py\n"
    )

    (file,) = parse_diff(diff)

    assert (file.path, file.old_path, file.status) == ("new.py", "old.py", "renamed")


def test_a_binary_file_is_recorded_without_lines() -> None:
    diff = "diff --git a/i.png b/i.png\nBinary files a/i.png and b/i.png differ\n"

    (file,) = parse_diff(diff)

    assert file.status == "binary"
    assert file.lines == []


def test_a_hunk_header_without_counts_means_one_line() -> None:
    """`@@ -3 +3 @@` is a one-line hunk; git omits a count of 1."""
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -3 +3 @@\n-old\n+new\n"

    (file,) = parse_diff(diff)

    assert [(line.kind, line.new_line) for line in file.lines] == [("del", None), ("add", 3)]


def test_no_newline_marker_is_not_a_line() -> None:
    diff = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n"
        "\\ No newline at end of file\n+new\n\\ No newline at end of file\n"
    )

    (file,) = parse_diff(diff)

    assert [line.kind for line in file.lines] == ["del", "add"]


def test_empty_input_parses_to_nothing() -> None:
    assert parse_diff("") == []


# ------------------------------------------------------------------ rendering


def test_the_numbered_view_puts_the_new_line_number_beside_each_line() -> None:
    (file,) = parse_diff(MODIFIED)

    rendered = render([file], budget_tokens=500, estimator=TokenEstimator(), numbered=True)

    assert "   11 +    b = 3" in rendered.text
    assert "-    b = 2" in rendered.text, "a removed line is shown, without a number"
    assert rendered.omitted == []


def test_the_plain_view_has_no_numbers() -> None:
    (file,) = parse_diff(MODIFIED)

    rendered = render([file], budget_tokens=500, estimator=TokenEstimator(), numbered=False)

    assert "+    b = 3" in rendered.text
    assert "   11 " not in rendered.text


def test_a_gap_between_hunks_is_marked() -> None:
    diff = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "@@ -1,1 +1,1 @@\n-a\n+A\n@@ -50,1 +50,1 @@\n-b\n+B\n"
    )

    rendered = render(parse_diff(diff), budget_tokens=500, estimator=TokenEstimator())

    assert "  ..." in rendered.text


def test_files_that_do_not_fit_are_named_not_dropped() -> None:
    """A review that says "looks good" about a diff it only half saw is the failure."""
    header = "diff --git a/{n}.py b/{n}.py\n--- a/{n}.py\n+++ b/{n}.py\n@@ -1,1 +1,1 @@\n-a\n+"
    big = header + "x" * 400 + "\n"
    files = parse_diff(big.format(n="one") + big.format(n="two") + big.format(n="three"))

    rendered = render(files, budget_tokens=200, estimator=TokenEstimator())

    assert rendered.included, "something was shown"
    assert rendered.omitted, "and what was not shown is named"
    assert set(rendered.included) | set(rendered.omitted) == {"one.py", "two.py", "three.py"}


def test_one_enormous_file_is_cut_rather_than_skipped() -> None:
    """Skipping it would hide that a lockfile changed at all; cutting shows that and how
    much."""
    lines = "".join(f"+line {n}\n" for n in range(2000))
    diff = f"diff --git a/lock.txt b/lock.txt\n--- a/lock.txt\n+++ b/lock.txt\n@@ -0,0 +1,2000 @@\n{lines}"

    rendered = render(parse_diff(diff), budget_tokens=300, estimator=TokenEstimator())

    assert rendered.included == ["lock.txt"]
    assert rendered.truncated
    assert "file cut here" in rendered.text


# --------------------------------------------------------------- verification


@pytest.fixture
def files():
    return parse_diff(MODIFIED)


def test_a_citation_of_a_shown_line_is_verified(files) -> None:
    (citation,) = verify_citations("`src/billing.py:11` assigns the wrong value.", files)

    assert citation.verified


def test_a_citation_of_a_context_line_is_verified(files) -> None:
    """The model was shown it, so citing it is not an invention."""
    (citation,) = verify_citations("see src/billing.py:10", files)

    assert citation.verified


def test_a_line_that_was_not_shown_is_not_verified(files) -> None:
    (citation,) = verify_citations("src/billing.py:88 divides by zero", files)

    assert not citation.verified
    assert "not in the diff" in citation.reason


def test_a_line_past_the_end_of_the_hunk_is_not_verified(files) -> None:
    """The hunk shows new-side lines 10-13. Line 14 exists in the file, but the model was
    never shown it, so citing it is a guess that happens to be in range."""
    (citation,) = verify_citations("src/billing.py:14", files)

    assert not citation.verified


def test_a_file_outside_the_diff_is_not_verified(files) -> None:
    (citation,) = verify_citations("utils.py:3 is affected", files)

    assert not citation.verified
    assert "not a file in the reviewed diff" in citation.reason


def test_a_dropped_directory_still_resolves_when_unambiguous(files) -> None:
    """Models routinely write `billing.py` for `src/billing.py`; flagging correct findings
    for that would teach users to ignore the footer."""
    (citation,) = verify_citations("billing.py:11", files)

    assert citation.verified


def test_an_ambiguous_suffix_is_not_guessed() -> None:
    files = parse_diff(MODIFIED + MODIFIED.replace("src/billing", "lib/billing"))

    (citation,) = verify_citations("billing.py:11", files)

    assert not citation.verified
    assert "ambiguous" in citation.reason


def test_a_line_range_is_checked_by_its_start(files) -> None:
    (citation,) = verify_citations("src/billing.py:11-13", files)

    assert citation.verified
    assert citation.line == 11


def test_prose_that_is_not_a_citation_is_ignored(files) -> None:
    """`step 3: 12 files` has no file extension, so it is not a path."""
    assert verify_citations("In step 3: 12 files changed, at 10:30 today.", files) == []


def test_the_same_citation_is_reported_once(files) -> None:
    results = verify_citations("src/billing.py:11 and again src/billing.py:11", files)

    assert len(results) == 1


def test_the_footer_lists_only_the_failures(files) -> None:
    citations = verify_citations("src/billing.py:11 and src/billing.py:99", files)

    footer = unverified_footer(citations)

    assert "src/billing.py:99" in footer
    assert "src/billing.py:11" not in footer


def test_the_footer_is_empty_when_everything_checks_out(files) -> None:
    assert unverified_footer(verify_citations("src/billing.py:11", files)) == ""
