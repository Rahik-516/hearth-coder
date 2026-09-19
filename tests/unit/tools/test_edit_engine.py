"""The edit engine — docs/safety-and-tool-use.md §7.2.

Written before ``tools/edit_engine.py`` (CLAUDE.md rule 3). This file *is* M5's second
acceptance criterion: "the edit engine test matrix passes: CRLF files, BOM, tabs vs
spaces, trailing whitespace, ambiguous matches, missing matches with hints, and a stale
file" (docs/implementation-roadmap.md M5).

The engine's whole job is to make a model's approximate idea of a file's contents land
exactly, or refuse. The dangerous outcome is not a refusal — it is a near-miss that edits
the wrong region, because the preview the user approved was computed from the same wrong
match and looks entirely reasonable.
"""

from __future__ import annotations

import pytest

from hearth.tools.edit_engine import (
    MAX_WRITE_BYTES,
    FileForm,
    apply_edit,
    diff_stats,
    encode_with_form,
    read_form,
)

PY_SOURCE = """def finalize(self, invoice_id):
    total = sum(line.amount for line in self.lines)
    return total
"""


# ------------------------------------------------------------------- file form


def test_plain_utf8_lf_round_trips() -> None:
    data = b"alpha\nbeta\n"
    form = read_form(data)

    assert form.encoding == "utf-8"
    assert form.line_ending == "\n"
    assert encode_with_form(form.decode(data), form) == data


def test_crlf_is_detected_and_restored() -> None:
    """A Windows-authored file must come back as CRLF, byte for byte.

    This is the WSL2 case from §16.3: the repo is edited from both sides, and an edit that
    silently rewrote every line ending would show up as a whole-file diff in git.
    """
    data = b"alpha\r\nbeta\r\n"
    form = read_form(data)

    assert form.line_ending == "\r\n"
    assert encode_with_form(form.decode(data), form) == data


def test_a_bom_survives_an_edit() -> None:
    data = "\ufeffalpha\nbeta\n".encode()
    form = read_form(data)

    assert form.encoding == "utf-8-sig"
    assert encode_with_form(form.decode(data), form) == data
    assert encode_with_form(form.decode(data), form).startswith(b"\xef\xbb\xbf")


def test_the_model_never_sees_carriage_returns() -> None:
    """Decoded text is always LF, whatever the file is.

    Matching has to happen in LF space: the model writes `old_string` with `\\n`, and
    requiring it to guess a file's line endings would make every CRLF file unmatchable.
    """
    form = read_form(b"a\r\nb\r\n")

    assert "\r" not in form.decode(b"a\r\nb\r\n")


def test_a_missing_trailing_newline_is_preserved() -> None:
    data = b"alpha\nbeta"
    form = read_form(data)

    assert form.had_trailing_newline is False
    assert encode_with_form(form.decode(data), form) == data


def test_binary_content_is_refused() -> None:
    form = read_form(b"\x00\x01binary")

    assert form.usable is False


def test_undecodable_bytes_are_refused_rather_than_mangled() -> None:
    """latin-1 would "work" and corrupt the file on write-back.

    ``util.text.decode_text`` replaces undecodable bytes, which is right for indexing and
    catastrophic for editing — the round trip would not be lossless.
    """
    form = read_form(b"valid \xff\xfe invalid")

    assert form.usable is False


# ---------------------------------------------------------------- exact match


def test_an_exact_unique_match_is_applied() -> None:
    outcome = apply_edit(PY_SOURCE, "return total", "return total.quantize(CENTS)")

    assert outcome.ok
    assert outcome.strategy == "exact"
    assert outcome.new_text is not None
    assert "return total.quantize(CENTS)" in outcome.new_text
    assert outcome.badges == ()


def test_an_exact_match_in_a_crlf_file_works_with_lf_input() -> None:
    """The reason matching happens in LF space."""
    data = b"def f():\r\n    return 1\r\n"
    form = read_form(data)

    outcome = apply_edit(form.decode(data), "    return 1", "    return 2")

    assert outcome.ok
    assert outcome.new_text is not None
    assert encode_with_form(outcome.new_text, form) == b"def f():\r\n    return 2\r\n"


# ------------------------------------------------------------ fuzzy strategies


def test_trailing_whitespace_differences_are_tolerated() -> None:
    """The model reproduced a whole line but dropped the spaces someone left at its end.

    The needle has to span the line ending for this to be the *normalized* strategy's
    job — a bare fragment like `    return 1` is an exact substring of a line with
    trailing whitespace, so exact matching gets there first (and should; see below).
    """
    source = "def f():\n    return 1   \n"

    outcome = apply_edit(source, "    return 1\n", "    return 2\n")

    assert outcome.ok
    assert outcome.strategy == "normalized"
    assert "FUZZY-MATCH" in outcome.badges


def test_a_fragment_match_leaves_trailing_whitespace_alone() -> None:
    """Replacing part of a line must not rewrite the rest of it.

    Trailing whitespace is ugly but it is not this edit's business. Normalising it away
    here would put an unrequested change in a diff the user is being asked to approve.
    """
    source = "def f():\n    return 1   \n"

    outcome = apply_edit(source, "return 1", "return 2")

    assert outcome.ok
    assert outcome.strategy == "exact"
    assert outcome.new_text == "def f():\n    return 2   \n"


def test_indentation_insensitive_matching_reindents_to_the_file() -> None:
    """A model that guesses the wrong indent depth should still land the edit.

    But the replacement has to be re-indented to the *file's* level, not pasted at the
    model's — otherwise a correct match produces syntactically broken Python.
    """
    source = "class A:\n    def f(self):\n        return 1\n"

    outcome = apply_edit(source, "def f(self):\n    return 1", "def f(self):\n    return 2")

    assert outcome.ok
    assert outcome.strategy == "indent"
    assert "FUZZY-MATCH" in outcome.badges
    assert outcome.new_text == "class A:\n    def f(self):\n        return 2\n"


def test_tabs_and_spaces_are_not_silently_conflated() -> None:
    """A tab-indented file edited with spaces must not become mixed.

    Indentation-insensitive matching compares *depth*, and re-indents using the file's own
    indent character, so the result stays internally consistent.
    """
    source = "def f():\n\treturn 1\n"

    outcome = apply_edit(source, "    return 1", "    return 2")

    assert outcome.ok
    assert outcome.new_text == "def f():\n\treturn 2\n"


def test_an_exact_match_is_preferred_over_a_fuzzy_one() -> None:
    """Strategy order matters: exact first, always (§7.2)."""
    source = "x = 1\nx = 1   \n"

    outcome = apply_edit(source, "x = 1\n", "x = 2\n")

    assert outcome.strategy == "exact"


# ----------------------------------------------------------- ambiguity refusal


def test_multiple_matches_are_refused_with_their_line_numbers() -> None:
    source = "a = 1\nb = 2\na = 1\n"

    outcome = apply_edit(source, "a = 1", "a = 9")

    assert not outcome.ok
    assert outcome.replacements == 2
    assert outcome.error is not None
    assert "2" in outcome.error
    assert "1" in (outcome.hint or "") and "3" in (outcome.hint or ""), "should cite both lines"


def test_replace_all_applies_every_match() -> None:
    source = "a = 1\nb = 2\na = 1\n"

    outcome = apply_edit(source, "a = 1", "a = 9", replace_all=True)

    assert outcome.ok
    assert outcome.replacements == 2
    assert outcome.new_text == "a = 9\nb = 2\na = 9\n"


def test_replace_all_does_not_rescue_a_zero_match() -> None:
    outcome = apply_edit(PY_SOURCE, "nonexistent", "x", replace_all=True)

    assert not outcome.ok


# ------------------------------------------------------------- missing matches


def test_a_missing_match_reports_the_closest_region() -> None:
    """The hint is the difference between one wasted step and three.

    A model that guessed the text slightly wrong needs to see what is actually there, with
    line numbers, so its next attempt is a correction rather than another guess.
    """
    source = "def finalize(self):\n    total = compute()\n    return total\n"

    outcome = apply_edit(source, "    total = compute_total()", "    total = 0")

    assert not outcome.ok
    assert outcome.replacements == 0
    assert outcome.hint is not None
    assert "total = compute()" in outcome.hint
    assert "2" in outcome.hint, "the hint cites line numbers"


def test_a_hopeless_match_still_fails_cleanly() -> None:
    outcome = apply_edit(PY_SOURCE, "\u2603 nothing like this exists \u2603", "x")

    assert not outcome.ok
    assert outcome.error is not None


def test_an_empty_old_string_is_refused() -> None:
    """Otherwise it matches at offset 0 and prepends, which no caller means to ask for."""
    outcome = apply_edit(PY_SOURCE, "", "x")

    assert not outcome.ok


def test_replacing_text_with_itself_is_refused() -> None:
    """A no-op edit would checkpoint, reindex and ask for approval, all to change nothing."""
    outcome = apply_edit(PY_SOURCE, "return total", "return total")

    assert not outcome.ok
    assert outcome.error is not None


# ---------------------------------------------------------------- diff stats


def test_diff_stats_counts_added_and_removed_lines() -> None:
    added, removed = diff_stats("a\nb\nc\n", "a\nB\nc\nd\n")

    assert added == 2  # B and d
    assert removed == 1  # b


def test_a_large_diff_is_flagged() -> None:
    before = "".join(f"line {n}\n" for n in range(400))
    after = "".join(f"changed {n}\n" for n in range(400))

    outcome = apply_edit(before, before, after)

    assert outcome.ok
    assert "LARGE" in outcome.badges


def test_a_small_diff_is_not_flagged() -> None:
    outcome = apply_edit(PY_SOURCE, "return total", "return total  # rounded")

    assert "LARGE" not in outcome.badges


# ----------------------------------------------------------- parse guard


def test_an_edit_that_breaks_the_syntax_is_flagged() -> None:
    """M5 acceptance: an edit introducing a syntax error shows PARSE-ERRORS-INTRODUCED."""
    outcome = apply_edit(PY_SOURCE, "    return total", "    return total(((", language="python")

    assert outcome.ok, "the badge warns; it does not block"
    assert "PARSE-ERRORS-INTRODUCED" in outcome.badges


def test_a_clean_edit_is_not_flagged() -> None:
    outcome = apply_edit(PY_SOURCE, "    return total", "    return round(total)", language="python")

    assert "PARSE-ERRORS-INTRODUCED" not in outcome.badges


def test_a_file_that_already_had_errors_is_judged_on_the_delta() -> None:
    """Plenty of real files do not parse cleanly — vendored snippets, newer syntax.

    Refusing to edit those, or badging every edit to them, would make the signal useless.
    What matters is whether *this* edit made the file worse.
    """
    broken = "def f(:\n    pass\n"

    outcome = apply_edit(broken, "    pass", "    return 1", language="python")

    assert outcome.ok
    assert "PARSE-ERRORS-INTRODUCED" not in outcome.badges


def test_the_guard_is_silent_when_the_language_is_unknown() -> None:
    """No grammar is not the same as no errors, so the badge is simply absent."""
    outcome = apply_edit(PY_SOURCE, "    return total", "    return total(((")

    assert outcome.ok
    assert "PARSE-ERRORS-INTRODUCED" not in outcome.badges


# ------------------------------------------------------------------- size cap


def test_the_write_size_cap_is_two_megabytes() -> None:
    """§7.2 item 3. Stated as a constant so the tool and the test agree on the number."""
    assert MAX_WRITE_BYTES == 2 * 1024 * 1024


# ------------------------------------------------------------------ form reuse


@pytest.mark.parametrize(
    "data",
    [
        b"a\nb\n",
        b"a\r\nb\r\n",
        "\ufeffa\nb\n".encode(),
        "\ufeffa\r\nb\r\n".encode(),
        b"no trailing newline",
        b"",
    ],
)
def test_decode_encode_is_lossless_for_every_supported_form(data: bytes) -> None:
    """The property the whole module rests on.

    If the round trip is not lossless, then every edit rewrites parts of the file nobody
    asked to change — and the diff the user approved was a lie about the bytes.
    """
    form = read_form(data)
    if not form.usable:
        pytest.skip("not a text form")

    assert encode_with_form(form.decode(data), form) == data


def test_form_is_hashable_so_it_survives_the_approval_round_trip() -> None:
    """``FileForm`` is carried in ``Prepared.payload`` from prepare() to execute().

    It has to be a value, not a handle on an open file: the approval in between can take
    arbitrarily long, and the form must still describe the file that was previewed.
    """
    form = read_form(b"a\r\n")

    assert isinstance(form, FileForm)
    assert hash(form) == hash(FileForm(encoding="utf-8", line_ending="\r\n"))
