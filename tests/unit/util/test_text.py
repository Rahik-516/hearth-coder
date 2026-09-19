"""Identifier splitting and file-text helpers.

The two documented examples in docs/system-design.md §6.5 are pinned as tests, because
retrieval quality depends on them and they are easy to regress while "tidying" the regex.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hearth.util.text import (
    average_line_length,
    build_search_query,
    build_search_text,
    count_lines,
    decode_text,
    detect_encoding,
    dominant_line_ending,
    expand_identifier,
    extract_identifiers,
    is_probably_binary,
    split_identifier,
)


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("InvoiceService", ["invoice", "service"]),
        ("STRIPE_WEBHOOK_SECRET", ["stripe", "webhook", "secret"]),
        ("finalize", ["finalize"]),
        ("parseURLQuery", ["parse", "url", "query"]),
        ("HTTPServer", ["http", "server"]),
        ("snake_case_name", ["snake", "case", "name"]),
        ("mixed_CamelAnd_snake", ["mixed", "camel", "and", "snake"]),
        ("XMLHttpRequest", ["xml", "http", "request"]),
    ],
)
def test_split_identifier_cases(identifier: str, expected: list[str]) -> None:
    assert split_identifier(identifier) == expected


def test_documented_example_invoice_service() -> None:
    """docs/system-design.md §6.5: InvoiceService.finalize -> invoice service finalize invoiceservice."""
    forms = []
    for part in extract_identifiers("InvoiceService.finalize"):
        for form in expand_identifier(part):
            if form not in forms:
                forms.append(form)

    assert forms == ["invoice", "service", "invoiceservice", "finalize"]


def test_documented_example_stripe_secret() -> None:
    """docs/system-design.md §6.5: STRIPE_WEBHOOK_SECRET keeps its verbatim form too."""
    forms = expand_identifier("STRIPE_WEBHOOK_SECRET")

    assert forms[:3] == ["stripe", "webhook", "secret"]
    assert "stripe_webhook_secret" in forms


def test_single_characters_are_dropped_as_noise() -> None:
    assert split_identifier("aB") == []
    assert "x" not in split_identifier("xCoordinate")


def test_digits_survive() -> None:
    assert split_identifier("utf8Decoder") == ["utf", "8", "decoder"]


def test_extract_identifiers_preserves_order_and_dedupes() -> None:
    found = extract_identifiers("def finalize(self): return finalize_id")
    assert found == ["def", "finalize", "self", "return", "finalize_id"]


def test_build_search_text_keeps_the_original() -> None:
    """Exact-phrase queries must still work, so the original text is never discarded."""
    original = "class InvoiceService:"
    enriched = build_search_text(original)

    assert enriched.startswith(original)
    assert "invoice service" in enriched or ("invoice" in enriched and "service" in enriched)


def test_build_search_text_handles_textless_input() -> None:
    assert build_search_text("") == ""
    assert build_search_text("!!!") == "!!!"


def test_query_and_document_expand_the_same_way() -> None:
    """If the two sides disagreed, a matching query would silently miss."""
    document = build_search_text("InvoiceService")
    for term in build_search_query("invoice service"):
        assert term in document


# ------------------------------------------------------------------ properties


@given(st.text())
def test_split_identifier_never_raises(text: str) -> None:
    parts = split_identifier(text)
    assert all(isinstance(p, str) for p in parts)


@given(st.text())
def test_split_parts_are_lowercase_and_alphanumeric(text: str) -> None:
    for part in split_identifier(text):
        assert part == part.lower()
        assert part.isalnum()


@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1))
def test_build_search_text_is_a_superset(text: str) -> None:
    """Enrichment only ever adds; it must never lose the original content."""
    assert build_search_text(text).startswith(text)


@given(st.text())
def test_expand_identifier_has_no_duplicates(text: str) -> None:
    forms = expand_identifier(text)
    assert len(forms) == len(set(forms))


@given(st.lists(st.sampled_from(["Invoice", "Service", "HTTP", "url", "id", "2"]), min_size=1))
def test_concatenated_identifiers_split_back_apart(words: list[str]) -> None:
    identifier = "".join(words)
    parts = split_identifier(identifier)
    for word in words:
        if len(word) >= 2 and not word.isdigit():
            assert word.lower() in parts or word.lower() in "".join(parts)


# ----------------------------------------------------------------- file text


def test_detect_encoding_utf8_and_bom() -> None:
    assert detect_encoding(b"hello") == "utf-8"
    assert detect_encoding(b"\xef\xbb\xbfhello") == "utf-8-sig"
    assert detect_encoding(b"\xff\xfe\x00bad") == "latin-1"


def test_decode_text_never_raises() -> None:
    assert decode_text(b"\xff\xfe invalid utf-8 \xc3") is not None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a\nb\nc", "\n"),
        ("a\r\nb\r\n", "\r\n"),
        ("a\rb\r", "\r"),
        ("no newlines", "\n"),
        ("", "\n"),
    ],
)
def test_dominant_line_ending(text: str, expected: str) -> None:
    assert dominant_line_ending(text) == expected


def test_binary_detection_uses_nul_byte() -> None:
    assert is_probably_binary(b"text\x00more") is True
    assert is_probably_binary(b"plain text") is False
    # A NUL beyond the sniff window is not considered.
    assert is_probably_binary(b"a" * 9000 + b"\x00") is False


def test_count_lines_counts_unterminated_final_line() -> None:
    assert count_lines("a\nb") == 2
    assert count_lines("a\nb\n") == 2
    assert count_lines("") == 0


def test_average_line_length_flags_minified_shape() -> None:
    assert average_line_length("short\nlines\n") < 10
    assert average_line_length("x" * 500) > 300
