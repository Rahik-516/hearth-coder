"""Structured chunking for JSON, YAML and TOML — the last I1 item.

These files have no bundled grammar, so before this they went to the sliding-window
chunker, which cuts at arbitrary line counts: a window could begin mid-table and end
mid-table, telling the model neither which section it was reading nor where it ended.

Splitting is textual rather than parsed, which is a deliberate trade — `json` and
`tomllib` report no line numbers and PyYAML is not a runtime dependency, so a parse would
buy correctness the chunker cannot use, since every chunk needs a citable line range. The
tests that matter are therefore the ones where a naive text split goes wrong: braces
inside strings, indented keys, and files small enough that splitting helps nobody.
"""

from __future__ import annotations

import itertools

import pytest

from hearth.indexing.chunker import (
    STRUCTURED_MIN_TOKENS,
    chunk_file,
    chunk_structured,
    estimate_tokens,
)


def big(text: str) -> str:
    """Pad a sample past the size at which splitting is worthwhile."""
    padding = "\n".join(f"    filler_{i} = {i}" for i in range(400))
    return text.replace("@@PAD@@", padding)


TOML = big(
    """
[tool.ruff]
line-length = 110
@@PAD@@

[tool.mypy]
strict = true

[[profile]]
family = "qwen"
"""
).strip()

YAML = big(
    """
service:
  name: billing
  ports:
    - 8080
  env:
    nested: "not a boundary"
@@PAD@@

database:
  host: localhost

logging:
  level: info
"""
).strip()

JSON = big(
    """
{
  "name": "hearth",
  "scripts": {
    "test": "pytest",
    "nested": {"deep": "value"}
  },
  "tricky": "a string with { and } and \\" inside",
@@PAD@@
  "dependencies": {
    "numpy": "^1.26"
  }
}
"""
).strip()


# ----------------------------------------------------------------------- TOML


def test_toml_splits_on_table_headers() -> None:
    chunks = chunk_structured(TOML, "toml")

    labels = [chunk.symbol_path for chunk in chunks]
    assert "tool.ruff" in labels
    assert "tool.mypy" in labels


def test_a_toml_array_of_tables_is_a_boundary() -> None:
    """`[[profile]]` starts a new entry; merging it into the previous table hides it."""
    labels = [chunk.symbol_path for chunk in chunk_structured(TOML, "toml")]

    assert "profile" in labels


# ----------------------------------------------------------------------- YAML


def test_yaml_splits_on_column_zero_keys() -> None:
    chunks = chunk_structured(YAML, "yaml")

    labels = [chunk.symbol_path for chunk in chunks]
    assert "service" in labels
    assert "database" in labels


def test_an_indented_yaml_key_is_not_a_boundary() -> None:
    """`name:` under `service:` belongs to it — splitting there cuts a value from its key."""
    labels = [chunk.symbol_path for chunk in chunk_structured(YAML, "yaml")]

    assert "name" not in labels
    assert "nested" not in labels


def test_a_yaml_document_separator_splits() -> None:
    text = "alpha: 1\n" + ("x: 1\n" * 400) + "---\nbeta: 2\n"

    labels = [chunk.symbol_path for chunk in chunk_structured(text, "yaml")]

    assert "document" in labels


# ----------------------------------------------------------------------- JSON


def test_json_splits_on_depth_one_keys() -> None:
    chunks = chunk_structured(JSON, "json")

    labels = [chunk.symbol_path for chunk in chunks]
    assert "name" in labels
    assert "scripts" in labels
    assert "dependencies" in labels


def test_a_nested_json_key_is_not_a_boundary() -> None:
    """`"test"` lives at depth 2 inside `scripts`; promoting it would fragment the object."""
    labels = [chunk.symbol_path for chunk in chunk_structured(JSON, "json")]

    assert "test" not in labels
    assert "deep" not in labels


def test_braces_and_quotes_inside_a_string_do_not_move_the_depth() -> None:
    """The case a naive brace counter gets wrong, and gets wrong for the whole rest of
    the file rather than just that line."""
    labels = [chunk.symbol_path for chunk in chunk_structured(JSON, "json")]

    # `dependencies` comes after the tricky string. If the `{` or the escaped quote inside
    # it had shifted the depth, this key would never be seen at depth 1.
    assert "dependencies" in labels


# -------------------------------------------------------------------- dispatch


@pytest.mark.parametrize(("language", "text"), [("toml", TOML), ("yaml", YAML), ("json", JSON)])
def test_chunk_file_uses_the_structured_strategy(language: str, text: str) -> None:
    result = chunk_file(source=text.encode(), text=text, language=language, parse_result=None)

    assert result.strategy == "structured"
    assert len(result.chunks) > 1


@pytest.mark.parametrize("language", ["toml", "yaml", "json"])
def test_a_small_file_is_left_to_the_window_chunker(language: str) -> None:
    """The top of a short config explains its bottom; splitting it loses that for nothing."""
    text = 'alpha: 1\nbeta: 2\n' if language == "yaml" else '{"a": 1}'
    assert estimate_tokens(text) < STRUCTURED_MIN_TOKENS

    result = chunk_file(source=text.encode(), text=text, language=language, parse_result=None)

    assert result.strategy == "window"


def test_a_flat_file_with_no_sections_falls_back() -> None:
    """One section is not a split, and returning a single chunk would just be the file."""
    text = "\n".join(f"key_{i} = {i}" for i in range(500))

    assert chunk_structured(text, "toml") == []


def test_empty_input_produces_nothing() -> None:
    assert chunk_structured("", "toml") == []
    assert chunk_structured("", "json") == []


# ------------------------------------------------------------------- integrity


@pytest.mark.parametrize(("language", "text"), [("toml", TOML), ("yaml", YAML), ("json", JSON)])
def test_line_ranges_point_at_the_real_lines(language: str, text: str) -> None:
    """A chunk whose line range is wrong produces a citation that points at other code."""
    lines = text.splitlines()

    for chunk in chunk_structured(text, language):
        assert 1 <= chunk.start_line <= chunk.end_line <= len(lines)
        span = "\n".join(lines[chunk.start_line - 1 : chunk.end_line]).strip()
        assert span == chunk.text


@pytest.mark.parametrize(("language", "text"), [("toml", TOML), ("yaml", YAML), ("json", JSON)])
def test_chunks_cover_the_file_without_overlapping(language: str, text: str) -> None:
    """Overlap would double-count content; gaps would lose it."""
    chunks = chunk_structured(text, language)

    for earlier, later in itertools.pairwise(chunks):
        assert earlier.end_line < later.start_line


def test_content_before_the_first_section_is_kept() -> None:
    """A licence header ahead of the first table belongs to no section.

    Found by checking line coverage on a real file: lines 1-3 appeared in no chunk at all,
    so the header was silently absent from the index — searchable nowhere, citable never.
    """
    pad = "\n".join(f"filler_{i} = {i}" for i in range(400))
    text = f"# Copyright 2026 Example Corp\n# Licensed under MIT\n\n[tool.a]\n{pad}\n\n[tool.b]\nx = 1\n"

    chunks = chunk_structured(text, "toml")

    assert chunks[0].start_line == 1
    assert "Copyright" in chunks[0].text
    assert chunks[0].kind == "preamble"


@pytest.mark.parametrize(("language", "text"), [("toml", TOML), ("yaml", YAML), ("json", JSON)])
def test_every_non_blank_line_lands_in_some_chunk(language: str, text: str) -> None:
    """Content the index never sees cannot be retrieved, cited, or noticed as missing."""
    chunks = chunk_structured(text, language)
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(range(chunk.start_line, chunk.end_line + 1))

    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            assert number in covered, f"line {number} is in no chunk: {line!r}"
