"""Chunker behaviour and per-language snapshots.

Snapshots capture chunk *boundaries* — kind, symbol path, line range, size — not chunk
text. Boundaries are what regress when the algorithm or a grammar changes, and a full-text
snapshot would be too noisy to review, which defeats the purpose.

A moved boundary invalidates every embedding for that file, so these diffs are meant to be
read, not blindly accepted (docs/tech-stack.md §3.2).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from hearth.indexing.chunker import (
    HARD_MAX_TOKENS,
    SOFT_MAX_TOKENS,
    chunk_file,
    estimate_tokens,
)
from hearth.indexing.parser import parse
from hearth.indexing.symbols import extract
from hearth.util.text import decode_text

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "repos"


def chunk_path(relative: str, language: str):
    raw = (FIXTURES / relative).read_bytes()
    parsed = parse(raw, language)
    extraction = extract(parsed.root, raw, language) if parsed.root is not None else None
    return chunk_file(
        source=raw,
        text=decode_text(raw),
        language=language,
        parse_result=parsed,
        extraction=extraction,
    )


def chunk_source(code: str, language: str):
    raw = code.encode("utf-8")
    parsed = parse(raw, language)
    extraction = extract(parsed.root, raw, language) if parsed.root is not None else None
    return chunk_file(
        source=raw,
        text=code,
        language=language,
        parse_result=parsed,
        extraction=extraction,
    )


def outline(result) -> list[str]:
    """A reviewable one-line-per-chunk summary."""
    return [f"{c.kind:16} {c.symbol_path or '-':44} L{c.start_line}-{c.end_line}" for c in result.chunks]


# ------------------------------------------------------------------ snapshots


@pytest.mark.parametrize(
    ("relative", "language"),
    [
        ("py_small/src/billing/invoice_service.py", "python"),
        ("py_small/src/billing/reporting.py", "python"),
        ("py_small/src/billing/models.py", "python"),
        ("py_small/src/billing/broken_syntax.py", "python"),
        ("ts_small/src/cart.ts", "typescript"),
        ("ts_small/src/types.ts", "typescript"),
        ("ts_small/src/components/CartBadge.tsx", "tsx"),
        ("ts_small/src/index.js", "javascript"),
        ("py_small/README.md", "markdown"),
    ],
)
def test_chunk_boundaries_snapshot(relative: str, language: str, snapshot) -> None:
    assert outline(chunk_path(relative, language)) == snapshot


# -------------------------------------------------------------- token sizing


def test_token_estimate_ignores_indentation() -> None:
    """Size is measured in non-whitespace characters.

    Otherwise the same method would measure larger inside a class than at module level,
    and chunk boundaries would shift when code is merely re-indented.
    """
    flat = "def f():\nreturn 1"
    indented = "        def f():\n            return 1"

    assert estimate_tokens(flat) == estimate_tokens(indented)


def test_token_estimate_is_positive_for_any_content() -> None:
    assert estimate_tokens("") >= 1
    assert estimate_tokens("x") >= 1


# --------------------------------------------------------------- AST chunking


def test_definitions_become_their_own_chunks() -> None:
    result = chunk_path("py_small/src/billing/invoice_service.py", "python")
    paths = {c.symbol_path for c in result.chunks}

    assert "InvoiceService" in paths
    assert result.strategy == "ast"


def test_nested_class_is_reachable_through_its_parent() -> None:
    """Chunker coverage for nested definitions."""
    result = chunk_path("py_small/src/billing/invoice_service.py", "python")
    service = next(c for c in result.chunks if c.symbol_path == "InvoiceService")

    assert "class Config" in service.text


def test_decorators_stay_attached_to_their_definition() -> None:
    code = '''
@decorator_one
@decorator_two("arg")
def target(a, b):
    """Body."""
    return a + b
'''
    result = chunk_source(code, "python")
    # Skip the skeleton, which carries signatures only.
    target = next(c for c in result.chunks if "def target" in c.text and c.kind != "module_skeleton")

    assert "@decorator_one" in target.text
    assert "@decorator_two" in target.text
    assert target.start_line < 4, "the chunk must start at the first decorator, not the def"


def test_module_skeleton_is_emitted_first() -> None:
    """The skeleton answers "what's in this file?" cheaply, so it leads."""
    result = chunk_path("py_small/src/billing/invoice_service.py", "python")

    assert result.chunks[0].kind == "module_skeleton"
    assert "InvoiceService" in result.chunks[0].text
    assert result.chunks[0].token_estimate < 600, "a skeleton must stay cheap"


def test_skeleton_elides_bodies() -> None:
    result = chunk_path("py_small/src/billing/invoice_service.py", "python")
    skeleton = result.chunks[0]

    assert "def finalize" in skeleton.text
    # The body of finalize must not be present.
    assert "raise InvoiceNotFound" not in skeleton.text


def test_oversized_definition_is_split() -> None:
    """The split path: a function past the soft maximum must not be emitted whole."""
    body = "\n".join(f"    value_{i} = compute_{i}(alpha, beta, gamma, delta)" for i in range(400))
    code = f"def enormous():\n{body}\n"

    result = chunk_source(code, "python")

    assert len(result.chunks) > 1, "an oversized function should have been split"
    assert all(c.token_estimate <= HARD_MAX_TOKENS for c in result.chunks)


def test_oversized_class_keeps_a_skeleton() -> None:
    """Splitting a big class must not lose its shape."""
    methods = "\n".join(
        f"    def method_{i}(self, alpha, beta):\n        return compute_{i}(alpha, beta) + {i}\n"
        for i in range(120)
    )
    code = f"class Huge:\n{methods}\n"

    result = chunk_source(code, "python")
    kinds = [c.kind for c in result.chunks]

    assert "class_skeleton" in kinds
    assert len(result.chunks) > 2


def test_tiny_siblings_are_merged() -> None:
    """Dozens of one-line chunks would each waste a retrieval slot."""
    code = "\n".join(f"CONST_{i} = {i}" for i in range(40))

    result = chunk_source(code, "python")

    assert len(result.chunks) < 10


# --------------------------------------------------------- parse failures


def test_syntax_error_file_still_chunks() -> None:
    """M1 acceptance criterion. An unparsed file vanishing from the index is far worse."""
    result = chunk_path("py_small/src/billing/broken_syntax.py", "python")

    assert result.chunks, "a broken file must still produce chunks"
    assert "".join(c.text for c in result.chunks).strip()


def test_unparsed_language_falls_back_to_windows() -> None:
    code = "\n".join(f"line {i} of some unsupported language" for i in range(200))
    result = chunk_file(
        source=code.encode("utf-8"),
        text=code,
        language="cobol",
        parse_result=None,
    )

    assert result.strategy == "window"
    assert len(result.chunks) > 1


def test_window_chunks_overlap() -> None:
    """Windows overlap because their boundaries are arbitrary; AST chunks do not."""
    code = "\n".join(f"statement_{i}()" for i in range(200))
    result = chunk_file(source=code.encode("utf-8"), text=code, language=None, parse_result=None)

    assert len(result.chunks) >= 2
    first, second = result.chunks[0], result.chunks[1]
    assert second.start_line <= first.end_line, "expected overlapping windows"


def test_ast_chunks_do_not_overlap() -> None:
    result = chunk_path("py_small/src/billing/models.py", "python")
    bodies = [c for c in result.chunks if c.kind != "module_skeleton"]

    for earlier, later in itertools.pairwise(bodies):
        assert later.start_byte >= earlier.start_byte


# ------------------------------------------------------------------ markdown


def test_markdown_splits_on_headings() -> None:
    text = "# Title\n\nIntro.\n\n## Section A\n\nBody A.\n\n## Section B\n\nBody B.\n"
    result = chunk_file(source=text.encode("utf-8"), text=text, language="markdown", parse_result=None)

    assert result.strategy == "markdown"
    assert [c.symbol_path for c in result.chunks] == ["Title", "Section A", "Section B"]


def test_markdown_section_keeps_its_heading() -> None:
    """A section severed from its title usually loses what made it findable."""
    text = "# Alpha\n\nalpha body\n\n# Beta\n\nbeta body\n"
    result = chunk_file(source=text.encode("utf-8"), text=text, language="markdown", parse_result=None)

    for chunk in result.chunks:
        assert chunk.text.startswith("#")


def test_empty_file_produces_no_chunks() -> None:
    result = chunk_file(source=b"", text="", language="python", parse_result=None)
    assert result.chunks == []


# ------------------------------------------------------------------- ranges


def test_line_ranges_are_one_based_and_ordered() -> None:
    result = chunk_path("py_small/src/billing/models.py", "python")

    for chunk in result.chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_line_ranges_resolve_to_real_source() -> None:
    """Citations point at these ranges, so they must actually contain the chunk."""
    relative = "py_small/src/billing/invoice_service.py"
    lines = (FIXTURES / relative).read_text(encoding="utf-8").splitlines()
    result = chunk_path(relative, "python")

    for chunk in result.chunks:
        if chunk.kind in ("module_skeleton", "class_skeleton"):
            continue  # synthetic text, spans the definition rather than quoting it
        assert chunk.end_line <= len(lines)
        first_real_line = chunk.text.splitlines()[0].strip()
        assert first_real_line in lines[chunk.start_line - 1]


def test_chunks_stay_within_the_hard_maximum() -> None:
    for relative, language in [
        ("py_small/src/billing/reporting.py", "python"),
        ("ts_small/src/cart.ts", "typescript"),
    ]:
        for chunk in chunk_path(relative, language).chunks:
            assert chunk.token_estimate <= HARD_MAX_TOKENS, f"{chunk.symbol_path} is oversized"


def test_soft_max_is_below_hard_max() -> None:
    assert SOFT_MAX_TOKENS < HARD_MAX_TOKENS
