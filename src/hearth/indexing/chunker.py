"""Splitting files into retrievable chunks.

The algorithm is in docs/system-design.md §6.4. Three things about it are load-bearing:

* **Size is measured in non-whitespace characters**, then converted to estimated tokens.
  Counting raw characters would make an indented Python method look twice the size of the
  same code at module level, and chunk boundaries would shift with formatting.
* **AST chunks do not overlap.** Their boundaries are semantic, so overlap would only
  duplicate tokens. Only the fallback window chunker overlaps.
* **A file that fails to parse still chunks**, via the window path. An unparsed file would
  silently vanish from the index, which is far worse than imprecise boundaries — and
  work-in-progress code is exactly what an agent gets asked about.

Skeletons (module-level and per class) are what answer "what is in this file?" cheaply.
They carry signatures with bodies elided, so they cost a fraction of the tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hearth.indexing.parser import ParseResult, node_text
from hearth.indexing.symbols import Definition, ExtractionResult
from hearth.util.text import count_lines

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tree_sitter import Node

#: Target chunk size in estimated tokens (docs/system-design.md §6.4).
TARGET_TOKENS = 400
SOFT_MAX_TOKENS = 800
HARD_MAX_TOKENS = 1500
MIN_TOKENS = 40

#: Characters per token, for the cheap pre-tokenizer estimate. Code is denser than prose;
#: the calibrated estimator in core/context refines this at request time.
_CHARS_PER_TOKEN = 3.5

#: Fallback window sizing, in lines.
_WINDOW_LINES = 60
_WINDOW_OVERLAP_LINES = 9  # ~15%

#: Node types that are definitions worth emitting as their own chunk.
_DEFINITION_NODES = frozenset(
    {
        "function_definition",
        "class_definition",
        "decorated_definition",
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "abstract_class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "method_definition",
        "lexical_declaration",
        "export_statement",
    }
)

_CLASS_NODES = frozenset(
    {"class_definition", "class_declaration", "abstract_class_declaration", "interface_declaration"}
)


@dataclass
class Chunk:
    """One indexable unit of a file."""

    kind: str
    text: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    symbol_path: str | None = None
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if not self.token_estimate:
            self.token_estimate = estimate_tokens(self.text)


@dataclass
class ChunkingResult:
    chunks: list[Chunk] = field(default_factory=list)
    strategy: str = "ast"

    @property
    def total_tokens(self) -> int:
        return sum(c.token_estimate for c in self.chunks)


def estimate_tokens(text: str) -> int:
    """Estimate tokens from non-whitespace character count.

    Whitespace-insensitive by design: the same function indented inside a class must not
    be measured as larger than at module level, or chunk boundaries would move with
    formatting changes.
    """
    dense = sum(1 for char in text if not char.isspace())
    return max(1, round(dense / _CHARS_PER_TOKEN))


def chunk_file(
    *,
    source: bytes,
    text: str,
    language: str | None,
    parse_result: ParseResult | None,
    extraction: ExtractionResult | None = None,
) -> ChunkingResult:
    """Chunk a file, choosing a strategy by what is actually available.

    Falls back to windows whenever there is no usable tree — an unparseable file, or a
    language with no bundled grammar.
    """
    if language == "markdown":
        return ChunkingResult(chunks=_chunk_markdown(text), strategy="markdown")

    if language in _STRUCTURED_LANGUAGES and estimate_tokens(text) >= STRUCTURED_MIN_TOKENS:
        structured = chunk_structured(text, language)
        if structured:
            return ChunkingResult(chunks=structured, strategy="structured")

    if parse_result is None or not parse_result.usable or parse_result.root is None:
        return ChunkingResult(chunks=_chunk_windows(text), strategy="window")

    chunks = _chunk_ast(parse_result.root, source, extraction)
    if not chunks:
        return ChunkingResult(chunks=_chunk_windows(text), strategy="window")

    skeleton = _module_skeleton(extraction, text)
    if skeleton is not None:
        chunks.insert(0, skeleton)

    return ChunkingResult(chunks=chunks, strategy="ast")


# ----------------------------------------------------------------------- AST


def _chunk_ast(root: Node, source: bytes, extraction: ExtractionResult | None) -> list[Chunk]:
    """Walk top-level nodes, emitting definitions and merging the rest into preambles."""
    by_span = _definitions_by_span(extraction)
    chunks: list[Chunk] = []
    pending: list[Node] = []

    for child in root.named_children:
        if _is_definition(child):
            chunks.extend(_flush_preamble(pending, source))
            pending = []
            chunks.extend(_chunk_definition(child, source, by_span))
        else:
            pending.append(child)

    chunks.extend(_flush_preamble(pending, source))
    return [c for c in chunks if c.text.strip()]


def _chunk_definition(node: Node, source: bytes, by_span: dict[tuple[int, int], Definition]) -> list[Chunk]:
    """Emit one definition, splitting it when it exceeds the soft maximum."""
    text = node_text(node, source)
    tokens = estimate_tokens(text)
    definition = _lookup(node, by_span)
    kind = _chunk_kind(node, definition)

    if tokens <= SOFT_MAX_TOKENS:
        return [_make_chunk(node, source, kind, definition)]

    chunks: list[Chunk] = []

    # A class that is too large becomes a skeleton plus its members, so the class's shape
    # stays retrievable even though its body is split up.
    if node.type in _CLASS_NODES:
        skeleton = _class_skeleton(node, source, definition)
        if skeleton is not None:
            chunks.append(skeleton)

    body = node.child_by_field_name("body")
    members = list(body.named_children) if body is not None else []

    if not members:
        return [_make_chunk(node, source, kind, definition)]

    for group in _greedy_merge(members, source):
        if len(group) == 1 and estimate_tokens(node_text(group[0], source)) > SOFT_MAX_TOKENS:
            chunks.extend(_chunk_definition(group[0], source, by_span))
            continue
        chunks.append(_make_group_chunk(group, source, by_span))

    return chunks or [_make_chunk(node, source, kind, definition)]


def _greedy_merge(nodes: list[Node], source: bytes) -> list[list[Node]]:
    """Group consecutive siblings up to the target size.

    Merging tiny siblings matters: a class of one-line methods would otherwise produce
    dozens of chunks too small to carry meaning, each costing a retrieval slot.
    """
    groups: list[list[Node]] = []
    current: list[Node] = []
    current_tokens = 0

    for node in nodes:
        tokens = estimate_tokens(node_text(node, source))

        if current and current_tokens + tokens > TARGET_TOKENS:
            groups.append(current)
            current = [node]
            current_tokens = tokens
        else:
            current.append(node)
            current_tokens += tokens

    if current:
        groups.append(current)
    return groups


def _flush_preamble(nodes: list[Node], source: bytes) -> list[Chunk]:
    """Merge consecutive non-definition siblings into preamble chunks.

    This is where imports, module constants and configuration end up — the part of a file
    that answers "what does this module depend on?".
    """
    if not nodes:
        return []

    chunks: list[Chunk] = []
    for group in _greedy_merge(nodes, source):
        chunks.append(_make_group_chunk(group, source, {}, kind="preamble"))
    return chunks


def _make_chunk(node: Node, source: bytes, kind: str, definition: Definition | None) -> Chunk:
    return Chunk(
        kind=kind,
        text=node_text(node, source),
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        symbol_path=definition.symbol_path if definition else None,
    )


def _make_group_chunk(
    nodes: list[Node],
    source: bytes,
    by_span: dict[tuple[int, int], Definition],
    *,
    kind: str | None = None,
) -> Chunk:
    first, last = nodes[0], nodes[-1]
    definition = _lookup(first, by_span) if len(nodes) == 1 else None
    resolved_kind = kind or (_chunk_kind(first, definition) if len(nodes) == 1 else "preamble")

    return Chunk(
        kind=resolved_kind,
        text=source[first.start_byte : last.end_byte].decode("utf-8", errors="replace"),
        start_line=first.start_point[0] + 1,
        end_line=last.end_point[0] + 1,
        start_byte=first.start_byte,
        end_byte=last.end_byte,
        symbol_path=definition.symbol_path if definition else None,
    )


def _chunk_kind(node: Node, definition: Definition | None) -> str:
    if definition is not None:
        return definition.kind if definition.kind in ("class", "method", "function") else "preamble"
    if node.type in _CLASS_NODES:
        return "class"
    if "function" in node.type or "method" in node.type:
        return "function"
    return "preamble"


# ------------------------------------------------------------------ skeletons


def _module_skeleton(extraction: ExtractionResult | None, text: str) -> Chunk | None:
    """Imports plus top-level signatures, bodies elided.

    Answers "what's in this file?" in a handful of tokens, and feeds the repo map.
    """
    if extraction is None or not extraction.definitions:
        return None

    lines: list[str] = []
    for imp in extraction.imports[:40]:
        names = ", ".join(imp.names[:8])
        lines.append(f"import {imp.module_spec}" + (f" ({names})" if names else ""))

    if lines:
        lines.append("")

    for definition in extraction.definitions:
        if definition.parent is not None:
            continue
        indent = ""
        lines.append(f"{indent}{definition.signature or definition.name}")
        for child in extraction.definitions:
            if child.parent is definition and child.signature:
                lines.append(f"    {child.signature}")

    body = "\n".join(lines).strip()
    if not body:
        return None

    return Chunk(
        kind="module_skeleton",
        text=body,
        start_line=1,
        # count_lines, not `count("\n") + 1`: a file ending in a newline has N lines, not
        # N+1. The naive form produced an end_line one past the end of every such file,
        # which is a citation that does not resolve.
        end_line=max(1, count_lines(text)),
        start_byte=0,
        end_byte=len(text.encode("utf-8")),
        symbol_path=None,
    )


def _class_skeleton(node: Node, source: bytes, definition: Definition | None) -> Chunk | None:
    """A class header plus its member signatures, bodies elided."""
    body = node.child_by_field_name("body")
    if body is None:
        return None

    header = source[node.start_byte : body.start_byte].decode("utf-8", errors="replace").strip()
    lines = [header]

    for member in body.named_children:
        member_body = member.child_by_field_name("body")
        end = member_body.start_byte if member_body is not None else member.end_byte
        signature = source[member.start_byte : end].decode("utf-8", errors="replace").strip()
        if signature:
            lines.append("    " + " ".join(signature.split()))

    if len(lines) <= 1:
        return None

    return Chunk(
        kind="class_skeleton",
        text="\n".join(lines),
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        symbol_path=definition.symbol_path if definition else None,
    )


# ------------------------------------------------------------------ documents


def _chunk_markdown(text: str) -> list[Chunk]:
    """Split on headings, so a section stays whole.

    Headings are the document's own structure; splitting anywhere else would cut a section
    from its title, which is usually the part that makes it findable.
    """
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    start_line = 1
    heading: str | None = None
    offset = 0
    start_byte = 0

    def flush(end_line: int, end_byte: int) -> None:
        body = "\n".join(current).strip()
        if body:
            chunks.append(
                Chunk(
                    kind="section",
                    text=body,
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=start_byte,
                    end_byte=end_byte,
                    symbol_path=heading,
                )
            )

    for index, line in enumerate(lines, start=1):
        line_bytes = len(line.encode("utf-8")) + 1
        if line.startswith("#") and current:
            flush(index - 1, offset)
            current = []
            start_line = index
            start_byte = offset
        if line.startswith("#"):
            heading = line.lstrip("#").strip() or heading
        current.append(line)
        offset += line_bytes

    flush(len(lines), offset)
    return _split_oversized(chunks)


def _split_oversized(chunks: list[Chunk]) -> list[Chunk]:
    """Break any chunk past the hard maximum into windows."""
    result: list[Chunk] = []
    for chunk in chunks:
        if chunk.token_estimate <= HARD_MAX_TOKENS:
            result.append(chunk)
            continue
        for window in _chunk_windows(chunk.text, start_line_offset=chunk.start_line - 1):
            window.symbol_path = chunk.symbol_path
            result.append(window)
    return result


# ------------------------------------------------------------------- windows


def _chunk_windows(text: str, *, start_line_offset: int = 0) -> list[Chunk]:
    """Blank-line-aware sliding window with overlap.

    The last resort: no grammar, or a tree we could not use. Overlap exists here (unlike
    AST chunks) because the boundaries are arbitrary, so a definition split across two
    windows should appear whole in at least one.
    """
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    index = 0
    byte_offset = 0

    while index < len(lines):
        end = min(index + _WINDOW_LINES, len(lines))
        end = _extend_to_blank_line(lines, end)

        window = lines[index:end]
        body = "\n".join(window)
        size = len(body.encode("utf-8"))

        if body.strip():
            chunks.append(
                Chunk(
                    kind="window",
                    text=body,
                    start_line=index + 1 + start_line_offset,
                    end_line=end + start_line_offset,
                    start_byte=byte_offset,
                    end_byte=byte_offset + size,
                )
            )

        if end >= len(lines):
            break

        step = max(1, (end - index) - _WINDOW_OVERLAP_LINES)
        byte_offset += len("\n".join(lines[index : index + step]).encode("utf-8")) + 1
        index += step

    return chunks


def _extend_to_blank_line(lines: list[str], end: int, *, look_ahead: int = 8) -> int:
    """Nudge a window boundary to the next blank line, so it lands between blocks."""
    limit = min(end + look_ahead, len(lines))
    for candidate in range(end, limit):
        if not lines[candidate].strip():
            return candidate
    return end


# ------------------------------------------------------------------- helpers


def _is_definition(node: Node) -> bool:
    return node.type in _DEFINITION_NODES


def _definitions_by_span(
    extraction: ExtractionResult | None,
) -> dict[tuple[int, int], Definition]:
    if extraction is None:
        return {}
    return {(d.start_byte, d.end_byte): d for d in extraction.definitions}


def _lookup(node: Node, by_span: dict[tuple[int, int], Definition]) -> Definition | None:
    """Find the definition matching a node's span.

    Decorated definitions are a special case: the query captures the inner definition
    while the chunker walks the outer ``decorated_definition``, so a containment search
    recovers the pairing.
    """
    exact = by_span.get((node.start_byte, node.end_byte))
    if exact is not None:
        return exact

    for (start, end), definition in by_span.items():
        if node.start_byte <= start and end <= node.end_byte:
            return definition
    return None


# ------------------------------------------------------------------ structured data


#: Data languages whose own syntax marks section boundaries.
_STRUCTURED_LANGUAGES = frozenset({"json", "yaml", "toml"})

#: Below this, a config file is more useful whole: the top of a `pyproject.toml` explains
#: the bottom, and splitting a 40-line file into six fragments loses that for nothing.
STRUCTURED_MIN_TOKENS = 300

#: A TOML table header at column zero: `[tool.ruff]` or `[[profile]]`.
_TOML_SECTION = re.compile(r"^\[\[?[^\]\s]+\]?\]\s*(?:#.*)?$")

#: A YAML mapping key at column zero. Anchored there on purpose — an indented key belongs
#: to the block above it, and splitting on one would cut a value from the key that names it.
_YAML_TOP_KEY = re.compile(r"^(?:\"[^\"]+\"|'[^']+'|[A-Za-z0-9_.\-]+)\s*:(?:\s|$)")


def chunk_structured(text: str, language: str) -> list[Chunk]:
    """Split a data file on its own top-level structure.

    JSON, YAML and TOML have no bundled grammar, so they would otherwise land in the
    sliding-window chunker, which cuts at arbitrary line counts: a window can begin in the
    middle of one table and end in the middle of the next, and retrieving it tells the
    model neither which section it is reading nor where the section ends.

    Splitting is **textual**, not parsed, and that is a deliberate trade. ``json`` and
    ``tomllib`` report no line numbers, and PyYAML is not a runtime dependency (rule 7), so
    a parse would buy correctness the chunker cannot use — every chunk needs a line range
    to be citable. The cost is that a top-level key inside a multi-line string can be
    mistaken for a boundary; the result is a split in a slightly wrong place, which is what
    the window chunker does everywhere by definition.

    Returns an empty list when the file has no structure worth splitting on, so the caller
    can fall back.
    """
    lines = text.splitlines()
    if not lines:
        return []

    boundaries = _structure_boundaries(lines, language)
    if len(boundaries) < 2:
        # One section is not a split. Returning nothing lets the caller keep the file
        # whole, which is the right answer for a flat config.
        return []

    chunks: list[Chunk] = []
    byte_offsets = _line_byte_offsets(lines)

    # Anything before the first boundary — a licence header, a JSON file's opening brace,
    # TOML keys outside any table — belongs to no section and would otherwise be dropped
    # from the index entirely. Silently losing content is worse than an extra chunk.
    if boundaries[0][0] > 0:
        preamble = "\n".join(lines[: boundaries[0][0]]).strip()
        if preamble:
            chunks.append(
                Chunk(
                    kind="preamble",
                    text=preamble,
                    start_line=1,
                    end_line=boundaries[0][0],
                    start_byte=0,
                    end_byte=byte_offsets[boundaries[0][0]],
                    symbol_path=None,
                )
            )

    for position, (start_index, label) in enumerate(boundaries):
        end_index = boundaries[position + 1][0] if position + 1 < len(boundaries) else len(lines)
        body = "\n".join(lines[start_index:end_index]).strip()
        if not body:
            continue

        chunks.append(
            Chunk(
                kind="section",
                text=body,
                start_line=start_index + 1,
                end_line=end_index,
                start_byte=byte_offsets[start_index],
                end_byte=byte_offsets[end_index - 1] + len(lines[end_index - 1].encode("utf-8")),
                symbol_path=label,
            )
        )

    return chunks


def _structure_boundaries(lines: list[str], language: str) -> list[tuple[int, str]]:
    """``(line index, label)`` for each top-level section, in order."""
    if language == "toml":
        return [
            (index, line.strip().strip("[]"))
            for index, line in enumerate(lines)
            if _TOML_SECTION.match(line)
        ]

    if language == "yaml":
        found: list[tuple[int, str]] = []
        for index, line in enumerate(lines):
            # `---` starts a new document; treat it as a boundary so multi-document files
            # do not merge into one chunk.
            if line.rstrip() == "---":
                found.append((index, "document"))
            elif _YAML_TOP_KEY.match(line):
                found.append((index, line.split(":", 1)[0].strip().strip("\"'")))
        return found

    if language == "json":
        return _json_boundaries(lines)

    return []


def _json_boundaries(lines: list[str]) -> list[tuple[int, str]]:
    """Keys at depth 1 of a JSON object, found by tracking brace depth.

    Depth is counted outside strings, so a brace inside a value cannot shift it. Escapes
    are honoured for the same reason — `"a\\""` must not be read as an unterminated string,
    which would make every following line look like it was inside one.
    """
    found: list[tuple[int, str]] = []
    depth = 0
    in_string = False
    escaped = False

    for index, line in enumerate(lines):
        line_start_depth = depth
        key: str | None = None
        buffer: list[str] = []

        for char in line:
            if escaped:
                escaped = False
                if in_string:
                    buffer.append(char)
                continue
            if char == "\\" and in_string:
                escaped = True
                continue
            if char == '"':
                if in_string and line_start_depth == 1 and key is None:
                    key = "".join(buffer)
                in_string = not in_string
                buffer = []
                continue
            if in_string:
                buffer.append(char)
            elif char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1

        if key is not None and line_start_depth == 1:
            found.append((index, key))

    return found


def _line_byte_offsets(lines: list[str]) -> list[int]:
    """Byte offset of each line's start, for chunk spans."""
    offsets: list[int] = []
    running = 0
    for line in lines:
        offsets.append(running)
        running += len(line.encode("utf-8")) + 1  # +1 for the newline
    offsets.append(running)
    return offsets
