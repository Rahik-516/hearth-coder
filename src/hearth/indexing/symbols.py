"""Symbol, reference and import extraction from a parsed tree.

This builds a **name-based** graph: approximate, but language-agnostic, cheap, and good
enough for "who calls X" and for ranking the repo map. Precise semantic references need a
language server and are a Phase 3 addition (docs/system-design.md §6.6).

Being honest about the approximation matters. Two different ``save`` methods in unrelated
classes are one name here. That is a known limit, recorded in the revisit triggers
(docs/system-design.md §18), not a bug to be surprised by later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hearth.indexing.parser import load_query, node_text, run_matches

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tree_sitter import Node

#: Definition capture prefix -> symbol kind stored in the schema.
_KIND_BY_CAPTURE: dict[str, str] = {
    "definition.class": "class",
    "definition.function": "function",
    "definition.method": "method",
    "definition.interface": "interface",
    "definition.type": "type",
    "definition.constant": "const",
    "definition.variable": "var",
    "definition.module": "module",
}

#: Node types that introduce a class-like scope, so functions inside become methods.
_CLASS_LIKE = frozenset(
    {
        "class_definition",
        "class_declaration",
        "class",
        "interface_declaration",
        "object_type",
    }
)

#: How much of a definition to keep as its signature, in characters.
_MAX_SIGNATURE_CHARS = 300


@dataclass
class Definition:
    """One extracted definition."""

    name: str
    kind: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    signature: str | None = None
    exported: bool | None = None
    parent: Definition | None = None

    @property
    def symbol_path(self) -> str:
        """Dotted-ish path used for display and FTS, e.g. ``InvoiceService > finalize``."""
        parts = [self.name]
        current = self.parent
        while current is not None:
            parts.append(current.name)
            current = current.parent
        return " > ".join(reversed(parts))


@dataclass
class Reference:
    name: str
    line: int
    kind: str


@dataclass
class Import:
    module_spec: str
    names: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    definitions: list[Definition] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.definitions or self.references or self.imports)


def extract(root: Node, source: bytes, language: str) -> ExtractionResult:
    """Run the vendored tags query and assemble symbols, refs and imports.

    Returns an empty result rather than raising when the language has no query, so an
    unsupported language degrades to text chunks instead of failing the file.
    """
    query = load_query(language)
    if query is None:
        return ExtractionResult()

    definitions: list[Definition] = []
    references: list[Reference] = []
    imports: dict[str, Import] = {}

    for match in run_matches(query, root):
        _collect_definition(match, source, definitions)
        _collect_reference(match, source, references)
        _collect_import(match, source, imports)

    definitions.sort(key=lambda d: (d.start_byte, -d.end_byte))
    _assign_parents(definitions)
    _refine_kinds(definitions)

    return ExtractionResult(
        definitions=definitions,
        references=_dedupe_references(references),
        imports=list(imports.values()),
    )


def _collect_definition(match: dict[str, list[Node]], source: bytes, out: list[Definition]) -> None:
    capture = next((key for key in match if key.startswith("definition.")), None)
    if capture is None:
        return

    name_nodes = match.get("name")
    if not name_nodes:
        return

    node = match[capture][0]
    name_node = name_nodes[0]
    name = node_text(name_node, source)
    if not name:
        return

    out.append(
        Definition(
            name=name,
            kind=_KIND_BY_CAPTURE.get(capture, "var"),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            signature=_signature(node, source),
            exported=_is_exported(name),
        )
    )


def _collect_reference(match: dict[str, list[Node]], source: bytes, out: list[Reference]) -> None:
    capture = next((key for key in match if key.startswith("reference.")), None)
    if capture is None:
        return

    for name_node in match.get("name", []):
        name = node_text(name_node, source)
        if name:
            out.append(
                Reference(
                    name=name,
                    line=name_node.start_point[0] + 1,
                    kind=capture.split(".", 1)[1],
                )
            )


def _collect_import(match: dict[str, list[Node]], source: bytes, out: dict[str, Import]) -> None:
    """Record one import.

    Every match carries its own module, so names cannot drift onto a neighbouring import.
    Repeated ``from x import ...`` lines merge into one entry per module spec.
    """
    module_nodes = match.get("import.module")
    if not module_nodes:
        return

    spec = _unquote(node_text(module_nodes[0], source))
    if not spec:
        return

    entry = out.setdefault(spec, Import(module_spec=spec))
    for node in match.get("import.name", []):
        name = node_text(node, source)
        if name and name not in entry.names:
            entry.names.append(name)


def _assign_parents(definitions: list[Definition]) -> None:
    """Nest definitions by byte containment.

    The list is pre-sorted by start byte with wider spans first, so a simple stack gives
    correct nesting for classes inside classes and closures inside functions.
    """
    stack: list[Definition] = []
    for definition in definitions:
        while stack and definition.start_byte >= stack[-1].end_byte:
            stack.pop()
        definition.parent = stack[-1] if stack else None
        stack.append(definition)


def _refine_kinds(definitions: list[Definition]) -> None:
    """A function directly inside a class is a method."""
    for definition in definitions:
        if definition.kind != "function":
            continue
        parent = definition.parent
        if parent is not None and parent.kind in ("class", "interface"):
            definition.kind = "method"


def _signature(node: Node, source: bytes) -> str | None:
    """Text from the definition's start up to its body.

    Keeping the signature but not the body is what makes class skeletons useful: they
    answer "what's in this file" at a fraction of the token cost.
    """
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    raw = source[node.start_byte : end].decode("utf-8", errors="replace").strip()

    if not raw:
        return None
    collapsed = " ".join(raw.split())
    if len(collapsed) > _MAX_SIGNATURE_CHARS:
        return collapsed[: _MAX_SIGNATURE_CHARS - 1] + "…"
    return collapsed


def _unquote(text: str) -> str:
    """Strip the quotes a JS/TS module specifier carries as a string literal.

    Python module specs arrive bare, so this is a no-op for them. Storing ``"./types"``
    with quotes would break import resolution against file paths later.
    """
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'`":
        return stripped[1:-1]
    return stripped


def _is_exported(name: str) -> bool:
    """Python convention: a leading underscore means private.

    TypeScript and JavaScript use explicit `export`, handled by their own queries; this
    default is a reasonable approximation until those captures exist.
    """
    return not name.startswith("_")


def _dedupe_references(references: list[Reference]) -> list[Reference]:
    """Collapse identical (name, line, kind) triples produced by overlapping patterns."""
    seen: set[tuple[str, int, str]] = set()
    unique: list[Reference] = []
    for ref in references:
        key = (ref.name, ref.line, ref.kind)
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    return unique
