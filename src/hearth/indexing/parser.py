"""Tree-sitter parsing, with grammars from pinned wheels and queries vendored in-repo.

Nothing is downloaded or compiled at runtime. Grammars come from wheels pinned in
``pyproject.toml``; tag queries are files in ``queries/<language>/tags.scm`` that ship with
the package. That is what makes symbol extraction deterministic, reviewable and offline
(docs/tech-stack.md §6.2).

Imports of ``tree_sitter`` and the grammar modules are **lazy**. They are slow to import
and are not needed on the chat path, and CLI startup has a 1.5s budget
(docs/system-design.md §15).

Parsers and compiled queries are cached per language: constructing them is expensive
relative to parsing one file, and indexing parses thousands.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hearth.indexing.languages import GRAMMAR_MODULES, is_parseable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tree_sitter import Language, Node, Parser, Query, Tree

_QUERIES_DIR = Path(__file__).parent / "queries"

#: language id -> (module name, attribute holding the PyCapsule)
#: Re-exported from `languages`, which owns the map so availability detection and
#: loading cannot disagree about which grammars exist.
_GRAMMAR_MODULES = GRAMMAR_MODULES


class ParserUnavailableError(RuntimeError):
    """No grammar is bundled for this language in this build."""


@dataclass(frozen=True)
class ParseResult:
    """A parsed file.

    ``status`` mirrors ``files.parse_status`` in the schema:

    * ``ok`` — parsed with no error nodes
    * ``partial`` — parsed, but the tree contains ERROR or MISSING nodes. Tree-sitter
      recovers from syntax errors, so a partial tree is still worth chunking; an unparsed
      file would simply vanish from the index.
    * ``failed`` — no tree at all
    """

    tree: Tree | None
    status: str
    error_count: int = 0

    @property
    def root(self) -> Node | None:
        return None if self.tree is None else self.tree.root_node

    @property
    def usable(self) -> bool:
        return self.tree is not None


@lru_cache(maxsize=8)
def get_language(language: str) -> Language:
    """Load and cache a tree-sitter ``Language``."""
    if language not in _GRAMMAR_MODULES:
        raise ParserUnavailableError(f"no bundled grammar for {language!r}")

    import importlib

    from tree_sitter import Language as TSLanguage

    module_name, attribute = _GRAMMAR_MODULES[language]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - a broken install
        raise ParserUnavailableError(f"grammar module {module_name!r} is not installed") from exc

    return TSLanguage(getattr(module, attribute)())


@lru_cache(maxsize=8)
def get_parser(language: str) -> Parser:
    """A cached parser for one language."""
    from tree_sitter import Parser as TSParser

    return TSParser(get_language(language))


def parse(source: bytes, language: str) -> ParseResult:
    """Parse source bytes. Never raises for malformed input.

    A file with syntax errors still yields a tree, because tree-sitter's error recovery is
    what lets the chunker index work-in-progress code — which is most of the code an agent
    is asked about.
    """
    if not is_parseable(language):
        return ParseResult(tree=None, status="skipped")

    try:
        parser = get_parser(language)
    except ParserUnavailableError:
        return ParseResult(tree=None, status="skipped")

    try:
        tree = parser.parse(source)
    except Exception:
        return ParseResult(tree=None, status="failed")

    errors = count_error_nodes(tree.root_node)
    return ParseResult(
        tree=tree,
        status="partial" if errors else "ok",
        error_count=errors,
    )


def count_error_nodes(node: Node) -> int:
    """Count ERROR and MISSING nodes.

    Used two ways: to set ``parse_status`` here, and by the edit engine's parse guard,
    which compares counts before and after an edit to detect newly introduced breakage
    (docs/safety-and-tool-use.md §7.2).
    """
    if not node.has_error:
        return 0

    total = 0
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "ERROR" or current.is_missing:
            total += 1
        stack.extend(child for child in current.children if child.has_error or child.is_missing)
    return total


@lru_cache(maxsize=16)
def load_query(language: str, name: str = "tags") -> Query | None:
    """Compile a vendored query file. None when the language has no query."""
    path = _QUERIES_DIR / language / f"{name}.scm"
    if not path.is_file():
        return None

    from tree_sitter import Query as TSQuery
    from tree_sitter import QueryError

    try:
        return TSQuery(get_language(language), path.read_text(encoding="utf-8"))
    except (QueryError, ParserUnavailableError, OSError):
        # A broken vendored query must not take down indexing; symbols are simply absent.
        return None


def run_query(query: Query, node: Node) -> dict[str, list[Node]]:
    """Execute a query, returning ``capture name -> nodes`` across all matches."""
    from tree_sitter import QueryCursor

    cursor = QueryCursor(query)
    captures: dict[str, list[Any]] = cursor.captures(node)
    return {name: list(nodes) for name, nodes in captures.items()}


def run_matches(query: Query, node: Node) -> list[dict[str, list[Node]]]:
    """Execute a query, returning one capture dict per match.

    Preferred over :func:`run_query` whenever captures must be *paired* — a definition
    with its name, an import with its module. A flat capture list loses which ``@name``
    belongs to which ``@definition``, and re-deriving that by position is guesswork.
    """
    from tree_sitter import QueryCursor

    cursor = QueryCursor(query)
    return [dict(captures) for _pattern_index, captures in cursor.matches(node)]


def node_text(node: Node, source: bytes) -> str:
    """Decode a node's source span."""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def available_languages() -> tuple[str, ...]:
    """Languages with both a bundled grammar and a vendored tags query."""
    return tuple(
        language for language in sorted(_GRAMMAR_MODULES) if (_QUERIES_DIR / language / "tags.scm").is_file()
    )
