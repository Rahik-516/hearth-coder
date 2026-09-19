"""Query analysis: deterministic, no LLM.

Extracts identifiers, paths, language hints and an intent label
(docs/system-design.md §7.1). Intent only ever adjusts *weights* — it never excludes a
retriever, because a misclassified query would then silently lose its best results.

LLM query rewriting is deliberately absent. It costs a full generation before retrieval
can even start, which on a local model is seconds, and the eval decides whether it is
worth enabling per tier.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from hearth.util.text import expand_identifier

#: Takes candidate identifiers, returns those that exist in the symbol table.
SymbolLookup = Callable[[list[str]], list[str]]

#: Identifier shapes worth checking against the symbol table.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

#: Anything with a separator or a known code extension.
_PATH_LIKE = re.compile(
    r"[\w./-]*/[\w./-]+|[\w-]+\.(?:py|pyi|ts|tsx|js|jsx|md|json|ya?ml|toml|sql|rs|go|java)\b"
)

_BACKTICKED = re.compile(r"`([^`]+)`")

#: Prefixes that signal intent, matched against whole words.
#:
#: Prefixes rather than an exact word list, because "where are totals *tested*" is a
#: question about tests and an exact-match list silently misses it — which then applies
#: the test-file *penalty* to exactly the query that wanted test files.
_TEST_PREFIXES = ("test", "spec", "fixture", "mock", "assert")
_DOC_PREFIXES = ("readme", "doc", "changelog", "guide", "tutorial")

_LANGUAGE_HINTS: dict[str, str] = {
    "python": "python",
    "py": "python",
    "typescript": "typescript",
    "ts": "typescript",
    "tsx": "tsx",
    "react": "tsx",
    "javascript": "javascript",
    "js": "javascript",
    "go": "go",
    "golang": "go",
    "rust": "rust",
    "java": "java",
    "markdown": "markdown",
}

#: Words too common to be useful identifiers on their own.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "do",
        "does",
        "did",
        "how",
        "what",
        "where",
        "when",
        "why",
        "which",
        "who",
        "we",
        "i",
        "it",
        "this",
        "that",
        "these",
        "those",
        "and",
        "or",
        "not",
        "of",
        "in",
        "on",
        "to",
        "for",
        "with",
        "from",
        "by",
        "at",
        "as",
        "if",
        "can",
        "could",
        "would",
        "should",
        "use",
        "used",
        "using",
        "get",
        "set",
        "code",
        "file",
        "files",
        "find",
    }
)


class Intent(StrEnum):
    SYMBOL = "symbol"
    PATH = "path"
    TEST = "test"
    DOCS = "docs"
    CONCEPTUAL = "conceptual"


@dataclass
class QueryAnalysis:
    """What a query appears to be asking for."""

    raw: str
    intent: Intent = Intent.CONCEPTUAL
    identifiers: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    #: Identifiers that actually exist in the symbol table. The strongest signal available
    #: without running a model.
    known_symbols: list[str] = field(default_factory=list)

    @property
    def is_symbol_query(self) -> bool:
        return self.intent is Intent.SYMBOL


def analyze(
    query: str,
    *,
    known_symbol_lookup: SymbolLookup | None = None,
) -> QueryAnalysis:
    """Analyse a query.

    ``known_symbol_lookup`` checks candidate identifiers against the index. Without it the
    analysis still works; it just cannot promote a query to ``symbol`` intent, and will
    treat "where is InvoiceService" as conceptual.
    """
    analysis = QueryAnalysis(raw=query)
    lowered = query.lower()
    words = set(re.findall(r"[a-z0-9_]+", lowered))

    analysis.paths = _extract_paths(query)
    analysis.identifiers = _extract_identifiers(query, analysis.paths)
    analysis.languages = sorted({_LANGUAGE_HINTS[w] for w in words if w in _LANGUAGE_HINTS})
    analysis.terms = _extract_terms(query)

    if known_symbol_lookup is not None and analysis.identifiers:
        analysis.known_symbols = known_symbol_lookup(analysis.identifiers)

    analysis.intent = _classify(analysis, words)
    return analysis


def _classify(analysis: QueryAnalysis, words: set[str]) -> Intent:
    """Assign intent. Order matters: a confirmed symbol beats a topical word.

    Checking symbols first means "where are the auth tests defined" with a real
    ``AuthTests`` symbol is a symbol query, not a test query — the user named a thing.
    """
    if analysis.known_symbols:
        return Intent.SYMBOL
    if analysis.paths:
        return Intent.PATH
    if any(word.startswith(_DOC_PREFIXES) for word in words):
        return Intent.DOCS
    if any(word.startswith(_TEST_PREFIXES) for word in words):
        return Intent.TEST
    return Intent.CONCEPTUAL


def _extract_paths(query: str) -> list[str]:
    found: list[str] = []
    for match in _PATH_LIKE.finditer(query):
        candidate = match.group(0).strip(".,;:()[]\"'")
        if candidate and candidate not in found:
            found.append(candidate)
    return found


def _extract_identifiers(query: str, paths: list[str]) -> list[str]:
    """Identifier-shaped tokens, excluding ones that are really paths.

    Backticked terms are always kept: someone writing `finalize` has told us it is a name.
    """
    found: list[str] = []

    for match in _BACKTICKED.finditer(query):
        token = match.group(1).strip()
        if token and token not in found:
            found.append(token)

    path_text = " ".join(paths)
    for match in _IDENTIFIER.finditer(query):
        token = match.group(0)
        if token in path_text:
            continue
        if token.lower() in _STOPWORDS:
            continue
        # A bare lowercase word is only interesting if it looks like code: snake_case,
        # a dotted path, or mixed case. Otherwise it is prose.
        if token.islower() and "_" not in token and "." not in token:
            continue
        if token not in found:
            found.append(token)
    return found


def _extract_terms(query: str) -> list[str]:
    """Expanded search terms, matching how documents were indexed."""
    terms: list[str] = []
    for word in re.findall(r"[A-Za-z0-9_.]+", query):
        if word.lower() in _STOPWORDS:
            continue
        for form in expand_identifier(word):
            if form and form not in terms and form not in _STOPWORDS:
                terms.append(form)
    return terms
