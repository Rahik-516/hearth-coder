"""Language detection and support levels.

Support level decides what the indexer can extract, not merely what it recognises
(docs/system-design.md §6.3):

* **FULL** — AST chunks, symbols, references, imports, signatures, skeletons.
* **STRUCTURAL** — AST chunks, symbols, signatures. No reference graph.
* **DOCUMENT** — heading/section chunks.
* **DATA** — whole file when small; structured chunks in Phase 2.
* **FALLBACK** — blank-line-aware sliding window. Everything else.

MVP ships FULL for Python, TypeScript/TSX and JavaScript/JSX. A language listed below at
FULL without a bundled grammar degrades to FALLBACK at runtime rather than failing.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import PurePosixPath


class SupportLevel(StrEnum):
    FULL = "full"
    STRUCTURAL = "structural"
    DOCUMENT = "document"
    DATA = "data"
    FALLBACK = "fallback"


#: Extension -> language id. Lowercase, leading dot included.
_BY_EXTENSION: dict[str, str] = {
    # Full (MVP)
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    # Full (Phase 2)
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    # Structural (Phase 2)
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    # Documents
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".txt": "text",
    # Data / config
    ".json": "json",
    ".jsonc": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".sql": "sql",
    ".tf": "hcl",
    ".tfvars": "hcl",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
}

#: Exact filenames that imply a language regardless of extension.
_BY_FILENAME: dict[str, str] = {
    "dockerfile": "dockerfile",
    "containerfile": "dockerfile",
    "makefile": "make",
    "justfile": "just",
    "gemfile": "ruby",
    "rakefile": "ruby",
    "pyproject.toml": "toml",
    "cargo.toml": "toml",
    ".gitignore": "text",
    ".hearthignore": "text",
    "readme": "markdown",
    "license": "text",
}

#: Interpreter name found in a shebang -> language.
_BY_SHEBANG: dict[str, str] = {
    "python": "python",
    "python3": "python",
    "node": "javascript",
    "bash": "bash",
    "sh": "bash",
    "zsh": "bash",
    "ruby": "ruby",
    "php": "php",
}

_SUPPORT: dict[str, SupportLevel] = {
    "python": SupportLevel.FULL,
    "typescript": SupportLevel.FULL,
    "tsx": SupportLevel.FULL,
    "javascript": SupportLevel.FULL,
    "go": SupportLevel.FULL,
    "rust": SupportLevel.FULL,
    "java": SupportLevel.FULL,
    "c": SupportLevel.STRUCTURAL,
    "cpp": SupportLevel.STRUCTURAL,
    "csharp": SupportLevel.STRUCTURAL,
    "ruby": SupportLevel.STRUCTURAL,
    "php": SupportLevel.STRUCTURAL,
    "kotlin": SupportLevel.STRUCTURAL,
    "swift": SupportLevel.STRUCTURAL,
    "bash": SupportLevel.STRUCTURAL,
    "markdown": SupportLevel.DOCUMENT,
    "rst": SupportLevel.DOCUMENT,
    "text": SupportLevel.DOCUMENT,
    "json": SupportLevel.DATA,
    "yaml": SupportLevel.DATA,
    "toml": SupportLevel.DATA,
    "ini": SupportLevel.DATA,
    "sql": SupportLevel.DATA,
    "hcl": SupportLevel.DATA,
    "xml": SupportLevel.DATA,
    "html": SupportLevel.DATA,
    "css": SupportLevel.DATA,
    "dockerfile": SupportLevel.DATA,
    "make": SupportLevel.DATA,
    "just": SupportLevel.DATA,
}

#: Language -> (module, factory attribute) for every grammar Hearth knows how to load.
#: The single source of truth: ``parser.py`` loads through it, and availability is derived
#: from it rather than tracked in a second list that can disagree.
GRAMMAR_MODULES: dict[str, tuple[str, str]] = {
    # Always installed (docs/tech-stack.md §19.1).
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    # The `langs-extra` extra. Absent in a default install, which is why availability is
    # detected rather than declared.
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
}


@lru_cache(maxsize=1)
def grammar_available() -> frozenset[str]:
    """Languages whose grammar wheel is installed in *this* environment.

    Detected with ``find_spec`` rather than by importing: importing seven grammar modules
    to answer "which are present" would cost startup time on every CLI path, and the
    optional ones are absent in a default install, where the import would simply fail.

    Cached, since the answer cannot change within a process.
    """
    from importlib.util import find_spec

    present = set()
    for language, (module, _) in GRAMMAR_MODULES.items():
        try:
            if find_spec(module) is not None:
                present.add(language)
        except (ImportError, ValueError):
            # A broken or partially installed distribution is "not available", not a
            # crash: the language degrades to the fallback chunker like any other.
            continue
    return frozenset(present)


def detect_language(path: str, *, first_line: str | None = None) -> str | None:
    """Identify a file's language from its path, falling back to a shebang.

    Filename wins over extension, because ``Dockerfile`` and ``Makefile`` have none, and
    ``pyproject.toml`` is more usefully TOML than whatever a future ``.toml`` rule says.
    """
    name = PurePosixPath(path).name
    lowered = name.lower()

    if lowered in _BY_FILENAME:
        return _BY_FILENAME[lowered]

    suffix = PurePosixPath(lowered).suffix
    if suffix in _BY_EXTENSION:
        return _BY_EXTENSION[suffix]

    # Dockerfile.dev, Dockerfile.prod
    if lowered.startswith("dockerfile"):
        return "dockerfile"

    if first_line:
        interpreter = _interpreter_from_shebang(first_line)
        if interpreter:
            return interpreter

    return None


def _interpreter_from_shebang(first_line: str) -> str | None:
    """Parse ``#!/usr/bin/env python3`` and friends."""
    if not first_line.startswith("#!"):
        return None

    parts = first_line[2:].strip().split()
    if not parts:
        return None

    candidates = [PurePosixPath(parts[0]).name]
    if candidates[0] == "env" and len(parts) > 1:
        candidates.append(PurePosixPath(parts[1]).name)

    for candidate in reversed(candidates):
        stripped = candidate.rstrip("0123456789.")
        for key in (candidate, stripped):
            if key in _BY_SHEBANG:
                return _BY_SHEBANG[key]
    return None


def declared_support_level(language: str | None) -> SupportLevel:
    """The level a language is declared at, ignoring whether its grammar is installed.

    :func:`support_level` deliberately reports FALLBACK when the grammar is missing, which
    is what the indexer needs and the wrong answer for anything asking *which languages
    have been downgraded* — that caller would be told none of them had.
    """
    if language is None:
        return SupportLevel.FALLBACK
    return _SUPPORT.get(language, SupportLevel.FALLBACK)


def support_level(language: str | None) -> SupportLevel:
    """How deeply a language can be analysed in this build.

    A language whose grammar is not bundled degrades to FALLBACK, so an unpulled grammar
    produces window chunks rather than an error.
    """
    if language is None:
        return SupportLevel.FALLBACK

    level = _SUPPORT.get(language, SupportLevel.FALLBACK)
    if level in (SupportLevel.FULL, SupportLevel.STRUCTURAL) and language not in grammar_available():
        return SupportLevel.FALLBACK
    return level


def is_parseable(language: str | None) -> bool:
    """Whether tree-sitter can parse this language in this build."""
    return language in grammar_available()


def known_languages() -> set[str]:
    return set(_SUPPORT)
