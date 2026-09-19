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

#: Languages with a grammar wheel bundled in this build (docs/tech-stack.md §19.1).
GRAMMAR_AVAILABLE: frozenset[str] = frozenset({"python", "typescript", "tsx", "javascript"})


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


def support_level(language: str | None) -> SupportLevel:
    """How deeply a language can be analysed in this build.

    A language whose grammar is not bundled degrades to FALLBACK, so an unpulled grammar
    produces window chunks rather than an error.
    """
    if language is None:
        return SupportLevel.FALLBACK

    level = _SUPPORT.get(language, SupportLevel.FALLBACK)
    if level in (SupportLevel.FULL, SupportLevel.STRUCTURAL) and language not in GRAMMAR_AVAILABLE:
        return SupportLevel.FALLBACK
    return level


def is_parseable(language: str | None) -> bool:
    """Whether tree-sitter can parse this language in this build."""
    return language in GRAMMAR_AVAILABLE


def known_languages() -> set[str]:
    return set(_SUPPORT)
