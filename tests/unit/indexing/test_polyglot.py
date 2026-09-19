"""Go, Rust and Java — the I1 language expansion.

These grammars ship in the optional ``langs-extra`` extra, so every test here skips when
they are absent rather than failing. That is the same contract the indexer honours: a
missing grammar degrades the language to the fallback chunker, it does not break indexing.

What is asserted is the *capture contract* the symbol graph depends on — definitions with
their names, references, imports — rather than exact node counts, which would break on
every grammar bump for no useful reason.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hearth.indexing.languages import SupportLevel, grammar_available, support_level
from hearth.indexing.parser import load_query, parse, run_matches
from hearth.indexing.pipeline import Indexer
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "repos" / "polyglot"

LANGUAGES = ("go", "rust", "java")

SOURCES = {"go": "src/server.go", "rust": "src/cache.rs", "java": "src/Invoice.java"}


def require(language: str) -> None:
    if language not in grammar_available():
        pytest.skip(f"{language} grammar not installed (uv sync --extra langs-extra)")


def captures_for(language: str) -> dict[str, list[str]]:
    """Capture name -> the source text of each captured node."""
    require(language)
    source = (FIXTURE / SOURCES[language]).read_bytes()
    result = parse(source, language)
    query = load_query(language)
    assert query is not None, f"no tags query for {language}"
    assert result.usable and result.root is not None

    found: dict[str, list[str]] = {}
    for match in run_matches(query, result.root):
        for name, nodes in match.items():
            for node in nodes:
                text = source[node.start_byte : node.end_byte].decode("utf-8", "replace")
                found.setdefault(name, []).append(text)
    return found


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_language_is_full_support_when_its_grammar_is_present(language: str) -> None:
    require(language)

    assert support_level(language) is SupportLevel.FULL


@pytest.mark.parametrize("language", LANGUAGES)
def test_a_tags_query_compiles(language: str) -> None:
    """A query that fails to compile returns None and silently costs every symbol."""
    require(language)

    assert load_query(language) is not None


@pytest.mark.parametrize("language", LANGUAGES)
def test_definitions_and_names_are_captured(language: str) -> None:
    captures = captures_for(language)

    definitions = [key for key in captures if key.startswith("definition.")]
    assert definitions, f"{language}: no definitions captured"
    assert captures.get("name"), f"{language}: definitions captured without names"


@pytest.mark.parametrize("language", LANGUAGES)
def test_imports_are_captured_with_their_module(language: str) -> None:
    captures = captures_for(language)

    assert captures.get("import"), f"{language}: no imports captured"
    assert captures.get("import.module"), f"{language}: imports captured without a module"


@pytest.mark.parametrize("language", LANGUAGES)
def test_references_are_captured(language: str) -> None:
    """Without references there is no graph, which is what separates FULL from STRUCTURAL."""
    captures = captures_for(language)

    references = [key for key in captures if key.startswith("reference.")]
    assert references, f"{language}: no references captured"


def test_go_captures_a_method_by_its_own_name() -> None:
    """`func (s *Server) Serve()` is `Serve`, not `Server.Serve`.

    Qualifying it with the receiver would make every type's method collide on lookup.
    """
    captures = captures_for("go")

    assert "Serve" in captures["name"]
    assert "Register" in captures["name"]


def test_rust_attributes_methods_to_the_impl_type() -> None:
    """Rust has no other syntax linking a method to its type, so the impl block carries it."""
    captures = captures_for("rust")

    assert "Cache" in captures["name"]
    assert "insert" in captures["name"]


def test_java_separates_fields_from_locals() -> None:
    """`total` is a local inside computeSubtotal and must not enter the symbol table."""
    captures = captures_for("java")

    assert "computeSubtotal" in captures["name"]
    assert "total" not in captures["name"]


def test_the_polyglot_fixture_indexes_end_to_end(tmp_path: Path) -> None:
    """The real check: symbols reach the database, not just the query."""
    for language in LANGUAGES:
        require(language)

    workspace = tmp_path / "polyglot"
    shutil.copytree(FIXTURE, workspace)
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    Indexer(root=workspace, repository=IndexRepository(connection)).run()

    rows = connection.execute(
        """
        SELECT f.language AS language, COUNT(s.id) AS symbols
        FROM files f LEFT JOIN symbols s ON s.file_id = f.id
        GROUP BY f.language
        """
    ).fetchall()
    by_language = {str(row["language"]): int(row["symbols"]) for row in rows}

    for language in LANGUAGES:
        assert by_language.get(language, 0) > 0, f"{language} produced no symbols: {by_language}"
