"""End-to-end indexing against the fixture repos.

Every M1 acceptance criterion from docs/implementation-roadmap.md is asserted here, so
they stay true rather than being verified once by hand:

* re-indexing an unchanged tree performs zero parses
* editing one file reindexes exactly that file
* secret files never reach `files` or `chunks`
* `InvoiceService finalize` returns the definition chunk first
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hearth.indexing.pipeline import Indexer
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "repos"


@pytest.fixture
def repo(tmp_path: Path) -> IndexRepository:
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    return IndexRepository(connection)


@pytest.fixture
def py_small(tmp_path: Path) -> Path:
    """A disposable copy. Tests here write to the tree, so never touch the original."""
    destination = tmp_path / "py_small"
    shutil.copytree(FIXTURES / "py_small", destination)
    return destination


def index(root: Path, repo: IndexRepository, *, force: bool = False):
    return Indexer(root=root, repository=repo).run(force=force)


# ------------------------------------------------------- incremental indexing


def test_first_run_indexes_the_tree(py_small: Path, repo: IndexRepository) -> None:
    stats = index(py_small, repo)

    assert stats.added > 0
    assert stats.parsed > 0
    assert repo.count_chunks() > 0
    assert repo.count_symbols() > 0


def test_reindex_with_no_changes_parses_nothing(py_small: Path, repo: IndexRepository) -> None:
    """M1 acceptance criterion, verified by counter rather than by timing."""
    first = index(py_small, repo)
    second = index(py_small, repo)

    assert second.parsed == 0
    assert second.files_written == 0
    assert second.unchanged == first.added
    assert repo.count_chunks() == first.chunks_written


def test_editing_one_file_reindexes_only_that_file(py_small: Path, repo: IndexRepository) -> None:
    """M1 acceptance criterion."""
    index(py_small, repo)

    target = py_small / "src" / "billing" / "errors.py"
    target.write_text(
        target.read_text(encoding="utf-8") + "\n\nclass NewlyAddedError(BillingError):\n    pass\n",
        encoding="utf-8",
    )

    stats = index(py_small, repo)

    assert stats.parsed == 1
    assert stats.modified == 1
    assert stats.added == 0
    assert repo.find_symbol("NewlyAddedError"), "the new symbol should be searchable"


def test_deleting_a_file_removes_it_from_the_index(py_small: Path, repo: IndexRepository) -> None:
    index(py_small, repo)
    assert repo.get_file("src/billing/payments.py") is not None

    (py_small / "src" / "billing" / "payments.py").unlink()
    stats = index(py_small, repo)

    assert stats.deleted == 1
    assert repo.get_file("src/billing/payments.py") is None

    # The README also mentions TokenBucket, so the query still matches something. What
    # must be gone is every chunk belonging to the deleted file.
    hits = repo.search_lexical('"tokenbucket"', limit=50)
    assert all(row["path"] != "src/billing/payments.py" for row in hits)


def test_rebuild_reindexes_everything(py_small: Path, repo: IndexRepository) -> None:
    first = index(py_small, repo)
    forced = index(py_small, repo, force=True)

    assert forced.parsed == first.parsed
    assert forced.modified == first.added


# -------------------------------------------------------------------- secrets


def test_secret_files_never_reach_the_index(repo: IndexRepository) -> None:
    """M1 acceptance criterion, asserted against the `malicious` fixture."""
    stats = index(FIXTURES / "malicious", repo)

    indexed = repo.file_paths()
    for secret in (
        ".env",
        ".env.production",
        "server.pem",
        "id_rsa",
        ".npmrc",
        "terraform.tfstate",
    ):
        assert secret not in indexed

    assert stats.skipped_by_reason.get("secret-file", 0) >= 6


def test_no_secret_content_is_searchable(repo: IndexRepository) -> None:
    """The stronger claim: not just absent as files, but absent from chunk text."""
    index(FIXTURES / "malicious", repo)

    for token in ("hunter2", "whsec", "npm_FAKE_token_value", "BEGIN PRIVATE KEY"):
        assert repo.search_lexical(f'"{token.lower()}"') == [], f"{token} leaked into the index"


def test_ordinary_files_in_the_malicious_repo_are_still_indexed(repo: IndexRepository) -> None:
    """A filter that excluded everything would pass the tests above while being useless."""
    index(FIXTURES / "malicious", repo)

    assert "src/safe.py" in repo.file_paths()


def test_generated_files_are_metadata_only(repo: IndexRepository) -> None:
    index(FIXTURES / "malicious", repo)

    record = repo.get_file("src/schema_pb2.py")
    assert record is not None
    assert record.is_generated is True

    chunks = repo.connection.execute(
        "SELECT COUNT(*) FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = ?",
        ("src/schema_pb2.py",),
    ).fetchone()[0]
    assert chunks == 0


# --------------------------------------------------------------------- search


def test_search_returns_the_definition_first(py_small: Path, repo: IndexRepository) -> None:
    """M1 acceptance criterion: the headline retrieval check."""
    index(py_small, repo)

    results = repo.search_lexical('"invoice" OR "service" OR "invoiceservice" OR "finalize"', limit=5)

    assert results, "expected matches"
    top = results[0]
    assert top["path"] == "src/billing/invoice_service.py"
    assert top["symbol_path"] == "InvoiceService"
    assert top["kind"] == "class"


def test_definitions_outrank_incidental_mentions(py_small: Path, repo: IndexRepository) -> None:
    """BM25 alone favours a short docstring that mentions a symbol over the symbol itself."""
    index(py_small, repo)

    results = repo.search_lexical('"invoiceservice"', limit=10)
    kinds = [r["kind"] for r in results]

    assert kinds[0] in ("class", "method", "function")


def test_camel_case_is_findable_by_its_parts(py_small: Path, repo: IndexRepository) -> None:
    index(py_small, repo)

    assert repo.search_lexical('"tokenbucket"'), "exact identifier should match"
    assert repo.search_lexical('"token" OR "bucket"'), "split identifier should match"


def test_symbols_are_recorded_with_nesting(py_small: Path, repo: IndexRepository) -> None:
    index(py_small, repo)

    found = repo.find_symbol("finalize")
    assert any(r["path"] == "src/billing/invoice_service.py" for r in found)


# ------------------------------------------------------------------ languages


def test_typescript_fixture_indexes(repo: IndexRepository) -> None:
    stats = index(FIXTURES / "ts_small", repo)

    assert stats.parsed >= 4, "ts, tsx and js files should all be parsed"
    assert repo.find_symbol("ShoppingCart")
    assert repo.find_symbol("RateLimiter")


def test_tsx_and_js_are_both_covered(repo: IndexRepository) -> None:
    index(FIXTURES / "ts_small", repo)
    languages = {f.language for f in repo.all_files()}

    assert {"typescript", "tsx", "javascript"} <= languages


def test_broken_syntax_file_is_indexed_not_dropped(py_small: Path, repo: IndexRepository) -> None:
    """A file that fails to parse must still be searchable, flagged rather than absent."""
    index(py_small, repo)

    record = repo.get_file("src/billing/broken_syntax.py")
    assert record is not None
    assert record.parse_status in ("partial", "failed")

    chunks = repo.connection.execute(
        "SELECT COUNT(*) FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = ?",
        ("src/billing/broken_syntax.py",),
    ).fetchone()[0]
    assert chunks > 0


def test_installing_a_grammar_reindexes_the_files_it_unlocks(tmp_path: Path, monkeypatch) -> None:
    """Installing a language grammar must not leave its files stranded as windows.

    Change detection compares size, mtime and hash, so a newly installed grammar — which
    changes what Hearth can *extract* without touching a single byte — is invisible to it.
    Before this was handled, `uv sync --extra langs-extra` followed by `hearth index`
    reported "4 unchanged" and produced zero Go, Rust and Java symbols, permanently,
    unless the user happened to think of `--rebuild`. Nothing suggested they should.
    """
    from hearth.indexing import languages, pipeline

    source = Path(__file__).resolve().parents[1] / "fixtures" / "repos" / "polyglot"
    if not {"go", "rust", "java"} <= languages.grammar_available():
        pytest.skip("language grammars not installed (uv sync --extra langs-extra)")

    workspace = tmp_path / "polyglot"
    shutil.copytree(source, workspace)
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    repo = IndexRepository(connection)

    # As if the extra were not installed.
    monkeypatch.setattr(pipeline, "grammar_available", lambda: frozenset({"python"}))
    monkeypatch.setattr(languages, "grammar_available", lambda: frozenset({"python"}))
    pipeline.Indexer(root=workspace, repository=repo).run()
    assert repo.count_symbols() == 0, "premise: no grammar means no symbols"

    # The user installs the extra and re-runs `hearth index`.
    monkeypatch.undo()
    stats = pipeline.Indexer(root=workspace, repository=repo).run()

    assert stats.modified >= 3, "the unlocked files must be re-indexed"
    assert repo.count_symbols() > 0

    # And the M1 criterion still holds: a further run parses nothing.
    again = pipeline.Indexer(root=workspace, repository=repo).run()
    assert again.parsed == 0
    assert again.modified == 0
