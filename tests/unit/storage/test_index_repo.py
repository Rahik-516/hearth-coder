"""Storage: migrations, and the chunks/FTS consistency invariant.

The FTS tests are the important ones. A contentless FTS5 table maintains nothing
automatically, so every way a chunk can disappear must be matched by an explicit FTS
delete. If it isn't, search returns rowids for chunks that no longer exist — citations
pointing at deleted code, discovered long after the cause.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hearth.storage.db import connect, has_fts5, table_names
from hearth.storage.errors import SchemaTooNewError
from hearth.storage.index_repo import (
    ChunkRecord,
    FilePayload,
    FileRecord,
    ImportRecord,
    IndexRepository,
    RefRecord,
    SymbolRecord,
)
from hearth.storage.migrate import current_version, latest_version, migrate


@pytest.fixture
def repo(tmp_path: Path) -> IndexRepository:
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    return IndexRepository(connection)


def payload(path: str = "src/a.py", *, chunk_texts: tuple[str, ...] = ("def alpha(): ...",)) -> FilePayload:
    return FilePayload(
        file=FileRecord(
            path=path,
            language="python",
            size_bytes=100,
            mtime_ns=123,
            content_hash="deadbeef",
        ),
        chunks=[
            ChunkRecord(
                kind="function",
                start_line=1 + i,
                end_line=2 + i,
                start_byte=0,
                end_byte=10,
                text=text,
                embed_text_hash=f"hash{i}",
                token_estimate=5,
                symbol_path=f"alpha{i}",
            )
            for i, text in enumerate(chunk_texts)
        ],
        symbols=[SymbolRecord(name="alpha", kind="function", start_line=1, end_line=2)],
        refs=[RefRecord(name="beta", line=2, kind="call")],
        imports=[ImportRecord(module_spec="os", names=["path"])],
    )


# ------------------------------------------------------------------ migrations


def test_migrate_creates_every_table(repo: IndexRepository) -> None:
    names = table_names(repo.connection)
    assert {"meta", "files", "chunks", "chunks_fts", "embeddings", "symbols", "refs", "imports"} <= names


def test_migrate_records_and_is_idempotent(tmp_path: Path) -> None:
    connection = connect(tmp_path / "index.db")
    first = migrate(connection, database="index")

    assert first == latest_version("index")
    assert current_version(connection) == first
    assert migrate(connection, database="index") == first  # no-op second time


def test_summaries_table_is_deliberately_absent(repo: IndexRepository) -> None:
    """Summaries are Phase 2 (docs/system-design.md §6.9); M1 must not ship them."""
    assert "summaries" not in table_names(repo.connection)


def test_schema_from_the_future_is_refused(tmp_path: Path) -> None:
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    IndexRepository(connection).set_meta("schema_version", "9999")

    with pytest.raises(SchemaTooNewError):
        migrate(connection, database="index")


def test_fts5_is_available(repo: IndexRepository) -> None:
    assert has_fts5(repo.connection) is True


def test_foreign_keys_are_enforced(repo: IndexRepository) -> None:
    """Without this pragma, deleting a file silently orphans its chunks."""
    with pytest.raises(sqlite3.IntegrityError):
        repo.connection.execute(
            "INSERT INTO chunks(file_id, kind, start_line, end_line, start_byte, end_byte, "
            "text, embed_text_hash, token_estimate) VALUES(9999, 'function', 1, 2, 0, 1, 'x', 'h', 1)"
        )


# ----------------------------------------------------------------------- writes


def test_replace_file_writes_everything(repo: IndexRepository) -> None:
    repo.replace_files([payload()])

    assert repo.count_files() == 1
    assert repo.count_chunks() == 1
    assert repo.count_symbols() == 1
    assert repo.get_file("src/a.py") is not None


def test_replace_file_is_idempotent(repo: IndexRepository) -> None:
    repo.replace_files([payload()])
    repo.replace_files([payload()])

    assert repo.count_files() == 1
    assert repo.count_chunks() == 1


def test_reindexing_replaces_rather_than_accumulates(repo: IndexRepository) -> None:
    repo.replace_files([payload(chunk_texts=("one", "two", "three"))])
    assert repo.count_chunks() == 3

    repo.replace_files([payload(chunk_texts=("only one now",))])
    assert repo.count_chunks() == 1


def test_cascade_removes_derived_rows(repo: IndexRepository) -> None:
    repo.replace_files([payload()])
    repo.delete_files(["src/a.py"])

    assert repo.count_files() == 0
    assert repo.count_chunks() == 0
    assert repo.count_symbols() == 0
    assert repo.connection.execute("SELECT COUNT(*) FROM refs").fetchone()[0] == 0
    assert repo.connection.execute("SELECT COUNT(*) FROM imports").fetchone()[0] == 0


# -------------------------------------------------------------- FTS invariant


def fts_rowids(repo: IndexRepository) -> set[int]:
    rows = repo.connection.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'alpha0 OR one OR two'"
    )
    return {int(r["rowid"]) for r in rows}


def test_search_finds_an_indexed_chunk(repo: IndexRepository) -> None:
    repo.replace_files([payload(chunk_texts=("def finalize(self): ...",))])

    results = repo.search_lexical("finalize")

    assert results
    assert results[0]["path"] == "src/a.py"


def test_deleting_a_file_removes_its_fts_rows(repo: IndexRepository) -> None:
    """The invariant. A contentless FTS table cascades nothing on its own."""
    repo.replace_files([payload(chunk_texts=("unique_token_xyz",))])
    assert repo.search_lexical("unique_token_xyz")

    repo.delete_files(["src/a.py"])

    assert repo.search_lexical("unique_token_xyz") == []


def test_reindexing_removes_stale_fts_rows(repo: IndexRepository) -> None:
    """Text removed from a file must stop being searchable."""
    repo.replace_files([payload(chunk_texts=("token_before_edit",))])
    assert repo.search_lexical("token_before_edit")

    repo.replace_files([payload(chunk_texts=("token_after_edit",))])

    assert repo.search_lexical("token_before_edit") == []
    assert repo.search_lexical("token_after_edit")


def test_fts_rowids_always_resolve_to_live_chunks(repo: IndexRepository) -> None:
    """The failure this guards: search returning rowids for deleted chunks."""
    repo.replace_files([payload("src/a.py", chunk_texts=("alpha", "beta"))])
    repo.replace_files([payload("src/b.py", chunk_texts=("gamma",))])
    repo.delete_files(["src/a.py"])
    repo.replace_files([payload("src/b.py", chunk_texts=("gamma", "delta"))])

    orphans = repo.connection.execute(
        "SELECT COUNT(*) FROM chunks_fts "
        "WHERE chunks_fts MATCH 'alpha OR beta OR gamma OR delta' "
        "AND rowid NOT IN (SELECT id FROM chunks)"
    ).fetchone()[0]

    assert orphans == 0


def test_identifier_split_makes_camel_case_searchable(repo: IndexRepository) -> None:
    """The point of search_text enrichment: "invoice service" must find InvoiceService."""
    repo.replace_files([payload(chunk_texts=("class InvoiceService: pass",))])

    assert repo.search_lexical("invoice")
    assert repo.search_lexical("service")
    assert repo.search_lexical("InvoiceService")


# ----------------------------------------------------------------------- meta


def test_meta_roundtrip(repo: IndexRepository) -> None:
    repo.set_meta("embed_model", "qwen3-embedding:0.6b")
    assert repo.get_meta("embed_model") == "qwen3-embedding:0.6b"

    repo.set_meta("embed_model", "other")
    assert repo.get_meta("embed_model") == "other"
    assert repo.get_meta("absent") is None


def test_file_states_supports_change_detection(repo: IndexRepository) -> None:
    repo.replace_files([payload()])
    states = repo.file_states()

    assert states["src/a.py"] == (100, 123, "deadbeef")
