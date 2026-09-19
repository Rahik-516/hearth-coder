"""Opening a SQLite connection with the pragmas Hearth depends on.

``check_same_thread=False`` is required, not incidental: the retrieval engine's dense
vector search runs on the thread that owns the connection while other work happens
concurrently, and a connection that enforced single-thread use would make that combination
impossible rather than merely slow (docs/system-design.md §13, the M2 lesson recorded in
CLAUDE.md — a broad ``except sqlite3.Error`` once silently disabled dense retrieval
entirely because of exactly this).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def table_names(connection: sqlite3.Connection) -> set[str]:
    """Every table and view name in the database, for schema assertions in tests."""
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
    ).fetchall()
    return {row["name"] for row in rows}


def has_fts5(connection: sqlite3.Connection) -> bool:
    """Whether this SQLite build has FTS5 compiled in.

    `hearth doctor` and the index commands check this before building the schema, because
    an FTS5-less SQLite fails with an opaque "no such module" error deep inside a
    migration rather than a message that says what to actually do about it.
    """
    try:
        connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
        connection.execute("DROP TABLE _fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def connect(path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    """Open (creating if needed) a Hearth SQLite database with its required pragmas.

    ``read_only=True`` opens via a ``file:`` URI in mode=ro, for the second connection a
    background thread uses to query the index while the main thread may still be writing
    to it — WAL mode makes a concurrent read safe, and opening read-only makes it explicit
    that this handle must never migrate the schema or write a row.
    """
    path = Path(path)
    # ``isolation_level=None`` turns off sqlite3's implicit transaction management, making
    # the explicit BEGIN in :class:`transaction` the only one. Without it the driver opens
    # its own transaction before a DML statement, and the next explicit BEGIN fails with
    # "cannot start a transaction within a transaction" — which is invisible on a first
    # index and breaks every re-index, since only then does a delete run before a batch.
    if read_only:
        connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro",
            uri=True,
            check_same_thread=False,
            isolation_level=None,
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    connection.row_factory = sqlite3.Row
    if not read_only:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


class transaction:
    """Context manager wrapping one write in BEGIN/COMMIT, rolling back on error.

    A thin wrapper rather than relying on sqlite3's implicit transaction handling, because
    the implicit behaviour differs between DML and DDL in ways that are easy to get wrong
    silently — an explicit BEGIN removes the ambiguity.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def __enter__(self) -> sqlite3.Connection:
        self._connection.execute("BEGIN")
        return self._connection

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self._connection.commit()
        else:
            self._connection.rollback()
