"""Read/write access to ``state.db``.

This file is **not disposable**. It holds conversation history, checkpoints and trust
records, so everything here preserves data rather than rebuilding it.

Messages are stored as opaque JSON rather than flat columns, because the shape varies by
role — an assistant turn carries tool calls, a tool turn carries a call id — and modelling
that relationally would mean a join per message for no benefit. The ``epoch`` column is
what lets a resumed session rebuild the same cache-stable prefix.

**This module deliberately does not know about ``hearth.llm``.** ``storage`` and ``llm``
are sibling layers that may not import each other (docs/project-structure.md §2), so
messages cross this boundary as JSON strings and ``core.session`` owns the conversion.
That is not ceremony: it is what keeps persistence usable by a future frontend or tool
that has no opinion about message types.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

from hearth.storage.db import transaction


def _now_ms() -> int:
    """Current time in milliseconds.

    Milliseconds rather than seconds because `resume` orders by `updated_at`, and two
    sessions created within the same second would otherwise tie and resume
    non-deterministically -- which is easy to hit, since starting a session and asking the
    first question happen together.
    """
    return int(time.time() * 1000)


def new_session_id() -> str:
    """Sortable-ish, readable session id."""
    return f"s_{uuid.uuid4().hex[:16]}"


@dataclass
class SessionRecord:
    """One conversation."""

    id: str
    created_at: int
    updated_at: int
    title: str | None = None
    mode: str = "chat"
    permission_level: str = "supervised"
    model: str | None = None
    num_ctx: int | None = None
    workspace: str | None = None
    message_count: int = 0

    @property
    def display_title(self) -> str:
        return self.title or "(untitled)"


@dataclass
class StoredMessage:
    """One persisted message: its role, its serialized body, and its metadata."""

    seq: int
    role: str
    content_json: str
    token_estimate: int = 0
    epoch: int = 0
    created_at: int = field(default_factory=_now_ms)


class StateRepository:
    """Queries and writes over ``state.db``."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    # -------------------------------------------------------------- sessions

    def create_session(
        self,
        *,
        session_id: str | None = None,
        mode: str = "chat",
        model: str | None = None,
        num_ctx: int | None = None,
        workspace: str | None = None,
        title: str | None = None,
    ) -> SessionRecord:
        now = _now_ms()
        record = SessionRecord(
            id=session_id or new_session_id(),
            created_at=now,
            updated_at=now,
            title=title,
            mode=mode,
            model=model,
            num_ctx=num_ctx,
            workspace=workspace,
        )
        with transaction(self._connection):
            self._connection.execute(
                "INSERT INTO sessions(id, created_at, updated_at, title, mode, "
                "permission_level, model, num_ctx, workspace) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.created_at,
                    record.updated_at,
                    record.title,
                    record.mode,
                    record.permission_level,
                    record.model,
                    record.num_ctx,
                    record.workspace,
                ),
            )
        return record

    def get_session(self, session_id: str) -> SessionRecord | None:
        row = self._connection.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n "
            "FROM sessions s WHERE s.id = ?",
            (session_id,),
        ).fetchone()
        return None if row is None else _session_from_row(row)

    def latest_session(self, *, workspace: str | None = None) -> SessionRecord | None:
        """Most recently updated session, for ``hearth resume``.

        Ordered by timestamp then insertion order. Milliseconds narrow the tie window but
        do not close it — creating a session and asking its first question can land in the
        same millisecond — and resuming the wrong conversation is a confusing failure.
        """
        if workspace is None:
            row = self._connection.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n "
                "FROM sessions s ORDER BY s.updated_at DESC, s.rowid DESC LIMIT 1"
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n "
                "FROM sessions s WHERE s.workspace = ? ORDER BY s.updated_at DESC, s.rowid DESC LIMIT 1",
                (workspace,),
            ).fetchone()
        return None if row is None else _session_from_row(row)

    def list_sessions(self, *, limit: int = 20) -> list[SessionRecord]:
        rows = self._connection.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n "
            "FROM sessions s ORDER BY s.updated_at DESC, s.rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_session_from_row(row) for row in rows]

    def touch_session(self, session_id: str, *, title: str | None = None) -> None:
        """Mark a session as recently used, optionally naming it."""
        with transaction(self._connection):
            if title is None:
                self._connection.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (_now_ms(), session_id),
                )
            else:
                self._connection.execute(
                    "UPDATE sessions SET updated_at = ?, title = COALESCE(title, ?) WHERE id = ?",
                    (_now_ms(), title, session_id),
                )

    def set_model(self, session_id: str, model: str, num_ctx: int | None = None) -> None:
        with transaction(self._connection):
            self._connection.execute(
                "UPDATE sessions SET model = ?, num_ctx = COALESCE(?, num_ctx), updated_at = ? WHERE id = ?",
                (model, num_ctx, _now_ms(), session_id),
            )

    def delete_session(self, session_id: str) -> bool:
        with transaction(self._connection):
            cursor = self._connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    # -------------------------------------------------------------- messages

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content_json: str,
        token_estimate: int = 0,
        epoch: int = 0,
    ) -> int:
        """Append one message, returning its sequence number.

        Append-only by construction: the next seq is derived from what is stored, so two
        writers cannot silently overwrite each other's turn.
        """
        with transaction(self._connection):
            row = self._connection.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            seq = int(row[0])

            self._connection.execute(
                "INSERT INTO messages(session_id, seq, role, content_json, token_estimate, "
                "epoch, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    seq,
                    role,
                    content_json,
                    token_estimate,
                    epoch,
                    _now_ms(),
                ),
            )
            self._connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (_now_ms(), session_id),
            )
        return seq

    def load_messages(self, session_id: str, *, epoch: int | None = None) -> list[StoredMessage]:
        """Load a session's history in order, optionally restricted to one epoch."""
        if epoch is None:
            rows = self._connection.execute(
                "SELECT seq, role, content_json, token_estimate, epoch, created_at FROM messages "
                "WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT seq, role, content_json, token_estimate, epoch, created_at FROM messages "
                "WHERE session_id = ? AND epoch = ? ORDER BY seq",
                (session_id, epoch),
            ).fetchall()

        return [
            StoredMessage(
                seq=int(row["seq"]),
                role=str(row["role"]),
                content_json=str(row["content_json"]),
                token_estimate=int(row["token_estimate"]),
                epoch=int(row["epoch"]),
                created_at=int(row["created_at"]),
            )
            for row in rows
        ]

    def current_epoch(self, session_id: str) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(epoch), 0) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0])

    def count_messages(self, session_id: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        return int(row[0])

    def clear_messages(self, session_id: str) -> int:
        """Drop a session's history, keeping the session itself. Backs ``/clear``."""
        with transaction(self._connection):
            cursor = self._connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        return cursor.rowcount

    # ------------------------------------------------------------------ meta

    def set_meta(self, key: str, value: str) -> None:
        with transaction(self._connection):
            self._connection.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_meta(self, key: str) -> str | None:
        row = self._connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    # ----------------------------------------------------------------- trust

    def is_trusted(self, config_sha256: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM trust WHERE project_config_sha256 = ?", (config_sha256,)
        ).fetchone()
        return row is not None

    def record_trust(self, config_sha256: str) -> None:
        with transaction(self._connection):
            self._connection.execute(
                "INSERT INTO trust(project_config_sha256, trusted_at) VALUES(?, ?) "
                "ON CONFLICT(project_config_sha256) DO UPDATE SET trusted_at = excluded.trusted_at",
                (config_sha256, _now_ms()),
            )

    def revoke_trust(self, config_sha256: str) -> bool:
        with transaction(self._connection):
            cursor = self._connection.execute(
                "DELETE FROM trust WHERE project_config_sha256 = ?", (config_sha256,)
            )
        return cursor.rowcount > 0

    # ---------------------------------------------------------------- grants

    def add_grant(self, session_id: str, grant_key: str) -> None:
        """Record an "always for this session" approval.

        Keys are exact strings (docs/safety-and-tool-use.md §5.4) and the primary key
        makes re-granting idempotent — the user pressing `s` twice is not an error.
        """
        with transaction(self._connection):
            self._connection.execute(
                "INSERT INTO grants(session_id, grant_key, created_at) VALUES(?, ?, ?) "
                "ON CONFLICT(session_id, grant_key) DO NOTHING",
                (session_id, grant_key, _now_ms()),
            )

    def grants_for(self, session_id: str) -> frozenset[str]:
        """Every grant key active for one session.

        Scoped to the session by the schema's foreign key, so a grant cannot outlive the
        conversation in which the user gave it.
        """
        rows = self._connection.execute(
            "SELECT grant_key FROM grants WHERE session_id = ?", (session_id,)
        ).fetchall()
        return frozenset(str(row["grant_key"]) for row in rows)

    def clear_grants(self, session_id: str) -> int:
        """Drop every grant for a session. Backs a `/grants clear` style escape hatch."""
        with transaction(self._connection):
            cursor = self._connection.execute("DELETE FROM grants WHERE session_id = ?", (session_id,))
        return cursor.rowcount

    # ----------------------------------------------------------- checkpoints

    def add_checkpoint(
        self,
        *,
        session_id: str,
        step: int,
        path: str,
        before_blob: str | None,
        after_blob: str | None,
    ) -> int:
        """Record one file's before/after for one write step.

        ``before_blob`` of NULL means the file did not exist, which is what makes undo
        able to tell "restore these bytes" from "remove this file".
        """
        with transaction(self._connection):
            cursor = self._connection.execute(
                "INSERT INTO checkpoints(session_id, step, path, before_blob, after_blob, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (session_id, step, path, before_blob, after_blob, _now_ms()),
            )
        return int(cursor.lastrowid or 0)

    def checkpoints_for_step(self, session_id: str, step: int) -> list[sqlite3.Row]:
        """Every file touched by one step, in the order it was written."""
        return list(
            self._connection.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? AND step = ? ORDER BY id",
                (session_id, step),
            ).fetchall()
        )

    def checkpoint_rows(self, session_id: str) -> list[sqlite3.Row]:
        """Every checkpoint for a session, newest step first."""
        return list(
            self._connection.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY step DESC, id ASC",
                (session_id,),
            ).fetchall()
        )

    def mark_step_reverted(self, session_id: str, step: int) -> int:
        with transaction(self._connection):
            cursor = self._connection.execute(
                "UPDATE checkpoints SET reverted = 1 WHERE session_id = ? AND step = ?",
                (session_id, step),
            )
        return cursor.rowcount


def _session_from_row(row: sqlite3.Row) -> SessionRecord:
    keys = row.keys()
    return SessionRecord(
        id=str(row["id"]),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        title=row["title"],
        mode=str(row["mode"]),
        permission_level=str(row["permission_level"]),
        model=row["model"],
        num_ctx=int(row["num_ctx"]) if row["num_ctx"] is not None else None,
        workspace=row["workspace"],
        message_count=int(row["n"]) if "n" in keys else 0,
    )


def dumps(value: object) -> str:
    """JSON with stable key order, so stored rows diff cleanly."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
