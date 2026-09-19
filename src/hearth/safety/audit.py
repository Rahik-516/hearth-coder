"""The audit log.

Append-only JSONL, one file per month, one object per line, fsynced per record
(docs/safety-and-tool-use.md §13). Every tool call is recorded with what was asked, what
was decided, who decided it, and what happened.

Design choices worth stating:

* **`O_APPEND` and fsync per record.** The log's value is that it survives the thing it is
  recording. A buffered write that is lost when a runaway command takes the process down
  documents exactly the wrong subset of history.
* **Redacted before writing.** Arguments can carry credentials, and this file is
  permanent. Redaction happens here rather than at the call site so it cannot be forgotten.
* **JSONL, not a database.** It must stay readable by `jq` and by a person five years from
  now, and appending is the only operation it needs.
* **A failed write never fails the tool call.** An unwritable log is worth a warning, not
  a refusal to work — the alternative punishes the user for a disk problem.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hearth.safety.secrets import redact_value

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass
class AuditRecord:
    """One auditable event."""

    tool: str
    risk: str
    decision: str
    session: str
    step: int = 0
    args: dict[str, Any] = field(default_factory=dict)
    decided_by: str | None = None
    rule_id: str | None = None
    project: str | None = None
    duration_ms: float | None = None
    exit_code: int | None = None
    error: str | None = None
    output_sha256: str | None = None
    output_bytes: int | None = None
    model: str | None = None
    badges: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize, with arguments redacted.

        Keys are sorted so successive records diff cleanly and a human scanning the file
        sees fields in the same order every time.
        """
        payload: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "session": self.session,
            "step": self.step,
            "tool": self.tool,
            "risk": self.risk,
            "args": redact_value(self.args),
            "decision": self.decision,
        }
        optional = {
            "decided_by": self.decided_by,
            "rule_id": self.rule_id,
            "project": self.project,
            "duration_ms": self.duration_ms,
            "exit_code": self.exit_code,
            "error": redact_value(self.error) if self.error else None,
            "output_sha256": self.output_sha256,
            "output_bytes": self.output_bytes,
            "model": self.model,
            "badges": self.badges or None,
            "hearth_version": _version(),
        }
        payload.update({k: v for k, v in optional.items() if v is not None})
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)


class AuditLog:
    """Writes audit records to a monthly JSONL file."""

    def __init__(self, directory: Path, *, fsync: bool = True) -> None:
        self._directory = directory
        self._fsync = fsync

    def path_for(self, when: float | None = None) -> Path:
        stamp = datetime.fromtimestamp(when or time.time(), tz=UTC)
        return self._directory / f"{stamp:%Y-%m}.jsonl"

    def write(self, record: AuditRecord) -> bool:
        """Append one record. Returns whether it was written.

        Never raises: an unwritable audit log is a problem to report, not a reason to
        refuse the work the user asked for.
        """
        line = record.to_json()
        target = self.path_for()

        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            # O_APPEND makes the write atomic against other writers, so two Hearth
            # processes on the same repo cannot interleave halves of a line.
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            handle = os.open(target, flags, 0o600)
            try:
                os.write(handle, (line + "\n").encode("utf-8"))
                if self._fsync:
                    os.fsync(handle)
            finally:
                os.close(handle)
        except OSError:
            logger.exception("could not write audit record to %s", target)
            return False
        return True

    def read_records(self, *, limit: int | None = None, when: float | None = None) -> list[dict[str, Any]]:
        """Read back records, newest last. Malformed lines are skipped, not fatal."""
        target = self.path_for(when)
        if not target.is_file():
            return []

        records: list[dict[str, Any]] = []
        try:
            for line in target.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn final line from a killed process should not hide the rest.
                    continue
        except OSError:
            return []

        return records[-limit:] if limit else records


def _version() -> str:
    from hearth import __version__

    return __version__
