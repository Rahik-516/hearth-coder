"""Numbered SQL migrations.

Hand-rolled rather than Alembic: there is no ORM, the schemas are small and hand-written,
and ``index.db`` can simply be dropped and rebuilt when a migration would be awkward
(docs/tech-stack.md §5.2).

Migrations are plain ``.sql`` files named ``NNNN_description.sql``, applied in numeric
order inside one transaction each. The applied version lives in ``meta.schema_version``.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from hearth.storage.db import transaction
from hearth.storage.errors import MigrationError, SchemaTooNewError

_MIGRATIONS_ROOT = Path(__file__).parent / "migrations"
_FILENAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")


def discover(database: str) -> list[Migration]:
    """Load migrations for a database ("index" or "state"), ordered by version."""
    directory = _MIGRATIONS_ROOT / database
    if not directory.is_dir():
        raise MigrationError(f"no migrations directory for {database!r}")

    migrations: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = _FILENAME.match(path.name)
        if match is None:
            raise MigrationError(f"migration {path.name!r} does not match NNNN_description.sql")
        migrations.append(Migration(version=int(match.group(1)), name=match.group(2), path=path))

    versions = [m.version for m in migrations]
    if len(set(versions)) != len(versions):
        raise MigrationError(f"duplicate migration versions in {directory}")
    return migrations


def current_version(connection: sqlite3.Connection) -> int:
    """Applied schema version. 0 means an empty database."""
    try:
        row = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.OperationalError:
        return 0  # no meta table yet
    if row is None:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError) as exc:
        raise MigrationError(f"meta.schema_version is not an integer: {row['value']!r}") from exc


def migrate(connection: sqlite3.Connection, *, database: str = "index") -> int:
    """Apply pending migrations. Returns the resulting schema version.

    Each migration runs in its own transaction, so a failure leaves the database at the
    last version that fully applied rather than halfway through one.
    """
    migrations = discover(database)
    if not migrations:
        raise MigrationError(f"no migrations found for {database!r}")

    latest = max(m.version for m in migrations)
    applied = current_version(connection)

    if applied > latest:
        raise SchemaTooNewError(found=applied, supported=latest)
    if applied == latest:
        return applied

    for migration in migrations:
        if migration.version <= applied:
            continue
        try:
            with transaction(connection):
                connection.executescript(migration.sql)
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(migration.version),),
                )
        except sqlite3.Error as exc:
            raise MigrationError(f"migration {migration.version:04d}_{migration.name} failed: {exc}") from exc
        applied = migration.version

    return applied


def latest_version(database: str = "index") -> int:
    """Highest migration version available on disk."""
    return max(m.version for m in discover(database))
