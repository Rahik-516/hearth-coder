"""Domain exceptions for storage."""

from __future__ import annotations


class StorageError(Exception):
    """Base class for every error raised by ``hearth.storage``."""


class MigrationError(StorageError):
    """A migration could not be applied, or the schema is from a newer Hearth."""


class SchemaTooNewError(MigrationError):
    """The database was written by a newer Hearth than this one.

    Downgrading is not supported. For ``index.db`` the fix is to rebuild, since it is
    disposable; for ``state.db`` it is to upgrade Hearth again.
    """

    def __init__(self, found: int, supported: int) -> None:
        self.found = found
        self.supported = supported
        super().__init__(
            f"database schema version {found} is newer than the supported version "
            f"{supported}. Upgrade Hearth, or rebuild the index with `hearth index --rebuild`."
        )
