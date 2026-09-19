"""Content-addressed blob storage, under the project's data directory.

Backs checkpoints (the before/after bytes of every write) and, from M6, captured command
output. Two properties earn their keep:

* **Addressed by content**, so re-editing the same file all session stores one copy of
  each distinct version rather than one per write.
* **Immutable once written.** A digest always names the same bytes, which is what lets a
  checkpoint hold a digest and trust it later without a lock or a generation counter.

Blobs live outside the repository being worked on (docs/project-structure.md §3), so
nothing here ever appears in the user's `git status`.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from hearth.util.hashing import blob_hash

#: Hex characters used as the shard directory name. Two gives 256 buckets, which keeps
#: any single directory small enough to list quickly on every filesystem Hearth targets.
_SHARD = 2

#: A digest is lowercase hex and nothing else.
#:
#: Enforced rather than assumed because the shard join — ``root / digest[:2] / digest`` —
#: reads a digest as a *path*. While digests only ever came from :meth:`put` that was
#: safe; once one can arrive from a model (``read_output`` pages by ``output_id``), a
#: value like ``../../etc/passwd`` would walk straight out of the store. Validating here
#: rather than at the call site means every caller is covered, including the next one.
_DIGEST = re.compile(r"^[0-9a-f]{16,128}$")


class InvalidDigestError(ValueError):
    """A digest that is not a plain lowercase hex string."""


class BlobStore:
    """A directory of immutable, content-addressed files."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, digest: str) -> Path:
        """Where a digest lives. Sharded by its first two characters.

        Raises:
            InvalidDigestError: for anything that is not lowercase hex. The join treats a
                digest as a path, so this is a jail, not a formatting check.
        """
        if not _DIGEST.match(digest):
            raise InvalidDigestError(f"not a digest: {digest!r}")
        return self._root / digest[:_SHARD] / digest

    def has(self, digest: str) -> bool:
        try:
            return self.path_for(digest).is_file()
        except InvalidDigestError:
            return False

    def put(self, data: bytes) -> str:
        """Store bytes and return their digest. A no-op if already present.

        Written to a temp file in the destination directory and then renamed, so a crash
        mid-write cannot leave a truncated blob under a digest that claims to be complete
        — which a later restore would happily write over a user's file.
        """
        digest = blob_hash(data)
        target = self.path_for(digest)
        if target.is_file():
            return digest

        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            Path(temporary).replace(target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return digest

    def get(self, digest: str) -> bytes:
        """The bytes for a digest.

        Raises:
            FileNotFoundError: if the blob is absent. Callers must not substitute empty
                content — an emptied file and a missing checkpoint are different facts.
        """
        return self.path_for(digest).read_bytes()

    def count(self) -> int:
        """How many blobs are stored. For diagnostics and a future garbage collection."""
        if not self._root.is_dir():
            return 0
        return sum(1 for path in self._root.rglob("*") if path.is_file())

    def total_bytes(self) -> int:
        if not self._root.is_dir():
            return 0
        return sum(path.stat().st_size for path in self._root.rglob("*") if path.is_file())
