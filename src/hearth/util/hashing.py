"""Content hashing.

Two hashes, chosen for different jobs (docs/system-design.md §13):

* **blake2b-128** for file content. It is fast and only needs to detect change, so 128
  bits is ample and halves the storage versus SHA-256.
* **SHA-256** for blobs and embedding cache keys, where the value is content-addressed and
  shared across runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_FILE_DIGEST_BYTES = 16  # 128 bits
_READ_CHUNK = 1 << 20  # 1 MiB


def content_hash(data: bytes | str) -> str:
    """blake2b-128 hex digest of file content."""
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.blake2b(payload, digest_size=_FILE_DIGEST_BYTES).hexdigest()


def file_hash(path: Path) -> str:
    """blake2b-128 of a file, streamed so a large file does not land in memory."""
    digest = hashlib.blake2b(digest_size=_FILE_DIGEST_BYTES)
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def blob_hash(data: bytes | str) -> str:
    """SHA-256 hex digest, for content-addressed storage and cache keys."""
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()
