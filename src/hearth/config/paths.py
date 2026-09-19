"""Filesystem locations for config, per-project data, and logs.

Hearth never writes index or state data into the repository being worked on
(docs/project-structure.md §3). Everything lives under platformdirs locations, keyed by a
stable project id so two checkouts of the same repo at different paths get separate
indexes.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from platformdirs import user_config_path, user_data_path

APP_NAME = "hearth"

#: Length of the hex project id. 16 hex chars = 64 bits, ample against accidental
#: collision across one developer's checkouts.
_PROJECT_ID_LEN = 16

_SLUG_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def global_config_dir() -> Path:
    """``~/.config/hearth`` on Linux; the OS equivalent elsewhere."""
    return user_config_path(APP_NAME, appauthor=False)


def global_config_file() -> Path:
    """The user's global config. A protected path — the agent may never write it."""
    return global_config_dir() / "config.toml"


def data_dir() -> Path:
    """``~/.local/share/hearth`` on Linux; the OS equivalent elsewhere."""
    return user_data_path(APP_NAME, appauthor=False)


def project_id(root: Path) -> str:
    """Stable id for a repository root.

    The first 16 hex characters of the SHA-256 of the canonical, resolved root path
    (docs/project-structure.md §3). Resolution matters: it means a symlinked path and its
    target share one index rather than silently building two.
    """
    canonical = root.resolve().as_posix()
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:_PROJECT_ID_LEN]


def project_slug(root: Path) -> str:
    """Human-readable directory-name component, for eyeballing the data directory."""
    name = root.resolve().name or "root"
    slug = _SLUG_UNSAFE.sub("-", name).strip("-.")
    return (slug or "root")[:48]


def project_data_dir(root: Path) -> Path:
    """``<data>/projects/<slug>-<id>/`` — index, state, blobs and trash for one repo."""
    return data_dir() / "projects" / f"{project_slug(root)}-{project_id(root)}"


def index_db_path(root: Path) -> Path:
    """Disposable: always rebuildable from the working tree."""
    return project_data_dir(root) / "index.db"


def state_db_path(root: Path) -> Path:
    """Not disposable: sessions, checkpoints, grants and trust records live here."""
    return project_data_dir(root) / "state.db"


def blobs_dir(root: Path) -> Path:
    """Content-addressed snapshots backing checkpoints, and full tool outputs."""
    return project_data_dir(root) / "blobs"


def trash_dir(root: Path) -> Path:
    """Where ``delete_file`` moves things, so a delete stays reversible."""
    return project_data_dir(root) / "trash"


def audit_dir() -> Path:
    """Append-only audit log, shared across projects: ``audit/YYYY-MM.jsonl``."""
    return data_dir() / "audit"


def logs_dir() -> Path:
    """Application logs, and ``--debug`` prompt dumps (off by default)."""
    return data_dir() / "logs"


def project_config_file(root: Path) -> Path:
    """Optional, committable, user-authored project settings."""
    return root / ".hearth" / "config.toml"


def project_local_config_file(root: Path) -> Path:
    """Personal, uncommitted overrides; ``hearth init`` offers to gitignore this."""
    return root / ".hearth" / "local.toml"


def project_instructions_files(root: Path) -> tuple[Path, ...]:
    """Candidate project-instruction files, in precedence order.

    ``AGENTS.md`` is preferred; ``HEARTH.md`` is accepted for projects that want a
    Hearth-specific file (docs/project-structure.md §3).
    """
    return (root / "AGENTS.md", root / "HEARTH.md")
