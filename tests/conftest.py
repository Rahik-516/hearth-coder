"""Shared test fixtures.

Two properties this file is responsible for:

1. **No egress.** ``pytest-socket`` is enabled through ``addopts`` in ``pyproject.toml``
   with loopback allowed for opt-in live tests. This is a product requirement, not a
   convenience (docs/tech-stack.md §11.1) — do not disable it.
2. **No real repositories.** Agent tools are never pointed at a real checkout in tests.
   Fixture repos are copied to a tmp directory first (docs/project-structure.md §5).
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

FIXTURE_REPOS = Path(__file__).parent / "fixtures" / "repos"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An empty tmp directory standing in for a workspace root."""
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Iterator[object]:
    """Copy a named fixture repo to a tmp directory and return its path.

    Usage::

        def test_something(fixture_repo):
            repo = fixture_repo("py_small")

    Every test that touches files works on the copy, never on the original, so a test
    that writes or deletes cannot corrupt the fixtures.
    """

    def _copy(name: str) -> Path:
        source = FIXTURE_REPOS / name
        if not source.is_dir():
            pytest.skip(f"fixture repo {name!r} does not exist yet")
        destination = tmp_path / name
        shutil.copytree(source, destination)
        return destination

    yield _copy
