"""Scaffolding smoke tests.

These assert that the skeleton itself is sound: the package imports without side effects,
the CLI entry point runs, the layer packages all exist, and the egress guard is active.
They are cheap and they fail loudly if the project layout drifts from
docs/project-structure.md §1.
"""

from __future__ import annotations

import importlib
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from pytest_socket import SocketBlockedError, SocketConnectBlockedError

import hearth

SRC = Path(__file__).resolve().parents[2] / "src" / "hearth"

LAYER_PACKAGES = [
    "hearth.cli",
    "hearth.server",
    "hearth.core",
    "hearth.workflows",
    "hearth.tools",
    "hearth.retrieval",
    "hearth.indexing",
    "hearth.safety",
    "hearth.llm",
    "hearth.storage",
    "hearth.git",
    "hearth.config",
    "hearth.util",
]


def test_version_is_exposed() -> None:
    assert hearth.__version__ == "0.1.0"


@pytest.mark.parametrize("module", LAYER_PACKAGES)
def test_layer_package_imports(module: str) -> None:
    """Every layer named in the import-linter contracts must exist and import cleanly."""
    importlib.import_module(module)


def test_cli_version_runs() -> None:
    """`python -m hearth --version` works end to end."""
    result = subprocess.run(
        [sys.executable, "-m", "hearth", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "hearth 0.1.0" in result.stdout


def test_sockets_are_blocked() -> None:
    """The egress guard is active.

    If this fails, --disable-socket was removed from addopts. Zero egress is the top
    quality attribute and it is verified here (docs/tech-stack.md §11.1).
    """
    # pytest-socket raises SocketBlockedError when socket() construction itself is blocked,
    # and SocketConnectBlockedError when construction succeeds but connect() is blocked
    # (the case here, since it's a real socket.socket() call). Neither is a subclass of
    # the other — both are direct RuntimeError subclasses — so both are caught.
    with (
        pytest.raises((SocketBlockedError, SocketConnectBlockedError)),
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
    ):
        sock.connect(("93.184.216.34", 80))


def test_prompts_directory_exists() -> None:
    """Prompts live on disk as Markdown, never inline in Python (docs/project-structure.md §4)."""
    assert (SRC / "prompts").is_dir()
