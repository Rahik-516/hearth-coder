"""Resolving what the user typed — I3 workflows.

Ambiguity and the jail are the two things worth pinning down. A name defined in three files
must not resolve to the first one, because a test or a refactor written against the wrong
`parse()` looks like success; and a path that leaves the workspace must be refused here,
before any workflow spends a model call on it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hearth.indexing.pipeline import Indexer
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate
from hearth.workflows.targets import ResolvedTarget, TargetError, resolve_target

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "repos"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(FIXTURES / "py_small", root)
    return root


@pytest.fixture
def repository(workspace: Path, tmp_path: Path) -> IndexRepository:
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    repo = IndexRepository(connection)
    Indexer(root=workspace, repository=repo).run()
    return repo


def test_an_existing_file_resolves_to_itself(workspace: Path) -> None:
    result = resolve_target(workspace, "src/billing/errors.py", repository=None)

    assert result == ResolvedTarget(path="src/billing/errors.py")


def test_windows_separators_resolve(workspace: Path) -> None:
    result = resolve_target(workspace, "src\\billing\\errors.py", repository=None)

    assert isinstance(result, ResolvedTarget)
    assert result.path == "src/billing/errors.py"


def test_a_unique_symbol_resolves_to_its_definition(
    workspace: Path, repository: IndexRepository
) -> None:
    result = resolve_target(workspace, "InvoiceService", repository=repository)

    assert isinstance(result, ResolvedTarget)
    assert result.path == "src/billing/invoice_service.py"
    assert result.symbol == "InvoiceService"
    assert result.start_line and result.end_line and result.end_line >= result.start_line


def test_path_double_colon_symbol_selects_within_a_file(
    workspace: Path, repository: IndexRepository
) -> None:
    result = resolve_target(
        workspace, "src/billing/invoice_service.py::InvoiceService", repository=repository
    )

    assert isinstance(result, ResolvedTarget)
    assert result.symbol == "InvoiceService"


def test_a_symbol_not_in_the_named_file_is_an_error(
    workspace: Path, repository: IndexRepository
) -> None:
    result = resolve_target(workspace, "src/billing/errors.py::InvoiceService", repository=repository)

    assert isinstance(result, TargetError)
    assert "not defined in" in result.message


def test_an_unknown_name_is_an_error(workspace: Path, repository: IndexRepository) -> None:
    result = resolve_target(workspace, "no_such_thing", repository=repository)

    assert isinstance(result, TargetError)
    assert "no file or symbol" in result.message


def test_an_ambiguous_name_lists_where_it_is_defined_and_picks_none(
    workspace: Path, repository: IndexRepository
) -> None:
    """The point of the module: a name in several places is a question, not a coin flip."""
    (workspace / "src" / "billing" / "dup_a.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (workspace / "src" / "billing" / "dup_b.py").write_text("def helper():\n    return 2\n", encoding="utf-8")
    Indexer(root=workspace, repository=repository).run()

    result = resolve_target(workspace, "helper", repository=repository)

    assert isinstance(result, TargetError)
    assert "dup_a.py::helper" in result.message
    assert "dup_b.py::helper" in result.message


def test_a_symbol_without_an_index_says_how_to_get_one(workspace: Path) -> None:
    result = resolve_target(workspace, "InvoiceService", repository=None)

    assert isinstance(result, TargetError)
    assert "hearth index" in result.message


@pytest.mark.parametrize("hostile", ["../outside.py", "../../etc/passwd", "/etc/passwd"])
def test_a_path_outside_the_workspace_is_refused(workspace: Path, hostile: str) -> None:
    result = resolve_target(workspace, hostile, repository=None)

    assert isinstance(result, TargetError)


def test_a_protected_path_is_refused(workspace: Path) -> None:
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    result = resolve_target(workspace, ".git/config", repository=None)

    assert isinstance(result, TargetError)
    assert "protected" in result.message


def test_a_path_shaped_target_is_never_retried_as_a_symbol(
    workspace: Path, repository: IndexRepository
) -> None:
    """`../x.py` is not a symbol name, and looking it up would be a guess."""
    result = resolve_target(workspace, "../InvoiceService.py", repository=repository)

    assert isinstance(result, TargetError)
    assert "outside the workspace" in result.message


def test_a_missing_file_is_an_error(workspace: Path) -> None:
    result = resolve_target(workspace, "src/billing/nope.py", repository=None)

    assert isinstance(result, TargetError)
    assert "not a file" in result.message


def test_an_empty_target_is_an_error(workspace: Path) -> None:
    assert isinstance(resolve_target(workspace, "  ", repository=None), TargetError)
