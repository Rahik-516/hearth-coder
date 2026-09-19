"""The fixture repos must keep the properties later milestones assert against.

M1's acceptance criteria name specific things — `InvoiceService.finalize` must be
findable, a syntax-error file must still chunk, an oversized function must exist to force
chunk splitting. If someone "tidies up" a fixture, the failure would otherwise surface as
a confusing chunker or retrieval regression much later.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOS = Path(__file__).resolve().parents[1] / "fixtures" / "repos"
PY_SMALL = REPOS / "py_small"


def test_py_small_exists() -> None:
    assert PY_SMALL.is_dir()


def test_invoice_service_defines_the_m1_retrieval_target() -> None:
    """`hearth search "InvoiceService finalize"` must have something to find."""
    source = (PY_SMALL / "src" / "billing" / "invoice_service.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "InvoiceService" in classes

    methods = {n.name for n in classes["InvoiceService"].body if isinstance(n, ast.FunctionDef)}
    assert "finalize" in methods


def test_invoice_service_has_a_nested_class_and_decorated_methods() -> None:
    """Chunker coverage: nested definitions, and decorators attaching to what follows."""
    source = (PY_SMALL / "src" / "billing" / "invoice_service.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    service = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "InvoiceService")
    nested = [n for n in service.body if isinstance(n, ast.ClassDef)]
    decorated = [n for n in service.body if isinstance(n, ast.FunctionDef) and n.decorator_list]

    assert nested, "expected a nested class"
    assert decorated, "expected at least one decorated method"


def test_broken_syntax_file_is_genuinely_unparseable() -> None:
    """If this ever parses, the "still chunks a broken file" test proves nothing."""
    source = (PY_SMALL / "src" / "billing" / "broken_syntax.py").read_text(encoding="utf-8")
    with pytest.raises(SyntaxError):
        ast.parse(source)


def test_every_other_python_file_parses() -> None:
    """Exactly one file is broken; the rest must be valid, or fixture tests are noise."""
    unparseable = []
    for path in sorted(PY_SMALL.rglob("*.py")):
        if path.name == "broken_syntax.py":
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - only on fixture breakage
            unparseable.append(path.name)

    assert unparseable == []


def test_oversized_function_exists_to_force_chunk_splitting() -> None:
    source = (PY_SMALL / "src" / "billing" / "reporting.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    target = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_monthly_statement"
    )
    assert target.end_lineno is not None
    assert target.end_lineno - target.lineno > 60, "function is no longer oversized"


def test_vocabulary_gap_case_is_present() -> None:
    """ "Where do we throttle API calls?" must be answerable via TokenBucket."""
    source = (PY_SMALL / "src" / "billing" / "payments.py").read_text(encoding="utf-8")

    assert "class TokenBucket" in source
    assert "throttl" in source.lower(), "the concept must appear in prose, not the class name"


def test_fixture_has_its_own_tests() -> None:
    """`run_tests` and the test-writer workflow need a green baseline to work against."""
    tests = list((PY_SMALL / "tests").glob("test_*.py"))
    assert tests
