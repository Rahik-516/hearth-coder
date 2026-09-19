"""Graph expansion — docs/system-design.md §7.5.

Expansion is the part of retrieval that adds context nothing asked for, so the tests are
mostly about restraint: it pays in signatures and never in bodies, it does not repeat what
the retrieved chunks already contain, and it stays inside its budget.

The one that matters most is the ambiguity test. Resolving a callee by name is approximate,
and a *wrong* signature is worse than no signature — the model cannot tell it is wrong, so
it reasons confidently against an interface that does not exist.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hearth.indexing.pipeline import Indexer
from hearth.retrieval.expansion import Expansion, expand
from hearth.retrieval.types import FusedResult
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "repos"


@pytest.fixture
def indexed(tmp_path: Path):
    workspace = tmp_path / "py_small"
    shutil.copytree(FIXTURES / "py_small", workspace)
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    Indexer(root=workspace, repository=IndexRepository(connection)).run()
    return connection


def result_for(connection, *, name: str) -> FusedResult:
    """A fused result standing for one symbol, built from the symbols table.

    Built from `symbols` rather than by looking up a chunk with a matching `symbol_path`,
    because whether a class splits into per-method chunks depends on its size: py_small's
    classes fit in one chunk each, so no `Parent > child` chunk exists there, while a
    larger repository produces them constantly. Expansion consumes a path, a symbol path
    and a line range, and this supplies exactly that for whichever symbol the test names.
    """
    row = connection.execute(
        """
        SELECT f.path AS path, s.kind AS kind, s.start_line AS start, s.end_line AS end,
               s.name AS name, parent.name AS parent
        FROM symbols s
        JOIN files f ON f.id = s.file_id
        LEFT JOIN symbols parent ON parent.id = s.parent_id
        WHERE s.name = ?
        LIMIT 1
        """,
        (name,),
    ).fetchone()
    assert row is not None, f"fixture has no symbol named {name}"

    parent = row["parent"]
    symbol_path = f"{parent} > {row['name']}" if parent else str(row["name"])
    return FusedResult(
        chunk_id=0,
        path=str(row["path"]),
        kind=str(row["kind"]),
        symbol_path=symbol_path,
        start_line=int(row["start"]),
        end_line=int(row["end"]),
        text="",
    )


def method_names(connection, *, limit: int = 8) -> list[str]:
    """Symbols that are nested inside another symbol."""
    rows = connection.execute(
        """
        SELECT s.name AS name FROM symbols s
        WHERE s.parent_id IS NOT NULL
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [str(row["name"]) for row in rows]


# ------------------------------------------------------------------- parent skeletons


def test_a_retrieved_method_gains_its_class(indexed) -> None:
    """A method without its class reads as a free function."""
    result = result_for(indexed, name="finalize")

    expansions = expand(indexed, [result], budget_tokens=500)

    parents = [e for e in expansions if e.kind == "parent"]
    assert parents, f"no parent skeleton for {result.symbol_path}"
    assert "InvoiceService" in parents[0].text


def test_the_parent_skeleton_carries_signatures_not_bodies(indexed) -> None:
    """The whole point: structure without paying for implementations."""
    result = result_for(indexed, name="finalize")

    parent = next(e for e in expand(indexed, [result], budget_tokens=500) if e.kind == "parent")

    assert "return" not in parent.text, "a body leaked into the skeleton"
    assert parent.text.count("\n") < 40


def test_a_top_level_function_gains_no_parent(indexed) -> None:
    """Nothing encloses it, so there is nothing to add."""
    expansions = expand(indexed, [result_for(indexed, name="build_monthly_statement")], budget_tokens=500)

    assert not [e for e in expansions if e.kind == "parent"]


# -------------------------------------------------------------- callee signatures


def test_callees_are_signatures_from_other_files(indexed) -> None:
    result = result_for(indexed, name="finalize")

    callees = [e for e in expand(indexed, [result], budget_tokens=800) if e.kind == "callee"]

    for callee in callees:
        assert callee.path != result.path, "a callee in the same file is already in view"
        assert "\n" not in callee.text, "callees are one-line signatures"


def test_an_ambiguous_name_is_not_resolved(indexed) -> None:
    """Two classes may define `save`. A wrong signature is worse than none.

    Checked structurally rather than by naming a symbol: every callee that *was* resolved
    must be the only definition of that name outside its own file.
    """
    results = [result_for(indexed, name=n) for n in method_names(indexed, limit=6)]

    callees = [e for e in expand(indexed, results, budget_tokens=2000) if e.kind == "callee"]

    for callee in callees:
        name = callee.reason  # "called from <citation>" — resolve by path/line instead
        assert name  # keep the message useful if the next assertion fires
        defining = indexed.execute(
            """
            SELECT COUNT(DISTINCT f.path) AS files
            FROM symbols s JOIN files f ON f.id = s.file_id
            WHERE s.name = (
                SELECT s2.name FROM symbols s2
                JOIN files f2 ON f2.id = s2.file_id
                WHERE f2.path = ? AND s2.start_line = ?
                LIMIT 1
            )
            """,
            (callee.path, callee.line),
        ).fetchone()
        assert int(defining["files"]) == 1, f"{callee.citation} resolved an ambiguous name"


def test_expansion_does_not_repeat_the_retrieved_chunks(indexed) -> None:
    """Budget spent restating what is already in context is budget wasted."""
    results = [result_for(indexed, name=n) for n in method_names(indexed, limit=4)]

    expansions = expand(indexed, results, budget_tokens=2000)

    retrieved = {(r.path, r.start_line) for r in results}
    assert not [e for e in expansions if (e.path, e.line) in retrieved]


# -------------------------------------------------------------------- caller hints


def test_caller_hints_appear_only_for_symbol_intent(indexed) -> None:
    """"Where is X used" is the question they answer; elsewhere they are noise."""
    result = result_for(indexed, name="finalize")

    neutral = expand(indexed, [result], intent="how", symbols=["finalize"], budget_tokens=800)
    targeted = expand(indexed, [result], intent="symbol", symbols=["finalize"], budget_tokens=800)

    assert not [e for e in neutral if e.kind == "caller"]
    assert [e for e in targeted if e.kind == "caller"]


def test_a_caller_hint_names_its_enclosing_function(indexed) -> None:
    """`path:line` alone makes the reader open the file to learn where they landed."""
    result = result_for(indexed, name="finalize")

    hints = [
        e
        for e in expand(indexed, [result], intent="symbol", symbols=["finalize"], budget_tokens=800)
        if e.kind == "caller"
    ]

    assert hints
    assert any("—" in hint.text for hint in hints), "no hint named an enclosing symbol"


def test_caller_hints_are_capped(indexed) -> None:
    """§7.5.3 says the top 5; an unbounded list would swamp the chunks that matched."""
    result = result_for(indexed, name="finalize")

    hints = [
        e
        for e in expand(indexed, [result], intent="symbol", symbols=["Decimal"], budget_tokens=4000)
        if e.kind == "caller"
    ]

    assert len(hints) <= 5


# ------------------------------------------------------------------------ budget


def test_expansion_stays_within_its_budget(indexed) -> None:
    results = [result_for(indexed, name=n) for n in method_names(indexed, limit=8)]

    expansions = expand(indexed, results, intent="symbol", symbols=["finalize"], budget_tokens=120)

    spent = sum(int(len(e.text) / 4) + 1 for e in expansions)
    assert spent <= 120


def test_a_zero_budget_expands_nothing(indexed) -> None:
    result = result_for(indexed, name="finalize")

    assert expand(indexed, [result], budget_tokens=0) == []


def test_no_results_expand_to_nothing(indexed) -> None:
    assert expand(indexed, [], budget_tokens=500) == []


def test_the_parent_is_offered_the_budget_before_callees(indexed) -> None:
    """Priority order is the ranking: a parent skeleton outearns several callees.

    The budget here fits the parent and only some of the callees, which is the case where
    the ordering decides something. A budget too small for the parent is a different
    situation — see the test below.
    """
    result = result_for(indexed, name="finalize")
    parent = next(e for e in expand(indexed, [result], budget_tokens=4000) if e.kind == "parent")
    parent_cost = int(len(parent.text) / 4) + 1

    expansions = expand(indexed, [result], budget_tokens=parent_cost + 10)

    assert expansions[0].kind == "parent", "the parent must get the budget first"


def test_an_unaffordable_parent_does_not_waste_the_whole_budget(indexed) -> None:
    """Skipped, not fatal: a budget with no room for the class still has room for callees.

    Returning nothing because the most valuable item did not fit would spend the budget on
    silence.
    """
    result = result_for(indexed, name="finalize")

    expansions = expand(indexed, [result], budget_tokens=20)

    assert not [e for e in expansions if e.kind == "parent"], "premise: the class cannot fit"
    assert expansions, "the remaining budget should still buy something"


def test_every_expansion_explains_itself(indexed) -> None:
    """Expansion changes what the model sees without appearing in any retriever's rank."""
    result = result_for(indexed, name="finalize")

    for expansion in expand(indexed, [result], intent="symbol", symbols=["finalize"], budget_tokens=800):
        assert isinstance(expansion, Expansion)
        assert expansion.reason.strip()
        assert expansion.kind in {"parent", "callee", "caller"}
