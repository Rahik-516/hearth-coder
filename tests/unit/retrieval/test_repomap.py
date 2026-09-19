"""The repo map — docs/system-design.md §7.7.

The I1 acceptance criterion is that **the map fits its budget within ±5%**, which is two
claims, not one: it must never overrun (a map that evicts the conversation is worse than
no map), and it must not undershoot badly either (a 200-token map handed a 4000-token
budget has thrown away most of what the model needed).

The ranking tests are written against a fixture repository with a deliberate shape — one
widely used module, one leaf nobody calls — so "important" has a checkable meaning rather
than being whatever the algorithm happens to output.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hearth.indexing.pipeline import Indexer
from hearth.retrieval.repomap import MapEntry, RepoMapBuilder, _elide, render_entries
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


@pytest.fixture
def builder(indexed) -> RepoMapBuilder:
    return RepoMapBuilder(indexed)


# ------------------------------------------------------------------- the budget


@pytest.mark.parametrize("budget", [200, 500, 1200, 3000])
def test_the_map_never_exceeds_its_budget(builder: RepoMapBuilder, budget: int) -> None:
    """Overrunning evicts the conversation the map exists to inform."""
    result = builder.build(budget_tokens=budget)

    assert result.estimated_tokens <= budget


@pytest.mark.parametrize("budget", [500, 800, 1000])
def test_the_map_uses_most_of_its_budget(builder: RepoMapBuilder, budget: int) -> None:
    """The other half of ±5%: a map that undershoots wasted the space it was given.

    These budgets are all smaller than py_small's full map (~1090 tokens), so the budget
    is what binds. The `omitted` assertion guards that premise — without it the test would
    keep passing while measuring nothing, which is what happened when it was first written
    against a budget larger than the fixture.
    """
    result = builder.build(budget_tokens=budget)

    assert result.omitted > 0, "this budget should be the binding constraint"
    assert result.estimated_tokens >= budget * 0.95


def test_a_tiny_budget_undershoots_by_at_most_one_definition(
    builder: RepoMapBuilder,
) -> None:
    """±5% is not achievable at every budget, and the reason is worth pinning down.

    Definitions are indivisible, so the map lands within one entry of the budget — around
    10-20 tokens. At a realistic 1000-token budget that is well inside 5%; at 200 tokens a
    single entry is 7% of it, so the guarantee is "within one definition", and ±5% follows
    only once the budget is large enough for that to be true.
    """
    budget = 200
    result = builder.build(budget_tokens=budget)
    largest_omitted = max(
        (len(f"│{entry.signature}\n") // 4 + 1 for entry in builder.build(budget_tokens=100_000).entries),
        default=0,
    )

    assert result.estimated_tokens <= budget
    assert result.estimated_tokens >= budget - largest_omitted


def test_a_budget_larger_than_the_repository_is_not_padded(builder: RepoMapBuilder) -> None:
    """Undershooting is correct when there is simply nothing left to say."""
    result = builder.build(budget_tokens=100_000)

    assert result.omitted == 0
    assert result.estimated_tokens < 100_000


def test_a_zero_budget_yields_an_empty_map(builder: RepoMapBuilder) -> None:
    assert builder.build(budget_tokens=0).empty


def test_the_budget_is_measured_with_the_supplied_estimator(builder: RepoMapBuilder) -> None:
    """A session's estimator is calibrated to its model; the map must use it, not a guess."""

    class Pessimistic:
        def estimate(self, text: str) -> int:
            return len(text)  # one token per character

    result = builder.build(budget_tokens=400, estimator=Pessimistic())

    assert result.estimated_tokens <= 400
    assert len(result.text) <= 400


# ------------------------------------------------------------------- the ranking


def test_the_map_names_the_repositorys_central_module(builder: RepoMapBuilder) -> None:
    result = builder.build(budget_tokens=2000)

    paths = {entry.path for entry in result.entries}
    assert "src/billing/invoice_service.py" in paths


def test_personalization_pulls_a_file_up_the_ranking(builder: RepoMapBuilder) -> None:
    """The same repository maps differently in two sessions — that is the point of it."""
    target = "src/billing/payments.py"

    neutral = builder.build(budget_tokens=2000)
    focused = builder.build(budget_tokens=2000, personalization={target: 5.0})

    def first_index(entries: tuple[MapEntry, ...]) -> int:
        paths = [entry.path for entry in entries]
        return paths.index(target) if target in paths else len(paths)

    assert first_index(focused.entries) <= first_index(neutral.entries)


def test_personalizing_an_unknown_path_is_ignored(builder: RepoMapBuilder) -> None:
    """A mentioned file may have been deleted since; that is not an error."""
    result = builder.build(budget_tokens=1000, personalization={"does/not/exist.py": 9.0})

    assert not result.empty


def test_an_empty_index_maps_to_nothing(tmp_path: Path) -> None:
    connection = connect(tmp_path / "empty.db")
    migrate(connection, database="index")

    result = RepoMapBuilder(connection).build(budget_tokens=1000)

    assert result.empty
    assert result.entries == ()


def test_one_huge_file_cannot_claim_the_whole_map(builder: RepoMapBuilder) -> None:
    """Otherwise the map becomes a table of contents for the largest module."""
    result = builder.build(budget_tokens=1500)

    per_file = {path: 0 for path in {entry.path for entry in result.entries}}
    for entry in result.entries:
        per_file[entry.path] += 1

    assert len(per_file) > 1
    assert max(per_file.values()) < len(result.entries)


# ------------------------------------------------------------------ the rendering


def test_entries_are_grouped_by_file_in_line_order() -> None:
    """Within a file the map reads like the file, not like the ranking."""
    entries = [
        MapEntry("a.py", "second", "function", "def second() -> None", line=40, score=0.9),
        MapEntry("a.py", "first", "function", "def first() -> None", line=10, score=0.5),
    ]

    text = render_entries(entries)

    assert text.index("def first") < text.index("def second")
    assert text.count("a.py:") == 1


def test_a_method_is_shown_under_its_class() -> None:
    """`def finalize(...)` with no owner is ambiguous in any file with two classes."""
    entries = [
        MapEntry(
            "s.py", "finalize", "method", "def finalize(self) -> Invoice",
            line=20, score=0.9, parent="class InvoiceService:",
        )
    ]

    text = render_entries(entries)

    assert "│class InvoiceService:" in text
    assert "│    def finalize(self) -> Invoice" in text


def test_source_outranks_tests(builder: RepoMapBuilder) -> None:
    """Tests reference production symbols constantly, which is what PageRank rewards.

    Without a discount they crowd out the code under test, and an architecture question
    gets a map of the test suite.
    """
    result = builder.build(budget_tokens=1500)
    paths = [entry.path for entry in result.entries]

    source = [p for p in paths if not p.startswith("tests/")]
    assert source, "the map must show source"
    assert paths.index(source[0]) == 0, "the highest-ranked entry should be source"


def test_a_long_signature_is_elided_rather_than_left_whole() -> None:
    """One definition eating a tenth of the budget says what three could have said."""
    long_signature = "def create(" + ", ".join(f"param_{i}: str = 'x'" for i in range(30)) + ") -> None"

    elided = _elide(long_signature)

    assert len(elided) < len(long_signature)
    assert elided.startswith("def create(")
    assert elided.endswith("-> None")


def test_a_multi_line_signature_becomes_one_line() -> None:
    """Multi-line definitions arrive with newlines intact and render ragged in the gutter."""
    ragged = "def f(\n    a: int,\n    b: int,\n) -> int:"

    assert _elide(ragged) == "def f( a: int, b: int, ) -> int:"


def test_rendering_nothing_is_empty_rather_than_a_stray_header() -> None:
    assert render_entries([]) == ""
