"""The retrieval engine: the public entry point for finding context.

Runs query analysis, then the retrievers, then weighted RRF fusion, then diversity capping
(docs/system-design.md §7).

**Retrieval degrades rather than fails.** If embeddings are missing, incomplete, or the
provider is unreachable, dense search is simply skipped and the lexical, symbol and path
retrievers carry the query. That is what makes a large repository usable a minute into
indexing instead of after a full embed — an M2 acceptance criterion — and it is why
``RetrievalResult`` reports coverage: partial dense results should be visible, not
silently worse.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from hearth.retrieval.expansion import Expansion, expand
from hearth.retrieval.fusion import FusionContext, FusionWeights, fuse
from hearth.retrieval.query_analysis import Intent, QueryAnalysis, analyze
from hearth.retrieval.retrievers import (
    DEFAULT_TOP_K,
    bm25_search,
    dense_search,
    path_search,
    symbol_search,
)
from hearth.retrieval.types import Candidate, FusedResult, Retriever
from hearth.storage.db import connect
from hearth.storage.vector_index import NumpyVectorIndex

#: Which retrievers run. "hybrid" is the product; the others exist for the eval.
logger = logging.getLogger(__name__)

RetrievalMode = Literal["hybrid", "lexical", "dense"]

#: Sentinel for the not-yet-computed database path.
_UNSET: Any = object()

#: Chunks per file in the final list, unless the file was explicitly named.
DEFAULT_MAX_PER_FILE = 3

#: Tokens graph expansion may spend on signatures. Deliberately a small fraction of the
#: retrieval budget: expansion supports the chunks that matched, and a large allowance
#: would let supporting context outweigh the answer it is supporting (§7.5).
DEFAULT_EXPANSION_BUDGET = 400


@dataclass
class RetrievalResult:
    """Fused results plus everything `--explain` needs."""

    query: str
    analysis: QueryAnalysis
    results: list[FusedResult] = field(default_factory=list)
    per_retriever: dict[Retriever, list[Candidate]] = field(default_factory=dict)
    #: Signature-only additions from graph expansion (§7.5). Separate from ``results``
    #: because they were never ranked — they are context the chunks needed, not matches.
    expansions: list[Expansion] = field(default_factory=list)

    #: (embedded, total) distinct chunk texts. Dense results are partial below 1.0.
    coverage: tuple[int, int] = (0, 0)
    dense_used: bool = False
    dense_skipped_reason: str | None = None
    duration_ms: float = 0.0

    @property
    def coverage_fraction(self) -> float:
        embedded, total = self.coverage
        return 1.0 if total == 0 else embedded / total

    @property
    def is_fully_embedded(self) -> bool:
        return self.coverage_fraction >= 1.0


class RetrievalEngine:
    """Hybrid retrieval over one index."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        vector_index: NumpyVectorIndex | None = None,
        weights: FusionWeights | None = None,
        max_per_file: int = DEFAULT_MAX_PER_FILE,
        expansion_budget: int = DEFAULT_EXPANSION_BUDGET,
    ) -> None:
        self._connection = connection
        self._vector_index = vector_index
        self._expansion_budget = expansion_budget
        self._weights = weights or FusionWeights()
        self._max_per_file = max_per_file
        self._database_path_cache: str | None = _UNSET

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyse a query, resolving identifiers against the symbol table."""
        return analyze(query, known_symbol_lookup=self._known_symbols)

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 10,
        query_vector: np.ndarray | None = None,
        context: FusionContext | None = None,
        top_k: int = DEFAULT_TOP_K,
        mode: RetrievalMode = "hybrid",
    ) -> RetrievalResult:
        """Retrieve context for a query.

        ``query_vector`` is supplied by the caller rather than computed here, because
        embedding is async and this layer is deliberately synchronous.

        ``mode`` selects which retrievers run:

        * ``hybrid`` — all four, fused. The product behaviour.
        * ``lexical`` — BM25, symbol and path. What happens before embeddings finish.
        * ``dense`` — embeddings alone.

        The two single-signal modes exist for the eval, which has to show that hybrid
        beats *both* of its halves rather than merely working.
        """
        import time

        started = time.perf_counter()
        analysis = self.analyze_query(query)

        candidate_lists: list[Sequence[Candidate]] = []
        per_retriever: dict[Retriever, list[Candidate]] = {}

        dense_used = False
        skipped_reason: str | None = None
        coverage = (0, 0)

        if self._vector_index is not None:
            coverage = self._vector_index.coverage()

        run_dense = True
        if mode == "lexical":
            skipped_reason, run_dense = "lexical-only requested", False
        elif self._vector_index is None:
            skipped_reason, run_dense = "no vector index", False
        elif query_vector is None:
            skipped_reason, run_dense = "no query embedding available", False
        elif self._vector_index.count() == 0:
            skipped_reason, run_dense = "no embeddings yet", False

        # The retrievers are independent, so they run concurrently (the fan-out in
        # docs/system-design.md §7). This is a real win rather than bookkeeping: sqlite3
        # and NumPy both release the GIL, so BM25 (the slowest) overlaps the dense matmul
        # instead of adding to it. Each thread gets its own read-only connection, because
        # SQLite serializes work on a shared one and nothing would overlap.
        tasks: dict[Retriever, Any] = {}
        if mode != "dense":
            tasks[Retriever.BM25] = lambda c: bm25_search(c, analysis, limit=top_k)
            tasks[Retriever.SYMBOL] = lambda c: symbol_search(c, analysis, limit=top_k)
            tasks[Retriever.PATH] = lambda c: path_search(c, analysis, limit=top_k)
        if run_dense and query_vector is not None and self._vector_index is not None:
            index = self._vector_index
            tasks[Retriever.DENSE] = lambda c: dense_search(
                c,
                index,
                query_vector,
                limit=top_k,
                languages=analysis.languages,
            )

        found_by = self._run_retrievers(tasks)

        for name, found in found_by.items():
            per_retriever[name] = list(found)
            if found:
                candidate_lists.append(found)
                if name is Retriever.DENSE:
                    dense_used = True

        if run_dense and not dense_used and skipped_reason is None:
            skipped_reason = "dense search returned nothing"

        fused = fuse(
            candidate_lists,
            intent=analysis.intent,
            weights=self._weights,
            context=context,
            limit=limit * 4,
        )
        diversified = self._apply_diversity(fused, analysis, limit=limit)
        expansions = self._expand(diversified, analysis)

        return RetrievalResult(
            query=query,
            analysis=analysis,
            results=diversified,
            expansions=expansions,
            per_retriever=per_retriever,
            coverage=coverage,
            dense_used=dense_used,
            dense_skipped_reason=skipped_reason,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    # -------------------------------------------------------------- internals

    def _run_retrievers(
        self, tasks: dict[Retriever, Callable[[sqlite3.Connection], list[Candidate]]]
    ) -> dict[Retriever, list[Candidate]]:
        """Run retrievers, concurrently when a second connection is available.

        Falls back to sequential execution when extra connections cannot be opened — an
        in-memory database, or a path SQLite will not reopen. Correctness never depends on
        the parallel path.
        """
        if len(tasks) <= 1:
            return {name: fn(self._connection) for name, fn in tasks.items()}

        database = self._database_path()
        if database is None:
            return {name: fn(self._connection) for name, fn in tasks.items()}

        results: dict[Retriever, list[Candidate]] = {}
        names = list(tasks)

        # Dense must run on this thread, because it reads through the vector index, which
        # holds the engine's own connection — and sqlite3 rejects cross-thread use. It is
        # also the slowest after BM25, so keeping it here overlaps the two heaviest
        # retrievers rather than serialising them.
        primary = Retriever.DENSE if Retriever.DENSE in tasks else names[0]
        rest = [name for name in names if name != primary]

        with ThreadPoolExecutor(max_workers=max(1, len(rest))) as pool:
            futures = {pool.submit(self._with_connection, database, tasks[name]): name for name in rest}
            results[primary] = tasks[primary](self._connection)

            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except sqlite3.Error:
                    # One retriever failing must not lose the others' results — but it
                    # must not vanish either. Swallowing this silently is how a broken
                    # dense retriever once looked like a merely unhelpful one.
                    logger.exception("retriever %s failed", name.value)
                    results[name] = []

        return {name: results.get(name, []) for name in names}

    @staticmethod
    def _with_connection(
        database: str, fn: Callable[[sqlite3.Connection], list[Candidate]]
    ) -> list[Candidate]:
        connection = connect(database, read_only=True)
        try:
            return fn(connection)
        finally:
            connection.close()

    def _database_path(self) -> str | None:
        """The on-disk path behind this connection, or None for in-memory databases."""
        if self._database_path_cache is _UNSET:
            path: str | None = None
            try:
                for row in self._connection.execute("PRAGMA database_list"):
                    if row[1] == "main" and row[2]:
                        path = str(row[2])
                        break
            except sqlite3.Error:  # pragma: no cover - defensive
                path = None
            self._database_path_cache = path
        return self._database_path_cache

    def _expand(
        self, results: list[FusedResult], analysis: QueryAnalysis
    ) -> list[Expansion]:
        """Add the signatures the final chunks need to be readable (§7.5).

        Runs after diversity rather than before, because expansion is defined over the
        chunks that will actually be sent: expanding candidates that are about to be cut
        would spend the budget on context for text the model never sees.

        Failures are swallowed. Expansion is an enhancement to a result that is already
        correct without it, and a malformed symbol table should cost the answer some
        signatures rather than cost it the retrieval.
        """
        if not results:
            return []

        try:
            return expand(
                self._connection,
                results,
                intent=str(analysis.intent),
                symbols=analysis.identifiers,
                budget_tokens=self._expansion_budget,
            )
        except Exception:
            return []

    def _apply_diversity(
        self, results: list[FusedResult], analysis: QueryAnalysis, *, limit: int
    ) -> list[FusedResult]:
        """Cap chunks per file, so one large file cannot monopolise the budget.

        A file the query named explicitly is exempt: if someone asks about
        ``invoice_service.py``, filling the result list from it is the right answer.
        """
        named = {p.lower() for p in analysis.paths}
        per_file: dict[str, int] = {}
        kept: list[FusedResult] = []

        for result in results:
            explicit = any(fragment in result.path.lower() for fragment in named)
            count = per_file.get(result.path, 0)

            if not explicit and count >= self._max_per_file:
                continue

            per_file[result.path] = count + 1
            kept.append(result)
            if len(kept) >= limit:
                break

        return kept

    def _known_symbols(self, identifiers: list[str]) -> list[str]:
        """Which candidate identifiers actually exist as symbols."""
        if not identifiers:
            return []

        leaves = [name.rsplit(".", 1)[-1] for name in identifiers]
        placeholders = ",".join("?" * len(leaves))
        rows = self._connection.execute(
            f"SELECT DISTINCT name FROM symbols WHERE name IN ({placeholders}) COLLATE NOCASE",  # noqa: S608 - placeholders only
            tuple(leaves),
        ).fetchall()

        found = {str(row[0]).lower() for row in rows}
        return [name for name in identifiers if name.rsplit(".", 1)[-1].lower() in found]


def intent_of(query: str) -> Intent:
    """Analyse a query without an index. Used by tests and tooling."""
    return analyze(query).intent
