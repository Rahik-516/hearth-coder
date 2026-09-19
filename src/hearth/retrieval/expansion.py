"""Graph expansion: the signatures a retrieved chunk needs to be understandable.

docs/system-design.md §7.5. Retrieval returns the chunks that match a query; this adds the
small amount of surrounding structure that makes them *readable*, and nothing more.

The governing constraint is that expansion pays in signatures, never in bodies. A method
retrieved without its class reads as a free function, and a call to `TokenBucket.consume`
means nothing without that method's parameters — but pulling in whole implementations to
supply either would spend the retrieval budget on code the question never asked about, and
push the chunks that actually matched out of the context.

Three kinds, each answering a different failure:

* **Parent skeleton** — a method arrives without the class it belongs to, so the model
  cannot see its siblings or what `self` is.
* **Callee signatures** — the chunk calls something defined elsewhere, and the model has
  to guess the interface it is calling.
* **Caller hints** — for a "where is this used" question, the definition alone is the
  wrong half of the answer.

Everything here reads from ``refs`` and ``symbols``, which the indexer already populated.
No parsing happens at query time: retrieval runs on the interactive path, and a tree-sitter
pass over every retrieved chunk would cost more than the retrieval it is expanding.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from hearth.retrieval.types import FusedResult

#: Caller hints shown for a symbol-intent query (§7.5.3).
MAX_CALLER_HINTS = 5

#: Callee signatures per retrieved chunk. Small on purpose: a chunk that calls twenty
#: things is usually calling the standard library, and expanding all of it would bury the
#: chunk that matched.
MAX_CALLEES_PER_CHUNK = 4

#: Identifiers this short are loop variables and accessors far more often than they are
#: meaningful callees.
_MIN_CALLEE_LENGTH = 4

#: Reference kinds worth resolving. An attribute access rarely names a definition the
#: model needs the signature of, and resolving it produces noise.
_CALLEE_REF_KINDS = ("call", "type")

#: Characters per token, for the expansion budget. Matches repomap's fallback.
_CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class Expansion:
    """One signature-only addition, with why it was added.

    ``reason`` exists for ``hearth search --explain``: expansion changes what the model
    sees without appearing in any retriever's ranking, so without it an operator debugging
    a bad answer cannot tell where a line came from.
    """

    path: str
    line: int
    text: str
    #: parent | callee | caller
    kind: str
    reason: str

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.line}"


def expand(
    connection: sqlite3.Connection,
    results: Sequence[FusedResult],
    *,
    intent: str = "",
    symbols: Iterable[str] = (),
    budget_tokens: int = 400,
) -> list[Expansion]:
    """Expand the final results with parents, callees and (for symbol intent) callers.

    Args:
        results: The chunks that survived fusion and diversity — the ones about to be
            packed into the prompt.
        intent: The query's intent. Caller hints are added only for ``symbol``, since
            "where is X used" is the question they answer and they are noise otherwise.
        symbols: Identifiers the query named, for caller hints.
        budget_tokens: Ceiling for everything added here.

    Returns:
        Expansions in priority order: parents first, then callers, then callees. The
        ordering is the ranking — parents are almost always worth their tokens, callees
        only sometimes — so a caller that truncates this list keeps the valuable half.
    """
    if not results or budget_tokens <= 0:
        return []

    covered = _covered_symbols(results)
    seen: set[tuple[str, int]] = set()

    parents = _parent_skeletons(connection, results, covered=covered, seen=seen)
    callers = (
        _caller_hints(connection, symbols, seen=seen) if intent == "symbol" else []
    )
    callees = _callee_signatures(connection, results, covered=covered, seen=seen)

    return _within_budget([*parents, *callers, *callees], budget_tokens=budget_tokens)


# ------------------------------------------------------------------ parent skeletons


def _parent_skeletons(
    connection: sqlite3.Connection,
    results: Sequence[FusedResult],
    *,
    covered: set[tuple[str, str]],
    seen: set[tuple[str, int]],
) -> list[Expansion]:
    """The enclosing class of every retrieved method, as signatures only.

    Skipped when the class body is already in the results: a chunk that contains the class
    definition has already paid for it, and repeating it would spend budget to say the same
    thing twice.
    """
    expansions: list[Expansion] = []

    for result in results:
        parent = _parent_name(result.symbol_path)
        if parent is None or (result.path, parent) in covered:
            continue

        skeleton = _class_skeleton(connection, path=result.path, class_name=parent)
        if skeleton is None:
            continue

        line, text = skeleton
        if (result.path, line) in seen:
            continue
        seen.add((result.path, line))
        covered.add((result.path, parent))

        expansions.append(
            Expansion(
                path=result.path,
                line=line,
                text=text,
                kind="parent",
                reason=f"class enclosing {result.symbol_path}",
            )
        )

    return expansions


def _class_skeleton(
    connection: sqlite3.Connection, *, path: str, class_name: str
) -> tuple[int, str] | None:
    """A class's own signature plus its members' signatures, bodies elided."""
    rows = connection.execute(
        """
        SELECT s.name AS name, s.kind AS kind, s.signature AS signature,
               s.start_line AS line, s.parent_id AS parent_id, s.id AS id
        FROM symbols s
        JOIN files f ON f.id = s.file_id
        WHERE f.path = ?
          AND (s.name = ? OR s.parent_id = (
                SELECT s2.id FROM symbols s2
                JOIN files f2 ON f2.id = s2.file_id
                WHERE f2.path = ? AND s2.name = ?
                LIMIT 1
              ))
        ORDER BY s.start_line
        """,
        (path, class_name, path, class_name),
    ).fetchall()

    if not rows:
        return None

    header: str | None = None
    header_line = 0
    members: list[str] = []

    for row in rows:
        signature = _one_line(str(row["signature"] or row["name"]))
        if str(row["name"]) == class_name and row["parent_id"] is None:
            header, header_line = signature, int(row["line"])
        else:
            members.append(f"    {signature}")

    if header is None:
        return None
    return header_line, "\n".join([header, *members])


# -------------------------------------------------------------- callee signatures


def _callee_signatures(
    connection: sqlite3.Connection,
    results: Sequence[FusedResult],
    *,
    covered: set[tuple[str, str]],
    seen: set[tuple[str, int]],
) -> list[Expansion]:
    """Signatures of what the retrieved chunks call, when it is defined in this repository.

    Resolution is by name, which is approximate: two classes may define ``save``. Rather
    than guess, a name defined in several places is skipped — an arbitrary pick would be
    wrong about as often as it is right, and a wrong signature is worse than none, because
    the model has no way to tell it is wrong.
    """
    expansions: list[Expansion] = []

    for result in results:
        names = _referenced_names(connection, result)
        added = 0

        for name in names:
            if added >= MAX_CALLEES_PER_CHUNK:
                break

            definition = _unique_definition(connection, name, exclude_path=result.path)
            if definition is None:
                continue

            path, line, signature = definition
            if (path, line) in seen or (path, name) in covered:
                continue
            seen.add((path, line))

            expansions.append(
                Expansion(
                    path=path,
                    line=line,
                    text=signature,
                    kind="callee",
                    reason=f"called from {result.citation}",
                )
            )
            added += 1

    return expansions


def _referenced_names(connection: sqlite3.Connection, result: FusedResult) -> list[str]:
    """Identifiers the chunk references, most-repeated first.

    Taken from ``refs`` by line range rather than by re-parsing the chunk text: the indexer
    already resolved these with a real grammar, and a regex over the text at query time
    would disagree with it in exactly the confusing cases (a name in a comment or string).
    """
    rows = connection.execute(
        """
        SELECT r.name AS name, COUNT(*) AS hits
        FROM refs r
        JOIN files f ON f.id = r.file_id
        WHERE f.path = ?
          AND r.line BETWEEN ? AND ?
          AND r.kind IN (?, ?)
        GROUP BY r.name
        ORDER BY hits DESC, r.name
        """,
        (result.path, result.start_line, result.end_line, *_CALLEE_REF_KINDS),
    ).fetchall()

    return [
        str(row["name"])
        for row in rows
        if len(str(row["name"]).strip("_")) >= _MIN_CALLEE_LENGTH
    ]


def _unique_definition(
    connection: sqlite3.Connection, name: str, *, exclude_path: str
) -> tuple[str, int, str] | None:
    """Where ``name`` is defined, when exactly one file outside ``exclude_path`` defines it."""
    rows = connection.execute(
        """
        SELECT f.path AS path, s.start_line AS line, s.signature AS signature, s.name AS name
        FROM symbols s
        JOIN files f ON f.id = s.file_id
        WHERE s.name = ? AND f.path != ?
        LIMIT 3
        """,
        (name, exclude_path),
    ).fetchall()

    if len(rows) != 1:
        return None

    row = rows[0]
    return str(row["path"]), int(row["line"]), _one_line(str(row["signature"] or row["name"]))


# -------------------------------------------------------------------- caller hints


def _caller_hints(
    connection: sqlite3.Connection,
    symbols: Iterable[str],
    *,
    seen: set[tuple[str, int]],
) -> list[Expansion]:
    """Where a symbol is referenced, as ``path:line — enclosing function`` (§7.5.3).

    Only for symbol intent. "Where is `finalize` used" is answered by the call sites, and
    returning the definition alone answers the opposite question.
    """
    expansions: list[Expansion] = []

    for symbol in symbols:
        rows = connection.execute(
            """
            SELECT f.path AS path, r.line AS line
            FROM refs r
            JOIN files f ON f.id = r.file_id
            WHERE r.name = ?
            ORDER BY f.path, r.line
            LIMIT ?
            """,
            (symbol, MAX_CALLER_HINTS),
        ).fetchall()

        for row in rows:
            path, line = str(row["path"]), int(row["line"])
            if (path, line) in seen:
                continue
            seen.add((path, line))

            enclosing = _enclosing_symbol(connection, path=path, line=line)
            where = f" — {enclosing}" if enclosing else ""
            expansions.append(
                Expansion(
                    path=path,
                    line=line,
                    text=f"{path}:{line}{where}",
                    kind="caller",
                    reason=f"references {symbol}",
                )
            )

    return expansions


def _enclosing_symbol(connection: sqlite3.Connection, *, path: str, line: int) -> str | None:
    """The innermost definition containing a line.

    Innermost, so a call inside a method is attributed to the method rather than its class:
    ``finalize`` locates the reader far better than ``InvoiceService`` does.
    """
    row = connection.execute(
        """
        SELECT s.name AS name
        FROM symbols s
        JOIN files f ON f.id = s.file_id
        WHERE f.path = ? AND s.start_line <= ? AND s.end_line >= ?
        ORDER BY (s.end_line - s.start_line)
        LIMIT 1
        """,
        (path, line, line),
    ).fetchone()

    return str(row["name"]) if row else None


# ------------------------------------------------------------------------ helpers


def _covered_symbols(results: Sequence[FusedResult]) -> set[tuple[str, str]]:
    """``(path, symbol)`` pairs where a retrieved chunk *is* that symbol.

    Only the leaf of each symbol path counts. Marking every ancestor as covered would read
    "InvoiceService > finalize" as proof that the class is in context, when the chunk is
    only one of its methods — and then suppress the very skeleton §7.5.1 exists to add.
    """
    covered: set[tuple[str, str]] = set()
    for result in results:
        if not result.symbol_path:
            continue
        leaf = result.symbol_path.split(SYMBOL_PATH_SEPARATOR)[-1].strip()
        if leaf:
            covered.add((result.path, leaf))
    return covered


#: How the chunker joins nested symbol names: ``InvoiceService > finalize``.
SYMBOL_PATH_SEPARATOR = " > "


def _parent_name(symbol_path: str | None) -> str | None:
    """The innermost enclosing symbol, or None for a top-level definition."""
    if not symbol_path or SYMBOL_PATH_SEPARATOR not in symbol_path:
        return None
    return symbol_path.rsplit(SYMBOL_PATH_SEPARATOR, 1)[0].split(SYMBOL_PATH_SEPARATOR)[-1].strip() or None


def _one_line(signature: str) -> str:
    return " ".join(signature.split())


def _within_budget(expansions: list[Expansion], *, budget_tokens: int) -> list[Expansion]:
    """Take expansions in priority order, skipping any that will not fit.

    Priority order is preserved — parents are offered the budget before callees — but an
    item too large to fit is stepped over rather than ending the selection. A class
    skeleton can be several hundred tokens; when it does not fit, returning nothing would
    waste a budget that still had room for three useful callee signatures.

    Not a best-fit search: reordering to pack the budget tightly would let a handful of
    cheap callees displace a parent skeleton worth more than all of them.
    """
    kept: list[Expansion] = []
    spent = 0

    for expansion in expansions:
        cost = int(len(expansion.text) / _CHARS_PER_TOKEN) + 1
        if spent + cost > budget_tokens:
            continue
        spent += cost
        kept.append(expansion)

    return kept
