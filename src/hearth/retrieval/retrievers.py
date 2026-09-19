"""The four retrievers (docs/system-design.md §7.2).

Each covers a question class the others handle badly, which is the whole argument for
hybrid retrieval (docs/system-design.md §1.2):

* **BM25** — exact terms, error strings, config keys
* **Dense** — conceptual questions, where the words in the question never appear in the code
* **Symbol** — "where is X defined"
* **Path** — "the auth middleware file"

Every retriever returns ranked ``Candidate`` lists. Fusion consumes ranks, not scores, so
retrievers do not need comparable score scales — which is exactly why RRF is used.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import numpy as np

from hearth.retrieval.query_analysis import QueryAnalysis
from hearth.retrieval.types import Candidate, Retriever
from hearth.storage.vector_index import NumpyVectorIndex

#: Rows fetched per retriever before fusion. Generous: fusion is cheap, and a candidate
#: missing here can never be recovered later.
DEFAULT_TOP_K = 40


def bm25_search(
    connection: sqlite3.Connection,
    analysis: QueryAnalysis,
    *,
    limit: int = DEFAULT_TOP_K,
) -> list[Candidate]:
    """Lexical search over the contentless FTS5 index.

    Column weights favour a symbol-name hit over a body mention. Terms are OR'd and each
    is quoted, because FTS5 treats ``-`` and ``:`` as operators and a bare identifier is a
    syntax error rather than a miss.
    """
    terms = analysis.terms or [t for t in analysis.raw.split() if t]
    if not terms:
        return []

    terms = select_selective_terms(connection, terms)
    match = " OR ".join(f'"{term}"' for term in terms if term)
    if not match:
        return []

    try:
        rows = connection.execute(
            """
            SELECT c.id, f.path, c.kind, c.symbol_path, c.start_line, c.end_line,
                   c.text, f.language, bm25(chunks_fts, 1.0, 4.0, 2.0) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN files  f ON f.id = c.file_id
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # A query FTS5 cannot parse is a miss, not a crash: retrieval must degrade rather
        # than take down the turn.
        return []

    return [_candidate(row, rank, Retriever.BM25, -float(row["score"])) for rank, row in enumerate(rows)]


def dense_search(
    connection: sqlite3.Connection,
    vector_index: NumpyVectorIndex,
    query_vector: np.ndarray,
    *,
    limit: int = DEFAULT_TOP_K,
    languages: Sequence[str] = (),
) -> list[Candidate]:
    """Semantic search over chunk embeddings.

    Vectors are keyed by content hash, and several chunks can share one. Each hash is
    resolved to its chunks, keeping the best-scoring chunk per hash so a duplicated
    docstring does not occupy the whole result list.
    """
    if vector_index.count() == 0:
        return []

    candidate_hashes = _hashes_for_languages(connection, languages) if languages else None
    hits = vector_index.search(query_vector, k=limit * 2, candidates=candidate_hashes)
    if not hits:
        return []

    by_hash = {hit.embed_text_hash: hit.score for hit in hits}
    placeholders = ",".join("?" * len(by_hash))
    rows = connection.execute(
        f"""
        SELECT c.id, f.path, c.kind, c.symbol_path, c.start_line, c.end_line,
               c.text, f.language, c.embed_text_hash
        FROM chunks c JOIN files f ON f.id = c.file_id
        WHERE c.embed_text_hash IN ({placeholders})
        """,  # noqa: S608 - placeholders only
        tuple(by_hash),
    ).fetchall()

    scored = sorted(
        ((by_hash[str(row["embed_text_hash"])], row) for row in rows),
        key=lambda pair: pair[0],
        reverse=True,
    )

    seen_hashes: set[str] = set()
    candidates: list[Candidate] = []
    for score, row in scored:
        text_hash = str(row["embed_text_hash"])
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)
        candidates.append(_candidate(row, len(candidates), Retriever.DENSE, score))
        if len(candidates) >= limit:
            break
    return candidates


def symbol_search(
    connection: sqlite3.Connection,
    analysis: QueryAnalysis,
    *,
    limit: int = DEFAULT_TOP_K,
) -> list[Candidate]:
    """Find chunks containing definitions whose name matches a query identifier.

    Tried in decreasing confidence: exact case-sensitive, then case-insensitive, then
    prefix. Stopping at the first tier that produces hits avoids a vague prefix match
    outranking an exact one.
    """
    names = analysis.identifiers or analysis.known_symbols
    if not names:
        return []

    leaf_names = [name.rsplit(".", 1)[-1] for name in names]
    rows: list[sqlite3.Row] = []

    for clause, params in (
        ("s.name = ?", leaf_names),
        ("s.name = ? COLLATE NOCASE", leaf_names),
        ("s.name LIKE ? || '%' COLLATE NOCASE", leaf_names),
    ):
        where = " OR ".join(clause for _ in params)
        rows = connection.execute(
            f"""
            SELECT DISTINCT c.id, f.path, c.kind, c.symbol_path, c.start_line, c.end_line,
                   c.text, f.language, s.start_line AS def_line
            FROM symbols s
            JOIN files f  ON f.id = s.file_id
            JOIN chunks c ON c.file_id = f.id
                         AND s.start_line BETWEEN c.start_line AND c.end_line
            WHERE {where}
            ORDER BY LENGTH(c.text)
            LIMIT ?
            """,  # noqa: S608 - clause built from a fixed template
            (*params, limit),
        ).fetchall()
        if rows:
            break

    return [_candidate(row, rank, Retriever.SYMBOL, 1.0 / (rank + 1)) for rank, row in enumerate(rows)]


def path_search(
    connection: sqlite3.Connection,
    analysis: QueryAnalysis,
    *,
    limit: int = DEFAULT_TOP_K,
) -> list[Candidate]:
    """Match chunks by path.

    When the query names a path, only that path is used. Falling back to generic terms in
    that case is actively harmful: "src/billing/payments.py" also yields the terms `src`
    and `billing`, which match every sibling file and bury the one that was asked for.

    Terms are used only when no explicit path was given, so "invoice service" still finds
    ``invoice_service.py`` even when nothing in that file's body matches.
    """
    explicit = [p.lower() for p in analysis.paths]
    fragments = explicit or [term for term in analysis.terms if len(term) >= 4]
    if not fragments:
        return []

    # Match against `files` first, not the chunks join. A leading-wildcard LIKE cannot use
    # an index, so it scans whatever it is given — and there are three orders of magnitude
    # more chunks than files. Scanning 2K paths instead of 100K chunk rows took this
    # retriever from ~72ms to sub-millisecond at benchmark scale.
    where = " OR ".join("LOWER(path) LIKE '%' || ? || '%'" for _ in fragments)
    file_rows = connection.execute(
        f"SELECT id, path, language FROM files WHERE {where}",  # noqa: S608 - fixed template
        tuple(fragments),
    ).fetchall()
    if not file_rows:
        return []

    # Rank files by match quality, not by path length. Sorting by length made `errors.py`
    # outrank `payments.py` for the query "src/billing/payments.py".
    ranked_files = sorted(
        file_rows,
        key=lambda row: (-_path_match_quality(str(row["path"]), fragments), len(str(row["path"]))),
    )[: max(1, limit)]

    file_ids = [int(row["id"]) for row in ranked_files]
    order = {file_id: position for position, file_id in enumerate(file_ids)}
    placeholders = ",".join("?" * len(file_ids))

    rows = connection.execute(
        f"""
        SELECT c.id, f.path, c.kind, c.symbol_path, c.start_line, c.end_line,
               c.text, f.language, c.file_id
        FROM chunks c JOIN files f ON f.id = c.file_id
        WHERE c.file_id IN ({placeholders})
        ORDER BY c.start_line
        LIMIT ?
        """,  # noqa: S608 - placeholders only
        (*file_ids, limit * 4),
    ).fetchall()

    # Preserve file ranking: chunks from the best-matching file come first.
    ordered = sorted(rows, key=lambda row: (order[int(row["file_id"])], int(row["start_line"])))

    return [
        _candidate(row, rank, Retriever.PATH, 1.0 / (rank + 1)) for rank, row in enumerate(ordered[:limit])
    ]


def _path_match_quality(path: str, fragments: Sequence[str]) -> float:
    """How strongly a path matches the query fragments, 0.0 to 1.0.

    An exact or suffix match beats a substring match, which beats a match on a parent
    directory. Coverage breaks ties, so a fragment spanning most of the path wins over one
    matching a single segment.
    """
    lowered = path.lower()
    best = 0.0

    for fragment in fragments:
        if not fragment:
            continue
        if lowered == fragment:
            best = max(best, 1.0)
        elif lowered.endswith(fragment):
            best = max(best, 0.9)
        elif fragment in lowered:
            coverage = len(fragment) / max(len(lowered), 1)
            best = max(best, 0.3 + 0.5 * coverage)
    return best


# ------------------------------------------------------------------- helpers


#: A term appearing in more than this fraction of chunks is treated as a stopword for
#: this corpus. Such a term barely changes BM25's ranking but forces it to score nearly
#: every row, which measured as ~95ms of a 150ms budget at 100K chunks.
MAX_TERM_DOCUMENT_FRACTION = 0.25

#: Never reduce a query below this many terms. A query made entirely of common words
#: ("how does the service handle errors") must still return something.
MIN_TERMS = 2


def select_selective_terms(
    connection: sqlite3.Connection,
    terms: Sequence[str],
    *,
    max_fraction: float = MAX_TERM_DOCUMENT_FRACTION,
) -> list[str]:
    """Drop terms that appear in most of the corpus.

    Identifier expansion guarantees these exist: ``InvoiceService`` becomes ``invoice``,
    ``service`` and ``invoiceservice``, and ``service`` is in most files of a service
    codebase. Keeping it costs almost the whole retrieval budget and changes the ranking
    very little, because a term matching everything discriminates between nothing.

    Falls back to the original terms whenever frequencies are unavailable (an older index
    without the vocab table) or when filtering would leave too few — retrieval must
    degrade, never return nothing.
    """
    if len(terms) <= MIN_TERMS:
        return list(terms)

    try:
        total = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        if total == 0:
            return list(terms)

        placeholders = ",".join("?" * len(terms))
        rows = connection.execute(
            f"SELECT term, doc FROM chunks_fts_vocab WHERE term IN ({placeholders})",  # noqa: S608 - placeholders only
            tuple(terms),
        ).fetchall()
    except sqlite3.Error:
        return list(terms)

    frequency = {str(row[0]): int(row[1]) for row in rows}
    limit = total * max_fraction

    # Unknown terms are kept: absent from the vocab table means absent from the corpus,
    # which is maximally selective rather than minimally.
    selective = [term for term in terms if frequency.get(term, 0) <= limit]

    if len(selective) >= MIN_TERMS:
        return selective

    # Everything was common. Keep the rarest few rather than dropping the query.
    ranked = sorted(terms, key=lambda term: frequency.get(term, 0))
    return ranked[:MIN_TERMS]


def _hashes_for_languages(connection: sqlite3.Connection, languages: Sequence[str]) -> set[str]:
    placeholders = ",".join("?" * len(languages))
    rows = connection.execute(
        "SELECT DISTINCT c.embed_text_hash FROM chunks c JOIN files f ON f.id = c.file_id "  # noqa: S608 - placeholders only
        f"WHERE f.language IN ({placeholders})",
        tuple(languages),
    ).fetchall()
    return {str(row[0]) for row in rows}


def _candidate(row: sqlite3.Row, rank: int, retriever: Retriever, score: float) -> Candidate:
    return Candidate(
        chunk_id=int(row["id"]),
        path=str(row["path"]),
        kind=str(row["kind"]),
        symbol_path=row["symbol_path"],
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        text=str(row["text"]),
        language=row["language"],
        score=score,
        rank=rank,
        retriever=retriever,
    )
