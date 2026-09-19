"""The repo map: a ranked, budgeted skeleton of the whole repository.

An adaptation of the approach Aider popularised (docs/system-design.md §7.7). Retrieval
answers "where is X"; the repo map answers "what is this repository, and how do its parts
connect" — the question a model otherwise has to guess at, because no single retrieved
chunk contains the shape of a codebase.

The pipeline is four steps, and each exists for a reason worth keeping:

1. **Graph.** Nodes are files; an edge A→B carries the identifiers A references that B
   defines. This is a *use* graph rather than an import graph, because imports say what a
   file is allowed to touch while references say what it actually leans on.

2. **Weight.** An identifier's weight rises with how specific it looks and falls with how
   many files define it — an IDF-like measure. Without it, `get`, `run` and `name` would
   dominate the graph by sheer frequency and connect everything to everything, which is
   the same as connecting nothing.

3. **Rank.** Personalised PageRank. Importance is *recursive*: a file matters because
   files that matter use it. The personalisation vector is where the session leans in —
   pinned, mentioned, edited and top-retrieved files pull rank toward what the user is
   actually working on, so the same repository maps differently in two conversations.

4. **Render.** Binary search for the largest number of definitions that fits the token
   budget. A map that overruns its budget does not degrade gracefully: it evicts the
   conversation, which is the one thing the map exists to inform.

**Computed once per cache epoch** (§9.2). The map sits near the front of the prompt, so
recomputing it mid-session would change the cached prefix and throw away the KV cache on
every turn — a 28x prefill penalty on the reference machine to refresh a summary nobody
asked for. It is rebuilt at compaction, on `/map refresh`, or when the epoch advances.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

#: Estimates how many tokens a rendered fragment costs.
Measure = Callable[[str], int]

#: Identifiers too generic to carry structural signal. They appear everywhere, so an edge
#: built on them says only "both files are Python", which the graph already knows.
_STOP_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "self", "cls", "init", "main", "run", "get", "set", "add", "new", "name", "value",
        "data", "item", "items", "key", "keys", "result", "results", "args", "kwargs",
        "type", "types", "str", "int", "float", "bool", "list", "dict", "tuple",
        "object", "none", "true", "false", "error", "errors", "test", "tests", "path",
    }
)

#: Below this length an identifier is treated as generic regardless of how rare it is:
#: a short name that happens to be defined once is far more often a loop variable than a
#: meaningful symbol.
_MIN_IDENTIFIER_LENGTH = 4

#: Where the specificity curve flattens. Past roughly this length a name is already as
#: distinctive as it is going to get, and letting weight grow without bound would let one
#: very long identifier outvote a dozen ordinary ones.
_SPECIFICITY_SATURATION = 16.0

#: A definition in this many files or more is treated as ubiquitous and dropped. Keeps
#: `Config`, `Error` and their kin from wiring the whole graph together.
_UBIQUITY_RATIO = 0.25

#: How much of the budget a single file's entries may claim, so one enormous module
#: cannot crowd every other file out of the map.
_MAX_FILE_SHARE = 0.35

#: PageRank's damping factor. The networkx default, kept explicit because the map's shape
#: depends on it and a silent library change should not quietly re-rank a repository.
_DAMPING = 0.85

#: Characters per token. Only used when no estimator is supplied.
_CHARS_PER_TOKEN = 4.0

#: Longest signature the map will print before dropping its parameter list.
_MAX_SIGNATURE_CHARS = 110

#: Rank multiplier for test and fixture files. They reference production symbols
#: constantly, which is what PageRank rewards, so without a discount they crowd out the
#: code under test.
_TEST_RANK_PENALTY = 0.15


@dataclass(frozen=True)
class MapEntry:
    """One definition the map may show."""

    path: str
    name: str
    kind: str
    signature: str
    line: int
    score: float
    #: Enclosing class or interface, when the symbol is nested inside one.
    parent: str | None = None


@dataclass
class RepoMap:
    """A rendered map, with what it cost and what it left out."""

    text: str
    entries: tuple[MapEntry, ...] = ()
    #: Definitions that ranked but did not fit.
    omitted: int = 0
    estimated_tokens: int = 0
    #: Files the graph knew about, before budgeting.
    ranked_files: int = 0

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    @property
    def files_shown(self) -> int:
        return len({entry.path for entry in self.entries})


@dataclass
class _Definition:
    """A symbol as the ranker sees it."""

    path: str
    name: str
    kind: str
    signature: str
    line: int
    parent: str | None


@dataclass
class _Graph:
    """The use graph, plus what it took to build."""

    edges: dict[tuple[str, str], float] = field(default_factory=dict)
    #: (defining file, identifier) -> weight arriving from everywhere that uses it.
    incoming: dict[tuple[str, str], float] = field(default_factory=dict)
    files: set[str] = field(default_factory=set)


class RepoMapBuilder:
    """Builds repo maps for one index.

    Holds no session state: the caller supplies the personalisation for the turn, which is
    what keeps a builder reusable across sessions that care about different files.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def build(
        self,
        *,
        budget_tokens: int,
        personalization: Mapping[str, float] | None = None,
        estimator: object | None = None,
    ) -> RepoMap:
        """Rank every definition and render as many as the budget allows.

        Args:
            budget_tokens: Hard ceiling. The result never exceeds it.
            personalization: Workspace-relative path -> weight, for files the session is
                working on. Unknown paths are ignored rather than rejected, since a path
                may have been deleted since it was mentioned.
            estimator: Anything with ``.estimate(text) -> int``. Defaults to a
                characters-per-token approximation.
        """
        if budget_tokens <= 0:
            return RepoMap(text="")

        definitions = self._load_definitions()
        if not definitions:
            return RepoMap(text="")

        graph = self._build_graph(definitions)
        ranks = self._rank(graph, personalization or {})
        ranked = self._score(definitions, graph, ranks)

        return self._render_within_budget(
            ranked, budget_tokens=budget_tokens, estimator=estimator, ranked_files=len(graph.files)
        )

    # ------------------------------------------------------------------ loading

    def _load_definitions(self) -> list[_Definition]:
        """Every named definition, with its enclosing symbol when it has one.

        Only symbols that carry a signature are useful here: the map's whole output is
        signatures, and a definition without one renders as a bare name that tells the
        model nothing it could not have guessed from the filename.
        """
        rows = self._connection.execute(
            """
            SELECT f.path AS path, s.name AS name, s.kind AS kind,
                   s.signature AS signature, s.start_line AS line,
                   parent.name AS parent_name, parent.signature AS parent_signature,
                   parent.kind AS parent_kind
            FROM symbols s
            JOIN files f ON f.id = s.file_id
            LEFT JOIN symbols parent ON parent.id = s.parent_id
            WHERE s.name IS NOT NULL AND s.name != ''
            """
        ).fetchall()

        return [
            _Definition(
                path=str(row["path"]),
                name=str(row["name"]),
                kind=str(row["kind"]),
                signature=_elide(str(row["signature"] or row["name"])),
                line=int(row["line"]),
                parent=_parent_header(row),
            )
            for row in rows
        ]

    def _load_references(self) -> dict[str, dict[str, int]]:
        """Referencing file -> identifier -> how many times it appears there."""
        rows = self._connection.execute(
            """
            SELECT f.path AS path, r.name AS name, COUNT(*) AS hits
            FROM refs r
            JOIN files f ON f.id = r.file_id
            GROUP BY f.path, r.name
            """
        ).fetchall()

        references: dict[str, dict[str, int]] = defaultdict(dict)
        for row in rows:
            references[str(row["path"])][str(row["name"])] = int(row["hits"])
        return references

    # -------------------------------------------------------------------- graph

    def _build_graph(self, definitions: list[_Definition]) -> _Graph:
        """Edges from each file to the files defining what it references."""
        definers: dict[str, set[str]] = defaultdict(set)
        for definition in definitions:
            definers[definition.name].add(definition.path)

        references = self._load_references()
        total_files = len({d.path for d in definitions} | set(references))
        graph = _Graph(files={d.path for d in definitions})

        ubiquitous = max(2, int(total_files * _UBIQUITY_RATIO))

        for source, names in references.items():
            graph.files.add(source)
            for name, hits in names.items():
                targets = definers.get(name)
                if not targets or len(targets) >= ubiquitous:
                    continue

                weight = _identifier_weight(name, defined_in=len(targets), total_files=total_files)
                if weight <= 0.0:
                    continue

                # sqrt, not the raw count: a file calling something forty times depends on
                # it more than one calling it twice, but not twenty times more, and the
                # linear form lets a single hot loop dominate the whole graph.
                contribution = weight * math.sqrt(hits)

                for target in targets:
                    if target == source:
                        # Self-edges say a file uses itself, which is true of every file
                        # and would inflate exactly the large modules already over-ranked.
                        continue
                    graph.edges[(source, target)] = graph.edges.get((source, target), 0.0) + contribution
                    key = (target, name)
                    graph.incoming[key] = graph.incoming.get(key, 0.0) + contribution

        return graph

    def _rank(self, graph: _Graph, personalization: Mapping[str, float]) -> dict[str, float]:
        """Personalised PageRank over the use graph.

        networkx is imported here rather than at module scope: it pulls in a large
        dependency tree, and `hearth --version` should not pay for it (CLAUDE.md style).
        """
        if not graph.files:
            return {}

        import networkx as nx  # type: ignore[import-untyped]

        digraph = nx.DiGraph()
        digraph.add_nodes_from(graph.files)
        for (source, target), weight in graph.edges.items():
            digraph.add_edge(source, target, weight=weight)

        vector = {path: max(0.0, float(personalization.get(path, 0.0))) for path in graph.files}
        if sum(vector.values()) <= 0.0:
            vector = None  # type: ignore[assignment]
        else:
            # A floor under every node: a personalisation vector with zeros everywhere else
            # makes PageRank ignore whole components of the graph, so a session that pinned
            # one file would get a map of that file's neighbourhood and nothing else.
            floor = 1.0 / len(graph.files)
            vector = {path: weight + floor for path, weight in vector.items()}

        try:
            ranks = nx.pagerank(digraph, alpha=_DAMPING, weight="weight", personalization=vector)
        except Exception:
            # Convergence failures are real on pathological graphs, and a repo map is a
            # nicety: falling back to uniform rank degrades the ordering rather than the
            # session (the budget and rendering still work).
            uniform = 1.0 / len(graph.files)
            ranks = dict.fromkeys(graph.files, uniform)

        return {str(path): float(rank) for path, rank in ranks.items()}

    def _score(
        self, definitions: list[_Definition], graph: _Graph, ranks: dict[str, float]
    ) -> list[MapEntry]:
        """Definition score: its file's rank times how much the graph leans on that name.

        The product is what makes the map structural rather than a popularity list. A
        widely used helper in an unimportant file and an unused symbol in a central one
        both score low; what survives is a name the repository depends on, in a file the
        repository depends on.
        """
        entries: list[MapEntry] = []
        for definition in definitions:
            rank = ranks.get(definition.path, 0.0) * _test_penalty(definition.path)
            incoming = graph.incoming.get((definition.path, definition.name), 0.0)

            # The +1 keeps un-referenced definitions rankable. A public entry point that
            # nothing inside the repository calls is often exactly what a newcomer needs
            # to see, and dropping it would make the map describe internals only.
            score = rank * (1.0 + incoming)

            entries.append(
                MapEntry(
                    path=definition.path,
                    name=definition.name,
                    kind=definition.kind,
                    signature=definition.signature,
                    line=definition.line,
                    score=score,
                    parent=definition.parent,
                )
            )

        entries.sort(key=lambda entry: (-entry.score, entry.path, entry.line))
        return entries

    # ------------------------------------------------------------------ render

    def _render_within_budget(
        self,
        ranked: list[MapEntry],
        *,
        budget_tokens: int,
        estimator: object | None,
        ranked_files: int,
    ) -> RepoMap:
        """Binary search for the largest prefix of ``ranked`` that fits.

        Searching over the *count* rather than trimming the rendered text keeps whole
        definitions intact: a map cut mid-signature is worse than a shorter map, because
        the model cannot tell a truncated signature from a real one.
        """
        measure = _measurer(estimator)

        capped = _cap_per_file(ranked, budget_tokens=budget_tokens, measure=measure)
        if not capped:
            return RepoMap(text="", ranked_files=ranked_files, omitted=len(ranked))

        low, high = 0, len(capped)
        best_text, best_count, best_tokens = "", 0, 0

        while low <= high:
            middle = (low + high) // 2
            text = render_entries(capped[:middle])
            tokens = measure(text)
            if tokens <= budget_tokens:
                best_text, best_count, best_tokens = text, middle, tokens
                low = middle + 1
            else:
                high = middle - 1

        return RepoMap(
            text=best_text,
            entries=tuple(capped[:best_count]),
            omitted=len(ranked) - best_count,
            estimated_tokens=best_tokens,
            ranked_files=ranked_files,
        )


def render_entries(entries: list[MapEntry]) -> str:
    """Render definitions grouped by file, in the §7.7 format.

    Files appear in the order their best entry ranked, and definitions within a file in
    line order — ranking inside a file would scramble a class's methods, which is how the
    file reads.
    """
    if not entries:
        return ""

    order: list[str] = []
    grouped: dict[str, list[MapEntry]] = defaultdict(list)
    for entry in entries:
        if entry.path not in grouped:
            order.append(entry.path)
        grouped[entry.path].append(entry)

    blocks: list[str] = []
    for path in order:
        rows = sorted(grouped[path], key=lambda entry: entry.line)
        lines = [f"{path}:"]
        shown_parents: set[str] = set()

        for entry in rows:
            if entry.parent and entry.parent not in shown_parents:
                # The enclosing class is printed even when it did not rank on its own:
                # a bare `def finalize(...)` with no owner is ambiguous in any file that
                # has more than one class.
                lines.append(f"│{entry.parent}")
                shown_parents.add(entry.parent)
            indent = "    " if entry.parent else ""
            lines.append(f"│{indent}{entry.signature.strip()}")
            if entry.parent is None and entry.kind in {"class", "interface", "struct"}:
                shown_parents.add(entry.signature.strip())

        blocks.append("\n".join(lines))

    return "\n".join(blocks)


def _cap_per_file(
    ranked: list[MapEntry], *, budget_tokens: int, measure: Measure
) -> list[MapEntry]:
    """Drop entries once one file has claimed too much of the budget.

    Applied before the binary search, so the search chooses among entries that are already
    spread across the repository. Without it a 3000-line module legitimately outranks
    everything and the map becomes a table of contents for one file.
    """
    per_file_ceiling = max(1, int(budget_tokens * _MAX_FILE_SHARE))
    spent: dict[str, int] = defaultdict(int)
    kept: list[MapEntry] = []

    for entry in ranked:
        cost = measure(f"│{entry.signature.strip()}\n")
        # The first entry from a file is always kept: otherwise a file whose single
        # definition is enormous would vanish from the map entirely rather than appear once.
        if spent[entry.path] > 0 and spent[entry.path] + cost > per_file_ceiling:
            continue
        spent[entry.path] += cost
        kept.append(entry)

    return kept


def _parent_header(row: sqlite3.Row) -> str | None:
    """How an enclosing symbol is printed above its members.

    The parser stores a class's signature with its bases (``class Invoice(BaseModel):``),
    which is more informative than the bare name — it is often the only place the map shows
    an inheritance relationship. Falls back to reconstructing a header from the kind when
    no signature was captured.
    """
    name = row["parent_name"]
    if not name:
        return None

    signature = row["parent_signature"]
    if signature:
        return _elide(str(signature))

    kind = str(row["parent_kind"] or "class")
    keyword = kind if kind in {"class", "interface", "struct", "trait", "enum"} else "class"
    return f"{keyword} {name}:"


def _elide(signature: str) -> str:
    """Collapse a signature to one readable line.

    Two problems to fix. Multi-line definitions arrive with their newlines and indentation
    intact, which renders as a ragged block inside the map's ``│`` gutter. And a long
    parameter list can run to several hundred characters — one definition eating a tenth of
    the budget to say what three would have said. Parameters are dropped rather than cut
    mid-word, so what remains is still a valid-looking signature.
    """
    flattened = " ".join(signature.split())
    if len(flattened) <= _MAX_SIGNATURE_CHARS:
        return flattened

    opening = flattened.find("(")
    if opening == -1:
        return flattened[: _MAX_SIGNATURE_CHARS - 1].rstrip() + "…"

    tail = flattened[flattened.rfind(")") + 1 :] if ")" in flattened else ""
    return f"{flattened[:opening]}(…){tail}".strip()


def _test_penalty(path: str) -> float:
    """How much to discount a path that looks like tests or fixtures.

    Not an exclusion: in a test-heavy library the tests are part of the architecture, and a
    map that pretended otherwise would be lying about the repository. But an architecture
    question is almost always about the source, and test files reference production symbols
    constantly — which is exactly what PageRank rewards, so without this they crowd out the
    code they are testing.
    """
    lowered = path.lower()
    parts = lowered.replace("\\", "/").split("/")
    if any(part in {"tests", "test", "testing", "fixtures", "__tests__", "spec"} for part in parts):
        return _TEST_RANK_PENALTY
    filename = parts[-1] if parts else lowered
    if filename.startswith("test_") or filename.endswith(("_test.py", ".test.ts", ".spec.ts")):
        return _TEST_RANK_PENALTY
    return 1.0


def _identifier_weight(name: str, *, defined_in: int, total_files: int) -> float:
    """How much structural signal one identifier carries.

    Two forces, per §7.7: specificity (long, distinctive names mean something) and rarity
    (a name defined in many files distinguishes nothing). Their product is the weight.
    """
    lowered = name.lower().strip("_")
    if len(lowered) < _MIN_IDENTIFIER_LENGTH or lowered in _STOP_IDENTIFIERS:
        return 0.0

    specificity = min(len(lowered), _SPECIFICITY_SATURATION) / _SPECIFICITY_SATURATION
    # Classic IDF, with the +1 keeping a name defined in every file at a small positive
    # weight rather than exactly zero — it still carries a little signal.
    rarity = math.log(1.0 + (total_files / max(1, defined_in)))
    return specificity * rarity


def _measurer(estimator: object | None) -> Measure:
    """A ``text -> tokens`` function, from an estimator or the fallback ratio."""
    estimate = getattr(estimator, "estimate", None)
    if callable(estimate):
        return lambda text: int(estimate(text))
    return lambda text: math.ceil(len(text) / _CHARS_PER_TOKEN)
