"""``/doc architecture | readme | api``: documents whose facts come from code (§12.3).

A generated document fails in a particular way: it reads well and describes something that
is not there. A model asked to "write the architecture doc" will name a `services/` package
because most repositories have one. So the document is built the other way round.

**The structure is code.** Components, file counts, the dependency diagram, entry points,
environment variables, the directory layout and the whole of the API reference are gathered
from the index and the tree and rendered from a template. None of it is the model's to
invent, and none of it can be wrong in the way prose is wrong.

**The model writes two bounded things:** an overview paragraph, and one sentence saying what
each component is for. Both are given the facts they may draw on, both are cleaned so that
model text cannot add a heading or a code fence and so cannot alter the document's shape,
and both are then checked: any `path` the prose mentions that is not in the tree is reported
before the user is asked to approve the file.

**The result is verified structurally** — required headings present, fences balanced, the
Mermaid diagram referring only to components that exist — which is what the roadmap's
"verified by structural checks" asks for and what makes a document safe to write without
reading every sentence.

The dependency picture is derived from *names*: a component uses another when it references
a top-level symbol that only that component defines. It is language-agnostic, which resolved
imports are not, and it skips a name defined in more than one component rather than guess,
the same rule the retrieval layer applies to callee signatures.
"""

from __future__ import annotations

import json
import re
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

from hearth.core.runner import ChatRunner
from hearth.core.session import Session
from hearth.indexing.languages import SupportLevel, declared_support_level
from hearth.prompts import load
from hearth.storage.index_repo import IndexRepository
from hearth.tools.gateway import ToolGateway

DocKind = Literal["architecture", "readme", "api"]

#: Where each document is written. Conventional locations; the write is an approved diff,
#: so an existing file is shown as one rather than silently replaced.
DOC_PATHS: dict[str, str] = {
    "architecture": "docs/ARCHITECTURE.md",
    "readme": "README.md",
    "api": "docs/API.md",
}

#: Directory names that hold no source worth documenting as a component.
_NON_SOURCE_DIRS = frozenset(
    {
        "tests", "test", "__tests__", "docs", "doc", "examples", "example", "scripts",
        "benchmarks", "bench", "fixtures", "vendor", "node_modules", ".venv", "dist", "build",
    }
)  # fmt: skip

#: Symbol kinds that are definitions other code refers to *by name*. Methods are excluded on
#: purpose: `run` and `get` are defined everywhere, and counting them would connect
#: components that share nothing but a verb.
_NAMED_KINDS = frozenset({"function", "class", "interface", "type", "const", "var"})

MAX_ENV_FILES = 2_000
MAX_ENV_FILE_BYTES = 200_000
MAX_ENV_VARS = 40
MAX_TREE_LINES = 60
MAX_EDGES_SHOWN = 40
MAX_API_SYMBOLS = 1_500
MAX_ROLE_CHARS = 220
MAX_OVERVIEW_CHARS = 900
#: Symbols shown to the model per component. Enough to characterise it, few enough that the
#: prompt stays a few hundred tokens and the model has less room to embroider.
MODEL_SYMBOLS_PER_COMPONENT = 12

_ENV_PATTERNS = (
    re.compile(r"""os\.environ\[\s*['"](\w+)['"]\s*\]"""),
    re.compile(r"""os\.environ\.get\(\s*['"](\w+)['"]"""),
    re.compile(r"""os\.getenv\(\s*['"](\w+)['"]"""),
    re.compile(r"""process\.env\.(\w+)"""),
    re.compile(r"""process\.env\[\s*['"](\w+)['"]\s*\]"""),
    re.compile(r"""os\.Getenv\(\s*"(\w+)\""""),
    re.compile(r"""env::var\(\s*"(\w+)\""""),
)
_BACKTICKED = re.compile(r"`([^`\n]+)`")
_PATHLIKE = re.compile(r"^[\w.\-/]+\.\w{1,6}$")


# ------------------------------------------------------------------------- facts


@dataclass(frozen=True)
class Component:
    """A unit of the project's structure: a package directory, or a module in a flat one."""

    name: str
    #: Directory prefix (``src/pkg/core``) for a package, exact file path for a module.
    path: str
    kind: Literal["package", "module"]
    files: tuple[str, ...]
    #: Top-level symbols shown to the model when it is asked what the component is for.
    symbols: tuple[str, ...] = ()

    def contains(self, file_path: str) -> bool:
        if self.kind == "module":
            return file_path == self.path
        return file_path.startswith(self.path + "/")


@dataclass
class ProjectFacts:
    """Everything a document may state, gathered without a model."""

    name: str
    description: str = ""
    components: list[Component] = field(default_factory=list)
    #: (from, to) -> how many distinct names `from` uses that only `to` defines.
    edges: dict[tuple[str, str], int] = field(default_factory=dict)
    entry_points: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    tree: list[str] = field(default_factory=list)
    test_command: str | None = None
    #: Every indexed file path, for checking that a path a document mentions exists.
    files: frozenset[str] = frozenset()
    #: Path -> (symbol rows) for the API reference.
    symbols: list[dict[str, object]] = field(default_factory=list)


@dataclass
class DocOutcome:
    kind: str
    path: str
    text: str = ""
    #: The document is structurally broken and was not written. Should be impossible from a
    #: template; the check exists so that stays true when the template changes.
    problems: list[str] = field(default_factory=list)
    #: Things a reader should double-check before approving: paths in model prose that are
    #: not in the tree.
    warnings: list[str] = field(default_factory=list)
    status: Literal["written", "rejected", "failed", "refused"] = "failed"
    detail: str = ""


def gather_facts(root: Path, repository: IndexRepository) -> ProjectFacts:
    """Collect every fact a document may state. Reads the index and the tree; no model."""
    records = [record for record in repository.all_files() if record.language and not record.is_generated]
    paths = sorted(record.path for record in records)
    # Components come from *code* only. A README or `pyproject.toml` at the repository root
    # is indexed (it has a language) but is not a component, and counting it as source put a
    # file directly in the root, which stopped the descent at the top and reported the whole
    # project as one component named `src`. Declared level rather than effective: a Go file
    # whose grammar is not installed is still source.
    code = sorted(
        record.path
        for record in records
        if declared_support_level(record.language) in (SupportLevel.FULL, SupportLevel.STRUCTURAL)
    )
    symbols = repository.symbols_for_docs()

    manifest = _read_manifest(root)
    components = _with_symbols(find_components(code), symbols)

    return ProjectFacts(
        name=_text(manifest, "name") or root.name,
        description=_text(manifest, "description"),
        components=components,
        edges=component_edges(components, symbols, repository.reference_pairs()),
        entry_points=_entry_points(root, code, manifest),
        env_vars=_environment_variables(root, code),
        tree=_tree(paths),
        test_command=_test_command(root, paths, manifest),
        # Every indexed file, not just code: a document may legitimately mention the
        # README or the manifest, and flagging those as invented would be a false alarm.
        files=frozenset(paths),
        symbols=symbols,
    )


def find_components(paths: list[str]) -> list[Component]:
    """Work out the project's components from its source layout.

    Descend through single-child directories (``src/`` -> ``pkg/``), since those are
    packaging, not architecture. At the first directory that branches, each source-bearing
    subdirectory is a component. In a flat package with no subdirectories the modules
    themselves are the components, which is the honest granularity for a small project.
    """
    source = [path for path in paths if not _is_non_source(path)]
    if not source:
        return []

    root = ""
    while True:
        prefix = f"{root}/" if root else ""
        inside = [path for path in source if path.startswith(prefix)]
        direct = [
            path
            for path in inside
            if "/" not in path[len(prefix) :] and PurePosixPath(path).name != "__init__.py"
        ]
        subdirs = sorted(
            {path[len(prefix) :].split("/", 1)[0] for path in inside if "/" in path[len(prefix) :]}
        )

        if not direct and len(subdirs) == 1:
            root = f"{prefix}{subdirs[0]}"
            continue
        break

    prefix = f"{root}/" if root else ""
    inside = [path for path in source if path.startswith(prefix)]

    if subdirs:
        components = []
        for name in subdirs:
            member = tuple(path for path in inside if path.startswith(f"{prefix}{name}/"))
            components.append(Component(name, f"{prefix}{name}", "package", member))
        return components

    modules = [
        Component(PurePosixPath(path).stem, path, "module", (path,))
        for path in inside
        if PurePosixPath(path).name != "__init__.py"
    ]
    # Sorted here rather than trusting the caller's order: the diagram and the table are
    # rendered from this list, and the same project must always produce the same document.
    return sorted(modules, key=lambda component: component.name)


def component_edges(
    components: list[Component],
    symbols: list[dict[str, object]],
    references: list[tuple[str, str]],
) -> dict[tuple[str, str], int]:
    """Which components use which, by the names they reference.

    A name defined in more than one component is skipped: attributing a use of `parse` to
    whichever component happened to be first would draw a dependency that is not there, and
    a wrong arrow in a diagram is worse than a missing one because nobody double-checks it.
    """
    owner: dict[str, str] = {}
    for component in components:
        for file in component.files:
            owner[file] = component.name

    defined_in: dict[str, set[str]] = {}
    for row in symbols:
        if row["parent"] is not None or row["kind"] not in _NAMED_KINDS:
            continue
        owning = owner.get(str(row["path"]))
        if owning is not None:
            defined_in.setdefault(str(row["name"]), set()).add(owning)

    used: dict[tuple[str, str], set[str]] = {}
    for path, name in references:
        source = owner.get(path)
        candidates = defined_in.get(name)
        if source is None or not candidates or len(candidates) != 1:
            continue
        (target,) = candidates
        if target != source:
            used.setdefault((source, target), set()).add(name)

    return {edge: len(names) for edge, names in used.items()}


# ---------------------------------------------------------------------- rendering


def render_mermaid(components: list[Component], edges: dict[tuple[str, str], int]) -> str:
    """The component diagram, built from the edges. Deterministic: same facts, same text."""
    names = sorted(component.name for component in components)
    ids = {name: f"n{index}" for index, name in enumerate(names)}

    lines = ["graph LR"]
    lines.extend(f'    {ids[name]}["{name}"]' for name in names)

    shown = sorted(edges.items(), key=lambda item: (-item[1], item[0]))[:MAX_EDGES_SHOWN]
    for (source, target), _weight in sorted(shown):
        lines.append(f"    {ids[source]} --> {ids[target]}")
    return "\n".join(lines)


def render_architecture(facts: ProjectFacts, overview: str, roles: dict[str, str]) -> str:
    parts = [f"# {facts.name} — architecture", "", "## Overview", "", overview or _NO_OVERVIEW, ""]

    parts += ["## Components", "", "| Component | Files | Role |", "|---|---|---|"]
    for component in facts.components:
        role = roles.get(component.name) or "—"
        parts.append(f"| `{component.name}` | {len(component.files)} | {role} |")
    parts.append("")

    parts += ["## How the components depend on each other", ""]
    if facts.edges:
        parts += ["```mermaid", render_mermaid(facts.components, facts.edges), "```", ""]
        hidden = len(facts.edges) - MAX_EDGES_SHOWN
        parts += [
            "An arrow means the source component references a top-level name that only the "
            "target defines. Names defined in several components are not counted."
            + (
                f" The {hidden} weakest of {len(facts.edges)} dependencies are not drawn."
                if hidden > 0
                else ""
            ),
            "",
        ]
    else:
        parts += ["No dependencies between components were found in the index.", ""]

    parts += ["## Entry points", ""]
    parts += [f"- `{entry}`" for entry in facts.entry_points] or ["None found."]
    parts.append("")

    parts += ["## Configuration", ""]
    if facts.env_vars:
        parts += ["Environment variables read by the code:", ""]
        parts += [f"- `{name}` (`{path}`)" for name, path in sorted(facts.env_vars.items())]
    else:
        parts.append("No environment variables are read.")
    parts.append("")

    parts += ["## Directory layout", "", "```text", *facts.tree, "```", ""]
    return "\n".join(parts)


def render_readme(facts: ProjectFacts, overview: str) -> str:
    parts = [f"# {facts.name}", ""]
    if facts.description:
        parts += [facts.description, ""]
    parts += ["## Overview", "", overview or _NO_OVERVIEW, ""]

    parts += ["## Usage", ""]
    parts += [f"- `{entry}`" for entry in facts.entry_points] or ["No entry points were found."]
    parts.append("")

    parts += ["## Tests", ""]
    parts.append(
        f"```bash\n{facts.test_command}\n```" if facts.test_command else "No test setup was detected."
    )
    parts.append("")

    parts += ["## Project layout", "", "| Component | Files |", "|---|---|"]
    parts += [f"| `{c.name}` | {len(c.files)} |" for c in facts.components]
    parts.append("")
    return "\n".join(parts)


def render_api(facts: ProjectFacts) -> str:
    """Every public symbol, by component and file, with its signature and location."""
    parts = [f"# {facts.name} — API reference", ""]

    public = [row for row in facts.symbols if _is_public(row) and str(row["path"]) in facts.files]
    truncated = len(public) > MAX_API_SYMBOLS
    public = public[:MAX_API_SYMBOLS]

    owner = {file: component.name for component in facts.components for file in component.files}
    by_component: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in public:
        component = owner.get(str(row["path"]), "(other)")
        by_component.setdefault(component, {}).setdefault(str(row["path"]), []).append(row)

    for component in sorted(by_component):
        parts += [f"## {component}", ""]
        for path in sorted(by_component[component]):
            parts += [f"### `{path}`", ""]
            for row in by_component[component][path]:
                signature = str(row["signature"] or row["name"]).strip()
                indent = "  " if row["parent"] is not None else ""
                parts.append(f"{indent}- `{signature}` — line {row['start_line']}")
            parts.append("")

    if truncated:
        parts += [f"_Showing the first {MAX_API_SYMBOLS} public symbols._", ""]
    if len(parts) <= 2:
        parts += ["No public symbols were found in the index.", ""]
    return "\n".join(parts)


_NO_OVERVIEW = "_No overview was generated._"


# --------------------------------------------------------------------- prose


def clean_prose(text: str, *, max_chars: int) -> str:
    """Make model text safe to place in a document without altering its shape.

    Fenced blocks and heading lines are removed outright. A model that adds `## Security`
    to its overview has added a section the document's structure does not know about, and
    a stray fence would swallow everything after it.
    """
    without_fences = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    lines = [line.rstrip() for line in without_fences.splitlines() if not line.lstrip().startswith("#")]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars]
    boundary = max(cut.rfind(". "), cut.rfind(".\n"))
    return cut[: boundary + 1] if boundary > max_chars // 2 else cut.rstrip() + "…"


def clean_role(text: str) -> str:
    """One sentence from a model reply: the first line, unbulleted, unquoted, and short."""
    for line in text.strip().splitlines():
        stripped = line.strip().lstrip("-*• ").strip().strip("\"'")
        if stripped and not stripped.startswith("#"):
            role = clean_prose(stripped, max_chars=MAX_ROLE_CHARS)
            return role.replace("|", "\\|")
    return ""


def unknown_paths(prose: str, known_files: frozenset[str]) -> list[str]:
    """Backticked paths in prose that are not files in the tree.

    Only path-shaped tokens are checked (`a/b.py`, `x.toml`); a backticked `compute_tax` is
    a symbol, which this check cannot judge and does not pretend to. A path is accepted
    when it names an indexed file exactly or as a trailing path (models drop leading
    directories), so this errs towards silence: a false alarm on every document teaches
    people to stop reading the warnings.
    """
    unknown: list[str] = []
    for token in _BACKTICKED.findall(prose):
        candidate = token.strip()
        if not _PATHLIKE.match(candidate) or " " in candidate:
            continue
        if candidate in known_files or any(path.endswith("/" + candidate) for path in known_files):
            continue
        if candidate not in unknown:
            unknown.append(candidate)
    return unknown


# ------------------------------------------------------------------ verification

_REQUIRED_HEADINGS: dict[str, tuple[str, ...]] = {
    "architecture": (
        "## Overview",
        "## Components",
        "## How the components depend on each other",
        "## Entry points",
        "## Configuration",
        "## Directory layout",
    ),
    "readme": ("## Overview", "## Usage", "## Tests", "## Project layout"),
    "api": (),
}


def verify_document(kind: str, text: str, facts: ProjectFacts) -> list[str]:
    """Structural checks. Returns problems; empty means the document is well-formed.

    These are properties of the *shape* — headings present, fences balanced, the diagram
    referring only to components that exist, no template placeholder left behind. They can
    only fail if the template or the prose cleaning has a bug, which is exactly why they
    are checked: a document is written to disk on the strength of them.
    """
    problems: list[str] = []

    for heading in _REQUIRED_HEADINGS.get(kind, ()):
        if not re.search(rf"^{re.escape(heading)}\s*$", text, flags=re.MULTILINE):
            problems.append(f"missing the {heading!r} section")

    if text.count("```") % 2:
        problems.append("a code fence is not closed")
    if "{{" in text or "}}" in text:
        problems.append("a template placeholder was left in the document")

    if kind == "architecture":
        problems += _verify_mermaid(text, facts)
        for component in facts.components:
            if f"| `{component.name}` |" not in text:
                problems.append(f"component {component.name!r} is missing from the table")

    if kind == "api":
        problems += _verify_api_symbols(text, facts)

    return problems


def _verify_mermaid(text: str, facts: ProjectFacts) -> list[str]:
    match = re.search(r"```mermaid\n(.*?)\n```", text, flags=re.DOTALL)
    if match is None:
        return [] if not facts.edges else ["the dependency diagram is missing"]

    block = match.group(1).splitlines()
    problems: list[str] = []
    if not block or not block[0].startswith("graph "):
        problems.append("the diagram does not start with a graph declaration")

    known = {component.name for component in facts.components}
    labels: dict[str, str] = {}
    for line in block[1:]:
        node = re.match(r'^\s*(\w+)\["([^"]+)"\]\s*$', line)
        edge = re.match(r"^\s*(\w+)\s*-->\s*(\w+)\s*$", line)
        if node:
            labels[node.group(1)] = node.group(2)
            if node.group(2) not in known:
                problems.append(f"the diagram names {node.group(2)!r}, which is not a component")
        elif edge:
            for end in edge.groups():
                if end not in labels:
                    problems.append(f"the diagram draws an edge to undefined node {end!r}")
        else:
            problems.append(f"the diagram has a line it cannot read: {line.strip()!r}")
    return problems


def _verify_api_symbols(text: str, facts: ProjectFacts) -> list[str]:
    """Every `path` heading in an API document names an indexed file."""
    problems = []
    for path in re.findall(r"^### `([^`]+)`", text, flags=re.MULTILINE):
        if path not in facts.files:
            problems.append(f"the reference lists {path!r}, which is not an indexed file")
    return problems


# --------------------------------------------------------------------- workflow


async def run_doc(
    *,
    kind: DocKind,
    runner: ChatRunner,
    session: Session,
    gateway: ToolGateway,
    root: Path,
    repository: IndexRepository,
) -> DocOutcome:
    """Gather facts, have the model write the bounded prose, verify, and write the file."""
    path = DOC_PATHS[kind]
    facts = gather_facts(root, repository)

    if not facts.components and kind != "readme":
        return DocOutcome(
            kind,
            path,
            status="refused",
            detail="the index holds no source files to document. Run `hearth index`.",
        )

    overview = ""
    roles: dict[str, str] = {}
    if kind in ("architecture", "readme"):
        overview = clean_prose(
            await runner.complete(session, _overview_prompt(kind, facts), num_predict=350),
            max_chars=MAX_OVERVIEW_CHARS,
        )
    if kind == "architecture":
        for component in facts.components:
            # No symbols means the parser extracted nothing from it (a file with a syntax
            # error, say). The model is not asked: given nothing to go on it answers "not
            # defined by any listed files", which is a sentence about its own prompt rather
            # than about the code, and it would sit in the table as though it were a role.
            # The row renders a dash instead.
            if not component.symbols:
                continue
            reply = await runner.complete(session, _role_prompt(facts, component), num_predict=80)
            roles[component.name] = clean_role(reply)

    text = {
        "architecture": lambda: render_architecture(facts, overview, roles),
        "readme": lambda: render_readme(facts, overview),
        "api": lambda: render_api(facts),
    }[kind]()

    outcome = DocOutcome(kind, path, text=text)
    outcome.problems = verify_document(kind, text, facts)
    if outcome.problems:
        outcome.status = "failed"
        outcome.detail = "the generated document failed its structural checks, so it was not written."
        return outcome

    model_prose = "\n".join([overview, *roles.values()])
    outcome.warnings = [
        f"the text mentions `{path_}`, which is not a file in this project"
        for path_ in unknown_paths(model_prose, facts.files)
    ]

    return await _write(gateway, root, outcome)


async def _write(gateway: ToolGateway, root: Path, outcome: DocOutcome) -> DocOutcome:
    """Write through the gateway, so the user sees the diff and approves it.

    An existing file is read first: `write_file` refuses to overwrite a file the session has
    not read, and that rule is not one a workflow gets to skip.
    """
    if (root / outcome.path).is_file():
        await gateway.call("read_file", {"path": outcome.path}, call_id=f"wf-doc-{uuid.uuid4().hex[:8]}")

    result = await gateway.call(
        "write_file",
        {"path": outcome.path, "content": outcome.text},
        call_id=f"wf-doc-{uuid.uuid4().hex[:8]}",
    )
    if result.ok:
        outcome.status = "written"
        outcome.detail = result.content
    elif result.error is not None and result.error.value in ("rejected", "denied"):
        outcome.status = "rejected"
        outcome.detail = result.content
    else:
        outcome.status = "failed"
        outcome.detail = result.content
    return outcome


def _overview_prompt(kind: str, facts: ProjectFacts) -> str:
    components = (
        "\n".join(f"- {component.name} ({len(component.files)} file(s))" for component in facts.components)
        or "(none found)"
    )
    return (
        load("workflows/doc_readme" if kind == "readme" else "workflows/doc_architecture")
        .replace("{{name}}", facts.name)
        .replace("{{description}}", facts.description or "(none given)")
        .replace("{{components}}", components)
        .replace(
            "{{entry_points}}", "\n".join(f"- {entry}" for entry in facts.entry_points) or "(none found)"
        )
    )


def _role_prompt(facts: ProjectFacts, component: Component) -> str:
    shown = "\n".join(f"- {symbol}" for symbol in component.symbols) or "(no top-level symbols)"
    files = "\n".join(f"- {file}" for file in component.files[:12])
    return (
        load("workflows/doc_component")
        .replace("{{project}}", facts.name)
        .replace("{{component}}", component.name)
        .replace("{{files}}", files)
        .replace("{{symbols}}", shown)
    )


# ------------------------------------------------------------------ fact helpers


def _is_non_source(path: str) -> bool:
    return any(part in _NON_SOURCE_DIRS for part in PurePosixPath(path).parts[:-1])


def _is_public(row: dict[str, object]) -> bool:
    if str(row["name"]).startswith("_"):
        return False
    if row["parent"] is not None and str(row["parent"]).startswith("_"):
        return False
    exported = row["exported"]
    return exported is None or bool(exported)


def _with_symbols(components: list[Component], symbols: list[dict[str, object]]) -> list[Component]:
    """Attach each component's top-level symbol signatures, for the model's role prompt."""
    result = []
    for component in components:
        signatures = [
            str(row["signature"] or row["name"]).strip()
            for row in symbols
            if row["parent"] is None
            and row["kind"] in _NAMED_KINDS
            and component.contains(str(row["path"]))
            and not str(row["name"]).startswith("_")
        ][:MODEL_SYMBOLS_PER_COMPONENT]
        result.append(
            Component(component.name, component.path, component.kind, component.files, tuple(signatures))
        )
    return result


def _text(manifest: dict[str, object], key: str) -> str:
    """A string from the manifest, or empty. Manifests are untrusted shapes: a `name` that
    is a number or a list in somebody's package.json must not reach a document as one."""
    value = manifest.get(key)
    return value.strip() if isinstance(value, str) else ""


def _read_manifest(root: Path) -> dict[str, object]:
    """Name, description and scripts from whichever manifest the project has."""
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            project = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {})
        except (OSError, tomllib.TOMLDecodeError):
            project = {}
        if project:
            return {
                "name": project.get("name", ""),
                "description": project.get("description", ""),
                "scripts": dict(project.get("scripts", {})),
                "kind": "python",
            }

    package = root / "package.json"
    if package.is_file():
        try:
            data = json.loads(package.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if isinstance(data, dict):
            bin_field = data.get("bin")
            scripts = (
                {data["name"]: bin_field}
                if isinstance(bin_field, str) and data.get("name")
                else dict(bin_field or {})
            )
            return {
                "name": data.get("name", ""),
                "description": data.get("description", ""),
                "scripts": scripts,
                "npm_scripts": dict(data.get("scripts", {})),
                "main": data.get("main", ""),
                "kind": "node",
            }

    cargo = root / "Cargo.toml"
    if cargo.is_file():
        try:
            package_table = tomllib.loads(cargo.read_text(encoding="utf-8")).get("package", {})
        except (OSError, tomllib.TOMLDecodeError):
            package_table = {}
        return {
            "name": package_table.get("name", ""),
            "description": package_table.get("description", ""),
            "kind": "rust",
        }

    gomod = root / "go.mod"
    if gomod.is_file():
        match = re.search(
            r"^module\s+(\S+)", gomod.read_text(encoding="utf-8", errors="replace"), re.MULTILINE
        )
        return {"name": match.group(1).rsplit("/", 1)[-1] if match else "", "kind": "go"}

    return {}


def _entry_points(root: Path, paths: list[str], manifest: dict[str, object]) -> list[str]:
    found: list[str] = []
    scripts = manifest.get("scripts")
    if isinstance(scripts, dict):
        found += [f"{name} → {target}" for name, target in sorted(scripts.items())]
    found += [f"python -m {_module_name(path)}" for path in paths if path.endswith("__main__.py")]
    if manifest.get("main"):
        found.append(f"main: {manifest['main']}")
    found += [
        f"go run ./{PurePosixPath(path).parent}" for path in paths if path.endswith("main.go") and "/" in path
    ]
    return list(dict.fromkeys(found))


def _module_name(path: str) -> str:
    parts = list(PurePosixPath(path).parts[:-1])
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)


def _test_command(root: Path, paths: list[str], manifest: dict[str, object]) -> str | None:
    npm = manifest.get("npm_scripts")
    if isinstance(npm, dict) and "test" in npm:
        return "npm test"
    if manifest.get("kind") == "rust":
        return "cargo test"
    if manifest.get("kind") == "go":
        return "go test ./..."
    pyproject = (
        (root / "pyproject.toml").read_text(encoding="utf-8", errors="replace")
        if (root / "pyproject.toml").is_file()
        else ""
    )
    if "pytest" in pyproject or (root / "pytest.ini").is_file() or (root / "conftest.py").is_file():
        return "pytest"
    if any(PurePosixPath(path).name.startswith("test_") for path in paths):
        return "pytest"
    return None


def _environment_variables(root: Path, paths: list[str]) -> dict[str, str]:
    """Environment variables the source reads, with the first file that reads each."""
    found: dict[str, str] = {}
    for path in paths[:MAX_ENV_FILES]:
        if len(found) >= MAX_ENV_VARS:
            break
        if _is_non_source(path):
            continue
        try:
            file = root / path
            if file.stat().st_size > MAX_ENV_FILE_BYTES:
                continue
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pattern in _ENV_PATTERNS:
            for name in pattern.findall(text):
                found.setdefault(name, path)
    return dict(sorted(found.items())[:MAX_ENV_VARS])


def _tree(paths: list[str]) -> list[str]:
    """A directory outline three levels deep, with file counts."""
    counts: dict[str, int] = {}
    for path in paths:
        parts = PurePosixPath(path).parts[:-1]
        for depth in range(1, min(len(parts), 3) + 1):
            key = "/".join(parts[:depth])
            counts[key] = counts.get(key, 0) + 1

    # Sorted by path *parts*, not by the joined string: "a-b" sorts before "a/x" as text,
    # which would put a sibling directory between a directory and its own children.
    lines = [
        f"{'  ' * key.count('/')}{PurePosixPath(key).name}/  ({counts[key]} file(s))"
        for key in sorted(counts, key=lambda key: key.split("/"))
    ]
    if len(lines) > MAX_TREE_LINES:
        return [*lines[:MAX_TREE_LINES], f"… {len(lines) - MAX_TREE_LINES} more"]
    return lines
