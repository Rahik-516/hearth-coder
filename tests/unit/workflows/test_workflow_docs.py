"""`/doc` — I3.

The design claim is that a generated document's *structure* is code and only two bounded
pieces of prose are the model's. So most of this file tests the code half, on synthetic
trees where the right answer is known, and the model half is tested by what it is **not**
allowed to do: add a heading, open a code fence, name a file that is not there.

The workflow tests use a scripted model, a real index of a `py_small` copy, and the real
gateway, so "the document was written" means a diff went through approval and bytes landed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hearth.core.bus import EventBus
from hearth.core.runner import ChatRunner
from hearth.core.session import Mode, Session
from hearth.indexing.pipeline import Indexer
from hearth.llm.scripted_provider import ScriptedProvider, ScriptedResponse
from hearth.safety.checkpoints import CheckpointStore
from hearth.safety.policy import ConfigView, SessionView
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository
from hearth.tools.base import ToolContext
from hearth.tools.channel import ApprovalAsk, ApprovalReply
from hearth.tools.gateway import ToolGateway, make_policy
from hearth.tools.read_fs import ReadFileTool
from hearth.tools.registry import ToolRegistry
from hearth.tools.write_fs import WriteFileTool
from hearth.workflows.docs import (
    Component,
    ProjectFacts,
    clean_prose,
    clean_role,
    component_edges,
    find_components,
    gather_facts,
    render_architecture,
    render_mermaid,
    run_doc,
    unknown_paths,
    verify_document,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "repos"


def sym(path: str, name: str, kind: str = "function", parent: str | None = None) -> dict[str, object]:
    return {
        "path": path,
        "name": name,
        "kind": kind,
        "signature": f"def {name}()",
        "start_line": 1,
        "end_line": 2,
        "exported": True,
        "parent": parent,
    }


# ------------------------------------------------------------------- components


def test_single_child_directories_are_packaging_not_architecture() -> None:
    paths = ["src/pkg/core/a.py", "src/pkg/core/b.py", "src/pkg/cli/c.py", "src/pkg/__init__.py"]

    components = find_components(paths)

    assert [c.name for c in components] == ["cli", "core"]
    assert components[1].files == ("src/pkg/core/a.py", "src/pkg/core/b.py")
    assert all(c.kind == "package" for c in components)


def test_a_flat_package_has_its_modules_as_components() -> None:
    """The honest granularity for a small project."""
    components = find_components(["src/pkg/__init__.py", "src/pkg/models.py", "src/pkg/errors.py"])

    assert [c.name for c in components] == ["errors", "models"]
    assert all(c.kind == "module" for c in components)


def test_tests_and_docs_are_not_components() -> None:
    paths = ["src/pkg/core/a.py", "tests/unit/test_a.py", "docs/conf.py", "examples/demo.py"]

    assert [c.name for c in find_components(paths)] == ["a"]


def test_source_files_beside_subdirectories_stop_the_descent() -> None:
    """`src/pkg/main.py` next to `src/pkg/util/` means `pkg` itself is the level that
    branches; descending further would hide the top of the structure."""
    paths = ["src/pkg/main.py", "src/pkg/util/a.py", "src/pkg/io/b.py"]

    assert [c.name for c in find_components(paths)] == ["io", "util"]


def test_an_empty_project_has_no_components() -> None:
    assert find_components([]) == []
    assert find_components(["tests/test_a.py"]) == []


# ----------------------------------------------------------------------- edges


def components_for(*names: str) -> list[Component]:
    return [Component(n, f"src/{n}", "package", (f"src/{n}/m.py",)) for n in names]


def test_a_component_using_anothers_name_draws_an_edge() -> None:
    symbols = [sym("src/core/m.py", "Session", "class")]

    edges = component_edges(components_for("core", "cli"), symbols, [("src/cli/m.py", "Session")])

    assert edges == {("cli", "core"): 1}


def test_a_name_defined_in_two_components_draws_nothing() -> None:
    """A wrong arrow in a diagram is worse than a missing one: nobody double-checks it."""
    symbols = [sym("src/core/m.py", "parse"), sym("src/io/m.py", "parse")]

    edges = component_edges(components_for("core", "io", "cli"), symbols, [("src/cli/m.py", "parse")])

    assert edges == {}


def test_methods_do_not_connect_components() -> None:
    """`run` and `get` are defined everywhere; counting them would join components that
    share nothing but a verb."""
    symbols = [sym("src/core/m.py", "run", "method", parent="Engine")]

    assert component_edges(components_for("core", "cli"), symbols, [("src/cli/m.py", "run")]) == {}


def test_using_your_own_name_is_not_a_dependency() -> None:
    symbols = [sym("src/core/m.py", "Session", "class")]

    assert component_edges(components_for("core"), symbols, [("src/core/m.py", "Session")]) == {}


def test_the_edge_weight_counts_distinct_names() -> None:
    symbols = [sym("src/core/m.py", "A"), sym("src/core/m.py", "B")]
    refs = [("src/cli/m.py", "A"), ("src/cli/m.py", "B"), ("src/cli/x.py", "A")]

    edges = component_edges(components_for("core", "cli"), symbols, refs)

    assert edges == {("cli", "core"): 2}


# --------------------------------------------------------------------- rendering


def facts_with_edges() -> ProjectFacts:
    components = components_for("cli", "core")
    return ProjectFacts(
        name="demo",
        components=components,
        edges={("cli", "core"): 3},
        entry_points=["demo → demo.cli:main"],
        env_vars={"DEMO_HOME": "src/core/m.py"},
        tree=["src/  (2 file(s))"],
        files=frozenset({"src/cli/m.py", "src/core/m.py"}),
    )


def test_the_mermaid_diagram_is_deterministic() -> None:
    facts = facts_with_edges()

    assert render_mermaid(facts.components, facts.edges) == render_mermaid(
        list(reversed(facts.components)), dict(reversed(list(facts.edges.items())))
    )


def test_the_mermaid_diagram_names_components_and_edges() -> None:
    facts = facts_with_edges()

    diagram = render_mermaid(facts.components, facts.edges)

    assert diagram.startswith("graph LR")
    assert '["cli"]' in diagram and '["core"]' in diagram
    assert "n0 --> n1" in diagram


def test_a_rendered_architecture_document_passes_its_own_checks() -> None:
    facts = facts_with_edges()
    document = render_architecture(
        facts, "It is a demo.", {"cli": "The command line.", "core": "The engine."}
    )

    assert verify_document("architecture", document, facts) == []
    assert "| `cli` | 1 | The command line. |" in document
    assert "`DEMO_HOME`" in document


def test_a_missing_role_is_a_dash_not_an_invention() -> None:
    document = render_architecture(facts_with_edges(), "Overview.", {})

    assert "| `core` | 1 | — |" in document


# ------------------------------------------------------------------ verification


def good_document() -> tuple[str, ProjectFacts]:
    facts = facts_with_edges()
    return render_architecture(facts, "Overview.", {}), facts


def test_a_missing_section_is_a_problem() -> None:
    document, facts = good_document()

    problems = verify_document("architecture", document.replace("## Entry points", "## Other"), facts)

    assert any("Entry points" in problem for problem in problems)


def test_an_unclosed_fence_is_a_problem() -> None:
    document, facts = good_document()

    assert any("fence" in p for p in verify_document("architecture", document + "\n```python\n", facts))


def test_a_leftover_placeholder_is_a_problem() -> None:
    document, facts = good_document()

    assert any("placeholder" in p for p in verify_document("architecture", document + "{{diff}}", facts))


def test_a_diagram_naming_a_nonexistent_component_is_a_problem() -> None:
    document, facts = good_document()

    problems = verify_document("architecture", document.replace('["core"]', '["ghost"]'), facts)

    assert any("ghost" in p and "not a component" in p for p in problems)


def test_a_diagram_edge_to_an_undefined_node_is_a_problem() -> None:
    document, facts = good_document()

    problems = verify_document("architecture", document.replace("n0 --> n1", "n0 --> n9"), facts)

    assert any("undefined node" in p for p in problems)


def test_a_component_missing_from_the_table_is_a_problem() -> None:
    document, facts = good_document()

    problems = verify_document("architecture", document.replace("| `core` |", "| `xcore` |"), facts)

    assert any("'core' is missing" in p for p in problems)


def test_an_api_reference_naming_an_unindexed_file_is_a_problem() -> None:
    facts = ProjectFacts(name="demo", files=frozenset({"a.py"}))

    problems = verify_document("api", "# x\n\n### `ghost.py`\n", facts)

    assert any("ghost.py" in p for p in problems)


# --------------------------------------------------------------------- the prose


def test_headings_and_fences_are_stripped_from_model_prose() -> None:
    """The model cannot add a section, or open a fence that swallows the document."""
    text = "It is a tool.\n\n## Security\n\n```python\nrm_everything()\n```\n\nIt is fast."

    cleaned = clean_prose(text, max_chars=500)

    assert "##" not in cleaned
    assert "```" not in cleaned
    assert "rm_everything" not in cleaned
    assert "It is a tool." in cleaned and "It is fast." in cleaned


def test_long_prose_is_cut_at_a_sentence() -> None:
    text = "First sentence here. " * 40

    cleaned = clean_prose(text, max_chars=100)

    assert cleaned.endswith(".")
    assert len(cleaned) <= 100


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Owns the retrieval engine.", "Owns the retrieval engine."),
        ("- Owns the retrieval engine.", "Owns the retrieval engine."),
        ('"Owns the retrieval engine."', "Owns the retrieval engine."),
        ("# Heading\nOwns the retrieval engine.", "Owns the retrieval engine."),
        ("", ""),
    ],
)
def test_a_role_is_one_clean_line(reply: str, expected: str) -> None:
    assert clean_role(reply) == expected


def test_a_pipe_in_a_role_cannot_break_the_table() -> None:
    assert "|" not in clean_role("Handles a | b").replace("\\|", "")


def test_paths_the_prose_invents_are_reported() -> None:
    known = frozenset({"src/pkg/core.py", "pyproject.toml"})

    unknown = unknown_paths("See `src/pkg/core.py`, `services/billing.py` and `compute_tax`.", known)

    assert unknown == ["services/billing.py"]


def test_a_path_with_a_dropped_directory_is_accepted() -> None:
    """Models routinely drop a leading directory; flagging it teaches people to ignore the
    warnings."""
    assert unknown_paths("`core.py` does it", frozenset({"src/pkg/core.py"})) == []


# ========================================================================== facts


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


def test_facts_come_from_the_index_and_the_manifest(workspace: Path, repository: IndexRepository) -> None:
    facts = gather_facts(workspace, repository)

    assert facts.name == "py-small-fixture"
    assert "billing service" in facts.description
    assert {"models", "errors", "invoice_service"} <= {c.name for c in facts.components}
    assert facts.test_command == "pytest"
    assert "src/billing/models.py" in facts.files


def test_the_fixtures_real_dependencies_are_found(workspace: Path, repository: IndexRepository) -> None:
    """`invoice_service` raises the errors and builds the models, so both edges exist."""
    facts = gather_facts(workspace, repository)

    assert ("invoice_service", "errors") in facts.edges
    assert ("invoice_service", "models") in facts.edges


def test_environment_variables_are_found_with_their_file(workspace: Path, tmp_path: Path) -> None:
    (workspace / "src" / "billing" / "config.py").write_text(
        'import os\n\nKEY = os.environ["BILLING_KEY"]\nURL = os.getenv("BILLING_URL", "x")\n',
        encoding="utf-8",
    )
    connection = connect(tmp_path / "env.db")
    migrate(connection, database="index")
    repo = IndexRepository(connection)
    Indexer(root=workspace, repository=repo).run()

    facts = gather_facts(workspace, repo)

    assert facts.env_vars == {
        "BILLING_KEY": "src/billing/config.py",
        "BILLING_URL": "src/billing/config.py",
    }


def test_the_tree_lists_a_directory_before_its_children(workspace: Path, repository: IndexRepository) -> None:
    tree = gather_facts(workspace, repository).tree

    assert tree.index(next(line for line in tree if line.startswith("src/"))) < tree.index(
        next(line for line in tree if line.strip().startswith("billing/"))
    )


def test_a_package_json_project_is_read(tmp_path: Path) -> None:
    root = tmp_path / "js"
    (root / "src").mkdir(parents=True)
    (root / "src" / "index.js").write_text("export const a = 1\n", encoding="utf-8")
    (root / "package.json").write_text(
        '{"name": "widget", "description": "A widget", "bin": {"widget": "src/cli.js"},'
        ' "scripts": {"test": "vitest"}}',
        encoding="utf-8",
    )
    connection = connect(tmp_path / "js.db")
    migrate(connection, database="index")
    repo = IndexRepository(connection)
    Indexer(root=root, repository=repo).run()

    facts = gather_facts(root, repo)

    assert facts.name == "widget"
    assert facts.entry_points == ["widget → src/cli.js"]
    assert facts.test_command == "npm test"


# ======================================================================= workflow


class Channel:
    def __init__(self, decision: str = "approve") -> None:
        self.decision = decision
        self.asks: list[ApprovalAsk] = []

    async def proposed(self, **_: object) -> None: ...

    async def started(self, **_: object) -> None: ...

    async def finished(self, **_: object) -> None: ...

    async def request_approval(self, ask: ApprovalAsk) -> ApprovalReply:
        self.asks.append(ask)
        return ApprovalReply(decision=self.decision)


def build(workspace: Path, tmp_path: Path, script: list[ScriptedResponse], decision: str = "approve"):
    connection = connect(tmp_path / "state.db")
    migrate(connection, database="state")
    state = StateRepository(connection)
    session_row = state.create_session(workspace=str(workspace))
    checkpoints = CheckpointStore(state, BlobStore(tmp_path / "blobs"))
    channel = Channel(decision)
    gateway = ToolGateway(
        registry=ToolRegistry([ReadFileTool(), WriteFileTool()]),
        context=ToolContext(workspace=workspace, checkpoints=checkpoints.bind(session_row.id)),
        channel=channel,
        policy=make_policy(SessionView(mode="agent", level="supervised"), ConfigView()),
        session_id=session_row.id,
    )
    session = Session(id=session_row.id, workspace=workspace, model="scripted", num_ctx=8192, mode=Mode.AGENT)
    provider = ScriptedProvider(script)
    return session, gateway, ChatRunner(provider=provider, bus=EventBus()), provider, channel


async def doc(kind, workspace, tmp_path, repository, script, decision="approve"):
    session, gateway, runner, provider, channel = build(workspace, tmp_path, script, decision)
    outcome = await run_doc(
        kind=kind,
        runner=runner,
        session=session,
        gateway=gateway,
        root=workspace,
        repository=repository,
    )
    return outcome, provider, channel


OVERVIEW = "A small billing service that creates, finalises and pays invoices."


async def test_an_architecture_document_is_written_and_well_formed(
    workspace: Path, tmp_path: Path, repository: IndexRepository
) -> None:
    outcome, _provider, _channel = await doc(
        "architecture",
        workspace,
        tmp_path,
        repository,
        [ScriptedResponse(OVERVIEW), ScriptedResponse("Handles one part of billing.")],
    )

    assert outcome.status == "written", outcome.detail
    assert outcome.problems == []
    written = (workspace / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert written == outcome.text
    assert OVERVIEW in written
    assert "Handles one part of billing." in written
    assert "```mermaid" in written


async def test_the_model_cannot_add_structure_to_the_document(
    workspace: Path, tmp_path: Path, repository: IndexRepository
) -> None:
    """The model's overview tries to add a section and open a fence. Neither survives, so
    the document has exactly the sections the template gave it."""
    hostile = "A billing library.\n\n## Secret section\n\n```python\nimport os\n```\n"

    outcome, _provider, _channel = await doc(
        "architecture",
        workspace,
        tmp_path,
        repository,
        [ScriptedResponse(hostile), ScriptedResponse("Role.")],
    )

    assert outcome.status == "written"
    assert "Secret section" not in outcome.text
    assert "import os" not in outcome.text
    assert outcome.text.count("```") % 2 == 0


async def test_a_path_the_model_invents_is_flagged_before_approval(
    workspace: Path, tmp_path: Path, repository: IndexRepository
) -> None:
    outcome, _provider, _channel = await doc(
        "architecture",
        workspace,
        tmp_path,
        repository,
        [ScriptedResponse("Billing logic lives in `services/billing.py`."), ScriptedResponse("Role.")],
    )

    assert outcome.status == "written"
    assert any("services/billing.py" in warning for warning in outcome.warnings)


async def test_the_diff_goes_through_approval(
    workspace: Path, tmp_path: Path, repository: IndexRepository
) -> None:
    _outcome, _provider, channel = await doc(
        "architecture",
        workspace,
        tmp_path,
        repository,
        [ScriptedResponse(OVERVIEW), ScriptedResponse("Role.")],
    )

    assert any("docs/ARCHITECTURE.md" in ask.preview for ask in channel.asks)


async def test_a_rejected_document_is_not_written(
    workspace: Path, tmp_path: Path, repository: IndexRepository
) -> None:
    outcome, _provider, _channel = await doc(
        "architecture",
        workspace,
        tmp_path,
        repository,
        [ScriptedResponse(OVERVIEW), ScriptedResponse("Role.")],
        decision="reject",
    )

    assert outcome.status == "rejected"
    assert not (workspace / "docs" / "ARCHITECTURE.md").exists()


async def test_an_existing_document_is_replaced_through_the_read_first_rule(
    workspace: Path, tmp_path: Path, repository: IndexRepository
) -> None:
    """`write_file` refuses to overwrite a file the session has not read, and a workflow
    does not get to skip that rule — it reads first."""
    (workspace / "docs").mkdir()
    (workspace / "docs" / "ARCHITECTURE.md").write_text("# old\n", encoding="utf-8")

    outcome, _provider, _channel = await doc(
        "architecture",
        workspace,
        tmp_path,
        repository,
        [ScriptedResponse(OVERVIEW), ScriptedResponse("Role.")],
    )

    assert outcome.status == "written", outcome.detail
    assert "# old" not in (workspace / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")


async def test_the_api_reference_needs_no_model(
    workspace: Path, tmp_path: Path, repository: IndexRepository
) -> None:
    """Deterministic all the way down: every entry is a symbol the index holds."""
    outcome, provider, _channel = await doc(
        "api", workspace, tmp_path, repository, [ScriptedResponse("unused")]
    )

    assert outcome.status == "written", outcome.detail
    assert provider.requests == []
    assert "### `src/billing/errors.py`" in outcome.text
    assert "InvoiceNotFound" in outcome.text


async def test_private_symbols_stay_out_of_the_api_reference(
    workspace: Path, tmp_path: Path, repository: IndexRepository
) -> None:
    (workspace / "src" / "billing" / "hidden.py").write_text(
        "def _internal():\n    return 1\n\n\ndef public():\n    return 2\n", encoding="utf-8"
    )
    Indexer(root=workspace, repository=repository).run()

    outcome, _provider, _channel = await doc(
        "api", workspace, tmp_path, repository, [ScriptedResponse("unused")]
    )

    assert "public" in outcome.text
    assert "_internal" not in outcome.text


async def test_a_readme_is_written_with_the_manifests_facts(
    workspace: Path, tmp_path: Path, repository: IndexRepository
) -> None:
    outcome, _provider, _channel = await doc(
        "readme", workspace, tmp_path, repository, [ScriptedResponse(OVERVIEW)]
    )

    assert outcome.status == "written", outcome.detail
    assert outcome.text.startswith("# py-small-fixture")
    assert "Fixture repository: a small billing service" in outcome.text
    assert "pytest" in outcome.text


async def test_an_unindexed_project_is_refused_before_the_model_is_asked(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    connection = connect(tmp_path / "empty.db")
    migrate(connection, database="index")
    empty = IndexRepository(connection)

    outcome, provider, _channel = await doc(
        "architecture", root, tmp_path, empty, [ScriptedResponse("unused")]
    )

    assert outcome.status == "refused"
    assert "hearth index" in outcome.detail
    assert provider.requests == []


async def test_a_component_with_no_symbols_is_not_described_by_the_model(
    workspace: Path, tmp_path: Path, repository: IndexRepository
) -> None:
    """Found in a live run: asked about `broken_syntax`, a file the parser could not read,
    the model answered "not defined by any listed files or symbols" — a sentence about its
    own prompt, sitting in the table as though it were a role. With nothing to go on the
    model is not asked; the row shows a dash."""
    outcome, provider, _channel = await doc(
        "architecture",
        workspace,
        tmp_path,
        repository,
        [ScriptedResponse(OVERVIEW), ScriptedResponse("Handles one part of billing.")],
    )

    assert "| `broken_syntax` | 1 | — |" in outcome.text
    described = [c for c in gather_facts(workspace, repository).components if c.symbols]
    assert len(provider.requests) == 1 + len(described), (
        "an overview, then one call per component with symbols"
    )
