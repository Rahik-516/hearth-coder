"""``hearth eval retrieval`` and ``hearth embed``.

These are the commands that touch a real model, so they live apart from the deterministic
index/search commands.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from hearth.cli.render import ChatRenderer
from hearth.config import paths
from hearth.config.loader import LoadedConfig, load_config
from hearth.core.bus import EventBus
from hearth.core.limits import TurnLimits
from hearth.core.runner import AgentTurnResult, ChatRunner
from hearth.core.session import Mode, SessionStore
from hearth.evals.retrieval_eval import (
    EvalDataError,
    EvalDependencyError,
    EvalQuestion,
    EvalReport,
    QueryEmbedder,
    evaluate_mode,
    format_report,
    load_questions,
)
from hearth.indexing.embedder import Embedder
from hearth.indexing.pipeline import Indexer
from hearth.llm.errors import LLMError
from hearth.llm.ollama_provider import OllamaProvider
from hearth.llm.profiles import ProfileRegistry
from hearth.retrieval.engine import RetrievalEngine
from hearth.retrieval.repomap import RepoMapBuilder
from hearth.safety.checkpoints import CheckpointStore
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate
from hearth.storage.vector_index import NumpyVectorIndex
from hearth.tools.registry import build_default_registry

console = Console()

eval_app = typer.Typer(help="Measure retrieval and model quality.", no_args_is_help=True)

#: Fixture repos the retrieval eval runs against, and their question sets.
EVAL_SETS: dict[str, tuple[str, str]] = {
    "py_small": ("tests/fixtures/repos/py_small", "evals/retrieval/py_small.yaml"),
    "ts_small": ("tests/fixtures/repos/ts_small", "evals/retrieval/ts_small.yaml"),
    # The I1 acceptance subset. Runs against Hearth itself rather than a fixture: these
    # are architecture questions, and a 10-file fixture has no architecture to ask about.
    "global": (".", "evals/retrieval/global_questions.yaml"),
}


def register(app: typer.Typer) -> None:
    app.add_typer(eval_app, name="eval")
    app.command()(embed)


def _config(root: Path) -> LoadedConfig:
    return load_config(project_root=root)


def _provider(loaded: LoadedConfig) -> OllamaProvider:
    return OllamaProvider(
        host=loaded.config.ollama.host,
        allow_remote_host=loaded.config.ollama.allow_remote_host,
        timeout_s=loaded.config.ollama.request_timeout_s,
    )


def _embedder(loaded: LoadedConfig, vector_index: NumpyVectorIndex, provider: OllamaProvider) -> Embedder:
    models = loaded.config.models
    return Embedder(
        provider=provider,
        vector_index=vector_index,
        model=models.embed,
        dimensions=models.embed_dimensions,
        profile=ProfileRegistry.load().for_model(models.embed),
        on_cpu=models.embed_placement == "cpu",
    )


# ---------------------------------------------------------------------- embed


def embed(
    workspace: Path = typer.Option(None, "--workspace", "-w", help="Repository to embed."),
    limit: int = typer.Option(None, "--limit", help="Stop after this many chunk texts."),
) -> None:
    """Fill in missing embeddings for the index.

    Safe to interrupt: each batch commits before the next starts, so a restart re-embeds
    nothing already stored.
    """
    from hearth.cli.app import find_workspace_root

    root = (workspace or find_workspace_root()).resolve()
    loaded = _config(root)

    database = paths.index_db_path(root)
    if not database.exists():
        console.print("[red]No index.[/red] Run [bold]hearth index[/bold] first.")
        raise typer.Exit(1)

    connection = connect(database)
    migrate(connection, database="index")
    vector_index = NumpyVectorIndex(
        connection,
        model_id=loaded.config.models.embed,
        dims=loaded.config.models.embed_dimensions or 1024,
    )

    embedded, total = vector_index.coverage()
    console.print(f"Coverage before: {embedded}/{total}")

    provider = _provider(loaded)
    embedder = _embedder(loaded, vector_index, provider)

    async def run() -> None:
        try:
            stats = await embedder.run(limit=limit)
        finally:
            await provider.close()

        console.print(f"[green]{stats.summary()}[/green] in {stats.duration_s:.1f}s")
        for error in stats.errors[:3]:
            console.print(f"  [red]·[/red] {error}")

    try:
        asyncio.run(run())
    except LLMError as exc:
        console.print(f"[red]Embedding failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    after, total = vector_index.coverage()
    console.print(f"Coverage after:  {after}/{total}")


# ----------------------------------------------------------------- eval suite


@eval_app.command("retrieval")
def eval_retrieval(
    sets: list[str] = typer.Option(None, "--set", "-s", help="Eval sets to run. Default: all."),
    repo_root: Path = typer.Option(
        None, "--repo-root", help="Hearth checkout containing fixtures and eval data."
    ),
    skip_embed: bool = typer.Option(
        False, "--skip-embed", help="Do not embed first; measures lexical-only behaviour."
    ),
    show_misses: bool = typer.Option(True, "--misses/--no-misses", help="List missed questions."),
) -> None:
    """Measure retrieval quality against the fixture question sets.

    Indexes each fixture into a temporary database, embeds it, then runs lexical-only,
    dense-only and hybrid so the three are directly comparable.
    """
    from hearth.cli.app import find_workspace_root

    base = (repo_root or find_workspace_root()).resolve()
    selected = sets or list(EVAL_SETS)

    loaded = _config(base)
    reports: list[EvalReport] = []

    for name in selected:
        if name not in EVAL_SETS:
            console.print(f"[red]Unknown eval set {name!r}.[/red] Known: {', '.join(EVAL_SETS)}")
            raise typer.Exit(1)

        fixture_rel, questions_rel = EVAL_SETS[name]
        fixture = base / fixture_rel
        questions_path = base / questions_rel

        if not fixture.is_dir() or not questions_path.is_file():
            console.print(f"[red]Missing fixture or question set for {name}.[/red]")
            raise typer.Exit(1)

        try:
            questions = load_questions(questions_path)
        except (EvalDependencyError, EvalDataError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

        console.print(f"[bold]{name}[/bold]: {len(questions)} questions")
        reports.append(
            _run_one_set(
                name=name,
                fixture=fixture,
                questions=questions,
                loaded=loaded,
                skip_embed=skip_embed,
            )
        )

    console.print()
    failures = 0
    for report in reports:
        console.print(format_report(report, show_misses=show_misses))
        console.print()

        hybrid = report.hybrid
        if hybrid is None:
            continue
        if hybrid.recall_at(10) < 0.80:
            console.print(
                f"[red]FAIL[/red] {report.name}: hybrid recall@10 "
                f"{hybrid.recall_at(10):.2f} is below the 0.80 acceptance bar."
            )
            failures += 1
        elif not report.hybrid_beats_parts():
            console.print(
                f"[yellow]WARN[/yellow] {report.name}: hybrid does not beat both single-signal baselines."
            )
        else:
            console.print(f"[green]PASS[/green] {report.name}")

    raise typer.Exit(1 if failures else 0)


def _run_one_set(
    *,
    name: str,
    fixture: Path,
    questions: list[EvalQuestion],
    loaded: LoadedConfig,
    skip_embed: bool,
) -> EvalReport:
    """Index a fixture into a scratch database and measure every mode."""
    import tempfile

    report = EvalReport(name=name)

    with tempfile.TemporaryDirectory(prefix=f"hearth-eval-{name}-") as tmp:
        connection = connect(Path(tmp) / "index.db")
        migrate(connection, database="index")
        Indexer(root=fixture, repository=IndexRepository(connection)).run()

        dims = loaded.config.models.embed_dimensions or 1024
        vector_index = NumpyVectorIndex(connection, model_id=loaded.config.models.embed, dims=dims)
        engine = RetrievalEngine(connection=connection, vector_index=vector_index)

        embed_query = None
        if not skip_embed:
            embed_query = _embed_fixture(loaded, vector_index)

        report.coverage = vector_index.coverage()

        report.modes["lexical"] = evaluate_mode(engine, questions, mode="lexical")
        if embed_query is not None:
            report.modes["dense"] = evaluate_mode(engine, questions, mode="dense", embed_query=embed_query)
            report.modes["hybrid"] = evaluate_mode(engine, questions, mode="hybrid", embed_query=embed_query)
        else:
            # Without embeddings, "hybrid" is lexical plus symbol and path — still worth
            # reporting so a --skip-embed run is not silently mislabelled.
            report.modes["hybrid"] = evaluate_mode(engine, questions, mode="hybrid")

    return report


def _embed_fixture(loaded: LoadedConfig, vector_index: NumpyVectorIndex) -> QueryEmbedder:
    """Embed a fixture index and return a synchronous query-embedding callable."""
    provider = _provider(loaded)
    embedder = _embedder(loaded, vector_index, provider)

    async def fill() -> None:
        try:
            await embedder.run()
        finally:
            await provider.close()

    with console.status("embedding fixture…"):
        asyncio.run(fill())

    cache: dict[str, np.ndarray] = {}

    def embed_query(text: str) -> np.ndarray:
        if text in cache:
            return cache[text]

        async def once() -> np.ndarray:
            local = _provider(loaded)
            try:
                return await _embedder(loaded, vector_index, local).embed_query(text)
            finally:
                await local.close()

        vector = asyncio.run(once())
        cache[text] = vector
        return vector

    return embed_query


@eval_app.command("tasks")
def eval_tasks(
    repo_root: Path = typer.Option(
        None, "--repo-root", help="Hearth checkout containing tests/fixtures/repos."
    ),
    model: str = typer.Option(None, "--model", "-m", help="Model to evaluate."),
    only: list[str] = typer.Option(None, "--task", "-t", help="Run only these tasks."),
    keep: bool = typer.Option(False, "--keep", help="Keep the disposable workspaces."),
    max_steps: int = typer.Option(None, "--max-steps", help="Override the profile's step limit."),
) -> None:
    """Run the agent task suite and report pass/fail with timings.

    Each task runs **headless with every side effect allowed**, in a throwaway copy of a
    fixture repository. That combination is deliberate: the eval measures whether the loop
    can finish a change on its own, which a run that stops for approval cannot answer,
    and the blast radius stays inside a directory that is deleted afterwards.
    """
    from hearth.evals.task_eval import TASKS, EvalReport, prepare_workspace, score

    root = (repo_root or Path.cwd()).resolve()
    selected = [task for task in TASKS if not only or task.name in only]
    if not selected:
        console.print(f"[yellow]no matching tasks[/yellow] — have: {', '.join(t.name for t in TASKS)}")
        raise typer.Exit(1)

    loaded = _config(root)
    chat_model = model or loaded.config.models.chat
    report = EvalReport()

    with tempfile.TemporaryDirectory(prefix="hearth-task-eval-") as scratch:
        for spec in selected:
            workspace = prepare_workspace(
                spec, repo_root=root, destination=Path(scratch) / spec.name
            )
            console.print(f"[dim]{spec.name}: {workspace}[/dim]")
            started = time.monotonic()

            result = _run_task(workspace, chat_model, spec.prompt, max_steps=max_steps)
            report.outcomes.append(
                score(
                    spec,
                    workspace,
                    steps=result.steps,
                    tool_calls=result.tool_calls,
                    reason=result.reason,
                    started=started,
                )
            )
            if keep:
                kept = root / f"task-eval-{spec.name}"
                shutil.copytree(workspace, kept, dirs_exist_ok=True)
                console.print(f"[dim]kept: {kept}[/dim]")

    console.print()
    console.print(report.render())
    raise typer.Exit(0 if report.passed == report.total else 1)


def _run_task(workspace: Path, model: str, prompt: str, *, max_steps: int | None) -> AgentTurnResult:
    """One headless agent run against a disposable workspace."""
    from hearth.cli.chat_commands import _build_gateway, _build_runtime, _open_state

    # Index the copy first. A real user runs `hearth index` before asking for work, and an
    # unindexed workspace makes `search_code` fail on every call — which would measure the
    # eval's setup rather than the model.
    _index_workspace(workspace)

    loaded = load_config(project_root=workspace)
    provider, engine, embed_query, index_connection = _build_runtime(workspace, loaded)

    store = SessionStore(_open_state(workspace))
    session = store.create(
        workspace=workspace,
        model=model,
        num_ctx=loaded.config.models.num_ctx,
        mode=Mode.AGENT,
    )

    bus = EventBus()
    bus.subscribe(ChatRenderer(console=console))
    checkpoints = CheckpointStore(store.repo, BlobStore(paths.blobs_dir(workspace)))

    gateway, _grants = _build_gateway(
        workspace,
        loaded,
        session=session,
        bus=bus,
        checkpoints=checkpoints,
        index_connection=index_connection,
        engine=engine,
        headless=True,
        allow_edits=True,
        allow_tests=True,
        allow_commit=False,
    )

    profile = ProfileRegistry.load().for_model(model)
    limits = TurnLimits.from_profile(profile, "agent")
    if max_steps:
        limits = replace(limits, max_steps=max_steps)

    runner = ChatRunner(
        provider=provider,
        bus=bus,
        engine=engine,
        embed_query=embed_query,
        repo_map=RepoMapBuilder(index_connection) if index_connection is not None else None,
    )
    schemas = gateway_schemas(loaded, profile)

    async def main() -> AgentTurnResult:
        try:
            return await runner.run_agent_turn(
                session,
                prompt,
                gateway=gateway,
                tool_schemas=schemas,
                limits=limits,
                think="medium" if profile.supports_thinking else "off",
            )
        finally:
            await bus.close()
            await provider.close()

    try:
        return asyncio.run(main())
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        return AgentTurnResult(answer="", reason="error")


def gateway_schemas(loaded: LoadedConfig, profile: object) -> list[dict[str, object]]:
    """The tool schemas an agent session is shown, for the configured project."""
    registry = build_default_registry(test_command=loaded.config.project.test_command)
    return registry.for_mode(
        "agent", tool_reliability=getattr(profile, "tool_reliability", "high")
    ).schemas


def _index_workspace(workspace: Path) -> None:
    """Build the lexical index for a disposable eval workspace."""
    connection = connect(paths.index_db_path(workspace))
    migrate(connection, database="index")
    Indexer(root=workspace, repository=IndexRepository(connection)).run()
    connection.close()
