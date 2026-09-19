"""``hearth eval retrieval`` and ``hearth embed``.

These are the commands that touch a real model, so they live apart from the deterministic
index/search commands.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from hearth.config import paths
from hearth.config.loader import LoadedConfig, load_config
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
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate
from hearth.storage.vector_index import NumpyVectorIndex

console = Console()

eval_app = typer.Typer(help="Measure retrieval and model quality.", no_args_is_help=True)

#: Fixture repos the retrieval eval runs against, and their question sets.
EVAL_SETS: dict[str, tuple[str, str]] = {
    "py_small": ("tests/fixtures/repos/py_small", "evals/retrieval/py_small.yaml"),
    "ts_small": ("tests/fixtures/repos/ts_small", "evals/retrieval/ts_small.yaml"),
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
