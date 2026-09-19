"""``hearth init``, ``hearth index`` and ``hearth search``.

Registered onto the main Typer app in ``cli/app.py``. Kept separate so the app module
stays a thin command surface rather than growing a body per milestone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table

from hearth.config import paths
from hearth.config.errors import ConfigError
from hearth.config.loader import LoadedConfig, load_config
from hearth.indexing.filters import PathFilter, load_ignore_file
from hearth.indexing.pipeline import Indexer, IndexProgress, IndexStats
from hearth.retrieval.engine import RetrievalResult
from hearth.storage.db import connect, has_fts5
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate
from hearth.util.text import build_search_query

console = Console()


def register(app: typer.Typer) -> None:
    app.command()(init)
    app.command()(index)
    app.command()(search)


# --------------------------------------------------------------------- helpers


def _resolve_root(workspace: Path | None) -> Path:
    from hearth.cli.app import find_workspace_root

    return (workspace or find_workspace_root()).resolve()


def _load(root: Path) -> LoadedConfig:
    try:
        return load_config(project_root=root)
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1) from exc


def _open_index(root: Path, *, create: bool = True) -> IndexRepository:
    database = paths.index_db_path(root)
    if not create and not database.exists():
        console.print(f"[red]No index for {root}.[/red] Run [bold]hearth index[/bold] first.")
        raise typer.Exit(1)

    connection = connect(database)
    if not has_fts5(connection):
        console.print("[red]This SQLite build has no FTS5.[/red] Run `hearth doctor`.")
        raise typer.Exit(1)

    migrate(connection, database="index")
    return IndexRepository(connection)


def _build_filter(root: Path, loaded: LoadedConfig) -> PathFilter:
    def read(name: str) -> str | None:
        try:
            return (root / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    return PathFilter(
        exclude=loaded.config.index.exclude,
        include=loaded.config.index.include,
        gitignore=load_ignore_file(read(".gitignore")),
        hearthignore=load_ignore_file(read(".hearthignore")),
    )


# ----------------------------------------------------------------------- init


def init(
    workspace: Path = typer.Option(None, "--workspace", "-w", help="Repository to initialize."),
    create_config: bool = typer.Option(False, "--config", help="Also write a starter .hearth/config.toml."),
) -> None:
    """Create the project data directory for this repository."""
    root = _resolve_root(workspace)
    data_dir = paths.project_data_dir(root)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "blobs").mkdir(exist_ok=True)

    console.print(f"[green]Initialized[/green] {root}")
    console.print(f"  data directory: {data_dir}")
    console.print(f"  project id:     {paths.project_id(root)}")

    if create_config:
        config_path = paths.project_config_file(root)
        if config_path.exists():
            console.print(f"  [yellow]kept existing[/yellow] {config_path}")
        else:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(_STARTER_CONFIG, encoding="utf-8")
            console.print(f"  wrote {config_path}")

    console.print("\nNext: [bold]hearth index[/bold]")


_STARTER_CONFIG = """\
# Project settings for Hearth. Committable.
#
# Permission ALLOW rules here are ignored until you run `hearth trust`, because a cloned
# repository must not be able to widen its own permissions. Deny rules always apply.

[project]
# test_command = "uv run pytest -q {target}"
# lint_command = "uv run ruff check {target}"

[index]
exclude = []
"""


# ---------------------------------------------------------------------- index


def index(
    workspace: Path = typer.Option(None, "--workspace", "-w", help="Repository to index."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Discard the index and rebuild it."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print the summary."),
) -> None:
    """Build or update the lexical index for this repository."""
    root = _resolve_root(workspace)
    loaded = _load(root)

    if rebuild:
        database = paths.index_db_path(root)
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(database) + suffix)
            if candidate.exists():
                candidate.unlink()
        console.print("[yellow]Rebuilding from scratch.[/yellow]")

    repo = _open_index(root)

    if quiet:
        stats = Indexer(
            root=root,
            repository=repo,
            path_filter=_build_filter(root, loaded),
            max_file_bytes=loaded.config.index.max_file_bytes,
        ).run(force=rebuild)
    else:
        stats = _run_with_progress(root, repo, loaded, force=rebuild)

    _print_index_summary(root, stats)


def _run_with_progress(root: Path, repo: IndexRepository, loaded: LoadedConfig, *, force: bool) -> IndexStats:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("scanning", total=None)

        def on_progress(event: IndexProgress) -> None:
            if event.phase == "indexing":
                progress.update(
                    task,
                    description="indexing",
                    total=event.total,
                    completed=event.done,
                )
            else:
                progress.update(task, description=event.phase)

        return Indexer(
            root=root,
            repository=repo,
            path_filter=_build_filter(root, loaded),
            max_file_bytes=loaded.config.index.max_file_bytes,
            on_progress=on_progress,
        ).run(force=force)


def _print_index_summary(root: Path, stats: IndexStats) -> None:
    console.print(f"\n[bold]Indexed[/bold] {root}")

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("", style="bold")
    table.add_column("")

    table.add_row("files", f"{stats.added} added, {stats.modified} modified, {stats.deleted} deleted")
    table.add_row("unchanged", str(stats.unchanged))
    table.add_row("chunks", str(stats.chunks_written))
    table.add_row("symbols", str(stats.symbols_written))
    table.add_row("parsed", str(stats.parsed))
    table.add_row("elapsed", f"{stats.duration_s:.2f}s")
    console.print(table)

    notes: list[str] = []
    if stats.rebuilt_for_version:
        notes.append(
            f"rebuilt: chunking changed since this index was built ({stats.rebuilt_for_version})"
        )
    if stats.skipped_by_reason.get("secret-file"):
        notes.append(f"{stats.skipped_by_reason['secret-file']} secret file(s) excluded")
    if stats.generated_metadata_only:
        notes.append(f"{stats.generated_metadata_only} generated file(s) indexed as metadata only")
    if stats.skipped_binary:
        notes.append(f"{stats.skipped_binary} binary file(s) skipped")
    if stats.skipped_large:
        notes.append(f"{stats.skipped_large} oversized file(s) skipped")
    if stats.parse_failures:
        notes.append(f"{stats.parse_failures} file(s) failed to parse")

    for note in notes:
        console.print(f"  [dim]·[/dim] {note}")

    if stats.parsed == 0 and stats.files_written == 0:
        console.print("  [dim]· nothing changed — no files were parsed[/dim]")


# --------------------------------------------------------------------- search


def search(
    query: str = typer.Argument(..., help="What to look for."),
    workspace: Path = typer.Option(None, "--workspace", "-w", help="Repository to search."),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results."),
    lexical_only: bool = typer.Option(
        False, "--lexical-only", help="Skip dense search, even if embeddings exist."
    ),
    explain: bool = typer.Option(
        False, "--explain", help="Show query analysis, per-retriever ranks and adjustments."
    ),
    show_text: bool = typer.Option(False, "--text", help="Print the matching chunk text."),
) -> None:
    """Search the index.

    Hybrid by default: BM25, symbol, path and — when embeddings exist and Ollama is
    reachable — dense. Falls back to the lexical retrievers rather than failing, so search
    works while embeddings are still building.
    """
    root = _resolve_root(workspace)
    loaded = _load(root)
    repo = _open_index(root, create=False)

    if repo.count_chunks() == 0:
        console.print("[yellow]The index is empty.[/yellow] Run [bold]hearth index[/bold].")
        raise typer.Exit(1)

    from hearth.retrieval.engine import RetrievalEngine
    from hearth.storage.vector_index import NumpyVectorIndex

    vector_index = NumpyVectorIndex(
        repo.connection,
        model_id=loaded.config.models.embed,
        dims=loaded.config.models.embed_dimensions or 1024,
    )
    engine = RetrievalEngine(connection=repo.connection, vector_index=vector_index)

    query_vector = None
    if not lexical_only and vector_index.count() > 0:
        query_vector = _embed_query_or_none(loaded, query)

    result = engine.retrieve(
        query,
        limit=limit,
        query_vector=query_vector,
        mode="lexical" if lexical_only else "hybrid",
    )

    if explain:
        _print_explanation(result)

    if not result.results:
        console.print("[yellow]No matches.[/yellow]")
        raise typer.Exit(1)

    for position, found in enumerate(result.results, start=1):
        symbol = found.symbol_path or found.kind
        console.print(f"[bold]{position:>2}.[/bold] [cyan]{found.citation}[/cyan]  [dim]{symbol}[/dim]")

        if explain:
            ranks = "  ".join(f"{r.value}#{found.ranks[r]}" for r in found.retrievers)
            adjustments = "  ".join(f"{label} x{factor}" for label, factor in found.adjustments)
            console.print(f"     [dim]score {found.score:.5f}  ·  {ranks}[/dim]")
            if adjustments:
                console.print(f"     [dim]adjustments: {adjustments}[/dim]")

        if show_text:
            body = found.text
            truncated = body if len(body) <= 1200 else body[:1200] + "\n..."
            console.print(
                Syntax(
                    truncated,
                    found.language or "text",
                    line_numbers=False,
                    background_color="default",
                )
            )


def _print_explanation(result: RetrievalResult) -> None:
    """Show how the query was read and which retrievers contributed."""
    analysis = result.analysis
    console.print(f"[dim]query:     [/dim] {result.query}")
    console.print(f"[dim]intent:    [/dim] {analysis.intent.value}")
    if analysis.known_symbols:
        console.print(f"[dim]symbols:   [/dim] {', '.join(analysis.known_symbols)}")
    if analysis.paths:
        console.print(f"[dim]paths:     [/dim] {', '.join(analysis.paths)}")
    console.print(f"[dim]terms:     [/dim] {' '.join(analysis.terms[:12])}")

    counts = "  ".join(f"{name.value}={len(found)}" for name, found in sorted(result.per_retriever.items()))
    console.print(f"[dim]retrievers:[/dim] {counts or 'none'}")

    embedded, total = result.coverage
    if total:
        state = "complete" if result.is_fully_embedded else "partial"
        console.print(f"[dim]embeddings:[/dim] {embedded}/{total} ({state})")
    if not result.dense_used and result.dense_skipped_reason:
        console.print(f"[dim]dense:     [/dim] skipped — {result.dense_skipped_reason}")
    console.print(f"[dim]took:      [/dim] {result.duration_ms:.1f} ms")
    console.print()


def _embed_query_or_none(loaded: LoadedConfig, query: str) -> np.ndarray | None:
    """Embed a query, returning None if the model is unreachable.

    A missing Ollama must degrade search to lexical, not break it.
    """
    import asyncio

    from hearth.indexing.embedder import Embedder
    from hearth.llm.errors import LLMError
    from hearth.llm.ollama_provider import OllamaProvider
    from hearth.llm.profiles import ProfileRegistry
    from hearth.storage.vector_index import NumpyVectorIndex

    async def once() -> np.ndarray:
        provider = OllamaProvider(
            host=loaded.config.ollama.host,
            allow_remote_host=loaded.config.ollama.allow_remote_host,
            timeout_s=loaded.config.ollama.request_timeout_s,
        )
        try:
            embedder = Embedder(
                provider=provider,
                vector_index=NumpyVectorIndex(connect(":memory:"), model_id="unused", dims=1),
                model=loaded.config.models.embed,
                dimensions=loaded.config.models.embed_dimensions,
                profile=ProfileRegistry.load().for_model(loaded.config.models.embed),
                on_cpu=loaded.config.models.embed_placement != "gpu",
            )
            return await embedder.embed_query(query)
        finally:
            await provider.close()

    try:
        return asyncio.run(once())
    except LLMError:
        return None


def _build_fts_query(query: str) -> str:
    """Turn a user query into an FTS5 MATCH expression.

    Kept for callers that want the raw expression; the engine builds its own.
    """
    terms = build_search_query(query) or [t for t in query.split() if t]
    quoted = [f'"{term}"' for term in terms if term]
    if not quoted:
        return '""'
    return " OR ".join(quoted)
