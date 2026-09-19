"""``hearth chat``, ``hearth resume`` and ``hearth ask``.

Assembles the pieces a session needs — provider, retrieval, event bus, runner, REPL — and
hands control to the loop. Nothing here contains turn logic; that lives in
``core.runner`` so a non-terminal frontend gets the same behaviour.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from hearth.cli.approval import ApprovalPrompt
from hearth.cli.completers import ChatCompleter, extract_pins
from hearth.cli.render import ChatRenderer
from hearth.cli.repl import ChatREPL
from hearth.config import paths
from hearth.config.errors import ConfigError
from hearth.config.loader import LoadedConfig, load_config
from hearth.core.bus import EventBus
from hearth.core.runner import ChatRunner, QueryEmbedder
from hearth.core.session import Mode, Session, SessionStore
from hearth.indexing.embedder import Embedder
from hearth.llm.errors import LLMError, ProviderConfigError
from hearth.llm.ollama_provider import OllamaProvider
from hearth.llm.profiles import ProfileRegistry
from hearth.llm.types import Message
from hearth.retrieval.engine import RetrievalEngine
from hearth.safety.checkpoints import CheckpointStore
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository
from hearth.storage.vector_index import NumpyVectorIndex

console = Console()


def register(app: typer.Typer) -> None:
    app.command()(chat)
    app.command()(resume)
    app.command()(ask)


def _open_state(root: Path) -> StateRepository:
    """Open (and migrate) ``state.db`` for a workspace."""
    connection = connect(paths.state_db_path(root))
    migrate(connection, database="state")
    return StateRepository(connection)


def _open_index_connection(root: Path) -> sqlite3.Connection | None:
    """Open ``index.db`` read-write, or None when the repo was never indexed."""
    database = paths.index_db_path(root)
    if not database.exists():
        return None
    connection = connect(database)
    migrate(connection, database="index")
    return connection


def _build_runtime(
    root: Path, loaded: LoadedConfig
) -> tuple[OllamaProvider, RetrievalEngine | None, QueryEmbedder | None, sqlite3.Connection | None]:
    """Create the provider, retrieval engine and query embedder for a session."""
    provider = OllamaProvider(
        host=loaded.config.ollama.host,
        allow_remote_host=loaded.config.ollama.allow_remote_host,
        timeout_s=loaded.config.ollama.request_timeout_s,
    )

    index_connection = _open_index_connection(root)
    engine = None
    embed_query = None

    if index_connection is not None:
        dims = loaded.config.models.embed_dimensions or 1024
        vector_index = NumpyVectorIndex(index_connection, model_id=loaded.config.models.embed, dims=dims)
        engine = RetrievalEngine(connection=index_connection, vector_index=vector_index)

        if vector_index.count() > 0:
            embedder = Embedder(
                provider=provider,
                vector_index=vector_index,
                model=loaded.config.models.embed,
                dimensions=loaded.config.models.embed_dimensions,
                profile=ProfileRegistry.load().for_model(loaded.config.models.embed),
                on_cpu=loaded.config.models.embed_placement != "gpu",
            )

            async def embed_query(text: str) -> np.ndarray:
                return await embedder.embed_query(text)

    return provider, engine, embed_query, index_connection


def _load(root: Path) -> LoadedConfig:
    try:
        return load_config(project_root=root)
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1) from exc


def _resolve_root(workspace: Path | None) -> Path:
    from hearth.cli.app import find_workspace_root

    return (workspace or find_workspace_root()).resolve()


def _warn_if_unindexed(root: Path) -> None:
    if not paths.index_db_path(root).exists():
        console.print(
            "[yellow]This repository is not indexed.[/yellow] Answers will have no "
            "repository context. Run [bold]hearth index[/bold] first.\n"
        )


# ----------------------------------------------------------------------- chat


def chat(
    workspace: Path = typer.Option(None, "--workspace", "-w", help="Repository to chat about."),
    model: str = typer.Option(None, "--model", "-m", help="Override the chat model."),
    num_ctx: int = typer.Option(None, "--num-ctx", help="Override the context size."),
) -> None:
    """Ask questions about this repository, interactively."""
    root = _resolve_root(workspace)
    loaded = _load(root)
    _warn_if_unindexed(root)

    session_model = model or loaded.config.models.chat
    session_ctx = num_ctx or loaded.config.models.num_ctx

    state = _open_state(root)
    store = SessionStore(state)
    session = store.create(workspace=root, model=session_model, num_ctx=session_ctx, mode=Mode.CHAT)

    _run_repl(root, loaded, session, store)


def resume(
    session_id: str = typer.Argument(None, help="Session to resume. Defaults to the most recent."),
    workspace: Path = typer.Option(None, "--workspace", "-w", help="Repository."),
) -> None:
    """Continue a previous conversation."""
    root = _resolve_root(workspace)
    loaded = _load(root)

    state = _open_state(root)
    store = SessionStore(state)

    session = store.resume(session_id) if session_id else store.resume_latest(workspace=root)
    if session is None:
        console.print("[yellow]No session to resume.[/yellow] Start one with `hearth chat`.")
        raise typer.Exit(1)

    console.print(f"[dim]resuming {session.id} — {len(session.history)} message(s)[/dim]")
    _run_repl(root, loaded, session, store)


def _run_repl(root: Path, loaded: LoadedConfig, session: Session, store: SessionStore) -> None:
    try:
        provider, engine, embed_query, index_connection = _build_runtime(root, loaded)
    except ProviderConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    bus = EventBus()
    runner = ChatRunner(provider=provider, bus=bus, engine=engine, embed_query=embed_query)

    # Subscribed even in chat mode, where no tool can reach an approval. The frontend's
    # job is to be able to answer one whenever the core asks; wiring it only for agent
    # mode would mean an approval in any other path silently fails closed with no prompt,
    # which looks to the user like a hang (docs/safety-and-tool-use.md §6).
    bus.subscribe(ApprovalPrompt(bus=bus, console=console))

    checkpoints = CheckpointStore(store.repo, BlobStore(paths.blobs_dir(root)))

    history_file = paths.project_data_dir(root) / "repl_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    repl = ChatREPL(
        session=session,
        runner=runner,
        bus=bus,
        store=store,
        console=console,
        history_file=history_file,
        completer=ChatCompleter(index_connection),
        workspace=root,
        checkpoints=checkpoints,
    )

    async def main() -> None:
        try:
            await repl.run()
        finally:
            await bus.close()
            await provider.close()

    try:
        asyncio.run(main())
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


# ------------------------------------------------------------------------ ask


def ask(
    question: str = typer.Argument(..., help="The question to ask."),
    workspace: Path = typer.Option(None, "--workspace", "-w", help="Repository."),
    model: str = typer.Option(None, "--model", "-m", help="Override the chat model."),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", help="List retrieved context."),
) -> None:
    """Ask a single question and print the answer.

    The non-interactive path: one turn, no session persistence, useful for scripting and
    for checking retrieval quality without opening the REPL.
    """
    root = _resolve_root(workspace)
    loaded = _load(root)
    _warn_if_unindexed(root)

    text, pins = extract_pins(question)

    try:
        provider, engine, embed_query, _ = _build_runtime(root, loaded)
    except ProviderConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    session = Session(
        id="ask",
        workspace=root,
        model=model or loaded.config.models.chat,
        num_ctx=loaded.config.models.num_ctx,
    )
    for path in pins:
        session.pin(path)

    bus = EventBus()
    renderer = ChatRenderer(console=console, show_sources=show_sources)
    bus.subscribe(renderer)
    runner = ChatRunner(provider=provider, bus=bus, engine=engine, embed_query=embed_query)

    async def main() -> Message | None:
        try:
            result = await runner.run_turn(session, text or question)
            return Message(role="assistant", content=result.answer) if result.ok else None
        finally:
            await bus.close()
            await provider.close()

    try:
        answer = asyncio.run(main())
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    raise typer.Exit(0 if answer else 1)
