"""``hearth chat``, ``hearth resume`` and ``hearth ask``.

Assembles the pieces a session needs — provider, retrieval, event bus, runner, REPL — and
hands control to the loop. Nothing here contains turn logic; that lives in
``core.runner`` so a non-terminal frontend gets the same behaviour.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
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
from hearth.core.limits import TurnLimits
from hearth.core.runner import AgentTurnResult, ChatRunner, QueryEmbedder
from hearth.core.session import Mode, Session, SessionStore
from hearth.core.tool_channel import EventBusChannel
from hearth.indexing.embedder import Embedder
from hearth.llm.errors import LLMError, ProviderConfigError
from hearth.llm.ollama_provider import OllamaProvider
from hearth.llm.profiles import ProfileRegistry
from hearth.llm.types import Message
from hearth.retrieval.engine import RetrievalEngine
from hearth.retrieval.repomap import RepoMapBuilder
from hearth.safety.audit import AuditLog
from hearth.safety.checkpoints import CheckpointStore
from hearth.safety.invariants import privilege_escalation_dirs
from hearth.safety.policy import (
    ConfigView,
    Decision,
    PolicyRequest,
    SessionView,
    evaluate,
)
from hearth.safety.rules import compile_rules
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository
from hearth.storage.vector_index import NumpyVectorIndex
from hearth.tools.base import ToolContext
from hearth.tools.channel import ApprovalAsk, ApprovalReply
from hearth.tools.gateway import ToolGateway
from hearth.tools.registry import build_default_registry

console = Console()


def register(app: typer.Typer) -> None:
    app.command()(chat)
    app.command()(resume)
    app.command()(ask)
    app.command()(run)


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
    runner = ChatRunner(
        provider=provider,
        bus=bus,
        engine=engine,
        embed_query=embed_query,
        repo_map=RepoMapBuilder(index_connection) if index_connection is not None else None,
    )

    # Subscribed even in chat mode, where no tool can reach an approval. The frontend's
    # job is to be able to answer one whenever the core asks; wiring it only for agent
    # mode would mean an approval in any other path silently fails closed with no prompt,
    # which looks to the user like a hang (docs/safety-and-tool-use.md §6).
    bus.subscribe(ApprovalPrompt(bus=bus, console=console))

    checkpoints = CheckpointStore(store.repo, BlobStore(paths.blobs_dir(root)))

    history_file = paths.project_data_dir(root) / "repl_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    # The REPL starts in chat mode, but it is given a gateway anyway: `/mode agent`,
    # `/plan` and `/execute` all need one, and building it lazily on first use would mean
    # the failure — an unindexed workspace, say — surfaces three commands into a session
    # instead of at startup. Mode still decides which tools the model is shown; holding a
    # gateway grants nothing on its own.
    gateway, grants = _build_gateway(
        root,
        loaded,
        session=session,
        bus=bus,
        checkpoints=checkpoints,
        index_connection=index_connection,
        engine=engine,
        headless=False,
        allow_edits=False,
        allow_tests=False,
        allow_commit=False,
    )

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
        gateway=gateway,
        grants=grants,
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
        provider, engine, embed_query, index_connection = _build_runtime(root, loaded)
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
    runner = ChatRunner(
        provider=provider,
        bus=bus,
        engine=engine,
        embed_query=embed_query,
        repo_map=RepoMapBuilder(index_connection) if index_connection is not None else None,
    )

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


# ------------------------------------------------------------------------ run


def _build_gateway(
    root: Path,
    loaded: LoadedConfig,
    *,
    session: Session,
    bus: EventBus,
    checkpoints: CheckpointStore,
    index_connection: sqlite3.Connection | None,
    engine: RetrievalEngine | None,
    headless: bool,
    allow_edits: bool,
    allow_tests: bool,
    allow_commit: bool,
) -> tuple[ToolGateway, set[str]]:
    """Assemble the gateway for an agent session.

    The ``--allow-*`` flags are deliberately narrow: each widens exactly one risk class in
    headless mode and nothing else (docs/safety-and-tool-use.md §14). There is no
    ``--allow-all``, because the flag a user types is the only record of what they
    consented to before walking away.

    Returns the gateway and the live grant set it decides against. The set is returned
    rather than kept private because grants are *session* state, not gateway state: an
    approved plan adds edit grants to it and a superseding plan removes them, and both
    happen in the frontend, between turns, with no tool call in sight.
    """
    grants: set[str] = set()

    def policy(request: PolicyRequest) -> Decision:
        # Rebuilt per call so a grant added mid-run is visible to the next one.
        return evaluate(
            request,
            SessionView(
                mode=session.mode.value,
                level=session.permission_level.value,
                headless=headless,
                grants=frozenset(grants),
            ),
            ConfigView(
                rules=compile_rules(loaded.config.permissions, source="global"),
                protected_dirs=privilege_escalation_dirs(),
                headless_allow_edits=allow_edits,
                headless_allow_tests=allow_tests,
                headless_allow_commit=allow_commit,
            ),
        )

    context = ToolContext(
        workspace=root,
        index_connection=index_connection,
        retrieval_engine=engine,
        checkpoints=checkpoints.bind(session.id),
        blobs=BlobStore(paths.blobs_dir(root)),
    )

    gateway = ToolGateway(
        registry=build_default_registry(
            test_command=loaded.config.project.test_command,
            test_command_source=str(paths.project_config_file(root))
            if loaded.config.project.test_command
            else None,
        ),
        context=context,
        channel=_HeadlessChannel(bus) if headless else EventBusChannel(bus),
        audit=AuditLog(paths.audit_dir()),
        policy=policy,
        on_grant=grants.add,
        session_id=session.id,
    )
    return gateway, grants


class _HeadlessChannel(EventBusChannel):
    """Reports progress but never obtains an approval.

    The policy engine already converts an ask into a deny in headless mode, so this should
    be unreachable. It exists because "should be unreachable" is not a safety property: if
    some path ever does request an approval with nobody there to answer, the alternatives
    are denying it or waiting forever on a bus nobody is listening to.
    """

    async def request_approval(self, ask: ApprovalAsk) -> ApprovalReply | None:
        return None


def run(
    task: str = typer.Argument(..., help="What you want done, in a sentence or two."),
    workspace: Path = typer.Option(None, "--workspace", "-w", help="Repository to work in."),
    model: str = typer.Option(None, "--model", "-m", help="Override the chat model."),
    headless: bool = typer.Option(
        False, "--headless", help="Run without prompts. Every side effect is denied unless allowed."
    ),
    allow_edits: bool = typer.Option(False, "--allow-edits", help="Headless: permit file edits."),
    allow_tests: bool = typer.Option(
        False, "--allow-tests", help="Headless: permit running commands and tests."
    ),
    allow_commit: bool = typer.Option(False, "--allow-commit", help="Headless: permit git writes."),
    max_steps: int = typer.Option(None, "--max-steps", help="Override the model profile's step limit."),
) -> None:
    """Carry out a task: edit, run tests, and stop for approval before each side effect.

    Interactive by default. ``--headless`` fails closed — it does not skip approvals, it
    refuses everything that would have needed one, and names the flag that would have
    permitted it. That asymmetry is the point: an unattended run that quietly edits files
    is the failure mode the flags exist to prevent.
    """
    root = _resolve_root(workspace)
    loaded = _load(root)
    _warn_if_unindexed(root)

    try:
        provider, engine, embed_query, index_connection = _build_runtime(root, loaded)
    except ProviderConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    state = _open_state(root)
    store = SessionStore(state)
    session = store.create(
        workspace=root,
        model=model or loaded.config.models.chat,
        num_ctx=loaded.config.models.num_ctx,
        mode=Mode.AGENT,
    )
    store.set_title_from(session, task)

    bus = EventBus()
    bus.subscribe(ChatRenderer(console=console))
    if not headless:
        bus.subscribe(ApprovalPrompt(bus=bus, console=console))

    checkpoints = CheckpointStore(store.repo, BlobStore(paths.blobs_dir(root)))
    gateway, _grants = _build_gateway(
        root,
        loaded,
        session=session,
        bus=bus,
        checkpoints=checkpoints,
        index_connection=index_connection,
        engine=engine,
        headless=headless,
        allow_edits=allow_edits,
        allow_tests=allow_tests,
        allow_commit=allow_commit,
    )

    profile = ProfileRegistry.load().for_model(session.model)
    limits = TurnLimits.from_profile(profile, session.mode.value)
    if max_steps:
        limits = replace(limits, max_steps=max_steps)

    runner = ChatRunner(
        provider=provider,
        bus=bus,
        engine=engine,
        embed_query=embed_query,
        repo_map=RepoMapBuilder(index_connection) if index_connection is not None else None,
    )
    # Asked of the gateway rather than of a second registry built here. Two registries
    # is one too many: they are constructed with different arguments, so the schemas the
    # model was shown could describe a tool selection the gateway never had.
    availability = gateway.availability(
        session.mode.value, tool_reliability=profile.tool_reliability
    )

    async def main() -> AgentTurnResult:
        try:
            return await runner.run_agent_turn(
                session,
                task,
                gateway=gateway,
                tool_schemas=availability.schemas,
                limits=limits,
                think="medium" if profile.supports_thinking else "off",
            )
        finally:
            await bus.close()
            await provider.close()

    try:
        result = asyncio.run(main())
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        # Printed even here: §14.1 asks for the summary on every exit path, and "it died
        # three steps in" is the part a user needs when a run fails.
        console.print("[dim]run_result: reason=error steps=0 tool_calls=0[/dim]")
        raise typer.Exit(1) from exc

    if result.answer:
        store.save_message(session, Message(role="user", content=task))
        store.save_message(session, Message(role="assistant", content=result.answer))

    console.print(f"\n[dim]{result.summary()} · session {session.id}[/dim]")
    if not result.ok:
        console.print("[dim]`hearth undo` reverts the file changes from this run.[/dim]")
    raise typer.Exit(0 if result.ok else 1)
