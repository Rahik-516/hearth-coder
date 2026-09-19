"""``/undo``, ``/rewind``, ``/checkpoints`` and ``hearth undo``.

Shared by the REPL and the standalone command, because the interesting part is not the
plumbing but the **conflict prompt** (docs/safety-and-tool-use.md §7.3), and having two
copies of that would mean two chances to get it wrong.

A conflict means the file on disk no longer holds the bytes Hearth last wrote — someone
edited it in the meantime. Reverting anyway would discard their work, so the revert stops,
says which files and asks. That question is always answered by a human: nothing here
forces past a conflict on its own.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from hearth.config import paths
from hearth.safety.checkpoints import CheckpointStore, RevertReport
from hearth.storage.blobs import BlobStore
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository

#: Asks the user a yes/no question. Injectable so the conflict path can be tested.
Confirm = Callable[[str], bool]


def open_store(root: Path) -> CheckpointStore:
    """A checkpoint store for one repository."""
    database = paths.state_db_path(root)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection: sqlite3.Connection = connect(database)
    migrate(connection, database="state")
    return CheckpointStore(StateRepository(connection), BlobStore(paths.blobs_dir(root)))


def show_checkpoints(console: Console, store: CheckpointStore, session_id: str) -> None:
    """``/checkpoints`` — the write steps this session has taken."""
    steps = store.steps(session_id)
    if not steps:
        console.print("[dim]no file changes in this session[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Step", justify="right")
    table.add_column("When", style="dim")
    table.add_column("Files")
    table.add_column("", style="dim")

    for step in steps:
        table.add_row(
            str(step.step),
            _when(step.created_at),
            ", ".join(escape(path) for path in step.paths),
            "reverted" if step.reverted else "",
        )

    console.print(table)
    console.print("[dim]/undo reverts the newest step · /rewind <step> reverts everything after it[/dim]")


def undo(
    console: Console,
    store: CheckpointStore,
    session_id: str,
    *,
    workspace: Path,
    confirm: Confirm,
    step: int | None = None,
) -> bool:
    """``/undo`` — revert the most recent write step, or a named one."""
    target = step if step is not None else store.latest_step(session_id)
    if target is None:
        console.print("[dim]nothing to undo — no file changes in this session[/dim]")
        return False

    report = store.revert_step(session_id, target, workspace=workspace, trash=paths.trash_dir(workspace))
    return _settle(
        console,
        report,
        retry=lambda: store.revert_step(
            session_id,
            target,
            workspace=workspace,
            force=True,
            trash=paths.trash_dir(workspace),
        ),
        confirm=confirm,
        label=f"step {target}",
    )


def rewind(
    console: Console,
    store: CheckpointStore,
    session_id: str,
    step: int,
    *,
    workspace: Path,
    confirm: Confirm,
) -> bool:
    """``/rewind <step>`` — revert every write step after ``step``, newest first."""
    report = store.revert_after(session_id, step, workspace=workspace, trash=paths.trash_dir(workspace))
    return _settle(
        console,
        report,
        retry=lambda: store.revert_after(
            session_id, step, workspace=workspace, force=True, trash=paths.trash_dir(workspace)
        ),
        confirm=confirm,
        label=f"everything after step {step}",
    )


# ------------------------------------------------------------------ internals


def _settle(
    console: Console,
    report: RevertReport,
    *,
    retry: Callable[[], RevertReport],
    confirm: Confirm,
    label: str,
) -> bool:
    """Report a revert, and offer to force past conflicts.

    The force path exists but is never taken without an explicit yes: a conflict means
    somebody's unrelated work is about to be discarded, and that is not a decision Hearth
    gets to make for them.
    """
    if report.already_reverted:
        console.print(f"[dim]{label} was already reverted[/dim]")
        return False

    if report.conflicts:
        console.print(f"[yellow]{len(report.conflicts)} file(s) changed since Hearth wrote them:[/yellow]")
        for conflict in report.conflicts:
            state = "deleted since" if conflict.deleted else "edited since"
            console.print(f"  · {escape(conflict.path)} [dim]({state})[/dim]")
        console.print("[dim]Reverting would discard those changes.[/dim]")

        if not confirm("Overwrite them anyway?"):
            console.print("[dim]left alone — nothing was reverted[/dim]")
            return False

        report = retry()

    _announce(console, report, label=label)
    return report.touched > 0


def _announce(console: Console, report: RevertReport, *, label: str) -> None:
    if not report.touched:
        console.print(f"[dim]nothing to revert for {label}[/dim]")
        return

    parts = []
    if report.restored:
        parts.append(f"restored {len(report.restored)}")
    if report.removed:
        parts.append(f"removed {len(report.removed)}")
    console.print(f"[green]Reverted {label}[/green] — {', '.join(parts)}.")

    for path in report.restored:
        console.print(f"  · {escape(path)}")
    for path in report.removed:
        console.print(f"  · {escape(path)} [dim](removed; a copy is in Hearth's trash)[/dim]")


def _when(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).strftime("%H:%M:%S")


# ---------------------------------------------------------------- the command


def register(app: typer.Typer) -> None:
    app.command("undo")(undo_command)


def undo_command(
    workspace: Path = typer.Option(None, "--workspace", "-w", help="Repository to act on."),
    session: str = typer.Option(None, "--session", help="Session id. Defaults to the latest."),
    step: int = typer.Option(None, "--step", help="Revert this step instead of the latest."),
    list_only: bool = typer.Option(False, "--list", help="List the write steps and exit."),
) -> None:
    """Undo Hearth's file changes from outside the REPL."""
    from hearth.cli.app import find_workspace_root

    console = Console()
    root = (workspace or find_workspace_root()).resolve()
    store = open_store(root)

    session_id = session
    if session_id is None:
        record = store.repo.latest_session(workspace=str(root))
        if record is None:
            console.print("[dim]no sessions for this repository[/dim]")
            raise typer.Exit(0)
        session_id = record.id

    if list_only:
        show_checkpoints(console, store, session_id)
        raise typer.Exit(0)

    changed = undo(
        console,
        store,
        session_id,
        workspace=root,
        confirm=lambda question: typer.confirm(question, default=False),
        step=step,
    )
    raise typer.Exit(0 if changed else 1)
