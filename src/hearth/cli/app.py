"""Typer application — the CLI frontend.

Interactive sessions talk to the core only through the event bus, never by reaching into
services directly (docs/system-design.md §5.1). ``doctor`` is the one deliberate
exception: it is a one-shot diagnostic whose whole job is inspecting the environment, so
it reads config and probes the provider directly rather than running a session.

Commands are registered here as the milestones land.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from hearth import __version__
from hearth.cli.doctor import DoctorReport, Status, run_doctor_sync
from hearth.config.errors import ConfigError
from hearth.config.loader import load_config
from hearth.llm.errors import ProviderConfigError
from hearth.llm.ollama_provider import OllamaProvider
from hearth.llm.provider import LLMProvider

app = typer.Typer(
    name="hearth",
    help="Fully offline AI coding assistant and project agent for local Ollama models.",
    no_args_is_help=True,
    add_completion=False,
)

_STATUS_STYLE = {
    Status.PASS: ("PASS", "green"),
    Status.WARN: ("WARN", "yellow"),
    Status.FAIL: ("FAIL", "red"),
}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"hearth {__version__}")
        raise typer.Exit


@app.callback()
def cli(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Hearth command-line interface."""


def find_workspace_root(start: Path | None = None) -> Path:
    """Nearest enclosing git repository, falling back to the current directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


@app.command()
def doctor(
    workspace: Path = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Repository to check. Defaults to the enclosing git repository.",
    ),
    skip_ollama: bool = typer.Option(
        False,
        "--skip-ollama",
        help="Skip checks that contact the Ollama server.",
    ),
) -> None:
    """Check that this machine can run Hearth, and explain anything that can't."""
    console = Console()
    root = (workspace or find_workspace_root()).resolve()

    try:
        loaded = load_config(project_root=root)
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1) from exc

    provider: LLMProvider | None = None
    provider_error: str | None = None
    if not skip_ollama:
        try:
            provider = OllamaProvider(
                host=loaded.config.ollama.host,
                allow_remote_host=loaded.config.ollama.allow_remote_host,
                timeout_s=loaded.config.ollama.request_timeout_s,
            )
        except ProviderConfigError as exc:
            provider_error = str(exc)

    report = run_doctor_sync(loaded=loaded, workspace=root, provider=provider)
    _render(console, report, workspace=root)

    if provider_error:
        console.print()
        console.print("[red]FAIL[/red]  Ollama host")
        console.print(f"      {provider_error}")
        raise typer.Exit(1)

    raise typer.Exit(report.exit_code)


def _render(console: Console, report: DoctorReport, *, workspace: Path) -> None:
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("", width=4)
    table.add_column("Check", style="bold")
    table.add_column("Detail", overflow="fold")

    for result in report.results:
        label, style = _STATUS_STYLE[result.status]
        table.add_row(f"[{style}]{label}[/{style}]", result.name, result.detail)

    console.print()
    console.print(f"[bold]Hearth {__version__}[/bold]  ·  workspace: {workspace}")
    console.print()
    console.print(table)

    actionable = [r for r in report.results if r.fix and r.status is not Status.PASS]
    if actionable:
        console.print()
        console.print("[bold]What to do[/bold]")
        for result in actionable:
            console.print(f"  · [bold]{result.name}[/bold]: {result.fix}")

    console.print()
    if report.failed:
        console.print(f"[red]{len(report.failed)} check(s) failed[/red], {len(report.warned)} warning(s).")
    elif report.warned:
        console.print(f"[green]All checks passed[/green], with {len(report.warned)} warning(s).")
    else:
        console.print("[green]All checks passed.[/green]")


def _register_commands() -> None:
    """Attach command groups.

    Imported here rather than at module scope so ``hearth --version`` does not pay for
    loading the indexing stack (docs/system-design.md §15: 1.5s startup budget).
    """
    from hearth.cli import (
        chat_commands,
        checkpoint_commands,
        eval_commands,
        index_commands,
        trust_commands,
    )

    index_commands.register(app)
    eval_commands.register(app)
    chat_commands.register(app)
    trust_commands.register(app)
    checkpoint_commands.register(app)


_register_commands()


def main() -> None:
    """Console-script entry point (``hearth``) and ``python -m hearth``."""
    app()
