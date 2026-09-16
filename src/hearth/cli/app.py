"""Typer application — the CLI frontend.

Frontends talk to the core only through the event bus (docs/system-design.md §5.1). This
module must not import ``hearth.tools``, ``hearth.safety``, ``hearth.llm`` or
``hearth.storage``; the layer contracts in ``.importlinter`` enforce that.

Commands are registered here as the milestones land. Phase 0 provides ``--version``;
``doctor`` arrives with the LLM gateway (docs/implementation-roadmap.md, Phase 0 task 6).
"""

from __future__ import annotations

import typer

from hearth import __version__

app = typer.Typer(
    name="hearth",
    help="Fully offline AI coding assistant and project agent for local Ollama models.",
    no_args_is_help=True,
    add_completion=False,
)


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


def main() -> None:
    """Console-script entry point (``hearth``) and ``python -m hearth``."""
    app()
