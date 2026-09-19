"""``hearth trust`` — reviewing and recording trust in a project's own config.

The command exists because of one asymmetry: a project's `.hearth/config.toml` can
*tighten* permissions freely, but any rule that relaxes them is ignored until a human has
seen it (docs/safety-and-tool-use.md §5.5). So this is not a yes/no toggle — it is a
review screen, and the listing of relaxing rules is the substance of it.

Trust is recorded against the config file's bytes, so switching to a branch with a
different config, or a `git pull` that touches it, silently drops back to untrusted rather
than carrying approval forward to rules nobody read.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from hearth.config.errors import ConfigError
from hearth.config.loader import load_config
from hearth.config.paths import project_config_file, state_db_path
from hearth.safety.rules import Rule, compile_rules, relaxing_effect_count
from hearth.safety.trust import (
    config_fingerprint,
    is_project_trusted,
    revoke_project_trust,
    trust_project,
)
from hearth.storage.db import connect
from hearth.storage.migrate import migrate
from hearth.storage.state_repo import StateRepository

#: How much of the fingerprint to show. Enough to compare two configs by eye, not so much
#: that the output wraps.
_SHORT_DIGEST = 12


def register(app: typer.Typer) -> None:
    app.command("trust")(trust)


def trust(
    workspace: Path = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Repository to trust. Defaults to the enclosing git repository.",
    ),
    revoke: bool = typer.Option(
        False,
        "--revoke",
        help="Withdraw trust, so the project's allow rules are ignored again.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt. Only for scripts that already reviewed the rules.",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help="Print the current trust state and the rules it governs, and change nothing.",
    ),
) -> None:
    """Review a project's permission rules and decide whether to honour them."""
    from hearth.cli.app import find_workspace_root

    console = Console()
    root = (workspace or find_workspace_root()).resolve()
    config_path = project_config_file(root)

    fingerprint = config_fingerprint(config_path)
    if fingerprint is None:
        console.print(f"[dim]No project config at[/dim] {_display(config_path, root)}")
        console.print("[dim]Nothing to trust — the project supplies no permission rules.[/dim]")
        raise typer.Exit(0)

    # Load before showing anything: trusting a config that does not parse would record
    # approval for bytes Hearth cannot actually read, and the error belongs here rather
    # than at the start of the next session.
    try:
        load_config(project_root=root)
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1) from exc

    repo = _open_state(root)
    trusted = is_project_trusted(repo, config_path)
    rules = _project_rules(root)

    _render_state(console, config_path=config_path, root=root, fingerprint=fingerprint, trusted=trusted)

    if show:
        _render_rules(console, rules)
        raise typer.Exit(0)

    if revoke:
        _revoke(console, repo, config_path, trusted=trusted)
        raise typer.Exit(0)

    if trusted:
        console.print("[green]Already trusted at these bytes.[/green] Nothing to do.")
        raise typer.Exit(0)

    _render_rules(console, rules)

    relaxing = relaxing_effect_count(rules)
    if relaxing == 0:
        # Worth saying plainly: people run `hearth trust` expecting it to matter, and a
        # config with only deny and ask rules is already fully in effect.
        console.print(
            "[dim]This config has no relaxing rules, so trusting it changes nothing.[/dim]\n"
            "[dim]Its deny and ask rules already apply — those never need trust.[/dim]"
        )

    if not yes and not typer.confirm(f"Trust these {relaxing} relaxing rule(s)?", default=False):
        console.print("[yellow]Left untrusted.[/yellow] Allow rules stay ignored.")
        raise typer.Exit(0)

    recorded = trust_project(repo, config_path)
    console.print(f"[green]Trusted[/green] {_display(config_path, root)} at {recorded[:_SHORT_DIGEST]}…")
    console.print("[dim]Any edit to this file drops trust until you run `hearth trust` again.[/dim]")


# ------------------------------------------------------------------ internals


def _revoke(console: Console, repo: StateRepository, config_path: Path, *, trusted: bool) -> None:
    if not trusted:
        console.print("[dim]Not trusted, so there is nothing to revoke.[/dim]")
        return
    revoke_project_trust(repo, config_path)
    console.print("[yellow]Trust withdrawn.[/yellow] The project's allow rules are ignored again.")


def _render_state(
    console: Console, *, config_path: Path, root: Path, fingerprint: str, trusted: bool
) -> None:
    label = "[green]trusted[/green]" if trusted else "[yellow]not trusted[/yellow]"
    console.print()
    console.print(f"[bold]{_display(config_path, root)}[/bold]  ·  {label}")
    console.print(f"[dim]fingerprint {fingerprint[:_SHORT_DIGEST]}…[/dim]")
    console.print()


def _render_rules(console: Console, rules: tuple[Rule, ...]) -> None:
    """Every rule, with the relaxing ones marked.

    Deny and ask rules are shown even though trust does not govern them, because "what
    does this repository do to my permissions" is the question someone actually has when
    they run this — and answering only half of it invites the wrong conclusion.
    """
    if not rules:
        console.print("[dim]This config declares no permission rules.[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("", width=3)
    table.add_column("Effect", style="bold")
    table.add_column("Id", style="dim", overflow="fold")
    table.add_column("Matches", overflow="fold")

    for rule in rules:
        marker = "[yellow]![/yellow]" if rule.relaxing else " "
        style = {"allow": "yellow", "deny": "green", "ask": "cyan"}.get(rule.effect, "")
        # escape(): rule ids and argv globs are user-authored text, and Rich would read
        # `[project:allow:0]` or a bracketed glob as markup and silently drop it.
        table.add_row(
            marker,
            f"[{style}]{rule.effect}[/{style}]",
            escape(rule.id),
            escape(rule.conditions()),
        )

    console.print(table)
    console.print()
    console.print("[dim]! marks a rule that only takes effect once trusted.[/dim]")
    console.print()


def _project_rules(root: Path) -> tuple[Rule, ...]:
    """Compile the project's own rules, trust aside.

    Read straight from the project file rather than from the merged config, because the
    loader already drops untrusted project allow rules — which is correct for running a
    session and useless for a review screen that exists to show them.
    """
    from hearth.config.loader import read_project_permissions

    return compile_rules(read_project_permissions(root), source="project")


def _open_state(root: Path) -> StateRepository:
    path = state_db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection: sqlite3.Connection = connect(path)
    migrate(connection, database="state")
    return StateRepository(connection)


def _display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
