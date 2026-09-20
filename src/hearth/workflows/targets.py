"""Turning what the user typed into a file and, if they named one, a symbol.

`/test compute_tax` and `/refactor compute_tax ...` both start from a word the user
typed, and both have to become a concrete place in the tree before any model is involved.
That is deterministic work — the index knows where `compute_tax` is defined — and doing it
here means the model is handed a location rather than asked to find one, which is a job a
4B model does badly and a database does exactly.

**Ambiguity is an error, not a choice.** A name defined in three files does not resolve to
the first one. The workflow asks the user to say which (`path::name`), because a test or a
refactor written against the wrong `parse()` is worse than a question: it looks like
success, and the user finds out at review.

**The jail applies.** A target that resolves outside the workspace, or into `.git/`, is
refused here rather than discovered later by a tool — the same reason every tool resolves
its paths before touching them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hearth.safety.paths import is_inside_workspace, is_protected_write, relative_to_workspace
from hearth.storage.index_repo import IndexRepository

#: How many candidate locations an ambiguity message lists. Enough to choose from; past
#: this the name is too common for a list to help and the user needs `path::name`.
MAX_CANDIDATES_SHOWN = 6


@dataclass(frozen=True)
class ResolvedTarget:
    """Where a workflow's subject lives."""

    #: Workspace-relative POSIX path of the file.
    path: str
    #: The symbol, when the user named one rather than a whole file.
    symbol: str | None = None
    kind: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    signature: str | None = None

    def describe(self) -> str:
        if self.symbol is None:
            return self.path
        span = f":{self.start_line}-{self.end_line}" if self.start_line else ""
        return f"{self.symbol} ({self.kind or 'symbol'}) in {self.path}{span}"


@dataclass(frozen=True)
class TargetError:
    """Why a target could not be resolved, worded for the user."""

    message: str


def resolve_target(
    root: Path, target: str, *, repository: IndexRepository | None
) -> ResolvedTarget | TargetError:
    """Resolve a file path, a bare symbol name, or ``path::symbol``.

    Files win over symbols: if ``target`` names an existing file, that is what was meant,
    even when a symbol of the same name also exists. A user who typed ``models`` and has a
    ``models.py`` almost certainly meant the file, and one who meant the symbol can write
    ``path::models``.
    """
    target = target.strip()
    if not target:
        return TargetError("name a file or a symbol.")

    path_part, separator, symbol_part = target.partition("::")
    if separator and symbol_part:
        return _resolve_in_file(root, path_part, symbol_part, repository)

    as_file = _resolve_file(root, target)
    if isinstance(as_file, ResolvedTarget):
        return as_file
    # A path-shaped target that failed the jail is refused as such, never retried as a
    # symbol: `../../etc/passwd` is not a symbol name and looking it up would be a guess.
    if _looks_like_path(target):
        return as_file

    return _resolve_symbol(target, repository)


def _resolve_file(root: Path, target: str) -> ResolvedTarget | TargetError:
    normalised = target.replace("\\", "/")
    candidate = (root / normalised).resolve(strict=False)

    if not is_inside_workspace(candidate, root):
        return TargetError(f"{target} is outside the workspace.")

    relative = relative_to_workspace(candidate, root)
    if is_protected_write(Path(relative)):
        return TargetError(f"{relative} is a protected path.")
    if not candidate.is_file():
        return TargetError(f"{relative} is not a file in this workspace.")
    return ResolvedTarget(path=relative)


def _resolve_symbol(name: str, repository: IndexRepository | None) -> ResolvedTarget | TargetError:
    if repository is None:
        return TargetError(
            f"{name} is not a file, and there is no index to look it up as a symbol. "
            "Run `hearth index`, or give a file path."
        )

    matches = repository.find_symbol(name)
    if not matches:
        return TargetError(f"no file or symbol named {name!r}.")
    if len(matches) > 1:
        shown = matches[:MAX_CANDIDATES_SHOWN]
        listing = "\n".join(
            f"  {m['path']}::{m['name']}  ({m['kind']}, line {m['start_line']})" for m in shown
        )
        more = len(matches) - len(shown)
        tail = f"\n  … and {more} more" if more > 0 else ""
        return TargetError(
            f"{name!r} is defined in {len(matches)} places; say which:\n{listing}{tail}"
        )
    return _from_row(matches[0])


def _resolve_in_file(
    root: Path, path: str, symbol: str, repository: IndexRepository | None
) -> ResolvedTarget | TargetError:
    located = _resolve_file(root, path)
    if isinstance(located, TargetError):
        return located
    if repository is None:
        return TargetError("there is no index to look the symbol up in. Run `hearth index`.")

    inside = [m for m in repository.find_symbol(symbol) if m["path"] == located.path]
    if not inside:
        return TargetError(f"{symbol!r} is not defined in {located.path}.")
    if len(inside) > 1:
        lines = ", ".join(str(m["start_line"]) for m in inside)
        return TargetError(f"{symbol!r} is defined more than once in {located.path} (lines {lines}).")
    return _from_row(inside[0])


def _from_row(row: dict[str, object]) -> ResolvedTarget:
    return ResolvedTarget(
        path=str(row["path"]),
        symbol=str(row["name"]),
        kind=str(row["kind"]),
        start_line=int(row["start_line"]),  # type: ignore[call-overload]
        end_line=int(row["end_line"]),  # type: ignore[call-overload]
        signature=str(row["signature"]) if row.get("signature") else None,
    )


def _looks_like_path(target: str) -> bool:
    return "/" in target or "\\" in target or target.startswith(".") or "." in Path(target).name
