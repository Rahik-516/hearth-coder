"""Search tools: ``grep``, ``search_code``, ``find_symbol``, ``find_references``.

``grep`` prefers ripgrep and falls back to a pure-Python scan, so a missing binary makes
searches slower rather than impossible (docs/system-design.md §10).

The index-backed tools (``search_code``, ``find_symbol``, ``find_references``) answer the
question classes grep cannot: conceptual queries, and "who calls this". They degrade to a
clear message when the repository has not been indexed, rather than returning nothing and
letting the model conclude the code does not exist.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hearth.indexing.filters import PathFilter
from hearth.safety.paths import is_sensitive_read, is_symlink_to_outside, relative_to_workspace
from hearth.tools.base import Prepared, Risk, Tool, ToolContext
from hearth.tools.results import ErrorCode, ToolResult

MAX_MATCHES = 50
MAX_LINE_LENGTH = 300
_GREP_TIMEOUT_S = 30


class GrepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(description="Text or regular expression to find")
    path_glob: str | None = Field(default=None, description="Restrict to matching paths")
    regex: bool = Field(default=False, description="Treat the pattern as a regular expression")
    case_sensitive: bool = Field(default=False)
    max_results: int = Field(default=MAX_MATCHES, ge=1, le=500)


class GrepTool(Tool[GrepArgs]):
    name = "grep"
    description = "Search file contents for a string or regular expression."
    risk = Risk.READ
    args_model = GrepArgs
    concurrent_safe = True

    def prepare(self, args: GrepArgs, context: ToolContext) -> Prepared:
        if args.regex:
            try:
                re.compile(args.pattern)
            except re.error as exc:
                return Prepared(
                    summary=f"grep {args.pattern!r}",
                    error=ToolResult.failure(
                        ErrorCode.INVALID_ARGUMENTS,
                        f"invalid regular expression: {exc}. Set regex=false to search literally.",
                    ),
                )
        return Prepared(summary=f"grep {args.pattern!r}")

    def execute(self, args: GrepArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        matches = (
            self._ripgrep(args, context.workspace)
            if shutil.which("rg")
            else self._python_grep(args, context.workspace)
        )

        if matches is None:
            matches = self._python_grep(args, context.workspace) or []

        if not matches:
            return ToolResult.success(f"No matches for {args.pattern!r}.", pattern=args.pattern, count=0)

        shown = matches[: args.max_results]
        more = len(matches) - len(shown)
        footer = f"\n… {more} more match(es)" if more > 0 else ""

        return ToolResult.success(
            f"{len(matches)} match(es) for {args.pattern!r}:\n" + "\n".join(shown) + footer,
            pattern=args.pattern,
            count=len(matches),
        )

    def _ripgrep(self, args: GrepArgs, workspace: Path) -> list[str] | None:
        command = [
            "rg",
            "--line-number",
            "--no-heading",
            "--color=never",
            "--max-count",
            str(args.max_results),
        ]
        if not args.case_sensitive:
            command.append("--ignore-case")
        if not args.regex:
            command.append("--fixed-strings")
        if args.path_glob:
            command.extend(["--glob", args.path_glob])
        command.extend(["--", args.pattern, str(workspace)])

        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                command, capture_output=True, text=True, timeout=_GREP_TIMEOUT_S, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None

        # rg exits 1 for "no matches", which is not an error.
        if completed.returncode not in (0, 1):
            return None

        return [
            self._format_rg_line(line, workspace) for line in completed.stdout.splitlines() if line.strip()
        ]

    @staticmethod
    def _format_rg_line(line: str, workspace: Path) -> str:
        path, _, rest = line.partition(":")
        try:
            shown = relative_to_workspace(Path(path), workspace)
        except (OSError, ValueError):
            shown = path
        body = rest[:MAX_LINE_LENGTH]
        return f"{shown}:{body}"

    def _python_grep(self, args: GrepArgs, workspace: Path) -> list[str]:
        """Fallback scan. Slower than ripgrep, but always available."""
        flags = 0 if args.case_sensitive else re.IGNORECASE
        pattern = (
            re.compile(args.pattern, flags) if args.regex else re.compile(re.escape(args.pattern), flags)
        )
        path_filter = PathFilter()
        results: list[str] = []

        root = workspace.resolve(strict=False)

        for candidate in sorted(workspace.rglob(args.path_glob or "*")):
            if not candidate.is_file():
                continue
            # The jail, applied to what the walk found rather than to what the model
            # named. A symlink inside the workspace can point at `~/.ssh/id_rsa`, and
            # `rglob` reports it as an ordinary file under the workspace. Skipping links
            # whose target leaves the tree also matches ripgrep, which does not follow
            # symlinks by default — so the two backends agree on what is searchable.
            if is_symlink_to_outside(candidate, root) or is_sensitive_read(
                candidate.resolve(strict=False)
            ):
                continue
            shown = relative_to_workspace(candidate, workspace)
            if not path_filter.decide(shown).include:
                continue

            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    results.append(f"{shown}:{number}:{line.strip()[:MAX_LINE_LENGTH]}")
                    if len(results) >= args.max_results * 2:
                        return results
        return results


class SearchCodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="What to look for, in natural language or identifiers")
    limit: int = Field(default=8, ge=1, le=30)


class SearchCodeTool(Tool[SearchCodeArgs]):
    name = "search_code"
    description = "Search the codebase by meaning as well as by text."
    risk = Risk.READ
    args_model = SearchCodeArgs
    concurrent_safe = True

    def prepare(self, args: SearchCodeArgs, context: ToolContext) -> Prepared:
        if context.retrieval_engine is None:
            return Prepared(
                summary=f"search {args.query!r}",
                error=ToolResult.failure(
                    ErrorCode.NOT_FOUND,
                    "This repository is not indexed, so search_code is unavailable. Use grep instead.",
                ),
            )
        return Prepared(summary=f"search {args.query!r}")

    def execute(self, args: SearchCodeArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        # Lexical only: the tool path is synchronous and embedding a query is async
        # (docs/system-design.md §7.2). Pre-retrieval already gave the model the dense
        # results for the turn's question.
        result = context.retrieval_engine.retrieve(args.query, limit=args.limit, mode="lexical")

        if not result.results:
            return ToolResult.success(f"No results for {args.query!r}.", count=0)

        lines = [f"{found.citation}  ({found.symbol_path or found.kind})" for found in result.results]
        return ToolResult.success(
            f"{len(lines)} result(s) for {args.query!r}:\n" + "\n".join(lines),
            count=len(lines),
        )


class FindSymbolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Symbol name to locate")
    kind: str | None = Field(default=None, description="Restrict to a kind, e.g. 'class'")


class FindSymbolTool(Tool[FindSymbolArgs]):
    name = "find_symbol"
    description = "Find where a class, function or constant is defined."
    risk = Risk.READ
    args_model = FindSymbolArgs
    concurrent_safe = True

    def prepare(self, args: FindSymbolArgs, context: ToolContext) -> Prepared:
        if context.index_connection is None:
            return Prepared(
                summary=f"find_symbol {args.name}",
                error=_needs_index("find_symbol"),
            )
        return Prepared(summary=f"find_symbol {args.name}")

    def execute(self, args: FindSymbolArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        sql = (
            "SELECT s.name, s.kind, s.signature, s.start_line, s.end_line, f.path "
            "FROM symbols s JOIN files f ON f.id = s.file_id "
            "WHERE s.name = ? COLLATE NOCASE"
        )
        params: list[object] = [args.name]
        if args.kind:
            sql += " AND s.kind = ?"
            params.append(args.kind)
        sql += " ORDER BY f.path, s.start_line LIMIT 30"

        rows = context.index_connection.execute(sql, tuple(params)).fetchall()
        if not rows:
            return ToolResult.success(
                f"No symbol named {args.name!r} is indexed. "
                f"It may be defined in an unindexed language, or spelled differently.",
                count=0,
            )

        lines = [
            f"{row['path']}:{row['start_line']}-{row['end_line']}  "
            f"{row['kind']} {row['name']}" + (f"  —  {row['signature']}" if row["signature"] else "")
            for row in rows
        ]
        return ToolResult.success(
            f"{len(lines)} definition(s) of {args.name!r}:\n" + "\n".join(lines),
            count=len(lines),
        )


class FindReferencesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Symbol name to find uses of")
    limit: int = Field(default=30, ge=1, le=200)


class FindReferencesTool(Tool[FindReferencesArgs]):
    name = "find_references"
    description = "Find where a symbol is used or called."
    risk = Risk.READ
    args_model = FindReferencesArgs
    concurrent_safe = True

    def prepare(self, args: FindReferencesArgs, context: ToolContext) -> Prepared:
        if context.index_connection is None:
            return Prepared(
                summary=f"find_references {args.name}",
                error=_needs_index("find_references"),
            )
        return Prepared(summary=f"find_references {args.name}")

    def execute(self, args: FindReferencesArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        rows = context.index_connection.execute(
            "SELECT r.name, r.line, r.kind, f.path FROM refs r "
            "JOIN files f ON f.id = r.file_id "
            "WHERE r.name = ? COLLATE NOCASE ORDER BY f.path, r.line LIMIT ?",
            (args.name, args.limit),
        ).fetchall()

        if not rows:
            return ToolResult.success(f"No references to {args.name!r} are indexed.", count=0)

        lines = [f"{row['path']}:{row['line']}  ({row['kind']})" for row in rows]

        # The reference graph is name-based, so two unrelated methods with the same name
        # are indistinguishable here. Saying so stops the model reporting a count as fact
        # (docs/system-design.md §6.6).
        caveat = (
            "\n\nNote: references are matched by name, so results may include unrelated "
            "symbols that share it."
        )
        return ToolResult.success(
            f"{len(lines)} reference(s) to {args.name!r}:\n" + "\n".join(lines) + caveat,
            count=len(lines),
        )


def _needs_index(tool: str) -> ToolResult:
    return ToolResult.failure(
        ErrorCode.NOT_FOUND,
        f"This repository is not indexed, so {tool} is unavailable. Use grep to search the text instead.",
    )
