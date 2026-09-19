"""Read-only filesystem tools: ``read_file``, ``list_dir``, ``find_files``.

Every path goes through ``safety.paths.resolve_in_workspace`` — these tools never call
``open()`` on a model-supplied string directly, and a CI grep guard enforces that
(docs/project-structure.md §4).

``read_file`` also registers a content hash per file. That registry is what makes
read-before-write possible in M5: an edit whose file changed since it was read is refused
rather than clobbering someone's work (docs/system-design.md §10.1).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hearth.indexing.filters import PathFilter
from hearth.safety.paths import relative_to_workspace, resolve_in_workspace
from hearth.tools.base import Prepared, Risk, Tool, ToolContext
from hearth.tools.results import ErrorCode, ToolResult
from hearth.util.hashing import content_hash
from hearth.util.text import decode_text, is_probably_binary

#: Default lines per read. The model can page with offset/limit
#: (docs/system-design.md §9.3).
DEFAULT_READ_LIMIT = 400
MAX_READ_LIMIT = 2_000
MAX_FILE_BYTES = 2_000_000
MAX_LISTING_ENTRIES = 300


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Workspace-relative path of the file to read")
    offset: int = Field(default=1, ge=1, description="First line to read, 1-based")
    limit: int = Field(default=DEFAULT_READ_LIMIT, ge=1, le=MAX_READ_LIMIT, description="Maximum lines")


class ReadFileTool(Tool[ReadFileArgs]):
    name = "read_file"
    description = "Read a file's contents, with optional line offset and limit."
    risk = Risk.READ
    args_model = ReadFileArgs
    concurrent_safe = True

    def prepare(self, args: ReadFileArgs, context: ToolContext) -> Prepared:
        resolved = resolve_in_workspace(context.workspace, args.path, for_write=False)
        relative = relative_to_workspace(resolved, context.workspace)

        # Secret files are refused here as well as at index time: the index is not the
        # only way a path reaches this tool (docs/safety-and-tool-use.md §10).
        if PathFilter().is_secret(relative):
            return Prepared(
                summary=f"read {relative}",
                error=ToolResult.failure(
                    ErrorCode.PATH_REFUSED,
                    f"{relative} matches a secret-file pattern and is never readable.",
                ),
            )

        if not resolved.exists():
            return Prepared(
                summary=f"read {relative}",
                error=ToolResult.failure(
                    ErrorCode.NOT_FOUND,
                    f"{relative} does not exist. Use find_files to locate it.",
                ),
            )
        if resolved.is_dir():
            return Prepared(
                summary=f"read {relative}",
                error=ToolResult.failure(
                    ErrorCode.INVALID_ARGUMENTS,
                    f"{relative} is a directory. Use list_dir instead.",
                ),
            )

        return Prepared(
            summary=f"read {relative} lines {args.offset}-{args.offset + args.limit - 1}",
            payload={"resolved": resolved, "relative": relative},
        )

    def execute(self, args: ReadFileArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        resolved: Path = prepared.payload["resolved"]
        relative: str = prepared.payload["relative"]

        try:
            raw = resolved.read_bytes()
        except OSError as exc:
            return ToolResult.failure(ErrorCode.NOT_FOUND, f"could not read {relative}: {exc}")

        if len(raw) > MAX_FILE_BYTES:
            return ToolResult.failure(
                ErrorCode.TOO_LARGE,
                f"{relative} is {len(raw)} bytes. Read a range with offset and limit.",
            )
        if is_probably_binary(raw):
            return ToolResult.failure(
                ErrorCode.INVALID_ARGUMENTS, f"{relative} looks binary and cannot be read as text."
            )

        text = decode_text(raw)
        # Registered even for a partial read: the hash covers the whole file, which is what
        # a later edit needs to compare against.
        context.record_read(relative, content_hash(raw))

        lines = text.splitlines()
        start = args.offset - 1
        window = lines[start : start + args.limit]

        if not window and lines:
            return ToolResult.failure(
                ErrorCode.INVALID_ARGUMENTS,
                f"{relative} has {len(lines)} lines; offset {args.offset} is past the end.",
            )

        numbered = "\n".join(f"{number:>6}\t{line}" for number, line in enumerate(window, start=args.offset))
        shown_to = start + len(window)
        footer = (
            f"\n\n… {len(lines) - shown_to} more line(s); re-read with offset={shown_to + 1}"
            if shown_to < len(lines)
            else ""
        )

        return ToolResult.success(
            f"{relative} ({len(lines)} lines)\n{numbered}{footer}",
            path=relative,
            total_lines=len(lines),
            shown_lines=len(window),
        )


class ListDirArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(default=".", description="Directory to list, workspace-relative")
    depth: int = Field(default=2, ge=1, le=5, description="How many levels to descend")


class ListDirTool(Tool[ListDirArgs]):
    name = "list_dir"
    description = "List the files and directories under a path."
    risk = Risk.READ
    args_model = ListDirArgs
    concurrent_safe = True

    def prepare(self, args: ListDirArgs, context: ToolContext) -> Prepared:
        resolved = resolve_in_workspace(context.workspace, args.path, for_write=False)
        relative = relative_to_workspace(resolved, context.workspace)

        if not resolved.is_dir():
            return Prepared(
                summary=f"list {relative}",
                error=ToolResult.failure(ErrorCode.NOT_FOUND, f"{relative} is not a directory."),
            )
        return Prepared(summary=f"list {relative}", payload={"resolved": resolved, "relative": relative})

    def execute(self, args: ListDirArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        resolved: Path = prepared.payload["resolved"]
        relative: str = prepared.payload["relative"]
        path_filter = PathFilter()

        entries: list[str] = []
        truncated = False

        for current, dirnames, filenames in resolved.walk():
            level = len(current.relative_to(resolved).parts)
            if level >= args.depth:
                dirnames.clear()

            # Prune as we go rather than filtering afterwards, so a node_modules never
            # gets walked in the first place.
            dirnames[:] = sorted(d for d in dirnames if path_filter.decide(f"{d}/x").include or level == 0)

            for name in sorted(filenames):
                candidate = current / name
                shown = relative_to_workspace(candidate, context.workspace)
                if not path_filter.decide(shown).include:
                    continue
                entries.append(shown)
                if len(entries) >= MAX_LISTING_ENTRIES:
                    truncated = True
                    break
            if truncated:
                break

        if not entries:
            return ToolResult.success(f"{relative} contains no indexable files.", path=relative)

        footer = f"\n… truncated at {MAX_LISTING_ENTRIES} entries" if truncated else ""
        return ToolResult.success(
            f"{relative} ({len(entries)} entries)\n" + "\n".join(entries) + footer,
            path=relative,
            count=len(entries),
        )


class FindFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    glob: str = Field(description="Glob pattern, e.g. 'src/**/*.py'")
    limit: int = Field(default=100, ge=1, le=1_000)


class FindFilesTool(Tool[FindFilesArgs]):
    name = "find_files"
    description = "Find files by glob pattern."
    risk = Risk.READ
    args_model = FindFilesArgs
    concurrent_safe = True

    def prepare(self, args: FindFilesArgs, context: ToolContext) -> Prepared:
        if args.glob.startswith("/") or ".." in args.glob:
            return Prepared(
                summary=f"find {args.glob}",
                error=ToolResult.failure(
                    ErrorCode.PATH_REFUSED,
                    "Glob patterns are workspace-relative and may not contain '..'.",
                ),
            )
        return Prepared(summary=f"find {args.glob}")

    def execute(self, args: FindFilesArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        path_filter = PathFilter()
        matches: list[str] = []

        try:
            for candidate in sorted(context.workspace.glob(args.glob)):
                if not candidate.is_file():
                    continue
                shown = relative_to_workspace(candidate, context.workspace)
                if not path_filter.decide(shown).include:
                    continue
                matches.append(shown)
                if len(matches) >= args.limit:
                    break
        except (OSError, ValueError) as exc:
            return ToolResult.failure(ErrorCode.INVALID_ARGUMENTS, f"bad glob: {exc}")

        if not matches:
            return ToolResult.success(f"No files match {args.glob!r}.", pattern=args.glob, count=0)

        return ToolResult.success(
            f"{len(matches)} file(s) matching {args.glob!r}:\n" + "\n".join(matches),
            pattern=args.glob,
            count=len(matches),
        )
