"""The tool registry.

Decides which tools exist for a given mode and model profile
(docs/system-design.md §5.2). Two rules shape it:

* **Mode decides which tools exist**, separately from permissions. `chat` gets read-only
  tools; `agent` gets writes and exec. A tool that is not exposed cannot be called by
  mistake, which is a stronger guarantee than one that would be denied.
* **Weak models get fewer tools.** A profile with low tool reliability is given no tools at
  all rather than a long list it will call incorrectly — an unusable answer beats a
  confidently wrong tool call (docs/system-design.md §1.1).

The registered set is fixed for a session, because tool schemas are part of the cached
prompt prefix and changing them mid-session discards the KV cache
(docs/system-design.md §9.2, rule 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hearth.tools.base import Risk, Tool

#: Tools exposed per mode, in the order they appear in the schema.
_MODE_TOOLS: dict[str, tuple[str, ...]] = {
    "chat": (
        "read_file",
        "grep",
        "find_files",
        "list_dir",
        "find_symbol",
        "find_references",
    ),
    "plan": (
        "read_file",
        "grep",
        "find_files",
        "list_dir",
        "find_symbol",
        "find_references",
        "search_code",
        "git_status",
        "git_diff",
        "git_log",
    ),
    # Writes appear here and nowhere else. A tool that is not exposed cannot be called by
    # mistake, which is a stronger guarantee than one that would be denied — and it is why
    # `edit_file` is absent from chat and plan rather than merely refused there.
    #
    # Reads come first deliberately: the schema order is part of the cached prompt prefix,
    # and the model should meet the tools that let it look before the ones that change
    # things.
    "agent": (
        "read_file",
        "grep",
        "find_files",
        "list_dir",
        "find_symbol",
        "find_references",
        "search_code",
        "git_status",
        "git_diff",
        "git_log",
        "read_output",
        "todo_write",
        "edit_file",
        "write_file",
        "run_command",
        "run_tests",
        "git_add",
        "git_commit",
        "git_branch_create",
        "git_switch",
    ),
}

#: On a 12K window chat mode exposes at most this many tools, with one-line descriptions
#: (docs/system-design.md §9.1).
MAX_CHAT_TOOLS = 6


@dataclass(frozen=True)
class ToolAvailability:
    """Which tools a session gets, and why."""

    tools: tuple[Tool[Any], ...]
    reason: str

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)

    @property
    def schemas(self) -> list[dict[str, object]]:
        return [tool.schema() for tool in self.tools]


class ToolRegistry:
    """Holds tool instances and selects them per mode and profile."""

    def __init__(self, tools: list[Tool[Any]] | None = None) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any] | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def for_mode(
        self,
        mode: str,
        *,
        tool_reliability: str = "high",
        max_tools: int | None = None,
    ) -> ToolAvailability:
        """Select the tools a session should expose.

        Args:
            mode: ``chat``, ``plan`` or ``agent``.
            tool_reliability: From the model profile. ``low`` means no tools.
            max_tools: Cap, for small context windows.
        """
        if tool_reliability == "low":
            # A model that invents arguments does more harm with tools than without: it
            # burns steps, fills the context with errors, and ends up answering from
            # pre-retrieved context anyway (docs/system-design.md §5.2).
            return ToolAvailability(
                tools=(),
                reason="model profile reports low tool reliability",
            )

        wanted = _MODE_TOOLS.get(mode, ())
        selected: list[Tool[Any]] = [self._tools[name] for name in wanted if name in self._tools]

        limit = max_tools if max_tools is not None else (MAX_CHAT_TOOLS if mode == "chat" else None)
        if limit is not None and len(selected) > limit:
            selected = _truncate(selected, limit)
            return ToolAvailability(
                tools=tuple(selected),
                reason=f"{mode} mode, capped at {limit} tools for the context budget",
            )

        return ToolAvailability(tools=tuple(selected), reason=f"{mode} mode")

    def read_only_names(self) -> list[str]:
        return sorted(name for name, tool in self._tools.items() if tool.risk is Risk.READ)


def _truncate(selected: list[Tool[Any]], limit: int) -> list[Tool[Any]]:
    """Apply a tool-count cap without dropping the tools that define the mode.

    Reads are listed first so the model meets them first, which means a naive
    ``selected[:limit]`` silently removes ``edit_file`` and ``write_file`` — turning an
    agent session into a read-only one that still calls itself agent mode. Found the hard
    way: an ``agent`` availability capped at 8 exposed ten read tools and no writes.

    So the cap drops reads from the end and keeps everything else. Order is otherwise
    preserved, because the schema list is part of the cached prompt prefix.
    """
    reads = [tool for tool in selected if tool.risk is Risk.READ]
    others = [tool for tool in selected if tool.risk is not Risk.READ]

    # A floor of reads is reserved before anything else is fitted. Agent mode now carries
    # ten non-read tools, so filling with those first leaves zero reads at any realistic
    # cap — handing the model `edit_file` with no `read_file`, which cannot satisfy
    # read-before-write and so cannot edit anything at all.
    floor = min(len(reads), max(1, limit // 3))
    kept = [*reads[:floor], *others[: limit - floor]]

    # Spend anything left over on more reads; they are the cheapest tools to carry.
    if len(kept) < limit:
        kept.extend(reads[floor : floor + (limit - len(kept))])

    chosen = {id(tool) for tool in kept}
    return [tool for tool in selected if id(tool) in chosen]


def build_default_registry(
    *,
    test_command: str | None = None,
    test_command_source: str | None = None,
) -> ToolRegistry:
    """Every tool Hearth currently has.

    Registration is not exposure: `for_mode` decides what a session actually sees, so
    `edit_file` being registered here does not make it reachable from chat mode.

    Imported lazily by callers so `hearth --version` does not pay for it.

    Args:
        test_command: The project's configured test command, if it set one. It is a
            *value* here, not a permission — it is classified and approved exactly like a
            command the model composed (docs/safety-and-tool-use.md §5.6).
        test_command_source: Where that command came from, for the PROJECT-CONFIG badge.
    """
    from hearth.tools.git_read import GitDiffTool, GitLogTool, GitStatusTool
    from hearth.tools.git_write import (
        GitAddTool,
        GitBranchCreateTool,
        GitCommitTool,
        GitSwitchTool,
    )
    from hearth.tools.meta import TodoWriteTool
    from hearth.tools.output import ReadOutputTool
    from hearth.tools.read_fs import FindFilesTool, ListDirTool, ReadFileTool
    from hearth.tools.search import (
        FindReferencesTool,
        FindSymbolTool,
        GrepTool,
        SearchCodeTool,
    )
    from hearth.tools.shell import RunCommandTool
    from hearth.tools.tests import RunTestsTool
    from hearth.tools.write_fs import EditFileTool, WriteFileTool

    return ToolRegistry(
        [
            ReadFileTool(),
            ListDirTool(),
            FindFilesTool(),
            GrepTool(),
            SearchCodeTool(),
            FindSymbolTool(),
            FindReferencesTool(),
            GitStatusTool(),
            GitDiffTool(),
            GitLogTool(),
            EditFileTool(),
            WriteFileTool(),
            RunCommandTool(),
            RunTestsTool(test_command=test_command, source=test_command_source),
            GitAddTool(),
            GitCommitTool(),
            GitBranchCreateTool(),
            GitSwitchTool(),
            TodoWriteTool(),
            ReadOutputTool(),
        ]
    )
