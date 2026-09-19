"""``todo_write`` and the agent-mode prompt assembly.

The prompt half is snapshot-guarded per CLAUDE.md rule 6: prompts are reviewed like code,
and a change to one has to be a deliberate act with an updated snapshot rather than
something that drifts in alongside an unrelated edit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hearth.prompts import available, load, system_prompt_for
from hearth.tools.base import ToolContext
from hearth.tools.meta import TodoWriteTool, render_todos
from hearth.tools.results import ErrorCode


@pytest.fixture
def context(tmp_path: Path) -> ToolContext:
    root = tmp_path / "repo"
    root.mkdir()
    return ToolContext(workspace=root)


def run_todo(context: ToolContext, todos: list[dict[str, str]]):
    """``todo_write`` is META: no filesystem, no subprocess, nothing to contain."""
    tool = TodoWriteTool()
    args = tool.args_model.model_validate({"todos": todos})
    prepared = tool.prepare(args, context)  # type: ignore[arg-type]
    if prepared.failed:
        return prepared, prepared.error
    return prepared, tool.execute(args, context, prepared)  # type: ignore[arg-type]


def test_the_list_renders_with_a_mark_per_status(context: ToolContext) -> None:
    _, result = run_todo(
        context,
        [
            {"content": "Read the invoice service", "status": "completed"},
            {"content": "Fix the rounding", "status": "in_progress"},
            {"content": "Run the tests", "status": "pending"},
        ],
    )

    assert result.ok
    assert "[x] Read the invoice service" in result.content
    assert "[~] Fix the rounding" in result.content
    assert "[ ] Run the tests" in result.content
    assert result.metadata["completed"] == 1


def test_two_items_in_progress_is_refused(context: ToolContext) -> None:
    """Two active items means the model has lost track of what it is doing."""
    _, result = run_todo(
        context,
        [
            {"content": "Fix the rounding", "status": "in_progress"},
            {"content": "Run the tests", "status": "in_progress"},
        ],
    )

    assert not result.ok
    assert result.error is ErrorCode.INVALID_ARGUMENTS


def test_an_empty_list_is_rejected_by_validation(context: ToolContext) -> None:
    tool = TodoWriteTool()

    parsed = tool.validate({"todos": []})

    assert not isinstance(parsed, tool.args_model)


def test_the_display_is_the_bare_list(context: ToolContext) -> None:
    """The panel shows the list; the model gets a progress line above it."""
    _, result = run_todo(context, [{"content": "Fix the rounding", "status": "pending"}])

    assert result.display == "[ ] Fix the rounding"
    assert result.content.startswith("0/1 complete.")


def test_render_is_stable_for_the_cli(context: ToolContext) -> None:
    from hearth.tools.meta import TodoItem

    rendered = render_todos([TodoItem(content="One", status="completed")])

    assert rendered == "[x] One"


# ------------------------------------------------------------------- prompts


def test_agent_mode_gets_the_tool_protocol() -> None:
    prompt = system_prompt_for("agent")

    assert "Read a file before you edit it" in prompt
    assert "Run the tests you can run" in prompt


def test_chat_mode_does_not_get_the_tool_protocol() -> None:
    """A model told it can edit files, in a mode where it cannot, will try."""
    prompt = system_prompt_for("chat")

    assert "edit_file" not in prompt
    assert "no tools" in prompt


def test_the_prompt_order_is_core_then_tools_then_mode() -> None:
    """This string is the cached prefix; its order is load-bearing (§9.2)."""
    prompt = system_prompt_for("agent")

    assert prompt.index("You are Hearth") < prompt.index("## Tools")
    assert prompt.index("## Tools") < prompt.index("## This session: agent mode")


def test_the_agent_prompt_files_are_packaged() -> None:
    assert {"system_core", "system_tools", "mode_agent", "mode_chat"} <= set(available())


def test_agent_prompt_snapshot(snapshot) -> None:
    """Rule 6: a prompt change is deliberate, with the snapshot updated on purpose."""
    assert load("mode_agent") == snapshot


def test_tool_protocol_snapshot(snapshot) -> None:
    assert load("system_tools") == snapshot
