"""``todo_write``: the agent's visible plan for a multi-step task.

The tool holds no state. Each call carries the whole list and replaces the previous one,
which is what keeps the model's plan and the panel the user is reading from drifting apart
— an incremental "mark item 3 done" protocol needs both sides to agree on what item 3 was,
and a 4B model reliably does not.

Its value is mostly not for the model. A long agent turn is otherwise opaque: the user sees
tool calls scroll past with no way to tell whether the run is progressing or circling. The
list makes the intended shape of the work visible early enough to interrupt, which is the
point at which interrupting is cheap.

``META`` risk: it changes nothing outside the session, so it never asks for approval
(docs/safety-and-tool-use.md §2.1). It is the only tool in agent mode that is free.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hearth.tools.base import Prepared, Risk, Tool, ToolContext
from hearth.tools.results import ErrorCode, ToolResult

Status = Literal["pending", "in_progress", "completed"]

_MARKS: dict[str, str] = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}

#: More than this and the list has stopped being a plan and become a transcript.
MAX_TODOS = 20


class TodoItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, description="What needs doing, as an imperative phrase")
    status: Status = Field(default="pending", description="pending, in_progress or completed")


class TodoWriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    todos: list[TodoItem] = Field(
        min_length=1,
        max_length=MAX_TODOS,
        description="The complete list, replacing any previous one",
    )


class TodoWriteTool(Tool[TodoWriteArgs]):
    name = "todo_write"
    description = (
        "Record the plan for a multi-step task. Send the whole list every time; it "
        "replaces the previous one. Mark exactly one item in_progress."
    )
    risk = Risk.META
    args_model = TodoWriteArgs

    def prepare(self, args: TodoWriteArgs, context: ToolContext) -> Prepared:
        active = [item for item in args.todos if item.status == "in_progress"]
        if len(active) > 1:
            # Refused rather than tolerated: two active items means the model has stopped
            # tracking which thing it is doing, and the list is about to become fiction.
            return Prepared(
                summary="todo_write refused",
                error=ToolResult.failure(
                    ErrorCode.INVALID_ARGUMENTS,
                    f"{len(active)} items are in_progress; mark exactly one.",
                ),
            )

        return Prepared(summary=f"update {len(args.todos)} todo(s)", payload={})

    def execute(self, args: TodoWriteArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        rendered = render_todos(args.todos)
        done = sum(1 for item in args.todos if item.status == "completed")
        return ToolResult.success(
            f"{done}/{len(args.todos)} complete.\n{rendered}",
            display=rendered,
            total=len(args.todos),
            completed=done,
        )


def render_todos(todos: list[TodoItem]) -> str:
    """The list as both the model and the panel see it."""
    return "\n".join(f"{_MARKS[item.status]} {item.content}" for item in todos)
