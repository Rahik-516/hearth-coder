"""``read_output``: paging through a command's full output.

``run_command`` and ``run_tests`` show the model a head/tail window and store the whole
capture in a blob (docs/safety-and-tool-use.md §8.2). Without a way to read that blob the
``output_id`` in every result is a promise Hearth does not keep: a 400-line test failure
whose middle holds the actual traceback is unreachable, and the model's only recourse is
to re-run the command and hope for a smaller failure.

Paging is bounded by line, not by byte, because the thing being paged is program output
and a byte offset lands mid-line. The model asks for a range; it gets those lines with
their numbers, so a second call can continue where the first stopped.

READ risk: reading back output the session already produced and already showed a window
of reveals nothing new, so it never asks for approval.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hearth.storage.blobs import InvalidDigestError
from hearth.tools.base import Prepared, Risk, Tool, ToolContext
from hearth.tools.results import ErrorCode, ToolResult

#: Lines per page. Large enough that a stack trace arrives whole, small enough that two
#: pages do not fill the context the window was truncated to protect.
DEFAULT_LIMIT = 200

#: Ceiling on a single request, so `limit=100000` cannot undo the truncation.
MAX_LIMIT = 500


class ReadOutputArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_id: str = Field(description="The output_id from a previous run_command or run_tests")
    offset: int = Field(default=0, ge=0, description="First line to return, counting from 0")
    limit: int = Field(
        default=DEFAULT_LIMIT,
        ge=1,
        le=MAX_LIMIT,
        description=f"How many lines to return (max {MAX_LIMIT})",
    )


class ReadOutputTool(Tool[ReadOutputArgs]):
    name = "read_output"
    description = (
        "Read more of a command's output by its output_id, when the result was truncated."
    )
    risk = Risk.READ
    args_model = ReadOutputArgs
    concurrent_safe = True

    def prepare(self, args: ReadOutputArgs, context: ToolContext) -> Prepared:
        if context.blobs is None:
            return _refused(
                ErrorCode.NOT_FOUND,
                "no output store is available in this session; re-run the command instead",
            )
        return Prepared(summary=f"read output {args.output_id[:12]}")

    def execute(self, args: ReadOutputArgs, context: ToolContext, prepared: Prepared) -> ToolResult:
        try:
            raw = context.blobs.get(args.output_id)
        except InvalidDigestError:
            # Deliberately not echoed back: the value is model-supplied, and quoting an
            # arbitrary string into the transcript is how a malformed id becomes a
            # confusing second problem.
            return ToolResult.failure(
                ErrorCode.INVALID_ARGUMENTS,
                "output_id is not valid. Use the output_id exactly as the tool reported it.",
            )
        except OSError:
            return ToolResult.failure(
                ErrorCode.NOT_FOUND,
                f"no stored output for {args.output_id[:12]}. It may be from an older session; "
                "re-run the command to capture it again.",
            )

        lines = raw.decode("utf-8", errors="replace").splitlines()
        total = len(lines)

        if args.offset >= total:
            return ToolResult.success(
                f"That output has {total} line(s); offset {args.offset} is past the end.",
                total_lines=total,
                returned=0,
            )

        end = min(args.offset + args.limit, total)
        body = "\n".join(
            f"{number}\t{line}" for number, line in enumerate(lines[args.offset : end], start=args.offset + 1)
        )

        remaining = total - end
        footer = (
            f"\n\n[{remaining} more line(s); read from offset {end}]" if remaining else ""
        )

        result = ToolResult.success(
            f"lines {args.offset + 1}-{end} of {total}\n\n{body}{footer}",
            total_lines=total,
            returned=end - args.offset,
            remaining=remaining,
        )
        result.output_id = args.output_id
        return result


def _refused(code: ErrorCode, message: str) -> Prepared:
    return Prepared(summary="read_output refused", error=ToolResult.failure(code, message))
