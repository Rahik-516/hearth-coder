"""The Tool contract.

Every tool is a class with a Pydantic ``Args`` model, a risk level, a ``prepare()`` that
computes what *would* happen, and an ``execute()`` that does it
(docs/system-design.md §5.8).

**Separating prepare from execute is what makes approval honest.** The preview the user
approves is produced by ``prepare()``; ``execute()`` re-verifies its preconditions and
aborts if anything moved. Without that split, the diff shown and the bytes written are two
independent computations that merely tend to agree.

``prepare()` must not mutate anything. That invariant is why an approval prompt can be
generated for a call that is then rejected, with nothing to undo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from hearth.safety.policy import PolicyFacts
from hearth.safety.risk import Risk
from hearth.tools.results import ToolResult, invalid_arguments

# Re-exported: ``Risk`` is defined in ``safety`` because the policy engine switches on it
# and the ``policy-pure`` contract forbids ``safety.policy`` from importing ``tools``.
# Tools still import it from here, which is where it reads naturally.
__all__ = ["Prepared", "Risk", "Tool", "ToolContext"]


@dataclass
class ToolContext:
    """What a tool is allowed to know about the session it runs in.

    Deliberately narrow: a tool gets the workspace, the read registry and the services it
    needs — never the session, the provider, or the bus. Tools are deterministic, and
    nothing here lets one call a model (enforced by the `tools-no-llm` contract).
    """

    workspace: Path
    #: path -> content hash at last read. Backs read-before-write.
    read_hashes: dict[str, str] = field(default_factory=dict)
    #: Open index connection, for the search tools.
    index_connection: Any = None
    #: Retrieval engine, for `search_code`.
    retrieval_engine: Any = None
    #: Session-bound checkpoint writer, for the write tools. None disables checkpointing,
    #: which is why the write tools refuse to run without it — an unrevertable write is
    #: not a degraded write, it is a different and worse operation.
    checkpoints: Any = None
    #: Synchronous single-file reindex, called with a workspace-relative path after a
    #: write lands. None skips it.
    reindex: Any = None
    #: The current step within the turn. Set by the gateway before each call, so a
    #: checkpoint can be attributed to the step `/undo` will name.
    step: int = 0
    max_output_tokens: int = 4_000

    def record_read(self, path: str, content_hash: str) -> None:
        self.read_hashes[path] = content_hash

    def hash_at_last_read(self, path: str) -> str | None:
        return self.read_hashes.get(path)


@dataclass
class Prepared:
    """The outcome of ``prepare()``: what would happen, and how to describe it.

    ``payload`` carries whatever ``execute()`` needs so it does not recompute — and so it
    cannot silently compute something different from what was approved.
    """

    summary: str
    preview: str = ""
    badges: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    #: Set when prepare already knows the call cannot succeed.
    error: ToolResult | None = None
    #: What the policy engine needs to judge this call: the resolved path, the parsed
    #: argv, the classification flags.
    #:
    #: The tool states these rather than letting the gateway re-derive them, because the
    #: preview the user approves was computed from *this* resolution. A second resolution
    #: of the same string in the gateway would be a different computation that merely
    #: tends to agree — the exact failure the prepare/execute split exists to prevent.
    facts: PolicyFacts = field(default_factory=PolicyFacts)

    @property
    def failed(self) -> bool:
        return self.error is not None


class Tool[ArgsT: BaseModel](ABC):
    """Base class for every tool."""

    #: The name the model sees. A snake_case verb (docs/project-structure.md §4).
    name: ClassVar[str]
    #: One line, shown in the tool schema. Small models follow short descriptions better.
    description: ClassVar[str]
    risk: ClassVar[Risk]
    args_model: ClassVar[type[BaseModel]]

    #: Whether several calls to this tool may run concurrently. True only for reads.
    concurrent_safe: ClassVar[bool] = False

    def validate(self, raw: dict[str, Any]) -> ArgsT | ToolResult:
        """Parse arguments, or return a corrective error.

        ``extra="forbid"`` on the args model turns an invented parameter into a validation
        error the model can read, rather than a silently ignored one
        (docs/safety-and-tool-use.md §2.3).
        """
        try:
            return self.args_model.model_validate(raw)  # type: ignore[return-value]
        except ValidationError as exc:
            return invalid_arguments(
                self.name,
                _format_validation_error(exc),
                schema_hint=self.schema_hint(),
            )

    @abstractmethod
    def prepare(self, args: ArgsT, context: ToolContext) -> Prepared:
        """Compute what this call would do. **Must not mutate anything.**"""

    @abstractmethod
    def execute(self, args: ArgsT, context: ToolContext, prepared: Prepared) -> ToolResult:
        """Do it. Must re-verify any precondition ``prepare()`` relied on."""

    def schema(self) -> dict[str, Any]:
        """The JSON tool schema sent to the model, derived from ``args_model``.

        Generated rather than hand-written, so the schema the model sees and the schema
        that validates its call cannot drift apart.
        """
        parameters = self.args_model.model_json_schema()
        parameters.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def schema_hint(self) -> str:
        """A one-line argument summary, for error messages."""
        fields = self.args_model.model_fields
        parts = []
        for name, info in fields.items():
            marker = "" if info.is_required() else "?"
            parts.append(f"{name}{marker}")
        return f"{self.name}({', '.join(parts)})"


def _format_validation_error(exc: ValidationError) -> str:
    """Render a Pydantic error as something a model can act on."""
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "arguments"
        problems.append(f"{location}: {error['msg']}")
    return "; ".join(problems[:4])
