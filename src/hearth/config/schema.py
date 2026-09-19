"""Configuration schema.

Every section sets ``extra="forbid"``. A mistyped key is an error rather than a silently
ignored line — the alternative is a permission rule that looks applied but isn't, which is
exactly the failure the safety model cannot tolerate (docs/safety-and-tool-use.md §1.3).

The shape mirrors the worked example in docs/system-design.md §14.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Mode = Literal["chat", "plan", "agent"]
PermissionLevel = Literal["supervised", "auto-edit"]
EmbedPlacement = Literal["auto", "gpu", "cpu"]
SummaryMode = Literal["off", "lazy", "background"]
Tier = Literal["tier1", "tier2", "tier3", "tier4"]


class _Section(BaseModel):
    """Base for every config section: unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")


class OllamaConfig(_Section):
    """How to reach the local Ollama server.

    ``allow_remote_host`` is the single, explicit opt-out from the loopback guarantee. It
    is deliberately awkward to set by accident (docs/tech-stack.md §4.3).
    """

    host: str = "http://127.0.0.1:11434"
    allow_remote_host: bool = False
    min_version: str = "0.34.0"
    request_timeout_s: float = Field(default=300.0, gt=0)


class ModelsConfig(_Section):
    """Model selection and the memory-shaped parameters that go with it.

    ``num_ctx`` is always sent explicitly on every request and held constant for a
    session: Ollama's own default depends on detected VRAM and can be far smaller than a
    coding agent needs, and exceeding it truncates the prompt silently — taking the system
    prompt or tool definitions with it (docs/system-design.md §1.1).
    """

    tier: Tier = "tier1"
    chat: str = "qwen3.5:4b"
    eval_chat: str | None = None
    fast: str | None = None
    embed: str = "qwen3-embedding:0.6b"
    embed_dimensions: int | None = Field(default=1024, gt=0)
    num_ctx: int = Field(default=12288, ge=2048)
    keep_alive: str = "20m"
    embed_placement: EmbedPlacement = "auto"


class IndexConfig(_Section):
    """What gets indexed. Built-in ignore layers live in code, not here."""

    max_file_bytes: int = Field(default=1_000_000, gt=0)
    exclude: list[str] = Field(default_factory=list)
    include: list[str] = Field(default_factory=list)
    summaries: SummaryMode = "off"


class RetrievalWeights(_Section):
    """Fusion weights for weighted RRF. Tune these against `hearth eval retrieval`."""

    dense: float = Field(default=1.0, ge=0)
    bm25: float = Field(default=1.0, ge=0)
    symbol: float = Field(default=1.5, ge=0)
    path: float = Field(default=0.8, ge=0)


class RetrievalConfig(_Section):
    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)
    rrf_k: int = Field(default=60, gt=0)
    rerank: bool = False
    max_chunks_per_file: int = Field(default=3, gt=0)


class AgentConfig(_Section):
    default_mode: Mode = "chat"
    permission_level: PermissionLevel = "supervised"
    todo_tool: bool = True


class OfflineConfig(_Section):
    """``enforce = true`` denies commands classified as network-capable.

    The classifier is a heuristic, so this is defence in depth rather than a guarantee.
    True enforcement needs OS sandboxing (docs/safety-and-tool-use.md §8.3).
    """

    enforce: bool = False


class PermissionRule(_Section):
    """One allow/deny/ask rule.

    Matching semantics are specified in docs/safety-and-tool-use.md §5.4 and implemented
    in ``hearth.safety`` — notably, an ``argv`` rule can never match a command containing
    shell metacharacters, which defeats the `pytest; rm -rf ~` class of bypass.
    """

    id: str | None = None
    #: One tool name, ``"*"`` for any, or a list — docs/safety-and-tool-use.md §5.3 shows
    #: ``tool = ["edit_file", "write_file", "multi_edit"]`` for a single path rule.
    tool: str | list[str]
    argv: list[str] | None = None
    path: str | None = None
    env: bool = False
    reason: str | None = None

    @field_validator("tool")
    @classmethod
    def _tool_must_be_named(cls, value: str | list[str]) -> str | list[str]:
        names = [value] if isinstance(value, str) else value
        if not names or not all(name.strip() for name in names):
            raise ValueError("permission rule must name at least one tool")
        return value


class PermissionsConfig(_Section):
    allow: list[PermissionRule] = Field(default_factory=list)
    deny: list[PermissionRule] = Field(default_factory=list)
    ask: list[PermissionRule] = Field(default_factory=list)


class ProjectSettings(_Section):
    """Project-authored settings from ``<repo>/.hearth/config.toml``.

    ``test_command`` and ``lint_command`` are *values*, not permissions: they are resolved
    to argv and classified like any other command, and trusting a project grants them
    nothing (docs/safety-and-tool-use.md §5.6).
    """

    test_command: str | None = None
    lint_command: str | None = None
    languages: list[str] = Field(default_factory=list)


class HearthConfig(_Section):
    """The fully merged configuration."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    offline: OfflineConfig = Field(default_factory=OfflineConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    project: ProjectSettings = Field(default_factory=ProjectSettings)
