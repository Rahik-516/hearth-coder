"""Boundary types for the LLM gateway.

These are the vocabulary every layer above ``hearth.llm`` speaks. They deliberately do not
mirror the Ollama SDK's own models: keeping a translation boundary here is what lets a
second backend be added as an additive implementation of ``LLMProvider`` rather than a
rewrite (docs/tech-stack.md §4.2).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]

#: Per-mode thinking control. Ollama accepts a bool or a level; "off" maps to False.
ThinkLevel = Literal["off", "low", "medium", "high"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCall(_Model):
    """A tool invocation requested by the model.

    ``call_id`` is Hearth's own identifier. Local models frequently omit or repeat any id
    the API offers, and the approval flow needs a stable handle per proposed call.
    """

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(_Model):
    """One turn of conversation.

    ``thinking`` is kept separate from ``content`` so it can be rendered collapsed and
    dropped from history when a profile sets ``preserve_thinking = false`` — reasoning
    tokens are expensive to carry on a 12K context.
    """

    role: Role
    content: str = ""
    thinking: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    #: For role="tool": which call this result answers.
    tool_call_id: str | None = None
    tool_name: str | None = None


class Sampling(_Model):
    """Sampling overrides.

    Ollama honours the defaults baked into a model's Modelfile, so every field here is
    optional and only set when a model card recommends something specific
    (docs/model-recommendations.md §6.1).
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)


class ChatRequest(_Model):
    """A single chat completion request.

    ``num_ctx`` is **required**. There is no default and no None: Ollama's own default
    depends on detected VRAM and can be as small as 4K, and silently truncating a prompt
    takes the system prompt and tool definitions with it (docs/system-design.md §1.1).
    Making the field mandatory means that failure mode cannot be reached by forgetting.
    """

    model: str
    messages: list[Message]
    num_ctx: int = Field(ge=1)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    think: ThinkLevel = "off"
    #: JSON schema for structured output (plan mode, listwise rerank).
    format: dict[str, Any] | None = None
    keep_alive: str | None = None
    sampling: Sampling = Field(default_factory=Sampling)
    num_predict: int | None = None


class Usage(_Model):
    """Token and timing accounting for one request.

    ``prompt_tokens`` is what the *server* counted, which is how the TokenEstimator
    calibrates itself and how silent truncation is detected — a large gap between this and
    Hearth's own estimate means the prompt did not arrive intact
    (docs/system-design.md §5.3).
    """

    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    prefill_ms: float | None = None
    generation_ms: float | None = None
    total_ms: float | None = None

    @property
    def generation_tps(self) -> float | None:
        """Generated tokens per second, or None when the server reported no timing."""
        if not self.generated_tokens or not self.generation_ms:
            return None
        return self.generated_tokens / (self.generation_ms / 1000.0)


class ChatChunk(_Model):
    """One streamed piece of a response.

    Exactly one of the delta fields is normally populated. The final chunk has
    ``done=True`` and carries ``usage`` plus any accumulated ``tool_calls``.
    """

    content_delta: str = ""
    thinking_delta: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    done: bool = False
    done_reason: str | None = None
    usage: Usage | None = None


class ModelInfo(_Model):
    """What ``/api/show`` reports about a model."""

    name: str
    family: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    capabilities: list[str] = Field(default_factory=list)

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.capabilities

    @property
    def supports_thinking(self) -> bool:
        return "thinking" in self.capabilities

    @property
    def supports_embedding(self) -> bool:
        return "embedding" in self.capabilities

    @property
    def parameter_count_b(self) -> float | None:
        """Parameter count in billions, parsed from e.g. "4.0B". None when unknown.

        Used by profile size rules, which lower tool-reliability expectations and step
        limits for smaller models (docs/model-recommendations.md §6.1).
        """
        raw = (self.parameter_size or "").strip().upper()
        if not raw.endswith("B"):
            return None
        try:
            return float(raw[:-1])
        except ValueError:
            return None


class RunningModel(_Model):
    """One entry from ``/api/ps`` — what is loaded right now, and where.

    ``size_vram`` versus ``size`` is how ``hearth doctor`` answers the question that
    matters most on a 6 GB card: is this model actually on the GPU, or has it spilled to
    CPU where prefill will be many times slower?
    """

    name: str
    size: int | None = None
    size_vram: int | None = None
    context_length: int | None = None
    expires_at: str | None = None

    @property
    def gpu_fraction(self) -> float | None:
        """Fraction of the model resident in VRAM, 0.0 to 1.0. None when unknown."""
        if not self.size or self.size_vram is None:
            return None
        return self.size_vram / self.size

    @property
    def fully_on_gpu(self) -> bool:
        fraction = self.gpu_fraction
        return fraction is not None and fraction >= 0.99
