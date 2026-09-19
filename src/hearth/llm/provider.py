"""``LLMProvider`` — the boundary every model backend implements.

Kept deliberately small and native-shaped (``chat_stream`` mirrors Ollama's own streaming
API) rather than wrapping something more general like OpenAI's chat-completions shape,
because Hearth's whole premise is a loopback-only, tool-calling, thinking-aware local
model, and translating through a generic abstraction would either lose those features or
leak Ollama-specific behaviour through it anyway (docs/tech-stack.md §4.2).

``ScriptedProvider`` implements this for tests, so the entire agent loop, runner and CLI
are testable without a running Ollama instance.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from hearth.llm.types import ChatChunk, ChatRequest, ModelInfo, RunningModel


@runtime_checkable
class LLMProvider(Protocol):
    """What every model backend must supply."""

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Stream a chat completion, one chunk at a time.

        The final chunk has ``done=True`` and carries ``usage``. Implementations must not
        buffer the whole response before yielding — the whole point of streaming is that
        `on_text` sees tokens as they are generated.
        """
        yield  # type: ignore[misc] # pragma: no cover - a stub body, never executed.
        # The `yield` is load-bearing for the type checker, not the runtime: without one,
        # this reads as a plain coroutine returning an AsyncIterator, not an async
        # generator function, and every real async-generator implementation of this
        # protocol would then fail structural matching against `LLMProvider`.

    async def version(self) -> str:
        """The server's reported version, for `hearth doctor`."""
        ...

    async def show(self, model: str) -> ModelInfo:
        """What the server knows about one model: family, parameter size, capabilities."""
        ...

    async def running(self) -> list[RunningModel]:
        """Models currently loaded, and whether they are actually on the GPU."""
        ...

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None = None,
        on_cpu: bool = False,
    ) -> list[list[float]]:
        """Embed a batch of texts with an embedding model.

        ``on_cpu`` pins the request off the GPU so an interactive query embedding cannot
        evict the chat model from a small GPU mid-session (docs/system-design.md §6.7).
        """
        ...

    async def close(self) -> None:
        """Release any held connection. Safe to call more than once."""
        ...
