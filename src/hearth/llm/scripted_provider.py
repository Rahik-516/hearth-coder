"""A deterministic fake provider, for tests and demos that must not depend on Ollama.

Built against the same ``LLMProvider`` shape as the real backend, so a test written
against this one is exercising the same call sites the real provider will hit — the only
thing that differs is where the tokens come from.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from hearth.llm.errors import ProviderUnavailableError
from hearth.llm.types import ChatChunk, ChatRequest, ModelInfo, RunningModel, ToolCall, Usage

#: Fallback embedding width when a caller does not request a specific dimension count.
_DEFAULT_EMBED_DIMS = 8


@dataclass
class ScriptedResponse:
    """One canned turn: the text to stream, in ``chunk_size``-character pieces.

    Chunking matters for tests that assert on *streaming* behaviour — a renderer that only
    works when the whole answer arrives in one piece is a renderer that will stutter on a
    real model, which never delivers a turn as a single chunk.
    """

    text: str
    chunk_size: int = 1
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    done_reason: str = "stop"


#: Accepted alongside ``ScriptedResponse`` for the (text, tool_calls) shorthand some
#: tests use directly.
ScriptLike = ScriptedResponse | tuple[str, list[ToolCall]]


class ScriptedProvider:
    """Replays a fixed script of responses, one per call to :meth:`chat_stream`.

    If the script is shorter than the number of calls made, the last entry repeats —
    useful for a loop that should keep answering the same way until a limit stops it.
    """

    def __init__(
        self,
        script: list[ScriptLike] | None = None,
        *,
        models: dict[str, ModelInfo] | None = None,
    ) -> None:
        self._script = [_normalize(entry) for entry in (script or [ScriptedResponse("ok")])]
        self.requests: list[ChatRequest] = []
        #: One entry per :meth:`embed` call: ``(texts, model, dimensions, on_cpu)``.
        self.embed_calls: list[tuple[list[str], str, int | None, bool]] = []
        #: What :meth:`show` reports for a known model tag. A tag not in here behaves like
        #: an unpulled model on a real server: `show` fails rather than inventing a result,
        #: which is what makes ``hearth doctor``'s "ollama pull" suggestion testable.
        self._models = models or {}

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._script) - 1)
        response = self._script[index]

        if response.thinking:
            yield ChatChunk(thinking_delta=response.thinking)

        size = max(1, response.chunk_size)
        for start in range(0, len(response.text), size):
            yield ChatChunk(content_delta=response.text[start : start + size])
        if not response.text:
            yield ChatChunk(content_delta="")

        yield ChatChunk(
            tool_calls=response.tool_calls,
            done=True,
            done_reason=response.done_reason,
            usage=Usage(prompt_tokens=0, generated_tokens=len(response.text)),
        )

    async def version(self) -> str:
        return "scripted"

    async def show(self, model: str) -> ModelInfo:
        if model in self._models:
            return self._models[model]
        if not self._models:
            return ModelInfo(name=model, capabilities=["tools"])
        raise ProviderUnavailableError(f"model {model!r} is not pulled")

    async def running(self) -> list[RunningModel]:
        return []

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None = None,
        on_cpu: bool = False,
    ) -> list[list[float]]:
        """Deterministic fake vectors, stable per input text so caching tests are meaningful."""
        self.embed_calls.append((list(texts), model, dimensions, on_cpu))
        dims = dimensions or _DEFAULT_EMBED_DIMS
        return [_fake_vector(text, dims) for text in texts]

    async def close(self) -> None:
        return None


def _normalize(entry: ScriptLike) -> ScriptedResponse:
    if isinstance(entry, ScriptedResponse):
        return entry
    text, calls = entry
    return ScriptedResponse(text, tool_calls=list(calls))


def _fake_vector(text: str, dims: int) -> list[float]:
    """A stable, content-derived vector — identical text always yields identical output.

    Not remotely embedding-quality, but real caching, batching and normalization code
    cannot tell the difference between this and a real model's output at the boundary
    this class replaces, which is the entire point of a scripted provider.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=64).digest()
    return [(digest[i % len(digest)] / 127.5) - 1.0 for i in range(dims)]
