"""The real backend: Ollama's native API via the official async SDK.

Two refusals happen before any request leaves this class, not after a response comes
back: a non-loopback host (unless explicitly overridden) and a cloud-tagged model.
Checking afterward would mean the prompt already crossed the network by the time Hearth
noticed (docs/system-design.md §1, non-negotiable rule 1).

``num_ctx`` is required on every request (``ChatRequest`` enforces this at the type level)
and forwarded as-is into Ollama's ``options.num_ctx`` — never a default, because Ollama's
own default depends on detected VRAM and silently truncates the prompt if the context is
exceeded.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Literal

import ollama

from hearth.llm.errors import ProviderConfigError, ProviderUnavailableError
from hearth.llm.guards import ensure_loopback_host, ensure_not_cloud_model
from hearth.llm.types import (
    ChatChunk,
    ChatRequest,
    ModelInfo,
    RunningModel,
    ToolCall,
    Usage,
)

DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaProvider:
    """Talks to a local Ollama server over its native HTTP API."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        allow_remote_host: bool = False,
        timeout_s: float = 300.0,
    ) -> None:
        ensure_loopback_host(host, allow_remote_host=allow_remote_host)
        self._host = host
        try:
            self._client = ollama.AsyncClient(host=host, timeout=timeout_s)
        except Exception as exc:
            raise ProviderConfigError(f"could not construct an Ollama client: {exc}") from exc

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        ensure_not_cloud_model(request.model)

        options: dict[str, object] = {"num_ctx": request.num_ctx}
        sampling = request.sampling
        for key, value in (
            ("temperature", sampling.temperature),
            ("top_p", sampling.top_p),
            ("top_k", sampling.top_k),
            ("min_p", sampling.min_p),
            ("repeat_penalty", sampling.repeat_penalty),
            ("seed", sampling.seed),
        ):
            if value is not None:
                options[key] = value
        if sampling.stop:
            options["stop"] = list(sampling.stop)
        if request.num_predict is not None:
            options["num_predict"] = request.num_predict

        messages = [_to_ollama_message(message) for message in request.messages]
        think: Literal["low", "medium", "high"] | None = (
            None if request.think == "off" else request.think
        )

        try:
            stream = await self._client.chat(
                model=request.model,
                messages=messages,
                tools=request.tools or None,
                stream=True,
                think=think,
                format=request.format,
                options=options,
                keep_alive=request.keep_alive,
            )
        except Exception as exc:
            raise ProviderUnavailableError(f"chat request failed: {exc}") from exc

        try:
            async for part in stream:
                yield _to_chat_chunk(part)
        except Exception as exc:
            raise ProviderUnavailableError(f"chat stream failed: {exc}") from exc

    async def version(self) -> str:
        try:
            info = await self._client.list()
        except Exception as exc:
            raise ProviderUnavailableError(f"could not reach Ollama: {exc}") from exc
        return getattr(info, "version", None) or "unknown"

    async def show(self, model: str) -> ModelInfo:
        try:
            info = await self._client.show(model)
        except Exception as exc:
            raise ProviderUnavailableError(f"could not show {model!r}: {exc}") from exc

        details = getattr(info, "details", None)
        capabilities = list(getattr(info, "capabilities", None) or [])
        return ModelInfo(
            name=model,
            family=getattr(details, "family", None),
            parameter_size=getattr(details, "parameter_size", None),
            quantization=getattr(details, "quantization_level", None),
            context_length=None,
            capabilities=capabilities,
        )

    async def running(self) -> list[RunningModel]:
        try:
            response = await self._client.ps()
        except Exception as exc:
            raise ProviderUnavailableError(f"could not list running models: {exc}") from exc

        return [
            RunningModel(
                name=model.model,
                size=getattr(model, "size", None),
                size_vram=getattr(model, "size_vram", None),
                context_length=None,
                expires_at=str(getattr(model, "expires_at", "")) or None,
            )
            for model in getattr(response, "models", [])
        ]

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None = None,
        on_cpu: bool = False,
    ) -> list[list[float]]:
        options: dict[str, object] = {"num_gpu": 0} if on_cpu else {}
        try:
            response = await self._client.embed(
                model=model,
                input=list(texts),
                dimensions=dimensions,
                options=options or None,
            )
        except Exception as exc:
            raise ProviderUnavailableError(f"embed request failed: {exc}") from exc
        return [list(vector) for vector in response.embeddings]

    async def close(self) -> None:
        client = getattr(self._client, "_client", None)
        aclose = getattr(client, "aclose", None)
        if aclose is not None:
            await aclose()


def _to_ollama_message(message: object) -> dict[str, object]:
    payload: dict[str, object] = {"role": message.role, "content": message.content}  # type: ignore[attr-defined]
    if message.tool_calls:  # type: ignore[attr-defined]
        payload["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls  # type: ignore[attr-defined]
        ]
    return payload


def _to_chat_chunk(part: object) -> ChatChunk:
    message = getattr(part, "message", None)
    content = getattr(message, "content", "") or ""
    thinking = getattr(message, "thinking", "") or ""

    tool_calls: list[ToolCall] = []
    for index, call in enumerate(getattr(message, "tool_calls", None) or []):
        function = getattr(call, "function", None)
        name = getattr(function, "name", None) or ""
        arguments = dict(getattr(function, "arguments", None) or {})
        tool_calls.append(ToolCall(call_id=f"native-{index}", name=name, arguments=arguments))

    done = bool(getattr(part, "done", False))
    usage = None
    if done:
        prompt_tokens = getattr(part, "prompt_eval_count", None)
        generated_tokens = getattr(part, "eval_count", None)
        prompt_ns = getattr(part, "prompt_eval_duration", None)
        eval_ns = getattr(part, "eval_duration", None)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            prefill_ms=(prompt_ns / 1_000_000) if prompt_ns else None,
            generation_ms=(eval_ns / 1_000_000) if eval_ns else None,
            total_ms=((prompt_ns or 0) + (eval_ns or 0)) / 1_000_000 or None,
        )

    return ChatChunk(
        content_delta=content,
        thinking_delta=thinking,
        tool_calls=tool_calls,
        done=done,
        done_reason=getattr(part, "done_reason", None),
        usage=usage,
    )
