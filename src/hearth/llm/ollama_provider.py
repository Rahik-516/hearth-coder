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
        # `False`, not `None`: None leaves the decision to the server, and a thinking model
        # then thinks by default. The observed failure is silence — the model spends its
        # whole `num_predict` budget in `message.thinking` and returns empty content. "off"
        # has to actively disable it.
        think: Literal["low", "medium", "high"] | bool = (
            False if request.think == "off" else request.think
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
        """The server's version string, from ``/api/version``.

        Reached through the SDK's own httpx client because the SDK wraps every endpoint
        *except* this one, and the alternatives are worse: asking for the model list and
        reading a ``version`` attribute off it returns None (it is not there), which is
        what `hearth doctor` used to report as "version unknown" and then warn was below
        the configured minimum. Using the client Hearth already holds keeps this on the
        same loopback connection and adds no dependency.
        """
        transport = getattr(self._client, "_client", None)
        if transport is None:  # pragma: no cover - the SDK always builds one
            raise ProviderUnavailableError("the Ollama client exposes no HTTP transport")

        try:
            response = await transport.get("/api/version")
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # network, decode and status failures alike
            raise ProviderUnavailableError(f"could not reach Ollama: {exc}") from exc

        return str(payload.get("version") or "unknown")

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
            context_length=_context_length(info),
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


def _context_length(info: object) -> int | None:
    """The model's trained context window, from the architecture-prefixed key.

    Ollama reports it as ``<arch>.context_length`` — ``qwen35.context_length`` for this
    model — so the key cannot be hardcoded and is found by suffix instead. Worth having:
    it is how `hearth doctor` can tell the user that their configured ``num_ctx`` exceeds
    what the model was trained for, which Ollama itself answers by silently truncating.
    """
    model_info = getattr(info, "modelinfo", None) or getattr(info, "model_info", None) or {}
    if not isinstance(model_info, dict):
        return None

    for key, value in model_info.items():
        if str(key).endswith(".context_length") and isinstance(value, int):
            return value
    return None


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
