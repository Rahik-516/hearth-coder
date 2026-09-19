"""Live tests against a real local Ollama.

Excluded by default; run deliberately with ``uv run pytest -m live``.

House rules for anything added here (docs/implementation-roadmap.md, Phase 0):

* Keep prompts under ~8K tokens. The reference machine has 6 GB of VRAM.
* Run sequentially, never in parallel, and never load two chat models at once.
* Assert on *mechanics* — streaming shape, tool-call plumbing, usage accounting —
  never on model wording, which is not reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from hearth.config.loader import load_config
from hearth.llm.ollama_provider import OllamaProvider
from hearth.llm.profiles import ProfileRegistry
from hearth.llm.types import ChatRequest, Message, Sampling

pytestmark = pytest.mark.live

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


@pytest.fixture(scope="module")
def config():
    return load_config().config


@pytest.fixture
async def provider(config):
    instance = OllamaProvider(
        host=config.ollama.host,
        allow_remote_host=config.ollama.allow_remote_host,
        timeout_s=config.ollama.request_timeout_s,
    )
    yield instance
    await instance.close()


async def test_server_reports_a_version(provider) -> None:
    version = await provider.version()
    assert version
    assert version[0].isdigit()


async def test_chat_model_advertises_tool_support(provider, config) -> None:
    """Agent mode depends on this; the profile's autonomy settings assume it."""
    info = await provider.show(config.models.chat)

    assert info.supports_tools, f"{config.models.chat} cannot call tools"
    assert info.context_length is not None
    assert info.context_length >= config.models.num_ctx, "configured num_ctx exceeds what the model supports"


async def test_streaming_chat_produces_text_and_usage(provider, config) -> None:
    request = ChatRequest(
        model=config.models.chat,
        messages=[
            Message(role="system", content="Answer in one short sentence."),
            Message(role="user", content="What is 2 + 2?"),
        ],
        num_ctx=config.models.num_ctx,
        keep_alive=config.models.keep_alive,
        sampling=Sampling(temperature=0.0, seed=1),
        num_predict=64,
    )

    chunks = [chunk async for chunk in provider.chat_stream(request)]

    assert len(chunks) > 1, "expected a stream, not a single response"
    assert not any(c.done for c in chunks[:-1]), "only the final chunk may be done"

    final = chunks[-1]
    assert final.done is True
    assert final.usage is not None
    assert final.usage.prompt_tokens and final.usage.prompt_tokens > 0
    assert final.usage.generated_tokens and final.usage.generated_tokens > 0

    text = "".join(c.content_delta for c in chunks)
    assert text.strip(), "model produced no visible text"


async def test_num_ctx_is_honoured_not_defaulted(provider, config) -> None:
    """A small explicit num_ctx must be applied.

    This is the guard against Ollama's VRAM-dependent default silently replacing what
    Hearth asked for (docs/system-design.md §1.1).
    """
    request = ChatRequest(
        model=config.models.chat,
        messages=[Message(role="user", content="Say OK.")],
        num_ctx=4096,
        keep_alive=config.models.keep_alive,
        num_predict=16,
    )

    chunks = [chunk async for chunk in provider.chat_stream(request)]
    running = await provider.running()

    loaded = next((m for m in running if m.name == config.models.chat), None)
    assert chunks[-1].done
    if loaded is not None and loaded.context_length is not None:
        assert loaded.context_length == 4096, (
            f"asked for num_ctx=4096 but the server loaded {loaded.context_length}"
        )


async def test_native_tool_call_round_trip(provider, config) -> None:
    """The model should emit a structured tool call, and the SDK path should surface it."""
    request = ChatRequest(
        model=config.models.chat,
        messages=[
            Message(
                role="system",
                content="You must use the provided tool to answer weather questions.",
            ),
            Message(role="user", content="What is the weather in Paris?"),
        ],
        num_ctx=config.models.num_ctx,
        keep_alive=config.models.keep_alive,
        tools=[WEATHER_TOOL],
        sampling=Sampling(temperature=0.0, seed=1),
        num_predict=128,
    )

    calls = [call async for chunk in provider.chat_stream(request) for call in chunk.tool_calls]

    assert calls, "model did not emit a tool call"
    assert calls[0].name == "get_weather"
    assert calls[0].call_id, "every call needs a stable id for the approval flow"
    assert "paris" in str(calls[0].arguments).lower()


async def test_embeddings_are_normalized_and_stable(provider, config) -> None:
    vectors = await provider.embed(
        ["invoice finalization", "invoice finalization", "unrelated kitchen recipe"],
        model=config.models.embed,
        dimensions=config.models.embed_dimensions,
    )

    assert vectors.shape == (3, config.models.embed_dimensions)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3), "vectors must be unit length"

    identical = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])

    assert identical > 0.99, "the same text must embed to the same vector"
    assert identical > unrelated, "identical text must be closer than unrelated text"


async def test_profile_resolves_for_the_configured_model(provider, config) -> None:
    """The size rules must key off what the server actually reports."""
    info = await provider.show(config.models.chat)
    profile = ProfileRegistry.load().for_model(config.models.chat, parameter_count_b=info.parameter_count_b)

    assert profile.family != "unknown", f"no profile matches {config.models.chat}"
    assert profile.tools == "native"
