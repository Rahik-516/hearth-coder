"""A model that emits a broken tool call should cost a retry, not the task.

Found by the first live plan-mode eval: Ollama's server parses the model's tool calls, and a
sample that closed an XML element wrongly failed the whole request. The turn ended on that
one bad sample and the task scored zero — a sampling accident recorded as a capability
failure.

The line to hold is between two kinds of provider error. A malformed *model output* is
sampling, so the next attempt usually works and it is retried. A server that is *down* will
still be down, so it is not: retrying it would only make the failure slower.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from hearth.core.agent_loop import MAX_MALFORMED_RETRIES, AgentLoop
from hearth.core.bus import EventBus
from hearth.core.events import Notice
from hearth.llm.errors import MalformedOutputError, ProviderUnavailableError
from hearth.llm.ollama_provider import OllamaProvider, _chat_error
from hearth.llm.types import ChatChunk, ChatRequest, Message
from hearth.tools.base import ToolContext
from hearth.tools.gateway import ToolGateway
from hearth.tools.registry import ToolRegistry

XML_ERROR = "XML syntax error on line 3: element <function> closed by </parameter> (status code: -1)"


class FlakyProvider:
    """Raises the given errors in order, then answers."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        yield ChatChunk(content_delta="all done")
        yield ChatChunk(done=True, done_reason="stop")


def loop_for(provider: FlakyProvider, tmp_path) -> tuple[AgentLoop, EventBus, list[Notice]]:
    bus = EventBus()
    notices: list[Notice] = []
    bus.subscribe(lambda event: notices.append(event) if isinstance(event, Notice) else None)
    gateway = ToolGateway(registry=ToolRegistry([]), context=ToolContext(workspace=tmp_path))
    return AgentLoop(provider=provider, gateway=gateway, bus=bus), bus, notices


def request() -> ChatRequest:
    return ChatRequest(model="scripted", messages=[Message(role="user", content="hi")], num_ctx=4096)


# ---------------------------------------------------------------- classification


@pytest.mark.parametrize(
    "message",
    [
        XML_ERROR,
        "error parsing tool call: invalid character 'x' looking for beginning of value",
        "unexpected end of JSON input",
    ],
)
def test_a_model_output_parse_failure_is_classified_as_malformed(message: str) -> None:
    error = _chat_error("chat stream failed", RuntimeError(message))

    assert isinstance(error, MalformedOutputError)
    assert isinstance(error, ProviderUnavailableError), (
        "still a provider error for callers that catch the base"
    )


@pytest.mark.parametrize(
    "message",
    [
        "connection refused",
        "timed out",
        "model 'nope' not found",
        "500 internal server error",
    ],
)
def test_an_unavailable_server_is_not_classified_as_malformed(message: str) -> None:
    """The distinction that decides whether retrying makes sense."""
    error = _chat_error("chat request failed", RuntimeError(message))

    assert not isinstance(error, MalformedOutputError)
    assert isinstance(error, ProviderUnavailableError)


async def test_the_real_provider_raises_the_malformed_type_from_a_failed_stream() -> None:
    class Client:
        async def chat(self, **_: object):
            raise RuntimeError(XML_ERROR)

    provider = OllamaProvider(host="http://127.0.0.1:11434")
    provider._client = Client()  # type: ignore[assignment]

    with pytest.raises(MalformedOutputError):
        async for _ in provider.chat_stream(request()):
            pass


# ------------------------------------------------------------------------ the loop


async def test_one_bad_sample_is_retried_and_the_turn_succeeds(tmp_path) -> None:
    provider = FlakyProvider(MalformedOutputError(XML_ERROR))
    loop, _bus, _notices = loop_for(provider, tmp_path)

    outcome = await loop.run(base_request=request())

    assert outcome.answer == "all done"
    assert provider.calls == 2


async def test_a_retry_is_announced_to_the_user(tmp_path) -> None:
    provider = FlakyProvider(MalformedOutputError(XML_ERROR))
    loop, bus, notices = loop_for(provider, tmp_path)

    await loop.run(base_request=request())
    await bus.close()

    assert any("malformed tool call" in notice.message and "1/2" in notice.message for notice in notices)


async def test_retries_are_bounded(tmp_path) -> None:
    """A model that cannot produce a valid call in three tries is unable, not unlucky; the
    failure is reported rather than looped on."""
    provider = FlakyProvider(*[MalformedOutputError(XML_ERROR)] * (MAX_MALFORMED_RETRIES + 2))
    loop, _bus, _notices = loop_for(provider, tmp_path)

    with pytest.raises(MalformedOutputError):
        await loop.run(base_request=request())

    assert provider.calls == MAX_MALFORMED_RETRIES + 1


async def test_an_unavailable_server_is_not_retried(tmp_path) -> None:
    """A dead server will still be dead; retrying would only make the failure slower."""
    provider = FlakyProvider(ProviderUnavailableError("connection refused"))
    loop, _bus, _notices = loop_for(provider, tmp_path)

    with pytest.raises(ProviderUnavailableError):
        await loop.run(base_request=request())

    assert provider.calls == 1


async def test_success_on_the_last_allowed_retry_counts(tmp_path) -> None:
    provider = FlakyProvider(*[MalformedOutputError(XML_ERROR)] * MAX_MALFORMED_RETRIES)
    loop, _bus, _notices = loop_for(provider, tmp_path)

    outcome = await loop.run(base_request=request())

    assert outcome.answer == "all done"
    assert provider.calls == MAX_MALFORMED_RETRIES + 1
