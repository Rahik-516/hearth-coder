"""Domain exceptions for the LLM layer."""

from __future__ import annotations


class LLMError(Exception):
    """Base class for every error raised by ``hearth.llm``."""


class ProviderConfigError(LLMError):
    """The provider could not be constructed — a bad host, a refused cloud tag."""


class ProviderUnavailableError(LLMError):
    """The provider could not complete a request — the server is down, timed out, or
    returned something Hearth could not parse."""


class MalformedOutputError(ProviderUnavailableError):
    """The server could not parse what the *model* produced — a malformed tool call.

    Distinct from an unreachable server because the right response is opposite. A dead
    server will still be dead on the second attempt; a model that emitted a broken tool
    call is sampling, and the next sample usually is not broken. Callers may retry this
    one and must not retry the rest.
    """
