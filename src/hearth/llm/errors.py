"""Domain exceptions for the LLM layer."""

from __future__ import annotations


class LLMError(Exception):
    """Base class for every error raised by ``hearth.llm``."""


class ProviderConfigError(LLMError):
    """The provider could not be constructed — a bad host, a refused cloud tag."""


class ProviderUnavailableError(LLMError):
    """The provider could not complete a request — the server is down, timed out, or
    returned something Hearth could not parse."""
