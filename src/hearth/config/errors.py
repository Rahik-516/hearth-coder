"""Domain exceptions for configuration loading."""

from __future__ import annotations

from pathlib import Path


class ConfigError(Exception):
    """Base class for every error raised by ``hearth.config``."""


class ConfigParseError(ConfigError):
    """A config file could not be parsed as TOML, or could not be read."""

    def __init__(self, path: Path, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"could not read {path}: {detail}")


class ConfigValidationError(ConfigError):
    """A config file parsed, but failed schema validation.

    ``extra="forbid"`` on every section (docs/system-design.md) means a typo in a key name
    surfaces here rather than being silently ignored.
    """

    def __init__(self, path: Path | None, detail: str) -> None:
        self.path = path
        self.detail = detail
        location = f" in {path}" if path is not None else ""
        super().__init__(f"invalid configuration{location}: {detail}")
