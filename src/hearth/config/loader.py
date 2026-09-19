"""Layered configuration: defaults -> global -> project.

**Project allow rules are ignored until the project is trusted.** Deny and ask rules
always apply, because they can only tighten (docs/safety-and-tool-use.md §5.5) — a cloned
repository is allowed to ask for more caution than the user's own defaults, but it cannot
grant itself anything until a human has looked at what it is asking for. Silently applying
an untrusted project's allow rules would be the exact privilege escalation this project
control, so the alternative — a partial merge that looks applied but isn't — is the wrong
kind of surprising.

Layer order, later wins for scalars, lists append, tables merge recursively:

1. ``defaults.toml`` (bundled with Hearth)
2. the user's global config (``~/.config/hearth/config.toml``)
3. the project's own config (``<repo>/.hearth/config.toml``), gated on trust
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hearth.config import paths
from hearth.config.errors import ConfigParseError, ConfigValidationError
from hearth.config.schema import HearthConfig, PermissionsConfig

#: Layers whose *relaxing* rules are dropped unless the project has been trusted.
_UNTRUSTED_UNTIL_TRUSTED = frozenset({"project"})

_DEFAULTS_FILE = Path(__file__).parent / "defaults.toml"


@dataclass(frozen=True)
class ConfigSource:
    """One layer that was consulted while building the merged config."""

    layer: str
    path: Path
    existed: bool


@dataclass(frozen=True)
class LoadedConfig:
    """The merged, validated configuration, with provenance for `hearth doctor`."""

    config: HearthConfig
    sources: list[ConfigSource] = field(default_factory=list)
    #: Allow rules dropped because the project is untrusted. Non-zero means the CLI should
    #: mention `hearth trust`.
    dropped_allow_rules: int = 0

    @property
    def loaded_paths(self) -> list[Path]:
        return [source.path for source in self.sources if source.existed]


def load_config(
    project_root: Path | None = None,
    *,
    project_trusted: bool = False,
    global_config_path: Path | None = None,
) -> LoadedConfig:
    """Build the merged configuration for one workspace.

    Args:
        project_root: The repository being worked on. Omit for a project-less context
            (e.g. `hearth doctor` before a workspace is chosen) — only the defaults and
            global layers apply.
        project_trusted: Whether ``hearth trust`` has approved the current project config.
            Defaults to False — untrusted is the safe assumption.
        global_config_path: Override for the global config file, so tests never read the
            real user's ``~/.config/hearth/config.toml``. Defaults to that real path.
    """
    sources: list[ConfigSource] = []
    merged: dict[str, Any] = {}
    dropped = 0

    global_path = global_config_path if global_config_path is not None else paths.global_config_file()

    merged = _deep_merge(merged, _read_toml(_DEFAULTS_FILE, layer="defaults", sources=sources, optional=True))
    merged = _deep_merge(merged, _read_toml(global_path, layer="global", sources=sources, optional=True))

    candidates = [] if project_root is None else [("project", paths.project_config_file(project_root))]
    for layer, path in candidates:
        data = _read_toml(path, layer=layer, sources=sources, optional=True)
        if layer in _UNTRUSTED_UNTIL_TRUSTED and not project_trusted:
            data, removed = _strip_allow_rules(data)
            dropped += removed
        merged = _deep_merge(merged, data)

    try:
        config = HearthConfig.model_validate(merged)
    except ValidationError as exc:
        offending = sources[-1].path if sources else None
        raise ConfigValidationError(offending, _format_validation_error(exc)) from exc

    return LoadedConfig(config=config, sources=sources, dropped_allow_rules=dropped)


def read_project_permissions(project_root: Path) -> PermissionsConfig:
    """The project's own permission rules, with no trust filtering applied.

    :func:`load_config` deliberately drops untrusted project allow rules, which is right
    for running a session and useless for `hearth trust` — a review screen exists to show
    exactly the rules that are currently being ignored. Returns an empty section when the
    project has no config or declares no permissions.
    """
    path = paths.project_config_file(project_root)
    try:
        raw = path.read_bytes()
    except OSError:
        return PermissionsConfig()

    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigParseError(path, str(exc)) from exc

    permissions = data.get("permissions")
    if not isinstance(permissions, dict):
        return PermissionsConfig()

    try:
        return PermissionsConfig.model_validate(permissions)
    except ValidationError as exc:
        raise ConfigValidationError(path, _format_validation_error(exc)) from exc


def _read_toml(
    path: Path,
    *,
    layer: str,
    sources: list[ConfigSource],
    optional: bool = False,
) -> dict[str, Any]:
    """Read one TOML file, recording it as a source whether or not it exists."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        if not optional:
            raise
        sources.append(ConfigSource(layer=layer, path=path, existed=False))
        return {}
    except OSError as exc:
        raise ConfigParseError(path, str(exc)) from exc

    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigParseError(path, str(exc)) from exc

    sources.append(ConfigSource(layer=layer, path=path, existed=True))
    return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge ``overlay`` onto ``base``. Tables recurse, lists append, scalars replace."""
    result = dict(base)
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        elif isinstance(existing, list) and isinstance(value, list):
            result[key] = _append_unique(existing, value)
        else:
            result[key] = value
    return result


def _append_unique(base: list[Any], overlay: list[Any]) -> list[Any]:
    """Append, skipping exact duplicates so re-declaring the same rule twice is a no-op."""
    result = list(base)
    for item in overlay:
        if item not in result:
            result.append(item)
    return result


def _strip_allow_rules(data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Remove ``permissions.allow`` from an untrusted layer, returning how many were dropped."""
    permissions = data.get("permissions")
    if not isinstance(permissions, dict):
        return data, 0
    allow = permissions.get("allow")
    if not isinstance(allow, list) or not allow:
        return data, 0

    pruned_permissions = {k: v for k, v in permissions.items() if k != "allow"}
    pruned = {**data, "permissions": pruned_permissions}
    return pruned, len(allow)


def _format_validation_error(exc: ValidationError) -> str:
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "config"
        problems.append(f"{location}: {error['msg']}")
    return "; ".join(problems[:6])
