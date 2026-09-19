"""Model profiles: per-family defaults for tool reliability, step limits and thinking.

A profile answers "how much autonomy does this model deserve" from measured behaviour
rather than parameter count alone (docs/model-recommendations.md §6.1) — a smaller model
that reliably emits well-formed tool calls should get more steps than a larger one that
does not, though parameter count still lowers expectations when nothing more specific
matches, since it is the only signal available for a model nobody has profiled yet.

Bundled in ``profiles.toml`` and matched by family prefix against the requested model tag,
falling back to a conservative default for anything unrecognised.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

_BUNDLED = Path(__file__).parent / "profiles.toml"

#: Below this size a model's tool-calling gets systematically worse, not just noisier
#: (docs/model-recommendations.md §6.1) — small enough that even a matched profile's
#: reliability claim is downgraded when no more specific ``size_rules`` entry applies.
_SMALL_MODEL_PARAMS_B = 2.0


@dataclass(frozen=True)
class SizeRule:
    """A downgrade applied to a family's profile below a parameter-count threshold.

    Parameter count is a weak signal compared to a family match, so a size rule can only
    ever cap ``tool_reliability`` or ``max_steps`` for the ``agent`` mode down from what
    the family profile claims — never raise them (docs/model-recommendations.md §6.1).
    """

    max_params_b: float
    tool_reliability: str | None = None
    max_steps_agent: int | None = None


@dataclass(frozen=True)
class ModelProfile:
    """What Hearth assumes about one model family."""

    family: str
    #: How the model emits tool calls: ``native`` (the API's own tool_calls field),
    #: ``text_fallback`` (parsed out of the reply text) or ``none``
    #: (docs/model-recommendations.md §6.1).
    tools: str = "native"
    tool_reliability: str = "medium"
    supports_thinking: bool = False
    preserve_thinking: bool = False
    is_embedding: bool = False
    query_template: str | None = None
    document_template: str | None = None
    #: Steps allowed per mode, before a step limit or a small-model downgrade applies.
    max_steps: dict[str, int] = field(default_factory=lambda: {"chat": 3, "plan": 5, "agent": 8})
    size_rules: tuple[SizeRule, ...] = ()

    def max_steps_for(self, mode: str) -> int:
        return self.max_steps.get(mode, self.max_steps.get("agent", 6))


class ProfileRegistry:
    """Loaded profiles, matched against a requested model tag."""

    def __init__(self, profiles: list[ModelProfile], matches: dict[str, list[str]]) -> None:
        self._profiles = {profile.family: profile for profile in profiles}
        self._matches = matches

    @classmethod
    def load(cls, path: Path | None = None) -> ProfileRegistry:
        """Load the bundled profiles, or a caller-supplied override file."""
        raw = tomllib.loads((path or _BUNDLED).read_text(encoding="utf-8"))
        profiles: list[ModelProfile] = []
        matches: dict[str, list[str]] = {}

        for entry in raw.get("profile", []):
            size_rules = tuple(
                SizeRule(
                    max_params_b=rule["max_params_b"],
                    tool_reliability=rule.get("tool_reliability"),
                    max_steps_agent=rule.get("max_steps_agent"),
                )
                for rule in entry.get("size_rules", [])
            )
            profile = ModelProfile(
                family=entry["family"],
                tools=entry.get("tools", "native"),
                tool_reliability=entry.get("tool_reliability", "medium"),
                supports_thinking=entry.get("supports_thinking", False),
                preserve_thinking=entry.get("preserve_thinking", False),
                is_embedding=entry.get("is_embedding", False),
                query_template=entry.get("query_template"),
                document_template=entry.get("document_template"),
                max_steps=entry.get("max_steps", {"chat": 3, "plan": 5, "agent": 8}),
                size_rules=size_rules,
            )
            profiles.append(profile)
            matches[profile.family] = list(entry.get("match", [profile.family]))

        return cls(profiles, matches)

    def for_model(self, model: str, *, parameter_count_b: float | None = None) -> ModelProfile:
        """The profile for a model tag, downgraded for a small parameter count.

        A ``size_rules`` entry on the matched profile is preferred when it covers this
        parameter count; otherwise a small, unprofiled model falls back to the blunt
        across-the-board ``tool_reliability`` downgrade. Either way this only ever lowers
        expectations, never raises them, because a family match is real evidence about
        behaviour while a raw parameter count is a weak proxy the family match should
        always be preferred to.
        """
        lowered = model.lower()
        profile = self._match(lowered) or self._default()

        if parameter_count_b is None:
            return profile

        rule = _matching_size_rule(profile.size_rules, parameter_count_b)
        if rule is not None:
            if rule.tool_reliability is not None:
                profile = replace(profile, tool_reliability=rule.tool_reliability)
            if rule.max_steps_agent is not None:
                profile = replace(profile, max_steps={**profile.max_steps, "agent": rule.max_steps_agent})
        elif parameter_count_b < _SMALL_MODEL_PARAMS_B and profile.tool_reliability == "high":
            profile = replace(profile, tool_reliability="medium")

        return profile

    def _match(self, model: str) -> ModelProfile | None:
        best: tuple[int, ModelProfile] | None = None
        for family, patterns in self._matches.items():
            for pattern in patterns:
                if pattern == "*":
                    continue
                if pattern in model and (best is None or len(pattern) > best[0]):
                    best = (len(pattern), self._profiles[family])
        return best[1] if best else None

    def _default(self) -> ModelProfile:
        return self._profiles.get("default") or ModelProfile(family="default")


def _matching_size_rule(rules: tuple[SizeRule, ...], parameter_count_b: float) -> SizeRule | None:
    """The tightest rule whose threshold still covers this parameter count, if any."""
    covering = [rule for rule in rules if parameter_count_b <= rule.max_params_b]
    if not covering:
        return None
    return min(covering, key=lambda rule: rule.max_params_b)
