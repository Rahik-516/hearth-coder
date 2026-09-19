"""Step limits, retry budgets and loop detection.

Small models fail in characteristic ways: they repeat a call that already failed, they
call a tool that does not exist, or they never stop (docs/system-design.md §1.1, §16). Each
gets its own bound, because they need different responses.

* **Step limit** — how much work a turn may do. Ends the turn with a partial answer.
* **Retry budget** — how many *corrective* failures are tolerated. Separate from the step
  limit so a model that is making progress is not punished for one mistyped argument,
  while one that cannot get a schema right does not burn forty steps trying.
* **Loop detection** — identical calls repeating. Nudged once, then stopped, because a
  model repeating itself will keep repeating itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: Identical calls tolerated before nudging the model.
LOOP_NUDGE_THRESHOLD = 2
#: Identical calls tolerated before ending the turn.
LOOP_STOP_THRESHOLD = 3
#: How many recent calls are remembered for loop detection.
_HISTORY_WINDOW = 12


@dataclass
class TurnLimits:
    """Per-turn bounds, taken from the model profile and mode."""

    max_steps: int = 6
    max_retries: int = 3
    max_consecutive_failures: int = 3

    @classmethod
    def from_profile(cls, profile: Any, mode: str) -> TurnLimits:
        """Derive limits from a model profile.

        Autonomy scales with measured reliability: a profile marked ``medium`` gets fewer
        retries, because its failures are more often systematic than incidental.
        """
        max_steps = profile.max_steps_for(mode) if profile is not None else 6
        reliability = getattr(profile, "tool_reliability", "medium")

        retries = {"high": 4, "medium": 3, "low": 2}.get(reliability, 3)
        return cls(max_steps=max_steps, max_retries=retries, max_consecutive_failures=retries)


@dataclass
class StepTracker:
    """Tracks one turn's progress against its limits."""

    limits: TurnLimits = field(default_factory=TurnLimits)
    steps: int = 0
    retries: int = 0
    consecutive_failures: int = 0
    #: Fingerprints of recent calls, for loop detection.
    _recent: list[str] = field(default_factory=list, init=False)
    _nudged: set[str] = field(default_factory=set, init=False)

    def begin_step(self) -> None:
        self.steps += 1

    @property
    def steps_exhausted(self) -> bool:
        return self.steps >= self.limits.max_steps

    @property
    def retries_exhausted(self) -> bool:
        return self.retries >= self.limits.max_retries

    @property
    def failing_repeatedly(self) -> bool:
        return self.consecutive_failures >= self.limits.max_consecutive_failures

    def record_result(self, *, ok: bool, retryable: bool) -> None:
        """Record a tool outcome.

        Only *retryable* failures consume the retry budget. A denial is a decision, and
        spending retries on it would let policy refusals end a turn early.
        """
        if ok:
            self.consecutive_failures = 0
            return

        self.consecutive_failures += 1
        if retryable:
            self.retries += 1

    def observe_call(self, tool: str, arguments: Mapping[str, Any]) -> str | None:
        """Record a call and return a nudge message when it is repeating.

        Returns None while the call is novel, a nudge on the second repeat, and a stop
        signal on the third — escalating rather than cutting off immediately, because a
        model sometimes repeats once and then corrects itself.
        """
        fingerprint = _fingerprint(tool, arguments)
        self._recent.append(fingerprint)
        del self._recent[:-_HISTORY_WINDOW]

        occurrences = self._recent.count(fingerprint)

        if occurrences >= LOOP_STOP_THRESHOLD:
            return (
                f"STOP: {tool} has been called with identical arguments {occurrences} times "
                f"and returned the same result each time. Answer with what you have, or "
                f"explain what is missing."
            )
        if occurrences >= LOOP_NUDGE_THRESHOLD and fingerprint not in self._nudged:
            self._nudged.add(fingerprint)
            return (
                f"You already called {tool} with these arguments and got this result. "
                f"Try different arguments, a different tool, or answer with what you have."
            )
        return None

    def is_looping(self, tool: str, arguments: Mapping[str, Any]) -> bool:
        """Whether this exact call has repeated past the stop threshold."""
        return self._recent.count(_fingerprint(tool, arguments)) >= LOOP_STOP_THRESHOLD

    def stop_reason(self) -> str | None:
        """Why the turn should end, or None to continue."""
        if self.steps_exhausted:
            return "step_limit"
        if self.retries_exhausted:
            return "retry_budget"
        if self.failing_repeatedly:
            return "repeated_failures"
        return None


def _fingerprint(tool: str, arguments: Mapping[str, Any]) -> str:
    """Stable identity for a call.

    Arguments are serialized with sorted keys so that the same call written two ways is
    recognised as one — a model re-emitting a call rarely reproduces key order exactly.
    """
    try:
        payload = json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(sorted(arguments.items()))
    return hashlib.blake2b(f"{tool}:{payload}".encode(), digest_size=8).hexdigest()
