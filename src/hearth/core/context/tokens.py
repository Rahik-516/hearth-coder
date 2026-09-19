"""Token estimation, calibrated online.

Hearth ships no tokenizer. Every model family tokenizes differently, bundling one per
family would add weight and still be wrong for the next model, and the estimate only needs
to be good enough to *budget* — a few percent of error costs a few hundred tokens of
headroom, not correctness.

Instead: a fast character-based estimate, corrected by an exponential moving average of
``actual / estimated`` using the prompt token counts Ollama reports on every response
(docs/system-design.md §5.3). It converges within a few requests and adapts to whatever
model is loaded.

The estimator is also the truncation detector. If the server reports far *fewer* prompt
tokens than were sent, the prompt was silently cut — the failure mode that quietly removes
the system prompt or the tool definitions (docs/system-design.md §1.1, §16).
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Starting characters-per-token. Code is denser than prose: identifiers, punctuation and
#: indentation all tokenize finely. Calibration corrects this within a few requests.
DEFAULT_CHARS_PER_TOKEN = 3.6

#: Weight of each new observation in the moving average. Low enough to ride out one odd
#: request, high enough to converge in a handful.
DEFAULT_SMOOTHING = 0.25

#: Calibration is ignored outside this band. A ratio far from 1.0 is much more likely to
#: mean a truncated prompt than a genuinely mis-calibrated estimator, and folding that into
#: the average would corrupt every later estimate.
_MIN_RATIO = 0.5
_MAX_RATIO = 2.0

#: Reported prompt tokens below this fraction of the estimate mean the prompt was cut.
TRUNCATION_THRESHOLD = 0.80

#: Per-message overhead: role markers and chat-template scaffolding.
_MESSAGE_OVERHEAD_TOKENS = 4


@dataclass
class TokenEstimator:
    """Estimates token counts, correcting itself from server-reported totals."""

    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
    smoothing: float = DEFAULT_SMOOTHING
    observations: int = 0
    #: Every accepted ratio, for `/context` and for debugging a drifting estimate.
    history: list[float] = field(default_factory=list)

    def estimate(self, text: str) -> int:
        """Estimated tokens for a string. Always at least 1 for non-empty input."""
        if not text:
            return 0
        return max(1, round(len(text) / self.chars_per_token))

    def estimate_messages(self, messages: list[tuple[str, str]]) -> int:
        """Estimated tokens for ``(role, content)`` pairs, including per-message overhead."""
        total = 0
        for role, content in messages:
            total += self.estimate(content) + self.estimate(role) + _MESSAGE_OVERHEAD_TOKENS
        return total

    def calibrate(self, *, estimated: int, actual: int) -> bool:
        """Fold one observation into the moving average. Returns whether it was accepted.

        Rejects implausible ratios rather than absorbing them: a prompt that was truncated
        server-side reports a much smaller count, and treating that as calibration data
        would make every subsequent estimate worse.
        """
        if estimated <= 0 or actual <= 0:
            return False

        ratio = actual / estimated
        if not (_MIN_RATIO <= ratio <= _MAX_RATIO):
            return False

        # chars_per_token scales inversely with the ratio: if the server counted more
        # tokens than estimated, each token covers fewer characters.
        observed_cpt = self.chars_per_token / ratio
        if self.observations == 0:
            self.chars_per_token = observed_cpt
        else:
            self.chars_per_token = (1 - self.smoothing) * self.chars_per_token + self.smoothing * observed_cpt

        self.observations += 1
        self.history.append(ratio)
        return True

    def looks_truncated(self, *, estimated: int, actual: int) -> bool:
        """Whether a reported prompt size indicates the prompt was cut.

        Only flags *under*-reporting. Over-reporting means the estimate was low, which is a
        calibration matter; under-reporting by this much means content did not arrive.
        """
        if estimated <= 0 or actual <= 0:
            return False
        return actual < estimated * TRUNCATION_THRESHOLD

    def reset(self) -> None:
        """Return to the uncalibrated default. Used when the model changes."""
        self.chars_per_token = DEFAULT_CHARS_PER_TOKEN
        self.observations = 0
        self.history.clear()

    @property
    def is_calibrated(self) -> bool:
        return self.observations > 0

    @property
    def mean_ratio(self) -> float | None:
        """Mean accepted ratio, or None before any observation."""
        if not self.history:
            return None
        return sum(self.history) / len(self.history)


@dataclass(frozen=True)
class TruncationReport:
    """The outcome of comparing an estimate against a server-reported count."""

    estimated: int
    actual: int
    truncated: bool
    calibrated: bool

    @property
    def shortfall(self) -> int:
        return max(0, self.estimated - self.actual)

    def message(self) -> str:
        return (
            f"Ollama reported {self.actual} prompt tokens for an estimated {self.estimated}. "
            f"About {self.shortfall} tokens did not arrive — the prompt was truncated, which "
            f"can silently drop the system prompt. Lower models.num_ctx or reduce the "
            f"retrieval budget."
        )


def observe(estimator: TokenEstimator, *, estimated: int, actual: int) -> TruncationReport:
    """Record one request's outcome: calibrate, or flag truncation."""
    truncated = estimator.looks_truncated(estimated=estimated, actual=actual)
    calibrated = False if truncated else estimator.calibrate(estimated=estimated, actual=actual)
    return TruncationReport(
        estimated=estimated,
        actual=actual,
        truncated=truncated,
        calibrated=calibrated,
    )
