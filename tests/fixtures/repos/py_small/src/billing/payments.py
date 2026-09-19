"""Payment attempts, retries and rate limiting.

``TokenBucket`` exists deliberately: it is the fixture's vocabulary-gap case. A user asking
"where do we throttle API calls?" must find this class, even though the word "throttle"
never appears in its name (docs/system-design.md §6.9).
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, TypeVar
from uuid import UUID

from billing.errors import PaymentDeclined, RateLimited

T = TypeVar("T")


class TokenBucket:
    """Rate limiter: throttles outbound calls to the payment gateway.

    Refills continuously at ``rate`` tokens per second up to ``capacity``.
    """

    def __init__(self, capacity: int, rate: float) -> None:
        self.capacity = capacity
        self.rate = rate
        self._tokens = float(capacity)
        self._last = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last = now

    def try_acquire(self, tokens: int = 1) -> bool:
        """Take tokens if available. Returns False instead of blocking."""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def acquire(self, tokens: int = 1) -> None:
        """Take tokens or raise, so callers surface throttling rather than stalling."""
        if not self.try_acquire(tokens):
            raise RateLimited(f"rate limit exceeded; {self._tokens:.1f} tokens available")


@dataclass(frozen=True)
class PaymentResult:
    invoice_id: UUID
    amount: Decimal
    reference: str
    succeeded: bool


def with_retries(
    attempts: int = 3,
    base_delay: float = 0.1,
    jitter: float = 0.05,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry with exponential backoff and jitter.

    Only retries ``RateLimited``. A declined payment is a business outcome, not a
    transient fault, and retrying it would double-charge.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Exception | None = None
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except RateLimited as exc:
                    last_error = exc
                    delay = base_delay * (2**attempt) + random.uniform(0, jitter)
                    time.sleep(delay)
            raise last_error if last_error else RuntimeError("retry loop exited unexpectedly")

        return wrapper

    return decorator


class PaymentGateway:
    """Charges invoices, subject to throttling."""

    def __init__(self, bucket: TokenBucket | None = None) -> None:
        self._bucket = bucket or TokenBucket(capacity=10, rate=5.0)
        self._charges: dict[UUID, PaymentResult] = {}

    @with_retries(attempts=3)
    def charge(self, invoice_id: UUID, amount: Decimal, token: str) -> PaymentResult:
        self._bucket.acquire()

        if amount <= Decimal("0"):
            raise PaymentDeclined(invoice_id, "amount must be positive")
        if token.startswith("decline_"):
            raise PaymentDeclined(invoice_id, "card declined")

        result = PaymentResult(
            invoice_id=invoice_id,
            amount=amount,
            reference=f"ch_{invoice_id.hex[:12]}",
            succeeded=True,
        )
        self._charges[invoice_id] = result
        return result

    def lookup(self, invoice_id: UUID) -> PaymentResult | None:
        return self._charges.get(invoice_id)
