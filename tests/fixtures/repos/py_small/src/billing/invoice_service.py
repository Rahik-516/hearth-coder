"""Invoice lifecycle.

This module is the primary retrieval target in the fixture suite. The M1 acceptance
criterion asks that `hearth search "InvoiceService finalize"` return the definition of
``InvoiceService.finalize`` first, so keep that symbol name stable.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from functools import wraps
from typing import Any, Callable, Protocol
from uuid import UUID

from billing.errors import (
    InvoiceAlreadyFinalized,
    InvoiceNotFound,
    LineItemError,
)
from billing.models import Invoice, InvoiceStatus, LineItem

logger = logging.getLogger(__name__)

CENTS = Decimal("0.01")


def audited(action: str) -> Callable[..., Any]:
    """Decorator recording an audit line around a mutating operation.

    Present so the chunker has a decorated method to attach to its definition.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            logger.info("begin %s", action)
            try:
                result = func(self, *args, **kwargs)
            except Exception:
                logger.exception("failed %s", action)
                raise
            logger.info("completed %s", action)
            return result

        return wrapper

    return decorator


class InvoiceRepository(Protocol):
    """Storage seam, so the service can be tested without a database."""

    def get(self, invoice_id: UUID) -> Invoice | None: ...

    def save(self, invoice: Invoice) -> None: ...


class TaxPolicy(Protocol):
    def rate_for(self, country: str) -> Decimal: ...


@dataclass(frozen=True)
class FinalizationResult:
    """What finalization produced, for callers that want the detail."""

    invoice: Invoice
    subtotal: Decimal
    tax: Decimal
    total: Decimal


class InvoiceService:
    """Coordinates invoice state transitions.

    Contains a nested class and several decorated methods on purpose: the chunker tests
    assert that nested definitions and their decorators stay attached.
    """

    class Config:
        """Nested configuration class — chunker coverage for nested definitions."""

        round_half_up = True
        max_line_items = 500
        allow_zero_total = False

        @classmethod
        def describe(cls) -> str:
            return f"max_line_items={cls.max_line_items}"

    def __init__(self, repository: InvoiceRepository, tax_policy: TaxPolicy) -> None:
        self._repository = repository
        self._tax_policy = tax_policy

    @audited("finalize")
    def finalize(self, invoice_id: UUID) -> Invoice:
        """Finalize an invoice: total it, apply tax, and lock it against further edits.

        Raises:
            InvoiceNotFound: no invoice with that id.
            InvoiceAlreadyFinalized: the invoice is not a draft.
            LineItemError: the invoice has no lines, or too many.
        """
        invoice = self._repository.get(invoice_id)
        if invoice is None:
            raise InvoiceNotFound(invoice_id)
        if invoice.status is not InvoiceStatus.DRAFT:
            raise InvoiceAlreadyFinalized(invoice_id, invoice.status)

        self._validate_lines(invoice.lines)

        subtotal = self.compute_subtotal(invoice.lines)
        tax = self.compute_tax(subtotal, invoice.country)
        total = self._quantize(subtotal + tax)

        if total <= Decimal("0") and not self.Config.allow_zero_total:
            raise LineItemError("finalized invoice must have a positive total")

        finalized = invoice.with_totals(subtotal=subtotal, tax=tax, total=total)
        self._repository.save(finalized)
        return finalized

    def compute_subtotal(self, lines: Iterable[LineItem]) -> Decimal:
        """Sum line amounts.

        Starts from an explicit Decimal so an empty invoice yields Decimal("0.00")
        rather than an int, which would lose precision downstream.
        """
        total = sum((line.amount for line in lines), Decimal("0"))
        return self._quantize(total)

    def compute_tax(self, subtotal: Decimal, country: str) -> Decimal:
        rate = self._tax_policy.rate_for(country)
        return self._quantize(subtotal * rate)

    @audited("void")
    def void(self, invoice_id: UUID, reason: str) -> Invoice:
        invoice = self._repository.get(invoice_id)
        if invoice is None:
            raise InvoiceNotFound(invoice_id)

        voided = invoice.voided(reason)
        self._repository.save(voided)
        return voided

    def _validate_lines(self, lines: list[LineItem]) -> None:
        if not lines:
            raise LineItemError("invoice has no line items")
        if len(lines) > self.Config.max_line_items:
            raise LineItemError(f"invoice exceeds {self.Config.max_line_items} line items")

    @staticmethod
    def _quantize(value: Decimal) -> Decimal:
        return value.quantize(CENTS, rounding=ROUND_HALF_UP)
