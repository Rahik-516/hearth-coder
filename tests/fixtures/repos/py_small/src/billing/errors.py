"""Billing domain exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from billing.models import InvoiceStatus


class BillingError(Exception):
    """Base class for billing failures."""


class InvoiceNotFound(BillingError):
    def __init__(self, invoice_id: UUID) -> None:
        self.invoice_id = invoice_id
        super().__init__(f"invoice {invoice_id} not found")


class InvoiceAlreadyFinalized(BillingError):
    def __init__(self, invoice_id: UUID, status: InvoiceStatus) -> None:
        self.invoice_id = invoice_id
        self.status = status
        super().__init__(f"invoice {invoice_id} is {status.value}, not a draft")


class LineItemError(BillingError):
    """The invoice's line items are unusable."""


class PaymentDeclined(BillingError):
    def __init__(self, invoice_id: UUID, reason: str) -> None:
        self.invoice_id = invoice_id
        self.reason = reason
        super().__init__(f"payment for {invoice_id} declined: {reason}")


class RateLimited(BillingError):
    """Outbound gateway calls are being throttled."""
