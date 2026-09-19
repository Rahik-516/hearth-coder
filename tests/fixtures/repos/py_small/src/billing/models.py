"""Domain models for the billing fixture."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal
from enum import Enum
from uuid import UUID


class InvoiceStatus(str, Enum):
    DRAFT = "draft"
    FINALIZED = "finalized"
    PAID = "paid"
    VOID = "void"


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


@dataclass(frozen=True)
class LineItem:
    """A single billable line."""

    description: str
    quantity: int
    unit_price: Decimal

    @property
    def amount(self) -> Decimal:
        return self.unit_price * self.quantity

    def rename(self, description: str) -> LineItem:
        return replace(self, description=description)


@dataclass(frozen=True)
class Customer:
    id: UUID
    name: str
    email: str
    country: str = "US"


@dataclass(frozen=True)
class Invoice:
    """An invoice and its immutable state transitions."""

    id: UUID
    customer: Customer
    issued_on: date
    currency: Currency = Currency.USD
    status: InvoiceStatus = InvoiceStatus.DRAFT
    lines: list[LineItem] = field(default_factory=list)
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal | None = None
    void_reason: str | None = None

    @property
    def country(self) -> str:
        return self.customer.country

    @property
    def is_editable(self) -> bool:
        return self.status is InvoiceStatus.DRAFT

    def with_totals(self, *, subtotal: Decimal, tax: Decimal, total: Decimal) -> Invoice:
        return replace(
            self,
            subtotal=subtotal,
            tax=tax,
            total=total,
            status=InvoiceStatus.FINALIZED,
        )

    def voided(self, reason: str) -> Invoice:
        return replace(self, status=InvoiceStatus.VOID, void_reason=reason)

    def add_line(self, line: LineItem) -> Invoice:
        if not self.is_editable:
            raise ValueError("cannot add lines to a non-draft invoice")
        return replace(self, lines=[*self.lines, line])
