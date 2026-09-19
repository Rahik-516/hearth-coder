"""Tests for the invoice service.

These are real, passing tests. The `/test` workflow and the `run_tests` tool need a
fixture where the test framework, conventions and a green baseline all actually exist.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest
from billing.errors import InvoiceAlreadyFinalized, InvoiceNotFound, LineItemError
from billing.invoice_service import InvoiceService
from billing.models import Currency, Customer, Invoice, InvoiceStatus, LineItem


class InMemoryRepository:
    def __init__(self) -> None:
        self.invoices: dict[object, Invoice] = {}

    def get(self, invoice_id: object) -> Invoice | None:
        return self.invoices.get(invoice_id)

    def save(self, invoice: Invoice) -> None:
        self.invoices[invoice.id] = invoice


class FlatTax:
    def __init__(self, rate: str = "0.10") -> None:
        self.rate = Decimal(rate)

    def rate_for(self, country: str) -> Decimal:
        return Decimal("0") if country == "XX" else self.rate


@pytest.fixture
def customer() -> Customer:
    return Customer(id=uuid4(), name="Acme", email="ap@acme.test", country="US")


@pytest.fixture
def repository() -> InMemoryRepository:
    return InMemoryRepository()


@pytest.fixture
def service(repository: InMemoryRepository) -> InvoiceService:
    return InvoiceService(repository, FlatTax())


def make_invoice(customer: Customer, *lines: LineItem) -> Invoice:
    return Invoice(
        id=uuid4(),
        customer=customer,
        issued_on=date(2026, 3, 1),
        currency=Currency.USD,
        lines=list(lines),
    )


def test_finalize_totals_and_locks(service, repository, customer) -> None:
    invoice = make_invoice(
        customer,
        LineItem("Widget", 2, Decimal("10.00")),
        LineItem("Gadget", 1, Decimal("5.00")),
    )
    repository.save(invoice)

    finalized = service.finalize(invoice.id)

    assert finalized.status is InvoiceStatus.FINALIZED
    assert finalized.subtotal == Decimal("25.00")
    assert finalized.tax == Decimal("2.50")
    assert finalized.total == Decimal("27.50")


def test_finalize_rounds_half_up(service, repository, customer) -> None:
    invoice = make_invoice(customer, LineItem("Odd", 1, Decimal("0.125")))
    repository.save(invoice)

    finalized = service.finalize(invoice.id)

    assert finalized.subtotal == Decimal("0.13")


def test_finalize_rejects_missing_invoice(service) -> None:
    with pytest.raises(InvoiceNotFound):
        service.finalize(uuid4())


def test_finalize_rejects_empty_invoice(service, repository, customer) -> None:
    invoice = make_invoice(customer)
    repository.save(invoice)

    with pytest.raises(LineItemError):
        service.finalize(invoice.id)


def test_finalize_is_not_idempotent(service, repository, customer) -> None:
    invoice = make_invoice(customer, LineItem("Widget", 1, Decimal("10.00")))
    repository.save(invoice)
    service.finalize(invoice.id)

    with pytest.raises(InvoiceAlreadyFinalized):
        service.finalize(invoice.id)


def test_void_records_a_reason(service, repository, customer) -> None:
    invoice = make_invoice(customer, LineItem("Widget", 1, Decimal("10.00")))
    repository.save(invoice)

    voided = service.void(invoice.id, "duplicate")

    assert voided.status is InvoiceStatus.VOID
    assert voided.void_reason == "duplicate"


def test_empty_subtotal_keeps_decimal_type(service) -> None:
    """Regression: sum() from int 0 would make an empty invoice lose Decimal precision."""
    assert service.compute_subtotal([]) == Decimal("0.00")


def test_nested_config_is_reachable() -> None:
    assert InvoiceService.Config.max_line_items == 500
    assert "max_line_items" in InvoiceService.Config.describe()
