"""Reporting.

``build_monthly_statement`` is deliberately oversized — it exceeds the chunker's soft
maximum so the AST split-merge path has something to recurse into. Do not "clean it up";
its length is the point (see tests/fixtures/repos/README.md).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from billing.models import Currency, Invoice, InvoiceStatus


@dataclass
class StatementLine:
    label: str
    count: int
    gross: Decimal
    tax: Decimal
    net: Decimal


@dataclass
class MonthlyStatement:
    period_start: date
    period_end: date
    currency: Currency
    lines: list[StatementLine]
    totals: StatementLine
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "period": [self.period_start.isoformat(), self.period_end.isoformat()],
            "currency": self.currency.value,
            "lines": [vars(line) for line in self.lines],
            "totals": vars(self.totals),
            "warnings": list(self.warnings),
        }


def build_monthly_statement(
    invoices: list[Invoice],
    period_start: date,
    period_end: date,
    currency: Currency = Currency.USD,
    *,
    include_voided: bool = False,
    group_by_country: bool = True,
    warn_on_missing_totals: bool = True,
) -> MonthlyStatement:
    """Aggregate invoices into a monthly statement.

    Long by design: grouping, validation, aggregation and formatting all live here so the
    chunker has a function that must be split rather than emitted whole.
    """
    warnings: list[str] = []
    in_period: list[Invoice] = []

    for invoice in invoices:
        if invoice.issued_on < period_start or invoice.issued_on > period_end:
            continue
        if invoice.currency is not currency:
            warnings.append(
                f"invoice {invoice.id} is {invoice.currency.value}, expected {currency.value}"
            )
            continue
        if invoice.status is InvoiceStatus.VOID and not include_voided:
            continue
        if invoice.status is InvoiceStatus.DRAFT:
            warnings.append(f"invoice {invoice.id} is still a draft and was excluded")
            continue
        in_period.append(invoice)

    grouped: dict[str, list[Invoice]] = defaultdict(list)
    for invoice in in_period:
        key = invoice.country if group_by_country else "all"
        grouped[key].append(invoice)

    lines: list[StatementLine] = []
    running_gross = Decimal("0.00")
    running_tax = Decimal("0.00")
    running_net = Decimal("0.00")

    for label in sorted(grouped):
        bucket = grouped[label]
        gross = Decimal("0.00")
        tax = Decimal("0.00")

        for invoice in bucket:
            if invoice.total is None:
                if warn_on_missing_totals:
                    warnings.append(f"invoice {invoice.id} has no total and was skipped")
                continue
            gross += invoice.total
            tax += invoice.tax or Decimal("0.00")

        net = gross - tax
        lines.append(
            StatementLine(
                label=label,
                count=len(bucket),
                gross=gross,
                tax=tax,
                net=net,
            )
        )
        running_gross += gross
        running_tax += tax
        running_net += net

    totals = StatementLine(
        label="total",
        count=sum(line.count for line in lines),
        gross=running_gross,
        tax=running_tax,
        net=running_net,
    )

    if not lines:
        warnings.append("no invoices matched the requested period")

    return MonthlyStatement(
        period_start=period_start,
        period_end=period_end,
        currency=currency,
        lines=lines,
        totals=totals,
        warnings=warnings,
    )


def format_statement(statement: MonthlyStatement) -> str:
    """Render a statement as fixed-width text."""
    rows = [
        f"Statement {statement.period_start} to {statement.period_end} ({statement.currency.value})",
        "",
        f"{'Group':<16}{'Count':>8}{'Gross':>14}{'Tax':>12}{'Net':>14}",
    ]
    for line in statement.lines:
        rows.append(
            f"{line.label:<16}{line.count:>8}{line.gross:>14}{line.tax:>12}{line.net:>14}"
        )
    rows.append("")
    totals = statement.totals
    rows.append(
        f"{totals.label:<16}{totals.count:>8}{totals.gross:>14}{totals.tax:>12}{totals.net:>14}"
    )
    if statement.warnings:
        rows.append("")
        rows.append("Warnings:")
        rows.extend(f"  - {w}" for w in statement.warnings)
    return "\n".join(rows)
