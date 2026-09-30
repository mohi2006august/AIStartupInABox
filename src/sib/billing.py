"""Billing (FR-03): seat + usage pricing turned into invoices.

Invoices are computed locally and handed to a `BillingGateway`. `LocalGateway` stores them as
JSON with an idempotency key; a Stripe gateway plugs in behind the same two-method protocol.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Plan:
    name: str
    seat_price: Decimal
    included_requests_per_seat: int
    overage_per_request: Decimal


PLANS = {"team": Plan("team", Decimal("149"), 1000, Decimal("0.10"))}


@dataclass
class Subscription:
    tenant_id: str
    plan: str
    seats: int
    channel: str
    started: date
    trial: bool = False


@dataclass
class LineItem:
    description: str
    quantity: int
    unit_price: Decimal

    @property
    def amount(self) -> Decimal:
        return self.quantity * self.unit_price


@dataclass
class Invoice:
    tenant_id: str
    period: str  # YYYY-MM
    lines: list[LineItem] = field(default_factory=list)

    @property
    def total(self) -> Decimal:
        return sum((line.amount for line in self.lines), Decimal(0))

    @property
    def idempotency_key(self) -> str:
        return f"{self.tenant_id}:{self.period}"


def build_invoice(sub: Subscription, requests: int, period: str) -> Invoice:
    plan = PLANS[sub.plan]
    invoice = Invoice(sub.tenant_id, period)
    if sub.trial:
        invoice.lines.append(LineItem("Trial (usage not billed)", sub.seats, Decimal(0)))
        return invoice
    invoice.lines.append(LineItem(f"{plan.name} plan seats", sub.seats, plan.seat_price))
    overage = max(requests - sub.seats * plan.included_requests_per_seat, 0)
    if overage:
        invoice.lines.append(LineItem("Requests above included allowance", overage, plan.overage_per_request))
    return invoice


class BillingGateway(Protocol):
    def submit(self, invoice: Invoice) -> str: ...

    def submitted(self) -> list[str]: ...


class LocalGateway:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, key: str) -> Path:
        return self.directory / (key.replace(":", "_") + ".json")

    def submit(self, invoice: Invoice) -> str:
        """Idempotent: re-submitting the same tenant and period never bills twice."""
        path = self._path(invoice.idempotency_key)
        if not path.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
            payload = {"tenant_id": invoice.tenant_id, "period": invoice.period, "total": str(invoice.total),
                       "lines": [{**asdict(line), "unit_price": str(line.unit_price)} for line in invoice.lines]}
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return invoice.idempotency_key

    def submitted(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json")) if self.directory.exists() else []
