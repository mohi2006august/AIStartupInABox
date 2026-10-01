"""Stripe billing gateway (FR-03): push locally computed invoices to Stripe.

Pricing stays in `sib.billing` (seats plus overage, computed from the meter). This gateway turns
each `Invoice` into a Stripe invoice:

1. find or create the firm's Stripe customer (tenant id kept in customer metadata);
2. create a draft invoice (`collection_method=send_invoice`, `auto_advance=false`);
3. add one invoice item per line, attached to that draft;
4. finalize only when asked. By default invoices stay drafts for a human to review.

Every call carries an idempotency key, and progress is saved locally after each step, so a re-run
after a crash resumes instead of double-billing, even after Stripe's 24-hour key window.
Live-mode keys are refused unless explicitly allowed.
"""
from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

from sib.billing import BillingGateway, Invoice, LocalGateway
from sib.config import Settings

log = logging.getLogger("sib.stripe")

PAYMENT_EVENTS = {
    "invoice.finalized": "open",
    "invoice.paid": "paid",
    "invoice.payment_failed": "payment_failed",
    "invoice.voided": "void",
    "invoice.marked_uncollectible": "uncollectible",
}


class LiveKeyRefused(RuntimeError):
    pass


def to_minor_units(amount: Decimal) -> str:
    """Dollars to cents as a plain decimal string (Stripe's `unit_amount_decimal`)."""
    cents = (amount * 100).normalize()
    return format(cents, "f")


class _JsonStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


class StripeGateway:
    """`client` is a `stripe.StripeClient` (or anything with the same `v1` surface)."""

    def __init__(self, client: Any, state_dir: Path, customer_names: dict[str, str], *, currency: str = "usd",
                 days_until_due: int = 30, finalize: bool = False) -> None:
        self.client = client
        self.customer_names = customer_names
        self.currency, self.days_until_due, self.finalize = currency, days_until_due, finalize
        self._customers = _JsonStore(state_dir / "stripe_customers.json")
        self._invoices = _JsonStore(state_dir / "stripe_invoices.json")

    def _customer_id(self, tenant_id: str) -> str:
        customers = self._customers.load()
        if tenant_id not in customers:
            customer = self.client.v1.customers.create(
                params={"name": self.customer_names.get(tenant_id, tenant_id), "metadata": {"tenant_id": tenant_id}},
                options={"idempotency_key": f"sib-customer-{tenant_id}"},
            )
            customers[tenant_id] = customer.id
            self._customers.save(customers)
            log.info("stripe_customer_created", extra={"tenant": tenant_id, "customer": customer.id})
        return customers[tenant_id]

    def submit(self, invoice: Invoice) -> str:
        key = invoice.idempotency_key
        records = self._invoices.load()
        record = records.get(key)
        if record and record.get("complete"):
            return record["id"]
        if invoice.total <= 0:  # trials and empty months: nothing to collect, nothing to send
            return "skipped"
        customer = self._customer_id(invoice.tenant_id)
        if record is None:
            created = self.client.v1.invoices.create(
                params={
                    "customer": customer,
                    "collection_method": "send_invoice",
                    "days_until_due": self.days_until_due,
                    "auto_advance": False,
                    "pending_invoice_items_behavior": "exclude",
                    "currency": self.currency,
                    "description": f"AI research assistant, {invoice.period}",
                    "metadata": {"tenant_id": invoice.tenant_id, "period": invoice.period, "sib_key": key},
                },
                options={"idempotency_key": f"sib-invoice-{key}"},
            )
            record = {"id": created.id, "customer": customer, "items_added": 0, "status": "draft", "complete": False,
                      "total": str(invoice.total)}
            records[key] = record
            self._invoices.save(records)
        for index in range(record["items_added"], len(invoice.lines)):
            line = invoice.lines[index]
            self.client.v1.invoice_items.create(
                params={
                    "customer": customer,
                    "invoice": record["id"],
                    "currency": self.currency,
                    "description": line.description,
                    "quantity": line.quantity,
                    "unit_amount_decimal": to_minor_units(line.unit_price),
                },
                options={"idempotency_key": f"sib-item-{key}-{index}"},
            )
            record["items_added"] = index + 1
            self._invoices.save(records)
        if self.finalize and record["status"] == "draft":
            self.client.v1.invoices.finalize_invoice(record["id"], options={"idempotency_key": f"sib-finalize-{key}"})
            record["status"] = "open"
        record["complete"] = True
        self._invoices.save(records)
        log.info("stripe_invoice_submitted", extra={"key": key, "invoice": record["id"], "status": record["status"]})
        return record["id"]

    def submitted(self) -> list[str]:
        return sorted(k for k, r in self._invoices.load().items() if r.get("complete"))

    def statuses(self) -> dict[str, dict[str, Any]]:
        return self._invoices.load()

    def apply_event(self, event_type: str, invoice_obj: dict[str, Any]) -> bool:
        """Record a payment-status webhook against the matching local invoice. Returns True if matched."""
        status = PAYMENT_EVENTS.get(event_type)
        if status is None:
            return False
        records = self._invoices.load()
        key = (invoice_obj.get("metadata") or {}).get("sib_key")
        if key not in records:
            key = next((k for k, r in records.items() if r["id"] == invoice_obj.get("id")), None)
        if key is None:
            return False
        records[key]["status"] = status
        self._invoices.save(records)
        return True


def is_live_key(key: str) -> bool:
    return key.startswith(("sk_live_", "rk_live_"))


def stripe_client(settings: Settings) -> Any:
    import stripe  # optional dependency: pip install -e ".[stripe]"

    if is_live_key(settings.stripe_key) and not settings.stripe_allow_live:
        raise LiveKeyRefused("refusing a live Stripe key; set STRIPE_ALLOW_LIVE=true to bill real customers")
    return stripe.StripeClient(settings.stripe_key)


def make_gateway(settings: Settings, customer_names: dict[str, str], client: Any | None = None) -> BillingGateway:
    """Stripe when STRIPE_KEY is set, otherwise the local JSON gateway."""
    state_dir = Path(settings.state_dir)
    if not settings.stripe_key and client is None:
        return LocalGateway(state_dir / "invoices")
    return StripeGateway(client or stripe_client(settings), state_dir, customer_names,
                         currency=settings.billing_currency, finalize=settings.stripe_finalize)
