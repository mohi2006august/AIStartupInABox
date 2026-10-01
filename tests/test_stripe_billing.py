import hashlib
import hmac
import json
import time
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest

from sib.billing import LocalGateway, Subscription, build_invoice
from sib.config import Settings
from sib.stripe_billing import LiveKeyRefused, StripeGateway, make_gateway, stripe_client, to_minor_units

NAMES = {"firm_a": "Aurora Legal LLP"}
SUB = Subscription("firm_a", "team", 5, "referral", date(2026, 7, 1))


class FakeResource:
    def __init__(self, log: list, kind: str, fail_on: int | None = None) -> None:
        self.log, self.kind, self.fail_on, self.calls = log, kind, fail_on, 0

    def create(self, params=None, options=None):
        self.calls += 1
        if self.fail_on == self.calls:
            raise ConnectionError("network blip")
        self.log.append((self.kind, params, options))
        return SimpleNamespace(id=f"{self.kind}_{self.calls}")

    def finalize_invoice(self, invoice, params=None, options=None):
        self.log.append(("finalize", invoice, options))
        return SimpleNamespace(id=invoice, status="open")


class FakeStripe:
    """Records calls in the shape of stripe.StripeClient.v1."""

    def __init__(self, fail_item_on: int | None = None) -> None:
        self.log: list = []
        self.v1 = SimpleNamespace(customers=FakeResource(self.log, "cus"), invoices=FakeResource(self.log, "in"),
                                  invoice_items=FakeResource(self.log, "ii", fail_item_on))


def calls(fake: FakeStripe, kind: str) -> list:
    return [c for c in fake.log if c[0] == kind]


def test_minor_units():
    assert to_minor_units(Decimal("149")) == "14900"
    assert to_minor_units(Decimal("0.10")) == "10"
    assert to_minor_units(Decimal("0.125")) == "12.5"


def test_invoice_becomes_a_draft_stripe_invoice_with_one_item_per_line(tmp_path):
    fake = FakeStripe()
    gw = StripeGateway(fake, tmp_path, NAMES)
    invoice = build_invoice(SUB, 5200, "2026-10")  # 5 seats + 200 overage requests
    ref = gw.submit(invoice)
    (_, customer, cus_opts), = calls(fake, "cus")
    assert customer == {"name": "Aurora Legal LLP", "metadata": {"tenant_id": "firm_a"}}
    assert cus_opts == {"idempotency_key": "sib-customer-firm_a"}
    (_, inv, inv_opts), = calls(fake, "in")
    assert ref == "in_1" and inv["customer"] == "cus_1" and inv["auto_advance"] is False
    assert inv["collection_method"] == "send_invoice" and inv["pending_invoice_items_behavior"] == "exclude"
    assert inv["metadata"]["sib_key"] == "firm_a:2026-10" and inv_opts == {"idempotency_key": "sib-invoice-firm_a:2026-10"}
    items = [c[1] for c in calls(fake, "ii")]
    assert [(i["quantity"], i["unit_amount_decimal"], i["invoice"]) for i in items] == [(5, "14900", "in_1"), (200, "10", "in_1")]
    assert calls(fake, "finalize") == []  # drafts by default
    assert gw.submitted() == ["firm_a:2026-10"]


def test_resubmitting_never_double_bills(tmp_path):
    fake = FakeStripe()
    gw = StripeGateway(fake, tmp_path, NAMES)
    invoice = build_invoice(SUB, 10, "2026-10")
    assert gw.submit(invoice) == gw.submit(invoice)
    assert len(calls(fake, "in")) == 1 and len(calls(fake, "ii")) == 1
    gw.submit(build_invoice(SUB, 10, "2026-11"))  # next month reuses the customer
    assert len(calls(fake, "cus")) == 1 and len(calls(fake, "in")) == 2


def test_a_crash_mid_invoice_resumes_without_duplicates(tmp_path):
    flaky = FakeStripe(fail_item_on=2)
    invoice = build_invoice(SUB, 5200, "2026-10")
    with pytest.raises(ConnectionError):
        StripeGateway(flaky, tmp_path, NAMES).submit(invoice)
    healthy = FakeStripe()
    ref = StripeGateway(healthy, tmp_path, NAMES).submit(invoice)
    assert ref == "in_1"
    assert calls(healthy, "in") == [] and calls(healthy, "cus") == []  # invoice and customer already exist
    assert [c[1]["quantity"] for c in calls(healthy, "ii")] == [200]  # only the missing line


def test_trials_are_skipped_and_finalize_is_opt_in(tmp_path):
    fake = FakeStripe()
    trial = Subscription("firm_t", "team", 3, "ads", date(2026, 9, 1), trial=True)
    assert StripeGateway(fake, tmp_path, NAMES).submit(build_invoice(trial, 900, "2026-10")) == "skipped"
    assert fake.log == []
    StripeGateway(fake, tmp_path, NAMES, finalize=True).submit(build_invoice(SUB, 10, "2026-10"))
    assert calls(fake, "finalize") == [("finalize", "in_1", {"idempotency_key": "sib-finalize-firm_a:2026-10"})]


def test_webhook_events_update_status(tmp_path):
    gw = StripeGateway(FakeStripe(), tmp_path, NAMES)
    gw.submit(build_invoice(SUB, 10, "2026-10"))
    assert gw.apply_event("invoice.paid", {"id": "in_1", "metadata": {"sib_key": "firm_a:2026-10"}})
    assert gw.statuses()["firm_a:2026-10"]["status"] == "paid"
    assert gw.apply_event("invoice.payment_failed", {"id": "in_1", "metadata": {}})  # matched by id
    assert not gw.apply_event("invoice.paid", {"id": "in_unknown", "metadata": {}})
    assert not gw.apply_event("customer.created", {"id": "cus_1"})


def test_gateway_selection_and_live_key_guard(tmp_path):
    assert isinstance(make_gateway(Settings(state_dir=str(tmp_path)), NAMES), LocalGateway)
    assert isinstance(make_gateway(Settings(state_dir=str(tmp_path)), NAMES, client=FakeStripe()), StripeGateway)
    with pytest.raises(LiveKeyRefused):
        stripe_client(Settings(stripe_key="sk_live_abc"))


def _signed(payload: str, secret: str) -> str:
    t = int(time.time())
    sig = hmac.new(secret.encode(), f"{t}.{payload}".encode(), hashlib.sha256).hexdigest()
    return f"t={t},v1={sig}"


def test_webhook_endpoint_verifies_signatures(tmp_path):
    pytest.importorskip("stripe")
    from fastapi.testclient import TestClient

    from sib.api import create_app

    settings = Settings(state_dir=str(tmp_path / "state"), ledger_dir=str(tmp_path / "ledger"),
                        stripe_webhook_secret="whsec_test")
    StripeGateway(FakeStripe(), tmp_path / "state", NAMES).submit(build_invoice(
        Subscription("firm_aurora", "team", 5, "referral", date(2026, 7, 1)), 10, "2026-10"))
    client = TestClient(create_app(settings))
    payload = json.dumps({"id": "evt_1", "object": "event", "type": "invoice.paid",
                          "data": {"object": {"id": "in_1", "object": "invoice", "metadata": {"sib_key": "firm_aurora:2026-10"}}}})
    bad = client.post("/v1/stripe/webhook", content=payload, headers={"Stripe-Signature": _signed(payload, "whsec_wrong")})
    assert bad.status_code == 400
    ok = client.post("/v1/stripe/webhook", content=payload, headers={"Stripe-Signature": _signed(payload, "whsec_test")})
    assert ok.json() == {"received": True, "matched": True}
    status = client.get("/v1/billing/invoices", headers={"X-Tenant-Id": "firm_aurora"}).json()
    assert status["firm_aurora:2026-10"]["status"] == "paid"
    assert client.get("/v1/billing/invoices", headers={"X-Tenant-Id": "firm_birch"}).json() == {}
