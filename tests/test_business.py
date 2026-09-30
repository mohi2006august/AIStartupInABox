"""Economics calculations checked against hand-computed fixtures."""
from datetime import date
from decimal import Decimal

import pytest

from sib.billing import LocalGateway, Subscription, build_invoice
from sib.economics import (
    Channel, Scenario, UnitCosts, channel_economics, default_alive, gross_margin, runway_delta_months,
    runway_months, simulate, summary,
)
from sib.metering import Meter, UsageEvent, percentile


def scenario(**overrides) -> Scenario:
    base = dict(
        cash=Decimal("100000"), monthly_opex=Decimal("20000"), customers=10, seats_per_account=5,
        seat_price=Decimal("100"), requests_per_seat_per_month=200, draft_share=0.0,
        other_cogs_per_account=Decimal("50"), monthly_churn=0.02, trial_requests=100,
        channels=[Channel(name="ads", monthly_spend=Decimal("3000"), new_customers_per_month=1, trials_per_month=4),
                  Channel(name="referral", monthly_spend=Decimal("500"), new_customers_per_month=1, trials_per_month=2)],
    )
    return Scenario(**{**base, **overrides})


UNIT = UnitCosts(ask=Decimal("0.10"), draft=Decimal("0.50"))


def test_gross_margin_fixture():
    # ARPA 500; AI COGS 1000 req * 0.10 = 100; other 50 -> (500 - 150) / 500
    assert gross_margin(scenario(), UNIT) == pytest.approx(0.70)


def test_cac_includes_trial_ai_cogs_and_payback_uses_contribution():
    ads, referral = channel_economics(scenario(), UNIT)
    # ads: 3000 spend + 4 trials * 100 req * 0.10 = 3040; contribution 350/month
    assert ads.cac == Decimal("3040") and ads.trial_ai_cogs_per_customer == Decimal("40")
    assert ads.payback_months == pytest.approx(3040 / 350)
    assert referral.cac == Decimal("520") and referral.payback_months < ads.payback_months
    assert ads.ltv_to_cac == pytest.approx((350 / 0.02) / 3040)


def test_cheaper_requests_shorten_payback_and_extend_runway():
    cheap = UnitCosts(ask=Decimal("0.02"), draft=Decimal("0.50"))
    s = scenario()
    assert channel_economics(s, cheap)[0].payback_months < channel_economics(s, UNIT)[0].payback_months
    assert runway_delta_months(s, UNIT, cheap) >= 0
    assert runway_delta_months(s, UNIT, UNIT) == 0


def test_first_month_burn_fixture():
    row = simulate(scenario(), UNIT)[0]
    # revenue 5000; COGS 10 * 150 = 1500; opex 20000; acquisition 3500; trial COGS 6 * 100 * 0.10 = 60
    assert row.mrr == Decimal("5000")
    assert row.gross_burn == Decimal("25060") and row.net_burn == Decimal("20060")
    assert row.cash == Decimal("79940")


def test_runway_and_default_alive():
    dying = simulate(scenario(), UNIT)
    assert runway_months(dying) is not None and not default_alive(dying)
    healthy = simulate(scenario(monthly_opex=Decimal("3000"), cash=Decimal("200000")), UNIT)
    assert runway_months(healthy) is None and default_alive(healthy)


def test_fixed_gpu_cost_is_spread_over_customers():
    with_gpu = UnitCosts(ask=Decimal("0.10"), draft=Decimal("0.50"), fixed_monthly=Decimal("1000"))
    assert gross_margin(scenario(), with_gpu) == pytest.approx(0.70 - 100 / 500)


def test_summary_flags_margin_below_target():
    out = summary(scenario(), UNIT, margin_target=0.75)
    assert out["margin_alert"] is True and out["gross_margin"] == pytest.approx(0.70)
    assert len(out["cash_curve"]) == 36 and out["channels"][0]["name"] == "ads"


def test_invoice_seats_overage_and_trial():
    sub = Subscription("firm_a", "team", 5, "referral", date(2026, 9, 1))
    assert build_invoice(sub, 4000, "2026-09").total == Decimal("745")
    over = build_invoice(sub, 5200, "2026-09")
    assert over.total == Decimal("745") + 200 * Decimal("0.10")
    trial = Subscription("firm_t", "team", 3, "ads", date(2026, 9, 1), trial=True)
    assert build_invoice(trial, 900, "2026-09").total == 0


def test_gateway_is_idempotent(tmp_path):
    gateway = LocalGateway(tmp_path / "invoices")
    invoice = build_invoice(Subscription("firm_a", "team", 5, "referral", date(2026, 9, 1)), 10, "2026-09")
    assert gateway.submit(invoice) == gateway.submit(invoice)
    assert gateway.submitted() == ["firm_a_2026-09"]


def test_meter_aggregates_and_round_trips(tmp_path):
    meter = Meter()
    meter.record(UsageEvent(tenant_id="a", feature="ask", model="claude-opus-5-5", cost_usd=Decimal("0.02"), latency_ms=900))
    meter.record(UsageEvent(tenant_id="a", feature="ask", model="sib-8b-awq", cost_usd=Decimal("0.002"), latency_ms=300))
    meter.record(UsageEvent(tenant_id="b", feature="ask", model="claude-opus-5-5", cache_layer="exact", latency_ms=20, trial=True))
    assert meter.cost_by("tenant_id") == {"a": Decimal("0.022"), "b": Decimal(0)}
    assert meter.requests_by_tenant() == {"a": 2}
    assert meter.cost_per_1k("ask") == Decimal("0.022") / 3 * 1000
    assert meter.cache_hit_ratio() == pytest.approx(1 / 3)
    assert meter.self_hosted_share() == pytest.approx(0.5)
    meter.save(tmp_path / "usage.jsonl")
    assert Meter.load(tmp_path / "usage.jsonl").events == meter.events
    assert percentile([100, 200, 300, 400, 1000], 0.95) == 1000
