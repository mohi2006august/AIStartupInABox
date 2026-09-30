"""Economics console (FR-04): COGS, gross margin, burn, runway, CAC and payback by channel.

Two inputs: a `Scenario` (the business assumptions, from data/business.json) and `UnitCosts`
(AI cost per request, measured by replaying the workload through the stack). Every technical
change moves UnitCosts; this module turns that into margin, payback and runway.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

HORIZON_MONTHS = 36


class Channel(BaseModel):
    name: str
    monthly_spend: Decimal  # fully loaded: ads, tools and the salaries attributable to the channel
    new_customers_per_month: float
    trials_per_month: float


class Scenario(BaseModel):
    cash: Decimal
    monthly_opex: Decimal  # payroll, rent, tools: everything that is not COGS or acquisition spend
    customers: int
    seats_per_account: int
    seat_price: Decimal
    requests_per_seat_per_month: int
    draft_share: float
    other_cogs_per_account: Decimal  # hosting, support, payment fees
    monthly_churn: float
    trial_requests: int  # AI requests a typical trial consumes before converting or leaving
    channels: list[Channel]

    @property
    def arpa(self) -> Decimal:
        return self.seats_per_account * self.seat_price

    @property
    def requests_per_account(self) -> int:
        return self.seats_per_account * self.requests_per_seat_per_month


@dataclass(frozen=True)
class UnitCosts:
    ask: Decimal  # AI COGS per ask request
    draft: Decimal
    fixed_monthly: Decimal = Decimal(0)  # e.g. a GPU node: paid regardless of traffic

    def blended(self, draft_share: float) -> Decimal:
        share = Decimal(str(draft_share))
        return self.ask * (1 - share) + self.draft * share


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate(json.loads(path.read_text(encoding="utf-8")))


def ai_cogs_per_account(s: Scenario, unit: UnitCosts, customers: float | None = None) -> Decimal:
    n = Decimal(str(customers if customers is not None else s.customers))
    fixed = unit.fixed_monthly / n if n > 0 else Decimal(0)
    return s.requests_per_account * unit.blended(s.draft_share) + fixed


def contribution_per_account(s: Scenario, unit: UnitCosts) -> Decimal:
    return s.arpa - ai_cogs_per_account(s, unit) - s.other_cogs_per_account


def gross_margin(s: Scenario, unit: UnitCosts) -> float:
    return float(contribution_per_account(s, unit) / s.arpa)


@dataclass
class ChannelEconomics:
    name: str
    cac: Decimal
    trial_ai_cogs_per_customer: Decimal
    payback_months: float | None  # None: never pays back
    ltv_to_cac: float


def channel_economics(s: Scenario, unit: UnitCosts) -> list[ChannelEconomics]:
    """CAC includes the AI COGS of trials; payback = CAC / (ARPA - AI COGS - other COGS)."""
    contribution = contribution_per_account(s, unit)
    ltv = contribution / Decimal(str(s.monthly_churn)) if s.monthly_churn else Decimal(0)
    out = []
    for ch in s.channels:
        if ch.new_customers_per_month <= 0:
            continue
        new = Decimal(str(ch.new_customers_per_month))
        trial_cogs = Decimal(str(ch.trials_per_month)) * s.trial_requests * unit.blended(s.draft_share)
        cac = (ch.monthly_spend + trial_cogs) / new
        payback = float(cac / contribution) if contribution > 0 else None
        out.append(ChannelEconomics(ch.name, cac, trial_cogs / new, payback, float(ltv / cac) if cac else 0.0))
    return out


def blended_payback_months(s: Scenario, unit: UnitCosts) -> float | None:
    contribution = contribution_per_account(s, unit)
    new = sum(ch.new_customers_per_month for ch in s.channels)
    if contribution <= 0 or new <= 0:
        return None
    blended = unit.blended(s.draft_share)
    spend = sum((ch.monthly_spend + Decimal(str(ch.trials_per_month)) * s.trial_requests * blended for ch in s.channels), Decimal(0))
    return float(spend / Decimal(str(new)) / contribution)


@dataclass
class MonthRow:
    month: int
    customers: float
    mrr: Decimal
    cogs: Decimal
    gross_burn: Decimal
    net_burn: Decimal
    cash: Decimal


def simulate(s: Scenario, unit: UnitCosts, months: int = HORIZON_MONTHS) -> list[MonthRow]:
    rows: list[MonthRow] = []
    customers, cash = float(s.customers), s.cash
    new_per_month = sum(ch.new_customers_per_month for ch in s.channels)
    trials = Decimal(str(sum(ch.trials_per_month for ch in s.channels)))
    acquisition = sum((ch.monthly_spend for ch in s.channels), Decimal(0))
    blended = unit.blended(s.draft_share)
    for month in range(1, months + 1):
        n = Decimal(str(customers))
        mrr = n * s.arpa
        cogs = n * (s.requests_per_account * blended + s.other_cogs_per_account) + unit.fixed_monthly
        trial_cogs = trials * s.trial_requests * blended
        gross_burn = cogs + s.monthly_opex + acquisition + trial_cogs
        net_burn = gross_burn - mrr
        cash -= net_burn
        rows.append(MonthRow(month, customers, mrr, cogs, gross_burn, net_burn, cash))
        customers = customers * (1 - s.monthly_churn) + new_per_month
    return rows


def runway_months(rows: list[MonthRow]) -> float | None:
    """Months until cash runs out (interpolated within the month); None if it never does."""
    for r in rows:
        if r.cash < 0:
            cash_at_start = r.cash + r.net_burn
            return round(r.month - 1 + float(cash_at_start / r.net_burn), 2)
    return None


def default_alive(rows: list[MonthRow]) -> bool:
    """Reaches profitability on the current trajectory before the cash is gone."""
    for r in rows:
        if r.cash < 0:
            return False
        if r.net_burn <= 0:
            return True
    return False


def burn_multiple(s: Scenario, rows: list[MonthRow]) -> float | None:
    net_new_arr = (rows[1].mrr - rows[0].mrr) * 12 if len(rows) > 1 else Decimal(0)
    if net_new_arr <= 0:
        return None
    return float(rows[0].net_burn * 12 / net_new_arr)


def runway_delta_months(s: Scenario, before: UnitCosts, after: UnitCosts) -> float:
    """Re-run the simulation with the new burn. Runways beyond the horizon are capped at it."""
    b = runway_months(simulate(s, before)) or HORIZON_MONTHS
    a = runway_months(simulate(s, after)) or HORIZON_MONTHS
    return round(float(a - b), 2)


def summary(s: Scenario, unit: UnitCosts, margin_target: float = 0.75) -> dict[str, object]:
    rows = simulate(s, unit)
    ai = ai_cogs_per_account(s, unit)
    margin = gross_margin(s, unit)
    return {
        "mrr": float(rows[0].mrr),
        "arpa": float(s.arpa),
        "customers": s.customers,
        "ai_cogs_per_account": float(ai),
        "ai_cogs_share_of_arpa": float(ai / s.arpa),
        "other_cogs_per_account": float(s.other_cogs_per_account),
        "gross_margin": margin,
        "margin_target": margin_target,
        "margin_alert": margin < margin_target,
        "gross_burn": float(rows[0].gross_burn),
        "net_burn": float(rows[0].net_burn),
        "runway_months": runway_months(rows),
        "default_alive": default_alive(rows),
        "burn_multiple": burn_multiple(s, rows),
        "blended_payback_months": blended_payback_months(s, unit),
        "channels": [
            {"name": c.name, "cac": float(c.cac), "trial_ai_cogs_per_customer": float(c.trial_ai_cogs_per_customer),
             "payback_months": c.payback_months, "ltv_to_cac": c.ltv_to_cac}
            for c in channel_economics(s, unit)
        ],
        "cash_curve": [{"month": r.month, "cash": float(r.cash), "net_burn": float(r.net_burn),
                        "customers": round(r.customers, 1)} for r in rows],
    }
