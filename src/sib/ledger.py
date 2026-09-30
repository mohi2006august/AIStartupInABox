"""Decision ledger (FR-11) and `sib bench`: every technical change gets a measured price tag.

A change is a set of config switches. `bench` replays the same workload before and after,
converts the cost difference into gross margin and runway using the business scenario, applies
the quality gate, and writes the record to ledger/ as JSON plus a markdown timeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from sib.app import StackConfig
from sib.economics import (
    Scenario, UnitCosts, blended_payback_months, default_alive, gross_margin, load_scenario, runway_delta_months,
    runway_months, simulate,
)
from sib.quantize import monthly_gpu_cost
from sib.workload import BenchResult, run_workload

QUALITY_GATE_PTS = 2.0
MARGIN_TOLERANCE_PTS = 0.5
MIN_QUALITY_PER_MARGIN_PT = 2.0  # a change may spend margin only if each point buys this much quality


class DecisionRecord(BaseModel):
    id: str
    date: date
    change: str
    concepts: list[str]
    quality_before: float
    quality_after: float
    p95_latency_before_ms: int
    p95_latency_after_ms: int
    cogs_per_1k_before: Decimal
    cogs_per_1k_after: Decimal
    gross_margin_delta_pts: float
    runway_delta_months: float
    decision: Literal["kept", "reverted"]
    # Context beyond the core schema:
    quality_metric: str = "ask"
    gross_margin_before: float = 0.0
    gross_margin_after: float = 0.0
    payback_before_months: float | None = None
    payback_after_months: float | None = None
    runway_before_months: float | None = None  # None: cash never runs out within the horizon
    runway_after_months: float | None = None
    default_alive_after: bool = False
    customers: int = 0
    reason: str = ""


@dataclass(frozen=True)
class Change:
    id: str
    title: str
    concepts: tuple[str, ...]
    overrides: dict[str, object]
    metric: str  # which quality score the record reports: ask | draft


# The M1 MVP: frontier API for everything, off-the-shelf embeddings, no caches, no router.
BASELINE = StackConfig(tuned_embeddings=False)

CHANGES: list[Change] = [
    Change("tuned-embeddings", "Contrastively tuned embeddings for retrieval",
           ("Embeddings", "RAG", "Vector Database"), {"tuned_embeddings": True}, "ask"),
    Change("caching", "Exact + semantic + provider prompt caching",
           ("Caches", "Tokenization"), {"exact_cache": True, "semantic_cache": True, "prompt_cache": True}, "ask"),
    Change("router", "Distilled router sends easy asks to a small API model",
           ("Knowledge Distillation", "LLMs"), {"router": True}, "ask"),
    Change("selfhost", "Self-host AWQ 8B with per-firm LoRA for drafting and easy asks",
           ("Quantization", "LoRA", "Fine Tuning", "Overfitting"), {"selfhost": True}, "draft"),
    Change("context-6800", "Shrink the ask context budget from 8,000 to 6,800 tokens",
           ("Context Window", "Tokenization"), {"ask_budget": 6800}, "ask"),
]
CHANGE_BY_ID = {c.id: c for c in CHANGES}


def default_scenario() -> Scenario:
    return load_scenario(Path(str(resources.files("sib") / "data" / "business.json")))


def unit_costs(result: BenchResult, config: StackConfig, s: Scenario) -> UnitCosts:
    """Self-hosting buys a whole GPU node: whatever traffic does not use is idle cost."""
    fixed = Decimal(0)
    if config.selfhost:
        used = Decimal(str(result.selfhost_cost_per_request)) * s.requests_per_account * s.customers
        fixed = max(monthly_gpu_cost() - used, Decimal(0))
    return UnitCosts(Decimal(str(result.cost_per_ask)), Decimal(str(result.cost_per_draft)), fixed)


class Ledger:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def records(self) -> list[DecisionRecord]:
        if not self.directory.exists():
            return []
        found = [DecisionRecord.model_validate_json(p.read_text(encoding="utf-8")) for p in self.directory.glob("*.json")]
        order = {c.id: i for i, c in enumerate(CHANGES)}
        return sorted(found, key=lambda r: (r.date, order.get(r.id, len(order))))

    def add(self, record: DecisionRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{record.id}.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
        (self.directory / "LEDGER.md").write_text(render_markdown(self.records()), encoding="utf-8")

    def current_config(self, excluding: str | None = None) -> StackConfig:
        """The production config: baseline plus every kept change."""
        config = BASELINE
        for record in self.records():
            if record.decision == "kept" and record.id != excluding and record.id in CHANGE_BY_ID:
                config = config.with_(**CHANGE_BY_ID[record.id].overrides)
        return config

    def latest_change_id(self) -> str:
        recorded = {r.id for r in self.records()}
        pending = [c.id for c in CHANGES if c.id not in recorded]
        return pending[0] if pending else CHANGES[-1].id


def bench(change_id: str, ledger: Ledger, scenario: Scenario | None = None, on: date | None = None,
          write: bool = True) -> DecisionRecord:
    if change_id == "latest":
        change_id = ledger.latest_change_id()
    if change_id not in CHANGE_BY_ID:
        raise KeyError(f"unknown change '{change_id}'; known: {', '.join(CHANGE_BY_ID)}")
    change, s = CHANGE_BY_ID[change_id], scenario or default_scenario()
    before_cfg = ledger.current_config(excluding=change_id)
    after_cfg = before_cfg.with_(**change.overrides)
    before, after = run_workload(before_cfg), run_workload(after_cfg)
    unit_b, unit_a = unit_costs(before, before_cfg, s), unit_costs(after, after_cfg, s)
    gm_b, gm_a = gross_margin(s, unit_b), gross_margin(s, unit_a)
    drops = {"ask": before.ask_quality - after.ask_quality, "draft": before.draft_quality - after.draft_quality}
    worst = max(drops, key=lambda k: drops[k])
    margin_pts = 100 * (gm_a - gm_b)
    gain = -min(drops.values())  # best quality improvement across ask and draft
    if drops[worst] > QUALITY_GATE_PTS:
        decision, reason = "reverted", f"quality gate: {worst} quality fell {drops[worst]:.1f} pts (limit {QUALITY_GATE_PTS})"
    elif margin_pts < -MARGIN_TOLERANCE_PTS and gain < MIN_QUALITY_PER_MARGIN_PT * -margin_pts:
        decision = "reverted"
        reason = f"gross margin fell {-margin_pts:.1f} pts at {s.customers} customers for a {gain:+.1f} pt quality change"
        if after_cfg.selfhost and not before_cfg.selfhost:
            even = breakeven_customers(before, after, before_cfg, after_cfg, s)
            reason += (f"; the GPU node costs ${monthly_gpu_cost():.0f}/month and traffic does not fill it yet"
                       + (f" (margin-positive from about {even} customers)" if even else ""))
    elif margin_pts < -MARGIN_TOLERANCE_PTS:
        decision, reason = "kept", f"buys {gain:+.1f} pts of quality for {-margin_pts:.1f} pts of margin"
    else:
        decision, reason = "kept", "quality within gate and margin not reduced"
    rows_b, rows_a = simulate(s, unit_b), simulate(s, unit_a)
    metric = change.metric
    record = DecisionRecord(
        id=change.id, date=on or date.today(), change=change.title, concepts=list(change.concepts),
        quality_before=round(getattr(before, f"{metric}_quality"), 2),
        quality_after=round(getattr(after, f"{metric}_quality"), 2),
        p95_latency_before_ms=before.p95_ask_ms, p95_latency_after_ms=after.p95_ask_ms,
        cogs_per_1k_before=Decimal(str(round(before.cogs_per_1k, 2))),
        cogs_per_1k_after=Decimal(str(round(after.cogs_per_1k, 2))),
        gross_margin_delta_pts=round(100 * (gm_a - gm_b), 2),
        runway_delta_months=runway_delta_months(s, unit_b, unit_a),
        decision=decision, quality_metric=metric,
        gross_margin_before=round(gm_b, 4), gross_margin_after=round(gm_a, 4),
        payback_before_months=blended_payback_months(s, unit_b), payback_after_months=blended_payback_months(s, unit_a),
        runway_before_months=runway_months(rows_b), runway_after_months=runway_months(rows_a),
        default_alive_after=default_alive(rows_a), customers=s.customers, reason=reason,
    )
    if write:
        ledger.add(record)
    return record


def breakeven_customers(before: BenchResult, after: BenchResult, before_cfg: StackConfig, after_cfg: StackConfig,
                        s: Scenario, limit: int = 500) -> int | None:
    """Smallest customer count at which the change stops costing gross margin."""
    for n in range(s.customers, limit + 1):
        at_n = s.model_copy(update={"customers": n})
        if gross_margin(at_n, unit_costs(after, after_cfg, at_n)) >= gross_margin(at_n, unit_costs(before, before_cfg, at_n)):
            return n
    return None


def current_unit_costs(ledger: Ledger, s: Scenario) -> tuple[UnitCosts, BenchResult, StackConfig]:
    config = ledger.current_config()
    result = run_workload(config)
    return unit_costs(result, config, s), result, config


def _signed(value: float, unit: str = "", digits: int = 1) -> str:
    if abs(value) < 10 ** -digits / 2:
        return "="
    return f"{'+' if value > 0 else '−'}{abs(value):.{digits}f}{unit}"


def render_row(r: DecisionRecord) -> list[str]:
    gate_failed = r.decision == "reverted" and r.reason.startswith("quality gate")
    return [
        r.date.isoformat(), r.change,
        _signed(r.quality_after - r.quality_before) + (" ✖" if gate_failed else ""),
        _signed((r.p95_latency_after_ms - r.p95_latency_before_ms) / 1000, "s"),
        _signed(float(r.cogs_per_1k_after - r.cogs_per_1k_before), "", 2).replace("+", "+$").replace("−", "−$"),
        _signed(r.gross_margin_delta_pts, " pts"),
        "→ default alive" if r.default_alive_after and r.runway_before_months is not None
        and r.runway_after_months is None else _signed(r.runway_delta_months, " mo"),
        r.decision,
    ]


def render_markdown(records: list[DecisionRecord]) -> str:
    lines = ["# Decision ledger", "",
             "Generated by `sib bench`. Each row is one technical change measured on the same replayed workload.", "",
             "| Date | Change | Quality | p95 | COGS/1k | Gross margin | Runway | Decision |", "|---|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(render_row(r)) + " |" for r in records]
    kept = [r for r in records if r.decision == "kept"]
    if kept:
        first, last = kept[0], kept[-1]
        summary_line = (f"Gross margin {100 * first.gross_margin_before:.0f}% → {100 * last.gross_margin_after:.0f}%"
                        f" · CAC payback {first.payback_before_months:.1f} → {last.payback_after_months:.1f} mo"
                        f" · default alive {'✔' if last.default_alive_after else '✖'}"
                        f" (kept changes, {last.customers} customers)")
        lines += ["", summary_line]
    reverted = [r for r in records if r.decision == "reverted"]
    if reverted:
        lines += ["", "## Reverted", ""] + [f"- **{r.change}**: {r.reason}." for r in reverted]
    return "\n".join(lines) + "\n"


def to_json(records: list[DecisionRecord]) -> list[dict[str, object]]:
    return [json.loads(r.model_dump_json()) for r in records]
