"""Metering (FR-03): every request becomes a usage event carrying its cost."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, Field

from sib.llm import CATALOG


class UsageEvent(BaseModel):
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_id: str
    feature: str  # ask | draft | summarize
    model: str
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal = Decimal(0)
    latency_ms: int = 0
    cache_layer: str = "none"  # none | exact | semantic
    route_class: str = ""
    escalated: bool = False
    trial: bool = False  # trial usage is not billed; its cost belongs in CAC


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]


class Meter:
    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    def record(self, event: UsageEvent) -> None:
        self.events.append(event)

    def select(self, tenant_id: str | None = None, feature: str | None = None) -> list[UsageEvent]:
        return [e for e in self.events
                if (tenant_id is None or e.tenant_id == tenant_id) and (feature is None or e.feature == feature)]

    def cost_by(self, key: str) -> dict[str, Decimal]:
        out: dict[str, Decimal] = defaultdict(Decimal)
        for e in self.events:
            out[str(getattr(e, key))] += e.cost_usd
        return dict(out)

    def requests_by_tenant(self, billable_only: bool = True) -> dict[str, int]:
        return dict(Counter(e.tenant_id for e in self.events if not (billable_only and e.trial)))

    def total_cost(self, feature: str | None = None) -> Decimal:
        return sum((e.cost_usd for e in self.select(feature=feature)), Decimal(0))

    def cost_per_1k(self, feature: str | None = None) -> Decimal:
        events = self.select(feature=feature)
        return self.total_cost(feature) / len(events) * 1000 if events else Decimal(0)

    def p95_latency_ms(self, feature: str | None = None) -> int:
        return percentile([e.latency_ms for e in self.select(feature=feature)], 0.95)

    def cache_hit_ratio(self) -> float:
        return sum(e.cache_layer != "none" for e in self.events) / len(self.events) if self.events else 0.0

    def prompt_cache_read_share(self) -> float:
        read = sum(e.cache_read_tokens for e in self.events)
        total = read + sum(e.input_tokens + e.cache_write_tokens for e in self.events)
        return read / total if total else 0.0

    def model_mix(self) -> dict[str, float]:
        served = [e for e in self.events if e.cache_layer == "none"]
        counts = Counter(e.model for e in served)
        return {m: c / len(served) for m, c in counts.items()} if served else {}

    def self_hosted_share(self) -> float:
        return sum(share for m, share in self.model_mix().items() if CATALOG[m].tier == "self_hosted")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(e.model_dump_json() for e in self.events), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Meter":
        meter = cls()
        if path.exists():
            meter.events = [UsageEvent.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return meter
