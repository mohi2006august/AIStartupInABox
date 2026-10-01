"""HTTP API and web console.

The tenant comes from the `X-Tenant-Id` header. That is a stand-in for SSO: in production it is
derived from the authenticated session, never from a client-supplied value.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from sib import __version__
from sib.app import Answer, Stack, drafting_assets
from sib.config import Settings, get_settings
from sib.drafting import AUDIENCES, DOC_TYPES
from sib.economics import summary
from sib.ledger import BASELINE, CHANGE_BY_ID, CHANGES, DecisionRecord, Ledger, bench, default_scenario, to_json, unit_costs
from sib.mcp_gateway import PolicyError
from sib.metering import UsageEvent
from sib.router import INTENTS
from sib.workload import BenchResult, run_workload

log = logging.getLogger("sib.api")


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    remember: bool = False


class AssistIn(BaseModel):
    text: str = Field(min_length=3, max_length=2000)


class DraftIn(BaseModel):
    matter_id: str
    doc_type: Literal["demand_letter", "termination_notice", "client_update"] = "demand_letter"
    audience: Literal["court", "opposing_counsel", "client"] = "opposing_counsel"
    instruction: str = Field(min_length=3, max_length=2000, description="What provision the draft must rely on")
    action: str = Field(min_length=3, max_length=500, description="What the recipient is asked to do")
    days: int = Field(14, ge=1, le=365)


class ToolCallIn(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class BenchIn(BaseModel):
    change: str = "latest"
    customers: int | None = Field(None, ge=1, le=10_000)


def _answer_out(a: Answer) -> dict[str, Any]:
    return {"text": a.text, "citations": [asdict(c) for c in a.citations], "model": a.model,
            "route_class": a.route_class, "cache_layer": a.cache_layer, "escalated": a.escalated,
            "cost_usd": float(a.cost_usd), "latency_ms": a.latency_ms}


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ledger = Ledger(Path(settings.ledger_dir))
        self.usage_path = Path(settings.state_dir) / "usage.jsonl"
        self.lock = threading.Lock()
        self._persisted = 0
        self._measured: BenchResult | None = None
        self._baseline: BenchResult | None = None
        self.reload()

    def reload(self) -> None:
        """Rebuild the stack on the current production config (after a ledger change)."""
        self.config = self.ledger.current_config()
        live = self.config.with_(live=True) if self.settings.live_models else self.config
        self.stack = Stack(live)  # benchmarks and economics always replay on the simulator
        self._persisted = 0
        self._measured = None

    def persist_usage(self) -> None:
        events: list[UsageEvent] = self.stack.meter.events[self._persisted :]
        if events:
            self.usage_path.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_path.open("a", encoding="utf-8") as fh:
                fh.writelines(e.model_dump_json() + "\n" for e in events)
            self._persisted = len(self.stack.meter.events)

    def measured(self) -> BenchResult:
        """Unit costs come from replaying the benchmark workload on the production config."""
        if self._measured is None:
            self._measured = run_workload(self.config)
        return self._measured

    def baseline(self) -> BenchResult:
        """The M1 MVP (frontier API for everything) for before/after comparisons."""
        if self._baseline is None:
            self._baseline = run_workload(BASELINE)
        return self._baseline


def create_app(settings: Settings | None = None) -> FastAPI:
    state = AppState(settings or get_settings())
    api = FastAPI(title="AI Startup-in-a-Box", version=__version__)
    api.state.sib = state

    def tenant(x_tenant_id: str = Header(..., min_length=1)) -> str:
        if x_tenant_id not in state.stack.corpus.tenants:
            raise HTTPException(401, "unknown tenant")
        return x_tenant_id

    def guarded(fn, *args: Any, **kwargs: Any) -> Any:
        with state.lock:
            try:
                result = fn(*args, **kwargs)
            except KeyError as exc:
                raise HTTPException(404, str(exc.args[0]) if exc.args else "not found") from exc
            except PolicyError as exc:
                raise HTTPException(403, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            state.persist_usage()
            return result

    @api.get("/", response_class=HTMLResponse, include_in_schema=False)
    def console() -> str:
        return (resources.files("sib") / "web" / "index.html").read_text(encoding="utf-8")

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "selfhost_healthy": state.stack.selfhost_healthy}

    @api.get("/v1/tenants")
    def tenants() -> list[dict[str, str]]:
        return [{"id": k, "name": v} for k, v in state.stack.corpus.tenants.items()]

    @api.get("/v1/matters")
    def matters(tenant_id: str = Depends(tenant)) -> list[dict[str, str]]:
        return [{"id": m.id, "name": m.name, "kind": m.kind} for m in state.stack.corpus.matters_for(tenant_id)]

    @api.post("/v1/matters/{matter_id}/ask")
    def ask(matter_id: str, body: AskIn, tenant_id: str = Depends(tenant)) -> dict[str, Any]:
        return _answer_out(guarded(state.stack.ask, tenant_id, matter_id, body.question, remember=body.remember))

    @api.get("/v1/matters/{matter_id}/summary")
    def matter_summary(matter_id: str, tenant_id: str = Depends(tenant)) -> list[dict[str, str]]:
        return [asdict(c) for c in guarded(state.stack.summarize, tenant_id, matter_id)]

    @api.post("/v1/matters/{matter_id}/assist")
    def assist(matter_id: str, body: AssistIn, tenant_id: str = Depends(tenant)) -> dict[str, Any]:
        """One box for everything: the distilled intent classifier picks the feature."""
        from sib.app import trained_router

        intent, _ = trained_router(state.config.tuned_embeddings, state.config.selfhost).classify(body.text)
        if intent == "ask":
            return {"intent": intent, "answer": _answer_out(guarded(state.stack.ask, tenant_id, matter_id, body.text))}
        if intent == "summarize":
            return {"intent": intent, "summary": [asdict(c) for c in guarded(state.stack.summarize, tenant_id, matter_id)]}
        if intent == "draft":
            return {"intent": intent, "next": "POST /v1/drafts", "doc_types": DOC_TYPES, "audiences": AUDIENCES}
        return {"intent": intent, "next": "POST /v1/tools/{name}", "tools": sorted(state.stack.gateway.tools),
                "note": "email and calendar writes wait for lawyer approval"}

    @api.post("/v1/drafts")
    def create_draft(body: DraftIn, tenant_id: str = Depends(tenant)) -> dict[str, Any]:
        d = guarded(state.stack.draft, tenant_id, body.matter_id, body.doc_type, body.audience, body.instruction,
                    body.action, body.days)
        return {"text": d.text, "model": d.model, "used_adapter": d.used_adapter, "note": d.note,
                "provision": _answer_out(d.fact), "cost_usd": float(d.cost_usd), "latency_ms": d.latency_ms}

    @api.get("/v1/economics/summary")
    def economics(customers: int | None = None) -> dict[str, Any]:
        s = default_scenario()
        if customers:
            s = s.model_copy(update={"customers": customers})
        with state.lock:
            result, base = state.measured(), state.baseline()
        unit = unit_costs(result, state.config, s)
        out = summary(s, unit, state.settings.margin_target)
        out["measured"] = {"cost_per_ask": result.cost_per_ask, "cost_per_draft": result.cost_per_draft,
                           "cogs_per_1k": result.cogs_per_1k, "fixed_monthly": float(unit.fixed_monthly)}
        before = summary(s, unit_costs(base, BASELINE, s), state.settings.margin_target)
        out["baseline"] = {k: before[k] for k in ("gross_margin", "ai_cogs_per_account", "runway_months", "default_alive",
                                                    "blended_payback_months", "net_burn", "cash_curve")}
        return out

    @api.get("/v1/ledger")
    def ledger_records() -> dict[str, Any]:
        recorded = {r.id for r in state.ledger.records()}
        return {"records": to_json(state.ledger.records()),
                "pending": [{"id": c.id, "title": c.title} for c in CHANGES if c.id not in recorded]}

    @api.post("/v1/ledger")
    def record_decision(record: DecisionRecord) -> dict[str, str]:
        """Record a decision measured elsewhere (e.g. a pricing change)."""
        with state.lock:
            state.ledger.add(record)
            if record.id in CHANGE_BY_ID:
                state.reload()
        return {"status": "recorded", "id": record.id}

    @api.post("/v1/ledger/bench")
    def run_bench(body: BenchIn) -> dict[str, Any]:
        s = default_scenario()
        if body.customers:
            s = s.model_copy(update={"customers": body.customers})
        with state.lock:
            try:
                record = bench(body.change, state.ledger, scenario=s)
            except KeyError as exc:
                raise HTTPException(404, str(exc.args[0])) from exc
            state.reload()
        return to_json([record])[0]

    @api.get("/v1/ops")
    def ops() -> dict[str, Any]:
        from sib.app import trained_router

        with state.lock:
            result = state.measured()
        router = trained_router(state.config.tuned_embeddings, state.config.selfhost) if state.config.router else None
        reports = drafting_assets().reports
        return {
            "config": asdict(state.config),
            "model_mix": result.model_mix,
            "cache_hit_ratio": result.cache_hit_ratio,
            "prompt_cache_read_share": result.prompt_cache_read_share,
            "escalation_rate": result.escalation_rate,
            "p95_ask_ms": result.p95_ask_ms,
            "router": {"table": router.table, "hard_threshold": router.hard_threshold, "intents": INTENTS} if router else None,
            "adapters": {t: {"accepted": r.accepted, "gain_pts": round(r.gain_pts, 1), "rank": r.rank, "reason": r.reason}
                         for t, r in reports.items()},
        }

    @api.get("/v1/tools")
    def tools() -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "requires_approval": t.requires_approval,
                 "input_schema": t.input_schema} for t in state.stack.gateway.tools.values()]

    @api.post("/v1/tools/{name}")
    def call_tool(name: str, body: ToolCallIn, tenant_id: str = Depends(tenant),
                  x_user: str = Header("lawyer")) -> dict[str, Any]:
        return guarded(state.stack.gateway.call, tenant_id, x_user, name, body.arguments)

    @api.get("/v1/approvals")
    def approvals(tenant_id: str = Depends(tenant)) -> list[dict[str, Any]]:
        return [{"id": a.id, "tool": a.tool, "user": a.user, "args": a.args, "status": a.status}
                for a in state.stack.gateway.pending.values() if a.tenant_id == tenant_id]

    @api.post("/v1/approvals/{approval_id}/{verdict}")
    def decide(approval_id: str, verdict: Literal["approve", "deny"], tenant_id: str = Depends(tenant),
               x_user: str = Header("partner")) -> dict[str, Any]:
        fn = state.stack.gateway.approve if verdict == "approve" else state.stack.gateway.deny
        return guarded(fn, tenant_id, approval_id, x_user)

    @api.get("/v1/billing/invoices")
    def billing_invoices(tenant_id: str = Depends(tenant)) -> dict[str, Any]:
        """Stripe invoice ids and payment status for this firm (empty when Stripe is not configured)."""
        from sib.stripe_billing import StripeGateway

        gw = StripeGateway(None, Path(state.settings.state_dir), state.stack.corpus.tenants)
        return {k: v for k, v in gw.statuses().items() if k.split(":")[0] == tenant_id}

    @api.post("/v1/stripe/webhook", include_in_schema=False)
    async def stripe_webhook(request: Request, stripe_signature: str = Header("")) -> dict[str, Any]:
        """Payment-status updates from Stripe. The signature is verified before anything is read."""
        secret = state.settings.stripe_webhook_secret
        if not secret:
            raise HTTPException(404, "Stripe webhooks are not configured")
        import stripe

        from sib.stripe_billing import StripeGateway

        payload = await request.body()
        try:
            event = stripe.Webhook.construct_event(payload, stripe_signature, secret)
        except stripe.SignatureVerificationError as exc:
            raise HTTPException(400, "invalid signature") from exc
        except ValueError as exc:
            raise HTTPException(400, "invalid payload") from exc
        matched = False
        if event.type.startswith("invoice."):
            gw = StripeGateway(None, Path(state.settings.state_dir), state.stack.corpus.tenants)
            matched = gw.apply_event(event.type, event.data.object.to_dict())
        log.info("stripe_webhook", extra={"type": event.type, "matched": matched})
        return {"received": True, "matched": matched}

    return api
