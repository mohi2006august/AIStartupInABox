"""`sib` command line: serve, evals, economics, bench, ledger, ask, draft, mcp, billing."""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import typer

from sib.config import get_settings

app = typer.Typer(add_completion=False, help="AI Startup-in-a-Box: a legal research SaaS wired to its unit economics.")


def _ledger():
    from sib.ledger import Ledger

    return Ledger(Path(get_settings().ledger_dir))


def _scenario(customers: Optional[int]):
    from sib.ledger import default_scenario

    s = default_scenario()
    return s.model_copy(update={"customers": customers}) if customers else s


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API and web console."""
    import uvicorn

    uvicorn.run("sib.api:create_app", factory=True, host=host, port=port)


@app.command()
def evals(update_baseline: bool = typer.Option(False, help="Record the current scores as the regression baseline."),
          as_json: bool = typer.Option(False, "--json")) -> None:
    """Run every quality gate against the current production config (baseline + kept ledger changes)."""
    from sib.evals import gates_to_json, load_baseline, run_gates, save_baseline

    settings = get_settings()
    baseline_path = Path(settings.evals_dir) / "baseline.json"
    config = _ledger().current_config()
    gates, result = run_gates(config, _scenario(None), load_baseline(baseline_path), settings.margin_target)
    if as_json:
        typer.echo(json.dumps(gates_to_json(gates), indent=2))
    else:
        for g in gates:
            mark = "PASS" if g.passed else ("FAIL" if g.blocking else "ALERT")
            typer.echo(f"{mark:5}  {g.name:40} {g.value:>12}  (threshold {g.threshold}) {g.detail}")
    if update_baseline:
        save_baseline(baseline_path, result)
        typer.echo(f"baseline written to {baseline_path}")
    failed = [g for g in gates if g.blocking and not g.passed]
    if failed:
        typer.echo(f"{len(failed)} blocking gate(s) failed", err=True)
        raise typer.Exit(1)


@app.command()
def economics(customers: Optional[int] = typer.Option(None, help="Override the customer count."),
              as_json: bool = typer.Option(False, "--json")) -> None:
    """Recompute COGS, margin, burn, runway and CAC payback from measured unit costs."""
    from sib.economics import summary
    from sib.ledger import current_unit_costs

    s = _scenario(customers)
    unit, result, _ = current_unit_costs(_ledger(), s)
    out = summary(s, unit, get_settings().margin_target)
    out["measured"] = {"cost_per_ask": result.cost_per_ask, "cost_per_draft": result.cost_per_draft,
                       "cogs_per_1k": result.cogs_per_1k, "fixed_monthly": float(unit.fixed_monthly)}
    if as_json:
        typer.echo(json.dumps(out, indent=2, default=str))
        return
    runway = out["runway_months"]
    typer.echo(f"Customers {out['customers']}  ARPA ${out['arpa']:.0f}  MRR ${out['mrr']:,.0f}")
    typer.echo(f"AI COGS/account ${out['ai_cogs_per_account']:.2f} ({out['ai_cogs_share_of_arpa']:.1%} of ARPA)")
    typer.echo(f"Gross margin {out['gross_margin']:.1%} (target {out['margin_target']:.0%})"
               + ("  ALERT: below target" if out["margin_alert"] else ""))
    typer.echo(f"Net burn ${out['net_burn']:,.0f}/month  runway {'beyond horizon' if runway is None else f'{runway} months'}"
               f"  default alive: {'yes' if out['default_alive'] else 'no'}")
    typer.echo(f"Blended CAC payback {out['blended_payback_months']:.1f} months")
    for ch in out["channels"]:  # type: ignore[union-attr]
        typer.echo(f"  {ch['name']:14} CAC ${ch['cac']:,.0f}  payback {ch['payback_months']:.1f} mo  LTV:CAC {ch['ltv_to_cac']:.1f}")


@app.command()
def bench(change: str = typer.Option("latest", help="Change id, or 'latest' for the next unmeasured change."),
          customers: Optional[int] = typer.Option(None), run_all: bool = typer.Option(False, "--all"),
          dry_run: bool = typer.Option(False, help="Measure without writing a ledger record.")) -> None:
    """Measure a change before vs after and record its impact in the decision ledger."""
    from sib.ledger import CHANGES, bench as run_bench

    ledger = _ledger()
    ids = [c.id for c in CHANGES] if run_all else [change]
    for change_id in ids:
        r = run_bench(change_id, ledger, scenario=_scenario(customers), on=date.today(), write=not dry_run)
        typer.echo(f"{r.id}: {r.decision.upper()}  quality {r.quality_before:.1f} -> {r.quality_after:.1f} ({r.quality_metric})"
                   f"  p95 {r.p95_latency_before_ms} -> {r.p95_latency_after_ms} ms"
                   f"  COGS/1k ${r.cogs_per_1k_before} -> ${r.cogs_per_1k_after}"
                   f"  GM {r.gross_margin_delta_pts:+.1f} pts  runway {r.runway_delta_months:+.1f} mo")
        typer.echo(f"  {r.reason}")


@app.command()
def ledger() -> None:
    """Print the decision ledger."""
    from sib.ledger import render_markdown

    typer.echo(render_markdown(_ledger().records()))


@app.command()
def ask(tenant: str, matter: str, question: str) -> None:
    """Ask a cited question about one matter."""
    from sib.app import Stack

    answer = Stack(_ledger().current_config()).ask(tenant, matter, question)
    typer.echo(answer.text)
    for c in answer.citations:
        typer.echo(f"  [{c.title} · {c.chunk_id}]")
    typer.echo(f"  model {answer.model}, cost ${answer.cost_usd:.4f}, {answer.latency_ms} ms")


@app.command()
def draft(tenant: str, matter: str, instruction: str, action: str = "remedy the breach",
          doc_type: str = "demand_letter", audience: str = "opposing_counsel") -> None:
    """Draft a letter in the firm's house style, grounded in the matter's documents."""
    from sib.app import Stack

    d = Stack(_ledger().current_config()).draft(tenant, matter, doc_type, audience, instruction, action)
    typer.echo(d.text)
    typer.echo(f"\n  model {d.model}, adapter {'yes' if d.used_adapter else 'no'}, cost ${d.cost_usd:.4f} {d.note}")


@app.command()
def mcp(tenant: str = typer.Option(...), user: str = typer.Option(...)) -> None:
    """Serve the MCP gateway over stdio, bound to one firm and user."""
    from sib.corpus import build_corpus
    from sib.mcp_gateway import DriveConnector, McpGateway, McpServer

    logging.basicConfig(level=logging.WARNING)  # stdout is the protocol channel
    McpServer(McpGateway(DriveConnector(build_corpus().documents)), tenant, user).serve()


@app.command()
def billing(period: str = typer.Option(..., help="YYYY-MM")) -> None:
    """Invoice each subscription from the metered usage recorded by the API."""
    from sib.billing import LocalGateway, build_invoice
    from sib.metering import Meter
    from sib.subscriptions import SUBSCRIPTIONS

    settings = get_settings()
    meter = Meter.load(Path(settings.state_dir) / "usage.jsonl")
    counts = meter.requests_by_tenant()
    gateway = LocalGateway(Path(settings.state_dir) / "invoices")
    for sub in SUBSCRIPTIONS:
        invoice = build_invoice(sub, counts.get(sub.tenant_id, 0), period)
        key = gateway.submit(invoice)
        typer.echo(f"{key}: ${invoice.total} ({counts.get(sub.tenant_id, 0)} requests)")


if __name__ == "__main__":
    app()
