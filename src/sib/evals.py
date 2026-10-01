"""Quality gates (`make evals`): retrieval, faithfulness, regressions, latency, router,
overfitting, tenant isolation, cache precision and the COGS budget.

Blocking gates fail the release; alerts (gross margin below target) are reported but do not block.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sib.app import FRONTIER, Stack, StackConfig, drafting_assets, trained_router
from sib.corpus import build_corpus
from sib.economics import Scenario, ai_cogs_per_account, gross_margin
from sib.ledger import QUALITY_GATE_PTS, unit_costs
from sib.router import INTENTS, synthetic_requests, teacher_intent_probs
from sib.workload import BenchResult, run_workload

P95_BUDGET_MS = 6000
FAITHFULNESS_MIN = 0.95
RECALL_MIN = 0.95
COGS_SHARE_OF_ARPA_MAX = 0.20


@dataclass
class Gate:
    name: str
    value: float
    threshold: float
    passed: bool
    blocking: bool = True
    detail: str = ""


def _gate(name: str, value: float, threshold: float, higher_is_better: bool = True, **kw: object) -> Gate:
    passed = value >= threshold if higher_is_better else value <= threshold
    return Gate(name, round(float(value), 4), threshold, passed, **kw)  # type: ignore[arg-type]


def retrieval_recall(stack: Stack) -> float:
    qs = stack.corpus.questions
    hits = [any(q.expected in r.chunk.text for r in stack.index.retrieve(q.tenant_id, q.matter_id, q.question, stack.config.top_k))
            for q in qs]
    return float(np.mean(hits))


def tenant_isolation_violations(stack: Stack) -> int:
    violations = 0
    tenants = list(stack.corpus.tenants)
    for q in stack.corpus.questions:
        answer = stack.ask(q.tenant_id, q.matter_id, q.question)
        for cite in answer.citations:
            chunk = stack.index._chunks.get(q.tenant_id, {}).get(cite.chunk_id)
            violations += chunk is None or chunk.matter_id != q.matter_id
        for other in tenants:
            if other == q.tenant_id:
                continue
            try:
                stack.ask(other, q.matter_id, q.question)
                violations += 1
            except KeyError:
                pass
    return violations


def semantic_false_hits(config: StackConfig) -> int:
    """Ask every question twice in a matter-shuffled order; a cached answer to a different clause is a false hit."""
    stack = Stack(config.with_(semantic_cache=True, exact_cache=False))
    wrong = 0
    for q in stack.corpus.questions + stack.corpus.questions[::-1]:
        answer = stack.ask(q.tenant_id, q.matter_id, q.question)
        wrong += answer.cache_layer == "semantic" and q.expected not in answer.text
    return wrong


def router_gates(config: StackConfig) -> list[Gate]:
    router = trained_router(config.tuned_embeddings, config.selfhost)
    corpus = build_corpus()
    held_out = synthetic_requests([q.question for q in corpus.questions], n_per_intent=40, seed=99)
    agree = np.mean([router.classify(t)[0] == INTENTS[int(teacher_intent_probs(t).argmax())] for t in held_out])
    stack = Stack(config.with_(router=True, exact_cache=False, semantic_cache=False))
    cheap = [(q, a) for q in corpus.questions if (a := stack.ask(q.tenant_id, q.matter_id, q.question)).model != FRONTIER]
    precision = float(np.mean([q.expected in a.text for q, a in cheap])) if cheap else 1.0
    return [
        _gate("router.intent_agreement_with_teacher", float(agree), 0.90),
        _gate("router.cheap_route_precision", precision, 0.98, detail=f"{len(cheap)} asks answered by non-frontier models"),
    ]


def overfitting_gates() -> list[Gate]:
    gates = []
    for tenant_id, report in drafting_assets().reports.items():
        detail = f"rank {report.rank}, {report.epochs_run} epochs, gap {report.generalisation_gap_pts:.1f} pts, {report.reason}"
        if report.accepted:
            gates.append(_gate(f"lora.{tenant_id}.holdout_gain_pts", report.gain_pts, 2.0, detail=detail))
            gates.append(_gate(f"lora.{tenant_id}.canary_memorised", report.canary_memorised, 0.5, higher_is_better=False))
        else:  # falling back to the base model is the guard working, not a failure
            gates.append(Gate(f"lora.{tenant_id}.fallback_to_base", 1.0, 1.0, True, detail=detail))
    return gates


def run_gates(config: StackConfig, scenario: Scenario, baseline: dict[str, float] | None, margin_target: float,
              result: BenchResult | None = None) -> tuple[list[Gate], BenchResult]:
    result = result or run_workload(config)
    stack = Stack(config.with_(exact_cache=False, semantic_cache=False))
    gates = [
        _gate("rag.recall_at_k", retrieval_recall(stack), RECALL_MIN, detail=f"k={config.top_k}"),
        _gate("answer.faithfulness", result.faithfulness, FAITHFULNESS_MIN),
        _gate("latency.p95_ask_ms", result.p95_ask_ms, P95_BUDGET_MS, higher_is_better=False),
        _gate("isolation.violations", tenant_isolation_violations(stack), 0, higher_is_better=False),
        _gate("cache.semantic_false_hits", semantic_false_hits(config), 0, higher_is_better=False,
              blocking=config.semantic_cache,
              detail="" if config.semantic_cache else "semantic cache is off in this config; shown for information"),
    ]
    for metric in ("ask_quality", "draft_quality"):
        value = getattr(result, metric)
        if baseline and metric in baseline:
            gates.append(_gate(f"regression.{metric}", value, round(baseline[metric] - QUALITY_GATE_PTS, 2),
                               detail=f"baseline {baseline[metric]:.2f}, max drop {QUALITY_GATE_PTS} pts"))
        else:
            gates.append(Gate(f"regression.{metric}", round(value, 2), 0.0, True, detail="no baseline recorded yet"))
    if config.router:
        gates += router_gates(config)
    gates += overfitting_gates()
    unit = unit_costs(result, config, scenario)
    share = float(ai_cogs_per_account(scenario, unit) / scenario.arpa)
    gates.append(_gate("budget.ai_cogs_share_of_arpa", share, COGS_SHARE_OF_ARPA_MAX, higher_is_better=False))
    gates.append(_gate("business.gross_margin", gross_margin(scenario, unit), margin_target, blocking=False,
                       detail="alert only: margin is managed through the ledger, not the release gate"))
    return gates, result


def load_baseline(path: Path) -> dict[str, float] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_baseline(path: Path, result: BenchResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ask_quality": round(result.ask_quality, 2), "draft_quality": round(result.draft_quality, 2),
                                "faithfulness": round(result.faithfulness, 4)}, indent=2) + "\n", encoding="utf-8")


def gates_to_json(gates: list[Gate]) -> list[dict[str, object]]:
    return [asdict(g) for g in gates]
