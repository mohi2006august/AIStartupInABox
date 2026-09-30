"""A reproducible request stream for benchmarks: lawyers working matters in sessions.

Questions repeat (Zipf) and are often rephrased slightly, which is what makes caching pay. Time
advances between requests so prompt-cache TTLs expire realistically between sessions.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from sib.app import Stack, StackConfig, drafting_assets
from sib.corpus import Corpus, EvalQuestion
from sib.drafting import AUDIENCES, DOC_TYPES, style_match
from sib.llm import ABSTAIN, CATALOG
from sib.metering import percentile

ACTIONS = {"lease": "remedy the breach of the lease", "employment": "confirm compliance with the employment terms",
           "supply": "settle the outstanding invoices"}


@dataclass
class Request:
    kind: str  # ask | draft
    question: EvalQuestion
    text: str  # the question as typed (possibly rephrased)
    gap_s: float
    doc_type: str = ""
    audience: str = ""


def rephrase(question: str, rng: random.Random) -> str:
    return rng.choice([question, question, question, question.lower().rstrip("?"), "Quick question - " + question,
                       question + " Thanks."])


def build_workload(corpus: Corpus, n_sessions: int = 70, draft_share: float = 0.15, seed: int = 11) -> list[Request]:
    rng = random.Random(seed)
    by_matter: dict[str, list[EvalQuestion]] = {}
    for q in corpus.questions:
        by_matter.setdefault(q.matter_id, []).append(q)
    for qs in by_matter.values():
        rng.shuffle(qs)
    out: list[Request] = []
    matters = sorted(by_matter)
    for _ in range(n_sessions):
        qs = by_matter[rng.choice(matters)]
        weights = [1 / (rank + 1) for rank in range(len(qs))]
        for i in range(rng.randint(6, 10)):
            q = rng.choices(qs, weights)[0]
            gap = 900.0 if i == 0 else 40.0
            if rng.random() < draft_share:
                out.append(Request("draft", q, q.question, gap, rng.choice(DOC_TYPES), rng.choice(AUDIENCES)))
            else:
                out.append(Request("ask", q, rephrase(q.question, rng), gap))
    return out


@dataclass
class BenchResult:
    requests: int
    ask_quality: float  # 0-100: answer contains the expected value
    draft_quality: float  # 0-100: half house-style match, half correct provision
    faithfulness: float  # share of non-abstaining answers that are verbatim in the matter's documents
    p95_ask_ms: int
    cogs_per_1k: float  # blended over the ask/draft mix
    cost_per_ask: float
    cost_per_draft: float
    selfhost_cost_per_request: float
    cache_hit_ratio: float
    prompt_cache_read_share: float
    model_mix: dict[str, float]
    escalation_rate: float
    abstain_rate: float


def run_workload(config: StackConfig, corpus: Corpus | None = None, workload: list[Request] | None = None) -> BenchResult:
    stack = Stack(config, corpus=corpus)
    workload = workload or build_workload(stack.corpus)
    assets = drafting_assets()
    matter_text: dict[str, str] = {}
    for doc in stack.corpus.documents:
        matter_text[doc.matter_id] = matter_text.get(doc.matter_id, "") + " " + doc.text
    ask_ok, draft_scores, faithful, abstained = [], [], [], 0
    for req in workload:
        stack.clock.advance(req.gap_s)
        q = req.question
        if req.kind == "ask":
            answer = stack.ask(q.tenant_id, q.matter_id, req.text)
            ask_ok.append(q.expected in answer.text)
            if answer.text == ABSTAIN:
                abstained += 1
            else:
                faithful.append(answer.text in matter_text[q.matter_id])
        else:
            kind = stack.matter(q.tenant_id, q.matter_id).kind
            draft = stack.draft(q.tenant_id, q.matter_id, req.doc_type, req.audience, req.text, ACTIONS[kind])
            style = style_match(draft.choices, assets.styles[q.tenant_id], req.doc_type, req.audience)
            draft_scores.append(0.5 * style + 0.5 * float(q.expected in draft.text))
    meter = stack.meter
    asks, drafts = meter.select(feature="ask"), meter.select(feature="draft")
    selfhost_cost = sum(float(e.cost_usd) for e in meter.events if CATALOG[e.model].tier == "self_hosted")
    return BenchResult(
        requests=len(workload),
        ask_quality=100 * float(np.mean(ask_ok)),
        draft_quality=100 * float(np.mean(draft_scores)) if draft_scores else 0.0,
        faithfulness=float(np.mean(faithful)) if faithful else 1.0,
        p95_ask_ms=percentile([e.latency_ms for e in asks], 0.95),
        cogs_per_1k=float(meter.cost_per_1k()),
        cost_per_ask=float(meter.total_cost("ask")) / max(len(asks), 1),
        cost_per_draft=float(meter.total_cost("draft")) / max(len(drafts), 1),
        selfhost_cost_per_request=selfhost_cost / len(workload),
        cache_hit_ratio=meter.cache_hit_ratio(),
        prompt_cache_read_share=meter.prompt_cache_read_share(),
        model_mix=meter.model_mix(),
        escalation_rate=sum(e.escalated for e in meter.events) / len(meter.events),
        abstain_rate=abstained / max(len(ask_ok), 1),
    )
