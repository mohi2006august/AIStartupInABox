"""The stack: one object that wires every component into `ask`, `draft` and `summarize`.

Request flow (SYSTEM_ARCHITECTURE.md section 3): context manager -> retrieval -> token budget ->
caches -> router -> model -> verification -> metering. `StackConfig` holds the switches that
`sib bench` flips to measure a change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from decimal import Decimal
from functools import lru_cache

import numpy as np

from sib.caches import CachedEmbedder, ExactCache, PromptCache, SemanticCache
from sib.context import ContextManager
from sib.corpus import Corpus, Matter, build_corpus, stems
from sib.drafting import (
    AdapterReport, DraftRequest, FirmStyle, LoRAAdapter, StyleModel, featurize, few_shot_choices, fit_firm_adapter,
    model_choices, render_draft, sample_examples,
)
from sib.embeddings import STOPWORDS, HashingEmbedder, TunedEmbedder
from sib.llm import ABSTAIN, CATALOG, FRONTIER, LLMRequest, Provider, Usage, cost_usd, latency_ms, make_provider
from sib.mcp_gateway import DriveConnector, McpGateway
from sib.metering import Meter, UsageEvent
from sib.quantize import awq_quantize, dequantize
from sib.rag import Chunk, RagIndex, is_supported
from sib.router import DistilledRouter, choose_models, synthetic_requests, teacher_intent_probs
from sib.tokens import TokenBudgeter, count_tokens

log = logging.getLogger("sib.app")

SELF_HOSTED = "sib-8b-awq"
CHEAP_API = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are a legal research assistant working inside a law firm's matter workspace. Answer strictly from the "
    "passages supplied for this matter and cite the passage you relied on. If the passages do not contain the "
    "answer, say so plainly instead of guessing: a confident wrong answer is worse than no answer. Never draw on "
    "documents from another matter or another firm. Quote operative wording exactly, keep defined terms "
    "capitalised as drafted, and do not give an opinion on the merits unless asked. You assist a qualified lawyer "
    "who remains responsible for the advice; nothing you produce is filed or sent without their approval."
)


@dataclass(frozen=True)
class StackConfig:
    tuned_embeddings: bool = True
    exact_cache: bool = False
    semantic_cache: bool = False
    prompt_cache: bool = False
    router: bool = False
    selfhost: bool = False  # a GPU node serves drafting (AWQ base + per-firm LoRA) and easy routed asks
    ask_budget: int = 8000  # production tokens per ask prompt
    draft_budget: int = 12000
    top_k: int = 4
    few_shot_examples: int = 20
    semantic_threshold: float = 0.82
    # The synthetic corpus is far terser than real contracts. Token counts are multiplied by these
    # factors before pricing so unit costs sit at a production-like scale. They are assumptions.
    input_token_scale: int = 25
    output_token_scale: int = 8
    live: bool = False  # call the real Claude API for API-tier models

    def with_(self, **changes: object) -> "StackConfig":
        return replace(self, **changes)


class SimClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class Citation:
    doc_id: str
    chunk_id: str
    title: str
    quote: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    model: str
    route_class: str = ""
    cache_layer: str = "none"
    supported: bool = True
    escalated: bool = False
    cost_usd: Decimal = Decimal(0)
    latency_ms: int = 0


@dataclass
class Draft:
    text: str
    choices: dict[str, int]
    model: str
    used_adapter: bool
    fact: Answer
    cost_usd: Decimal
    latency_ms: int
    note: str = ""


@dataclass
class _Prepared:
    prefix: str
    passages: list[str]
    chunks: list[Chunk]  # aligned with passages
    features: tuple[float, float, float]
    dropped: int


def _add(a: Usage, b: Usage) -> Usage:
    return Usage(a.input_tokens + b.input_tokens, a.cache_read_tokens + b.cache_read_tokens,
                 a.cache_write_tokens + b.cache_write_tokens, a.output_tokens + b.output_tokens)


# --- Trained assets, built once per process ------------------------------------------------

FIRM_EXAMPLE_COUNTS = {"firm_aurora": 300, "firm_birch": 300, "firm_cedar": 60}


@dataclass
class DraftingAssets:
    base: StyleModel
    quantised: StyleModel
    styles: dict[str, FirmStyle]
    examples: dict[str, list]
    adapters: dict[str, LoRAAdapter]
    reports: dict[str, AdapterReport] = field(default_factory=dict)


@lru_cache(maxsize=1)
def drafting_assets() -> DraftingAssets:
    base = StyleModel()
    base.train_base(sample_examples(3000, seed=1))
    calib = np.stack([featurize(s, d, a) for s, d, a, _ in sample_examples(200, seed=4)])
    quantised = StyleModel()
    quantised.W = dequantize(awq_quantize(base.W, calib, bits=4)[0])
    assets = DraftingAssets(base, quantised, {}, {}, {})
    for tenant_id, n in FIRM_EXAMPLE_COUNTS.items():
        style = FirmStyle.generate(tenant_id)
        examples = sample_examples(n, seed=5, style=style)
        adapter, report = fit_firm_adapter(base, tenant_id, examples, rank=8)
        assets.styles[tenant_id], assets.examples[tenant_id], assets.reports[tenant_id] = style, examples, report
        if adapter is not None:
            assets.adapters[tenant_id] = adapter
        log.info("firm_adapter", extra={"tenant": tenant_id, "accepted": report.accepted, "reason": report.reason})
    return assets


@lru_cache(maxsize=2)
def tuned_embedder(tuned: bool) -> HashingEmbedder | TunedEmbedder:
    base = HashingEmbedder()
    if not tuned:
        return base
    embedder = TunedEmbedder(base)
    embedder.fit(build_corpus().training_pairs)
    return embedder


class Stack:
    def __init__(self, config: StackConfig | None = None, corpus: Corpus | None = None, clock: SimClock | None = None) -> None:
        self.config = config or StackConfig()
        self.corpus = corpus or build_corpus()
        self.clock = clock or SimClock()
        c = self.config
        self.embedder = CachedEmbedder(tuned_embedder(c.tuned_embeddings))
        self.index = RagIndex(self.embedder)
        self.index.ingest(self.corpus.documents)
        self.budgeter = TokenBudgeter({"ask": c.ask_budget // c.input_token_scale, "draft": c.draft_budget // c.input_token_scale})
        self.context = ContextManager(self.budgeter)
        self.exact = ExactCache(clock=self.clock)
        self.semantic = SemanticCache(threshold=c.semantic_threshold, clock=self.clock)
        self.prompt_cache = PromptCache(min_tokens=max(512 // c.input_token_scale, 1), clock=self.clock)
        self.meter = Meter()
        self.gateway = McpGateway(DriveConnector(self.corpus.documents))
        self.selfhost_healthy = True
        self._providers: dict[str, Provider] = {}
        self._matters: dict[tuple[str, str], Matter] = {(m.tenant_id, m.id): m for m in self.corpus.matters}
        for m in self.corpus.matters:
            self.context.pin(m.tenant_id, m.id, f"Matter: {m.name} ({m.kind}).")
        self.router: DistilledRouter | None = trained_router(c.tuned_embeddings, c.selfhost) if c.router else None

    # --- helpers ---------------------------------------------------------------------------

    def provider(self, model_id: str) -> Provider:
        if model_id not in self._providers:
            self._providers[model_id] = make_provider(model_id, live=self.config.live)
        return self._providers[model_id]

    def matter(self, tenant_id: str, matter_id: str) -> Matter:
        matter = self._matters.get((tenant_id, matter_id))
        if matter is None:  # same error whether it does not exist or belongs to another firm
            raise KeyError(f"matter '{matter_id}' not found")
        return matter

    def _scaled(self, usage: Usage) -> Usage:
        s_in, s_out = self.config.input_token_scale, self.config.output_token_scale
        return Usage(usage.input_tokens * s_in, usage.cache_read_tokens * s_in, usage.cache_write_tokens * s_in,
                     usage.output_tokens * s_out)

    def _prepare(self, tenant_id: str, matter_id: str, question: str, feature: str = "ask") -> _Prepared:
        hits = self.index.retrieve(tenant_id, matter_id, question, k=self.config.top_k)
        packed = self.context.build(tenant_id, matter_id, feature, SYSTEM_PROMPT, [h.chunk.text for h in hits], question)
        chunks = [hits[r].chunk for r in packed.passage_ranks]
        terms = {w for w in stems(question) if w not in STOPWORDS}
        cover = [len(terms & set(stems(p))) / len(terms) if terms else 0.0 for p in packed.passages]
        features = (cover[0] if cover else 0.0, max(cover, default=0.0), len(terms) / 20)
        return _Prepared(packed.prefix, packed.passages, chunks, features, len(hits) - len(chunks))

    def _complete(self, model_id: str, prep: _Prepared, question: str) -> tuple[str, Citation | None, bool, Usage]:
        req = LLMRequest(prep.prefix, prep.passages, question)
        if self.config.prompt_cache and CATALOG[model_id].tier != "self_hosted":
            req.cached_prefix_tokens, req.cache_write_tokens = self.prompt_cache.apply(model_id, prep.prefix)
        out = self.provider(model_id).complete(req)
        usage = self._scaled(out.usage)
        if out.citation is None or out.text == ABSTAIN:
            return ABSTAIN, None, True, usage
        chunk = prep.chunks[out.citation - 1]
        citation = Citation(chunk.doc_id, chunk.id, chunk.title, chunk.text)
        return out.text, citation, is_supported(out.text, chunk.text), usage

    def _pick_model(self, question: str, prep: _Prepared) -> tuple[str, str]:
        if self.router is None:
            return FRONTIER, ""
        decision = self.router.route(question, prep.features, FRONTIER, intent="ask")
        if CATALOG[decision.model].tier == "self_hosted" and not self.selfhost_healthy:
            log.warning("selfhost_failover", extra={"class": decision.request_class, "to": FRONTIER})
            return FRONTIER, decision.request_class + ":failover"
        return decision.model, decision.request_class

    def _answer(self, model_id: str, prep: _Prepared, question: str) -> tuple[Answer, Decimal, int, Usage]:
        """Generate, verify, and escalate to the frontier model once if the claim is unsupported."""
        text, citation, supported, usage = self._complete(model_id, prep, question)
        spec = CATALOG[model_id]
        cost, latency = cost_usd(spec, usage), latency_ms(spec, usage, question + model_id) + 60
        escalated, final_model = False, model_id
        if not supported and model_id != FRONTIER:
            text, citation, supported, usage2 = self._complete(FRONTIER, prep, question)
            cost += cost_usd(CATALOG[FRONTIER], usage2)
            latency += latency_ms(CATALOG[FRONTIER], usage2, question + FRONTIER)
            escalated, final_model = True, FRONTIER
            usage = _add(usage, usage2)
        if not supported:
            text, citation = ABSTAIN, None
        answer = Answer(text, [citation] if citation else [], final_model, supported=supported, escalated=escalated)
        return answer, cost, latency, usage

    def _meter(self, tenant_id: str, feature: str, answer_model: str, cost: Decimal, latency: int,
               usage: Usage | None = None, **fields: object) -> None:
        u = usage or Usage()
        self.meter.record(UsageEvent(
            tenant_id=tenant_id, feature=feature, model=answer_model, cost_usd=cost, latency_ms=latency,
            input_tokens=u.input_tokens, cache_read_tokens=u.cache_read_tokens, cache_write_tokens=u.cache_write_tokens,
            output_tokens=u.output_tokens, **fields))

    # --- features --------------------------------------------------------------------------

    def ask(self, tenant_id: str, matter_id: str, question: str, *, remember: bool = False, trial: bool = False,
            model: str | None = None) -> Answer:
        """Cited Q&A over one matter (FR-01). `model` pins the model, bypassing caches and router."""
        self.matter(tenant_id, matter_id)
        c = self.config
        use_caches = model is None
        if use_caches and c.exact_cache and (hit := self.exact.get(tenant_id, matter_id, question)) is not None:
            return self._from_cache(tenant_id, hit, "exact", 15, trial)
        query_vec = self.embedder.embed_one(question)
        if use_caches and c.semantic_cache and (hit := self.semantic.get(tenant_id, matter_id, query_vec)) is not None:
            return self._from_cache(tenant_id, hit, "semantic", 45, trial)
        prep = self._prepare(tenant_id, matter_id, question)
        model_id, route_class = (model, "pinned") if model else self._pick_model(question, prep)
        answer, cost, latency, usage = self._answer(model_id, prep, question)
        answer.route_class, answer.cost_usd, answer.latency_ms = route_class, cost, latency
        self._meter(tenant_id, "ask", answer.model, cost, latency, usage, route_class=route_class,
                    escalated=answer.escalated, trial=trial)
        if use_caches and answer.supported and answer.text != ABSTAIN:
            if c.exact_cache:
                self.exact.put(tenant_id, matter_id, question, answer)
            if c.semantic_cache:
                self.semantic.put(tenant_id, matter_id, query_vec, answer)
        if remember:
            self.context.record(tenant_id, matter_id, "user", question)
            self.context.record(tenant_id, matter_id, "assistant", answer.text)
        return answer

    def _from_cache(self, tenant_id: str, cached: Answer, layer: str, latency: int, trial: bool) -> Answer:
        answer = replace(cached, cache_layer=layer, cost_usd=Decimal(0), latency_ms=latency, escalated=False)
        self._meter(tenant_id, "ask", cached.model, Decimal(0), latency, cache_layer=layer, trial=trial)
        return answer

    def draft(self, tenant_id: str, matter_id: str, doc_type: str, audience: str, instruction: str, action: str,
              days: int = 14, trial: bool = False) -> Draft:
        """Draft in the firm's style (FR-06), grounded in one provision retrieved from the matter."""
        matter = self.matter(tenant_id, matter_id)
        assets = drafting_assets()
        adapter = assets.adapters.get(tenant_id)
        note = ""
        if self.config.selfhost and adapter is not None and self.selfhost_healthy:
            model_id, used_adapter = SELF_HOSTED, True
            choices = model_choices(assets.quantised, doc_type, audience, adapter)
            style_prompt_tokens = 0
        else:
            model_id, used_adapter = FRONTIER, False
            shots = assets.examples[tenant_id][: self.config.few_shot_examples]
            choices = few_shot_choices(shots, doc_type, audience)
            style_prompt_tokens = sum(count_tokens(f"{s} {d} {a} {v}") for s, d, a, v in shots)
            if self.config.selfhost:
                note = "self-host node down: failed over to API" if adapter is not None else \
                    f"no accepted adapter ({assets.reports[tenant_id].reason}): API model with few-shot style"
        prep = self._prepare(tenant_id, matter_id, instruction, feature="draft")
        fact, fact_cost, fact_latency, fact_usage = self._answer(model_id, prep, instruction)
        client, counterparty = matter.name.split(" v. ")
        text = render_draft(choices, DraftRequest(tenant_id, doc_type, audience, client, counterparty, matter.name, action, days),
                            self.corpus.tenants[tenant_id])
        if fact.citations:
            text += f'\n\nProvision relied on: "{fact.text}" [{fact.citations[0].title}]'
        spec = CATALOG[model_id]
        usage = self._scaled(Usage(input_tokens=style_prompt_tokens + count_tokens(instruction) + count_tokens(action),
                                   output_tokens=count_tokens(text)))
        cost = fact_cost + cost_usd(spec, usage)
        latency = fact_latency + latency_ms(spec, usage, instruction + model_id)
        self._meter(tenant_id, "draft", model_id, cost, latency, _add(fact_usage, usage), escalated=fact.escalated, trial=trial,
                    route_class="draft:lora" if used_adapter else "draft:few_shot")
        return Draft(text, choices, model_id, used_adapter, fact, cost, latency, note)

    def summarize(self, tenant_id: str, matter_id: str, max_points: int = 5) -> list[Citation]:
        """Extractive matter summary: the leading provision of each agreement chunk, with sources."""
        self.matter(tenant_id, matter_id)
        chunks = [c for c in self.index._chunks.get(tenant_id, {}).values()
                  if c.matter_id == matter_id and c.doc_id.endswith("-agreement")][:max_points]
        spec = CATALOG[CHEAP_API]
        usage = self._scaled(Usage(input_tokens=count_tokens(SYSTEM_PROMPT) + sum(count_tokens(c.text) for c in chunks),
                                   output_tokens=sum(count_tokens(c.text) for c in chunks) // 2))
        self._meter(tenant_id, "summarize", CHEAP_API, cost_usd(spec, usage), latency_ms(spec, usage, matter_id), usage)
        return [Citation(c.doc_id, c.id, c.title, c.text) for c in chunks]


# --- Router distillation -------------------------------------------------------------------

QUALITY_TOLERANCE_PTS = 1.0


@lru_cache(maxsize=4)
def trained_router(tuned: bool, selfhost: bool = False) -> DistilledRouter:
    """Distil the router on a separate training corpus (different matters, different phrasings).

    Difficulty labels come from the models themselves: a question is "hard" to the extent that the
    cheap models disagree with the frontier model on it. The routing table then sends each class
    to the cheapest model whose measured quality stays within tolerance of the frontier's.
    """
    candidates = ([SELF_HOSTED] if selfhost else []) + [CHEAP_API, FRONTIER]
    train = build_corpus(seed=99, split="train")
    stack = Stack(StackConfig(tuned_embeddings=tuned), corpus=train)
    router = DistilledRouter()
    requests = synthetic_requests([q.question for q in train.questions])
    router.fit_intent(requests, np.stack([teacher_intent_probs(t) for t in requests]))
    features, p_hard, preps = [], [], []
    outcomes: dict[str, list[tuple[bool, Decimal]]] = {m: [] for m in candidates}
    for q in train.questions:
        prep = stack._prepare(q.tenant_id, q.matter_id, q.question)
        teacher = stack._complete(FRONTIER, prep, q.question)[0]
        disagree = [stack._complete(m, prep, q.question)[0] != teacher for m in candidates[:-1]]
        features.append(router.features(q.question, prep.features))
        p_hard.append(float(np.mean(disagree)))
        preps.append(prep)
        for model_id in candidates:  # quality and cost with verification and escalation included
            answer, cost, _, _ = stack._answer(model_id, prep, q.question)
            outcomes[model_id].append((q.expected in answer.text, cost))
    router.fit_difficulty(np.stack(features), np.array(p_hard, dtype=np.float32))
    scores = router.difficulty.predict_proba(np.stack(features))[:, 1]
    # Calibrate the easy/hard cut-off: a missed hard question costs quality, so pick the threshold
    # whose routing table is cheapest while every class stays within tolerance of the frontier.
    best: tuple[Decimal, float, dict[str, str], dict[str, dict[str, float]]] | None = None
    for threshold in (0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05):
        classes = {"ask:easy": np.flatnonzero(scores < threshold), "ask:hard": np.flatnonzero(scores >= threshold)}
        classes = {cls: idx for cls, idx in classes.items() if len(idx)}
        quality = {cls: {m: 100 * float(np.mean([outcomes[m][i][0] for i in idx])) for m in candidates}
                   for cls, idx in classes.items()}
        cost = {cls: {m: sum((outcomes[m][i][1] for i in idx), Decimal(0)) / len(idx) for m in candidates}
                for cls, idx in classes.items()}
        target = {cls: q[FRONTIER] - QUALITY_TOLERANCE_PTS for cls, q in quality.items()}
        table = choose_models(quality, cost, target, fallback=FRONTIER)
        total = sum((cost[cls][table[cls]] * len(idx) for cls, idx in classes.items()), Decimal(0))
        if best is None or total < best[0]:
            best = (total, threshold, table, quality)
    assert best is not None
    _, router.hard_threshold, router.table, quality = best
    router.report = {"quality": quality, "table": router.table, "hard_threshold": router.hard_threshold,
                     "label_hard_rate": float(np.mean(p_hard))}
    return router
