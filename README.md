# AI Startup-in-a-Box

> The ultimate capstone that uses all 15 concepts: build, launch and run a real AI SaaS product end to end, from model stack to unit economics, and make it profitable.

**Concepts:** `RAG`, `MCP`, `Vector Database`, `Tokenization`, `Context Window`, `Fine Tuning`, `Overfitting`, `Embeddings`, `Knowledge Distillation`, `LoRA`, `Burn Rate`, `CAC`, `LLMs`, `Caches`, `Quantization`

A vertical AI SaaS, an **AI research assistant for small law firms**, wired to its own unit economics. Lawyers ask cited questions about their matters, draft letters in their firm's house style, and approve email and calendar actions. Founders see COGS, gross margin, burn, runway and CAC payback, and every technical change is benchmarked into a decision ledger with its dollar impact.

Everything runs offline with no API keys: a deterministic simulator stands in for the models so the whole loop (product, evals, economics, ledger) is reproducible. [docs/assumptions.md](docs/assumptions.md) lists what is measured and what is modelled.

## Quick start

```bash
pip install -e ".[dev]"
make test        # unit, isolation and economics-fixture tests
make evals       # quality gates on the production config
make economics   # margin, burn, runway, CAC payback from measured unit costs
make dev         # API + web console on http://127.0.0.1:8000
```

`make` targets set `PYTHONPATH=src`, so they also work without installing. Without `make`, run `python -m sib <command>`.

| Command | What it does |
|---|---|
| `sib bench --change latest` | Measure the next change before vs after and record it in `ledger/` |
| `sib bench --all` | Re-measure the whole change sequence |
| `sib bench --change selfhost --customers 50 --dry-run` | See how a decision changes at a different scale |
| `sib ledger` | Print the decision ledger |
| `sib ask firm_aurora aurora-m1 "How much warning is needed to end the tenancy?"` | Cited answer from one matter |
| `sib draft firm_aurora aurora-m1 "How much warning is needed to end the tenancy?"` | Letter in the firm's style, grounded in a provision |
| `sib mcp --tenant firm_aurora --user lawyer1` | MCP server over stdio (drive, email, calendar) |
| `sib billing --period 2026-10` | Invoice metered API usage |

## Results so far

From [ledger/LEDGER.md](ledger/LEDGER.md), measured on the same replayed workload of 566 requests, priced at 10 customers:

| Change | Quality | p95 | COGS/1k | Gross margin | Runway | Decision |
|---|---|---|---|---|---|---|
| Contrastively tuned embeddings | +30.9 | = | +$0.76 | −0.4 pts | −0.1 mo | kept |
| Exact + semantic + prompt caching | = | −0.3 s | −$22.25 | +13.6 pts | +10.3 mo | kept |
| Distilled router (easy asks to Haiku) | = | = | −$2.16 | +1.5 pts | → default alive | kept |
| Self-host AWQ 8B + per-firm LoRA | +3.9 | = | −$14.77 | −3.5 pts | = | reverted |
| Context budget 8,000 → 6,800 tokens | −2.6 ✖ | −0.1 s | −$1.51 | +0.9 pts | = | reverted |

Gross margin 59% → 73%, CAC payback 8.1 → 6.4 months, and the company goes from default dead (cash out in month 19) to default alive.

Two results are worth reading closely:

- **Self-hosting is the right move later, not now.** It cuts COGS per 1k requests by two thirds and improves draft quality, but at 10 customers the GPU node sits mostly idle and margin falls. The bench finds the break-even at about 15 customers; at 50 customers the same change adds +5.8 points of margin. The ledger keeps it reverted until there is volume to fill the node.
- **Gross margin is still below the 75% target** (73.4%). `make evals` reports this as an alert rather than a release blocker. The next lever is self-hosting once the customer count passes break-even.

## How it fits together

```text
question ─► context manager ─► hybrid retrieval ─► token budget ─► caches ─► router ─► model ─► verify ─► meter
            (matter memory)     (BM25 + tuned       (per feature)  (exact,    (distilled  (Opus /   (cited claim
                                 embeddings, RRF)                   semantic,   intent +    Haiku /   supported?
                                                                    prompt,     difficulty) 8B AWQ)   else escalate)
                                                                    embedding)
```

| Requirement | Module |
|---|---|
| FR-01 Cited RAG | `rag.py` (chunking, BM25, RRF, verification), `embeddings.py` (InfoNCE-tuned adapter), `vectordb.py` (per-tenant collections, IVF) |
| FR-02 MCP integrations | `mcp_gateway.py` (approval-gated writes, audit log, JSON-RPC stdio server) |
| FR-03 Metering + billing | `metering.py`, `billing.py`, `subscriptions.py` |
| FR-04 Economics console | `economics.py`, `api.py`, `web/index.html` |
| FR-05 Long-matter context | `context.py` (pinned facts, compaction, budgeted packing, lost-in-the-middle ordering) |
| FR-06 Drafting model | `drafting.py` (SFT base, per-firm LoRA, early stopping, canary probe, fallback to base) |
| FR-07 Distilled router | `router.py`, `app.trained_router` (soft-label distillation, cost-calibrated threshold) |
| FR-08 Quantised self-hosting | `quantize.py` (int8/int4, AWQ-style scaling, serving cost per MTok) |
| FR-09 Caching stack | `caches.py` |
| FR-10 Tokenizer-aware prompts | `tokens.py` (byte-level BPE, fertility, per-feature budgets, compact JSON) |
| FR-11 Decision ledger | `ledger.py`, `workload.py`, `evals.py` |

Planning documents:

| File | Contents |
|---|---|
| [BRAIN.md](BRAIN.md) | Concept primers, experiments, decision log |
| [CLAUDE.md](CLAUDE.md) | Instructions for AI coding assistants |
| [DESIGN.md](DESIGN.md) | UX flows, screens, visual language |
| [PRD.md](PRD.md) | Problem, users, requirements, metrics, roadmap |
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) | Components, data flow, deployment, failure modes |
| [TECHNICAL.md](TECHNICAL.md) | Stack, data models, APIs, algorithms |

## API

`make dev` serves the web console at `/` and these endpoints. The tenant comes from the `X-Tenant-Id` header, a stand-in for SSO.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/v1/matters/{id}/ask` | Cited Q&A |
| POST | `/v1/matters/{id}/assist` | One box: the distilled intent classifier picks ask, summarize, draft or action |
| GET | `/v1/matters/{id}/summary` | Cited matter summary |
| POST | `/v1/drafts` | Draft in firm style |
| GET | `/v1/economics/summary?customers=N` | Margin, burn, runway, CAC payback, cash curve vs the MVP stack |
| GET / POST | `/v1/ledger` | Read the ledger / record a decision measured elsewhere |
| POST | `/v1/ledger/bench` | Measure a change and switch the stack to the new production config |
| GET | `/v1/ops` | Model mix, cache hits, router table, adapter status |
| POST | `/v1/tools/{name}`, `/v1/approvals/{id}/approve` | MCP tools; writes wait for approval |

## Using real models

```bash
pip install -e ".[claude]"
export LIVE_MODELS=true   # plus ANTHROPIC_API_KEY or an `ant auth login` profile
make dev
```

API-tier requests in the console then go to Claude (Opus 5.5 with prompt caching and server-side refusal fallbacks, Haiku 4.5 for routed easy asks). Benchmarks and economics always replay on the simulator so the ledger stays reproducible.

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md). Built by **mohi2006august**, with Claude as pair-programming co-author.
