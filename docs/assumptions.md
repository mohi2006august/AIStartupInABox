# Assumptions and simulation boundaries

Everything runs offline with no API keys. That makes the whole loop reproducible, but it means some
numbers are measured and some are modelled. This page lists which is which, so nobody mistakes a
modelled number for a measured one.

## Measured by running code

| What | Where |
|---|---|
| Retrieval recall, answer correctness, faithfulness | `sib.evals`, `sib.workload` on the synthetic corpus |
| Embedding tuning (InfoNCE on a linear adapter) and its recall gain | `sib.embeddings`, `sib.rag` |
| Cache hit ratios (exact, semantic, prompt prefix) and semantic-cache false hits | `sib.caches`, `sib.workload` |
| Router distillation, threshold calibration and routed quality | `sib.router`, `sib.app.trained_router` |
| LoRA training, early stopping, canary memorisation, held-out gain | `sib.drafting` |
| Quantisation error (int8, int4, AWQ-style scaling) and accuracy after quantising | `sib.quantize` |
| Token counts per request (approximate tokenizer) | `sib.tokens` |
| Economics arithmetic (margin, CAC, payback, burn, runway) | `sib.economics`, checked against fixtures in `tests/test_business.py` |

## Modelled (assumptions you should replace with your own numbers)

| Assumption | Value | Where |
|---|---|---|
| Claude API prices | Opus 5.5 $4 / $20 per MTok, Haiku 4.5 $1 / $5; cache reads ~0.1x, writes 1.25x | `sib.llm.CATALOG` |
| Model capability in the simulator | vocabulary knowledge: Opus 1.0, Haiku 0.66, 8B fp16 0.50, 8B AWQ 0.45 | `sib.llm.CATALOG` |
| Latency | time to first token + prefill + decode; Opus 12 ms/token, Haiku 6, 8B AWQ 9; 0-40% deterministic tail | `sib.llm.latency_ms` |
| Production token scale | the synthetic clauses are terse, so input tokens are priced at 25x and output at 8x | `StackConfig.input_token_scale`, `output_token_scale` |
| GPU node | $1.20/hour, 1,500 tok/s at fp16, 40% utilisation; int4 up to 3.4x faster | `sib.quantize` |
| Self-hosting cost | the whole node is paid for; traffic that does not fill it is idle cost | `sib.ledger.unit_costs` |
| Business scenario | cash $350k, opex $22k/month, 10 customers x 5 seats at $149, 900 requests per seat, 15% drafts, other COGS $110/account, 2.5% monthly churn, three channels | `src/sib/data/business.json` |

## Known optimism

- The embedding adapter's query-generation settings (word dropout, two synthetic queries per clause)
  were chosen while looking at hard-question recall on the eval set. Treat the 1.0 recall@4 as a
  ceiling for this corpus, not a forecast for real firm documents.
- The simulator's "models" answer by extracting a sentence. Real models paraphrase, so the
  faithfulness check (`answer text appears verbatim in the matter's documents`) is stricter than a
  production verifier can be; production needs an entailment-style check.
- The router's training corpus uses different matters and paraphrases from the eval corpus, but the
  same clause families. A new clause type would need new training data.

## Swapping in real components

| Component | Today | Production |
|---|---|---|
| LLMs | `SimProvider` | `ClaudeProvider` (`pip install -e ".[claude]"`, set `LIVE_MODELS=true` and credentials); vLLM for self-hosted tiers |
| Vector store | in-memory, per-tenant collections | Postgres + pgvector with row-level security |
| MCP connectors | in-memory drive, email and calendar | Google Drive / SharePoint, Gmail / Graph, calendar APIs behind the same three classes |
| Billing | `LocalGateway` writes invoices as JSON when `STRIPE_KEY` is unset | `StripeGateway` (draft invoices, idempotent, webhook status); tested against a fake client and Stripe's signature scheme, not yet against a real test-mode account |
| Tenant identity | `X-Tenant-Id` header | SSO session |
