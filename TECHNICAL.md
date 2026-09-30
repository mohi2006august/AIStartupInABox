# AI Startup-in-a-Box: Technical Specification

> The ultimate capstone that uses all 15 concepts: build, launch and run a real AI SaaS product end to end, from model stack to unit economics, and make it profitable.
>
> **Concepts:** `RAG`, `MCP`, `Vector Database`, `Tokenization`, `Context Window`, `Fine Tuning`, `Overfitting`, `Embeddings`, `Knowledge Distillation`, `LoRA`, `Burn Rate`, `CAC`, `LLMs`, `Caches`, `Quantization`

## 1. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| App | Next.js + FastAPI | Fast iteration. |
| Data | Postgres + pgvector, Redis, S3 | Lean. |
| Models | Claude API + open models on vLLM | Quality + margin. |
| Training | TRL, PEFT, AutoAWQ | Customisation + compression. |
| Integrations | MCP servers | Standard. |
| Business | Stripe, warehouse + dbt | Billing + analytics. |

## 2. Repository Layout

```text
startup-in-a-box/
├── product/        # web/, api/, rag/, drafting/, context/
├── models/         # router_distill/, sft/, lora_per_firm/, quantize/, serving/
├── platform/       # mcp_gateway/, caches/, token_budgets/
├── business/       # metering/, billing/, economics/, cac/, burn/
├── ledger/         # decision records (markdown + json)
├── evals/          # rag/, drafting/, router/, overfitting/
└── docs/           # links to projects 01–27 used as building blocks
```

## 3. Data Models

```python
class DecisionRecord(BaseModel):
    id: str
    date: date
    change: str                         # "Route 'summaries' to self-hosted Qwen-7B AWQ"
    concepts: list[str]                 # ["Quantization", "LLMs", "Caches"]
    quality_before: float
    quality_after: float
    p95_latency_before_ms: int
    p95_latency_after_ms: int
    cogs_per_1k_before: Decimal
    cogs_per_1k_after: Decimal
    gross_margin_delta_pts: float
    runway_delta_months: float
    decision: Literal["kept", "reverted"]
```

## 4. API Surface

| Method | Endpoint / Tool | Purpose |
|---|---|---|
| POST | /v1/matters/{id}/ask | Cited Q&A. |
| POST | /v1/drafts | Draft in firm style. |
| GET | /v1/economics/summary | Margin, burn, runway, CAC/payback. |
| POST | /v1/ledger | Record decision + measured impact. |
| cli | sib bench --change CHANGE_ID | Measure quality/latency/cost before vs after. |

## 5. Core Algorithms

### Margin-aware routing

For each request class, choose the model minimising cost subject to `quality ≥ target` (from evals). The router is a distilled classifier predicting class + difficulty.

### Runway impact of a tech change

`Δburn = Δcogs_per_request × projected_requests`. Re-run the runway simulation with the new burn and record Δrunway in the ledger.

### CAC payback with AI COGS

`payback = CAC / (ARPA − AI_COGS_per_account − other_COGS)`. Tech optimisations shorten payback directly.

### Per-firm LoRA with overfitting guard

Small firm datasets (hundreds of docs), so use rank ≤ 16, early stop on a firm held-out set and a canary check, and fall back to the base model if the gain is < 2 pts.

## 6. Configuration

| Variable | Purpose |
|---|---|
| ANTHROPIC_API_KEY | Frontier model |
| VLLM_URL | Self-hosted models |
| STRIPE_KEY | Billing |
| DATABASE_URL | Postgres |
| MARGIN_TARGET | Gross margin alert threshold |

## 7. Testing Strategy

- Eval gates per component in CI.
- Tenant isolation suite.
- Economics calculations vs fixtures.
- Ledger benchmarks reproducible.

## 8. Performance Budgets

- p95 answer < 6 s.
- COGS per active firm < 20% of ARPA.

## 9. Local Setup

```bash
make dev
make evals
make economics
sib bench --change latest
```
