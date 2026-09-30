# AI Startup-in-a-Box: Product Requirements Document

> The ultimate capstone that uses all 15 concepts: build, launch and run a real AI SaaS product end to end, from model stack to unit economics, and make it profitable.
>
> **Concepts:** `RAG`, `MCP`, `Vector Database`, `Tokenization`, `Context Window`, `Fine Tuning`, `Overfitting`, `Embeddings`, `Knowledge Distillation`, `LoRA`, `Burn Rate`, `CAC`, `LLMs`, `Caches`, `Quantization`

## 1. Overview

Build and operate a real vertical AI SaaS, for example an 'AI research assistant for small law firms'. The product: RAG over firm documents (vector DB, tuned embeddings), MCP integrations (email, drive, calendar), context-window management for long matters, a fine-tuned + LoRA-per-firm drafting model, distilled classifiers for cheap routing, quantised self-hosted models for margin, multi-level caching and tokenizer-aware prompt design. The business layer includes metered billing, a CAC/payback dashboard and a burn/runway model. Every technical decision is logged with its measured impact on COGS, gross margin and runway.

## 2. Problem Statement

Learning concepts one by one doesn't teach how they interact in a real business. A technical choice (bigger model, longer context, no cache) changes COGS, which changes burn, runway and how much you can spend on CAC. Few people ever build the whole loop.

## 3. Target Users & Personas

| Persona | Needs |
|---|---|
| Solo founder / small team | A complete blueprint for an AI startup. |
| Advanced learner | See every concept working together. |
| Investor/mentor | Evaluate a business where tech and economics are connected. |

## 4. Goals

- Launch to 10 paying customers.
- Gross margin ≥ 75% through technical optimisation.
- CAC payback ≤ 9 months.
- A living 'decision ledger' linking each technical change to its $ impact.

## 5. Non-Goals

- Raising a large round (default alive is the aim).
- Perfect feature parity with incumbents.

## 6. User Stories

- As founder, I switch drafting from a frontier API to a quantised, LoRA-tuned 8B model and see gross margin go from 58% to 79% at −2% quality.
- As founder, I see enabling prompt caching + semantic cache added 1.8 months of runway.
- As founder, I see LinkedIn CAC payback 14 months vs referral 3 months, and shift budget.
- As a user (lawyer), I ask across 800 matter documents and get cited answers; drafts come in my firm's style.

## 7. Functional Requirements

| ID | Priority | Feature | Description |
|---|---|---|---|
| FR-01 | P0 | Product: cited RAG over firm docs | Hybrid retrieval, tuned embeddings, vector DB, citations (CiteSure-style). |
| FR-02 | P0 | Integrations via MCP | Email, Drive/SharePoint, calendar; approval-gated actions. |
| FR-03 | P0 | Metering + billing | Usage-based + seat pricing via Stripe. |
| FR-04 | P0 | Economics console | COGS per customer, gross margin, burn, runway, CAC and payback by channel. |
| FR-05 | P1 | Long-matter context management | Budgeted packing + matter memory (ContextPacker/RecallMCP-style). |
| FR-06 | P1 | Drafting model | SFT on public legal drafting + per-firm LoRA adapters; overfitting guard on small firm data. |
| FR-07 | P1 | Distilled router/classifiers | Intent + difficulty classifier distilled from frontier model routes requests to cheap vs expensive models. |
| FR-08 | P1 | Quantised self-hosting | AWQ/fp8 models on vLLM for high-volume paths. |
| FR-09 | P1 | Caching stack | Exact, semantic, provider prompt caching, embedding cache. |
| FR-10 | P1 | Tokenizer-aware prompts | Token budgets per feature; compact formats; multilingual fertility checks. |
| FR-11 | P2 | Decision ledger | Every tech change records before/after quality, latency, COGS, runway impact. |

## 8. Non-Functional Requirements

- Client data isolation and confidentiality (law-firm grade).
- Quality gates: no release if eval drops > 2 pts.
- p95 answer < 6 s.

## 9. Success Metrics

| Metric | Target |
|---|---|
| Paying customers | 10 → 50 |
| Gross margin | ≥ 75% |
| CAC payback | ≤ 9 months |
| Burn multiple | ≤ 1.5 |
| Answer faithfulness | ≥ 0.95 |

## 10. Release Plan

| Phase | Scope |
|---|---|
| M1 (weeks 1–4) | MVP: RAG + citations + Drive integration via MCP + Stripe; 3 design partners. |
| M2 (weeks 5–8) | Metering, economics console, caching stack, token budgets. |
| M3 (weeks 9–14) | Drafting model: SFT + per-firm LoRA with overfitting guard; distilled router. |
| M4 (weeks 15–20) | Quantised self-hosting for high-volume paths; margin optimisation. |
| M5 (weeks 21–26) | Growth: CAC tracking by channel, referral loop, decision ledger published. |

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Doing too much at once | Milestones gated by customer value and margin metrics; the MVP uses APIs only. |
| Self-hosting ops burden | Only move paths with volume and a proven margin case (ledger). |
| Legal-domain errors | Citations + verification + lawyer-in-the-loop; no autonomous filing. |

## 12. Open Questions

- Vertical choice (legal vs accounting vs real estate)?
- Seat vs usage pricing mix?
