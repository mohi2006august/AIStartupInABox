# AI Startup-in-a-Box: Design Document

> The ultimate capstone that uses all 15 concepts: build, launch and run a real AI SaaS product end to end, from model stack to unit economics, and make it profitable.
>
> **Concepts:** `RAG`, `MCP`, `Vector Database`, `Tokenization`, `Context Window`, `Fine Tuning`, `Overfitting`, `Embeddings`, `Knowledge Distillation`, `LoRA`, `Burn Rate`, `CAC`, `LLMs`, `Caches`, `Quantization`

## 1. Design Principles

- Every technical decision has a price tag and a quality score.
- Ship the simplest thing that makes a customer pay, then optimise margins with evidence.

## 2. Key User Flows

### Founder weekly review

1. Open economics console.
2. Check margin, burn, runway, CAC payback.
3. Review ledger entries from this week.
4. Pick next optimisation from ranked opportunities.

### Lawyer daily use

1. Open matter.
2. Ask questions with citations.
3. Generate draft in firm style.
4. Send via email integration (approval).

## 3. Screens & Surfaces

| Screen | What it shows / does |
|---|---|
| Matter workspace | Q&A with citations, drafts, integrations. |
| Economics console | MRR, gross margin, AI COGS, burn, runway, CAC/payback. |
| Decision ledger | Timeline of changes with quality/cost/runway deltas. |
| Model ops | Router mix, cache hits, self-hosted vs API share. |

## 4. Wireframe (Primary Screen)

```text
Decision ledger                              quality   p95     COGS/1k   GM      runway
2026-10-02 Semantic + prompt caching          =        −1.9s   −$4.10    +9 pts  +1.8 mo  kept
2026-10-19 Distilled router (small model 64%) −0.6     −0.7s   −$3.30    +7 pts  +1.2 mo  kept
2026-11-07 Self-host drafting AWQ + LoRA      −1.1     +0.2s   −$5.80    +12 pts +2.4 mo  kept
2026-11-21 Context window 32k → 12k budget    −2.9 ✖   −0.4s   −$1.20    +2 pts  +0.3 mo  reverted
Gross margin 58% → 81% · CAC payback 13.5 → 8.2 mo · default alive ✔
```

## 5. Visual Language

| Token | Value |
|---|---|
| Primary accent | `#B45309` (gold, capstone) |
| Neutral surface | `#0F1117` dark / `#FAFAF9` light |
| Success / Warning / Danger | `#22C55E` / `#F59E0B` / `#EF4444` |
| Type | Inter for UI, JetBrains Mono for code, tokens and numbers |
| Spacing scale | 4 / 8 / 12 / 16 / 24 / 32 px |
| Radius | 8px cards, 6px inputs, full for pills |

## 6. States

| State | Treatment |
|---|---|
| Quality gate failed | Change blocked + ledger 'reverted'. |
| Default alive | Celebration state on economics console. |

## 7. Accessibility

- WCAG 2.2 AA contrast in both themes; never encode meaning with colour alone (pair with icon or label).
- Full keyboard navigation with visible focus rings; `/` focuses search, `?` opens shortcuts.
- Charts ship with a data-table fallback and `aria-describedby` summaries.
- Respect `prefers-reduced-motion`; streaming text is announced politely via `aria-live`.
