# CLAUDE.md: AI Startup-in-a-Box

Guidance for Claude Code (and other AI assistants) working in this project.

## Project Summary

The ultimate capstone that uses all 15 concepts: build, launch and run a real AI SaaS product end to end, from model stack to unit economics, and make it profitable. Build and operate a real vertical AI SaaS, for example an 'AI research assistant for small law firms'. The product: RAG over firm documents (vector DB, tuned embeddings), MCP integrations (email, drive, calendar), context-window management for long matters, a fine-tuned + LoRA-per-firm drafting model, distilled classifiers for cheap routing, quantised self-hosted models for margin, multi-level caching and tokenizer-aware prompt design. The business layer includes metered billing, a CAC/payback dashboard and a burn/runway model. Every technical decision is logged with its measured impact on COGS, gross margin and runway.

## Read First

- `PRD.md`: what we are building and why; requirement IDs (FR-xx) are referenced in commits.
- `SYSTEM_ARCHITECTURE.md`: components, data flow and failure modes.
- `TECHNICAL.md`: stack, data models, APIs and algorithms.
- `DESIGN.md`: UX flows, screens and visual tokens.
- `BRAIN.md`: concept primers, experiments and the decision log.

## Commands

| Command | What it does |
|---|---|
| `make dev` | Run full stack locally |
| `make evals` | All quality gates |
| `make economics` | Recompute economics |
| `sib bench --change latest` | Measure a change |

## Code Conventions

- Every PR touching models/prompts/caches must include a `sib bench` result and a ledger entry.
- Reuse building blocks from projects 01–27 rather than rewriting.
- Keep functions small and typed; no silent `except`. Log with structured fields.
- Every new feature references its FR-ID in the PR description and ships with tests.
- Secrets only come from environment variables. Never hard-code keys or commit `.env`.

## Gotchas

- Don't optimise margin before you have paying customers.
- Include AI COGS of trials and free tiers in CAC.

## When Unsure

- Prefer the simplest design that satisfies the PRD; record bigger trade-offs in `BRAIN.md` → Decision Log.
- Run the eval / benchmark suite before claiming a quality or performance improvement.
- Ask before adding a new paid dependency, external service or data store.
