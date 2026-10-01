# AI Startup-in-a-Box: Brain

> Second brain for this project: concepts, mental models, research, ideas and decisions. Keep it messy and alive.

## 1. Why This Project

This is the final exam. Each earlier project is a building block, and here they compound. You learn that in AI startups, architecture decisions are financial decisions.

## 2. Concept Primers

### RAG

Retrieval-Augmented Generation: fetch relevant passages from an external corpus at query time and put them in the prompt so the LLM answers from evidence instead of memory.

- **Chunking**: Splitting documents into retrievable units (fixed tokens, sentence windows, semantic or structure-aware splits).
- **Hybrid retrieval**: Combining sparse (BM25) and dense (embedding) search, usually merged with Reciprocal Rank Fusion.
- **Re-ranking**: A cross-encoder rescoring the top-k candidates for precision before they reach the prompt.
- **Grounding / faithfulness**: Whether every claim in the answer is supported by a retrieved passage.

> **Mental model:** Retrieval sets the ceiling and generation sets the floor. If the right chunk never gets retrieved, no prompt can save the answer, so measure recall@k before tuning prompts.

### MCP

Model Context Protocol: an open JSON-RPC protocol that lets AI clients (Claude, IDEs, agents) discover and call tools, read resources and use prompts exposed by servers in a standard way.

- **Server**: Exposes tools, resources and prompts over stdio or Streamable HTTP.
- **Tool**: A typed function the model can call (a JSON Schema input and a structured result).
- **Resource**: Read-only context addressed by URI (files, rows, docs) that the client can attach.
- **Transport / auth**: stdio for local servers, Streamable HTTP + OAuth 2.1 for remote servers.

> **Mental model:** Think of MCP as USB-C for AI. Write the integration once and every compliant client can use it. Tool descriptions are prompts, so write them for the model.

### Vector Database

A datastore optimised for storing high-dimensional vectors and answering approximate nearest-neighbour (ANN) queries, usually with metadata filtering.

- **ANN index**: HNSW (graph), IVF (clustered lists), PQ (compressed codes), DiskANN (SSD-resident graph).
- **Recall@k**: Fraction of true k nearest neighbours returned; the quality side of the speed trade-off.
- **Filtered search**: Applying metadata predicates pre-, post- or in-graph during traversal.
- **Distance metric**: Cosine, dot product or L2; must match how the embeddings were trained.

> **Mental model:** Every ANN index trades recall, latency, memory and build time against each other. Pick two to optimise and measure the other two.

### Tokenization

Converting raw text into the integer token IDs a model consumes. Subword algorithms (BPE, WordPiece, Unigram) balance vocabulary size against sequence length.

- **BPE**: Byte-Pair Encoding: iteratively merges the most frequent adjacent pair; byte-level BPE never hits unknown tokens.
- **Unigram LM**: Starts with a large vocabulary and prunes tokens that least reduce corpus likelihood (SentencePiece).
- **Fertility**: Average tokens per word; high fertility means higher cost and less effective context for that language.
- **Special tokens**: BOS/EOS/PAD and chat-template markers that structure model input.

> **Mental model:** Tokens are the unit of cost, latency and context. The same sentence can cost 3x more in Hindi than in English, and that 'token tax' is a product and fairness issue.

### Context Window

The maximum number of tokens (prompt plus output) a model can attend to in one call. Everything the model 'knows' about the task at inference time must fit inside it.

- **Lost in the middle**: Models attend less reliably to information buried mid-context than at the start or end.
- **Compaction**: Summarising or pruning older turns so long sessions fit the budget.
- **KV cache**: Stored attention keys/values; grows linearly with context length and dominates memory.
- **Needle-in-a-haystack**: Eval that plants a fact at varying depths to test long-context recall.

> **Mental model:** Context is a budget, not a bucket. More tokens cost more money and latency and can dilute attention, so curate what goes in.

### Fine Tuning

Continuing training of a pre-trained model on task- or domain-specific data to change its behaviour, style or knowledge.

- **SFT**: Supervised fine-tuning on prompt → ideal response pairs.
- **DPO / preference tuning**: Optimising directly on chosen-vs-rejected pairs without a separate reward model.
- **Catastrophic forgetting**: Losing general capabilities while specialising.
- **PEFT**: Parameter-efficient methods (LoRA, adapters, prefix tuning) that train a small fraction of weights.

> **Mental model:** Fine-tune for form (format, tone, task behaviour). Retrieve for facts. Data quality beats data quantity almost every time.

### Overfitting

When a model fits noise or specifics of the training set so closely that it performs worse on unseen data; train loss keeps dropping while validation loss rises.

- **Generalisation gap**: Difference between training and validation/test performance.
- **Regularisation**: Weight decay, dropout, data augmentation, early stopping, smaller capacity.
- **Memorisation**: Verbatim reproduction of training examples; measured with canaries or extraction probes.
- **Data leakage**: Test information leaking into training, which hides overfitting behind inflated scores.

> **Mental model:** A low training loss is a claim. The held-out set is the evidence. Always keep a test set you never tune against.

### Embeddings

Dense vector representations where geometric closeness reflects semantic similarity; the backbone of search, clustering, recommendation and RAG.

- **Contrastive training**: Pulling positive pairs together and pushing negatives apart (e.g. InfoNCE).
- **Matryoshka embeddings**: Vectors whose prefixes are themselves usable lower-dimensional embeddings.
- **Anisotropy**: Embeddings crowding into a narrow cone, which hurts cosine similarity spread.
- **Embedding drift**: Distribution shift in vectors over time or after a model upgrade.

> **Mental model:** An embedding model is an opinion about what 'similar' means. Benchmarks like MTEB are a starting point, so evaluate on your own queries.

### Knowledge Distillation

Training a small 'student' model to mimic a large 'teacher' model's outputs (soft labels, logits, rationales), keeping most of the quality at a fraction of the cost.

- **Soft targets**: Teacher probability distributions softened with temperature T carry 'dark knowledge'.
- **KD loss**: KL divergence between teacher and student distributions, mixed with hard-label cross-entropy.
- **Sequence-level KD**: Student trained on teacher-generated outputs (synthetic data distillation).
- **Capacity gap**: Very large teacher/student gaps can hurt; teacher assistants bridge them.

> **Mental model:** You are compressing a function, not a dataset. The teacher's uncertainty is signal, so preserve it.

### LoRA

Low-Rank Adaptation: freeze the base weights W and learn a low-rank update ΔW = B·A (rank r ≪ d), training well under 1% of parameters with small, swappable adapters.

- **Rank r / alpha**: Capacity of the update and its scaling (alpha/r).
- **Target modules**: Which projections get adapters (q,k,v,o, MLP).
- **QLoRA**: LoRA on top of a 4-bit quantised base model; fine-tune big models on one GPU.
- **Multi-LoRA serving**: One base model in memory serving many adapters per request (S-LoRA, Punica).

> **Mental model:** The base model is the OS and adapters are apps. They are cheap to train, store and switch, so give each tenant, task or style its own adapter.

### Burn Rate

How fast a company spends cash. Gross burn is total monthly spend. Net burn is spend minus revenue. Runway = cash ÷ net burn.

- **Gross vs net burn**: Total outflow vs outflow minus inflow.
- **Runway**: Months until cash hits zero at the current (or projected) net burn.
- **Burn multiple**: Net burn ÷ net new ARR; efficiency of growth spending (< 1.5 is good).
- **Default alive**: Whether current trajectory reaches profitability before cash runs out.

> **Mental model:** Burn is a lagging indicator of decisions made months ago. Model the decisions (hires, contracts, pricing), not only the totals.

### CAC

Customer Acquisition Cost: total sales and marketing spend ÷ new customers acquired over a period. It is judged against LTV and payback period.

- **Blended vs paid CAC**: All customers vs only those attributed to paid channels.
- **LTV:CAC**: Lifetime value relative to acquisition cost; ~3:1 is a common healthy benchmark.
- **CAC payback**: Months of gross-margin-adjusted revenue needed to recover CAC.
- **Attribution**: Assigning credit to touchpoints (first/last touch, linear, data-driven, MMM).

> **Mental model:** Blended CAC hides channel truth. Always slice by channel, cohort and segment, and include fully loaded costs (salaries, tools).

### LLMs

Large Language Models: transformer networks trained on large text corpora to predict the next token, then aligned (SFT and RLHF/RLAIF) to follow instructions and use tools.

- **Prompting**: System prompts, few-shot examples, structured outputs, tool use.
- **Sampling**: Temperature, top-p and max tokens shape determinism and length.
- **Evals**: Task-specific test sets plus LLM-as-judge with calibrated rubrics.
- **Routing**: Sending each request to the cheapest model that meets its quality bar.

> **Mental model:** Treat the LLM as a probabilistic component. Wrap it with evals, guardrails, retries and observability, the same way you would a flaky network dependency.

### Caches

Storing results of expensive operations for reuse. In LLM systems this means exact response caches, semantic caches, embedding caches, and provider-side prompt/KV caching.

- **Hit ratio**: Fraction of requests served from cache; the core ROI metric.
- **Semantic cache**: Returns a cached answer if a new query is similar enough (embedding distance < threshold).
- **Prompt caching**: Provider reuses computed KV for a repeated prompt prefix, cutting cost and latency.
- **Invalidation / TTL**: When cached answers become stale; famously one of the two hard problems.

> **Mental model:** A cache is a bet that the future looks like the past. A wrong semantic cache hit costs more than a miss, so tune thresholds for precision.

### Quantization

Representing weights (and sometimes activations/KV cache) with fewer bits (int8, int4, fp8) to cut memory and increase speed, with some accuracy loss.

- **PTQ vs QAT**: Post-training quantisation vs quantisation-aware training.
- **GPTQ / AWQ**: Weight-only int4 methods using second-order info or activation-aware scaling.
- **GGUF**: llama.cpp file format with k-quants (Q4_K_M, Q5_K_M, …) for CPU/edge inference.
- **Outliers**: A few large-magnitude activation channels that break naive quantisation.

> **Mental model:** Memory bandwidth, not FLOPs, bounds LLM decoding. Halving bytes per weight roughly doubles tokens/sec until you hit compute limits.

## 3. What I Want to Learn

- [ ] Integrating all 15 concepts in one product.
- [ ] Translating technical metrics into business metrics.
- [ ] Prioritising optimisations by ROI.
- [ ] Operating a real product with customers.

## 4. Reading & Research List

- All reading lists from projects 01–27
- Paul Graham, *Default Alive or Default Dead?*
- a16z, *The New Business of AI*
- Anthropic, *Building effective agents*
- Rob Fitzpatrick, *The Mom Test* (customer discovery)

## 5. Experiments to Run

| # | Hypothesis | How to test | Result so far (simulated, see docs/assumptions.md) |
|---|---|---|---|
| 1 | Caching + routing alone reach 75% margin before any self-hosting | Ledger sequence. | Not quite: 59% → 73.4% at 10 customers. Self-hosting closes the gap once volume fills a GPU node (+5.8 pts at 50 customers). |
| 2 | Per-firm LoRA improves draft acceptance ≥ 15% | A/B with design partners. | Style match on held-out firm examples: +11 to +42 pts over the base model, and above 20-shot prompting. Needs real design partners. |
| 3 | Referral loop has lowest CAC payback | Channel CAC tracking. | In the scenario: referral 1.7 mo vs LinkedIn 13.8 mo (assumed spend and conversion, not observed). |

## 6. Ideas Backlog

- Second vertical using the same platform.
- Self-serve onboarding with trial COGS caps.
- Open-source the decision-ledger tooling.

## 7. Decision Log

| Date | Decision | Reason |
|---|---|---|
| 2026-09-30 | APIs first, self-host later | Speed to customers beats margin in month 1; the ledger tells us when to switch. |
| 2026-09-30 | Decision ledger mandatory | Forces measurement of every optimisation. |
| 2026-09-30 | Offline simulator behind the provider interface | Reproducible evals and ledger with no keys; `ClaudeProvider` swaps in with `LIVE_MODELS=true`. |
| 2026-09-30 | Single-file web console served by FastAPI instead of Next.js | One deployable, no build step; revisit when the UI outgrows it. |
| 2026-09-30 | Stripe deferred behind a `BillingGateway` protocol | New paid service; invoices are computed and stored locally until design partners sign. |
| 2026-10-01 | Keep tuned embeddings | +30.9 quality pts on the ask eval for −0.4 margin pts. |
| 2026-10-01 | Keep the caching stack | +13.6 margin pts, +10.3 months runway, no quality change. Semantic threshold 0.82 is only safe with the tuned embedder (12 false hits with the base embedder). |
| 2026-10-01 | Keep the distilled router (Haiku for easy asks) | +1.5 margin pts; flips the scenario to default alive. Difficulty uses retrieval-coverage features only, because word features did not transfer to new phrasings. |
| 2026-10-01 | Revert self-hosting at 10 customers | −3.5 margin pts: the $864/month node sits mostly idle. Re-bench at 15+ customers. |
| 2026-10-01 | Revert the 6,800-token context budget | Quality gate: −2.6 pts. |
| 2026-10-01 | Add the Stripe gateway (approved) | Pricing stays local and auditable; Stripe only receives finished invoices. Drafts by default, live keys refused unless allowed, idempotency keys plus local progress so re-runs never double-bill. |

## 8. Scratchpad

- _Dump raw thoughts, links and half-formed ideas here._
- The verifier only catches unsupported claims. A weak model that cites the wrong clause verbatim passes verification; that is what the router's difficulty threshold is calibrated against.
- Embedding query-generation settings were picked while watching eval recall. Re-check on real firm documents before trusting the number.
- A canary that is memorised is the clearest overfitting signal at small data sizes: in these runs most adapters trained on 60-150 examples memorised canaries; none trained on 300 did.
