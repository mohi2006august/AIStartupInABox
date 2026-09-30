import numpy as np

from sib.caches import CachedEmbedder, ExactCache, PromptCache, SemanticCache
from sib.corpus import build_corpus
from sib.embeddings import HashingEmbedder, TunedEmbedder
from sib.rag import RagIndex, chunk_document, is_supported, reciprocal_rank_fusion


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_exact_cache_normalises_scopes_and_expires():
    clock = Clock()
    cache = ExactCache(ttl_s=10, clock=clock)
    cache.put("firm_a", "m1", "What is the notice period?", "90 days")
    assert cache.get("firm_a", "m1", "what is the   notice period") == "90 days"
    assert cache.get("firm_b", "m1", "What is the notice period?") is None
    assert cache.get("firm_a", "m2", "What is the notice period?") is None
    clock.t = 11
    assert cache.get("firm_a", "m1", "What is the notice period?") is None


def test_exact_cache_lru_and_invalidate():
    cache = ExactCache(max_entries=2)
    for q in ("a", "b", "c"):
        cache.put("t", "m", q, q)
    assert cache.get("t", "m", "a") is None and cache.get("t", "m", "c") == "c"
    assert cache.invalidate("t", "m") == 2


def test_semantic_cache_threshold_and_tenant_scope():
    e = HashingEmbedder()
    cache = SemanticCache(threshold=0.9)
    cache.put("firm_a", "m1", e.embed_one("how many days notice to terminate the lease"), "ninety")
    assert cache.get("firm_a", "m1", e.embed_one("how many days notice to terminate the lease please")) == "ninety"
    assert cache.get("firm_a", "m1", e.embed_one("what security deposit is required")) is None
    assert cache.get("firm_b", "m1", e.embed_one("how many days notice to terminate the lease")) is None


def test_prompt_cache_write_then_read_then_expire():
    clock = Clock()
    cache = PromptCache(ttl_s=300, min_tokens=50, clock=clock)
    prefix = "You are a careful legal research assistant. " * 30
    read, write = cache.apply("m", prefix)
    assert read == 0 and write > 0
    assert cache.apply("m", prefix) == (write, 0)
    clock.t = 1000
    assert cache.apply("m", prefix) == (0, write)
    assert cache.apply("m", "short") == (0, 0)


def test_embedding_cache_hits_on_repeat():
    cached = CachedEmbedder(HashingEmbedder())
    a = cached.embed(["alpha beta", "gamma"])
    b = cached.embed(["alpha beta", "delta"])
    assert np.allclose(a[0], b[0])
    assert cached.stats.hits == 1 and cached.stats.misses == 3


def test_chunks_are_sentence_aligned():
    doc = build_corpus().documents[0]
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    assert " ".join(c.text for c in chunks) == doc.text
    assert all(c.text.endswith(".") for c in chunks)


def test_rrf_prefers_items_ranked_high_in_both():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert fused["a"] == fused["b"] > fused["c"] > 0


def _recall(index, corpus, difficulty, k=3):
    qs = [q for q in corpus.questions if q.difficulty == difficulty]
    hit = sum(any(q.expected in r.chunk.text for r in index.retrieve(q.tenant_id, q.matter_id, q.question, k)) for q in qs)
    return hit / len(qs)


def test_retrieval_is_matter_scoped_and_tuning_helps_hard_questions():
    corpus = build_corpus()
    base = RagIndex(HashingEmbedder())
    base.ingest(corpus.documents)
    q = corpus.questions[0]
    hits = base.retrieve(q.tenant_id, q.matter_id, q.question)
    assert hits and all(r.chunk.tenant_id == q.tenant_id and r.chunk.matter_id == q.matter_id for r in hits)
    assert base.retrieve("firm_unknown", q.matter_id, q.question) == []
    assert _recall(base, corpus, "easy") >= 0.95

    tuned_embedder = TunedEmbedder(HashingEmbedder())
    tuned_embedder.fit(corpus.training_pairs)
    tuned = RagIndex(tuned_embedder)
    tuned.ingest(corpus.documents)
    assert _recall(tuned, corpus, "hard") > _recall(base, corpus, "hard")


def test_verification_rejects_unsupported_claims():
    passage = "Either party may terminate this Lease upon ninety (90) days written notice to the other party."
    assert is_supported(passage, passage)
    assert not is_supported("The agreement provides for a standard period of thirty (30) days in such cases.", passage)
