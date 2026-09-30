import numpy as np
import pytest

from sib.embeddings import HashingEmbedder, TunedEmbedder, matryoshka, recall_at_k
from sib.vectordb import VectorStore, ann_recall

PASSAGES = [
    "The lessee shall remit monthly rent no later than the fifth day of each month.",
    "Either party may terminate this agreement upon ninety days written notice.",
    "The employee shall not disclose confidential information to third parties.",
    "The supplier indemnifies the purchaser against all product liability claims.",
    "Disputes shall be resolved by binding arbitration in Mumbai.",
    "The lessor is responsible for structural repairs to the premises.",
]
PAIRS = [
    ("when is the tenant payment due", PASSAGES[0]),
    ("how do I end the contract", PASSAGES[1]),
    ("can staff share secrets", PASSAGES[2]),
    ("who pays if the goods injure someone", PASSAGES[3]),
    ("where are conflicts settled", PASSAGES[4]),
    ("who fixes the building", PASSAGES[5]),
]


def test_similar_texts_are_closer():
    e = HashingEmbedder()
    a, b, c = e.embed(["termination notice period", "notice of termination", "product liability indemnity"])
    assert a @ b > a @ c


def test_tuning_lowers_loss_and_improves_recall():
    base = HashingEmbedder()
    tuned = TunedEmbedder(base)
    queries = [p[0] for p in PAIRS]
    before = recall_at_k(base, queries, PASSAGES, list(range(6)), k=1)
    losses = tuned.fit(PAIRS)
    after = recall_at_k(tuned, queries, PASSAGES, list(range(6)), k=1)
    assert losses[-1] < losses[0]
    assert after >= before and after == 1.0


def test_matryoshka_is_normalised():
    v = matryoshka(HashingEmbedder().embed(PASSAGES), 64)
    assert v.shape == (6, 64)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_tenant_isolation_and_filter():
    e = HashingEmbedder()
    store = VectorStore(e.dim)
    store.add("firm_a", "a1", e.embed_one("termination notice"), {"matter": "m1"})
    store.add("firm_a", "a2", e.embed_one("termination notice clause"), {"matter": "m2"})
    store.add("firm_b", "b1", e.embed_one("termination notice"), {"matter": "m1"})
    q = e.embed_one("termination notice")
    assert {h.id for h in store.search("firm_a", q)} == {"a1", "a2"}
    assert [h.id for h in store.search("firm_a", q, where={"matter": "m2"})] == ["a2"]
    assert store.search("firm_c", q) == []
    with pytest.raises(ValueError):
        store.search("", q)


def test_ivf_recall_rises_with_nprobe():
    rng = np.random.default_rng(1)
    store = VectorStore(32)
    vecs = rng.normal(size=(400, 32)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    for i, v in enumerate(vecs):
        store.add("t", f"v{i}", v)
    col = store.collection("t")
    col.build_ivf(16)
    low, high = ann_recall(col, vecs[:20], 5, nprobe=1), ann_recall(col, vecs[:20], 5, nprobe=16)
    assert high == 1.0 and low < high
