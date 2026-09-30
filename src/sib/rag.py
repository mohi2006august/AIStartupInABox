"""Cited RAG over firm documents (FR-01): chunking, hybrid retrieval with RRF, citation verification."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sib.corpus import Document, stems
from sib.embeddings import STOPWORDS
from sib.llm import split_sentences
from sib.tokens import count_tokens
from sib.vectordb import VectorStore


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    tenant_id: str
    matter_id: str
    title: str
    text: str


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


def chunk_document(doc: Document, max_tokens: int = 45) -> list[Chunk]:
    """Sentence-aligned chunks: a clause is never split mid-sentence."""
    chunks: list[Chunk] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, doc.tenant_id, doc.matter_id, doc.title, " ".join(buf)))
            buf.clear()

    for sentence in split_sentences(doc.text):
        if buf and count_tokens(" ".join(buf + [sentence])) > max_tokens:
            flush()
        buf.append(sentence)
    flush()
    return chunks


def _terms(text: str) -> list[str]:
    return [w for w in stems(text) if w not in STOPWORDS]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.df: Counter[str] = Counter()

    def add(self, text: str) -> None:
        terms = _terms(text)
        self.docs.append(Counter(terms))
        self.lengths.append(len(terms))
        self.df.update(set(terms))

    def scores(self, query: str, candidates: list[int]) -> list[float]:
        n = len(self.docs)
        avg = sum(self.lengths) / n if n else 0.0
        out = []
        for i in candidates:
            tf, score = self.docs[i], 0.0
            for term in set(_terms(query)):
                if term in tf:
                    idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += idf * tf[term] * (self.k1 + 1) / (tf[term] + self.k1 * (1 - self.b + self.b * self.lengths[i] / avg))
            out.append(score)
        return out


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


class RagIndex:
    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder
        self.vectors = VectorStore(embedder.dim)
        self._chunks: dict[str, dict[str, Chunk]] = {}  # tenant -> chunk id -> chunk
        self._bm25: dict[str, BM25] = {}
        self._order: dict[str, list[str]] = {}

    def ingest(self, documents: list[Document]) -> int:
        chunks = [c for d in documents for c in chunk_document(d)]
        vectors = self.embedder.embed([c.text for c in chunks])
        for chunk, vec in zip(chunks, vectors):
            self.vectors.add(chunk.tenant_id, chunk.id, vec, {"matter": chunk.matter_id})
            self._chunks.setdefault(chunk.tenant_id, {})[chunk.id] = chunk
            self._bm25.setdefault(chunk.tenant_id, BM25()).add(chunk.text)
            self._order.setdefault(chunk.tenant_id, []).append(chunk.id)
        return len(chunks)

    def retrieve(self, tenant_id: str, matter_id: str, query: str, k: int = 3, mode: str = "hybrid") -> list[Retrieved]:
        """mode: 'dense' | 'sparse' | 'hybrid' (RRF of both). Always scoped to tenant and matter."""
        chunks = self._chunks.get(tenant_id, {})
        order = self._order.get(tenant_id, [])
        candidates = [i for i, cid in enumerate(order) if chunks[cid].matter_id == matter_id]
        if not candidates:
            return []
        rankings: list[list[str]] = []
        if mode in ("dense", "hybrid"):
            hits = self.vectors.search(tenant_id, self.embedder.embed_one(query), k=len(candidates), where={"matter": matter_id})
            rankings.append([h.id for h in hits])
        if mode in ("sparse", "hybrid"):
            scores = self._bm25[tenant_id].scores(query, candidates)
            ranked = sorted(zip(candidates, scores), key=lambda cs: -cs[1])
            rankings.append([order[i] for i, s in ranked if s > 0])
        fused = reciprocal_rank_fusion(rankings)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Retrieved(chunks[cid], score) for cid, score in top]


def support_score(answer: str, passage: str) -> float:
    """Share of the answer's content words (numbers included) that appear in the cited passage."""
    claim = set(_terms(answer))
    return len(claim & set(_terms(passage))) / len(claim) if claim else 0.0


def is_supported(answer: str, passage: str, threshold: float = 0.8) -> bool:
    return support_score(answer, passage) >= threshold
