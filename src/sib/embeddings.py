"""Embeddings (FR-01): a dependency-free hashing embedder plus a contrastively tuned adapter.

The base embedder is feature hashing over words and character trigrams. The adapter is a linear
map trained with InfoNCE on (query, relevant passage) pairs: "tuned embeddings" at laptop scale.
"""
from __future__ import annotations

import re
import zlib

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the this to was were will with "
    "what which who when where how does do did shall may can any all".split()
)


def tokenize_words(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_words(text: str) -> list[str]:
    return [w for w in tokenize_words(text) if w not in STOPWORDS]


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    h = zlib.crc32(feature.encode("utf-8"))
    return h % dim, 1.0 if (h >> 31) & 1 else -1.0


def normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class HashingEmbedder:
    name = "hash-v1"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in content_words(text):
            idx, sign = _bucket("w:" + word, self.dim)
            vec[idx] += sign
            padded = f"#{word}#"
            for i in range(len(padded) - 2):
                idx, sign = _bucket("c:" + padded[i : i + 3], self.dim)
                vec[idx] += 0.3 * sign
        return normalize(vec)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed_one(t) for t in texts])


class TunedEmbedder:
    """Base embedder followed by a learned linear adapter W (initialised to identity)."""

    def __init__(self, base: HashingEmbedder) -> None:
        self.base = base
        self.dim = base.dim
        self.W = np.eye(base.dim, dtype=np.float32)
        self.name = base.name + "+adapter"

    def embed(self, texts: list[str]) -> np.ndarray:
        return normalize(self.base.embed(texts) @ self.W.T)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def fit(
        self,
        pairs: list[tuple[str, ...]],
        epochs: int = 60,
        lr: float = 0.5,
        temperature: float = 0.1,
        identity_reg: float = 0.01,
    ) -> list[float]:
        """InfoNCE with in-batch negatives. Returns the loss curve.

        Pairs are (query, passage) or (query, passage, group). Rows sharing a group are near
        duplicates, so they are masked out as negatives instead of being pushed apart.
        """
        q = self.base.embed([p[0] for p in pairs])
        d = self.base.embed([p[1] for p in pairs])
        n = len(pairs)
        groups = np.array([p[2] if len(p) > 2 else str(i) for i, p in enumerate(pairs)])
        false_negative = (groups[:, None] == groups[None, :]) & ~np.eye(n, dtype=bool)
        eye = np.eye(self.dim, dtype=np.float32)
        losses: list[float] = []
        for _ in range(epochs):
            a, b = q @ self.W.T, d @ self.W.T
            logits = a @ b.T / temperature
            logits[false_negative] = -1e9
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            losses.append(float(-np.log(np.diag(probs) + 1e-12).mean()))
            g = (probs - np.eye(n, dtype=np.float32)) / (n * temperature)
            grad = (g @ b).T @ q + (g.T @ a).T @ d + identity_reg * (self.W - eye)
            self.W -= lr * grad.astype(np.float32)
        return losses


def matryoshka(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Truncate to a prefix and re-normalise: cheaper storage for some recall loss."""
    return normalize(vectors[..., :dim])


def recall_at_k(embedder, queries: list[str], passages: list[str], relevant: list[int], k: int = 3) -> float:
    qv, pv = embedder.embed(queries), embedder.embed(passages)
    top = np.argsort(-(qv @ pv.T), axis=1)[:, :k]
    return float(np.mean([relevant[i] in top[i] for i in range(len(queries))]))
