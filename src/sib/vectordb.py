"""Vector database (FR-01): per-tenant collections, metadata filters, exact and IVF ANN search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Hit:
    id: str
    score: float
    metadata: dict[str, Any]


class Collection:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.ids: list[str] = []
        self.metas: list[dict[str, Any]] = []
        self._rows: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None
        self._centroids: np.ndarray | None = None
        self._lists: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.stack(self._rows) if self._rows else np.zeros((0, self.dim), dtype=np.float32)
        return self._matrix

    def add(self, id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        if vector.shape != (self.dim,):
            raise ValueError(f"expected vector of dim {self.dim}, got {vector.shape}")
        self.ids.append(id)
        self.metas.append(metadata or {})
        self._rows.append(vector.astype(np.float32))
        self._matrix = None
        self._centroids = None  # index is stale

    def build_ivf(self, n_lists: int, iters: int = 10, seed: int = 0) -> None:
        """k-means clustering; at query time only `nprobe` nearest lists are scanned."""
        m = self.matrix
        n_lists = max(1, min(n_lists, len(m)))
        rng = np.random.default_rng(seed)
        centroids = m[rng.choice(len(m), n_lists, replace=False)].copy()
        assign = np.zeros(len(m), dtype=np.int64)
        for _ in range(iters):
            assign = np.argmax(m @ centroids.T, axis=1)
            for c in range(n_lists):
                members = m[assign == c]
                if len(members):
                    v = members.mean(axis=0)
                    centroids[c] = v / (np.linalg.norm(v) or 1.0)
        self._centroids = centroids
        self._lists = [np.flatnonzero(assign == c) for c in range(n_lists)]

    def search(
        self, query: np.ndarray, k: int = 5, where: dict[str, Any] | None = None, nprobe: int | None = None
    ) -> list[Hit]:
        if not len(self):
            return []
        if nprobe is not None and self._centroids is not None:
            probe = np.argsort(-(self._centroids @ query))[:nprobe]
            candidates = np.concatenate([self._lists[c] for c in probe])
        else:
            candidates = np.arange(len(self))
        if where:  # pre-filter: predicates applied before scoring
            candidates = np.array(
                [i for i in candidates if all(self.metas[i].get(f) == v for f, v in where.items())], dtype=np.int64
            )
        if not len(candidates):
            return []
        scores = self.matrix[candidates] @ query
        order = np.argsort(-scores)[:k]
        return [Hit(self.ids[candidates[i]], float(scores[i]), self.metas[candidates[i]]) for i in order]


class VectorStore:
    """One collection per tenant: a query can never touch vectors of another firm."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._tenants: dict[str, Collection] = {}

    def collection(self, tenant_id: str) -> Collection:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        return self._tenants.setdefault(tenant_id, Collection(self.dim))

    def add(self, tenant_id: str, id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        self.collection(tenant_id).add(id, vector, metadata)

    def search(self, tenant_id: str, query: np.ndarray, k: int = 5, **kwargs: Any) -> list[Hit]:
        return self.collection(tenant_id).search(query, k, **kwargs)


def ann_recall(collection: Collection, queries: np.ndarray, k: int, nprobe: int) -> float:
    """Fraction of the true top-k that the IVF index returns."""
    found = 0
    for q in queries:
        truth = {h.id for h in collection.search(q, k)}
        found += len(truth & {h.id for h in collection.search(q, k, nprobe=nprobe)})
    return found / (len(queries) * k)
