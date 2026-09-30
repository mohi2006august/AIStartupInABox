"""Caching stack (FR-09): exact, semantic, provider prompt-prefix and embedding caches.

Every response cache is keyed by tenant and scope (the matter), so one firm can never be served
an answer computed from another firm's documents.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from sib.tokens import count_tokens

Clock = Callable[[], float]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def normalize_query(text: str) -> str:
    return " ".join(text.lower().split()).rstrip("?!. ")


class ExactCache:
    """LRU + TTL cache on the normalised request."""

    def __init__(self, max_entries: int = 10_000, ttl_s: float = 3600, clock: Clock = time.monotonic) -> None:
        self.max_entries, self.ttl_s, self.clock = max_entries, ttl_s, clock
        self._data: OrderedDict[str, tuple[float, str, Any]] = OrderedDict()
        self.stats = CacheStats()

    def get(self, tenant_id: str, scope: str, query: str) -> Any | None:
        key = _digest(tenant_id, scope, normalize_query(query))
        entry = self._data.get(key)
        if entry is None or entry[0] < self.clock():
            self._data.pop(key, None)
            self.stats.misses += 1
            return None
        self._data.move_to_end(key)
        self.stats.hits += 1
        return entry[2]

    def put(self, tenant_id: str, scope: str, query: str, value: Any) -> None:
        key = _digest(tenant_id, scope, normalize_query(query))
        self._data[key] = (self.clock() + self.ttl_s, _digest(tenant_id, scope), value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def invalidate(self, tenant_id: str, scope: str) -> int:
        """Drop everything for a matter, e.g. when one of its documents changes."""
        tag = _digest(tenant_id, scope)
        stale = [k for k, v in self._data.items() if v[1] == tag]
        for k in stale:
            del self._data[k]
        return len(stale)


class SemanticCache:
    """Returns a cached answer when a new query is close enough to an old one.

    A wrong hit costs more than a miss, so the threshold is tuned for precision (see evals).
    """

    def __init__(self, threshold: float = 0.93, ttl_s: float = 3600, clock: Clock = time.monotonic) -> None:
        self.threshold, self.ttl_s, self.clock = threshold, ttl_s, clock
        self._data: dict[tuple[str, str], list[tuple[float, np.ndarray, Any]]] = {}
        self.stats = CacheStats()

    def get(self, tenant_id: str, scope: str, vector: np.ndarray) -> Any | None:
        now = self.clock()
        entries = [e for e in self._data.get((tenant_id, scope), []) if e[0] >= now]
        self._data[(tenant_id, scope)] = entries
        if entries:
            sims = np.stack([e[1] for e in entries]) @ vector
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                self.stats.hits += 1
                return entries[best][2]
        self.stats.misses += 1
        return None

    def put(self, tenant_id: str, scope: str, vector: np.ndarray, value: Any) -> None:
        self._data.setdefault((tenant_id, scope), []).append((self.clock() + self.ttl_s, vector, value))

    def invalidate(self, tenant_id: str, scope: str) -> int:
        return len(self._data.pop((tenant_id, scope), []))


class PromptCache:
    """Models provider-side prefix caching: a repeated stable prefix is billed at the cache-read rate.

    The first request pays a cache write; later requests within the TTL read it and refresh the TTL.
    Prefixes below the provider minimum are not cacheable.
    """

    def __init__(self, ttl_s: float = 300, min_tokens: int = 512, clock: Clock = time.monotonic) -> None:
        self.ttl_s, self.min_tokens, self.clock = ttl_s, min_tokens, clock
        self._expiry: dict[str, float] = {}
        self.stats = CacheStats()

    def apply(self, model_id: str, prefix: str) -> tuple[int, int]:
        """Returns (cache_read_tokens, cache_write_tokens) for this request's prefix."""
        tokens = count_tokens(prefix)
        if tokens < self.min_tokens:
            return 0, 0
        key, now = _digest(model_id, prefix), self.clock()
        live = key in self._expiry and self._expiry[key] >= now
        self._expiry[key] = now + self.ttl_s
        if live:
            self.stats.hits += 1
            return tokens, 0
        self.stats.misses += 1
        return 0, tokens


class CachedEmbedder:
    """Embedding cache: never pay to embed the same text twice with the same model."""

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder
        self.dim = embedder.dim
        self._data: dict[str, np.ndarray] = {}
        self.stats = CacheStats()

    @property
    def name(self) -> str:
        return self.embedder.name

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [_digest(self.embedder.name, t) for t in texts]
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self._data]
        self.stats.hits += len(texts) - len(missing)
        self.stats.misses += len(missing)
        if missing:
            for (k, _), vec in zip(missing, self.embedder.embed([t for _, t in missing])):
                self._data[k] = vec
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._data[k] for k in keys])

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def clear(self) -> None:
        """Call after re-tuning the embedder: old vectors are from a different model."""
        self._data.clear()
