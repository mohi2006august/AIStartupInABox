"""Tokenizer-aware prompt design (FR-10).

Two tokenizers: a trainable byte-level BPE (for teaching/fertility experiments) and a fast
approximate counter used on hot paths. Budgets are per feature; packing is priority-ordered.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Approximate tokens: ~4 ASCII chars/token; non-ASCII scripts cost ~1 token per 1.5 chars."""
    total = text.count("
")  # newline + indentation runs cost ~1 token each
    for piece in _WORD.findall(text):
        ascii_chars = sum(1 for c in piece if ord(c) < 128)
        other = len(piece) - ascii_chars
        total += max(1, -(-ascii_chars // 4)) if ascii_chars else 0
        total += -(-other * 2 // 3) if other else 0
    return total


def fertility(text: str) -> float:
    """Tokens per whitespace word; higher means more cost and less effective context."""
    words = text.split()
    return count_tokens(text) / len(words) if words else 0.0


def token_tax(text: str, english_baseline: str) -> float:
    """How many times more expensive `text` is than the English baseline for the same content."""
    base = count_tokens(english_baseline)
    return count_tokens(text) / base if base else 0.0


class BPETokenizer:
    """Byte-level BPE: starts from 256 bytes, so it never hits an unknown token."""

    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, corpus: str, vocab_size: int) -> None:
        ids = list(corpus.encode("utf-8"))
        while len(self.vocab) < vocab_size:
            pairs = Counter(zip(ids, ids[1:]))
            if not pairs:
                break
            pair, freq = pairs.most_common(1)[0]
            if freq < 2:
                break
            new_id = len(self.vocab)
            self.merges.append(pair)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            ids = self._merge(ids, pair, new_id)

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for offset, pair in enumerate(self.merges):
            ids = self._merge(ids, pair, 256 + offset)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


# --- Budgets -------------------------------------------------------------------------------

DEFAULT_BUDGETS: dict[str, int] = {
    "ask": 6000,
    "draft": 8000,
    "summarize": 4000,
    "classify": 400,
}


@dataclass
class Part:
    name: str
    text: str
    priority: int  # lower packs first
    tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = count_tokens(self.text)


@dataclass
class PackResult:
    text: str
    used: int
    budget: int
    included: list[str]
    dropped: list[str]
    truncated: list[str]


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


class TokenBudgeter:
    def __init__(self, budgets: dict[str, int] | None = None) -> None:
        self.budgets = dict(DEFAULT_BUDGETS if budgets is None else budgets)

    def budget_for(self, feature: str) -> int:
        if feature not in self.budgets:
            raise KeyError(f"no token budget configured for feature '{feature}'")
        return self.budgets[feature]

    def pack(self, feature: str, parts: list[Part], reserve_output: int = 0) -> PackResult:
        budget = self.budget_for(feature) - reserve_output
        remaining = budget
        kept: list[tuple[int, Part, str]] = []
        dropped: list[str] = []
        truncated: list[str] = []
        for idx, part in sorted(enumerate(parts), key=lambda ip: (ip[1].priority, ip[0])):
            if part.tokens <= remaining:
                kept.append((idx, part, part.text))
                remaining -= part.tokens
            elif remaining > 20:
                cut = truncate_to_tokens(part.text, remaining)
                kept.append((idx, part, cut))
                remaining -= count_tokens(cut)
                truncated.append(part.name)
            else:
                dropped.append(part.name)
        kept.sort(key=lambda k: k[0])  # restore original order for the prompt
        text = "\n\n".join(t for _, _, t in kept)
        return PackResult(text, budget - remaining, budget, [p.name for _, p, _ in kept], dropped, truncated)


def compact_json(obj: object) -> str:
    """Compact separators save ~10-15% tokens over pretty-printed JSON in tool payloads."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
