"""LLM layer: model catalog with prices, usage/cost accounting and two providers.

`SimProvider` is a deterministic offline stand-in so the whole product and its economics run with
no API keys. Its capability knob is vocabulary knowledge: weaker models miss colloquial synonyms
and occasionally produce an unsupported claim, which is what the verifier and evals must catch.
`ClaudeProvider` calls the real API when credentials are available.
"""
from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, Protocol

from sib.corpus import SYNONYM_TO_CANONICAL, stems
from sib.embeddings import STOPWORDS
from sib.tokens import count_tokens

log = logging.getLogger("sib.llm")

Tier = Literal["frontier", "cheap_api", "self_hosted"]
MTOK = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    tier: Tier
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_read_per_mtok: Decimal
    cache_write_per_mtok: Decimal
    ttft_ms: int  # latency figures are modelling assumptions, not measurements
    ms_per_output_token: float
    knowledge: float  # SimProvider only: share of colloquial synonyms the model understands


# API prices are Anthropic list prices per million tokens (cache read ~0.1x, cache write 1.25x input).
# Self-hosted prices are derived in sib.quantize.serving_cost_per_mtok from GPU cost and throughput.
CATALOG: dict[str, ModelSpec] = {
    "claude-opus-5-5": ModelSpec("claude-opus-5-5", "frontier", Decimal("4.00"), Decimal("20.00"),
                                 Decimal("0.20"), Decimal("5.00"), 900, 14.0, 1.0),
    "claude-haiku-4-5": ModelSpec("claude-haiku-4-5", "cheap_api", Decimal("1.00"), Decimal("5.00"),
                                  Decimal("0.10"), Decimal("1.25"), 350, 6.0, 0.80),
    "sib-8b-awq": ModelSpec("sib-8b-awq", "self_hosted", Decimal("0.16"), Decimal("0.16"),
                            Decimal("0.16"), Decimal("0.16"), 250, 9.0, 0.62),
    "sib-8b-fp16": ModelSpec("sib-8b-fp16", "self_hosted", Decimal("0.56"), Decimal("0.56"),
                             Decimal("0.56"), Decimal("0.56"), 320, 18.0, 0.66),
}
FRONTIER = "claude-opus-5-5"


@dataclass
class Usage:
    input_tokens: int = 0  # uncached input
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


def cost_usd(spec: ModelSpec, usage: Usage) -> Decimal:
    return (
        usage.input_tokens * spec.input_per_mtok
        + usage.cache_read_tokens * spec.cache_read_per_mtok
        + usage.cache_write_tokens * spec.cache_write_per_mtok
        + usage.output_tokens * spec.output_per_mtok
    ) / MTOK


def latency_ms(spec: ModelSpec, usage: Usage, jitter_key: str = "") -> int:
    """Modelled latency: time-to-first-token (cached prefix is ~free) plus decode time."""
    prefill = (usage.input_tokens + usage.cache_write_tokens) * 0.05 + usage.cache_read_tokens * 0.005
    jitter = 1.0 + (zlib.crc32(jitter_key.encode()) % 40) / 100  # deterministic 0-40% tail
    return int((spec.ttft_ms + prefill + usage.output_tokens * spec.ms_per_output_token) * jitter)


@dataclass
class LLMRequest:
    system: str  # stable prefix: instructions, firm style, matter memory (prompt-cacheable)
    passages: list[str]
    question: str
    max_tokens: int = 1024
    cached_prefix_tokens: int = 0  # set by the prompt cache layer
    cache_write_tokens: int = 0


@dataclass
class Completion:
    text: str
    model: str
    usage: Usage
    citation: int | None = None  # 1-based passage index the answer relies on
    refused: bool = False
    meta: dict[str, object] = field(default_factory=dict)


class Provider(Protocol):
    spec: ModelSpec

    def complete(self, req: LLMRequest) -> Completion: ...


ABSTAIN = "I could not find this in the matter documents."


def split_sentences(text: str) -> list[str]:
    out, buf = [], []
    for token in text.split():
        buf.append(token)
        if token.endswith((".", "?", "!")):
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out


def _usage_for(req: LLMRequest, output: str) -> Usage:
    total_in = count_tokens(req.system) + sum(count_tokens(p) for p in req.passages) + count_tokens(req.question)
    cached = min(req.cached_prefix_tokens, total_in)
    written = min(req.cache_write_tokens, total_in - cached)
    return Usage(total_in - cached - written, cached, written, count_tokens(output))


class SimProvider:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def knows(self, word: str) -> bool:
        return zlib.crc32(word.encode()) % 100 < self.spec.knowledge * 100

    def understand(self, text: str) -> set[str]:
        """Content words mapped to canonical drafting terms, as far as this model knows them."""
        out = set()
        for w in stems(text):
            if w in STOPWORDS:
                continue
            canonical = SYNONYM_TO_CANONICAL.get(w)
            out.add(canonical if canonical and self.knows(w) else w)
        return out

    def complete(self, req: LLMRequest) -> Completion:
        q = self.understand(req.question)
        best: tuple[float, int, str] | None = None
        for idx, passage in enumerate(req.passages, start=1):
            for sentence in split_sentences(passage):
                overlap = len(q & self.understand(sentence))
                tie_break = (zlib.crc32((sentence + self.spec.id).encode()) % 1000) / 1e6
                if best is None or overlap + tie_break > best[0]:
                    best = (overlap + tie_break, idx, sentence)
        if best is None or best[0] < 1:
            text, citation = ABSTAIN, None
        elif best[0] < 2 and self.spec.knowledge < 1.0:
            # Weak evidence + weak model: a confident, unsupported claim (what verification must catch).
            text, citation = "The agreement provides for a standard period of thirty (30) days in such cases.", best[1]
        else:
            text, citation = best[2], best[1]
        return Completion(text, self.spec.id, _usage_for(req, text), citation)


class ClaudeProvider:
    """Real Claude API. Requires `pip install anthropic` and credentials in the environment."""

    def __init__(self, spec: ModelSpec) -> None:
        import anthropic  # optional dependency, imported lazily

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.spec = spec

    def _messages(self, req: LLMRequest) -> list[dict[str, object]]:
        context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(req.passages, start=1))
        prompt = (
            f"Passages:\n{context}\n\nQuestion: {req.question}\n\n"
            "Answer only from the passages and end with the passage number in square brackets, e.g. [2]. "
            f"If the passages do not contain the answer, reply exactly: {ABSTAIN}"
        )
        return [{"role": "user", "content": prompt}]

    def complete(self, req: LLMRequest) -> Completion:
        anthropic = self._anthropic
        system = [{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}]
        try:
            if self.spec.id == "claude-haiku-4-5":
                resp = self.client.messages.create(
                    model=self.spec.id, max_tokens=req.max_tokens, system=system, messages=self._messages(req)
                )
            else:
                # Opus 5.5: thinking is always on, effort is the only depth control; server-side
                # fallbacks re-run a safety-declined request on another model inside the same call.
                resp = self.client.beta.messages.create(
                    model=self.spec.id, max_tokens=req.max_tokens, system=system, messages=self._messages(req),
                    output_config={"effort": "low"},
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                )
        except anthropic.RateLimitError:
            log.warning("claude_rate_limited", extra={"model": self.spec.id})
            raise
        except anthropic.APIStatusError as exc:
            log.error("claude_api_error", extra={"model": self.spec.id, "status": exc.status_code})
            raise
        except anthropic.APIConnectionError:
            log.error("claude_connection_error", extra={"model": self.spec.id})
            raise
        usage = Usage(
            resp.usage.input_tokens,
            resp.usage.cache_read_input_tokens or 0,
            resp.usage.cache_creation_input_tokens or 0,
            resp.usage.output_tokens,
        )
        if resp.stop_reason == "refusal":
            return Completion(ABSTAIN, self.spec.id, usage, None, refused=True)
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        citation = None
        if text.endswith("]") and "[" in text:
            tail = text[text.rindex("[") + 1 : -1]
            if tail.isdigit() and 1 <= int(tail) <= len(req.passages):
                citation, text = int(tail), text[: text.rindex("[")].strip()
        return Completion(text, resp.model, usage, citation)


def make_provider(model_id: str, live: bool = False) -> Provider:
    spec = CATALOG[model_id]
    if live and spec.tier != "self_hosted":
        return ClaudeProvider(spec)
    return SimProvider(spec)
