"""Long-matter context management (FR-05): matter memory, compaction and budgeted packing.

Context is a budget, not a bucket. The stable part (instructions, pinned facts, summary) forms a
prompt-cacheable prefix; the volatile part (recent turns, retrieved passages) is packed to fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sib.llm import split_sentences
from sib.tokens import Part, TokenBudgeter, count_tokens, truncate_to_tokens


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str


@dataclass
class MatterMemory:
    pinned: list[str] = field(default_factory=list)  # facts that must survive compaction
    summary_lines: list[str] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return " ".join(self.summary_lines)

    def tokens(self) -> int:
        return count_tokens(self.summary) + sum(count_tokens(t.text) for t in self.turns)


@dataclass
class PackedContext:
    prefix: str  # stable: system + pinned facts + summary (send with cache_control)
    passages: list[str]
    passage_ranks: list[int]  # original retrieval rank of each emitted passage (for citations)
    recent: str
    used_tokens: int
    budget: int
    dropped: list[str]


def order_for_attention(passages: list[str]) -> list[int]:
    """Counter 'lost in the middle': best passages go first and last, weakest in the middle."""
    front, back = [], []
    for rank in range(len(passages)):
        (front if rank % 2 == 0 else back).append(rank)
    return front + back[::-1]


class ContextManager:
    def __init__(self, budgeter: TokenBudgeter, compact_above: int = 600, keep_recent: int = 4, summary_cap: int = 200) -> None:
        self.budgeter = budgeter
        self.compact_above, self.keep_recent, self.summary_cap = compact_above, keep_recent, summary_cap
        self._memory: dict[tuple[str, str], MatterMemory] = {}

    def memory(self, tenant_id: str, matter_id: str) -> MatterMemory:
        return self._memory.setdefault((tenant_id, matter_id), MatterMemory())

    def pin(self, tenant_id: str, matter_id: str, fact: str) -> None:
        mem = self.memory(tenant_id, matter_id)
        if fact not in mem.pinned:
            mem.pinned.append(fact)

    def record(self, tenant_id: str, matter_id: str, role: str, text: str) -> bool:
        """Append a turn; returns True when the memory was compacted."""
        mem = self.memory(tenant_id, matter_id)
        mem.turns.append(Turn(role, text))
        if mem.tokens() > self.compact_above and len(mem.turns) > self.keep_recent:
            self._compact(mem)
            return True
        return False

    def _compact(self, mem: MatterMemory) -> None:
        """Extractive compaction: keep the first sentence of each older assistant answer."""
        old, mem.turns = mem.turns[: -self.keep_recent], mem.turns[-self.keep_recent :]
        lines = mem.summary_lines
        for turn in old:
            if turn.role == "assistant":
                first = split_sentences(turn.text)[:1]
                if first and first[0] not in lines:
                    lines.append(first[0])
        # Newest findings win when the summary itself outgrows its cap.
        while lines and count_tokens(" ".join(lines)) > self.summary_cap:
            lines.pop(0)

    def build(
        self, tenant_id: str, matter_id: str, feature: str, system: str, passages: list[str], question: str,
        reserve_output: int = 0,
    ) -> PackedContext:
        mem = self.memory(tenant_id, matter_id)
        budget = self.budgeter.budget_for(feature) - reserve_output
        prefix_parts = [system]
        if mem.pinned:
            prefix_parts.append("Pinned matter facts:\n" + "\n".join(f"- {f}" for f in mem.pinned))
        if mem.summary:
            prefix_parts.append("Matter summary so far: " + mem.summary)
        prefix = "\n\n".join(prefix_parts)
        fixed = count_tokens(prefix) + count_tokens(question)
        parts = [Part(f"passage:{i}", p, priority=i) for i, p in enumerate(passages)]
        recent_text = "\n".join(f"{t.role}: {t.text}" for t in mem.turns)
        if recent_text:
            parts.append(Part("recent", recent_text, priority=len(passages)))
        packed = TokenBudgeter({feature: max(budget - fixed, 0)}).pack(feature, parts)
        kept: list[tuple[int, str]] = []
        for i, passage in enumerate(passages):
            name = f"passage:{i}"
            if name in packed.truncated:
                room = max(budget - fixed - sum(count_tokens(t) for _, t in kept), 0)
                kept.append((i, truncate_to_tokens(passage, room)))
            elif name in packed.included:
                kept.append((i, passage))
        ordered = [kept[i] for i in order_for_attention([t for _, t in kept])]
        recent = recent_text if "recent" in packed.included and "recent" not in packed.truncated else ""
        used = fixed + sum(count_tokens(t) for _, t in ordered) + count_tokens(recent)
        return PackedContext(prefix, [t for _, t in ordered], [r for r, _ in ordered], recent, used, budget, packed.dropped)
