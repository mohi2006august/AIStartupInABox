from sib.context import ContextManager, order_for_attention
from sib.tokens import TokenBudgeter, count_tokens


def test_best_passages_go_to_the_edges():
    assert order_for_attention(["p0", "p1", "p2", "p3", "p4"]) == [0, 2, 4, 3, 1]


def test_compaction_keeps_pinned_facts_and_bounds_memory():
    ctx = ContextManager(TokenBudgeter({"ask": 2000}), compact_above=200, keep_recent=2, summary_cap=120)
    ctx.pin("firm_a", "m1", "Limitation expires on 14 March 2027.")
    compacted = False
    for i in range(40):
        ctx.record("firm_a", "m1", "user", f"Question number {i} about clause {i}?")
        compacted |= ctx.record("firm_a", "m1", "assistant", f"Finding {i} is that clause {i} applies. Further detail follows here.")
    mem = ctx.memory("firm_a", "m1")
    assert compacted and len(mem.turns) <= 6
    assert mem.tokens() < 450
    assert "Finding 35" in mem.summary and "Finding 30" in mem.summary and "Finding 2 " not in mem.summary and "Further detail" not in mem.summary
    packed = ctx.build("firm_a", "m1", "ask", "System.", ["passage one."], "What now?")
    assert "Limitation expires on 14 March 2027." in packed.prefix


def test_memory_is_tenant_scoped():
    ctx = ContextManager(TokenBudgeter())
    ctx.pin("firm_a", "m1", "secret fact")
    assert ctx.memory("firm_b", "m1").pinned == []


def test_build_fits_budget_and_drops_lowest_ranked():
    ctx = ContextManager(TokenBudgeter({"ask": 120}))
    passages = [f"Passage {i} " + "word " * 40 for i in range(5)]
    packed = ctx.build("firm_a", "m1", "ask", "System prompt.", passages, "Question?")
    assert packed.used_tokens <= 120
    assert 0 in packed.passage_ranks and 4 not in packed.passage_ranks
    assert len(packed.passages) == len(packed.passage_ranks) < 5
    assert count_tokens(packed.prefix) < 20
