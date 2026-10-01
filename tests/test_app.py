"""Stack behaviour, including the tenant-isolation suite."""
from decimal import Decimal

import pytest

from sib.app import CHEAP_API, FRONTIER, SELF_HOSTED, Stack, StackConfig
from sib.llm import ABSTAIN


@pytest.fixture(scope="module")
def stack() -> Stack:
    return Stack()


def _q(stack: Stack, difficulty: str = "easy", tenant: str = "firm_aurora"):
    return next(q for q in stack.corpus.questions if q.difficulty == difficulty and q.tenant_id == tenant)


def test_ask_returns_a_cited_correct_answer(stack):
    q = _q(stack, "hard")
    answer = stack.ask(q.tenant_id, q.matter_id, q.question)
    assert q.expected in answer.text and answer.model == FRONTIER
    cite = answer.citations[0]
    assert answer.text in cite.quote and cite.doc_id.startswith(q.matter_id)
    assert answer.cost_usd > 0 and answer.latency_ms > 0


def test_every_citation_stays_inside_the_asking_tenant_and_matter(stack):
    for q in stack.corpus.questions:
        answer = stack.ask(q.tenant_id, q.matter_id, q.question)
        for cite in answer.citations:
            chunk = stack.index._chunks[q.tenant_id][cite.chunk_id]
            assert chunk.tenant_id == q.tenant_id and chunk.matter_id == q.matter_id


def test_cross_tenant_matter_access_is_refused(stack):
    q = _q(stack, tenant="firm_aurora")
    with pytest.raises(KeyError):
        stack.ask("firm_birch", q.matter_id, q.question)
    with pytest.raises(KeyError):
        stack.draft("firm_birch", q.matter_id, "demand_letter", "client", q.question, "pay")


def test_caches_hit_within_a_tenant_and_never_across_tenants():
    stack = Stack(StackConfig(exact_cache=True, semantic_cache=True))
    a = _q(stack, tenant="firm_aurora")
    first = stack.ask(a.tenant_id, a.matter_id, a.question)
    assert first.cache_layer == "none"
    assert stack.ask(a.tenant_id, a.matter_id, a.question).cache_layer == "exact"
    again = stack.ask(a.tenant_id, a.matter_id, "Quick question - " + a.question)
    assert again.cache_layer == "semantic" and again.cost_usd == 0 and again.text == first.text
    b = _q(stack, tenant="firm_birch")
    assert stack.ask(b.tenant_id, b.matter_id, a.question).cache_layer == "none"


def test_semantic_cache_does_not_confuse_different_clauses():
    stack = Stack(StackConfig(semantic_cache=True))
    for q in stack.corpus.questions:
        answer = stack.ask(q.tenant_id, q.matter_id, q.question)
        assert q.expected in answer.text, q.question


def test_prompt_cache_cuts_the_cost_of_the_next_request_in_a_matter():
    stack = Stack(StackConfig(prompt_cache=True))
    q1, q2 = [q for q in stack.corpus.questions if q.matter_id == "aurora-m1"][:2]
    cold = stack.ask(q1.tenant_id, q1.matter_id, q1.question)
    stack.clock.advance(30)
    warm = stack.ask(q2.tenant_id, q2.matter_id, q2.question)
    assert warm.cost_usd < cold.cost_usd
    assert stack.meter.events[-1].cache_read_tokens > 0 and stack.meter.events[0].cache_write_tokens > 0


def test_router_sends_easy_questions_to_the_small_model():
    stack = Stack(StackConfig(router=True))
    easy, hard = _q(stack, "easy"), _q(stack, "hard")
    assert stack.ask(easy.tenant_id, easy.matter_id, easy.question).model == CHEAP_API
    routed = stack.ask(hard.tenant_id, hard.matter_id, hard.question)
    assert routed.model == FRONTIER and hard.expected in routed.text


def test_self_hosted_outage_fails_over_to_the_api():
    stack = Stack(StackConfig(router=True, selfhost=True))
    easy = _q(stack, "easy")
    assert stack.ask(easy.tenant_id, easy.matter_id, easy.question).model == SELF_HOSTED
    stack.selfhost_healthy = False
    answer = stack.ask(easy.tenant_id, easy.matter_id, easy.question)
    assert answer.model == FRONTIER and answer.route_class.endswith("failover")


def test_unsupported_claims_are_escalated_or_withheld(stack):
    results = [stack.ask(q.tenant_id, q.matter_id, q.question, model=SELF_HOSTED)
               for q in stack.corpus.questions if q.difficulty == "hard"]
    escalated = [a for a in results if a.escalated]
    assert escalated and all(a.model == FRONTIER for a in escalated)
    assert all(a.supported or a.text == ABSTAIN for a in results)


def test_draft_uses_firm_style_adapter_when_self_hosting():
    api = Stack()
    hosted = Stack(StackConfig(selfhost=True))
    q = next(q for q in api.corpus.questions if q.tenant_id == "firm_aurora" and q.clause == "notice")
    args = (q.tenant_id, q.matter_id, "termination_notice", "opposing_counsel", q.question, "vacate the premises")
    d_api, d_host = api.draft(*args), hosted.draft(*args)
    assert not d_api.used_adapter and d_api.model == FRONTIER
    assert d_host.used_adapter and d_host.model == SELF_HOSTED and d_host.cost_usd < d_api.cost_usd
    for d in (d_api, d_host):
        assert d.text.rstrip().endswith("[Commercial Lease Agreement]") and "Aurora Legal LLP" in d.text
    assert hosted.meter.events[-1].feature == "draft"


def test_summary_is_cited_and_metered(stack):
    before = len(stack.meter.events)
    points = stack.summarize("firm_aurora", "aurora-m1")
    assert points and all(p.doc_id == "aurora-m1-agreement" for p in points)
    assert stack.meter.events[before].feature == "summarize" and stack.meter.events[before].cost_usd > Decimal(0)


def test_memory_is_recorded_only_when_asked(stack):
    q = _q(stack)
    stack.ask(q.tenant_id, q.matter_id, q.question, remember=True)
    turns = stack.context.memory(q.tenant_id, q.matter_id).turns
    assert turns[-2].text == q.question and turns[-1].role == "assistant"
