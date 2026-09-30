from decimal import Decimal

from sib.corpus import CLAUSES, build_corpus
from sib.llm import ABSTAIN, CATALOG, LLMRequest, SimProvider, Usage, cost_usd, latency_ms


def test_corpus_is_deterministic_and_tenant_scoped():
    a, b = build_corpus(), build_corpus()
    assert [d.text for d in a.documents] == [d.text for d in b.documents]
    assert len(a.tenants) == 3 and len(a.matters) == 12
    for q in a.questions:
        doc = next(d for d in a.documents if d.id == f"{q.matter_id}-agreement")
        assert doc.tenant_id == q.tenant_id and q.expected in doc.text


def test_training_pairs_do_not_leak_eval_questions():
    corpus = build_corpus()
    eval_questions = {q.question for q in corpus.questions}
    assert not eval_questions & {p[0] for p in corpus.training_pairs}


def test_cost_matches_list_prices():
    usage = Usage(input_tokens=1_000_000, cache_read_tokens=1_000_000, output_tokens=100_000)
    assert cost_usd(CATALOG["claude-opus-5-5"], usage) == Decimal("4.00") + Decimal("0.20") + Decimal("2.00")


def test_cached_prefix_is_cheaper_and_faster():
    spec = CATALOG["claude-opus-5-5"]
    cold, warm = Usage(2000, 0, 0, 50), Usage(500, 1500, 0, 50)
    assert cost_usd(spec, warm) < cost_usd(spec, cold)
    assert latency_ms(spec, warm, "k") < latency_ms(spec, cold, "k")


def _passages():
    return [" ".join(c.template.format(notice="ninety (90)", rent="INR 1", day="fifth (5th)", deposit="INR 2",
                                       cap="INR 3", city="Pune") for c in CLAUSES["lease"])]


def test_frontier_answers_hard_question_with_citation():
    out = SimProvider(CATALOG["claude-opus-5-5"]).complete(
        LLMRequest("sys", _passages(), "How much warning is needed to end the tenancy?"))
    assert "ninety (90)" in out.text and out.citation == 1


def test_weaker_models_know_a_subset_of_the_frontier_vocabulary():
    strong, weak = SimProvider(CATALOG["claude-opus-5-5"]), SimProvider(CATALOG["sib-8b-awq"])
    words = ["tenant", "landlord", "vendor", "bond", "rival", "secret", "bill", "warning", "tenancy", "upkeep"]
    assert all(strong.knows(w) for w in words)
    assert 0 < sum(weak.knows(w) for w in words) < len(words)


def test_abstains_without_evidence():
    out = SimProvider(CATALOG["claude-opus-5-5"]).complete(LLMRequest("sys", ["Unrelated text."], "zebra migration?"))
    assert out.text == ABSTAIN and out.citation is None
