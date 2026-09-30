from sib.tokens import BPETokenizer, Part, TokenBudgeter, compact_json, count_tokens, fertility, token_tax


def test_bpe_roundtrip_and_compression():
    tok = BPETokenizer()
    corpus = "the court held that the court may hold " * 20
    tok.train(corpus, 300)
    ids = tok.encode("the court held")
    assert tok.decode(ids) == "the court held"
    assert len(ids) < len("the court held".encode())


def test_bpe_no_unknown_tokens_for_unseen_unicode():
    tok = BPETokenizer()
    tok.train("aaaa bbbb " * 10, 270)
    assert tok.decode(tok.encode("नमस्ते ⚖")) == "नमस्ते ⚖"


def test_token_tax_hindi_costs_more():
    en = "The tenant must pay rent on time."
    hi = "किरायेदार को समय पर किराया देना होगा।"
    assert token_tax(hi, en) > 1.0
    assert fertility(hi) > fertility(en)


def test_pack_respects_budget_and_priority():
    b = TokenBudgeter({"ask": 60})
    parts = [Part("system", "You are a legal assistant.", 0), Part("ctx", "clause " * 200, 2), Part("q", "What is the notice period?", 1)]
    res = b.pack("ask", parts)
    assert res.used <= 60
    assert "system" in res.included and "q" in res.included
    assert "ctx" in res.truncated or "ctx" in res.dropped


def test_missing_budget_raises():
    import pytest
    with pytest.raises(KeyError):
        TokenBudgeter().budget_for("nope")


def test_compact_json_is_cheaper():
    import json
    obj = {"items": [{"id": i, "name": f"doc{i}"} for i in range(20)]}
    assert count_tokens(compact_json(obj)) < count_tokens(json.dumps(obj, indent=2))
