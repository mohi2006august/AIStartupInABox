from decimal import Decimal

import numpy as np
import pytest

from sib.drafting import (
    AUDIENCES, DOC_TYPES, DraftRequest, FirmStyle, StyleModel, all_contexts, featurize, few_shot_choices,
    fit_firm_adapter, model_choices, render_draft, sample_examples, style_match, train_lora,
)
from sib.llm import CATALOG
from sib.quantize import awq_quantize, dequantize, memory_bytes, output_error, quantize, serving_cost_per_mtok


@pytest.fixture(scope="module")
def base() -> StyleModel:
    model = StyleModel()
    losses = model.train_base(sample_examples(3000, seed=1))
    assert losses[-1] < losses[0]
    return model


def clean_set(style: FirmStyle):
    return [(s, d, a, style.preferred(s, d, a)) for s, d, a in all_contexts()]


def test_base_learns_public_style_but_not_house_style(base):
    assert base.accuracy(sample_examples(400, seed=2, noise=0)) > 0.98
    assert base.accuracy(clean_set(FirmStyle.generate("firm_aurora"))) < 0.8


def test_lora_trains_few_params_and_leaves_base_frozen(base):
    style = FirmStyle.generate("firm_aurora")
    data = sample_examples(300, seed=5, style=style)
    before = base.W.copy()
    adapter, history = train_lora(base, data[60:], data[:60], rank=8)
    assert np.array_equal(base.W, before)
    assert adapter.n_params < 0.35 * base.W.size
    assert history["train"][-1] < history["train"][0]
    assert base.accuracy(clean_set(style), adapter) > base.accuracy(clean_set(style)) + 0.2


def test_guard_accepts_with_enough_data_and_beats_unguarded(base):
    style = FirmStyle.generate("firm_aurora")
    data = sample_examples(300, seed=5, style=style)
    adapter, report = fit_firm_adapter(base, "firm_aurora", data, rank=8)
    assert report.accepted and adapter is not None and report.gain_pts >= 2
    assert report.epochs_run < 400  # early stopped
    loose, loose_report = fit_firm_adapter(base, "firm_aurora", data, rank=64, guard=False)
    assert loose_report.canary_memorised > report.canary_memorised
    assert base.accuracy(clean_set(style), adapter) >= base.accuracy(clean_set(style), loose)


def test_guard_falls_back_to_base_on_tiny_firm_dataset(base):
    data = sample_examples(60, seed=5, style=FirmStyle.generate("firm_aurora"))
    adapter, report = fit_firm_adapter(base, "firm_aurora", data, rank=8)
    assert adapter is None and not report.accepted and "fall back" in report.reason


def test_guard_rejects_high_rank():
    with pytest.raises(ValueError):
        fit_firm_adapter(StyleModel(), "t", sample_examples(50, 1), rank=32)


def test_adapter_beats_few_shot_prompting(base):
    style = FirmStyle.generate("firm_aurora")
    adapter, _ = fit_firm_adapter(base, "firm_aurora", sample_examples(300, seed=5, style=style))
    shots = sample_examples(20, seed=9, style=style)
    contexts = [(d, a) for d in DOC_TYPES for a in AUDIENCES]
    lora = np.mean([style_match(model_choices(base, d, a, adapter), style, d, a) for d, a in contexts])
    shot = np.mean([style_match(few_shot_choices(shots, d, a), style, d, a) for d, a in contexts])
    assert lora > shot


def test_render_draft_uses_chosen_variants():
    req = DraftRequest("firm_aurora", "demand_letter", "opposing_counsel", "Acme Textiles", "Borealis Foods",
                       "Acme Textiles v. Borealis Foods", "pay the outstanding rent of INR 240,000",
                       on=__import__("datetime").date(2026, 9, 30))
    text = render_draft({s: 0 for s in __import__("sib.drafting", fromlist=["SLOT_NAMES"]).SLOT_NAMES}, req, "Aurora Legal LLP")
    assert "In the matter of Acme Textiles v. Borealis Foods" in text
    assert "Dated this 30th day of September, 2026" in text
    assert "our client shall pursue" in text and text.endswith("Aurora Legal LLP")


def test_quantisation_error_shrinks_with_bits_and_awq_never_loses():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(30, 64)).astype(np.float32)
    x = rng.normal(size=(200, 64)).astype(np.float32)
    x[:, :4] *= 25  # a few outlier activation channels, as in real transformers
    e8, e4 = output_error(w, dequantize(quantize(w, 8)), x), output_error(w, dequantize(quantize(w, 4)), x)
    assert e8 < e4
    t, alpha = awq_quantize(w, x, 4)
    assert output_error(w, dequantize(t), x) < e4 and alpha > 0
    assert t.q.dtype == np.int8 and np.abs(t.q).max() <= 7


def test_quantised_style_model_keeps_accuracy(base):
    test = sample_examples(400, seed=3, noise=0)
    calib = np.stack([featurize(s, d, a) for s, d, a, _ in sample_examples(200, seed=4)])
    q = StyleModel()
    q.W = dequantize(awq_quantize(base.W, calib, 4)[0])
    assert q.accuracy(test) >= base.accuracy(test) - 0.02
    assert memory_bytes(base.W.size, 4) == memory_bytes(base.W.size, 16) // 4


def test_catalog_self_hosted_prices_match_serving_model():
    assert serving_cost_per_mtok(4) == CATALOG["sib-8b-awq"].input_per_mtok == Decimal("0.16")
    assert serving_cost_per_mtok(16) == CATALOG["sib-8b-fp16"].input_per_mtok
