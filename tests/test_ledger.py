from datetime import date

import pytest

from sib.app import StackConfig
from sib.ledger import BASELINE, CHANGES, Ledger, bench, default_scenario, render_markdown
from sib.workload import build_workload, run_workload


def test_workload_is_reproducible_and_benchmarks_are_deterministic():
    from sib.corpus import build_corpus

    corpus = build_corpus()
    assert [r.text for r in build_workload(corpus)] == [r.text for r in build_workload(corpus)]
    a, b = run_workload(BASELINE), run_workload(BASELINE)
    assert a == b
    assert a.model_mix == {"claude-opus-5-5": 1.0} and a.cache_hit_ratio == 0


def test_caching_lowers_cogs_without_hurting_quality():
    before = run_workload(StackConfig())
    after = run_workload(StackConfig(exact_cache=True, semantic_cache=True, prompt_cache=True))
    assert after.cogs_per_1k < 0.7 * before.cogs_per_1k
    assert after.ask_quality >= before.ask_quality - 0.5
    assert after.cache_hit_ratio > 0.2 and after.prompt_cache_read_share > 0


def test_bench_records_kept_and_reverted_changes(tmp_path):
    ledger = Ledger(tmp_path)
    assert ledger.latest_change_id() == CHANGES[0].id
    kept = bench("caching", ledger, on=date(2026, 10, 1))
    assert kept.decision == "kept" and kept.cogs_per_1k_after < kept.cogs_per_1k_before
    assert kept.gross_margin_delta_pts > 0 and kept.runway_delta_months > 0
    assert ledger.current_config().exact_cache
    shrink = bench("context-6800", ledger, on=date(2026, 10, 2))
    assert shrink.decision == "reverted" and shrink.reason.startswith("quality gate")
    assert ledger.current_config().ask_budget == BASELINE.ask_budget
    md = (tmp_path / "LEDGER.md").read_text(encoding="utf-8")
    assert "reverted" in md and "kept" in md and "✖" in md


def test_self_hosting_waits_for_volume(tmp_path):
    ledger = Ledger(tmp_path)
    for change_id in ("tuned-embeddings", "caching", "router"):  # the production config it would join
        bench(change_id, ledger, on=date(2026, 10, 1))
    record = bench("selfhost", ledger, on=date(2026, 10, 1), write=False)
    assert record.decision == "reverted" and "GPU node" in record.reason
    busy = default_scenario().model_copy(update={"customers": 40})
    assert bench("selfhost", ledger, scenario=busy, on=date(2026, 10, 1), write=False).decision == "kept"


def test_unknown_change_is_rejected(tmp_path):
    with pytest.raises(KeyError):
        bench("make-it-faster", Ledger(tmp_path))


def test_markdown_handles_an_empty_ledger():
    assert render_markdown([]).startswith("# Decision ledger")
