import json

import pytest
from typer.testing import CliRunner

from sib.cli import app
from sib.config import get_settings

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # ledger/, evals/ and .sib_state/ are relative to the working directory
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_economics_json():
    result = runner.invoke(app, ["economics", "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["customers"] == 10 and 0 < out["gross_margin"] < 1 and out["measured"]["cost_per_ask"] > 0


def test_bench_writes_the_ledger_and_ledger_prints_it(tmp_path):
    result = runner.invoke(app, ["bench", "--change", "caching"])
    assert result.exit_code == 0 and "caching: KEPT" in result.output
    assert (tmp_path / "ledger" / "caching.json").exists()
    assert "Exact + semantic + provider prompt caching" in runner.invoke(app, ["ledger"]).output


def test_ask_and_draft():
    out = runner.invoke(app, ["ask", "firm_aurora", "aurora-m1", "Upon how many days written notice may either party terminate this Lease?"])
    assert out.exit_code == 0 and "days written notice" in out.output and "model claude-opus-5-5" in out.output
    draft = runner.invoke(app, ["draft", "firm_aurora", "aurora-m1", "Upon how many days written notice may either party terminate this Lease?"])
    assert draft.exit_code == 0 and "Aurora Legal LLP" in draft.output


def test_billing_invoices_metered_usage(tmp_path):
    from decimal import Decimal

    from sib.metering import Meter, UsageEvent

    meter = Meter()
    for _ in range(3):
        meter.record(UsageEvent(tenant_id="firm_aurora", feature="ask", model="claude-opus-5-5", cost_usd=Decimal("0.01")))
    meter.save(tmp_path / ".sib_state" / "usage.jsonl")
    out = runner.invoke(app, ["billing", "--period", "2026-10"])
    assert out.exit_code == 0 and "firm_aurora:2026-10: $745 (3 requests)" in out.output
    assert "firm_cedar:2026-10: $0" in out.output  # trial
