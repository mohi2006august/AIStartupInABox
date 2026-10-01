import pytest
from fastapi.testclient import TestClient

from sib.api import create_app
from sib.config import Settings

AURORA = {"X-Tenant-Id": "firm_aurora"}
BIRCH = {"X-Tenant-Id": "firm_birch"}
NOTICE_Q = "Upon how many days written notice may either party terminate this Lease?"


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    root = tmp_path_factory.mktemp("api")
    settings = Settings(state_dir=str(root / "state"), ledger_dir=str(root / "ledger"), evals_dir=str(root / "evals"))
    return TestClient(create_app(settings))


def test_health_and_console(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200 and "<title>" in page.text


def test_tenant_header_is_required_and_checked(client):
    assert client.get("/v1/matters").status_code == 422
    assert client.get("/v1/matters", headers={"X-Tenant-Id": "firm_nobody"}).status_code == 401
    matters = client.get("/v1/matters", headers=AURORA).json()
    assert matters and all(m["id"].startswith("aurora-") for m in matters)


def test_ask_is_cited_metered_and_tenant_scoped(client):
    out = client.post("/v1/matters/aurora-m1/ask", json={"question": NOTICE_Q}, headers=AURORA).json()
    assert "days written notice" in out["text"] and out["citations"] and out["cost_usd"] > 0
    assert client.post("/v1/matters/aurora-m1/ask", json={"question": NOTICE_Q}, headers=BIRCH).status_code == 404
    usage = client.app.state.sib.usage_path.read_text(encoding="utf-8").splitlines()
    assert usage and '"tenant_id":"firm_aurora"' in usage[-1]


def test_draft_and_summary(client):
    body = {"matter_id": "aurora-m1", "doc_type": "termination_notice", "audience": "opposing_counsel",
            "instruction": NOTICE_Q, "action": "vacate the premises"}
    out = client.post("/v1/drafts", json=body, headers=AURORA).json()
    assert "Aurora Legal LLP" in out["text"] and out["provision"]["citations"]
    assert client.post("/v1/drafts", json={**body, "audience": "jury"}, headers=AURORA).status_code == 422
    summary = client.get("/v1/matters/aurora-m1/summary", headers=AURORA).json()
    assert summary and summary[0]["doc_id"] == "aurora-m1-agreement"


def test_assist_dispatches_on_intent(client):
    assert client.post("/v1/matters/aurora-m1/assist", json={"text": NOTICE_Q}, headers=AURORA).json()["intent"] == "ask"
    act = client.post("/v1/matters/aurora-m1/assist", json={"text": "Email the client the latest demand letter"},
                      headers=AURORA).json()
    assert act["intent"] == "action" and "email_send" in act["tools"]


def test_tools_need_approval(client):
    pending = client.post("/v1/tools/email_send", headers=AURORA,
                          json={"arguments": {"to": "a@b.c", "subject": "Notice", "body": "Attached."}}).json()
    assert pending["status"] == "pending_approval"
    assert client.post(f"/v1/approvals/{pending['approval_id']}/approve", headers=BIRCH).status_code == 403
    assert [a["id"] for a in client.get("/v1/approvals", headers=AURORA).json()] == [pending["approval_id"]]
    done = client.post(f"/v1/approvals/{pending['approval_id']}/approve", headers=AURORA).json()
    assert done["result"]["status"] == "sent"
    assert client.post("/v1/tools/email_send", headers=AURORA, json={"arguments": {}}).status_code == 422


def test_economics_ops_and_ledger(client):
    econ = client.get("/v1/economics/summary").json()
    assert econ["customers"] == 10 and econ["measured"]["cost_per_ask"] > 0
    assert client.get("/v1/economics/summary?customers=40").json()["customers"] == 40
    assert client.get("/v1/ops").json()["model_mix"] == {"claude-opus-5-5": 1.0}
    ledger = client.get("/v1/ledger").json()
    assert ledger["records"] == [] and ledger["pending"][0]["id"] == "tuned-embeddings"
    record = client.post("/v1/ledger/bench", json={"change": "caching"}).json()
    assert record["decision"] == "kept"
    assert client.app.state.sib.config.exact_cache  # the stack now runs the kept change
    assert client.post("/v1/ledger/bench", json={"change": "nope"}).status_code == 404
