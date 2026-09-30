import io
import json

import pytest

from sib.corpus import build_corpus
from sib.mcp_gateway import DriveConnector, McpGateway, McpServer, PolicyError


@pytest.fixture
def gw() -> McpGateway:
    return McpGateway(DriveConnector(build_corpus().documents))


def test_read_tools_run_and_are_tenant_scoped(gw):
    out = gw.call("firm_aurora", "lawyer1", "drive_search", {"query": "lease agreement"})
    assert out["status"] == "ok" and out["result"]
    file_id = out["result"][0]["file_id"]
    assert gw.call("firm_aurora", "lawyer1", "drive_read", {"file_id": file_id})["result"]["text"]
    with pytest.raises(PolicyError):
        gw.call("firm_birch", "lawyer9", "drive_read", {"file_id": file_id})


def test_write_tools_need_approval_from_the_same_firm(gw):
    args = {"to": "counsel@example.com", "subject": "Notice", "body": "Please see attached."}
    out = gw.call("firm_aurora", "lawyer1", "email_send", args)
    assert out["status"] == "pending_approval" and gw.email.sent == {}
    with pytest.raises(PolicyError):
        gw.approve("firm_birch", out["approval_id"], "intruder")
    assert gw.approve("firm_aurora", out["approval_id"], "partner1")["result"]["status"] == "sent"
    assert len(gw.email.sent["firm_aurora"]) == 1
    with pytest.raises(PolicyError):
        gw.approve("firm_aurora", out["approval_id"], "partner1")  # no replay


def test_denied_action_never_executes(gw):
    out = gw.call("firm_aurora", "lawyer1", "calendar_create", {"title": "Hearing", "start": "2026-10-05T10:00"})
    assert gw.deny("firm_aurora", out["approval_id"], "partner1") == {"status": "denied"}
    assert gw.calendar.events == {}


def test_arguments_are_validated_and_calls_audited(gw):
    with pytest.raises(ValueError):
        gw.call("firm_aurora", "u", "email_send", {"to": "a@b.c"})
    with pytest.raises(ValueError):
        gw.call("firm_aurora", "u", "drive_search", {"query": "x", "limit": "many"})
    with pytest.raises(PolicyError):
        gw.call("firm_aurora", "u", "shell_exec", {})
    gw.call("firm_aurora", "u", "drive_search", {"query": "confidential secret"})
    entry = gw.audit[-1]
    assert entry.outcome == "executed" and "secret" not in entry.args_digest


def test_mcp_server_speaks_json_rpc(gw):
    server = McpServer(gw, "firm_aurora", "lawyer1")
    init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})
    assert init["result"]["protocolVersion"] == "2025-03-26" and "tools" in init["result"]["capabilities"]
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    tools = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    assert {t["name"] for t in tools} >= {"drive_search", "email_send"}
    assert next(t for t in tools if t["name"] == "email_send")["annotations"]["readOnlyHint"] is False
    call = server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                          "params": {"name": "email_send", "arguments": {"to": "a@b.c", "subject": "s", "body": "b"}}})
    assert json.loads(call["result"]["content"][0]["text"])["status"] == "pending_approval"
    bad = server.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope", "arguments": {}}})
    assert bad["result"]["isError"] is True
    assert server.handle({"jsonrpc": "2.0", "id": 5, "method": "resources/list"})["error"]["code"] == -32601


def test_stdio_transport_round_trip(gw):
    stdin = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"ping"}\nnot json\n')
    stdout = io.StringIO()
    McpServer(gw, "firm_aurora", "lawyer1").serve(stdin, stdout)
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[0] == {"jsonrpc": "2.0", "id": 1, "result": {}} and lines[1]["error"]["code"] == -32700
