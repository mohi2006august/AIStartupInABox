"""MCP gateway (FR-02): email, drive and calendar tools with policy, approvals and audit.

Read tools run immediately. Write tools (sending email, creating events) never run on the model's
say-so: they create a pending action that a human in the same firm must approve. Every call is
audited. `McpServer` exposes the same tools over MCP (JSON-RPC 2.0, newline-delimited stdio).

The connectors here are local in-memory stand-ins; swapping in Gmail/Drive/Graph clients means
implementing the same three small classes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TextIO

from sib import __version__
from sib.corpus import Document
from sib.embeddings import content_words
from sib.tokens import compact_json

log = logging.getLogger("sib.mcp")


# --- Connectors ----------------------------------------------------------------------------


class DriveConnector:
    def __init__(self, documents: list[Document]) -> None:
        self._docs: dict[str, dict[str, Document]] = {}
        for doc in documents:
            self._docs.setdefault(doc.tenant_id, {})[doc.id] = doc

    def search(self, tenant_id: str, query: str, limit: int = 5) -> list[dict[str, str]]:
        terms = set(content_words(query))
        scored = [(len(terms & set(content_words(d.title + " " + d.text))), d) for d in self._docs.get(tenant_id, {}).values()]
        ranked = sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])[:limit]
        return [{"file_id": d.id, "title": d.title, "matter": d.matter_id} for _, d in ranked]

    def read(self, tenant_id: str, file_id: str) -> dict[str, str]:
        doc = self._docs.get(tenant_id, {}).get(file_id)
        if doc is None:
            raise KeyError(f"file '{file_id}' not found")
        return {"file_id": doc.id, "title": doc.title, "text": doc.text}


class EmailConnector:
    def __init__(self) -> None:
        self.sent: dict[str, list[dict[str, str]]] = {}

    def send(self, tenant_id: str, to: str, subject: str, body: str) -> dict[str, str]:
        message = {"to": to, "subject": subject, "body": body}
        self.sent.setdefault(tenant_id, []).append(message)
        return {"status": "sent", "to": to, "subject": subject}

    def search(self, tenant_id: str, query: str) -> list[dict[str, str]]:
        terms = set(content_words(query))
        return [m for m in self.sent.get(tenant_id, []) if terms & set(content_words(m["subject"] + " " + m["body"]))]


class CalendarConnector:
    def __init__(self) -> None:
        self.events: dict[str, list[dict[str, Any]]] = {}

    def list(self, tenant_id: str, day: str) -> list[dict[str, Any]]:
        return [e for e in self.events.get(tenant_id, []) if e["start"].startswith(day)]

    def create(self, tenant_id: str, title: str, start: str, attendees: list[str] | None = None) -> dict[str, Any]:
        event = {"title": title, "start": start, "attendees": attendees or []}
        self.events.setdefault(tenant_id, []).append(event)
        return {"status": "created", **event}


# --- Gateway -------------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str  # written for the model: say what it does and when to use it
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    requires_approval: bool = False


@dataclass
class AuditEntry:
    at: str
    tenant_id: str
    user: str
    tool: str
    args_digest: str
    outcome: str  # executed | pending_approval | approved | denied | error


@dataclass
class PendingAction:
    id: str
    tenant_id: str
    user: str
    tool: str
    args: dict[str, Any]
    status: str = "pending"


class PolicyError(Exception):
    pass


_TYPES = {"string": str, "integer": int, "array": list, "boolean": bool, "object": dict}


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> None:
    for name in schema.get("required", []):
        if name not in args:
            raise ValueError(f"missing required argument '{name}'")
    props = schema.get("properties", {})
    for name, value in args.items():
        if name not in props:
            raise ValueError(f"unknown argument '{name}'")
        if not isinstance(value, _TYPES[props[name]["type"]]):
            raise ValueError(f"argument '{name}' must be of type {props[name]['type']}")


def _schema(required: list[str], **props: str) -> dict[str, Any]:
    return {"type": "object", "properties": {k: {"type": v} for k, v in props.items()},
            "required": required, "additionalProperties": False}


@dataclass
class McpGateway:
    drive: DriveConnector
    email: EmailConnector = field(default_factory=EmailConnector)
    calendar: CalendarConnector = field(default_factory=CalendarConnector)
    audit: list[AuditEntry] = field(default_factory=list)
    pending: dict[str, PendingAction] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tools: dict[str, Tool] = {t.name: t for t in [
            Tool("drive_search", "Search the firm's document drive by keywords. Use to locate a file before reading it.",
                 _schema(["query"], query="string", limit="integer"), self.drive.search),
            Tool("drive_read", "Read the full text of one drive file by file_id (from drive_search).",
                 _schema(["file_id"], file_id="string"), self.drive.read),
            Tool("email_search", "Search emails the firm has sent by keywords.",
                 _schema(["query"], query="string"), self.email.search),
            Tool("email_send", "Send an email on the lawyer's behalf. Never sends directly: creates a request a lawyer must approve.",
                 _schema(["to", "subject", "body"], to="string", subject="string", body="string"), self.email.send, True),
            Tool("calendar_list", "List calendar events for a day (YYYY-MM-DD).",
                 _schema(["day"], day="string"), self.calendar.list),
            Tool("calendar_create", "Create a calendar event. Requires lawyer approval before it is created.",
                 _schema(["title", "start"], title="string", start="string", attendees="array"), self.calendar.create, True),
        ]}

    def _audit(self, tenant_id: str, user: str, tool: str, args: dict[str, Any], outcome: str) -> None:
        digest = hashlib.sha256(compact_json(args).encode()).hexdigest()[:16]  # never log raw client content
        self.audit.append(AuditEntry(datetime.now(timezone.utc).isoformat(), tenant_id, user, tool, digest, outcome))
        log.info("mcp_tool_call", extra={"tenant": tenant_id, "tool": tool, "outcome": outcome})

    def call(self, tenant_id: str, user: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not tenant_id:
            raise PolicyError("tenant_id is required")
        tool = self.tools.get(tool_name)
        if tool is None:
            raise PolicyError(f"unknown tool '{tool_name}'")
        validate_args(tool.input_schema, args)
        if tool.requires_approval:
            action = PendingAction(f"apr_{len(self.pending) + 1:04d}", tenant_id, user, tool_name, args)
            self.pending[action.id] = action
            self._audit(tenant_id, user, tool_name, args, "pending_approval")
            return {"status": "pending_approval", "approval_id": action.id}
        try:
            result = tool.handler(tenant_id, **args)
        except KeyError as exc:
            self._audit(tenant_id, user, tool_name, args, "error")
            raise PolicyError(str(exc.args[0])) from exc
        self._audit(tenant_id, user, tool_name, args, "executed")
        return {"status": "ok", "result": result}

    def _resolve(self, tenant_id: str, approval_id: str) -> PendingAction:
        action = self.pending.get(approval_id)
        if action is None or action.tenant_id != tenant_id:  # same error either way: no cross-tenant probing
            raise PolicyError(f"approval '{approval_id}' not found")
        if action.status != "pending":
            raise PolicyError(f"approval '{approval_id}' is already {action.status}")
        return action

    def approve(self, tenant_id: str, approval_id: str, approver: str) -> dict[str, Any]:
        action = self._resolve(tenant_id, approval_id)
        result = self.tools[action.tool].handler(tenant_id, **action.args)
        action.status = "approved"
        self._audit(tenant_id, approver, action.tool, action.args, "approved")
        return {"status": "ok", "result": result}

    def deny(self, tenant_id: str, approval_id: str, approver: str) -> dict[str, Any]:
        action = self._resolve(tenant_id, approval_id)
        action.status = "denied"
        self._audit(tenant_id, approver, action.tool, action.args, "denied")
        return {"status": "denied"}


# --- MCP server (JSON-RPC 2.0 over stdio) --------------------------------------------------

SUPPORTED_PROTOCOLS = ["2025-06-18", "2025-03-26", "2024-11-05"]


class McpServer:
    """One server process is bound to one tenant and user; the client cannot choose another."""

    def __init__(self, gateway: McpGateway, tenant_id: str, user: str) -> None:
        self.gateway, self.tenant_id, self.user = gateway, tenant_id, user

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method, msg_id = message.get("method"), message.get("id")
        if msg_id is None:  # notification: no response
            return None
        try:
            result = self._dispatch(method, message.get("params") or {})
        except LookupError as exc:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": str(exc)}}
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _dispatch(self, method: str | None, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
            return {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "sib-mcp-gateway", "version": __version__}}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [{"name": t.name, "description": t.description, "inputSchema": t.input_schema,
                               "annotations": {"readOnlyHint": not t.requires_approval}}
                              for t in self.gateway.tools.values()]}
        if method == "tools/call":
            try:
                out = self.gateway.call(self.tenant_id, self.user, params.get("name", ""), params.get("arguments") or {})
            except (PolicyError, ValueError) as exc:  # tool errors go back to the model, not the protocol
                return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
            return {"content": [{"type": "text", "text": compact_json(out)}], "isError": False}
        raise LookupError(f"method not found: {method}")

    def serve(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                response: dict[str, Any] | None = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            else:
                response = self.handle(message)
            if response is not None:
                stdout.write(json.dumps(response) + "\n")
                stdout.flush()
