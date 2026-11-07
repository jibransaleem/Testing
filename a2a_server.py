"""
a2a_server.py  --  Knowledge Agent as an A2A (Agent2Agent) server

Deploy on Vercel (entrypoint: this file, exposes `app`) or run locally:
    uvicorn a2a_server:app --port 8000

Endpoints
    GET  /.well-known/agent-card.json   public Agent Card (discovery; /.well-known/agent.json also works)
    POST /a2a                           JSON-RPC 2.0:  message/send | tasks/get
    GET  /ui                            log viewer page (enter the API key in the page)
    GET  /logs                          recent requests + responses as JSON (needs the key)
    GET  /health                        liveness (add ?deep=1 for DB/vector status)

Skills (selected by a DATA part in the message):
    {"skill": "ping"}
    {"skill": "get_context", "request": {...ContextRequest...}}
    {"skill": "get_package", "request": {...PackageRequest...}}
    {"skill": "put_item",    "request": {...ItemRequest...}}
A plain TEXT part also works: it is treated as get_context; give project_id in message.metadata.

Env
    A2A_API_KEY   if set, POST /a2a needs  Authorization: Bearer <key>
    UPSTASH_REDIS_REST_URL / _TOKEN   shared log store so /ui works on Vercel
    PUBLIC_URL    public base URL shown in the Agent Card
    (no ingestion here: documents are already in the database)
    + everything the KB already uses (DATABASE_URL, JINA_API_KEY, GROQ_API_KEY, ...)
"""
from __future__ import annotations

import hmac
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from collections import deque

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from a2a_ui import UI_HTML

PROTOCOL_VERSION = "0.3.0"
AGENT_VERSION = "1.0.0"
API_KEY = os.getenv("A2A_API_KEY")

SKILLS = {
    "ping": "Check that the Knowledge Agent is reachable.",
    "get_context": "Ranked evidence passages with citations for a project question.",
    "get_package": "Knowledge half of the PO Agent's UC1 / UC2 context packages.",
    "put_item": "Store / version / retire an approved item (story, meeting summary, action item, decision...).",
}

# ----------------------------------------------------------------------------- request log (shown in /ui)
REDIS_URL = os.getenv("UPSTASH_REDIS_REST_URL") or os.getenv("KV_REST_API_URL")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN") or os.getenv("KV_REST_API_TOKEN")
LOG_SIZE = int(os.getenv("LOG_SIZE", "200"))
LOG_KEY = os.getenv("LOG_KEY", "kb:a2a:logs")
MAX_ENTRY_CHARS = 200_000
LOGS: deque = deque(maxlen=LOG_SIZE)


def _redis(commands):
    r = httpx.post(REDIS_URL.rstrip("/") + "/pipeline", json=commands, timeout=4,
                   headers={"Authorization": f"Bearer {REDIS_TOKEN}"})
    r.raise_for_status()
    return [x.get("result") for x in r.json()]


def _log(http_req: Request, **entry) -> None:
    entry["id"] = uuid.uuid4().hex[:8]
    entry["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC"
    fwd = http_req.headers.get("x-forwarded-for")
    entry["client"] = (fwd.split(",")[0].strip() if fwd else (http_req.client.host if http_req.client else None))
    LOGS.appendleft(entry)
    if REDIS_URL and REDIS_TOKEN:
        try:
            text = json.dumps(entry, default=str)
            if len(text) > MAX_ENTRY_CHARS:
                entry = {**entry, "response": {"truncated": True}, "request": {"truncated": True}}
                text = json.dumps(entry, default=str)
            _redis([["LPUSH", LOG_KEY, text], ["LTRIM", LOG_KEY, 0, LOG_SIZE - 1]])
        except Exception as e:
            print(f"log store write failed: {type(e).__name__}: {e}")


def _read_logs():
    if REDIS_URL and REDIS_TOKEN:
        try:
            raw = _redis([["LRANGE", LOG_KEY, 0, LOG_SIZE - 1]])[0] or []
            return [json.loads(x) for x in raw], "redis"
        except Exception as e:
            print(f"log store read failed: {type(e).__name__}: {e}")
    return list(LOGS), "memory"


app = FastAPI(title="Knowledge Agent (A2A)", version=AGENT_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Authorization", "Content-Type"])

# ----------------------------------------------------------------------------- lazy KB initialization
_iface = None
_lock = threading.Lock()
_run_lock = threading.Lock()

def get_interface():
    global _iface
    with _lock:
        if _iface is None:
            from agents import KnowledgeAgent, build_llm_client
            from db import get_connection
            from knowledge_interface import KnowledgeInterface
            from resilient_conn import ResilientConnection      # FIX: self-healing connection

            # FIX: pass the connection FACTORY (no brackets) so a dropped Neon connection is re-opened
            # automatically instead of failing forever with "the connection is closed".
            conn = ResilientConnection(get_connection)
            _iface = KnowledgeInterface(KnowledgeAgent(
                conn, build_llm_client()))
        return _iface


# ----------------------------------------------------------------------------- JSON-RPC helpers
def rpc_ok(rid, result) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": result})


def rpc_err(rid, code: int, message: str, data: Any = None, http: int = 200) -> JSONResponse:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": err}, status_code=http)


def _authorized(request: Request) -> bool:
    if not API_KEY:
        return True
    got = request.headers.get("authorization", "")
    return got.startswith("Bearer ") and hmac.compare_digest(got[7:], API_KEY)


# ----------------------------------------------------------------------------- Agent Card
def agent_card(base_url: str) -> Dict[str, Any]:
    base = (os.getenv("PUBLIC_URL") or base_url).rstrip("/")
    card = {
        "protocolVersion": PROTOCOL_VERSION,
        "name": "Knowledge Base Agent",
        "description": "Project knowledge base (SRS documents, decisions, PO items). Answers with cited evidence "
                       "and stores approved items sent by the PO Agent.",
        "url": f"{base}/a2a",
        "preferredTransport": "JSONRPC",
        "version": AGENT_VERSION,
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": False},
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json"],
        "skills": [{"id": k, "name": k, "description": v, "tags": ["knowledge", "po-agent"]}
                   for k, v in SKILLS.items()],
    }
    if API_KEY:
        card["securitySchemes"] = {"bearer": {"type": "http", "scheme": "bearer"}}
        card["security"] = [{"bearer": []}]
    return card


@app.get("/.well-known/agent-card.json")
@app.get("/.well-known/agent.json")
async def well_known(request: Request):
    return agent_card(str(request.base_url))


@app.get("/health")
async def health(deep: int = 0):
    out: Dict[str, Any] = {"status": "ok", "agent": "knowledge", "time": time.time()}
    if deep:
        try:
            with _run_lock:
                out["kb"] = get_interface().agent.status()
        except Exception as e:
            return JSONResponse({**out, "status": "error", "message": f"{type(e).__name__}: {e}"}, 503)
    return out


# ----------------------------------------------------------------------------- message -> skill call
def parse_message(message: Dict[str, Any]):
    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("message.parts must be a non-empty list")
    meta = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}

    for p in parts:
        if isinstance(p, dict) and isinstance(p.get("data"), dict):
            d = dict(p["data"])
            skill = d.pop("skill", None) or d.pop("action", None) or meta.get("skill")
            req = d["request"] if isinstance(d.get("request"), dict) else d
            if not skill:
                raise ValueError("data part needs a 'skill' (ping | get_context | get_package | put_item)")
            return str(skill), req

    text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")).strip()
    if not text:
        raise ValueError("message has no data part and no text part")
    if text.lower() == "ping":
        return "ping", {}
    req = {"query": text, **{k: v for k, v in meta.items() if k in ("project_id", "top_k")}}
    return meta.get("skill", "get_context"), req


def run_skill(skill: str, req: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    if skill == "ping":
        return {"status": "ok", "agent": "knowledge", "skills": list(SKILLS), "time": time.time()}
    if skill not in SKILLS:
        raise KeyError(skill)
    ctx = req.get("context") if isinstance(req.get("context"), dict) else {}
    req = {**req, "context": {**ctx, "request_id": ctx.get("request_id") or request_id}}
    iface = get_interface()
    with _run_lock:
        return {"get_context": iface.get_context, "get_package": iface.get_package,
                "put_item": iface.put_item}[skill](req)


def build_task(skill: str, result: Dict[str, Any], message: Dict[str, Any], started: float) -> Dict[str, Any]:
    failed = isinstance(result, dict) and result.get("status") == "error"
    task_id = uuid.uuid4().hex
    return {
        "kind": "task",
        "id": task_id,
        "contextId": message.get("contextId") or uuid.uuid4().hex,
        "status": {
            "state": "failed" if failed else "completed",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"kind": "message", "role": "agent", "messageId": uuid.uuid4().hex,
                        "parts": [{"kind": "text", "text": (result.get("error") or {}).get("message", "error")
                                   if failed else f"{skill} ok"}]},
        },
        "artifacts": [{
            "artifactId": uuid.uuid4().hex,
            "name": f"{skill}_result",
            "parts": [{"kind": "data", "data": result}],
        }],
        "metadata": {"skill": skill, "latency_ms": int((time.perf_counter() - started) * 1000)},
    }


# ----------------------------------------------------------------------------- log viewer
@app.get("/ui")
async def ui():
    return HTMLResponse(UI_HTML)


@app.get("/logs")
async def logs(request: Request):
    if not _authorized(request):
        return rpc_err(None, -32001, "Unauthorized: missing or wrong bearer token", http=401)
    entries, storage = _read_logs()
    return {"count": len(entries), "storage": storage, "logs": entries}


# ----------------------------------------------------------------------------- JSON-RPC endpoint
@app.post("/a2a")
async def a2a(request: Request):
    if not _authorized(request):
        _log(request, method=None, skill=None, outcome="unauthorized", http=401,
             error="missing or wrong bearer token")
        return rpc_err(None, -32001, "Unauthorized: missing or wrong bearer token", http=401)
    try:
        body = await request.json()
    except Exception:
        _log(request, method=None, skill=None, outcome="parse_error", http=400, error="body is not valid JSON")
        return rpc_err(None, -32700, "Parse error: body is not valid JSON", http=400)
    if not isinstance(body, dict) or body.get("jsonrpc") != "2.0" or "method" not in body:
        _log(request, method=None, skill=None, outcome="invalid_request", http=400, request_body=body,
             error="not a JSON-RPC 2.0 object with 'method'")
        return rpc_err(body.get("id") if isinstance(body, dict) else None, -32600,
                       "Invalid Request: expected a JSON-RPC 2.0 object with 'method'", http=400)

    rid, method = body.get("id"), body["method"]
    params = body.get("params") if isinstance(body.get("params"), dict) else {}

    if method in ("message/send", "message/stream"):
        if method == "message/stream":
            _log(request, method=method, skill=None, outcome="unsupported", error="streaming not supported")
            return rpc_err(rid, -32004, "Streaming is not supported; use message/send")
        message = params.get("message")
        if not isinstance(message, dict):
            _log(request, method=method, skill=None, outcome="invalid_params", error="'message' object required")
            return rpc_err(rid, -32602, "Invalid params: 'message' object required")
        started = time.perf_counter()
        skill, req = None, None
        try:
            skill, req = parse_message(message)
            result = run_skill(skill, req, str(message.get("messageId") or uuid.uuid4().hex))
        except KeyError as e:
            _log(request, method=method, skill=skill, request=req, outcome="unknown_skill", error=f"Unknown skill {e}")
            return rpc_err(rid, -32602, f"Unknown skill {e}", {"skills": list(SKILLS)})
        except ValueError as e:
            _log(request, method=method, skill=skill, request=req, outcome="invalid_params", error=str(e))
            return rpc_err(rid, -32602, f"Invalid params: {e}")
        except Exception as e:
            _log(request, method=method, skill=skill, request=req, outcome="internal_error",
                 latency_ms=int((time.perf_counter() - started) * 1000), error=f"{type(e).__name__}: {e}")
            return rpc_err(rid, -32603, f"Internal error: {type(e).__name__}: {e}")
        task = build_task(skill, result, message, started)
        _log(request, method=method, skill=skill, request=req, response=result,
             outcome=result.get("status", "ok") if isinstance(result, dict) else "ok",
             latency_ms=task["metadata"]["latency_ms"],
             error=(result.get("error") or {}).get("message") if isinstance(result, dict) else None)
        return rpc_ok(rid, task)

    _log(request, method=method, skill=None, outcome="unsupported", error=f"method {method} not supported")
    if method == "tasks/get":
        return rpc_err(rid, -32001, "Task not found: tasks are completed synchronously and not stored")
    if method == "tasks/cancel":
        return rpc_err(rid, -32002, "Task cannot be canceled: it already completed")

    return rpc_err(rid, -32601, f"Method not found: {method}")