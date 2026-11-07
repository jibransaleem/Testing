"""
Knowledge Agent  <->  PO Agent  context contract  (schema_version 1.0)

The PO Agent calls ONE method:

    response = knowledge_agent.get_context(request)      # dict in, dict out

* The request names the project explicitly. The project is authoritative:
  nothing in the query text can redirect retrieval to another project.
* get_context() NEVER raises. Failures come back in the same envelope with
  status="error" and a machine-readable error.code.
* Every response has the same top-level keys, whatever the outcome, so the
  PO Agent can parse it without branching on shape.
* Only chunks that pass SQL provenance verification are returned.

The message formats live in po_knowledge_contract.py (shared with the PO Agent).
This module is the Knowledge Agent's implementation of that contract.
"""
from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ingest import normalize_version, version_key
from po_knowledge_contract import (  # noqa: F401  (re-exported for callers)
    CONTEXT_REQUEST_SCHEMA, CONTEXT_RESPONSE_SCHEMA, SCHEMA_VERSION,
    STATUS_OK, STATUS_NO_RESULTS, STATUS_ERROR,
    ERR_INVALID_REQUEST, ERR_UNKNOWN_PROJECT, ERR_RETRIEVAL_FAILED, ERR_INTERNAL,
    WARN_VERSION_NOT_FOUND, WARN_DROPPED_UNVERIFIED, WARN_ANSWER_FAILED, WARN_ANSWER_UNSUPPORTED,
    ContextRequest, InvalidRequest, validate_request, validate_response,
    KNOWLEDGE_CATEGORIES, KNOWLEDGE_ORIGIN, PACKAGE_TYPES, PackageRequest,
    validate_package_request, validate_package_response,
)
from provenance import format_citation, retrieve_with_provenance
from version_aware import resolve_version_documents

NOT_FOUND = "Not found in project knowledge."


# ============================================================
# Response helpers
# ============================================================

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _envelope(request_id: str, query: str, status: str, *, project=None, versions=None,
              passages=None, answer=None, warnings=None, error=None, context=None,
              top_k=None, min_score=None, started: float = 0.0) -> Dict[str, Any]:
    """Single place that builds the response, so the shape can never vary."""
    passages = passages or []
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "status": status,
        "project": project,                       # {"project_id", "name"} | None
        "query": query,
        "versions": versions or {
            "requested": [], "available": [], "latest": None,
            "returned": [], "include_history": False,
        },
        "passages": passages,
        "answer": answer,                         # None unless generate_answer=True
        "warnings": warnings or [],
        "error": error,                           # None unless status == "error"
        "context": context or {},                 # echoed from the request
        "meta": {
            "top_k": top_k,
            "min_score": min_score,
            "returned": len(passages),
            "latency_ms": int((time.perf_counter() - started) * 1000) if started else 0,
            "generated_at": _now(),
        },
    }


def _error(request_id, query, code, message, details=None, **kw) -> Dict[str, Any]:
    return _envelope(request_id, query, STATUS_ERROR,
                     error={"code": code, "message": message, "details": details or {}}, **kw)


def error_response(request_id: str, query: str, code: str, message: str,
                   details: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Contract-shaped error envelope (used by the transport layer for failures before get_context runs)."""
    return _error(request_id, query, code, message, details, context=context)


def _list_versions(conn, project_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, version, status, title FROM documents "
        "WHERE project_id = %s AND doc_type != 'decision'", (project_id,)).fetchall()
    rows.sort(key=lambda r: version_key(r["version"]))
    return [{"document_id": r["id"], "version": normalize_version(r["version"]),
             "status": r["status"] or "current", "title": r["title"]} for r in rows]


def _latest(available: List[Dict[str, Any]]) -> Optional[str]:
    current = [a for a in available if a["status"] == "current"]
    pool = current or available
    return pool[-1]["version"] if pool else None


def _to_passage(rank: int, hit: Dict[str, Any]) -> Dict[str, Any]:
    s = hit["source"]
    version = normalize_version(s["document_version"]) if s.get("document_version") else None
    return {
        "passage_id": hit["chunk_id"],
        "rank": rank,
        "text": hit["text"],
        "score": round(float(hit["score"]), 4) if hit.get("score") is not None else None,
        "is_current": s["approval_status"] == "approved",
        "version": version,
        "citation": format_citation(hit),
        "source": {
            "project_id": s["project_id"],
            "project_name": s["project"],
            "document_id": s["document_id"],        # None for decisions, stories, etc.
            "document_title": s["document"],
            "document_version": version,
            "document_status": s["document_status"],
            "doc_type": s["doc_type"],
            "section": s["section"],
            "page_start": s["page_start"],
            "page_end": s["page_end"],
            "approval_status": s["approval_status"],
            "decision_id": s["decision_id"],
            "source_event": s["source_event"],
            "source_id": s["source_id"],
        },
    }


# ============================================================
# Entry point
# ============================================================

def get_context(agent, request: Any) -> Dict[str, Any]:
    """Validate -> resolve project -> version-aware retrieval -> provenance check -> envelope."""
    started = time.perf_counter()
    raw_ctx = request.get("context") if isinstance(request, dict) and isinstance(request.get("context"), dict) else {}
    request_id = str(raw_ctx.get("request_id") or f"req_{uuid.uuid4().hex[:12]}")
    raw_query = request.get("query") if isinstance(request, dict) and isinstance(request.get("query"), str) else ""

    try:
        req = ContextRequest.from_dict(request)
    except InvalidRequest as e:
        return _error(request_id, raw_query, ERR_INVALID_REQUEST, str(e),
                      context=raw_ctx, started=started)

    ctx = dict(req.context)
    ctx["request_id"] = request_id
    min_score = req.min_score if req.min_score is not None else agent.min_score
    common = dict(context=ctx, top_k=req.top_k, min_score=min_score, started=started)

    try:
        row = agent.conn.execute("SELECT id, name FROM projects WHERE id = %s", (req.project_id,)).fetchone()
        if not row:
            known = [r["id"] for r in agent.conn.execute("SELECT id FROM projects ORDER BY id").fetchall()]
            return _error(request_id, req.query, ERR_UNKNOWN_PROJECT,
                          f"Unknown project_id: {req.project_id}",
                          details={"available_project_ids": known}, **common)
        project = {"project_id": row["id"], "name": row["name"]}

        available = _list_versions(agent.conn, req.project_id)
        versions_info = {
            "requested": req.versions, "available": available, "latest": _latest(available),
            "returned": [], "include_history": bool(req.include_history or req.versions),
        }
        warnings: List[Dict[str, str]] = []

        if req.versions and not resolve_version_documents(agent.conn, req.project_id, req.versions):
            warnings.append({"code": WARN_VERSION_NOT_FOUND, "message":
                             f"No document found for version(s) {req.versions}. "
                             f"Available: {[a['version'] for a in available]}."})
            return _envelope(request_id, req.query, STATUS_NO_RESULTS, project=project,
                             versions=versions_info, warnings=warnings, **common)

        try:
            hits = retrieve_with_provenance(
                agent.conn, req.project_id, req.query, top_k=req.top_k,
                include_history=req.include_history, target_versions=req.versions or None,
                min_score=min_score, doc_types=req.doc_types)
        except Exception as e:  # noqa: BLE001 - agent boundary
            return _error(request_id, req.query, ERR_RETRIEVAL_FAILED,
                          f"{type(e).__name__}: {e}", project=project, versions=versions_info, **common)

        valid = [h for h in hits if h.get("provenance_valid")]
        dropped = [h["chunk_id"] for h in hits if not h.get("provenance_valid")]
        if dropped:
            warnings.append({"code": WARN_DROPPED_UNVERIFIED, "message":
                             f"{len(dropped)} chunk(s) failed provenance verification and were removed: {dropped}. "
                             f"Run reconcile() on the Knowledge Agent."})

        passages = [_to_passage(i, h) for i, h in enumerate(valid, 1)]
        versions_info["returned"] = sorted({p["version"] for p in passages if p["version"]}, key=version_key)

        answer = None
        if req.generate_answer and valid:
            answer, warn = _build_answer(agent, req.query, valid)
            if warn:
                warnings.append(warn)

        return _envelope(request_id, req.query, STATUS_OK if passages else STATUS_NO_RESULTS,
                         project=project, versions=versions_info, passages=passages,
                         answer=answer, warnings=warnings, **common)

    except Exception as e:  # noqa: BLE001 - never raise to the PO Agent
        return _error(request_id, req.query, ERR_INTERNAL, f"{type(e).__name__}: {e}", **common)


def _build_answer(agent, query: str, valid: List[Dict[str, Any]]):
    """Optional synthesized answer, grounded only in the verified passages returned."""
    try:
        if agent.llm_client:
            text, mode = agent._synthesize_with_llm(query, valid), "llm"
        else:
            text, mode = agent._synthesize_extractive(valid), "extractive"
    except Exception as e:  # noqa: BLE001
        return None, {"code": WARN_ANSWER_FAILED, "message": f"{type(e).__name__}: {e}"}

    if mode == "llm" and text.strip().rstrip(".").lower() == NOT_FOUND.rstrip(".").lower():
        return ({"text": NOT_FOUND, "mode": mode, "supported": False, "citations": []},
                {"code": WARN_ANSWER_UNSUPPORTED,
                 "message": "Passages were retrieved but do not explicitly answer the query."})
    return {"text": text, "mode": mode, "supported": True,
            "citations": [format_citation(h) for h in valid]}, None


# ============================================================
# PACKAGE interface: knowledge half of the PO Agent's UC1 / UC2 context packages
# ============================================================
_PKG_DEFAULT_QUERY = "project requirements, constraints, policies and decisions"
_PKG_DOC_TYPES = ["requirement", "decision"]
_PKG_MAX_TEXT = 800

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u00ad\u200b\ufeff]")
_SPLIT_WORD = re.compile(r"([A-Za-z]{2,})- (?=([a-z]+))")
_KEEP_HYPHEN = {"and", "or", "to", "of", "the", "for", "in", "on", "with", "by", "a", "an"}


def clean_text(text: str, max_chars: int = _PKG_MAX_TEXT) -> str:
    """PDF extraction hygiene for package text: drop control characters (e.g. a \\x88 bullet),
    rejoin 'de- signed' -> 'designed', collapse whitespace, cap the length on a word boundary."""
    t = _CONTROL.sub("", text or "")
    t = _SPLIT_WORD.sub(lambda m: m.group(0) if m.group(2) in _KEEP_HYPHEN else m.group(1), t)
    t = re.sub(r"\s+", " ", t).strip(" \t-•*")
    return t if len(t) <= max_chars else t[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _knowledge(requirements: List[Dict[str, str]], reason_if_empty: str) -> Dict[str, Any]:
    """Knowledge block. Goal / vision / themes are not stored by the Knowledge Agent: reported missing, never invented."""
    reasons = {
        "PRODUCT_GOAL": "the Knowledge Agent holds no Product Goal for this project",
        "PRODUCT_VISION": "the Knowledge Agent holds no Product Vision for this project",
        "STRATEGIC_THEMES": "the Knowledge Agent holds no strategic themes for this project",
        "REQUIREMENTS_DECISIONS": reason_if_empty,
    }
    available = ["REQUIREMENTS_DECISIONS"] if requirements else []
    return {
        "origin": KNOWLEDGE_ORIGIN, "product_goal": None, "product_vision": None, "strategic_themes": [],
        "requirements": requirements, "project_requirements": [r["text"] for r in requirements],
        "available_context": available,
        "missing_context": [{"category": c, "reason": reasons[c]} for c in KNOWLEDGE_CATEGORIES if c not in available],
    }


def _package_envelope(request_id, package_type, status, *, project=None, knowledge=None, warnings=None,
                      error=None, context=None, top_k=None, started=0.0) -> Dict[str, Any]:
    k = knowledge or _knowledge([], "not assessed")
    return {
        "schema_version": SCHEMA_VERSION, "request_id": request_id, "status": status,
        "package_type": package_type, "project": project, "knowledge": k,
        "warnings": warnings or [], "error": error, "context": context or {},
        "meta": {"top_k": top_k, "returned": len(k["requirements"]),
                 "latency_ms": int((time.perf_counter() - started) * 1000) if started else 0,
                 "generated_at": _now()},
    }


def _package_error(request_id, package_type, code, message, details=None, **kw) -> Dict[str, Any]:
    return _package_envelope(
        request_id, package_type, STATUS_ERROR,
        knowledge=_knowledge([], f"not assessed: {code}"),
        error={"code": code, "message": message, "details": details or {}}, **kw)


def get_package(agent, request: Any) -> Dict[str, Any]:
    """Validate -> verified retrieval of CURRENT requirements/decisions -> PO-model-shaped knowledge block.
    Never raises. Uses get_context() underneath, so project isolation, version filtering (current only)
    and SQL provenance verification all apply."""
    started = time.perf_counter()
    raw_ctx = request.get("context") if isinstance(request, dict) and isinstance(request.get("context"), dict) else {}
    request_id = str(raw_ctx.get("request_id") or f"req_{uuid.uuid4().hex[:12]}")
    raw_type = request.get("package_type") if isinstance(request, dict) and request.get("package_type") in PACKAGE_TYPES else None
    try:
        req = PackageRequest.from_dict(request)
    except InvalidRequest as e:
        return _package_error(request_id, raw_type, ERR_INVALID_REQUEST, str(e), context=raw_ctx, started=started)

    ctx = {**req.context, "request_id": request_id}
    try:
        inner = get_context(agent, {
            "project_id": req.project_id, "query": req.focus_text or _PKG_DEFAULT_QUERY, "top_k": req.top_k,
            "doc_types": _PKG_DOC_TYPES, "context": {"request_id": request_id, "purpose": req.package_type}})
        if inner["status"] == STATUS_ERROR:
            return _package_error(request_id, req.package_type, inner["error"]["code"], inner["error"]["message"],
                                  inner["error"]["details"], context=ctx, top_k=req.top_k, started=started)

        requirements, seen = [], set()
        for p in inner["passages"]:
            text = clean_text(p["text"])
            key = re.sub(r"\W+", " ", text.lower()).strip()
            if not text or key in seen:
                continue
            seen.add(key)
            requirements.append({
                "requirement_id": p["passage_id"],
                "kind": "DECISION" if p["source"]["doc_type"] == "decision" else "REQUIREMENT",
                "text": text, "source": p["citation"]})

        reason = ("no requirement or decision matched the focus text" if req.focus_text
                  else "no requirements or decisions are stored for this project")
        return _package_envelope(
            request_id, req.package_type, STATUS_OK if requirements else STATUS_NO_RESULTS,
            project=inner["project"], knowledge=_knowledge(requirements, reason), warnings=inner["warnings"],
            context=ctx, top_k=req.top_k, started=started)
    except Exception as e:  # noqa: BLE001 - never raise to the PO Agent
        return _package_error(request_id, req.package_type, ERR_INTERNAL, f"{type(e).__name__}: {e}",
                              context=ctx, top_k=req.top_k, started=started)