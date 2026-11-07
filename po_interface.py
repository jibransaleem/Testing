"""
po_interface.py  --  PO AGENT side: asks the Knowledge Agent for context

Put this file in the PO Agent project (with po_knowledge_contract.py). It imports no Knowledge Agent
code: it only needs an object with `.get_package(dict)` and `.handle(dict)` (knowledge_interface.KnowledgeInterface).

    connector = KnowledgeConnector(knowledge_interface, project_map={"proj-agiliro-001": "proj_project1"})

WRITE (store approved project knowledge)
    connector.put_item(pid, "user_story", "PBI-12", text, title="Reset password", approved=True)
    store_pbi(connector, pid, committed_pbi)             # formats a committed PBI/proposal and stores it
    connector.retire_item(pid, "user_story", "PBI-12")   # no longer in force

UC1  (ContextPackageContract)
    ctx = enrich_context(ctx, connector)          # fills project_requirements from the Knowledge Agent

UC2  (AlignmentContextPackage)
    source = strategic_source(connector, focus_text=f"{pbi.title}. {pbi.user_story_narrative}")
    pkg = await build_alignment_context(pid, key, strategic_source=source)

Raw access
    pkg  = connector.get_package(pid, "UC2_ALIGNMENT", focus_text="...")   # PackageResponse (typed)
    resp = connector.ask(pid, "login requirements")                        # ranked passages with citations

What the Knowledge Agent supplies vs what stays with the PO Agent
    Knowledge Agent: product_goal, product_vision, strategic_themes, requirements/decisions
                     (names and shapes identical to the PO models, so ProjectRequirement(**r) just works).
    PO Agent:        raw_source_content, source_version_id, existing_backlog_titles, team_constraints,
                     target_pbi, related_pbis, evidence, fingerprint.
Nothing here raises: failures come back as status="error" (details["origin"] == "po_interface" when found
on the PO side), and the adapters then degrade to "no extra context".
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from po_knowledge_contract import (
    ERR_INTERNAL, ERR_INVALID_REQUEST, KNOWLEDGE_CATEGORIES, KNOWLEDGE_ORIGIN, PKG_UC1, PKG_UC2, SCHEMA_VERSION,
    STATUS_ERROR, ContextRequest, ContextResponse, ContractError, InvalidRequest, ItemRequest, ItemResponse,
    OP_RETIRE, OP_UPSERT, PackageRequest, PackageResponse,
)

_NOW = lambda: time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())  # noqa: E731


def _error_context(rid, query, code, message, reason, ctx):
    return ContextResponse.from_dict({
        "schema_version": SCHEMA_VERSION, "request_id": rid, "status": STATUS_ERROR, "project": None,
        "query": query, "versions": {"requested": [], "available": [], "latest": None, "returned": [],
                                     "include_history": False},
        "passages": [], "answer": None, "warnings": [],
        "error": {"code": code, "message": message, "details": {"origin": "po_interface", "reason": reason}},
        "context": {**(ctx or {}), "request_id": rid},
        "meta": {"top_k": None, "min_score": None, "returned": 0, "latency_ms": 0, "generated_at": _NOW()}})


def _error_package(rid, ptype, code, message, reason, ctx):
    reasons = {c: f"not assessed: {message}" for c in KNOWLEDGE_CATEGORIES}
    return PackageResponse.from_dict({
        "schema_version": SCHEMA_VERSION, "request_id": rid, "status": STATUS_ERROR, "package_type": ptype,
        "project": None,
        "knowledge": {"origin": KNOWLEDGE_ORIGIN, "product_goal": None, "product_vision": None,
                      "strategic_themes": [], "requirements": [], "project_requirements": [],
                      "available_context": [],
                      "missing_context": [{"category": c, "reason": reasons[c]} for c in KNOWLEDGE_CATEGORIES]},
        "warnings": [],
        "error": {"code": code, "message": message, "details": {"origin": "po_interface", "reason": reason}},
        "context": {**(ctx or {}), "request_id": rid},
        "meta": {"top_k": None, "returned": 0, "latency_ms": 0, "generated_at": _NOW()}})


def _error_item(rid, code, message, reason, ctx):
    return ItemResponse.from_dict({
        "schema_version": SCHEMA_VERSION, "request_id": rid, "status": STATUS_ERROR, "project": None, "item": None,
        "warnings": [], "error": {"code": code, "message": message,
                                  "details": {"origin": "po_interface", "reason": reason}},
        "context": {**(ctx or {}), "request_id": rid}, "meta": {"latency_ms": 0, "generated_at": _NOW()}})


class KnowledgeConnector:
    def __init__(self, knowledge_interface, project_map: Optional[Mapping[str, str]] = None,
                 requester: str = "po_agent"):
        self.interface = knowledge_interface
        self.project_map = dict(project_map or {})      # PO project id -> Knowledge Agent project id
        self.requester = requester

    def _ids(self, project_id: str, purpose: Optional[str], request_id: Optional[str]):
        rid = request_id or f"po_{uuid.uuid4().hex[:12]}"
        ka = self.project_map.get(project_id, project_id)
        ctx: Dict[str, Any] = {"request_id": rid, "requester": self.requester}
        if purpose:
            ctx["purpose"] = purpose
        if ka != project_id:
            ctx["po_project_id"] = project_id
        return rid, ka, ctx

    # ------------------------------------------------------------------ packages (UC1 / UC2)
    def get_package(self, project_id: str, package_type: str, *, focus_text: Optional[str] = None,
                    top_k: int = 20, request_id: Optional[str] = None) -> PackageResponse:
        """Knowledge half of the PO context package. package_type: 'UC1_CONTEXT' | 'UC2_ALIGNMENT'."""
        rid, ka, ctx = self._ids(project_id, None, request_id)
        try:
            req = PackageRequest.from_dict({"project_id": ka, "package_type": package_type,
                                            "focus_text": focus_text, "top_k": top_k, "context": ctx})
        except InvalidRequest as e:
            return _error_package(rid, package_type if package_type in (PKG_UC1, PKG_UC2) else None,
                                  ERR_INVALID_REQUEST, str(e), "invalid_request_not_sent", ctx)
        try:
            raw = self.interface.get_package(req.to_dict())
        except Exception as e:  # noqa: BLE001
            return _error_package(rid, package_type, ERR_INTERNAL,
                                  f"Knowledge Agent call failed: {type(e).__name__}: {e}", "call_failed", ctx)
        try:
            pkg = PackageResponse.from_dict(raw)
        except ContractError as e:
            return _error_package(rid, package_type, ERR_INTERNAL, f"Malformed response: {e}",
                                  "malformed_response", ctx)
        bad = None
        if pkg.request_id != rid:
            bad = f"request_id mismatch ({pkg.request_id!r})"
        elif pkg.package_type not in (package_type, None):
            bad = f"package_type mismatch ({pkg.package_type!r})"
        elif pkg.project and pkg.project.project_id != ka:
            bad = f"response is for project {pkg.project.project_id!r}, asked for {ka!r}"
        if bad:
            return _error_package(rid, package_type, ERR_INTERNAL, f"Response rejected: {bad}",
                                  "response_rejected", ctx)
        return pkg

    # ------------------------------------------------------------------ WRITE: store approved items
    def put_item(self, project_id: str, item_type: str, item_id: str, text: Optional[str] = None, *,
                 approved: bool, title: Optional[str] = None, operation: str = OP_UPSERT,
                 request_id: Optional[str] = None) -> ItemResponse:
        """Send an APPROVED item for the Knowledge Agent to store. `approved` must be True for upsert
        (never send drafts or proposals). Same item_id again: unchanged / new version. Never raises."""
        rid, ka, ctx = self._ids(project_id, None, request_id)
        try:
            req = ItemRequest.from_dict({"project_id": ka, "item_type": item_type, "item_id": item_id,
                                         "operation": operation, "text": text, "title": title,
                                         "approved": approved, "context": ctx})
        except InvalidRequest as e:
            return _error_item(rid, ERR_INVALID_REQUEST, str(e), "invalid_request_not_sent", ctx)
        try:
            raw = self.interface.put_item(req.to_dict())
        except Exception as e:  # noqa: BLE001
            return _error_item(rid, ERR_INTERNAL, f"Knowledge Agent call failed: {type(e).__name__}: {e}",
                               "call_failed", ctx)
        try:
            resp = ItemResponse.from_dict(raw)
        except ContractError as e:
            return _error_item(rid, ERR_INTERNAL, f"Malformed response: {e}", "malformed_response", ctx)
        bad = None
        if resp.request_id != rid:
            bad = f"request_id mismatch ({resp.request_id!r})"
        elif resp.project and resp.project.project_id != ka:
            bad = f"response is for project {resp.project.project_id!r}, asked for {ka!r}"
        elif resp.item and (resp.item.item_type, resp.item.item_id) != (item_type, item_id):
            bad = f"response is for item {resp.item.item_type}/{resp.item.item_id}"
        if bad:
            return _error_item(rid, ERR_INTERNAL, f"Response rejected: {bad}", "response_rejected", ctx)
        return resp

    def retire_item(self, project_id: str, item_type: str, item_id: str, *,
                    request_id: Optional[str] = None) -> ItemResponse:
        """The item is no longer in force: it stops appearing in normal answers (kept as history)."""
        return self.put_item(project_id, item_type, item_id, approved=False, operation=OP_RETIRE,
                             request_id=request_id)

    # ------------------------------------------------------------------ evidence passages
    def ask(self, project_id: str, query: str, *, top_k: int = 5, versions: Optional[Sequence[str]] = None,
            include_history: bool = False, doc_types: Optional[Sequence[str]] = None,
            min_score: Optional[float] = None, generate_answer: bool = False,
            purpose: Optional[str] = None, request_id: Optional[str] = None) -> ContextResponse:
        """Ranked source passages with citations (see ContextResponse)."""
        rid, ka, ctx = self._ids(project_id, purpose, request_id)
        try:
            req = ContextRequest.from_dict({
                "project_id": ka, "query": query, "top_k": top_k, "versions": list(versions or []),
                "include_history": include_history, "doc_types": list(doc_types) if doc_types else None,
                "min_score": min_score, "generate_answer": generate_answer, "context": ctx})
        except InvalidRequest as e:
            return _error_context(rid, query or "", ERR_INVALID_REQUEST, str(e), "invalid_request_not_sent", ctx)
        try:
            raw = self.interface.handle(req.to_dict())
        except Exception as e:  # noqa: BLE001
            return _error_context(rid, req.query, ERR_INTERNAL,
                                  f"Knowledge Agent call failed: {type(e).__name__}: {e}", "call_failed", ctx)
        try:
            resp = ContextResponse.from_dict(raw)
        except ContractError as e:
            return _error_context(rid, req.query, ERR_INTERNAL, f"Malformed response: {e}", "malformed_response", ctx)
        bad = None
        if resp.request_id != rid:
            bad = f"request_id mismatch ({resp.request_id!r})"
        elif resp.project and resp.project.project_id != ka:
            bad = f"response is for project {resp.project.project_id!r}, asked for {ka!r}"
        else:
            bad = next((f"passage {p.passage_id!r} is from project {p.source.project_id!r}"
                        for p in resp.passages if p.source.project_id != ka), None)
        if bad:
            return _error_context(rid, req.query, ERR_INTERNAL, f"Response rejected: {bad}", "response_rejected", ctx)
        return resp


# ----------------------------------------------------------------------------------------------------------
# UC1: ContextPackageContract
# ----------------------------------------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def enrich_context(ctx, connector: KnowledgeConnector, *, top_k: int = 20, max_query_chars: int = 1500):
    """Copy of a UC1 ContextPackageContract (model or dict) with the Knowledge Agent's current
    requirements/decisions appended to `project_requirements` (existing entries first, untouched).
    `product_goal` is filled only if the package's own goal is blank AND the Knowledge Agent holds one.
    Never raises; on any problem the original package is returned unchanged."""
    try:
        is_dict = isinstance(ctx, dict)
        get = (lambda k: ctx.get(k)) if is_dict else (lambda k: getattr(ctx, k))
        focus = (get("raw_source_content") or "")[:max_query_chars].strip() or None
        pkg = connector.get_package(get("project_id"), PKG_UC1, focus_text=focus, top_k=top_k)
        if pkg.status == STATUS_ERROR:
            return ctx
        existing = list(get("project_requirements") or [])
        have = {_norm(s) for s in existing}
        update: Dict[str, Any] = {}
        extra = [t for t in pkg.knowledge.project_requirements if _norm(t) not in have]
        if extra:
            update["project_requirements"] = existing + extra
        if pkg.knowledge.product_goal and not (get("product_goal") or "").strip():
            update["product_goal"] = pkg.knowledge.product_goal.text
        if not update:
            return ctx
        return {**ctx, **update} if is_dict else ctx.model_copy(update=update)
    except Exception:  # noqa: BLE001 - must never break the PO graph
        return ctx


# ----------------------------------------------------------------------------------------------------------
# UC2: strategic_source for uc2_context.build_alignment_context
# ----------------------------------------------------------------------------------------------------------
def strategic_source(connector: KnowledgeConnector, *, focus_text: Optional[str] = None, top_k: int = 20
                     ) -> Callable[[str], Any]:
    """A `strategic_source` (project_id -> StrategicContext) backed by the Knowledge Agent. Pass the audited
    PBI's title + story as `focus_text` so the most relevant requirements/decisions come first.
    Returns PO-native objects (ProjectRequirement, StrategicTheme); goal/vision stay None unless the
    Knowledge Agent holds them. Origin is 'knowledge_agent'. On failure: an empty StrategicContext."""
    def source(project_id: str):
        from uc2_context import StrategicContext                       # PO Agent modules, imported lazily
        from uc2_contracts import ProjectRequirement, StrategicTheme

        pkg = connector.get_package(project_id, PKG_UC2, focus_text=focus_text, top_k=top_k)
        k = pkg.knowledge
        if pkg.status == STATUS_ERROR:
            return StrategicContext(origin=KNOWLEDGE_ORIGIN)
        return StrategicContext(
            origin=KNOWLEDGE_ORIGIN,
            product_goal=k.product_goal.text if k.product_goal else None,
            product_goal_source=k.product_goal.source if k.product_goal else None,
            product_vision=k.product_vision.text if k.product_vision else None,
            product_vision_source=k.product_vision.source if k.product_vision else None,
            themes=[StrategicTheme(**t) for t in k.to_dict()["strategic_themes"]],
            requirements=[ProjectRequirement(**r) for r in k.to_dict()["requirements"]])
    return source


# ----------------------------------------------------------------------------------------------------------
# WRITE helper: store a committed PBI / story
# ----------------------------------------------------------------------------------------------------------
def pbi_to_text(pbi) -> str:
    """Readable text for a PBI / proposal (model or dict): title, story, acceptance criteria."""
    g = (lambda k: pbi.get(k)) if isinstance(pbi, dict) else (lambda k: getattr(pbi, k, None))
    lines = [str(g("title") or "").strip(), str(g("user_story_narrative") or "").strip()]
    criteria = [str(c).strip() for c in (g("acceptance_criteria") or []) if str(c).strip()]
    if criteria:
        lines.append("Acceptance criteria:")
        lines += [f"- {c}" for c in criteria]
    return "\n".join(x for x in lines if x)


def store_pbi(connector: KnowledgeConnector, project_id: str, pbi, *, item_id: Optional[str] = None,
              item_type: str = "user_story") -> ItemResponse:
    """Send a COMMITTED PBI/proposal (PBIProposalContract, dict, or any object with the same fields) to the
    Knowledge Agent. `item_id` defaults to target_pbi_id, else proposal_id. Refuses anything whose `status`
    is present and not COMMITTED, so drafts and pending proposals are never stored. Never raises."""
    try:
        g = (lambda k: pbi.get(k)) if isinstance(pbi, dict) else (lambda k: getattr(pbi, k, None))
        status = g("status")
        iid = item_id or g("target_pbi_id") or g("proposal_id")
        rid = f"po_{uuid.uuid4().hex[:12]}"
        if status is not None and str(getattr(status, "value", status)) != "COMMITTED":
            return _error_item(rid, ERR_INVALID_REQUEST,
                               f"Only COMMITTED items are stored (status is {status!r}).", "not_approved", {})
        return connector.put_item(project_id, item_type, str(iid or ""), pbi_to_text(pbi),
                                  title=g("title"), approved=True)
    except Exception as e:  # noqa: BLE001
        return _error_item(f"po_{uuid.uuid4().hex[:12]}", ERR_INTERNAL, f"{type(e).__name__}: {e}", "call_failed", {})