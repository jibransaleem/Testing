"""
po_interface.py  --  PO AGENT side: asks the Knowledge Agent for context

Put this file in the PO Agent project (with po_knowledge_contract.py). It imports no Knowledge Agent
code: it only needs an object with `.get_package(dict)`, `.handle(dict)` and `.put_item(dict)`
(knowledge_interface.KnowledgeInterface).

    connector = KnowledgeConnector(knowledge_interface,
                                   project_map={"proj-agiliro-001": "<projects.id UUID>"})

Knowledge Agent project ids are UUIDs (projects.id); get them from KnowledgeInterface.projects().
A PO project id that is not in project_map is sent as-is, so it must itself be that UUID.

WRITE (store approved project knowledge)
    connector.put_item(pid, "decision", "DEC-1", text, title="Hosting", approved=True)
    store_pbi(connector, pid, committed_pbi)
    connector.retire_item(pid, "decision", "DEC-1")

    Extra fields for the typed tables go in `context` (UUIDs must be real UUIDs of existing rows):
      meeting_summary / standup / retro_feedback : meeting_date, sprint_id, author (users.id)
      user_story   : requirement_id, acceptance_criteria (list)
      action_item  : meeting_id, sprint_id, owner (users.id), due_date, status (open|in_progress|done)

UC1:  ctx = enrich_context(ctx, connector)
UC2:  source = strategic_source(connector, focus_text=f"{pbi.title}. {pbi.user_story_narrative}")

Nothing here raises: failures come back as status="error" (details["origin"] == "po_interface" when found
on the PO side), and the adapters then degrade to "no extra context".
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import contracts.po_knowledge_contract as _contract
from contracts.po_knowledge_contract import (
    ContextRequest,
    ContextResponse,
    ContractError,
    InvalidRequest,
    ItemRequest,
    ItemResponse,
    PackageRequest,
    PackageResponse,
)

ERR_INTERNAL = getattr(_contract, "ERR_INTERNAL", "INTERNAL_ERROR")
ERR_INVALID_REQUEST = getattr(_contract, "ERR_INVALID_REQUEST", "INVALID_REQUEST")
KNOWLEDGE_CATEGORIES = getattr(
    _contract,
    "KNOWLEDGE_CATEGORIES",
    ("PRODUCT_GOAL", "PRODUCT_VISION", "STRATEGIC_THEMES", "REQUIREMENTS_DECISIONS"),
)
KNOWLEDGE_ORIGIN = getattr(_contract, "KNOWLEDGE_ORIGIN", "knowledge_agent")
PKG_UC1 = getattr(_contract, "PKG_UC1", "UC1_CONTEXT")
PKG_UC2 = getattr(_contract, "PKG_UC2", "UC2_ALIGNMENT")
SCHEMA_VERSION = getattr(_contract, "SCHEMA_VERSION", "1.2")
STATUS_ERROR = getattr(_contract, "STATUS_ERROR", "error")
OP_RETIRE = getattr(_contract, "OP_RETIRE", "retire")
OP_UPSERT = getattr(_contract, "OP_UPSERT", getattr(_contract, "OP_STORE", "store"))

_NOW = lambda: time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())  # noqa: E731


def _canon(value: Any) -> str:
    """A UUID in any spelling -> lower-case hyphenated (how the Knowledge Agent returns it)."""
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return str(value)


def _uuid_or_none(value: Any) -> Optional[str]:
    try:
        return str(uuid.UUID(str(value))) if value is not None else None
    except (ValueError, AttributeError, TypeError):
        return None


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
        self.project_map = dict(project_map or {})      # PO project id -> Knowledge Agent project id (UUID)
        self.requester = requester

    def _ids(self, project_id: str, purpose: Optional[str], request_id: Optional[str]):
        rid = request_id or f"po_{uuid.uuid4().hex[:12]}"
        ka = _canon(self.project_map.get(project_id, project_id))
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
        elif pkg.project and _canon(pkg.project.project_id) != ka:
            bad = f"response is for project {pkg.project.project_id!r}, asked for {ka!r}"
        if bad:
            return _error_package(rid, package_type, ERR_INTERNAL, f"Response rejected: {bad}",
                                  "response_rejected", ctx)
        return pkg

    # ------------------------------------------------------------------ WRITE
    def put_item(self, project_id: str, item_type: str, item_id: str, text: Optional[str] = None, *,
                 approved: bool, title: Optional[str] = None, operation: str = OP_UPSERT,
                 context: Optional[Dict[str, Any]] = None,
                 request_id: Optional[str] = None) -> ItemResponse:
        """Send an APPROVED item for the Knowledge Agent to store. `approved` must be True for upsert
        (never send drafts or proposals). Same item_id again: unchanged / new version. Never raises.
        `context` carries the typed-table fields; None values are dropped."""
        rid, ka, ctx = self._ids(project_id, None, request_id)
        extra = {k: v for k, v in (context or {}).items() if v is not None and v != ""}
        ctx = {**extra, **ctx}                                  # request_id / requester always win
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
        elif resp.project and _canon(resp.project.project_id) != ka:
            bad = f"response is for project {resp.project.project_id!r}, asked for {ka!r}"
        elif resp.item and (resp.item.item_type, resp.item.item_id) != (item_type, item_id):
            bad = f"response is for item {resp.item.item_type}/{resp.item.item_id}"
        if bad:
            return _error_item(rid, ERR_INTERNAL, f"Response rejected: {bad}", "response_rejected", ctx)
        return resp

    def retire_item(self, project_id: str, item_type: str, item_id: str, *,
                    request_id: Optional[str] = None) -> ItemResponse:
        """The item is no longer in force (kept as history). Not supported for user_story."""
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
        elif resp.project and _canon(resp.project.project_id) != ka:
            bad = f"response is for project {resp.project.project_id!r}, asked for {ka!r}"
        else:
            bad = next((f"passage {p.passage_id!r} is from project {p.source.project_id!r}"
                        for p in resp.passages if _canon(p.source.project_id) != ka), None)
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
    requirements/decisions appended to `project_requirements`. Never raises; on any problem the original
    package is returned unchanged."""
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
    """A `strategic_source` (project_id -> StrategicContext) backed by the Knowledge Agent.
    On failure: an empty StrategicContext."""
    def source(project_id: str):
        from uc.uc2_context import StrategicContext                       # PO Agent modules, imported lazily
        from uc.uc2_contracts import ProjectRequirement, StrategicTheme

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
def _criteria(pbi_get) -> List[str]:
    return [str(c).strip() for c in (pbi_get("acceptance_criteria") or []) if str(c).strip()]


def pbi_to_text(pbi, *, include_criteria: bool = True) -> str:
    """Readable text for a PBI / proposal (model or dict): title, story, acceptance criteria."""
    g = (lambda k: pbi.get(k)) if isinstance(pbi, dict) else (lambda k: getattr(pbi, k, None))
    lines = [str(g("title") or "").strip(), str(g("user_story_narrative") or "").strip()]
    criteria = _criteria(g) if include_criteria else []
    if criteria:
        lines.append("Acceptance criteria:")
        lines += [f"- {c}" for c in criteria]
    return "\n".join(x for x in lines if x)


def store_pbi(connector: KnowledgeConnector, project_id: str, pbi, *, item_id: Optional[str] = None,
              item_type: str = "user_story") -> ItemResponse:
    """Send a COMMITTED PBI/proposal to the Knowledge Agent. `item_id` defaults to target_pbi_key /
    target_pbi_id, else proposal_key / proposal_id. Refuses anything whose `status` is present and not
    COMMITTED. For 'user_story' the narrative goes to user_stories.story_text and the acceptance criteria go
    to acceptance_criteria (via context). Never raises."""
    try:
        g = (lambda k: pbi.get(k)) if isinstance(pbi, dict) else (lambda k: getattr(pbi, k, None))
        status = g("status")
        iid = (item_id or g("target_pbi_key") or g("target_pbi_id")
               or g("proposal_key") or g("proposal_id"))
        rid = f"po_{uuid.uuid4().hex[:12]}"
        if status is not None and str(getattr(status, "value", status)) != "COMMITTED":
            return _error_item(rid, ERR_INVALID_REQUEST,
                               f"Only COMMITTED items are stored (status is {status!r}).", "not_approved", {})
        title = g("title")
        if item_type == "user_story":
            text = str(g("user_story_narrative") or "").strip() or str(title or "").strip()
            context = {"acceptance_criteria": _criteria(g) or None,
                       "requirement_id": _uuid_or_none(g("requirement_id"))}
        else:
            text, context = pbi_to_text(pbi), None
        return connector.put_item(project_id, item_type, str(iid or ""), text,
                                  title=title, approved=True, context=context)
    except Exception as e:  # noqa: BLE001
        return _error_item(f"po_{uuid.uuid4().hex[:12]}", ERR_INTERNAL, f"{type(e).__name__}: {e}", "call_failed", {})