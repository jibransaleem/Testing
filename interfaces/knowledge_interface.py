"""
knowledge_interface.py  --  KNOWLEDGE AGENT side: receives a request from the PO Agent

    interface = KnowledgeInterface(knowledge_agent)        # or KnowledgeInterface.from_env()
    interface.get_package(package_request_dict)   # READ : PO's UC1/UC2 context package (knowledge half)
    interface.handle(context_request_dict)        # READ : ranked evidence passages with citations
    interface.put_item(item_request_dict)         # WRITE: PO stores an approved story / note / action item
    interface.projects()                          # project ids the PO can ask about (plain strings)

* The request/response format is defined in po_knowledge_contract.py.
* None of the methods raise: any failure comes back as a normal response with status="error".
* Every response is checked against the contract before it is returned.
* Ids are UUIDs in the database; everything leaving this module is plain JSON (str ids).
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from interfaces.context_interface import error_response, _package_error
from interfaces.knowledge_items import _error as _item_error
import contracts.po_knowledge_contract as _contract

ERR_INTERNAL = getattr(_contract, "ERR_INTERNAL", "INTERNAL_ERROR")
validate_item_response = getattr(_contract, "validate_item_response", None)
validate_package_response = getattr(_contract, "validate_package_response", None)
validate_response = getattr(_contract, "validate_response", None)


def _request_id(request: Any) -> str:
    ctx = request.get("context") if isinstance(request, dict) else None
    if isinstance(ctx, dict) and ctx.get("request_id"):
        return str(ctx["request_id"])
    return f"req_{uuid.uuid4().hex[:12]}"


def _context(request: Any) -> Dict[str, Any]:
    ctx = request.get("context") if isinstance(request, dict) else None
    return ctx if isinstance(ctx, dict) else {}


class KnowledgeInterface:
    def __init__(self, agent):
        self.agent = agent

    @classmethod
    def from_env(cls) -> "KnowledgeInterface":
        """Knowledge Agent from .env (Jina embeddings, optional Groq)."""
        from agents import KnowledgeAgent
        return cls(KnowledgeAgent.from_env())

    # ------------------------------------------------------------------ READ: passages
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Receive one PO request, return the contract response."""
        rid = _request_id(request)
        try:
            response = self.agent.get_context(request)
            if callable(validate_response):
                validate_response(response)
            return response
        except Exception as e:  # noqa: BLE001
            query = request.get("query") if isinstance(request, dict) and isinstance(request.get("query"), str) else ""
            return error_response(rid, query, ERR_INTERNAL, f"{type(e).__name__}: {e}",
                                  context=_context(request))

    get_context = handle

    # ------------------------------------------------------------------ READ: package
    def get_package(self, request: Dict[str, Any]) -> Dict[str, Any]:
        rid = _request_id(request)
        try:
            response = self.agent.get_package(request)
            if callable(validate_package_response):
                validate_package_response(response)
            return response
        except Exception as e:  # noqa: BLE001
            ptype = request.get("package_type") if isinstance(request, dict) else None
            return _package_error(rid, ptype if isinstance(ptype, str) else None, ERR_INTERNAL,
                                  f"{type(e).__name__}: {e}", context=_context(request))

    # ------------------------------------------------------------------ WRITE
    def put_item(self, request: Dict[str, Any]) -> Dict[str, Any]:
        rid = _request_id(request)
        try:
            response = self.agent.put_item(request)
            if callable(validate_item_response):
                validate_item_response(response)
            return response
        except Exception as e:  # noqa: BLE001
            return _item_error(rid, ERR_INTERNAL, f"{type(e).__name__}: {e}", context=_context(request))

    # ------------------------------------------------------------------ projects
    def projects(self) -> List[Dict[str, Optional[str]]]:
        """Project ids the PO Agent can ask about (for building its project_map). Plain strings."""
        return [{"project_id": None if p["project_id"] is None else str(p["project_id"]),
                 "name": p["name"]}
                for p in self.agent.list_projects()]