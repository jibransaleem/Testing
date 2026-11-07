"""
knowledge_interface.py  --  KNOWLEDGE AGENT side: receives a request from the PO Agent

Put this file in the Knowledge Agent project. Three methods:

    interface = KnowledgeInterface(knowledge_agent)        # or KnowledgeInterface.from_env()
    interface.get_package(package_request_dict)   # READ : PO's UC1/UC2 context package (knowledge half)
    interface.handle(context_request_dict)        # READ : ranked evidence passages with citations
    interface.put_item(item_request_dict)         # WRITE: PO stores an approved story / note / action item

* The request/response format is defined in po_knowledge_contract.py.
* handle() never raises: any failure comes back as a normal response with status="error".
* The response is checked against the contract before it is returned.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict

from context_interface import error_response
from context_interface import _package_error
from knowledge_items import _error as _item_error
from po_knowledge_contract import (
    ERR_INTERNAL, validate_item_response, validate_package_response, validate_response,
)


class KnowledgeInterface:
    def __init__(self, agent):
        self.agent = agent

    @classmethod
    def from_env(cls) -> "KnowledgeInterface":
        """Knowledge Agent from .env (Jina embeddings, optional Groq)."""
        from agents import KnowledgeAgent
        return cls(KnowledgeAgent.from_env())

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Receive one PO request, return the contract response."""
        ctx = request.get("context") if isinstance(request, dict) else None
        rid = str((ctx or {}).get("request_id") or f"req_{uuid.uuid4().hex[:12]}") if isinstance(ctx, dict) \
            else f"req_{uuid.uuid4().hex[:12]}"
        try:
            response = self.agent.get_context(request)   # validates the request against the contract
            validate_response(response)                  # we must obey the contract too
            return response
        except Exception as e:  # noqa: BLE001
            query = request.get("query") if isinstance(request, dict) and isinstance(request.get("query"), str) else ""
            return error_response(rid, query, ERR_INTERNAL, f"{type(e).__name__}: {e}")

    get_context = handle        # clearer name for the passage request; `handle` kept for compatibility

    def get_package(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """PO asks for the knowledge half of its UC1 / UC2 context package. Never raises; contract-checked."""
        ctx = request.get("context") if isinstance(request, dict) else None
        rid = str(ctx.get("request_id")) if isinstance(ctx, dict) and ctx.get("request_id") else f"req_{uuid.uuid4().hex[:12]}"
        try:
            response = self.agent.get_package(request)
            validate_package_response(response)
            return response
        except Exception as e:  # noqa: BLE001
            return _package_error(rid, None, ERR_INTERNAL, f"{type(e).__name__}: {e}")

    def put_item(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """PO sends an APPROVED item (user story, meeting summary, action item, ...) to store for a project.
        Idempotent per item_id; a changed item becomes a new version. Never raises; contract-checked."""
        ctx = request.get("context") if isinstance(request, dict) else None
        rid = str(ctx.get("request_id")) if isinstance(ctx, dict) and ctx.get("request_id") else f"req_{uuid.uuid4().hex[:12]}"
        try:
            response = self.agent.put_item(request)
            validate_item_response(response)
            return response
        except Exception as e:  # noqa: BLE001
            return _item_error(rid, ERR_INTERNAL, f"{type(e).__name__}: {e}")

    def projects(self):
        """Project ids the PO Agent can ask about (for building its project_map)."""
        return self.agent.list_projects()