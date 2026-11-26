

from __future__ import annotations

import importlib
import os
import uuid
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Optional contract validators
# ---------------------------------------------------------

def _load_contract():

    for mod_name in (
        "contracts.po_knowledge_contract",
        "po_knowledge_contract",
    ):
        try:
            return importlib.import_module(mod_name)
        except ImportError:
            continue

    return None


def _build_validators(mod):

    if mod is None:
        return {}, {}

    req_map = {
        "get_context": "validate_request",
        "get_package": "validate_package_request",
        "put_item": "validate_item_request",
    }

    resp_map = {
        "get_context": "validate_response",
        "get_package": "validate_package_response",
        "put_item": "validate_item_response",
    }

    reqs = {}
    resps = {}

    for skill, attr in req_map.items():
        fn = getattr(mod, attr, None)
        if callable(fn):
            reqs[skill] = fn

    for skill, attr in resp_map.items():
        fn = getattr(mod, attr, None)
        if callable(fn):
            resps[skill] = fn

    return reqs, resps


contract = _load_contract()

_REQ_CHECK, _RESP_CHECK = _build_validators(contract)

_ContractError = (
    getattr(contract, "ContractError", ValueError)
    if contract is not None
    else ValueError
)


# ---------------------------------------------------------
# Custom exception
# ---------------------------------------------------------

class A2AError(RuntimeError):

    def __init__(
        self,
        message: str,
        code: Optional[int] = None,
        data: Any = None,
    ):
        super().__init__(message)
        self.code = code
        self.data = data


# ---------------------------------------------------------
# URL helper
# ---------------------------------------------------------

def _resolve_endpoint(
    base_url: str,
    raw_url: Optional[str],
) -> str:

    if not raw_url:
        return f"{base_url}/a2a"

    if raw_url.startswith(("http://", "https://")):
        return raw_url

    return f"{base_url}/{raw_url.lstrip('/')}"


# ---------------------------------------------------------
# Knowledge Agent Client
# ---------------------------------------------------------

class KnowledgeA2AClient:

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        requester: str = "po_agent",
        validate: bool = True,
    ):

        self.base_url = base_url.rstrip("/")
        self.requester = requester

        self.validate = validate and contract is not None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        self.http = httpx.Client(
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )

        self._card = None

    # -----------------------------------------------------
    # Agent discovery
    # -----------------------------------------------------

    def agent_card(self, refresh: bool = False):

        if self._card is None or refresh:

            r = self.http.get(
                f"{self.base_url}/.well-known/agent-card.json"
            )

            if r.status_code == 404:
                r = self.http.get(
                    f"{self.base_url}/.well-known/agent.json"
                )

            r.raise_for_status()

            self._card = r.json()

        return self._card

    # -----------------------------------------------------
    # Health
    # -----------------------------------------------------

    def health(self, deep: bool = False):

        r = self.http.get(
            f"{self.base_url}/health",
            params={"deep": 1} if deep else None,
        )

        r.raise_for_status()

        return r.json()

    # -----------------------------------------------------
    # Core A2A communication
    # -----------------------------------------------------

    def call(
        self,
        skill: str,
        request: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        request = dict(request or {})

        # Inject request context
        ctx = (
            request.get("context")
            if isinstance(request.get("context"), dict)
            else {}
        )

        request["context"] = {
            "requester": self.requester,
            "request_id": ctx.get(
                "request_id",
                f"req_{uuid.uuid4().hex[:12]}",
            ),
            **ctx,
        }

        # Validate request
        if self.validate and skill in _REQ_CHECK:

            try:
                _REQ_CHECK[skill](request)

            except _ContractError as e:

                raise A2AError(
                    f"Request breaks contract: {e}",
                    -32602,
                ) from e

        # Build JSON-RPC payload
        payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "role": "user",
                    "messageId": uuid.uuid4().hex,
                    "parts": [
                        {
                            "kind": "data",
                            "data": {
                                "skill": skill,
                                "request": request,
                            },
                        }
                    ],
                }
            },
        }

        # Discover A2A endpoint
        card = self.agent_card()

        endpoint = _resolve_endpoint(
            self.base_url,
            card.get("url"),
        )

        # Send request
        try:
            r = self.http.post(
                endpoint,
                json=payload,
            )

        except httpx.RequestError as e:

            raise A2AError(
                f"HTTP connection failed: {e}"
            ) from e

        if r.status_code == 401:

            raise A2AError(
                "Unauthorized: invalid or missing API key",
                -32001,
            )

        if r.status_code >= 500:

            raise A2AError(
                f"Server error ({r.status_code}): {r.text[:500]}",
                -32603,
            )

        try:
            body = r.json()

        except ValueError as e:

            raise A2AError(
                f"Non-JSON response: {r.text[:500]}"
            ) from e

        if r.status_code >= 400:

            raise A2AError(
                f"HTTP error ({r.status_code}): {body}",
                r.status_code,
                body,
            )

        # Handle JSON-RPC errors
        if "error" in body:

            error = body["error"]

            raise A2AError(
                error.get("message", "A2A error"),
                error.get("code"),
                error.get("data"),
            )

        if not isinstance(body.get("result"), dict):

            raise A2AError(
                f"Malformed JSON-RPC response: {body}",
                -32603,
            )

        # Extract task artifact
        data = self._result_data(body["result"])

        # Validate response
        if self.validate and skill in _RESP_CHECK:

            try:
                _RESP_CHECK[skill](data)

            except _ContractError as e:

                raise A2AError(
                    f"Response breaks contract: {e}",
                    -32603,
                    data,
                ) from e

        return data

    # -----------------------------------------------------
    # Extract A2A task artifact
    # -----------------------------------------------------

    @staticmethod
    def _result_data(task: Dict[str, Any]):

        for artifact in task.get("artifacts") or []:

            for part in artifact.get("parts") or []:

                if isinstance(part.get("data"), dict):
                    return part["data"]

        raise A2AError(
            f"Task {task.get('id')} has no data artifact "
            f"(state={task.get('status', {}).get('state')})"
        )

    # -----------------------------------------------------
    # Knowledge Agent Skills
    # -----------------------------------------------------

    def ping(self):
        return self.call("ping")

    def get_context(self, request: Dict[str, Any]):
        return self.call("get_context", request)

    def get_package(self, request: Dict[str, Any]):

        # Get Package does not accept the "query" field.
        # Remove it if a shared request dictionary was supplied.
        package_request = dict(request)

        package_request.pop("query", None)

        return self.call(
            "get_package",
            package_request,
        )

    def put_item(self, request: Dict[str, Any]):
        return self.call("put_item", request)

    # -----------------------------------------------------
    # Close HTTP connection
    # -----------------------------------------------------

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---------------------------------------------------------
# CLIENT FACTORY
# ---------------------------------------------------------

def get_kb_client(
    timeout: float = 60.0,
    requester: str = "po_agent",
    validate: bool = True,
) -> KnowledgeA2AClient:

    """
    Load configuration from .env and return a ready-to-use
    Knowledge Agent client.
    """

    base_url = (
        os.getenv("A2A_URL")
        or os.getenv("KB_URL")
        or ""
    ).strip()

    api_key = (
        os.getenv("A2A_API_KEY")
        or os.getenv("KB_API_KEY")
        or ""
    ).strip()

    if not base_url:

        raise RuntimeError(
            "Missing A2A_URL. Add A2A_URL to your .env file."
        )

    if not api_key:

        raise RuntimeError(
            "Missing A2A_API_KEY. Add A2A_API_KEY to your .env file."
        )

    return KnowledgeA2AClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        requester=requester,
        validate=validate,
    )
