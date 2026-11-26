"""
po_kb_client.py -- PO AGENT side: sends requests to the Knowledge Agent (KB) over A2A.

    pip install httpx

    from po_kb_client import KnowledgeClient
    kb = KnowledgeClient("http://localhost:8000", project_id="<projects.id UUID>", verbose=True)

    r = kb.store_meeting_summary(meeting_id="sprint-12-nov", title="Sprint planning", date="2026-11-12",
                                 attendees=["Jibran", "Taha"], summary="Planned sprint 3.",
                                 decisions=["Attendance save target is 3 seconds"], approved=True)
    r.status    # stored | updated | unchanged
    kb.retire("meeting_summary", "sprint-12-nov")
    ctx = kb.get_context("What was decided about attendance save time?")

Writes are idempotent: same item_id + same content -> "unchanged"; changed content -> new version.

Foreign keys: sprint_id, meeting_id, requirement_id, author and owner are ids of existing rows, so they
must be UUIDs (author / owner = users.id). Anything else is refused here, before any call.
Item types -> tables: meeting_summary / standup / retro_feedback -> meetings, decision -> decisions,
action_item -> action_items, user_story -> user_stories. (approval_rationale / refinement_note have no table.)
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

OK_STATUSES = ("stored", "updated", "unchanged", "retired")
MEETING_TYPES = ("meeting_summary", "standup", "retro_feedback")


class KBError(Exception):
    """Base error: the KB could not be reached or spoke the wrong protocol."""


class KBUnavailable(KBError):
    pass


class KBAuthError(KBError):
    pass


class KBProtocolError(KBError):
    pass


class KBWriteRejected(KBError):
    """Raised by WriteResult.raise_for_error() when the KB answered with status=error."""


@dataclass
class WriteResult:
    status: str
    item_type: str
    item_id: str
    version: Optional[int] = None
    chunk_id: Optional[str] = None
    previous_chunk_id: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[Any] = field(default_factory=list)
    request_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in OK_STATUSES

    @property
    def changed(self) -> bool:
        return self.status in ("stored", "updated", "retired")

    def raise_for_error(self) -> "WriteResult":
        if not self.ok:
            raise KBWriteRejected(f"{self.error_code}: {self.error_message}")
        return self


def slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:n] or "item"


def stable_id(prefix: str, *parts: str) -> str:
    """Deterministic id, so the SAME real-world item always maps to the SAME item_id
    and an edit becomes a new VERSION, not a duplicate."""
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{slug(parts[-1] if parts else prefix, 30)}-{digest}"


def _bullets(items: Optional[List[str]]) -> str:
    return "\n".join(f"- {i}" for i in items) if items else ""


def _uuid(name: str, value: Optional[str]) -> Optional[str]:
    """None stays None; a UUID (any spelling) becomes lower-case hyphenated; anything else is refused."""
    if value is None or value == "":
        return None
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        raise ValueError(f"{name} must be the UUID of an existing row, got {value!r}") from None


class KnowledgeClient:
    def __init__(self, base_url: str, project_id: str, api_key: Optional[str] = None,
                 timeout: float = 120.0, retries: int = 2, verbose: bool = False, logger=print):
        self.base = base_url.rstrip("/")
        self.project_id = _uuid("project_id", project_id)
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.verbose = verbose
        self._log = logger

    # ------------------------------------------------------------------ transport
    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _show(self, title: str, obj: Any) -> None:
        if self.verbose:
            self._log(f"--- {title}\n{json.dumps(obj, indent=2, ensure_ascii=False, default=str)}")

    def _send(self, skill: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """One A2A call; returns the contract response dict (the data artifact)."""
        msg = {"kind": "message", "role": "user", "messageId": uuid.uuid4().hex,
               "parts": [{"kind": "data", "data": {"skill": skill, "request": request}}]}
        body = {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": "message/send", "params": {"message": msg}}
        self._show(f"PO AGENT -> KB  skill={skill}  (INPUT)", request)

        last: Optional[Exception] = None
        t0 = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                t0 = time.perf_counter()
                r = httpx.post(f"{self.base}/a2a", json=body, headers=self._headers(), timeout=self.timeout)
                break
            except httpx.TransportError as e:                  # safe to retry (idempotent)
                last = e
                time.sleep(0.5 * (2 ** attempt))
        else:
            raise KBUnavailable(f"KB not reachable at {self.base}: {last}")

        if r.status_code == 401:
            raise KBAuthError("KB rejected the API key (401)")
        try:
            payload = r.json()
        except ValueError:
            raise KBProtocolError(f"KB returned non-JSON (HTTP {r.status_code}): {r.text[:200]}")
        if "error" in payload:
            err = payload["error"]
            raise KBProtocolError(f"JSON-RPC error {err.get('code')}: {err.get('message')}")
        for art in payload.get("result", {}).get("artifacts", []):
            for part in art.get("parts", []):
                if isinstance(part.get("data"), dict):
                    ms = int((time.perf_counter() - t0) * 1000)
                    self._show(f"KB -> PO AGENT  status={part['data'].get('status')}  ({ms} ms)  (OUTPUT)", part["data"])
                    return part["data"]
        raise KBProtocolError(f"no data artifact in KB response: {payload}")

    # ------------------------------------------------------------------ writes
    def put_item(self, item_type: str, item_id: str, title: Optional[str], text: str, *, approved: bool,
                 context: Optional[Dict[str, Any]] = None,
                 request_id: Optional[str] = None, **extra: Any) -> WriteResult:
        """Store an APPROVED item. `approved=True` must be passed explicitly."""
        if approved is not True:
            raise ValueError("refusing to send: only human-approved items may be written to the KB "
                             "(pass approved=True after the user approved it)")
        if not (item_type and item_id and (text or "").strip()):
            raise ValueError("item_type, item_id and non-empty text are required")
        rid = request_id or f"po_{uuid.uuid4().hex[:12]}"
        ctx = {k: v for k, v in (context or {}).items() if v is not None}
        req = {"project_id": self.project_id, "item_type": item_type, "item_id": item_id,
               "title": title, "text": text, "approved": True,
               "context": {**ctx, "request_id": rid}, **extra}
        return self._to_result(self._send("put_item", req), item_type, item_id, rid)

    def retire(self, item_type: str, item_id: str, request_id: Optional[str] = None) -> WriteResult:
        """Retire an item (current version -> history). Retiring twice is fine ("unchanged").
        user_story cannot be retired: the KB answers INVALID_REQUEST."""
        rid = request_id or f"po_{uuid.uuid4().hex[:12]}"
        req = {"project_id": self.project_id, "item_type": item_type, "item_id": item_id,
               "operation": "retire", "context": {"request_id": rid}}
        return self._to_result(self._send("put_item", req), item_type, item_id, rid)

    @staticmethod
    def _to_result(data: Dict[str, Any], item_type: str, item_id: str, rid: str) -> WriteResult:
        item, err = data.get("item") or {}, data.get("error") or {}
        return WriteResult(status=data.get("status", "error"), item_type=item_type, item_id=item_id,
                           version=item.get("version"), chunk_id=item.get("chunk_id"),
                           previous_chunk_id=item.get("previous_chunk_id"),
                           error_code=err.get("code"), error_message=err.get("message"),
                           warnings=data.get("warnings") or [], request_id=data.get("request_id") or rid, raw=data)

    # ------------------------------------------------------------------ typed writers
    def store_meeting_summary(self, *, title: str, summary: str, approved: bool,
                              meeting_id: Optional[str] = None, date: Optional[str] = None,
                              attendees: Optional[List[str]] = None, decisions: Optional[List[str]] = None,
                              action_items: Optional[List[Dict[str, str]]] = None,
                              sprint_id: Optional[str] = None, author: Optional[str] = None,
                              item_type: str = "meeting_summary") -> WriteResult:
        """Stored in `meetings`. `item_type` picks meetings.meeting_type: meeting_summary -> 'meeting',
        standup -> 'standup', retro_feedback -> 'retrospective'. `sprint_id` = sprints.id, `author` = users.id."""
        if item_type not in MEETING_TYPES:
            raise ValueError(f"item_type must be one of {list(MEETING_TYPES)}")
        acts = [f"{a.get('owner', '?')}: {a.get('task', '')}" + (f" (due {a['due']})" if a.get("due") else "")
                for a in (action_items or [])]
        parts = [f"Meeting: {title}", f"Date: {date}" if date else "",
                 f"Attendees: {', '.join(attendees)}" if attendees else "",
                 f"Summary: {summary}",
                 f"Decisions:\n{_bullets(decisions)}" if decisions else "",
                 f"Action items:\n{_bullets(acts)}" if acts else ""]
        text = "\n".join(p for p in parts if p)                 # deterministic -> retries are "unchanged"
        item_id = meeting_id or stable_id("MTG", date or "", title)
        return self.put_item(item_type, item_id, title, text, approved=approved,
                             context={"meeting_date": date, "sprint_id": _uuid("sprint_id", sprint_id),
                                      "author": _uuid("author", author)})

    def store_user_story(self, *, title: str, as_a: str, i_want: str, so_that: str, approved: bool,
                         story_id: Optional[str] = None, requirement_id: Optional[str] = None,
                         acceptance_criteria: Optional[List[str]] = None) -> WriteResult:
        """Stored in `user_stories` (+ `acceptance_criteria`). Not searchable; cannot be retired."""
        text = f"As a {as_a}, I want {i_want} so that {so_that}."
        return self.put_item("user_story", story_id or stable_id("US", title), title, text, approved=approved,
                             context={"requirement_id": _uuid("requirement_id", requirement_id),
                                      "acceptance_criteria": acceptance_criteria})

    def store_action_item(self, *, description: str, approved: bool, title: Optional[str] = None,
                          action_id: Optional[str] = None, owner: Optional[str] = None,
                          due: Optional[str] = None, status: Optional[str] = None,
                          meeting_id: Optional[str] = None, sprint_id: Optional[str] = None) -> WriteResult:
        """Stored in `action_items`. `owner` = users.id, `due` YYYY-MM-DD, `status` open|in_progress|done."""
        return self.put_item("action_item", action_id or stable_id("ACT", title or description),
                             title, description, approved=approved,
                             context={"owner": _uuid("owner", owner), "due_date": due, "status": status,
                                      "meeting_id": _uuid("meeting_id", meeting_id),
                                      "sprint_id": _uuid("sprint_id", sprint_id)})

    def store_decision(self, *, title: str, decision: str, approved: bool,
                       decision_id: Optional[str] = None) -> WriteResult:
        """Stored in `decisions`; searchable and part of the UC1/UC2 packages. A changed decision becomes a
        new version that supersedes the old one."""
        return self.put_item("decision", decision_id or stable_id("DEC", title), title, decision,
                             approved=approved)

    # ------------------------------------------------------------------ reads
    def get_context(self, query: str, project_id: Optional[str] = None) -> Dict[str, Any]:
        return self._send("get_context", {"project_id": project_id or self.project_id, "query": query})

    def get_package(self, package_type: str = "UC1_CONTEXT", **extra: Any) -> Dict[str, Any]:
        """package_type: 'UC1_CONTEXT' | 'UC2_ALIGNMENT'."""
        return self._send("get_package", {"project_id": self.project_id, "package_type": package_type, **extra})

    def ping(self) -> Dict[str, Any]:
        return self._send("ping", {})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="PO Agent -> KB write demo")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--project", required=True, help="Knowledge Agent project id (UUID, projects.id)")
    ap.add_argument("--key", default=None)
    a = ap.parse_args()
    kb = KnowledgeClient(a.base, a.project, a.key, verbose=True)

    run = uuid.uuid4().hex[:6]
    meeting = dict(meeting_id=f"PO-DEMO-MTG-{run}", title="Sprint planning", date="2026-11-12",
                   attendees=["Jibran", "Taha"], summary="Planned the next sprint.",
                   action_items=[{"owner": "Ali", "task": "Add database index", "due": "2026-11-14"}])

    print("\n### 1. first write");  r1 = kb.store_meeting_summary(**meeting, decisions=["Save target stays 5 seconds"], approved=True)
    print("\n### 2. same again");   r2 = kb.store_meeting_summary(**meeting, decisions=["Save target stays 5 seconds"], approved=True)
    print("\n### 3. content changed"); r3 = kb.store_meeting_summary(**meeting, decisions=["Save target is 3 seconds"], approved=True)
    print("\n### 4. retire");       r4 = kb.retire("meeting_summary", meeting["meeting_id"])
    print("\n### 5. draft is refused on the PO side")
    try:
        kb.put_item("meeting_summary", "x", "t", "draft text", approved=False)
    except ValueError as e:
        print("refused:", e)
    print("\nRESULT:", [(r.status, r.version) for r in (r1, r2, r3, r4)])