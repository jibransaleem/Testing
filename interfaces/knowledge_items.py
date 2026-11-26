"""
knowledge_items.py  --  Knowledge Agent side: STORE approved items sent by the PO Agent

    put_item(agent, request_dict) -> ItemResponse dict        (never raises)

RULE: only APPROVED items are written ("approved": true). Anything else is rejected, nothing is stored.
(Retire requests carry no content, so no flag is needed.)

An item is identified by project_id + item_type + item_id.
    first time            -> stored      version 1
    same content again    -> unchanged   (safe to retry; nothing written)
    changed content       -> updated     new version; the previous version is deactivated (kept as history)
    operation 'retire'    -> retired     current version deactivated, no successor

WHERE EACH ITEM TYPE IS STORED
    item_type        typed table     chunk (searchable)   how a version is deactivated
    meeting_summary  meetings        yes (meeting_id)     its chunk is deleted (the meetings row stays as history)
    standup          meetings        yes                  same
    retro_feedback   meetings        yes                  same
    decision         decisions       yes (decision_id)    update: the new row's supersedes_id points at the old one
                                                          (v_chunks then shows the old one as superseded);
                                                          retire: decisions.approved = false
    action_item      action_items    no                   status = 'cancelled'
    user_story       user_stories    no                   no status exists: the highest version is current;
                                                          retire is refused

  * Requires the PATCHED v_chunks (decision superseded through supersedes_id), as utils/decisions.py does.
  * chunks can only point at a document, a decision or a meeting, so stories and action items are not
    searchable through get_context().
  * Version ids are deterministic uuid5(project, item_type, item_id, version). The chunk (when there is one)
    uses the SAME id as its typed row.
  * One write (typed row + chunk + deactivation) is ONE transaction and is read back before it ends; if
    anything is missing everything is rolled back and the caller gets WRITE_FAILED.

  context keys used
    meetings      meeting_date | date (YYYY-MM-DD), sprint_id, author (users.id)
    user_stories  requirement_id, acceptance_criteria (list or newline text)
    action_items  meeting_id, sprint_id, owner (users.id), due_date, status (open | in_progress | done)
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import date as _date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ingestion.embed import get_jina_embeddings
import contracts.po_knowledge_contract as _contract

ERR_INVALID_REQUEST = getattr(_contract, "ERR_INVALID_REQUEST", "INVALID_REQUEST")
ERR_ITEM_NOT_FOUND = getattr(_contract, "ERR_ITEM_NOT_FOUND", "ITEM_NOT_FOUND")
ERR_UNKNOWN_PROJECT = getattr(_contract, "ERR_UNKNOWN_PROJECT", "UNKNOWN_PROJECT")
ERR_WRITE_FAILED = getattr(_contract, "ERR_WRITE_FAILED", "WRITE_FAILED")
ITEM_RETIRED = getattr(_contract, "ITEM_RETIRED", "retired")
ITEM_STORED = getattr(_contract, "ITEM_STORED", "stored")
ITEM_UNCHANGED = getattr(_contract, "ITEM_UNCHANGED", "unchanged")
ITEM_UPDATED = getattr(_contract, "ITEM_UPDATED", "updated")
OP_RETIRE = getattr(_contract, "OP_RETIRE", "retire")
SCHEMA_VERSION = getattr(_contract, "SCHEMA_VERSION", "1.2")
STATUS_ERROR = getattr(_contract, "STATUS_ERROR", "error")
InvalidRequest = _contract.InvalidRequest
ItemRequest = _contract.ItemRequest

log = logging.getLogger("knowledge_items")

# Notes go to the server log; the response `warnings` stays [] unless the contract accepts strings.
WARNINGS_IN_RESPONSE = False

EVENT_PO_ITEM_SUBMITTED = "PO_ITEM_SUBMITTED"
EVENT_PO_ITEM_UPDATED = "PO_ITEM_UPDATED"

# Stable namespace for the deterministic version ids. Never change it: it would orphan stored items.
_ID_NS = uuid.UUID("5b1c6e0a-3f2d-4c7b-8e19-2a6d4f9b7c31")
MAX_VERSIONS = 10_000

# item_type -> (typed table, meetings.meeting_type)
KINDS: Dict[str, Tuple[str, Optional[str]]] = {
    "meeting_summary": ("meetings", "meeting"),
    "standup": ("meetings", "standup"),
    "retro_feedback": ("meetings", "retrospective"),
    "decision": ("decisions", None),
    "action_item": ("action_items", None),
    "user_story": ("user_stories", None),
}
# typed tables that also get a searchable chunk -> the chunks column that points at them
CHUNK_FK = {"meetings": "meeting_id", "decisions": "decision_id"}
_COMPARE = {
    "meetings": ("meeting_date", "summary", "meeting_type", "sprint_id", "author_id"),
    "decisions": ("description",),
    "action_items": ("meeting_id", "sprint_id", "description", "owner_id", "due_date", "status"),
    "user_stories": ("requirement_id", "title", "story_text"),
}
_ACTION_STATUSES = ("open", "in_progress", "done")


class ItemNotFound(LookupError):
    pass


class _BadExtra(ValueError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


# ---------------------------------------------------------------------------------------- small helpers
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", "" if text is None else str(text)).strip()


def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return "\n".join(_as_text(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    return str(v)


def _parse_uuid(value: Any) -> Optional[uuid.UUID]:
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _safe_rollback(conn) -> None:
    try:
        conn.rollback()
    except Exception:  # noqa: BLE001
        pass


def _envelope(rid, status, *, project=None, item=None, warnings=None, error=None, context=None, started=0.0):
    return {"schema_version": SCHEMA_VERSION, "request_id": rid, "status": status, "project": project,
            "item": item, "warnings": warnings or [], "error": error, "context": context or {},
            "meta": {"latency_ms": int((time.perf_counter() - started) * 1000) if started else 0,
                     "generated_at": _now()}}


def _error(rid, code, message, details=None, **kw):
    return _envelope(rid, STATUS_ERROR, error={"code": code, "message": message, "details": details or {}}, **kw)


def _item_uuid(project_id: str, item_type: str, item_id: str, version: int) -> str:
    return str(uuid.uuid5(_ID_NS, f"{project_id}|{item_type}|{item_id}|v{version}"))


def _ref(req: ItemRequest, version: int, item_uuid: str, previous: Optional[str], status: str) -> Dict[str, Any]:
    return {"item_type": req.item_type, "item_id": req.item_id, "version": version,
            "chunk_id": item_uuid, "previous_chunk_id": previous, "approval_status": status}


# ---------------------------------------------------------------------------------------- context extras
def _pick(merged: Dict[str, Any], request: Any, *names: str) -> Any:
    for n in names:
        for src in (merged, request if isinstance(request, dict) else {}):
            v = src.get(n)
            if v is not None and v != "":
                return v
    return None


def _uuid_field(name: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    u = _parse_uuid(value)
    if u is None:
        raise _BadExtra(f"{name} must be a UUID, got {value!r}")
    return str(u)


def _date_field(name: str, value: Any) -> Optional[_date]:
    if value is None:
        return None
    try:
        return _date.fromisoformat(str(value)[:10])
    except ValueError:
        raise _BadExtra(f"{name} must be YYYY-MM-DD, got {value!r}")


def _criteria(value: Any) -> List[str]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).splitlines()
    return [_norm(x) for x in items if _norm(x)]


def _extras(request: Any, req: ItemRequest, raw_ctx: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**raw_ctx, **(req.context or {})}
    table = KINDS[req.item_type][0]
    ex: Dict[str, Any] = {}
    if table == "meetings":
        ex["meeting_date"] = _date_field("meeting_date", _pick(merged, request, "meeting_date", "date"))
        ex["sprint_id"] = _uuid_field("sprint_id", _pick(merged, request, "sprint_id"))
        ex["author_id"] = _uuid_field("author", _pick(merged, request, "author", "author_id"))
    elif table == "user_stories":
        if not _norm(req.title):
            raise _BadExtra("title is required for a user_story")
        ex["requirement_id"] = _uuid_field("requirement_id", _pick(merged, request, "requirement_id"))
        ex["acceptance_criteria"] = _criteria(_pick(merged, request, "acceptance_criteria"))
    elif table == "action_items":
        ex["meeting_id"] = _uuid_field("meeting_id", _pick(merged, request, "meeting_id"))
        ex["sprint_id"] = _uuid_field("sprint_id", _pick(merged, request, "sprint_id"))
        ex["owner_id"] = _uuid_field("owner", _pick(merged, request, "owner", "owner_id"))
        ex["due_date"] = _date_field("due_date", _pick(merged, request, "due_date"))
        status = _pick(merged, request, "status") or "open"
        if status not in _ACTION_STATUSES:
            raise _BadExtra(f"action item status must be one of {list(_ACTION_STATUSES)}, got {status!r}")
        ex["status"] = status
    return ex


def _check_refs(conn, req: ItemRequest, ex: Dict[str, Any]) -> None:
    """Every foreign key is checked first: a clear message instead of a database error."""
    pid = req.project_id
    checks = []
    for field, table in (("sprint_id", "sprints"), ("meeting_id", "meetings")):
        if ex.get(field) is not None:
            checks.append((field, table, f"SELECT 1 FROM {table} WHERE id = %s AND project_id = %s", (ex[field], pid),
                           f"SELECT id FROM {table} WHERE project_id = %s ORDER BY id LIMIT 20", (pid,)))
    if ex.get("requirement_id") is not None:      # requirements belong to a project through their document
        checks.append(("requirement_id", "requirements",
                       "SELECT 1 FROM requirements r JOIN documents d ON d.id = r.document_id "
                       "WHERE r.id = %s AND d.project_id = %s", (ex["requirement_id"], pid),
                       "SELECT r.id FROM requirements r JOIN documents d ON d.id = r.document_id "
                       "WHERE d.project_id = %s ORDER BY r.id LIMIT 20", (pid,)))
    for field in ("author_id", "owner_id"):
        if ex.get(field) is not None:
            checks.append((field, "users", "SELECT 1 FROM users WHERE id = %s", (ex[field],),
                           "SELECT id FROM users ORDER BY id LIMIT 20", ()))
    for field, table, ok_sql, ok_params, known_sql, known_params in checks:
        if conn.execute(ok_sql, ok_params).fetchone():
            continue
        known = [str(r["id"]) for r in conn.execute(known_sql, known_params).fetchall()]
        raise _BadExtra(f"{field} '{ex[field]}' does not exist in table '{table}' for project {pid}",
                        {f"available_{table}_ids (max 20)": known})


# ---------------------------------------------------------------------------------------- typed rows
def _typed_values(req: ItemRequest, new_id: str, ex: Dict[str, Any], previous: Optional[str]) -> Dict[str, Any]:
    table, meeting_type = KINDS[req.item_type]
    base = {"id": new_id, "project_id": req.project_id}
    if table == "meetings":
        return {**base, "meeting_date": ex["meeting_date"], "summary": req.text, "meeting_type": meeting_type,
                "sprint_id": ex["sprint_id"], "author_id": ex["author_id"]}
    if table == "decisions":
        return {**base, "description": req.text, "approved": True, "supersedes_id": previous}
    if table == "action_items":
        return {**base, "meeting_id": ex["meeting_id"], "sprint_id": ex["sprint_id"], "description": req.text,
                "owner_id": ex["owner_id"], "status": ex["status"], "due_date": ex["due_date"]}
    return {**base, "requirement_id": ex["requirement_id"], "title": req.title, "story_text": req.text,
            "status": "ready"}                    # story_statuses has no 'approved'; 'ready' = approved by the PO


def _insert_typed(conn, req: ItemRequest, new_id: str, ex: Dict[str, Any], previous: Optional[str]) -> None:
    table = KINDS[req.item_type][0]
    values = _typed_values(req, new_id, ex, previous)
    cols = list(values)
    conn.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})",
                 [values[c] for c in cols])
    if table == "user_stories":
        for pos, criterion in enumerate(ex["acceptance_criteria"], 1):
            conn.execute("INSERT INTO acceptance_criteria (user_story_id, position, criterion) VALUES (%s, %s, %s)",
                         (new_id, pos, criterion))


def _insert_chunk(conn, req: ItemRequest, new_id: str, embedding, updated: bool) -> None:
    fk = CHUNK_FK[KINDS[req.item_type][0]]
    conn.execute(
        f"""INSERT INTO chunks (id, project_id, {fk}, chunk_index, chunk_text, section_label, source_event, embedding)
            VALUES (%s, %s, %s, 0, %s, %s, %s, %s::vector)""",
        (new_id, req.project_id, new_id, req.text, req.title,
         EVENT_PO_ITEM_UPDATED if updated else EVENT_PO_ITEM_SUBMITTED, embedding))


def _deactivate(conn, req: ItemRequest, item_uuid: str, warnings: List[str], retire: bool = False) -> None:
    table = KINDS[req.item_type][0]
    if table == "meetings":                       # v_chunks has no 'superseded' for meetings: drop the chunk
        conn.execute("DELETE FROM chunks WHERE meeting_id = %s", (item_uuid,))
    elif table == "decisions":
        if retire:                                # no successor: v_chunks shows approved=false as not approved
            conn.execute("UPDATE decisions SET approved = false WHERE id = %s", (item_uuid,))
        # on update nothing to do: the new row's supersedes_id already makes v_chunks mark this one superseded
    elif table == "action_items":
        conn.execute("UPDATE action_items SET status = 'cancelled' WHERE id = %s", (item_uuid,))
    else:
        warnings.append("user_stories has no superseded status: the older version stays as a row; "
                        "the highest version is the current one")


# ---------------------------------------------------------------------------------------- history
def _history(conn, req: ItemRequest) -> List[Dict[str, Any]]:
    """All versions of this item, newest first: {version, id, row, active}. Found by probing the deterministic ids."""
    table = KINDS[req.item_type][0]
    out: List[Dict[str, Any]] = []
    for version in range(1, MAX_VERSIONS + 1):
        iid = _item_uuid(req.project_id, req.item_type, req.item_id, version)
        row = conn.execute(f"SELECT * FROM {table} WHERE id = %s", (iid,)).fetchone()
        if row is None:
            break
        if table == "meetings":                   # active while its chunk exists
            active = conn.execute("SELECT 1 FROM chunks WHERE meeting_id = %s", (iid,)).fetchone() is not None
        elif table == "action_items":
            active = row["status"] != "cancelled"
        else:                                     # decisions: decided below; user_stories: always active
            active = True
        out.append({"version": version, "id": iid, "row": row, "active": active})
    out.reverse()
    if table == "decisions":                      # only the newest version can be current, and only if approved
        for i, v in enumerate(out):
            v["active"] = bool(v["row"]["approved"]) and i == 0
    return out


def _same(conn, req: ItemRequest, cur: Dict[str, Any], ex: Dict[str, Any]) -> bool:
    """Is the stored current version identical to what the PO just sent?"""
    table = KINDS[req.item_type][0]
    if table in CHUNK_FK:
        ch = conn.execute(f"SELECT section_label FROM chunks WHERE {CHUNK_FK[table]} = %s", (cur["id"],)).fetchone()
        if ch is None or _norm(ch["section_label"]) != _norm(req.title):
            return False
    new = _typed_values(req, cur["id"], ex, None)
    for col in _COMPARE[table]:
        if _norm(_as_text(cur["row"].get(col))) != _norm(_as_text(new.get(col))):
            return False
    if table == "user_stories":
        old = [r["criterion"] for r in conn.execute(
            "SELECT criterion FROM acceptance_criteria WHERE user_story_id = %s ORDER BY position",
            (cur["id"],)).fetchall()]
        if old != ex["acceptance_criteria"]:
            return False
    return True


def _verify_new(conn, req: ItemRequest, new_id: str) -> None:
    table = KINDS[req.item_type][0]
    if conn.execute(f"SELECT 1 FROM {table} WHERE id = %s", (new_id,)).fetchone() is None:
        raise RuntimeError(f"write could not be verified: no row in '{table}'")
    if table in CHUNK_FK:
        chk = conn.execute("SELECT approval_status, (embedding IS NOT NULL) AS has_embedding FROM v_chunks WHERE id = %s",
                           (new_id,)).fetchone()
        if not chk or chk["approval_status"] != "approved" or not chk["has_embedding"]:
            raise RuntimeError("write could not be verified: chunk missing, not approved or no embedding (v_chunks)")


def _verify_inactive(conn, req: ItemRequest, item_uuid: str) -> None:
    table = KINDS[req.item_type][0]
    if table in CHUNK_FK:
        chk = conn.execute("SELECT approval_status FROM v_chunks WHERE id = %s", (item_uuid,)).fetchone()
        if chk and chk["approval_status"] == "approved":
            raise RuntimeError("previous version is still approved in v_chunks "
                               "(is the patched v_chunks with supersedes_id installed?)")
    elif table == "action_items":
        row = conn.execute("SELECT status FROM action_items WHERE id = %s", (item_uuid,)).fetchone()
        if not row or row["status"] != "cancelled":
            raise RuntimeError("previous action item version was not cancelled")


# ---------------------------------------------------------------------------------------- entry point
def put_item(agent, request: Any) -> Dict[str, Any]:
    started = time.perf_counter()
    raw_ctx = request.get("context") if isinstance(request, dict) and isinstance(request.get("context"), dict) else {}
    rid = str(raw_ctx.get("request_id") or f"req_{uuid.uuid4().hex[:12]}")
    try:
        req = ItemRequest.from_dict(request)
    except InvalidRequest as e:
        return _error(rid, ERR_INVALID_REQUEST, str(e), context=raw_ctx, started=started)
    ctx = {**req.context, "request_id": rid}

    if req.item_type not in KINDS:
        return _error(rid, ERR_INVALID_REQUEST,
                      f"item_type '{req.item_type}' has no table in the schema. Supported: {sorted(KINDS)}. "
                      "Nothing was stored.", context=ctx, started=started)

    if req.operation != OP_RETIRE and not (isinstance(request, dict) and request.get("approved") is True):
        return _error(rid, ERR_INVALID_REQUEST,
                      "Item is not approved: send \"approved\": true (boolean) after the user approved it. "
                      "Nothing was stored.",
                      {"received_approved": request.get("approved") if isinstance(request, dict) else None},
                      context=ctx, started=started)

    extras: Dict[str, Any] = {}
    if req.operation != OP_RETIRE:
        try:
            extras = _extras(request, req, raw_ctx)
        except _BadExtra as e:
            return _error(rid, ERR_INVALID_REQUEST, f"{e}. Nothing was stored.", e.details, context=ctx, started=started)

    try:
        project_uuid = _parse_uuid(req.project_id)            # not a UUID -> cannot be a known project
        row = (agent.conn.execute("SELECT id, name FROM projects WHERE id = %s", (project_uuid,)).fetchone()
               if project_uuid else None)
        if not row:
            known = [str(r["id"]) for r in agent.conn.execute("SELECT id FROM projects ORDER BY id").fetchall()]
            return _error(rid, ERR_UNKNOWN_PROJECT, f"Unknown project_id: {req.project_id}",
                          {"available_project_ids": known}, context=ctx, started=started)
        req.project_id = str(row["id"])                       # canonical form (also used for the version ids)
        project = {"project_id": str(row["id"]), "name": row["name"]}

        if req.operation == OP_RETIRE:
            status, ref, warnings = _retire(agent.conn, req)
        else:
            _check_refs(agent.conn, req, extras)
            status, ref, warnings = _upsert(agent.conn, req, extras)
        for w in warnings:
            log.warning("put_item %s/%s: %s", req.item_type, req.item_id, w)
        if not WARNINGS_IN_RESPONSE:
            warnings = []
        return _envelope(rid, status, project=project, item=ref, warnings=warnings, context=ctx, started=started)
    except ItemNotFound as e:
        return _error(rid, ERR_ITEM_NOT_FOUND, str(e), context=ctx, started=started)
    except _BadExtra as e:
        return _error(rid, ERR_INVALID_REQUEST, f"{e}. Nothing was stored.", e.details, context=ctx, started=started)
    except Exception as e:  # noqa: BLE001 - never raise to the PO Agent
        _safe_rollback(agent.conn)
        return _error(rid, ERR_WRITE_FAILED, f"{type(e).__name__}: {e}. Nothing was stored.",
                      context=ctx, started=started)


def _upsert(conn, req: ItemRequest, ex: Dict[str, Any]):
    table = KINDS[req.item_type][0]
    warnings: List[str] = []
    versions = _history(conn, req)
    current = next((v for v in versions if v["active"]), None)

    if current and _same(conn, req, current, ex):
        return ITEM_UNCHANGED, _ref(req, current["version"], current["id"], None, "approved"), []

    new_version = (versions[0]["version"] if versions else 0) + 1
    new_id = _item_uuid(req.project_id, req.item_type, req.item_id, new_version)
    previous = current["id"] if current else None

    embedding = None
    if table in CHUNK_FK:                         # embed BEFORE writing: if Jina fails, nothing has been written
        vectors = get_jina_embeddings([req.text])
        if not vectors:
            raise RuntimeError("embedding service returned no vector")
        embedding = vectors[0]

    # One atomic transaction (conn.transaction(), never `with conn:`; the connection is on autocommit).
    with conn.transaction():
        _insert_typed(conn, req, new_id, ex, previous)
        if embedding is not None:
            _insert_chunk(conn, req, new_id, embedding, updated=bool(previous))
        if previous:
            _deactivate(conn, req, previous, warnings)
        _verify_new(conn, req, new_id)
        if previous:
            _verify_inactive(conn, req, previous)

    return (ITEM_UPDATED if previous else ITEM_STORED), \
        _ref(req, new_version, new_id, previous, "approved"), warnings


def _retire(conn, req: ItemRequest):
    table = KINDS[req.item_type][0]
    if table == "user_stories":
        raise _BadExtra("retire is not supported for user_story: user_stories has no retired/superseded status")
    versions = _history(conn, req)
    if not versions:
        raise ItemNotFound(f"No {req.item_type} with item_id '{req.item_id}' is stored for project {req.project_id}.")
    current = next((v for v in versions if v["active"]), None)
    if current is None:                           # already retired: idempotent
        return ITEM_UNCHANGED, _ref(req, versions[0]["version"], versions[0]["id"], None, "superseded"), []

    warnings: List[str] = []
    with conn.transaction():
        _deactivate(conn, req, current["id"], warnings, retire=True)
        _verify_inactive(conn, req, current["id"])
    return ITEM_RETIRED, _ref(req, current["version"], current["id"], None, "superseded"), warnings