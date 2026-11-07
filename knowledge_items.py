"""
knowledge_items.py  --  Knowledge Agent side: STORE approved items sent by the PO Agent

    put_item(agent, request_dict) -> ItemResponse dict        (never raises)

RULE: only APPROVED items are written.  The PO Agent must send  "approved": true  (the boolean true,
after the user approved the item).  Anything else is rejected and NOTHING is stored.
(Retire requests do not carry content, so they do not need the flag.)

What happens to an item (project_id + item_type + item_id identify it)
    first time            -> stored      version 1 (approved)
    same content again    -> unchanged   (safe to retry; nothing written)
    changed content       -> updated     new version approved, previous version -> 'superseded' (kept as history)
    operation 'retire'    -> retired     current version -> 'superseded', no successor
Superseded versions never appear in normal answers; they are returned only when history is requested.

WHERE IT IS STORED
    every item                -> table `chunks`  (search index + version ledger; text, status, pgvector embedding)
    item_type meeting_summary -> ALSO table `meetings`
    item_type user_story      -> ALSO table `user_stories`
  The typed row uses the SAME id as the chunk row (version-specific chunk_id), so they are linked.
  All rows of one write (chunk + typed row + status flip of the previous version) are written in ONE
  transaction and READ BACK before it ends: if anything is missing the whole write is rolled back and the
  caller gets WRITE_FAILED, so "stored" always means "really in Postgres".

  meetings        id=chunk_id | project_id | date=context.meeting_date (or context.date / request date)
                  summary=req.text | meeting_type=req.title | sprint_id=context.sprint_id | author=context.author
  user_stories    id=chunk_id | project_id | requirement_id=context.requirement_id | title=req.title
                  story_text=req.text | acceptance_criteria=context.acceptance_criteria
                  status='approved' | created_at=<utc time>
  Foreign keys are checked BEFORE writing: context.sprint_id must exist in `sprints` (same project) and
  context.requirement_id in `requirements` (same project), otherwise the request is rejected (nothing stored).
  If a typed table has a `status` column (user_stories has, meetings has not) it is flipped to
  'superseded' on update/retire as well.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import date as _date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from psycopg.types.json import Jsonb

from embed import get_jina_embeddings
from po_knowledge_contract import (
    ERR_INVALID_REQUEST, ERR_ITEM_NOT_FOUND, ERR_UNKNOWN_PROJECT, ERR_WRITE_FAILED,
    ITEM_RETIRED, ITEM_STORED, ITEM_UNCHANGED, ITEM_UPDATED, OP_RETIRE, SCHEMA_VERSION, STATUS_ERROR,
    InvalidRequest, ItemRequest,
)

log = logging.getLogger("knowledge_items")

# The response contract (po_knowledge_contract.validate_item_response, checked in KnowledgeInterface AFTER the
# write) may not accept free-text warnings. Keep False: notes go to the server log, `warnings` stays [].
# Set True only if the contract allows string warnings.
WARNINGS_IN_RESPONSE = False

EVENT_PO_ITEM_SUBMITTED = "PO_ITEM_SUBMITTED"
EVENT_PO_ITEM_UPDATED = "PO_ITEM_UPDATED"

# item_type -> typed table (everything else is stored in `chunks` only)
TYPED_TABLES = {"meeting_summary": "meetings", "user_story": "user_stories"}
# columns compared to decide "unchanged" vs "new version" (id / project_id / status are not compared)
_COMPARE = {
    "meeting_summary": ("date", "summary", "meeting_type", "sprint_id", "author"),
    "user_story": ("requirement_id", "title", "story_text", "acceptance_criteria"),
}


class ItemNotFound(LookupError):
    pass


class _BadExtra(ValueError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _as_text(v: Any) -> str:
    """Canonical text of any value (used to store into text columns and to compare)."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return "\n".join(_as_text(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    return str(v)


def _envelope(rid, status, *, project=None, item=None, warnings=None, error=None, context=None, started=0.0):
    return {"schema_version": SCHEMA_VERSION, "request_id": rid, "status": status, "project": project,
            "item": item, "warnings": warnings or [], "error": error, "context": context or {},
            "meta": {"latency_ms": int((time.perf_counter() - started) * 1000) if started else 0,
                     "generated_at": _now()}}


def _error(rid, code, message, details=None, **kw):
    return _envelope(rid, STATUS_ERROR, error={"code": code, "message": message, "details": details or {}}, **kw)


def _chunk_id(project_id: str, item_type: str, item_id: str, version: int) -> str:
    return f"{project_id}__{item_type}__{item_id}__v{version}"


def _version_of(chunk_id: str) -> int:
    m = re.search(r"__v(\d+)$", chunk_id)
    return int(m.group(1)) if m else 1


def _ref(req: ItemRequest, chunk_id: str, previous: Optional[str], status: str) -> Dict[str, Any]:
    return {"item_type": req.item_type, "item_id": req.item_id, "version": _version_of(chunk_id),
            "chunk_id": chunk_id, "previous_chunk_id": previous, "approval_status": status}


# ---------------------------------------------------------------------------------------- typed tables
_COLS: Dict[str, Dict[str, str]] = {}          # table -> {column: data_type}   (read once per process)


def _columns(conn, table: str) -> Dict[str, str]:
    cols = _COLS.get(table)
    if cols is None:
        rows = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s", (table,)).fetchall()
        cols = {r["column_name"]: r["data_type"] for r in rows}
        if not cols:
            raise RuntimeError(f"table '{table}' was not found in the database")
        _COLS[table] = cols
    return cols


def _pick(merged: Dict[str, Any], request: Any, *names: str) -> Any:
    """First non-empty value among context keys, then top-level request keys."""
    for n in names:
        for src in (merged, request if isinstance(request, dict) else {}):
            v = src.get(n)
            if v is not None and v != "":
                return v
    return None


def _extras(request: Any, req: ItemRequest, raw_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Values for the typed tables that come from the request context."""
    merged = {**raw_ctx, **(req.context or {})}
    ex = {
        "meeting_date": _pick(merged, request, "meeting_date", "date"),
        "sprint_id": _pick(merged, request, "sprint_id"),
        "author": _pick(merged, request, "author"),
        "requirement_id": _pick(merged, request, "requirement_id"),
        "acceptance_criteria": _pick(merged, request, "acceptance_criteria"),
    }
    if req.item_type == "user_story" and not _norm(req.title):
        raise _BadExtra("title is required for a user_story")
    if ex["meeting_date"] is not None:
        try:
            ex["meeting_date"] = _date.fromisoformat(str(ex["meeting_date"])[:10]).isoformat()
        except ValueError:
            raise _BadExtra(f"meeting date must be YYYY-MM-DD, got {ex['meeting_date']!r}")
    return ex


def _check_refs(conn, req: ItemRequest, ex: Dict[str, Any]) -> None:
    """sprint_id / requirement_id are foreign keys: reject with a clear message instead of a DB error."""
    checks = []
    if req.item_type == "meeting_summary" and ex.get("sprint_id") is not None:
        checks.append(("sprint_id", "sprints", str(ex["sprint_id"])))
    if req.item_type == "user_story" and ex.get("requirement_id") is not None:
        checks.append(("requirement_id", "requirements", str(ex["requirement_id"])))
    for field_name, table, value in checks:
        if conn.execute(f"SELECT 1 FROM {table} WHERE id = %s AND project_id = %s",
                        (value, req.project_id)).fetchone():
            continue
        known = [r["id"] for r in conn.execute(
            f"SELECT id FROM {table} WHERE project_id = %s ORDER BY id LIMIT 20", (req.project_id,)).fetchall()]
        raise _BadExtra(f"{field_name} '{value}' does not exist in table '{table}' for project {req.project_id}",
                        {f"available_{table}_ids (max 20)": known})


def _typed_values(req: ItemRequest, chunk_id: str, ex: Dict[str, Any]) -> Dict[str, Any]:
    if req.item_type == "meeting_summary":
        return {"id": chunk_id, "project_id": req.project_id, "date": ex["meeting_date"],
                "summary": req.text, "meeting_type": req.title,
                "sprint_id": ex["sprint_id"], "author": ex["author"]}
    return {"id": chunk_id, "project_id": req.project_id, "requirement_id": ex["requirement_id"],
            "title": req.title, "story_text": req.text,
            "acceptance_criteria": ex["acceptance_criteria"], "status": "approved"}


def _adapt(value: Any, data_type: str) -> Any:
    """Convert a Python value to what the column type needs."""
    if value is None:
        return None
    if data_type in ("json", "jsonb"):
        return Jsonb(value)
    if data_type == "ARRAY":
        return list(value) if isinstance(value, (list, tuple)) else [s for s in str(value).splitlines() if s.strip()]
    if data_type in ("text", "character varying", "character"):
        return _as_text(value)
    return value


def _insert_typed(conn, req: ItemRequest, chunk_id: str, ex: Dict[str, Any]) -> None:
    table = TYPED_TABLES[req.item_type]
    types = _columns(conn, table)
    cols, placeholders, params = [], [], []
    for col, val in _typed_values(req, chunk_id, ex).items():
        if col == "meeting_type" and val is None:             # keep the column default ('meeting')
            continue
        if col not in types:
            raise RuntimeError(f"column '{col}' was not found in table '{table}'")
        cols.append(f'"{col}"')
        placeholders.append("%s::date" if (col == "date" and types[col] == "date") else "%s")
        params.append(_adapt(val, types[col]))
    if req.item_type == "user_story" and "created_at" in types:
        cols.append('"created_at"')
        placeholders.append("%s")
        params.append(_now())
    conn.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(placeholders)})", params)


def _typed_row(conn, req: ItemRequest, chunk_id: str) -> Optional[Dict[str, Any]]:
    table = TYPED_TABLES[req.item_type]
    _columns(conn, table)                                    # clear error if the table is missing
    return conn.execute(f"SELECT * FROM {table} WHERE id = %s", (chunk_id,)).fetchone()


def _typed_differs(row: Dict[str, Any], req: ItemRequest, chunk_id: str, ex: Dict[str, Any]) -> bool:
    new = _typed_values(req, chunk_id, ex)
    for col in _COMPARE[req.item_type]:
        a, b = row.get(col), new.get(col)
        if col in ("summary", "story_text", "title", "meeting_type"):
            same = _norm(_as_text(a)) == _norm(_as_text(b))
        else:
            same = _as_text(a).strip() == _as_text(b).strip()
        if not same:
            return True
    return False


def _mirror_status(conn, req: ItemRequest, chunk_id: str, status: str) -> bool:
    """Set `status` on the typed row if that table has such a column. Returns False if it has none."""
    table = TYPED_TABLES[req.item_type]
    if "status" not in _columns(conn, table):
        return False
    conn.execute(f"UPDATE {table} SET status = %s WHERE id = %s", (status, chunk_id))
    return True


def _verify_typed(conn, req: ItemRequest, chunk_id: str) -> None:
    if _typed_row(conn, req, chunk_id) is None:
        raise RuntimeError(f"write could not be verified: no row in '{TYPED_TABLES[req.item_type]}'")


# ----------------------------------------------------------------------------------------------------------
def put_item(agent, request: Any) -> Dict[str, Any]:
    started = time.perf_counter()
    raw_ctx = request.get("context") if isinstance(request, dict) and isinstance(request.get("context"), dict) else {}
    rid = str(raw_ctx.get("request_id") or f"req_{uuid.uuid4().hex[:12]}")
    try:
        req = ItemRequest.from_dict(request)
    except InvalidRequest as e:
        return _error(rid, ERR_INVALID_REQUEST, str(e), context=raw_ctx, started=started)
    ctx = {**req.context, "request_id": rid}

    # --- approval gate: only items the user APPROVED may be written (strict boolean true) -----------------
    if req.operation != OP_RETIRE and not (isinstance(request, dict) and request.get("approved") is True):
        return _error(rid, ERR_INVALID_REQUEST,
                      "Item is not approved: send \"approved\": true (boolean) after the user approved it. "
                      "Nothing was stored.",
                      {"received_approved": request.get("approved") if isinstance(request, dict) else None},
                      context=ctx, started=started)

    extras: Dict[str, Any] = {}
    if req.operation != OP_RETIRE and req.item_type in TYPED_TABLES:
        try:
            extras = _extras(request, req, raw_ctx)
        except _BadExtra as e:
            return _error(rid, ERR_INVALID_REQUEST, f"{e}. Nothing was stored.", e.details, context=ctx, started=started)

    try:
        row = agent.conn.execute("SELECT id, name FROM projects WHERE id = %s", (req.project_id,)).fetchone()
        if not row:
            known = [r["id"] for r in agent.conn.execute("SELECT id FROM projects ORDER BY id").fetchall()]
            return _error(rid, ERR_UNKNOWN_PROJECT, f"Unknown project_id: {req.project_id}",
                          {"available_project_ids": known}, context=ctx, started=started)
        project = {"project_id": row["id"], "name": row["name"]}

        if req.operation == OP_RETIRE:
            status, ref, warnings = _retire(agent.conn, req)
        else:
            if req.item_type in TYPED_TABLES:
                _check_refs(agent.conn, req, extras)
            status, ref, warnings = _upsert(agent.conn, req, extras)
        if warnings:
            for w in warnings:
                log.warning("put_item %s/%s: %s", req.item_type, req.item_id, w)
                print(f"[knowledge_items] note for {req.item_type}/{req.item_id}: {w}")
            if not WARNINGS_IN_RESPONSE:
                warnings = []
        return _envelope(rid, status, project=project, item=ref, warnings=warnings, context=ctx, started=started)
    except ItemNotFound as e:
        return _error(rid, ERR_ITEM_NOT_FOUND, str(e), context=ctx, started=started)
    except _BadExtra as e:
        return _error(rid, ERR_INVALID_REQUEST, f"{e}. Nothing was stored.", e.details, context=ctx, started=started)
    except Exception as e:  # noqa: BLE001 - never raise to the PO Agent
        return _error(rid, ERR_WRITE_FAILED, f"{type(e).__name__}: {e}. Nothing was stored.",
                      context=ctx, started=started)


def _history(conn, req: ItemRequest) -> List[Tuple[str, str, Optional[str], str]]:
    """All versions of this item, newest first: (id, text, title, status)."""
    rows = conn.execute(
        "SELECT id, chunk_text, section_label, approval_status FROM chunks "
        "WHERE project_id = %s AND doc_type = %s AND source_id = %s",
        (req.project_id, req.item_type, req.item_id)).fetchall()
    versions = [(r["id"], r["chunk_text"], r["section_label"], r["approval_status"]) for r in rows]
    return sorted(versions, key=lambda v: _version_of(v[0]), reverse=True)


def _upsert(conn, req: ItemRequest, extras: Dict[str, Any]):
    typed = req.item_type in TYPED_TABLES
    warnings: List[str] = []
    versions = _history(conn, req)
    current = next((v for v in versions if v[3] == "approved"), None)

    if current and _norm(current[1]) == _norm(req.text) and _norm(current[2]) == _norm(req.title):
        if not typed:
            return ITEM_UNCHANGED, _ref(req, current[0], None, "approved"), []
        table = TYPED_TABLES[req.item_type]
        trow = _typed_row(conn, req, current[0])
        if trow is None:                                       # old item stored before typed tables: backfill
            with conn.transaction():
                _insert_typed(conn, req, current[0], extras)
                _verify_typed(conn, req, current[0])
            return ITEM_UNCHANGED, _ref(req, current[0], None, "approved"), \
                [f"copied the existing version into table '{table}'"]
        if not _typed_differs(trow, req, current[0], extras):
            return ITEM_UNCHANGED, _ref(req, current[0], None, "approved"), []
        # same text but context fields (date, sprint, author, criteria...) changed -> new version below

    new_version = (_version_of(versions[0][0]) if versions else 0) + 1
    new_id = _chunk_id(req.project_id, req.item_type, req.item_id, new_version)
    previous = current[0] if current else None

    # Embed BEFORE writing: if Jina fails, nothing has been written.
    embeddings = get_jina_embeddings([req.text])
    if not embeddings:
        raise RuntimeError("embedding service returned no vector")

    # One atomic transaction (conn.transaction(), never `with conn:`; the connection is on autocommit).
    with conn.transaction():
        conn.execute(
            """INSERT INTO chunks (id, project_id, document_id, decision_id, chunk_text, chunk_index,
                                   section_label, doc_type, approval_status, source_event, source_id, embedding)
               VALUES (%s, %s, NULL, NULL, %s, 0, %s, %s, 'approved', %s, %s, %s::vector)""",
            (new_id, req.project_id, req.text, req.title, req.item_type,
             EVENT_PO_ITEM_UPDATED if previous else EVENT_PO_ITEM_SUBMITTED, req.item_id, embeddings[0]))
        if typed:
            _insert_typed(conn, req, new_id, extras)
        if previous:
            conn.execute("UPDATE chunks SET approval_status = 'superseded' WHERE id = %s", (previous,))
            if typed and not _mirror_status(conn, req, previous, "superseded"):
                warnings.append(f"table '{TYPED_TABLES[req.item_type]}' has no status column: the older version "
                                f"stays as a row; the newest id is the current one")

        # Read-back verification INSIDE the transaction: if this fails the whole write is rolled back.
        check = conn.execute(
            "SELECT approval_status, (embedding IS NOT NULL) AS has_embedding FROM chunks WHERE id = %s",
            (new_id,)).fetchone()
        if not check or check["approval_status"] != "approved" or not check["has_embedding"]:
            raise RuntimeError("write could not be verified in the database (row missing or no embedding)")
        if typed:
            _verify_typed(conn, req, new_id)
        if previous:
            old = conn.execute("SELECT approval_status FROM chunks WHERE id = %s", (previous,)).fetchone()
            if not old or old["approval_status"] != "superseded":
                raise RuntimeError("previous version was not marked superseded")

    return (ITEM_UPDATED if previous else ITEM_STORED), _ref(req, new_id, previous, "approved"), warnings


def _retire(conn, req: ItemRequest):
    versions = _history(conn, req)
    if not versions:
        raise ItemNotFound(f"No {req.item_type} with item_id '{req.item_id}' is stored for project {req.project_id}.")
    current = next((v for v in versions if v[3] == "approved"), None)
    if current is None:                                        # already retired: idempotent
        return ITEM_UNCHANGED, _ref(req, versions[0][0], None, "superseded"), []

    warnings: List[str] = []
    with conn.transaction():
        conn.execute("UPDATE chunks SET approval_status = 'superseded' WHERE id = %s", (current[0],))
        if req.item_type in TYPED_TABLES and not _mirror_status(conn, req, current[0], "superseded"):
            warnings.append(f"table '{TYPED_TABLES[req.item_type]}' has no status column: the row was kept, "
                            f"it is retired only in the search index (chunks)")

    return ITEM_RETIRED, _ref(req, current[0], None, "superseded"), warnings