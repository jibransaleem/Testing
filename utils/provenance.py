"""
provenance.py  --  verify retrieved chunks against PostgreSQL and attach their source information.

Schema notes (normalized schema):
  * ids are UUIDs in the database; every id returned from here is a plain string.
  * `chunks` has no doc_type / approval_status / source_id columns. They are derived by the
    v_chunks view, so verification reads from v_chunks, not from chunks.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from .version_aware import retrieve_version_aware

CURRENT_ONLY = ("approved",)
WITH_HISTORY = ("approved", "superseded")

_TYPE_LABELS = {
    "requirement": "Requirement",
    "user_story": "User story",
    "meeting": "Meeting",
    "meeting_summary": "Meeting",
    "standup": "Standup",
    "retro_feedback": "Retro feedback",
    "action_item": "Action item",
    "decision": "Decision",
    "approval_rationale": "Approval rationale",
    "refinement_note": "Refinement note",
    "sync_record": "Sync record",
    "glossary": "Glossary",
}


def _s(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _uuid(value: Any) -> Optional[uuid.UUID]:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def enrich_with_provenance(
    conn,
    hits: list,
    project_id: str | None = None,
    allowed_statuses: tuple = CURRENT_ONLY,
) -> list:
    """Returns hits in the same order, each with provenance_valid / invalid_reason / source.
    A chunk is valid only if it exists, belongs to project_id and has an allowed approval status."""
    if not hits:
        return []

    ids = [u for u in (_uuid(h["chunk_id"]) for h in hits) if u is not None]
    rows = conn.execute(
        """SELECT c.id              AS chunk_id,
                  c.chunk_text      AS chunk_text,
                  c.section_label   AS section_label,
                  c.doc_type        AS doc_type,
                  c.approval_status AS approval_status,
                  c.project_id      AS chunk_project,
                  c.page_start      AS page_start,
                  c.page_end        AS page_end,
                  c.decision_id     AS decision_id,
                  c.source_event    AS source_event,
                  c.source_id       AS source_id,
                  d.title           AS doc_title,
                  d.id              AS doc_id,
                  d.version         AS doc_version,
                  d.status          AS doc_status,
                  p.name            AS project_name
           FROM v_chunks c
           LEFT JOIN documents d ON d.id = c.document_id
           JOIN projects p ON p.id = c.project_id
           WHERE c.id = ANY(%s)""",
        (ids,),
    ).fetchall() if ids else []
    by_id = {str(r["chunk_id"]): r for r in rows}

    wanted_project = _s(_uuid(project_id)) if project_id else None
    if project_id and wanted_project is None:
        wanted_project = str(project_id)                      # not a UUID: will simply never match

    enriched = []
    for h in hits:
        cid = str(h["chunk_id"])
        row = by_id.get(cid)
        if row is None:
            enriched.append({"chunk_id": cid, "text": h.get("text"), "score": h.get("score"),
                             "provenance_valid": False,
                             "invalid_reason": "chunk not found in database", "source": None})
            continue

        reason = None
        if wanted_project and str(row["chunk_project"]) != wanted_project:
            reason = "chunk belongs to a different project"
        elif row["approval_status"] not in allowed_statuses:
            reason = f"chunk status '{row['approval_status']}' not allowed for this query"

        enriched.append({
            "chunk_id": cid,
            "text": row["chunk_text"],
            "score": h.get("score"),
            "provenance_valid": reason is None,
            "invalid_reason": reason,
            "source": {
                "project": row["project_name"],
                "project_id": _s(row["chunk_project"]),
                "document": row["doc_title"],              # None for decision / meeting chunks
                "document_id": _s(row["doc_id"]),
                "document_version": row["doc_version"],
                "document_status": row["doc_status"],
                "section": row["section_label"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "doc_type": row["doc_type"],
                "approval_status": row["approval_status"],
                "decision_id": _s(row["decision_id"]),
                "source_event": row["source_event"],
                "source_id": _s(row["source_id"]),
            },
        })
    return enriched


def format_citation(hit: dict) -> str:
    """
    'Career Compass (v2), 3.2 Authentication, p.5'   (from a document or requirement)
    'Decision, Hosting'                              (decision / meeting: label + its title)
    """
    if not hit.get("provenance_valid"):
        return f"[UNRESOLVED chunk_id={hit['chunk_id']}]"
    s = hit["source"]

    if s.get("document"):
        version = s.get("document_version")
        head = f"{s['document']} (v{version})" if version else s["document"]
        parts = [head, s["section"] or "n/a"]
    else:
        label = _TYPE_LABELS.get(s.get("doc_type"), "Note")
        parts = [label]
        if s.get("section"):
            parts.append(s["section"])
        elif s.get("source_id"):
            parts.append(str(s["source_id"])[:8])

    if s.get("page_start"):
        p0, p1 = s["page_start"], s.get("page_end") or s["page_start"]
        parts.append(f"p.{p0}" if p0 == p1 else f"pp.{p0}-{p1}")
    if s["approval_status"] == "superseded":
        parts.append("SUPERSEDED")
    return ", ".join(parts)


def retrieve_with_provenance(conn, project_id: str, question: str,
                             top_k: int = 5, include_history: bool = False,
                             target_versions=None, min_score: float = 0.0,
                             doc_types=None):
    """Project isolation + version-aware filtering + SQL provenance verification."""
    hits = retrieve_version_aware(conn, project_id, question, top_k,
                                  include_history, target_versions, min_score, doc_types)
    allowed = WITH_HISTORY if (include_history or target_versions) else CURRENT_ONLY
    return enrich_with_provenance(conn, hits, project_id, allowed)