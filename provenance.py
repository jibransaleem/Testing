from __future__ import annotations

from version_aware import retrieve_version_aware

CURRENT_ONLY = ("approved",)
WITH_HISTORY = ("approved", "superseded")

# Human label for chunks that have no document to cite (see db_setup.CHUNK_DOC_TYPES).
_TYPE_LABELS = {
    "requirement": "Requirement",
    "user_story": "User story",
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


def enrich_with_provenance(
    conn,
    hits: list,
    project_id: str | None = None,
    allowed_statuses: tuple = CURRENT_ONLY,
) -> list:
    """Returns hits in the same order, each with provenance_valid / invalid_reason / source."""
    if not hits:
        return []

    ids = [h["chunk_id"] for h in hits]
    rows = conn.execute(
        """SELECT c.id            AS chunk_id,
                  c.chunk_text    AS chunk_text,
                  c.section_label AS section_label,
                  c.doc_type      AS doc_type,
                  c.approval_status AS approval_status,
                  c.project_id    AS chunk_project,
                  c.page_start    AS page_start,
                  c.page_end      AS page_end,
                  c.decision_id   AS decision_id,
                  c.source_event  AS source_event,
                  c.source_id     AS source_id,
                  d.title         AS doc_title,
                  d.id            AS doc_id,
                  d.version       AS doc_version,
                  d.status        AS doc_status,
                  p.name          AS project_name
           FROM chunks c
           LEFT JOIN documents d ON c.document_id = d.id
           JOIN projects p ON c.project_id = p.id
           WHERE c.id = ANY(%s)""",
        (ids,),
    ).fetchall()
    by_id = {r["chunk_id"]: r for r in rows}

    enriched = []
    for h in hits:
        row = by_id.get(h["chunk_id"])
        if row is None:
            enriched.append({**h, "provenance_valid": False,
                             "invalid_reason": "chunk not found in database", "source": None})
            continue

        reason = None
        if project_id and row["chunk_project"] != project_id:
            reason = "chunk belongs to a different project"
        elif row["approval_status"] not in allowed_statuses:
            reason = f"chunk status '{row['approval_status']}' not allowed for this query"

        enriched.append({
            "chunk_id": row["chunk_id"],
            "text": row["chunk_text"],
            "score": h.get("score"),
            "provenance_valid": reason is None,
            "invalid_reason": reason,
            "source": {
                "project": row["project_name"],
                "project_id": row["chunk_project"],
                "document": row["doc_title"],          # None for non-document chunks
                "document_id": row["doc_id"],
                "document_version": row["doc_version"],
                "document_status": row["doc_status"],
                "section": row["section_label"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "doc_type": row["doc_type"],
                "approval_status": row["approval_status"],
                "decision_id": row["decision_id"],
                "source_event": row["source_event"],
                "source_id": row["source_id"],
            },
        })
    return enriched


def format_citation(hit: dict) -> str:
    """
    'Career Compass (v2), 3.2 Authentication, p.5'   (from a document)
    'User story us_12'                               (not from a document)
    """
    if not hit.get("provenance_valid"):
        return f"[UNRESOLVED chunk_id={hit['chunk_id']}]"
    s = hit["source"]

    if s.get("document"):
        head = s["document"] if s.get("doc_type") == "decision" else f"{s['document']} (v{s['document_version']})"
        parts = [head, s["section"] or "n/a"]
    else:
        label = _TYPE_LABELS.get(s.get("doc_type"), "Note")
        parts = [f"{label} {s['source_id']}" if s.get("source_id") else label]
        if s.get("section"):
            parts.append(s["section"])

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