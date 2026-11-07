from __future__ import annotations

import uuid
from datetime import datetime, timezone

from embed import get_jina_embeddings

DECISION_DOC_TYPE = "decision"
EVENT_DECISION_RECORDED = "DECISION_RECORDED"


def record_decision(conn, project_id: str, description: str,
                    supersedes_id: str | None = None) -> dict:
    description = (description or "").strip()
    if not description:
        raise ValueError("Decision description cannot be empty.")

    with conn.cursor() as cur:
        project_exists = cur.execute("SELECT 1 FROM projects WHERE id = %s", (project_id,)).fetchone()
        if not project_exists:
            raise KeyError(f"Unknown project_id: {project_id}")

        if supersedes_id:
            old = cur.execute("SELECT project_id FROM decisions WHERE id = %s", (supersedes_id,)).fetchone()
            if not old:
                raise KeyError(f"Unknown decision to supersede: {supersedes_id}")
            if old["project_id"] != project_id:
                raise ValueError("A decision can only supersede a decision from the same project.")
            already_superseded = cur.execute(
                "SELECT 1 FROM decisions WHERE supersedes_id = %s", (supersedes_id,)).fetchone()
            if already_superseded:
                raise ValueError(f"Decision {supersedes_id} has already been superseded.")

    decision_id = f"dec_{uuid.uuid4().hex[:8]}"
    chunk_id = f"{decision_id}_chunk"

    # Embed first: if Jina is down, nothing is written and the caller gets a clean error
    # (no decision row without a searchable chunk).
    embeddings = get_jina_embeddings([description])
    embedding = embeddings[0] if embeddings else None

    # One atomic transaction. NOTE: conn.transaction(), never `with conn:` (psycopg3 closes the connection).
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO decisions (id, project_id, description, approved, supersedes_id, created_at) "
                "VALUES (%s, %s, %s, 1, %s, %s)",
                (decision_id, project_id, description, supersedes_id,
                 datetime.now(timezone.utc).isoformat()),
            )

            cur.execute(
                """INSERT INTO chunks (id, project_id, document_id, decision_id, chunk_text,
                                       chunk_index, doc_type, approval_status,
                                       source_event, source_id, embedding)
                   VALUES (%s, %s, NULL, %s, %s, 0, %s, 'approved', %s, %s, %s::vector)""",
                (chunk_id, project_id, decision_id, description, DECISION_DOC_TYPE,
                 EVENT_DECISION_RECORDED, decision_id, embedding),
            )

            if supersedes_id:
                cur.execute(
                    "UPDATE chunks SET approval_status = 'superseded' "
                    "WHERE decision_id = %s AND id != %s",
                    (supersedes_id, chunk_id),
                )

    return {"decision_id": decision_id, "project_id": project_id,
            "superseded": supersedes_id, "warnings": []}