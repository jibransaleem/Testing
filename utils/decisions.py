# This function records a new project decision in the database and generates its Jina embedding for AI-based semantic search.

# If it replaces an older decision, the old decision is kept (never deleted or overwritten), preserving the history.
# The function ensures that the project exists, the superseded decision (if any) belongs to the same project,
# and that it hasn't already been superseded. It returns a dictionary containing the new decision's ID,
# project ID, and any warnings.
#
# Schema note (normalized schema):
#   - ids are UUIDs, generated here
#   - chunks no longer stores doc_type / approval_status / source_id; they are derived by the v_chunks view
#   - "superseded" is therefore not written anywhere: a decision counts as superseded as soon as another
#     decision points at it through decisions.supersedes_id (the view works this out)

from __future__ import annotations

import uuid

import psycopg.errors

from ingestion.embed import get_jina_embeddings

DECISION_DOC_TYPE = "decision"   # kept for other modules that import it; v_chunks derives the same value
EVENT_DECISION_RECORDED = "DECISION_RECORDED"


def _as_uuid(value, what: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except ValueError:
        raise ValueError(f"Invalid {what}: {value!r} (expected a UUID).") from None


def record_decision(conn, project_id, description: str, supersedes_id=None) -> dict:
    description = (description or "").strip()
    if not description:
        raise ValueError("Decision description cannot be empty.")

    project_id = _as_uuid(project_id, "project_id")
    supersedes_id = _as_uuid(supersedes_id, "supersedes_id") if supersedes_id else None

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
            # supersedes_id is the ID of the old decision that the new decision will replace.
            already_superseded = cur.execute(
                "SELECT 1 FROM decisions WHERE supersedes_id = %s", (supersedes_id,)).fetchone()
            if already_superseded:
                raise ValueError(f"Decision {supersedes_id} has already been superseded.")

    decision_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    # Embed first: if Jina is down, nothing is written and the caller gets a clean error
    # (no decision row without a searchable chunk).
    embeddings = get_jina_embeddings([description])
    embedding = embeddings[0] if embeddings else None

    # One atomic transaction. NOTE: conn.transaction(), never `with conn:` (psycopg3 closes the connection).
    # 1. INSERT into decisions: saves the new decision as a new row (doesn't overwrite the old one).
    #    created_at is filled by the column default (now()).
    # 2. INSERT into chunks: saves the new decision's text and embedding so AI can search it.
    # 3. Nothing to update on the old decision: v_chunks marks it 'superseded' because the new row's
    #    supersedes_id now points at it.
    try:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO decisions (id, project_id, description, approved, supersedes_id) "
                    "VALUES (%s, %s, %s, true, %s)",
                    (decision_id, project_id, description, supersedes_id),
                )

                cur.execute(
                    """INSERT INTO chunks (id, project_id, decision_id, chunk_text,
                                           chunk_index, source_event, embedding)
                       VALUES (%s, %s, %s, %s, 0, %s, %s::vector)""",
                    (chunk_id, project_id, decision_id, description,
                     EVENT_DECISION_RECORDED, embedding),
                )
    except psycopg.errors.UniqueViolation:
        # Two callers superseded the same decision at once (uq_decisions_supersedes)
        if supersedes_id:
            raise ValueError(f"Decision {supersedes_id} has already been superseded.") from None
        raise

    return {"decision_id": str(decision_id), "project_id": str(project_id),
            "superseded": str(supersedes_id) if supersedes_id else None, "warnings": []}