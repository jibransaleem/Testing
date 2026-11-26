"""
version_aware.py  --  project-isolated, version-aware vector retrieval.

Schema notes (normalized schema):
- ids are UUIDs; ids returned from here are plain strings.
- doc_type / approval_status come from the v_chunks view, not from the chunks table.
- decisions and meetings live in their own tables, not in `documents`.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from ingestion.embed import get_jina_embeddings
from ingestion.ingest import normalize_version

_DOC_TYPE_ALIASES = {
    "meeting_summary": ("meeting_summary", "meeting"),
    "meeting": ("meeting", "meeting_summary"),
}


def _uuid(value) -> Optional[uuid.UUID]:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _vector_literal(vec) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _expand_doc_types(doc_types: Optional[List[str]]) -> Optional[List[str]]:
    if not doc_types:
        return None
    expanded: List[str] = []
    for dt in doc_types:
        for candidate in _DOC_TYPE_ALIASES.get(dt, (dt,)):
            if candidate not in expanded:
                expanded.append(candidate)
    return expanded


def resolve_version_documents(conn, project_id: str, requested: List[str]) -> List[uuid.UUID]:
    """
    Maps requested versions ('1', '2.1') to document ids of the project.
    Exact match first; a bare major ('1') also matches '1.x' if no exact one exists.
    """
    pid = _uuid(project_id)
    if pid is None:
        return []
    docs = conn.execute(
        "SELECT id, version FROM documents WHERE project_id = %s", (pid,)).fetchall()

    doc_ids = []
    for want in requested:
        want_n = normalize_version(want)
        exact = [d["id"] for d in docs if normalize_version(d["version"]) == want_n]
        if exact:
            doc_ids.extend(exact)
        elif "." not in want_n:
            doc_ids.extend(d["id"] for d in docs if normalize_version(d["version"]).split(".")[0] == want_n)
    return sorted(set(doc_ids), key=str)


def retrieve_version_aware(
    conn,
    project_id: str,
    question: str,
    top_k: int = 5,
    include_history: bool = False,
    target_versions: Optional[List[str]] = None,
    min_score: float = 0.0,
    doc_types: Optional[List[str]] = None,
) -> list[dict]:
    pid = _uuid(project_id)
    if pid is None:
        return []

    query_embeddings = get_jina_embeddings([question])
    if not query_embeddings:
        return []
    query_vector = _vector_literal(query_embeddings[0])

    where_clauses = ["project_id = %s", "embedding IS NOT NULL"]
    where_params: list = [pid]

    if target_versions:
        doc_ids = resolve_version_documents(conn, str(pid), target_versions)
        if not doc_ids:
            return []
        where_clauses.append("document_id = ANY(%s)")
        where_params.append(doc_ids)
        where_clauses.append("approval_status IN ('approved', 'superseded')")
    elif include_history:
        where_clauses.append("approval_status IN ('approved', 'superseded')")
    else:
        where_clauses.append("approval_status = 'approved'")

    expanded_doc_types = _expand_doc_types(doc_types)
    if expanded_doc_types:
        where_clauses.append("doc_type = ANY(%s)")
        where_params.append(expanded_doc_types)

    score_floor = 0.0 if min_score is None else float(min_score)

    # Placeholder order: SELECT vector, WHERE params, ORDER BY vector, LIMIT
    params = [query_vector] + where_params + [query_vector, top_k]

    sql = f"""
        SELECT
            id,
            chunk_text,
            1 - (embedding <=> %s::vector) AS similarity
        FROM v_chunks
        WHERE {" AND ".join(where_clauses)}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    rows = conn.execute(sql, tuple(params)).fetchall()

    hits = []
    for row in rows:
        score = row["similarity"] if row["similarity"] is not None else 0.0
        if score < score_floor:
            continue
        hits.append({
            "chunk_id": str(row["id"]),
            "text": row["chunk_text"],
            "score": float(score),
        })

    return hits