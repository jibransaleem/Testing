from __future__ import annotations

from typing import List, Optional

from embed import get_jina_embeddings
from ingest import normalize_version


def resolve_version_documents(conn, project_id: str, requested: List[str]) -> List[str]:
    """
    Maps requested versions ('1', '2.1') to document ids of the project.
    Exact match first; a bare major ('1') also matches '1.x' if no exact one exists.
    """
    with conn.cursor() as cur:
        docs = cur.execute(
            "SELECT id, version FROM documents WHERE project_id = %s AND doc_type != 'decision'",
            (project_id,),
        ).fetchall()

    doc_ids = []
    for want in requested:
        want_n = normalize_version(want)
        exact = [d["id"] for d in docs if normalize_version(d["version"]) == want_n]
        if exact:
            doc_ids.extend(exact)
        elif "." not in want_n:
            doc_ids.extend(d["id"] for d in docs if normalize_version(d["version"]).split(".")[0] == want_n)
    return sorted(set(doc_ids))


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
    # 1. Query embedding (Jina v3, 1024-dim)
    query_embeddings = get_jina_embeddings([question])
    if not query_embeddings:
        return []
    query_vector = query_embeddings[0]

    # 2. WHERE filters (where_params must stay in the same order as the %s in where_clauses)
    where_clauses = ["project_id = %s", "embedding IS NOT NULL"]
    where_params: list = [project_id]

    if target_versions:
        doc_ids = resolve_version_documents(conn, project_id, target_versions)
        if not doc_ids:
            return []
        where_clauses.append("document_id = ANY(%s)")
        where_params.append(doc_ids)
        where_clauses.append("approval_status IN ('approved', 'superseded')")
    elif include_history:
        where_clauses.append("approval_status IN ('approved', 'superseded')")
    else:
        where_clauses.append("approval_status = 'approved'")

    if doc_types:
        where_clauses.append("doc_type = ANY(%s)")
        where_params.append(list(doc_types))

    # 3. Placeholder order in the SQL is:
    #    SELECT vector, then WHERE params, then ORDER BY vector, then LIMIT
    params = [query_vector] + where_params + [query_vector, top_k]

    sql = f"""
        SELECT
            id,
            chunk_text,
            1 - (embedding <=> %s::vector) AS similarity
        FROM chunks
        WHERE {" AND ".join(where_clauses)}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    with conn.cursor() as cur:
        rows = cur.execute(sql, tuple(params)).fetchall()

    hits = []
    for row in rows:
        score = row["similarity"] if row["similarity"] is not None else 0.0
        if score < min_score:
            continue
        hits.append({
            "chunk_id": row["id"],
            "text": row["chunk_text"],
            "score": float(score),
        })

    return hits