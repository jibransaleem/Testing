# This file manages the embedding generation and synchronization of chunks stored in PostgreSQL
# Uses Jina AI Embeddings v3 to convert chunk text into 1024-dimensional vectors.
# Processes chunks in batches of 32 to avoid sending all texts at once.
# Finds chunks with missing embeddings and updates their vectors in PostgreSQL.
# Provides reconcile() to check embedding consistency and optionally repair missing embeddings.

# 1. Save chunks in PostgreSQL with their text and embedding = NULL (if embeddings haven't been generated yet).
# 2. Generate embeddings by sending chunk text to Jina AI.
# 3. Update PostgreSQL by saving the generated vectors in the corresponding chunk rows.
# # So, chunks are stored first, and embeddings are generated and saved afterward.

# Chunks are saved first to ensure the original text is safely stored even if the Jina API fails or is unavailable. Embeddings are generated separately so missing vectors can be retried without inserting the chunks again.
# Chunks are stored first to prevent data loss if the Jina API fails, and embeddings are generated separately in batches for efficient processing. This also allows failed embeddings to be retried without inserting the chunks again.
#
# Schema note (normalized schema):
#   chunks no longer stores doc_type / approval_status / source_id. They are derived by the
#   v_chunks view (from documents, decisions, meetings and chunks.requirement_id), so every
#   read below goes through v_chunks and the rows keep exactly the same fields as before.
#   Writes (embedding vectors) still go to the chunks table.
from __future__ import annotations

import os
import time
import uuid
import requests
from typing import Dict, Iterable, List
from dotenv import load_dotenv
from database.db import get_connection

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
BATCH_SIZE = 32

CHUNK_FIELDS = [
    "id", "chunk_text", "project_id", "doc_type", "approval_status",
    "source_event", "source_id", "document_id",
]


# ---------------------------------------------------------------- Jina
def get_jina_embeddings(texts: List[str], retries: int = 4) -> List[List[float]]:
    """Generate 1024-dim embeddings using Jina AI embeddings v3 API (with retries)."""
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY is missing from environment variables or .env file.")

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "jina-embeddings-v3",
        "task": "text-matching",
        "dimensions": 1024,
        "input": texts,
    }

    last_err = None
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 200:
            data = sorted(response.json().get("data", []), key=lambda x: x["index"])
            if len(data) != len(texts):
                raise RuntimeError(f"Jina returned {len(data)} embeddings for {len(texts)} texts")
            return [item["embedding"] for item in data]

        if response.status_code in (429, 500, 502, 503, 504):
            last_err = RuntimeError(f"Jina API error ({response.status_code}): {response.text}")
            time.sleep(2 ** attempt)
            continue

        raise RuntimeError(f"Jina API error ({response.status_code}): {response.text}")

    raise RuntimeError(f"Jina API failed after {retries} retries: {last_err}")


# ---------------------------------------------------------------- DB helpers
def fetch_chunk_rows(conn, where: str = "1=1", params: tuple = ()) -> List[dict]:
    """Fetch chunk rows ready for embedding/syncing.
    Reads from v_chunks (alias c), which supplies the derived doc_type, approval_status
    and source_id columns. `where` may reference any v_chunks column as c.<column>."""
    sql = (
        "SELECT c.id, c.chunk_text, c.project_id, c.doc_type, c.approval_status, "
        "c.source_event, c.source_id, c.document_id "
        "FROM v_chunks c "
        f"WHERE {where} ORDER BY c.project_id, c.document_id, c.chunk_index, c.id"
    )
    with conn.cursor() as cur:
        rows = cur.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def sync_chunks(conn, rows: Iterable[dict], batch_size: int = BATCH_SIZE) -> int:
    """Embed the given chunk rows and save vectors. Returns number embedded."""
    rows = [r for r in rows if (r["chunk_text"] or "").strip()]  # Jina rejects empty strings
    if not rows:
        return 0

    done = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        embeddings = get_jina_embeddings([r["chunk_text"] for r in batch])

        # conn.transaction(), NOT `with conn:` (psycopg3 closes the connection on `with conn:`)
        with conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE chunks SET embedding = %s::vector WHERE id = %s;",
                    [(emb, r["id"]) for r, emb in zip(batch, embeddings)],
                )
        done += len(batch)
        print(f"Processed batch {i // batch_size + 1}: embedded {done}/{len(rows)} chunks.")
    return done


def update_chunk_statuses(conn, statuses: Dict[str, str]) -> None:
    """Change the approval status of specific chunks.

    approval_status is no longer stored on chunks; it is derived from the chunk's parent.
    So this now updates the PARENT row of each chunk:
        'approved'   -> decisions.approved = true   (decision chunks)
                        documents.status  = 'current'   (document chunks)
        'rejected'   -> decisions.approved = false  (decision chunks)
        'superseded' -> documents.status  = 'superseded' (document chunks)
    Note: a document's chunks all share one status, so this affects the whole document.
    """
    if not statuses:
        return

    by_status: Dict[str, List[uuid.UUID]] = {}
    for cid, status in statuses.items():
        if status not in ("approved", "rejected", "superseded"):
            raise ValueError(f"Unsupported approval status: {status!r}")
        by_status.setdefault(status, []).append(uuid.UUID(str(cid)))

    with conn.transaction():
        for status, ids in by_status.items():
            if status in ("approved", "rejected"):
                conn.execute(
                    "UPDATE decisions SET approved = %s WHERE id IN "
                    "(SELECT decision_id FROM chunks WHERE id = ANY(%s) AND decision_id IS NOT NULL)",
                    (status == "approved", ids),
                )
            doc_status = {"approved": "current", "superseded": "superseded"}.get(status)
            if doc_status:
                conn.execute(
                    "UPDATE documents SET status = %s WHERE id IN "
                    "(SELECT document_id FROM chunks WHERE id = ANY(%s) "
                    "AND decision_id IS NULL AND document_id IS NOT NULL)",
                    (doc_status, ids),
                )


def reconcile(conn, repair: bool = True) -> dict:
    """Find chunks missing embeddings; with repair=True, embed and save them."""
    missing = fetch_chunk_rows(conn, "c.embedding IS NULL")

    report = {
        "total_chunks_missing_embedding": len(missing),
        "in_sync": len(missing) == 0,
        "repaired": False,
    }

    if repair and missing:
        report["embedded"] = sync_chunks(conn, missing)
        report["repaired"] = True
        left = conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunks WHERE embedding IS NULL AND btrim(chunk_text) <> ''"
        ).fetchone()["cnt"]
        report["in_sync"] = left == 0

    return report


# ---------------------------------------------------------------- entry points
def embed_missing_chunks() -> dict:
    """Find chunks in Postgres missing embeddings, embed them via Jina, and save."""
    conn = get_connection()
    try:
        rows = fetch_chunk_rows(conn, "c.embedding IS NULL")
        if not rows:
            print("All chunks already have embeddings. Nothing to sync.")
            return {"total_chunks": 0, "embedded": 0, "in_sync": True}

        print(f"Found {len(rows)} chunks missing embeddings. Generating via Jina...")
        embedded = sync_chunks(conn, rows)

        report = {"total_chunks": len(rows), "embedded": embedded, "in_sync": embedded == len(rows)}
        print("Embedding sync complete:", report)
        return report
    finally:
        conn.close()


def embed_and_sync() -> dict:
    """Backwards-compatible alias."""
    return embed_missing_chunks()