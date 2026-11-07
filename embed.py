from __future__ import annotations

import os
import time
import requests
from typing import Dict, Iterable, List
from dotenv import load_dotenv
from db import get_connection

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
    """Fetch chunk rows ready for embedding/syncing."""
    sql = (
        "SELECT c.id, c.chunk_text, c.project_id, c.doc_type, c.approval_status, "
        "c.source_event, c.source_id, c.document_id "
        "FROM chunks c "
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
    """Update approval status for specific chunk IDs in Postgres."""
    if not statuses:
        return
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET approval_status = %s WHERE id = %s;",
                [(status, cid) for cid, status in statuses.items()],
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


