from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader

from db import get_connection
from embed import sync_chunks
from parser import parse_srs

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOC_TYPE = "srs"
CHUNK_SIZE = 400      # words
CHUNK_OVERLAP = 80    # words

EMBED_REQUIREMENTS = True
REQ_DOC_TYPE = "requirement"
EVENT_DOCUMENT_INGESTED = "DOCUMENT_INGESTED"
EVENT_REQUIREMENT_CREATED = "REQUIREMENT_CREATED"

_PROJECT_KNOWLEDGE_TABLES = (
    "decisions", "user_stories", "meetings", "sprints", "backlog_items",
    "action_items", "chunks",
)

_UNEMBEDDED_SQL = (
    "SELECT id, chunk_text, project_id, doc_type, approval_status, "
    "source_event, source_id, document_id "
    "FROM chunks WHERE document_id = %s AND embedding IS NULL "
    "ORDER BY chunk_index, id"
)


# ============================================================
# Version helpers
# ============================================================

def normalize_version(v: str) -> str:
    """'1.0' -> '1', '2.10' -> '2.10', '2.0.0' -> '2'."""
    parts = [p for p in str(v).strip().lower().lstrip("v").split(".") if p != ""]
    try:
        nums = [int(p) for p in parts] or [1]
    except ValueError:
        return str(v)
    while len(nums) > 1 and nums[-1] == 0:
        nums.pop()
    return ".".join(str(n) for n in nums)


def version_key(v: str) -> Tuple[int, ...]:
    try:
        return tuple(int(p) for p in normalize_version(v).split("."))
    except ValueError:
        return (0,)


# ============================================================
# Filename metadata
# ============================================================

_VERSION_RE = re.compile(r"(?:^|[_\-\s])v(\d+(?:\.\d+)*)(?=$|[_\-\s])")


def parse_filename_metadata(filename_stem: str) -> Tuple[str, str, str]:
    clean = filename_stem.lower().strip()
    m = _VERSION_RE.search(clean)

    if m:
        version = normalize_version(m.group(1))
        base = clean[:m.start()] + clean[m.end():]
    else:
        version = "1"
        base = clean

    base = re.sub(r"[^a-z0-9]+", " ", base).strip()
    if not base:
        base = re.sub(r"[^a-z0-9]+", " ", clean).strip() or "untitled"

    return f"proj_{base.replace(' ', '_')}", base.title(), version


def make_document_id(project_id: str, version: str) -> str:
    return f"doc_{project_id[len('proj_'):]}_v{normalize_version(version)}"


# ============================================================
# PDF extraction + structure-aware chunking
# ============================================================

def extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        raise ValueError(f"Could not read PDF '{pdf_path.name}': {e}") from e

    pages = []
    for n, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append((n, text))
    return pages


_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,4})[.)]?\s+([A-Z][A-Za-z0-9 ,&/()'\-]{2,80})$")
_TOC_RE = re.compile(r"\.{3,}")


def _tokenize(pages: List[Tuple[int, str]]):
    tokens, heading = [], None
    for page_num, text in pages:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = _HEADING_RE.match(line)
            if m and not _TOC_RE.search(line):
                heading = f"{m.group(1)} {m.group(2).strip()}"
            for word in line.split():
                tokens.append((word, page_num, heading))
    return tokens


def chunk_pages(
    pages: List[Tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[dict]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = _tokenize(pages)
    if not tokens:
        return []

    step, n, start = chunk_size - overlap, len(tokens), 0
    chunks = []
    while start < n:
        window = tokens[start:start + chunk_size]
        idx = len(chunks) + 1
        chunks.append({
            "text": " ".join(w for w, _, _ in window),
            "section_label": window[0][2] or f"Section {idx}",
            "page_start": window[0][1],
            "page_end": window[-1][1],
        })
        if start + chunk_size >= n:
            break
        start += step
    return chunks


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ============================================================
# Lifecycle helpers
# (all of these must be called inside `with conn.transaction():`)
# ============================================================

def refresh_current_flags(conn, project_id: str, doc_type: str = DOC_TYPE) -> dict:
    docs = conn.execute(
        "SELECT id, version, status FROM documents WHERE project_id = %s AND doc_type = %s",
        (project_id, doc_type),
    ).fetchall()
    if not docs:
        return {}

    latest_id = max(docs, key=lambda d: version_key(d["version"]))["id"]
    changed = {}

    for doc in docs:
        doc_id, status = doc["id"], doc["status"]
        is_current = doc_id == latest_id
        new_doc_status = "current" if is_current else "superseded"
        if status != new_doc_status:
            conn.execute("UPDATE documents SET status = %s WHERE id = %s", (new_doc_status, doc_id))

        want = "approved" if is_current else "superseded"
        stale = "superseded" if is_current else "approved"
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM chunks WHERE document_id = %s AND decision_id IS NULL "
            "AND approval_status = %s", (doc_id, stale)).fetchall()]
        if ids:
            conn.execute(
                "UPDATE chunks SET approval_status = %s WHERE document_id = %s "
                "AND decision_id IS NULL AND approval_status = %s", (want, doc_id, stale))
            changed.update({i: want for i in ids})
    return changed


def _delete_requirements(conn, where: str, params: tuple) -> None:
    ids = f"SELECT id FROM requirements WHERE {where}"
    conn.execute(
        f"DELETE FROM requirement_dependencies "
        f"WHERE requirement_id IN ({ids}) OR depends_on_id IN ({ids})",
        params + params,
    )
    conn.execute(f"DELETE FROM traceability WHERE requirement_id IN ({ids})", params)
    conn.execute(f"UPDATE user_stories SET requirement_id = NULL WHERE requirement_id IN ({ids})", params)
    conn.execute(f"DELETE FROM requirements WHERE {where}", params)


def _delete_use_cases(conn, where: str, params: tuple) -> None:
    ids = f"SELECT id FROM use_cases WHERE {where}"
    conn.execute(f"DELETE FROM traceability WHERE use_case_id IN ({ids})", params)
    conn.execute(f"DELETE FROM use_cases WHERE {where}", params)


def _has_other_project_data(conn, project_id: str) -> bool:
    return any(
        conn.execute(f"SELECT 1 FROM {t} WHERE project_id = %s LIMIT 1", (project_id,)).fetchone()
        for t in _PROJECT_KNOWLEDGE_TABLES
    )


def _purge_project_rows(conn, project_id: str) -> None:
    p = (project_id,)
    conn.execute("DELETE FROM chunks WHERE project_id = %s", p)
    conn.execute("DELETE FROM action_items WHERE project_id = %s", p)
    conn.execute("DELETE FROM backlog_items WHERE project_id = %s", p)
    conn.execute("DELETE FROM user_stories WHERE project_id = %s", p)
    conn.execute("DELETE FROM meetings WHERE project_id = %s", p)
    conn.execute("DELETE FROM sprints WHERE project_id = %s", p)
    _delete_use_cases(conn, "project_id = %s", p)
    _delete_requirements(conn, "project_id = %s", p)
    conn.execute(
        "DELETE FROM revision_history WHERE document_id IN "
        "(SELECT id FROM documents WHERE project_id = %s)", p)
    conn.execute("DELETE FROM tech_stack WHERE project_id = %s", p)
    conn.execute(
        "DELETE FROM role_permissions WHERE role_id IN "
        "(SELECT id FROM roles WHERE project_id = %s)", p)
    conn.execute("DELETE FROM roles WHERE project_id = %s", p)
    conn.execute("DELETE FROM transitions WHERE project_id = %s", p)
    conn.execute("DELETE FROM states WHERE project_id = %s", p)
    conn.execute("UPDATE decisions SET supersedes_id = NULL WHERE project_id = %s", p)
    conn.execute("DELETE FROM decisions WHERE project_id = %s", p)
    conn.execute("DELETE FROM documents WHERE project_id = %s", p)
    conn.execute("DELETE FROM projects WHERE id = %s", p)


# ============================================================
# Ingest one file
# ============================================================

def ingest_file(pdf_path, conn=None, force: bool = False) -> dict:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()

    try:
        proj_id, proj_name, version = parse_filename_metadata(pdf_path.stem)
        doc_id = make_document_id(proj_id, version)
        content_hash = file_sha256(pdf_path)

        existing = conn.execute(
            "SELECT content_hash FROM documents WHERE id = %s", (doc_id,)
        ).fetchone()

        # Same file already ingested: skip, but repair any missing embeddings
        if existing and existing["content_hash"] == content_hash and not force:
            missing_rows = conn.execute(_UNEMBEDDED_SQL, (doc_id,)).fetchall()
            if not missing_rows:
                return {"outcome": "skipped", "project_id": proj_id, "document_id": doc_id,
                        "version": version, "message": f"Already ingested at version {version}."}
            print(f"Repairing {len(missing_rows)} missing embeddings for {pdf_path.name}...")
            sync_chunks(conn, missing_rows)
            return {"outcome": "repaired", "project_id": proj_id, "document_id": doc_id,
                    "version": version,
                    "message": f"Embedded {len(missing_rows)} missing chunks."}

        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError(f"No extractable text in '{pdf_path.name}'.")
        raw_text = "\n".join(text for _n, text in pages)

        parsed_srs = parse_srs(pdf_path)
        requirements = parsed_srs["requirements"]
        tech_stack = parsed_srs["tech_stack"]
        roles = parsed_srs["roles"]
        revision_history = parsed_srs["revision_history"]
        states_transitions = parsed_srs["state_transitions"]

        new_ids = [f"{doc_id}_chunk_{i}" for i in range(1, len(chunks) + 1)]
        new_req_ids = [f"{doc_id}_req_{i}" for i in range(1, len(requirements) + 1)]
        req_chunks = [
            (f"{rid}_chunk", rid, r) for rid, r in zip(new_req_ids, requirements)
            if EMBED_REQUIREMENTS and (r.get("raw_text") or "").strip()
        ]
        new_revhist_ids = [f"{doc_id}_rev_{i}" for i in range(1, len(revision_history) + 1)]
        new_tech_ids = [f"{proj_id}_tech_{i}" for i in range(1, len(tech_stack) + 1)]
        new_role_ids = [f"{proj_id}_role_{i}" for i in range(1, len(roles) + 1)]
        warnings: List[str] = []

        # ---- One atomic transaction for all relational data ----
        with conn.transaction():
            conn.execute(
                "INSERT INTO projects (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                (proj_id, proj_name),
            )
            story_links = conn.execute(
                "SELECT us.id, r.source_label FROM user_stories us "
                "JOIN requirements r ON r.id = us.requirement_id WHERE r.document_id = %s",
                (doc_id,),
            ).fetchall()

            conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
            _delete_requirements(conn, "document_id = %s", (doc_id,))
            conn.execute("DELETE FROM revision_history WHERE document_id = %s", (doc_id,))
            conn.execute("DELETE FROM tech_stack WHERE project_id = %s", (proj_id,))
            conn.execute(
                "DELETE FROM role_permissions WHERE role_id IN "
                "(SELECT id FROM roles WHERE project_id = %s)", (proj_id,),
            )
            conn.execute("DELETE FROM roles WHERE project_id = %s", (proj_id,))

            conn.execute(
                """INSERT INTO documents
                       (id, project_id, title, doc_type, version, status,
                        content_hash, source_filename, ingested_at, raw_text,
                        raw_structured_json)
                   VALUES (%s, %s, %s, %s, %s, 'current', %s, %s, %s, %s, %s)
                   ON CONFLICT(id) DO UPDATE SET
                       title = EXCLUDED.title, version = EXCLUDED.version,
                       content_hash = EXCLUDED.content_hash,
                       source_filename = EXCLUDED.source_filename,
                       ingested_at = EXCLUDED.ingested_at,
                       raw_text = EXCLUDED.raw_text,
                       raw_structured_json = EXCLUDED.raw_structured_json""",
                (doc_id, proj_id, proj_name, DOC_TYPE, version, content_hash,
                 pdf_path.name, datetime.now(timezone.utc).isoformat(), raw_text,
                 json.dumps(parsed_srs, default=str)),
            )

            for i, (cid, c) in enumerate(zip(new_ids, chunks), 1):
                conn.execute(
                    """INSERT INTO chunks
                           (id, project_id, document_id, chunk_text, chunk_index,
                            section_label, doc_type, approval_status, page_start, page_end,
                            source_event, source_id)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, 'approved', %s, %s, %s, %s)""",
                    (cid, proj_id, doc_id, c["text"], i, c["section_label"], DOC_TYPE,
                     c["page_start"], c["page_end"], EVENT_DOCUMENT_INGESTED, doc_id),
                )

            for rid, r in zip(new_req_ids, requirements):
                conn.execute(
                    """INSERT INTO requirements
                           (id, project_id, document_id, source_label, category,
                            nfr_category, raw_text, structured_fields)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (rid, proj_id, doc_id, r["source_label"], r["category"],
                     r["nfr_category"], r["raw_text"],
                     json.dumps(r["structured_fields"]) if r.get("structured_fields") else None),
                )

            for cid, rid, r in req_chunks:
                txt = f"{r['source_label']}: {r['raw_text']}" if r.get("source_label") else r["raw_text"]
                lbl = r.get("source_label") or r["category"]
                conn.execute(
                    """INSERT INTO chunks
                           (id, project_id, document_id, chunk_text, chunk_index,
                            section_label, doc_type, approval_status,
                            source_event, source_id)
                       VALUES (%s, %s, %s, %s, 0, %s, %s, 'approved', %s, %s)""",
                    (cid, proj_id, doc_id, txt, lbl, REQ_DOC_TYPE, EVENT_REQUIREMENT_CREATED, rid),
                )

            lost_links = 0
            for story in story_links:
                story_id, label = story["id"], story["source_label"]
                match = conn.execute(
                    "SELECT id FROM requirements WHERE document_id = %s AND source_label = %s",
                    (doc_id, label),
                ).fetchall() if label else []
                if len(match) == 1:
                    conn.execute("UPDATE user_stories SET requirement_id = %s WHERE id = %s",
                                 (match[0]["id"], story_id))
                else:
                    lost_links += 1
            if lost_links:
                warnings.append(f"{lost_links} user story link(s) could not be restored.")

            for rid, r in zip(new_revhist_ids, revision_history):
                conn.execute(
                    """INSERT INTO revision_history
                           (id, document_id, version, date, author, description)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (rid, doc_id, r["version"], r["date"], r["author"], r["description"]),
                )

            for tid, t in zip(new_tech_ids, tech_stack):
                conn.execute(
                    "INSERT INTO tech_stack (id, project_id, name, category) VALUES (%s, %s, %s, %s)",
                    (tid, proj_id, t["name"], t["category"]),
                )

            for rid, r in zip(new_role_ids, roles):
                conn.execute(
                    "INSERT INTO roles (id, project_id, role_name) VALUES (%s, %s, %s)",
                    (rid, proj_id, r["role_name"]),
                )
                for i, perm in enumerate(r["permissions"], 1):
                    conn.execute(
                        "INSERT INTO role_permissions (id, role_id, permission_text) VALUES (%s, %s, %s)",
                        (f"{rid}_perm_{i}", rid, perm),
                    )

            if states_transitions:
                conn.execute("DELETE FROM transitions WHERE project_id = %s", (proj_id,))
                conn.execute("DELETE FROM states WHERE project_id = %s", (proj_id,))
                state_name_to_id = {}
                state_rows = []
                for t in states_transitions:
                    for state_name in (t["from_state"], t["to_state"]):
                        if state_name not in state_name_to_id:
                            sid = f"{proj_id}_state_{len(state_name_to_id) + 1}"
                            state_name_to_id[state_name] = sid
                            state_rows.append((sid, proj_id, "Unspecified", state_name))
                for s_row in state_rows:
                    conn.execute(
                        "INSERT INTO states (id, project_id, entity_name, state_name) VALUES (%s, %s, %s, %s)",
                        s_row,
                    )
                for i, t in enumerate(states_transitions, 1):
                    conn.execute(
                        """INSERT INTO transitions (id, project_id, from_state_id, to_state_id, trigger)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (f"{proj_id}_trans_{i}", proj_id,
                         state_name_to_id[t["from_state"]], state_name_to_id[t["to_state"]],
                         t["trigger"]),
                    )

            refresh_current_flags(conn, proj_id)

        # ---- Embeddings (data above is already committed) ----
        unembedded = conn.execute(_UNEMBEDDED_SQL, (doc_id,)).fetchall()
        if unembedded:
            print(f"Generating Jina embeddings for {len(unembedded)} chunks in {pdf_path.name}...")
            sync_chunks(conn, unembedded)

        status = conn.execute(
            "SELECT status FROM documents WHERE id = %s", (doc_id,)
        ).fetchone()["status"]
        still_missing = conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunks WHERE document_id = %s AND embedding IS NULL",
            (doc_id,),
        ).fetchone()["cnt"]
        if still_missing:
            warnings.append(f"{still_missing} chunk(s) still have no embedding.")

        return {
            "outcome": "replaced" if existing else "created",
            "project_id": proj_id,
            "project_name": proj_name,
            "document_id": doc_id,
            "version": version,
            "document_status": status,
            "chunks_added": len(chunks),
            "requirement_chunks_added": len(req_chunks),
            "requirements_extracted": len(requirements),
            "warnings": warnings,
        }

    finally:
        if owns_conn and conn:
            conn.close()


def run_ingestion_for_file(pdf_path, conn=None) -> dict:
    return ingest_file(pdf_path, conn=conn)


def ingest_directory(data_dir, force: bool = False) -> List[dict]:
    """Ingest every *.pdf in a directory; each file gets its own connection so one failure never affects others."""
    results = []
    for pdf in sorted(Path(data_dir).glob("*.pdf")):
        conn = get_connection()
        try:
            res = ingest_file(pdf, conn=conn, force=force)
        except Exception as e:
            res = {"outcome": "error", "file": pdf.name, "message": str(e)}
        finally:
            conn.close()

        res.setdefault("file", pdf.name)
        results.append(res)
    return results


# ============================================================
# Removal
# ============================================================

def remove_document(conn, document_id: str) -> dict:
    row = conn.execute(
        "SELECT project_id, doc_type FROM documents WHERE id = %s", (document_id,)
    ).fetchone()
    if not row:
        raise KeyError(f"Unknown document_id: {document_id}")
    proj_id, doc_type = row["project_id"], row["doc_type"]

    with conn.transaction():
        conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        _delete_requirements(conn, "document_id = %s", (document_id,))
        _delete_use_cases(conn, "document_id = %s", (document_id,))
        conn.execute("DELETE FROM revision_history WHERE document_id = %s", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        refresh_current_flags(conn, proj_id, doc_type)
        remaining = conn.execute(
            "SELECT COUNT(*) AS cnt FROM documents WHERE project_id = %s", (proj_id,)
        ).fetchone()["cnt"]
        project_removed = False
        if remaining == 0 and not _has_other_project_data(conn, proj_id):
            _purge_project_rows(conn, proj_id)
            project_removed = True

    return {"document_id": document_id, "project_id": proj_id, "project_removed": project_removed}


def remove_project(conn, project_id: str) -> dict:
    if not conn.execute("SELECT 1 FROM projects WHERE id = %s", (project_id,)).fetchone():
        raise KeyError(f"Unknown project_id: {project_id}")

    n_chunks = conn.execute(
        "SELECT COUNT(*) AS cnt FROM chunks WHERE project_id = %s", (project_id,)
    ).fetchone()["cnt"]
    with conn.transaction():
        _purge_project_rows(conn, project_id)
    return {"project_id": project_id, "chunks_removed": n_chunks}


# ============================================================
# CLI: ingest everything in ./data
# ============================================================

def run_ingestion():
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return

    results = ingest_directory(DATA_DIR)
    for r in results:
        print(r)


if __name__ == "__main__":
    run_ingestion()