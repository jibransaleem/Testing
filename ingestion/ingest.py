# Ingestion: takes an SRS PDF, extracts text + structured info, stores everything in PostgreSQL,
# and generates Jina embeddings so the AI can search it.
#
# Updated for the normalized schema (UUID ids, lookup tables, users, entities/states,
# permissions catalog, technologies catalog, requirement_* link tables).
# Requires schema_patch.sql to have been applied.
#
# Pipeline per file:
#   1. Extract clean text with PyMuPDF (running headers/footers removed, hyphenation repaired)
#   2. Section-aware chunking (chunks never straddle two sections, cut at sentence ends)
#   3. parse_srs() -> requirements, roles, tech stack, states, revision history, dependencies
#   4. One atomic transaction writes all relational data
#   5. Embeddings are generated after commit; failures are recoverable on the next run

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
import uuid
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Tuple

import pymupdf
from psycopg.types.json import Jsonb

from database.db import get_connection
from .embed import sync_chunks
from Parsing.parser import parse_srs   # hybrid rules+LLM, same output schema as before

log = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOC_TYPE = "srs"
CHUNK_SIZE = 400          # words
CHUNK_OVERLAP = 80        # words
MIN_CHUNK_WORDS = 60      # sections shorter than this are merged into the next one
PREFIX_SECTION_LABEL = True   # prepend "[3.2 Heading]" to chunk text: better embeddings + LLM context

# Bump when the parser/chunker changes in a way that should re-process existing documents.
# It is folded into content_hash, so changing it triggers a one-time re-ingest of every file.
PARSER_VERSION = "2"

EMBED_REQUIREMENTS = True
EVENT_DOCUMENT_INGESTED = "DOCUMENT_INGESTED"
EVENT_REQUIREMENT_CREATED = "REQUIREMENT_CREATED"

# Tables holding project data that does NOT come from SRS ingestion.
_PROJECT_KNOWLEDGE_TABLES = (
    "decisions", "user_stories", "meetings", "sprints", "backlog_items",
    "action_items", "test_cases", "chunks",
)

# Reads through v_chunks so doc_type / approval_status / source_id keep the shape embed.py expects.
_UNEMBEDDED_SQL = (
    "SELECT id, chunk_text, project_id, doc_type, approval_status, "
    "source_event, source_id, document_id "
    "FROM v_chunks WHERE document_id = %s AND embedding IS NULL "
    "ORDER BY (requirement_id IS NOT NULL), chunk_index, id"
)

# ============================================================
# Deterministic UUIDs (re-ingesting the same file replaces the same rows).
# NEVER change this namespace, or every re-ingest will create duplicates.
# ============================================================

_NS = uuid.UUID("6f1c2b0e-8d4a-4c53-9a37-2f6f4b7d1e90")


def _uid(*parts) -> uuid.UUID:
    return uuid.uuid5(_NS, "|".join(str(p) for p in parts))


def _as_uuid(value) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _json_dumps(obj) -> str:
    return json.dumps(obj, default=str)


# ============================================================
# Text sanitising (Postgres rejects NUL bytes in text/jsonb; PDFs sometimes contain them)
# ============================================================

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def clean_text(s: str) -> str:
    return _CTRL_RE.sub("", s)


def sanitize(obj):
    """Recursively strip control characters from every string in a parsed structure."""
    if isinstance(obj, str):
        return clean_text(obj)
    if isinstance(obj, list):
        return [sanitize(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize(x) for x in obj)
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    return obj


# ============================================================
# Date parsing (revision_history.revised_on is a real DATE now)
# ============================================================

# Ambiguous numeric dates (01/02/2024) are read as day/month.
_DATE_FORMATS = (
    "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%Y/%m/%d",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
)


def parse_date(value) -> date | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


# ============================================================
# Requirement labels
# ============================================================

def assign_missing_labels(requirements: list) -> int:
    """Give every requirement without a printed id (e.g. FR-3) a generated one: FR-n for
    functional, NFR-n for nonfunctional, numbered in parser order and skipping labels already
    used. Generated labels are flagged in structured_fields['label_generated'] so they can be
    told apart from ids that were really printed in the SRS. Returns how many were generated."""
    used = {r["source_label"] for r in requirements if r.get("source_label")}
    counters = {"FR": 0, "NFR": 0}
    generated = 0
    for r in requirements:
        if r.get("source_label"):
            continue
        prefix = "NFR" if r.get("category") == "nonfunctional" else "FR"
        while True:
            counters[prefix] += 1
            label = f"{prefix}-{counters[prefix]}"
            if label not in used:
                break
        used.add(label)
        r["source_label"] = label
        r["structured_fields"] = {**(r.get("structured_fields") or {}), "label_generated": True}
        generated += 1
    return generated


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
# Filename metadata + ids
# ============================================================

_VERSION_RE = re.compile(r"(?:^|[_\-\s])v(\d+(?:\.\d+)*)(?=$|[_\-\s])")


def parse_filename_metadata(filename_stem: str) -> Tuple[str, str, str]:
    """Returns (project_key, project_name, version). project_key is only a stable slug
    used to derive the project's UUID; it is no longer stored."""
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


def filename_has_version(filename_stem: str) -> bool:
    return bool(_VERSION_RE.search(filename_stem.lower().strip()))


def make_project_id(project_key: str) -> uuid.UUID:
    return _uid("project", project_key)


def make_document_id(project_id: uuid.UUID, version: str) -> uuid.UUID:
    return _uid("document", project_id, normalize_version(version))


def _resolve_project_id(conn, project_key: str, project_name: str) -> uuid.UUID:
    """projects.name is UNIQUE: reuse an existing project with this name, else derive a UUID."""
    row = conn.execute("SELECT id FROM projects WHERE name = %s", (project_name,)).fetchone()
    return row["id"] if row else make_project_id(project_key)


# ============================================================
# PDF extraction + structure-aware chunking
# ============================================================

def _margin_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.lower())


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        with pymupdf.open(str(pdf_path)) as doc:
            return doc.page_count
    except Exception:
        return 0


def extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """PyMuPDF text (reading order), with repeated headers/footers removed and
    hyphenated line breaks repaired. Returns [(page_number, text)] for pages with text."""
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        raise ValueError(f"Could not read PDF '{pdf_path.name}': {e}") from e

    raw: List[Tuple[int, List[str]]] = []
    with doc:
        for n, page in enumerate(doc, 1):
            try:
                text = page.get_text("text", sort=True) or ""
            except Exception:
                text = ""
            lines = [clean_text(unicodedata.normalize("NFKC", ln)).strip() for ln in text.splitlines()]
            raw.append((n, [ln for ln in lines if ln]))

    repeated: set = set()
    if len(raw) >= 4:  # headers/footers: top/bottom 2 lines repeated on >=40% of pages
        counts: Counter = Counter()
        for _n, lines in raw:
            counts.update({_margin_key(ln) for ln in lines[:2] + lines[-2:]})
        repeated = {k for k, c in counts.items() if c >= max(3, 0.4 * len(raw))}

    pages = []
    for n, lines in raw:
        keep = [ln for i, ln in enumerate(lines)
                if not ((i < 2 or i >= len(lines) - 2) and _margin_key(ln) in repeated)]
        text = re.sub(r"(\w)-\n([a-z])", r"\1\2", "\n".join(keep))  # re-join hyphenated words
        if text.strip():
            pages.append((n, text))
    return pages


_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,4})[.)]?\s+([A-Z][A-Za-z0-9 ,&/()'\-]{2,80})$")
_TOC_RE = re.compile(r"\.{3,}")
_SMALL_WORDS = {"and", "or", "of", "the", "for", "to", "in", "on", "a", "an", "with", "by", "&", "vs"}


def _looks_like_title(title: str) -> bool:
    words = title.split()
    if not words or title.endswith((".", ",", ";")):
        return False
    ok = sum(1 for w in words if w[0].isupper() or w.lower() in _SMALL_WORDS or not w[0].isalpha())
    return ok / len(words) >= 0.8


def _heading_of(line: str):
    """Return 'N.N Title' if the line is a real heading, else None. Rejects list items
    such as '4. Automatic history logging...' (sentence-case, long, or high top-level number)."""
    if _TOC_RE.search(line):
        return None
    m = _HEADING_RE.match(line)
    if not m:
        return None
    num, title = m.group(1), m.group(2).strip()
    if not _looks_like_title(title):
        return None
    if "." not in num and (int(num) > 20 or len(title.split()) > 8):
        return None
    return f"{num} {title}"


def _tokenize(pages: List[Tuple[int, str]]):
    tokens, heading = [], None
    for page_num, text in pages:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            h = _heading_of(line)
            if h:
                heading = h
            for word in line.split():
                tokens.append((word, page_num, heading))
    return tokens


def chunk_pages(
    pages: List[Tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[dict]:
    """Section-aware: windows never cross a heading boundary (so section_label is always right),
    tiny sections are merged forward, and window ends snap to a sentence boundary."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = _tokenize(pages)
    if not tokens:
        return []

    segments: list = []
    for tok in tokens:
        if segments and segments[-1][0] == tok[2]:
            segments[-1][1].append(tok)
        else:
            segments.append((tok[2], [tok]))

    blocks, carry = [], []
    for _h, toks in segments:
        toks = carry + toks
        if len(toks) < MIN_CHUNK_WORDS:
            carry = toks
            continue
        carry = []
        blocks.append(toks)
    if carry:
        if blocks:
            blocks[-1] = blocks[-1] + carry
        else:
            blocks.append(carry)

    chunks: List[dict] = []
    for toks in blocks:
        n, start = len(toks), 0
        while start < n:
            end = min(start + chunk_size, n)
            if end < n:  # prefer ending on a sentence boundary within the last 25% of the window
                for j in range(end, start + int(chunk_size * 0.75), -1):
                    if toks[j - 1][0].endswith((".", "!", "?", ":")):
                        end = j
                        break
            window = toks[start:end]
            label = next((t[2] for t in window if t[2]), None)
            text = " ".join(w for w, _, _ in window)
            if PREFIX_SECTION_LABEL and label and not text.startswith(label):
                text = f"[{label}]\n{text}"
            chunks.append({
                "text": text,
                "section_label": label or f"Section {len(chunks) + 1}",
                "page_start": window[0][1],
                "page_end": window[-1][1],
            })
            if end >= n:
                break
            start = max(end - overlap, start + 1)
    return chunks


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ============================================================
# DB helpers
# (all of these must be called inside `with conn.transaction():`)
# ============================================================

_LOOKUP_TABLES = {"document_types", "requirement_categories", "nfr_categories", "technology_categories"}


def _insert_many(conn, sql: str, rows: list) -> None:
    """Batched insert (one round-trip instead of one per row)."""
    if rows:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)


def _ensure_codes(conn, table: str, codes) -> None:
    """Lookup tables are FK targets: make sure every code we are about to use exists."""
    if table not in _LOOKUP_TABLES:
        raise ValueError(f"Not a lookup table: {table}")
    _insert_many(
        conn,
        f"INSERT INTO {table} (code) VALUES (%s) ON CONFLICT DO NOTHING",
        [(c,) for c in sorted({c for c in codes if c})],
    )


def _ensure_user(conn, display_name, cache: dict):
    """Resolve a free-text author name to users.id.
    Reuses an existing user with the same display name; otherwise creates one with a
    placeholder email (users.email is NOT NULL) that you can later replace with the real one."""
    name = re.sub(r"\s+", " ", clean_text(str(display_name or ""))).strip()
    if not name:
        return None
    key = name.lower()
    if key in cache:
        return cache[key]

    row = conn.execute(
        "SELECT id FROM users WHERE lower(display_name) = %s ORDER BY id LIMIT 1", (key,)
    ).fetchone()
    if row:
        user_id = row["id"]
    else:
        slug = re.sub(r"[^a-z0-9]+", ".", key).strip(".") or "user"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        email = f"{slug[:40]}.{digest}@unknown.invalid"
        user_id = conn.execute(
            "INSERT INTO users (id, email, display_name) VALUES (%s, %s, %s) "
            "ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email RETURNING id",
            (_uid("user", key), email, name),
        ).fetchone()["id"]
    cache[key] = user_id
    return user_id


def refresh_current_flags(conn, project_id, doc_type: str = DOC_TYPE) -> dict:
    """Newest version of each project's document is 'current', older ones 'superseded'.
    Chunk approval is derived from documents.status (see v_chunks), so nothing else to update.
    Returns {document_id: new_status} for documents whose status changed."""
    docs = conn.execute(
        "SELECT id, version, status FROM documents WHERE project_id = %s AND doc_type = %s",
        (project_id, doc_type),
    ).fetchall()
    if not docs:
        return {}

    latest_id = max(docs, key=lambda d: version_key(d["version"]))["id"]
    changed = {}
    for doc in docs:
        new_status = "current" if doc["id"] == latest_id else "superseded"
        if doc["status"] != new_status:
            conn.execute("UPDATE documents SET status = %s WHERE id = %s", (new_status, doc["id"]))
            changed[doc["id"]] = new_status
    return changed


def _delete_requirements(conn, where: str, params: tuple) -> None:
    """`where` is a condition on the requirements table (which has no project_id any more)."""
    ids = f"SELECT id FROM requirements WHERE {where}"
    conn.execute(
        f"DELETE FROM requirement_dependencies "
        f"WHERE requirement_id IN ({ids}) OR depends_on_id IN ({ids})",
        params + params,
    )
    conn.execute(f"DELETE FROM requirement_use_cases WHERE requirement_id IN ({ids})", params)
    conn.execute(f"DELETE FROM requirement_test_cases WHERE requirement_id IN ({ids})", params)
    conn.execute(f"DELETE FROM chunks WHERE requirement_id IN ({ids})", params)
    conn.execute(f"UPDATE user_stories SET requirement_id = NULL WHERE requirement_id IN ({ids})", params)
    conn.execute(f"DELETE FROM requirements WHERE {where}", params)


def _delete_use_cases(conn, where: str, params: tuple) -> None:
    ids = f"SELECT id FROM use_cases WHERE {where}"
    conn.execute(f"DELETE FROM requirement_use_cases WHERE use_case_id IN ({ids})", params)
    conn.execute(f"DELETE FROM use_case_actors WHERE use_case_id IN ({ids})", params)
    conn.execute(f"DELETE FROM use_cases WHERE {where}", params)


def _has_other_project_data(conn, project_id) -> bool:
    return any(
        conn.execute(f"SELECT 1 FROM {t} WHERE project_id = %s LIMIT 1", (project_id,)).fetchone()
        for t in _PROJECT_KNOWLEDGE_TABLES
    )


def _purge_project_rows(conn, project_id) -> None:
    p = (project_id,)
    docs_of_project = "document_id IN (SELECT id FROM documents WHERE project_id = %s)"

    conn.execute("DELETE FROM chunks WHERE project_id = %s", p)
    conn.execute("DELETE FROM action_items WHERE project_id = %s", p)
    conn.execute(
        "DELETE FROM backlog_item_external_refs WHERE backlog_item_id IN "
        "(SELECT id FROM backlog_items WHERE project_id = %s)", p)
    conn.execute("DELETE FROM backlog_items WHERE project_id = %s", p)
    conn.execute(
        "DELETE FROM acceptance_criteria WHERE user_story_id IN "
        "(SELECT id FROM user_stories WHERE project_id = %s)", p)
    conn.execute("DELETE FROM user_stories WHERE project_id = %s", p)
    conn.execute("DELETE FROM meetings WHERE project_id = %s", p)
    conn.execute("DELETE FROM sprints WHERE project_id = %s", p)

    conn.execute(
        "DELETE FROM requirement_test_cases WHERE test_case_id IN "
        "(SELECT id FROM test_cases WHERE project_id = %s)", p)
    conn.execute("DELETE FROM test_cases WHERE project_id = %s", p)

    _delete_use_cases(conn, docs_of_project, p)
    _delete_requirements(conn, docs_of_project, p)
    conn.execute(
        "DELETE FROM revision_history WHERE document_id IN "
        "(SELECT id FROM documents WHERE project_id = %s)", p)

    conn.execute("DELETE FROM project_technologies WHERE project_id = %s", p)
    conn.execute("DELETE FROM project_members WHERE project_id = %s", p)
    conn.execute(
        "DELETE FROM role_permissions WHERE role_id IN "
        "(SELECT id FROM roles WHERE project_id = %s)", p)
    conn.execute("DELETE FROM roles WHERE project_id = %s", p)

    conn.execute(
        "DELETE FROM transitions WHERE entity_id IN "
        "(SELECT id FROM entities WHERE project_id = %s)", p)
    conn.execute(
        "DELETE FROM states WHERE entity_id IN "
        "(SELECT id FROM entities WHERE project_id = %s)", p)
    conn.execute("DELETE FROM entities WHERE project_id = %s", p)

    conn.execute("UPDATE decisions SET supersedes_id = NULL WHERE project_id = %s", p)
    conn.execute("DELETE FROM decisions WHERE project_id = %s", p)
    conn.execute("DELETE FROM documents WHERE project_id = %s", p)
    conn.execute("DELETE FROM projects WHERE id = %s", p)


def _store_dependencies(conn, requirements: list, req_ids: list, dependencies: list) -> Tuple[int, int]:
    """Resolve label-based dependency edges to requirement ids and store them.
    Uses a savepoint so a failure here can't abort the whole ingest.
    Returns (stored, unresolved)."""
    label_to_id, ambiguous = {}, set()
    for rid, r in zip(req_ids, requirements):
        lab = r.get("source_label")
        if not lab:
            continue
        if lab in label_to_id:
            ambiguous.add(lab)
        label_to_id[lab] = rid

    pairs, seen, unresolved = [], set(), 0
    for d in dependencies:
        src, dst = d.get("from_label"), d.get("depends_on_label")
        if not src or not dst or src in ambiguous or dst in ambiguous:
            unresolved += 1
            continue
        a, b = label_to_id.get(src), label_to_id.get(dst)
        if not a or not b or a == b:
            unresolved += 1
            continue
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
    if not pairs:
        return 0, unresolved

    with conn.transaction():  # nested -> SAVEPOINT
        _insert_many(
            conn,
            "INSERT INTO requirement_dependencies (requirement_id, depends_on_id) "
            "VALUES (%s, %s) ON CONFLICT DO NOTHING",
            pairs,
        )
    return len(pairs), unresolved


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
        proj_key, proj_name, file_version = parse_filename_metadata(pdf_path.stem)
        proj_id = _resolve_project_id(conn, proj_key, proj_name)
        content_hash = f"{file_sha256(pdf_path)}:p{PARSER_VERSION}"

        # Same content already ingested for this project (even if renamed): skip, repair embeddings only
        same = conn.execute(
            "SELECT id, version FROM documents WHERE project_id = %s AND content_hash = %s LIMIT 1",
            (proj_id, content_hash),
        ).fetchone()
        if same and not force:
            doc_id, version = same["id"], same["version"]
            missing_rows = conn.execute(_UNEMBEDDED_SQL, (doc_id,)).fetchall()
            if not missing_rows:
                return {"outcome": "skipped", "project_id": str(proj_id), "document_id": str(doc_id),
                        "version": version, "message": f"Already ingested at version {version}."}
            log.info("Repairing %d missing embeddings for %s", len(missing_rows), pdf_path.name)
            try:
                sync_chunks(conn, missing_rows)
            except Exception as e:
                return {"outcome": "repair_failed", "project_id": str(proj_id), "document_id": str(doc_id),
                        "version": version, "message": f"Embedding repair failed: {e}"}
            return {"outcome": "repaired", "project_id": str(proj_id), "document_id": str(doc_id),
                    "version": version, "message": f"Embedded {len(missing_rows)} missing chunks."}

        warnings: List[str] = []

        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError(f"No extractable text in '{pdf_path.name}'.")
        raw_text = "\n".join(text for _n, text in pages)
        total_pages = _pdf_page_count(pdf_path)
        if total_pages > len(pages):
            warnings.append(f"{total_pages - len(pages)} of {total_pages} page(s) had no extractable text "
                            f"(scanned/image-only); they are not in the searchable chunks.")

        # Parse BEFORE opening the transaction: LLM calls can take minutes
        parsed_srs = sanitize(parse_srs(pdf_path))
        requirements = parsed_srs["requirements"]
        tech_stack = parsed_srs["tech_stack"]
        roles = parsed_srs["roles"]
        revision_history = parsed_srs["revision_history"]
        states_transitions = parsed_srs["state_transitions"]
        dependencies = parsed_srs.get("dependencies", [])

        generated_labels = assign_missing_labels(requirements)
        if generated_labels:
            warnings.append(f"{generated_labels} requirement(s) had no id printed in the SRS; "
                            f"generated FR-/NFR- labels (flagged label_generated).")

        # Filename version wins; otherwise use the version printed inside the document
        version = file_version if filename_has_version(pdf_path.stem) \
            else normalize_version(parsed_srs.get("version") or "1")
        doc_id = make_document_id(proj_id, version)
        existing = conn.execute("SELECT 1 FROM documents WHERE id = %s", (doc_id,)).fetchone()

        chunk_ids = [_uid("chunk", doc_id, i) for i in range(1, len(chunks) + 1)]
        req_ids = [_uid("requirement", doc_id, i) for i in range(1, len(requirements) + 1)]
        req_chunks = [
            (_uid("requirement_chunk", rid), rid, r) for rid, r in zip(req_ids, requirements)
            if EMBED_REQUIREMENTS and (r.get("raw_text") or "").strip()
        ]
        dep_stored = 0
        now = datetime.now(timezone.utc)

        # ---- One atomic transaction for all relational data ----
        with conn.transaction():
            # Serialise concurrent ingests of the same project (uploads, retries, workers)
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (str(proj_id),))

            conn.execute(
                "INSERT INTO projects (id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (proj_id, proj_name),
            )

            # Lookup tables are FK targets: make sure every code we use exists
            _ensure_codes(conn, "document_types", [DOC_TYPE])
            _ensure_codes(conn, "requirement_categories", [r["category"] for r in requirements])
            _ensure_codes(conn, "nfr_categories", [r.get("nfr_category") for r in requirements])

            # Project-level data (tech/roles/states) may only come from the NEWEST version,
            # otherwise ingesting an older SRS silently overwrites newer knowledge.
            others = conn.execute(
                "SELECT version FROM documents WHERE project_id = %s AND doc_type = %s AND id <> %s",
                (proj_id, DOC_TYPE, doc_id),
            ).fetchall()
            is_latest = all(version_key(version) >= version_key(o["version"]) for o in others)

            story_links = conn.execute(
                "SELECT us.id, r.source_label, "
                "COALESCE((r.structured_fields ->> 'label_generated') = 'true', false) AS generated "
                "FROM user_stories us "
                "JOIN requirements r ON r.id = us.requirement_id WHERE r.document_id = %s",
                (doc_id,),
            ).fetchall()

            conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
            _delete_requirements(conn, "document_id = %s", (doc_id,))
            conn.execute("DELETE FROM revision_history WHERE document_id = %s", (doc_id,))

            clash = conn.execute(
                "SELECT id FROM documents WHERE project_id = %s AND content_hash = %s AND id <> %s",
                (proj_id, content_hash, doc_id),
            ).fetchone()
            if clash:
                raise ValueError(
                    f"Identical content is already stored as document {clash['id']}; "
                    f"remove it first if you really want it under version {version}.")

            conn.execute(
                """INSERT INTO documents
                       (id, project_id, title, doc_type, version, status,
                        content_hash, source_filename, ingested_at, raw_text,
                        raw_structured_json)
                   VALUES (%s, %s, %s, %s, %s, 'current', %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                       title = EXCLUDED.title, version = EXCLUDED.version,
                       content_hash = EXCLUDED.content_hash,
                       source_filename = EXCLUDED.source_filename,
                       ingested_at = EXCLUDED.ingested_at,
                       raw_text = EXCLUDED.raw_text,
                       raw_structured_json = EXCLUDED.raw_structured_json""",
                (doc_id, proj_id, proj_name, DOC_TYPE, version, content_hash,
                 pdf_path.name, now, raw_text, Jsonb(parsed_srs, dumps=_json_dumps)),
            )

            # Section chunks of the document
            _insert_many(conn, """INSERT INTO chunks
                       (id, project_id, document_id, chunk_text, chunk_index,
                        section_label, page_start, page_end, source_event)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [(cid, proj_id, doc_id, c["text"], i, c["section_label"],
                  c["page_start"], c["page_end"], EVENT_DOCUMENT_INGESTED)
                 for i, (cid, c) in enumerate(zip(chunk_ids, chunks), 1)])

            # Requirements (no project_id column: derived through document_id)
            _insert_many(conn, """INSERT INTO requirements
                       (id, document_id, source_label, category, nfr_category,
                        raw_text, structured_fields, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                [(rid, doc_id, r.get("source_label"), r["category"],
                  r.get("nfr_category"), r["raw_text"],
                  Jsonb(r["structured_fields"], dumps=_json_dumps) if r.get("structured_fields") else None,
                  now)
                 for rid, r in zip(req_ids, requirements)])

            def _req_chunk_text(r: dict) -> str:
                label = r.get("source_label")
                kind = r["category"] + (f"/{r['nfr_category']}" if r.get("nfr_category") else "")
                head = f"{label} ({kind})" if label else f"({kind})"
                return f"{head}: {r['raw_text']}"

            # One searchable chunk per requirement, linked through chunks.requirement_id
            _insert_many(conn, """INSERT INTO chunks
                       (id, project_id, document_id, requirement_id, chunk_text,
                        chunk_index, section_label, source_event)
                   VALUES (%s, %s, %s, %s, %s, 0, %s, %s)""",
                [(cid, proj_id, doc_id, rid, _req_chunk_text(r),
                  r.get("source_label") or r["category"], EVENT_REQUIREMENT_CREATED)
                 for cid, rid, r in req_chunks])

            if dependencies:
                try:
                    dep_stored, dep_unresolved = _store_dependencies(
                        conn, requirements, req_ids, dependencies)
                    if dep_unresolved:
                        warnings.append(f"{dep_unresolved} dependency link(s) could not be resolved to requirements.")
                except Exception as e:
                    warnings.append(f"Dependencies not stored: {e}")

            lost_links = 0
            for story in story_links:
                story_id, label = story["id"], story["source_label"]
                match = conn.execute(
                    "SELECT id FROM requirements WHERE document_id = %s AND source_label = %s",
                    (doc_id, label),
                ).fetchall() if label and not story["generated"] else []
                if len(match) == 1:
                    conn.execute("UPDATE user_stories SET requirement_id = %s WHERE id = %s",
                                 (match[0]["id"], story_id))
                else:
                    lost_links += 1
            if lost_links:
                warnings.append(f"{lost_links} user story link(s) could not be restored.")

            # Revision history: authors become users, dates become DATE,
            # and (document_id, version) must be unique and non-null
            user_cache: dict = {}
            rev_rows, seen_versions, dropped_rev, bad_dates = [], set(), 0, 0
            for r in revision_history:
                v = str(r.get("version") or "").strip()
                if not v or v in seen_versions:
                    dropped_rev += 1
                    continue
                seen_versions.add(v)
                revised_on = parse_date(r.get("date"))
                if r.get("date") and revised_on is None:
                    bad_dates += 1
                rev_rows.append((_uid("revision", doc_id, v), doc_id, v, revised_on,
                                 _ensure_user(conn, r.get("author"), user_cache),
                                 r.get("description")))
            _insert_many(conn, """INSERT INTO revision_history
                       (id, document_id, version, revised_on, author_id, description)
                   VALUES (%s, %s, %s, %s, %s, %s)""", rev_rows)
            if dropped_rev:
                warnings.append(f"{dropped_rev} revision row(s) skipped (missing or duplicate version).")
            if bad_dates:
                warnings.append(f"{bad_dates} revision date(s) could not be parsed and were stored as NULL.")

            if is_latest:
                # An empty parse must never wipe existing knowledge, so only replace when we found something

                # ---- Tech stack: shared catalog + per-project link ----
                tech = {}
                for t in tech_stack:
                    name = (t.get("name") or "").strip()
                    if name and name not in tech:
                        tech[name] = (t.get("category") or "").strip() or None
                if tech:
                    _ensure_codes(conn, "technology_categories", tech.values())
                    conn.execute("DELETE FROM project_technologies WHERE project_id = %s", (proj_id,))
                    tech_ids = []
                    for name, category in tech.items():
                        tech_ids.append(conn.execute(
                            "INSERT INTO technologies (id, name, category) VALUES (%s, %s, %s) "
                            "ON CONFLICT (name) DO UPDATE SET "
                            "category = COALESCE(technologies.category, EXCLUDED.category) "
                            "RETURNING id",
                            (_uid("technology", name), name, category),
                        ).fetchone()["id"])
                    _insert_many(conn,
                        "INSERT INTO project_technologies (project_id, technology_id) VALUES (%s, %s)",
                        [(proj_id, tid) for tid in tech_ids])
                else:
                    warnings.append("No tech stack extracted; existing tech stack kept.")

                # ---- Roles + permissions (upsert so project_members / use_case_actors links survive) ----
                role_perms: dict = {}
                for r in roles:
                    name = (r.get("role_name") or "").strip()
                    if not name:
                        continue
                    perms = role_perms.setdefault(name, [])
                    for p in r.get("permissions") or []:
                        p = (p or "").strip()
                        if p and p not in perms:
                            perms.append(p)
                if role_perms:
                    _insert_many(conn,
                        "INSERT INTO roles (id, project_id, role_name) VALUES (%s, %s, %s) "
                        "ON CONFLICT (project_id, role_name) DO NOTHING",
                        [(_uid("role", proj_id, n), proj_id, n) for n in role_perms])
                    role_rows = conn.execute(
                        "SELECT id, role_name FROM roles WHERE project_id = %s", (proj_id,)).fetchall()
                    role_id = {r["role_name"]: r["id"] for r in role_rows}

                    # Roles that disappeared from the SRS are removed only if nothing references them
                    stale = [r["id"] for r in role_rows if r["role_name"] not in role_perms]
                    if stale:
                        free = [x["id"] for x in conn.execute(
                            "SELECT r.id FROM roles r WHERE r.id = ANY(%s) "
                            "AND NOT EXISTS (SELECT 1 FROM project_members pm WHERE pm.role_id = r.id) "
                            "AND NOT EXISTS (SELECT 1 FROM use_case_actors ua WHERE ua.role_id = r.id)",
                            (stale,)).fetchall()]
                        if free:
                            conn.execute("DELETE FROM role_permissions WHERE role_id = ANY(%s)", (free,))
                            conn.execute("DELETE FROM roles WHERE id = ANY(%s)", (free,))

                    conn.execute("DELETE FROM role_permissions WHERE role_id = ANY(%s)",
                                 ([role_id[n] for n in role_perms],))
                    all_perms = sorted({p for ps in role_perms.values() for p in ps})
                    if all_perms:
                        _insert_many(conn,
                            "INSERT INTO permissions (id, permission_text) VALUES (%s, %s) "
                            "ON CONFLICT (permission_text) DO NOTHING",
                            [(_uid("permission", p), p) for p in all_perms])
                        perm_id = {r["permission_text"]: r["id"] for r in conn.execute(
                            "SELECT id, permission_text FROM permissions WHERE permission_text = ANY(%s)",
                            (all_perms,)).fetchall()}
                        _insert_many(conn,
                            "INSERT INTO role_permissions (role_id, permission_id) VALUES (%s, %s)",
                            [(role_id[n], perm_id[p]) for n, ps in role_perms.items() for p in ps])
                else:
                    warnings.append("No roles extracted; existing roles kept.")

                # ---- State machines: entities -> states -> transitions ----
                if states_transitions:
                    conn.execute(
                        "DELETE FROM transitions WHERE entity_id IN "
                        "(SELECT id FROM entities WHERE project_id = %s)", (proj_id,))
                    conn.execute(
                        "DELETE FROM states WHERE entity_id IN "
                        "(SELECT id FROM entities WHERE project_id = %s)", (proj_id,))
                    conn.execute("DELETE FROM entities WHERE project_id = %s", (proj_id,))

                    # state identity = (entity, name): "Pending" of an Order != "Pending" of a Ticket
                    entity_ids, entity_rows = {}, []
                    state_ids, state_rows = {}, []
                    trans_rows, seen_trans = [], set()
                    for t in states_transitions:
                        entity = (t.get("entity") or "Unspecified").strip() or "Unspecified"
                        if entity not in entity_ids:
                            entity_ids[entity] = _uid("entity", proj_id, entity)
                            entity_rows.append((entity_ids[entity], proj_id, entity))
                        for name in (t["from_state"], t["to_state"]):
                            if (entity, name) not in state_ids:
                                state_ids[(entity, name)] = _uid("state", entity_ids[entity], name)
                                state_rows.append((state_ids[(entity, name)], entity_ids[entity], name))
                        src, dst = state_ids[(entity, t["from_state"])], state_ids[(entity, t["to_state"])]
                        trig = t.get("trigger")
                        if (src, dst, trig) in seen_trans:
                            continue
                        seen_trans.add((src, dst, trig))
                        trans_rows.append((_uid("transition", src, dst, trig or ""),
                                           entity_ids[entity], src, dst, trig))
                    _insert_many(conn,
                        "INSERT INTO entities (id, project_id, name) VALUES (%s, %s, %s)", entity_rows)
                    _insert_many(conn,
                        "INSERT INTO states (id, entity_id, state_name) VALUES (%s, %s, %s)", state_rows)
                    _insert_many(conn,
                        'INSERT INTO transitions (id, entity_id, from_state_id, to_state_id, "trigger") '
                        "VALUES (%s, %s, %s, %s, %s)", trans_rows)
            else:
                warnings.append(f"Version {version} is older than the project's latest; "
                                f"tech stack, roles and states were left unchanged.")

            refresh_current_flags(conn, proj_id)

        # ---- Embeddings (data above is already committed) ----
        unembedded = conn.execute(_UNEMBEDDED_SQL, (doc_id,)).fetchall()
        if unembedded:
            log.info("Generating Jina embeddings for %d chunks in %s", len(unembedded), pdf_path.name)
            try:
                sync_chunks(conn, unembedded)
            except Exception as e:  # data is safe; the next run repairs missing embeddings
                warnings.append(f"Embedding failed ({e}); will be repaired on the next run.")

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
            "project_id": str(proj_id),
            "project_name": proj_name,
            "document_id": str(doc_id),
            "version": version,
            "document_status": status,
            "is_latest_version": is_latest,
            "chunks_added": len(chunks),
            "requirement_chunks_added": len(req_chunks),
            "requirements_extracted": len(requirements),
            "dependencies_stored": dep_stored,
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
        conn = None
        try:
            conn = get_connection()
            res = ingest_file(pdf, conn=conn, force=force)
        except Exception as e:
            log.exception("Ingestion failed for %s", pdf.name)
            res = {"outcome": "error", "file": pdf.name, "message": str(e)}
        finally:
            if conn:
                conn.close()

        res.setdefault("file", pdf.name)
        results.append(res)
    return results


# ============================================================
# Removal
# ============================================================

def remove_document(conn, document_id) -> dict:
    document_id = _as_uuid(document_id)
    row = conn.execute(
        "SELECT project_id, doc_type FROM documents WHERE id = %s", (document_id,)
    ).fetchone()
    if not row:
        raise KeyError(f"Unknown document_id: {document_id}")
    proj_id, doc_type = row["project_id"], row["doc_type"]

    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (str(proj_id),))
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

    return {"document_id": str(document_id), "project_id": str(proj_id), "project_removed": project_removed}


def remove_project(conn, project_id) -> dict:
    project_id = _as_uuid(project_id)
    if not conn.execute("SELECT 1 FROM projects WHERE id = %s", (project_id,)).fetchone():
        raise KeyError(f"Unknown project_id: {project_id}")

    n_chunks = conn.execute(
        "SELECT COUNT(*) AS cnt FROM chunks WHERE project_id = %s", (project_id,)
    ).fetchone()["cnt"]

    with conn.transaction():
        _purge_project_rows(conn, project_id)
    return {"project_id": str(project_id), "chunks_removed": n_chunks}


# ============================================================
# CLI: ingest everything in ./data
# ============================================================

def run_ingestion():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return

    results = ingest_directory(DATA_DIR)
    for r in results:
        print(r)


if __name__ == "__main__":
    run_ingestion()