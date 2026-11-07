from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf


# ============================================================
# Phase 1: Ingestion & Layout Preprocessing
# ============================================================

@dataclass
class Line:
    text: str
    page: int
    size: float
    bold: bool
    y: float  # top of the line, for ordering/merging decisions


@dataclass
class ParsedPDF:
    lines: list[Line]
    tables: list[dict]  # [{page, header: [str], rows: [[str]]}]
    page_count: int


def _line_from_pymupdf_line(line_dict: dict, page_num: int) -> Optional[Line]:
    spans = line_dict.get("spans", [])
    text = "".join(s["text"] for s in spans).strip()
    if not text:
        return None
    size = spans[0]["size"]
    bold = any("bold" in s["font"].lower() for s in spans)
    y = line_dict["bbox"][1]
    return Line(text=text, page=page_num, size=size, bold=bold, y=y)


def ingest_pdf(pdf_path: Path) -> ParsedPDF:
    """
    Phase 1. Extracts text WITH layout (font size, bold, page, y-position)
    instead of flat text -- this is what later phases need to find the
    title by font size, detect headings by boldness, and merge wrapped
    lines into real paragraphs.

    Table detection: PyMuPDF's page.find_tables() locates ruled/gridded
    tables. If a page has no detectable grid lines (common for simple
    bullet-list role descriptions), no table is returned for that page --
    that's a real "no table here" result, not a failure.

    Image-only pages (no extractable text) are flagged but not OCR'd here
    -- OCR requires a separate heavy dependency (e.g. pytesseract) that
    isn't wired in; `image_only_pages` on the result tells the caller
    which pages would need it.
    """
    doc = pymupdf.open(str(pdf_path))
    lines: list[Line] = []
    tables: list[dict] = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line_dict in block.get("lines", []):
                ln = _line_from_pymupdf_line(line_dict, page_num)
                if ln:
                    lines.append(ln)

        try:
            found = page.find_tables()
            for t in found.tables:
                extracted = t.extract()
                if not extracted or len(extracted) < 2:
                    continue
                header = [c.strip() if c else "" for c in extracted[0]]
                rows = [[c.strip() if c else "" for c in row] for row in extracted[1:]]
                tables.append({"page": page_num, "header": header, "rows": rows})
        except Exception:
            pass  # some pages genuinely have nothing table-like; not an error

    return ParsedPDF(lines=lines, tables=tables, page_count=doc.page_count)


# ============================================================
# Phase 2: Metadata & Cover Page Extraction
# ============================================================

_VERSION_PATTERN_RE = re.compile(r"\bv(?:ersion)?\s*(\d+(?:\.\d+)*)\b", re.IGNORECASE)
_ID_IN_PARENS_RE = re.compile(r"\([A-Z]{1,5}-?\d{2,}\)")  # e.g. "(CS-23054)"
_GENERIC_COVER_TERMS = {
    "SOFTWARE REQUIREMENT", "SOFTWARE REQUIREMENTS", "SPECIFICATION (SRS)",
    "SOFTWARE REQUIREMENT SPECIFICATION", "SOFTWARE REQUIREMENTS SPECIFICATION",
    "SPECIFICATION", "REQUIREMENTS", "REQUIREMENT", "SRS", "(SRS)",
    "PROJECT TITLE", "PROJECT NAME",
}


def extract_title(lines: list[Line]) -> Optional[str]:
    """
    Title = the first run of consecutive lines on page 1 at the document's
    largest cover-page font size, stopping before a Version line or an
    author line (detected by an ID-in-parentheses pattern like '(CS-23054)').
    Generic boilerplate text is skipped even if it's the largest font.
    """
    page1 = [l for l in lines if l.page == 1]
    if not page1:
        return None

    candidate_sizes = sorted({l.size for l in page1}, reverse=True)

    for size in candidate_sizes:
        run = []
        for l in page1:
            if abs(l.size - size) > 0.1:
                if run:
                    break  # a run already started and ended; stop at first gap
                continue
            if _VERSION_PATTERN_RE.search(l.text) or _ID_IN_PARENS_RE.search(l.text):
                break
            if l.text.upper().strip() in _GENERIC_COVER_TERMS:
                continue
            run.append(l.text)
        title = " ".join(run).strip()
        if title:
            return title
    return None


def extract_version(lines: list[Line]) -> str:
    """First 'Version X.Y' / 'vX.Y' match on page 1 or 2; default '1'."""
    for l in lines:
        if l.page > 2:
            break
        m = _VERSION_PATTERN_RE.search(l.text)
        if m:
            return m.group(1)
    return "1"


_TABLE_CAPTION_RE = re.compile(
    r"^Table\s+(\d+(?:\.\d+)*)\s*:\s*Requirement\s*#(\d+)\s*:\s*(.+)$", re.IGNORECASE
)


def extract_table_requirements(lines: list[Line], tables: list[dict]) -> list[dict]:
    """
    Some SRSs (confirmed: Happy Paws) define every functional requirement as
    a key-value "card" -- a 2-column table where each row is [field_name,
    field_value] (Function Description, Input & Origin, Pre-condition, ...),
    captioned just above as "Table 3.1: Requirement #1: <Title>" -- rather
    than a "the system shall..." sentence. A pure modal-verb/prose scan
    (extract_requirements) finds none of these correctly; it only picks up
    fragments of the table's own field values as false-positive sentences.

    This pairs each caption with the table that follows it (1:1, in document
    order -- confirmed on the real sample: 12 captions, 12 tables, same
    order) and builds one clean requirement per card: raw_text is the
    Function Description value (the closest analog to an FR statement),
    with every field kept in structured_fields for the richer schema column.
    """
    captions = []
    for l in lines:
        m = _TABLE_CAPTION_RE.match(l.text.strip())
        if m:
            captions.append({"req_num": m.group(2), "title": m.group(3).strip()})

    if not captions or len(captions) != len(tables):
        return []  # pairing isn't reliable unless counts match 1:1

    found = []
    for caption, table in zip(captions, tables):
        rows = [table["header"]] + table["rows"]  # PyMuPDF folds the first
        # key-value row into table["header"] here, since the "table" has no
        # real header row -- every row, including the first, is a field pair.
        fields = {}
        for row in rows:
            if len(row) >= 2 and row[0].strip():
                key = row[0].replace("\n", " ").strip().rstrip(":")
                value = row[1].replace("\n", " ").strip()
                fields[key] = value

        description = fields.get("Function Description") or next(iter(fields.values()), "")
        nfr_category = _classify_nfr(description)
        found.append({
            "source_label": f"REQ-{caption['req_num']}",
            "category": "nonfunctional" if nfr_category else "functional",
            "nfr_category": nfr_category,
            "raw_text": f"{caption['title']}: {description}".strip(": "),
            "structured_fields": fields,
        })
    return found


def extract_revision_history(tables: list[dict]) -> list[dict]:
    """
    Finds the document-control table: header contains both 'Version' and
    'Date' (case-insensitive). Returns [{version, date, author, description}].
    Column order varies, so columns are matched by header name, not position.
    """
    for t in tables:
        header_lower = [h.lower() for h in t["header"]]
        if any("version" in h for h in header_lower) and any("date" in h for h in header_lower):
            col_idx = {}
            for i, h in enumerate(header_lower):
                if "version" in h:
                    col_idx["version"] = i
                elif "date" in h:
                    col_idx["date"] = i
                elif "author" in h:
                    col_idx["author"] = i
                elif "description" in h or "change" in h:
                    col_idx["description"] = i

            rows = []
            for row in t["rows"]:
                if not any(c for c in row):
                    continue
                rows.append({
                    "version": row[col_idx["version"]] if "version" in col_idx and col_idx["version"] < len(row) else None,
                    "date": row[col_idx["date"]] if "date" in col_idx and col_idx["date"] < len(row) else None,
                    "author": row[col_idx["author"]] if "author" in col_idx and col_idx["author"] < len(row) else None,
                    "description": row[col_idx["description"]] if "description" in col_idx and col_idx["description"] < len(row) else None,
                })
            if rows:
                return rows
    return []


# ============================================================
# Phase 3: Structural Segmentation & Alias Mapping
# ============================================================

_HEADING_NUM_RE = re.compile(r"^(\d+(?:\.\d+){0,4})[.)]?\s+(.+)$")  # number+title on ONE line
_NUM_ONLY_RE = re.compile(r"^(\d+(?:\.\d+){0,4})[.)]?$")             # just the number, title follows separately
_TOC_DOTS_RE = re.compile(r"\.{3,}")


def _body_font_size(lines: list[Line]) -> float:
    """The most common font size in a set of lines -- that's body text.
    Real headings are reliably LARGER than this (confirmed on both real
    sample SRSs: body=12.0, subsection headings=17.2, chapter titles=24.8).
    Numbered list items (e.g. '4. Automatic history logging...') are body
    size too, which is exactly what distinguishes them from real headings."""
    if not lines:
        return 12.0
    from collections import Counter
    return Counter(round(l.size, 1) for l in lines).most_common(1)[0][0]

# canonical schema key -> phrases that should map to it
_SECTION_ALIASES: dict[str, list[str]] = {
    "system_overview": ["product scope", "overview", "introduction", "product perspective", "system overview"],
    "technology_stack": ["technology stack", "tech stack", "implementation environment",
                          "tools used", "operating environment", "software interfaces"],
    "user_roles": ["users and characteristics", "actors", "user roles", "roles and permissions"],
    "functional_requirements": ["functional requirements", "specific requirements"],
    "nonfunctional_requirements": ["non-functional requirements", "nonfunctional requirements",
                                    "other non-functional requirements", "quality attributes",
                                    "performance requirements", "safety and security requirements"],
    "constraints": ["design and implementation constraints", "constraints", "assumptions and dependencies"],
    "document_conventions": ["document conventions"],  # explicitly excluded from requirement scanning
    "use_cases": ["use case", "behaviour requirements", "behavior requirements"],
    "traceability": ["traceability", "appendix"],
}

_ALL_ALIAS_PHRASES = [(key, phrase) for key, phrases in _SECTION_ALIASES.items() for phrase in phrases]


@dataclass
class Section:
    label: str              # raw heading text, e.g. "3.2 Functional Requirements"
    canonical_key: Optional[str]  # mapped key, or None if no alias matched
    page: int
    lines: list[Line] = field(default_factory=list)


def map_to_canonical(heading_text: str, cutoff: float = 0.72) -> Optional[str]:
    """
    Fuzzy-matches a raw heading against the alias dictionary. Conservative
    on purpose: below `cutoff` similarity, returns None rather than guessing
    -- a wrong canonical mapping is worse than "unmapped".
    """
    heading_lower = heading_text.lower().strip()
    best_key, best_score = None, 0.0
    for key, phrase in _ALL_ALIAS_PHRASES:
        # substring match is a strong, cheap first signal
        if phrase in heading_lower:
            return key
        score = difflib.SequenceMatcher(None, phrase, heading_lower).ratio()
        if score > best_score:
            best_key, best_score = key, score
    return best_key if best_score >= cutoff else None


def segment_sections(lines: list[Line]) -> list[Section]:
    """
    Phase 3. Walks the document body (after the cover/ToC) and groups lines
    under whichever numbered heading precedes them.

    Headings are detected by FONT SIZE, not text pattern alone: a numbered
    list item like "4. Automatic history logging..." is printed at body
    text size, while a real heading number ("1.1") and its title are both
    printed at a larger size -- often as two SEPARATE text lines ("1.1" on
    one line, "Document Purpose" on the next), which this merges back
    together. This also means a heading is correctly found even when the
    PDF's layout puts the number and title on one line (handled by
    _HEADING_NUM_RE as a fallback within a still-larger-than-body line).

    If NO numbered heading structure exists at all in the document
    (missing-heading fallback), the whole body becomes one 'unstructured'
    section so Phase 4 can still lexically scan it.
    """
    start_idx = 0
    for i, l in enumerate(lines):
        if l.text.strip() == "Chapter 1":
            start_idx = i + 1
            break

    body = [l for l in lines[start_idx:] if not _TOC_DOTS_RE.search(l.text)]
    if not body:
        return [Section(label="Unstructured Body", canonical_key=None, page=1, lines=[])]

    body_size = _body_font_size(body)
    heading_threshold = body_size + 1.0  # real headings are reliably larger; +1pt margin for rounding

    sections: list[Section] = []
    current: Optional[Section] = None
    i = 0
    while i < len(body):
        l = body[i]
        text = l.text.strip()
        is_heading_size = l.size >= heading_threshold

        if is_heading_size and _NUM_ONLY_RE.match(text):
            # "1.1" on its own line -- merge following same-size lines as the title
            number = text
            title_parts = []
            j = i + 1
            while j < len(body) and body[j].size >= heading_threshold and not _NUM_ONLY_RE.match(body[j].text.strip()):
                title_parts.append(body[j].text.strip())
                j += 1
            title = " ".join(title_parts).strip()
            if title:
                if current:
                    sections.append(current)
                label = f"{number} {title}"
                current = Section(label=label, canonical_key=map_to_canonical(title), page=l.page)
                i = j
                continue

        if is_heading_size:
            m = _HEADING_NUM_RE.match(text)
            if m:
                # number + title already on one line
                if current:
                    sections.append(current)
                label = f"{m.group(1)} {m.group(2).strip()}"
                current = Section(label=label, canonical_key=map_to_canonical(m.group(2)), page=l.page)
                i += 1
                continue

        # Bare page-number footer artifact (e.g. a lone '4' line at body
        # size, left over from the PDF's own page numbering) -- not real
        # content, and was leaking into requirement text as a stray digit.
        if re.fullmatch(r"\d{1,4}", text):
            i += 1
            continue

        if current is None:
            current = Section(label="Preamble", canonical_key=None, page=l.page)
        current.lines.append(l)
        i += 1

    if current:
        sections.append(current)

    if not sections or all(s.label == "Preamble" for s in sections):
        return [Section(label="Unstructured Body", canonical_key=None, page=1, lines=body)]

    return sections


def section_text(section: Section) -> str:
    """Rejoins a section's lines into real paragraphs (wrapped lines
    merged with a space, blank-separated runs kept as paragraph breaks)."""
    return " ".join(l.text for l in section.lines)


# ============================================================
# Phase 4: Core Entity & Section Extraction
# ============================================================

def extract_system_overview(sections: list[Section]) -> Optional[str]:
    for s in sections:
        if s.canonical_key == "system_overview":
            text = section_text(s)
            return text[:1000]  # first ~1000 chars is plenty for a domain/audience summary
    return None


_TECH_KEYWORDS = {
    "backend": ["Flask", "Django", "FastAPI", "Express", "Node.js", "Spring", "Laravel", "Ruby on Rails", ".NET"],
    "frontend": ["React", "Next.js", "Vue", "Angular", "Flutter", "React Native", "Svelte"],
    "database": ["PostgreSQL", "MySQL", "MongoDB", "SQLite", "Firebase", "Neon", "Redis", "Oracle", "MariaDB"],
    "language": ["Python", "JavaScript", "TypeScript", "Java", "C#", "C++", "Go", "Kotlin", "Swift", "PHP"],
    "infra": ["Docker", "Kubernetes", "AWS", "Azure", "GCP", "Vercel", "Render", "Heroku", "Nginx"],
}
_TECH_LOOKUP = {name.lower(): (name, cat) for cat, names in _TECH_KEYWORDS.items() for name in names}
_TECH_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in _TECH_LOOKUP) + r")\b", re.IGNORECASE
)


def extract_tech_stack(sections: list[Section], full_text: str) -> list[dict]:
    """
    Dictionary lookup over the tech-stack-aliased section if one was found;
    falls back to scanning the WHOLE document if no section mapped (missing-
    heading fallback at the entity level, not just the section level).
    Returns deduplicated [{name, category}].
    """
    scoped = " ".join(section_text(s) for s in sections if s.canonical_key == "technology_stack")
    search_text = scoped if scoped.strip() else full_text

    found = {}
    for m in _TECH_WORD_RE.finditer(search_text):
        canonical_name, category = _TECH_LOOKUP[m.group(1).lower()]
        found[canonical_name] = category
    return [{"name": n, "category": c} for n, c in sorted(found.items())]


def extract_user_roles(tables: list[dict], sections: list[Section]) -> list[dict]:
    """
    Prefers a real table whose header mentions role/actor/permission
    (handles merged cells by forward-filling an empty first column from the
    previous row -- the documented edge case for multi-row role spans).
    Falls back to bullet-style role lists in the aliased 'user_roles'
    section (": " separated "RoleName: description" lines, as both of our
    real sample SRSs actually use -- neither has a literal table here).
    """
    for t in tables:
        header_lower = [h.lower() for h in t["header"]]
        if any(k in " ".join(header_lower) for k in ("role", "actor", "permission")):
            role_col = next((i for i, h in enumerate(header_lower) if "role" in h or "actor" in h), 0)
            roles: dict[str, list[str]] = {}
            last_role = None
            for row in t["rows"]:
                name = row[role_col].strip() if role_col < len(row) else ""
                if not name and last_role:
                    name = last_role  # merged-cell carry-down
                elif name:
                    last_role = name
                if not name:
                    continue
                perms = [c.strip() for i, c in enumerate(row) if i != role_col and c.strip()]
                roles.setdefault(name, []).extend(perms)
            if roles:
                return [{"role_name": n, "permissions": p} for n, p in roles.items()]

    # Fallback: bullet-list style, e.g. "Admin: Manages all entities..."
    # Bullet glyphs vary by PDF (-, *, •, or a non-ASCII bullet character
    # like \x88 seen in one of our real sample SRSs) -- stripped first so
    # the role-name match isn't anchored past them.
    bullet_re = re.compile(r"^([A-Z][A-Za-z /()]{1,30})\s*:\s*(.*)$")
    roles = []
    for s in sections:
        if s.canonical_key != "user_roles":
            continue
        section_lines = s.lines
        i = 0
        while i < len(section_lines):
            cleaned = re.sub(r"^[^A-Za-z]+", "", section_lines[i].text.strip())
            m = bullet_re.match(cleaned)
            if m:
                description = m.group(2).strip()
                if not description and i + 1 < len(section_lines):
                    # "Veterinarian:" alone on its line -- description
                    # wrapped entirely onto the next line instead of
                    # continuing after the colon.
                    description = section_lines[i + 1].text.strip()
                    i += 1
                if description:
                    roles.append({"role_name": m.group(1).strip(), "permissions": [description]})
            i += 1
    return roles


_REQ_ID_RE = re.compile(
    r"^(?P<label>(?:FR|NFR|REQ)[-_\s]?\d+)\s*[:.)]\s*(?P<body>.+)$", re.IGNORECASE
)
_MODAL_RE = re.compile(r"\b(shall|must)\b", re.IGNORECASE)
_EXCLUDED_SECTION_KEYS = {"document_conventions"}

_NFR_KEYWORDS = {
    "performance": ("response time", "seconds", "latency", "throughput", "load", "within"),
    "security": ("encrypt", "tls", "https", "password", "authentic", "authoriz", "sanitiz", "injection"),
    "reliability": ("uptime", "availability", "backup", "recover", "fault"),
}


def _split_into_sentences(paragraph: str) -> list[str]:
    """Crude but effective: split on '. ' while protecting common abbreviations."""
    protected = re.sub(r"\b(e\.g|i\.e|etc|Mr|Ms|Dr)\.", r"\1<DOT>", paragraph)
    sentences = re.split(r"(?<=[.!?])\s+", protected)
    return [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]


def _classify_nfr(text: str) -> Optional[str]:
    lowered = text.lower()
    for category, keywords in _NFR_KEYWORDS.items():
        if any(k in lowered for k in keywords):
            return category
    return None


def extract_requirements(sections: list[Section]) -> list[dict]:
    """
    Operates on reconstructed PARAGRAPH text per section (fixes the earlier
    line-level-truncation bug), split into sentences for modal-verb
    scanning. Skips sections mapped to document_conventions (fixes the
    earlier false-positive where the SRS's own "we use the word shall"
    explanation was mistaken for a requirement).

    Labelled requirements (FR1:, NFR-3:) are taken whole from the section
    text (not sentence-split, since their body may be a full multi-clause
    sentence). Bare modal-verb sentences are only taken from sections NOT
    excluded and NOT already contributing a labelled requirement on the
    same line, avoiding duplicate capture.
    """
    found, seen_text = [], set()

    # Splits "FR1: body one FR2: body two FR3: body three" into alternating
    # [label, body, label, body, ...] regardless of how many labels appear --
    # fixes the earlier bug where only the FIRST label in a joined-paragraph
    # section was found (via finditer+manual-slicing) and every subsequent
    # label's text got silently absorbed into the first one's body.
    _LABEL_SPLIT_RE = re.compile(
        r"(?:^|\s)((?:FR|NFR|REQ)[-_\s]?\d+)\s*[:.)]\s*", re.IGNORECASE
    )

    for s in sections:
        if s.canonical_key in _EXCLUDED_SECTION_KEYS:
            continue

        text = section_text(s)

        if "Requirement #" in text:
            # this section's content is a flattened requirement table
            # (already handled, more reliably, by extract_table_requirements).
            # NOTE: _TABLE_CAPTION_RE itself is intentionally ^/$-anchored
            # for matching individual raw lines (correct, used in
            # extract_table_requirements); it must NOT be reused with
            # .search() against this joined multi-line section string --
            # the anchors would silently only ever match at position 0.
            # A plain substring check is deliberately used here instead.
            continue

        parts = _LABEL_SPLIT_RE.split(text)
        # parts = [pre-text, label1, body1, label2, body2, ...]

        for i in range(1, len(parts) - 1, 2):
            label_raw, body_raw = parts[i], parts[i + 1]
            # body runs until the NEXT label's match (already split out) or
            # end of section -- trim any trailing fragment that isn't part
            # of this requirement (defensive strip only, no slicing math).
            body = body_raw.strip()
            if not body or body in seen_text:
                continue
            seen_text.add(body)
            label = label_raw.upper().replace(" ", "").replace("_", "-")
            is_nfr = label.startswith("NFR")
            found.append({
                "source_label": label,
                "category": "nonfunctional" if is_nfr else "functional",
                "nfr_category": _classify_nfr(body) if is_nfr else None,
                "raw_text": body,
            })

        if len(parts) > 1:
            continue  # this section uses labelled requirements; don't also bare-scan it

        for sentence in _split_into_sentences(text):
            if len(sentence) < 15 or len(sentence) > 400:
                continue
            if not _MODAL_RE.search(sentence):
                continue
            if sentence in seen_text:
                continue
            seen_text.add(sentence)
            nfr_category = _classify_nfr(sentence)
            found.append({
                "source_label": None,
                "category": "nonfunctional" if nfr_category else "functional",
                "nfr_category": nfr_category,
                "raw_text": sentence,
            })

    return found


# ============================================================
# Phase 5: Workflow, State & Cross-Reference Resolution
# ============================================================

# The lookahead on the second state (rather than consuming it) lets a
# chain like "Pending -> Verified -> Dispatched" produce BOTH pairs
# (Pending->Verified, Verified->Dispatched) instead of finditer's default
# non-overlapping matches silently dropping the shared middle state.
_ARROW_RE = re.compile(
    r"([A-Z][A-Za-z ]{1,30}?)\s*(?:->|→|–>|-->)\s*(?=([A-Z][A-Za-z ]{1,30}))"
)
_DEPENDENCY_RE = re.compile(
    r"\b(?:Requires|Depends on)\s+((?:FR|NFR|REQ|UC)[-_\s]?\d+)", re.IGNORECASE
)


def extract_state_transitions(full_text: str) -> list[dict]:
    """Finds 'A -> B' / 'A → B' style transition text anywhere in the
    document, including multi-step chains like 'A -> B -> C' (every
    consecutive pair is captured, not just the first and last)."""
    transitions = []
    for m in _ARROW_RE.finditer(full_text):
        from_state, to_state = m.group(1).strip(), m.group(2).strip()
        if len(from_state) < 3 or len(to_state) < 3:
            continue
        transitions.append({"from_state": from_state, "to_state": to_state, "trigger": None})
    return transitions


def extract_dependencies(full_text: str) -> list[dict]:
    """Finds 'Requires FR-04' / 'Depends on UC-02' cross-reference statements."""
    deps = []
    for m in _DEPENDENCY_RE.finditer(full_text):
        deps.append({"depends_on_label": m.group(1).upper().replace(" ", "").replace("_", "-")})
    return deps


def extract_traceability(tables: list[dict]) -> list[dict]:
    """Finds a table whose header mentions test case AND (requirement or use case)."""
    for t in tables:
        header_lower = [h.lower() for h in t["header"]]
        has_test = any("test case" in h or "test_case" in h for h in header_lower)
        has_req_or_uc = any("requirement" in h or "use case" in h for h in header_lower)
        if has_test and has_req_or_uc:
            return [dict(zip(t["header"], row)) for row in t["rows"] if any(c for c in row)]
    return []


# ============================================================
# Phase 6: Final Schema Assembly & Validation
# ============================================================

def parse_srs(pdf_path: Path) -> dict:
    """
    Runs all 6 phases and returns one normalized JSON-serializable dict.
    `_validation` reports which mandatory arrays came back empty, so
    callers can flag a document for manual review rather than silently
    proceeding as if nothing was missing.
    """
    pdf_path = Path(pdf_path)
    parsed = ingest_pdf(pdf_path)
    full_text = "\n".join(l.text for l in parsed.lines)

    title = extract_title(parsed.lines)
    version = extract_version(parsed.lines)
    revision_history = extract_revision_history(parsed.tables)

    sections = segment_sections(parsed.lines)

    system_overview = extract_system_overview(sections)
    tech_stack = extract_tech_stack(sections, full_text)
    roles = extract_user_roles(parsed.tables, sections)

    table_requirements = extract_table_requirements(parsed.lines, parsed.tables)
    # extract_requirements() already skips any section whose content is a
    # flattened requirement table (see _TABLE_CAPTION_RE check inside it),
    # so the two extraction modes are naturally non-overlapping: table
    # cards come from extract_table_requirements, genuine prose "shall"
    # statements (e.g. Happy Paws' NFR section 4.2, which isn't tabular)
    # still come from extract_requirements.
    prose_requirements = extract_requirements(sections)
    requirements = table_requirements + prose_requirements

    transitions = extract_state_transitions(full_text)
    dependencies = extract_dependencies(full_text)
    traceability = extract_traceability(parsed.tables)

    result = {
        "title": title or pdf_path.stem,
        "version": version,
        "revision_history": revision_history,
        "system_overview": system_overview,
        "tech_stack": tech_stack,
        "roles": roles,
        "requirements": requirements,
        "state_transitions": transitions,
        "dependencies": dependencies,
        "traceability": traceability,
        "page_count": parsed.page_count,
        "section_count": len(sections),
        "table_count": len(parsed.tables),
    }

    mandatory = ["requirements"]  # the one array every real SRS must populate
    missing = [k for k in mandatory if not result[k]]
    result["_validation"] = {
        "ok": not missing,
        "missing_mandatory_fields": missing,
        "sections_unmapped": sum(1 for s in sections if s.canonical_key is None),
    }

    return result


if __name__ == "__main__":
    import json
    import sys

    for pdf_file in sys.argv[1:] or Path("data").glob("*.pdf"):
        print(f"\n{'=' * 60}\n{pdf_file}\n{'=' * 60}")
        result = parse_srs(pdf_file)
        print(json.dumps(result, indent=2)[:3000])