from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from database.db import get_connection
from database.resilient_conn import ResilientConnection
from uc.uc2_contracts import AlignmentCitation, CitationKind, EvidenceProvenance, PBISnapshot

DEFAULT_RELATED_LIMIT = 50

_conn: Optional[ResilientConnection] = None


def _get_conn() -> ResilientConnection:
    global _conn
    if _conn is None:
        _conn = ResilientConnection(get_connection)
    return _conn


class TargetPBINotFoundError(LookupError):
    """No committed PBI with this key exists in this project."""


class TargetPBIVersionNotFoundError(LookupError):
    """The PBI exists in this project, but not at the requested version."""


def _require(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _str_or_none(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _parse_json_list(val: Any) -> List[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except ValueError:
            return []
    return []


def _extract_citations(
    raw_citations: Any,
    fallback_source_version_id: Optional[str],
) -> Tuple[List[AlignmentCitation], int]:
    """Convert persisted evidence citation entries into AlignmentCitation
    (SOURCE_SPAN, provenance=PERSISTED_METADATA). Malformed entries are counted."""
    out: List[AlignmentCitation] = []
    ignored = 0
    items = _parse_json_list(raw_citations)
    for c in items:
        if not isinstance(c, dict):
            ignored += 1
            continue
        ref_id = (
            _str_or_none(c.get("source_version_id"))
            or _str_or_none(c.get("reference_id"))
            or fallback_source_version_id
        )
        if not ref_id:
            ignored += 1
            continue
        try:
            out.append(
                AlignmentCitation(
                    kind=CitationKind.SOURCE_SPAN,
                    reference_id=ref_id,
                    reference_version=_str_or_none(c.get("reference_version")),
                    verbatim_snippet=c["verbatim_snippet"],
                    char_start=c["char_start"],
                    char_end=c["char_end"],
                    target_field=_str_or_none(c.get("target_field")),
                    rationale=str(c.get("agent_rationale") or c.get("rationale") or ""),
                    provenance=EvidenceProvenance.PERSISTED_METADATA,
                )
            )
        except Exception:  # noqa: BLE001 - malformed persisted citation is counted and skipped
            ignored += 1
    return out, ignored


def _load_acceptance_criteria(conn: ResilientConnection, pbi_version_ids: List[Any]) -> Dict[str, List[str]]:
    if not pbi_version_ids:
        return {}
    rows = conn.execute(
        """
        SELECT pbi_version_id, criterion
        FROM pbi_version_acceptance_criteria
        WHERE pbi_version_id = ANY(%s::uuid[])
        ORDER BY pbi_version_id, position
        """,
        ([str(v) for v in pbi_version_ids],),
    ).fetchall()
    grouped: Dict[str, List[str]] = {str(vid): [] for vid in pbi_version_ids}
    for row in rows:
        vid_key = str(row["pbi_version_id"])
        criterion = row.get("criterion")
        if isinstance(criterion, str) and criterion.strip():
            grouped.setdefault(vid_key, []).append(criterion)
    return grouped


def _to_snapshot(
    project_id: str,
    pbi_key: str,
    row: Dict[str, Any],
    acceptance_criteria: List[str],
) -> PBISnapshot:
    src_ver_id = _str_or_none(row.get("source_version_id"))
    raw_cites = (
        row.get("version_citations")
        if row.get("version_citations") is not None
        else row.get("proposal_citations")
    )
    cites, ignored = _extract_citations(raw_cites, src_ver_id)
    committed_at = row.get("committed_at")
    return PBISnapshot(
        project_id=str(project_id),
        pbi_key=str(pbi_key),
        version=int(row["version"]),
        proposal_id=_str_or_none(row.get("proposal_id")),
        title=_str_or_none(row.get("title")),
        pbi_type=_str_or_none(row.get("pbi_type")),
        user_story_narrative=_str_or_none(row.get("user_story_narrative")),
        persona=_str_or_none(row.get("persona")),
        acceptance_criteria=acceptance_criteria,
        source_version_id=src_ver_id,
        committed_at=committed_at.isoformat() if committed_at else None,
        evidence_citations=cites,
        ignored_citation_count=ignored,
    )


async def get_target_pbi(
    project_id: str,
    pbi_key: str,
    version: Optional[int] = None,
) -> Tuple[PBISnapshot, int]:
    """Return (snapshot of the requested version, latest version) for a PBI in
    `project_id`. version=None means the latest. Raises, never falls back."""
    pid = _require(project_id, "project_id")
    key = _require(pbi_key, "pbi_key")
    if version is not None and (isinstance(version, bool) or not isinstance(version, int) or version < 1):
        raise ValueError("version must be a positive integer or None")

    conn = _get_conn()
    pbi_row = conn.execute(
        """
        SELECT id, project_id, pbi_key
        FROM pbis
        WHERE project_id = %s::uuid AND pbi_key = %s
        """,
        (pid, key),
    ).fetchone()
    if pbi_row is None:
        raise TargetPBINotFoundError(f"No committed PBI {key!r} in project {pid!r}.")

    latest = conn.execute(
        """
        SELECT MAX(version) AS latest_version
        FROM pbi_versions
        WHERE pbi_id = %s::uuid
        """,
        (str(pbi_row["id"]),),
    ).fetchone()["latest_version"]
    if latest is None:
        raise TargetPBINotFoundError(f"No committed versions for PBI {key!r} in project {pid!r}.")

    wanted = int(latest) if version is None else version
    row = conn.execute(
        """
        SELECT
            v.id,
            v.pbi_id,
            v.version,
            v.proposal_id,
            v.title,
            v.user_story_narrative,
            v.persona,
            v.pbi_type,
            v.evidence_citations AS version_citations,
            v.committed_at,
            pr.source_version_id,
            pr.evidence_citations AS proposal_citations
        FROM pbi_versions v
        LEFT JOIN proposals pr
            ON pr.id = v.proposal_id AND pr.project_id = %s::uuid
        WHERE v.pbi_id = %s::uuid AND v.version = %s
        ORDER BY v.id
        LIMIT 1
        """,
        (pid, str(pbi_row["id"]), wanted),
    ).fetchone()
    if row is None:
        raise TargetPBIVersionNotFoundError(
            f"PBI {key!r} in project {pid!r} has no version {wanted} (latest is {int(latest)})."
        )

    ac_map = _load_acceptance_criteria(conn, [row["id"]])
    return (
        _to_snapshot(pid, key, row, ac_map.get(str(row["id"]), [])),
        int(latest),
    )


async def list_related_pbis(
    project_id: str,
    exclude_pbi_key: str,
    limit: int = DEFAULT_RELATED_LIMIT,
) -> Tuple[List[PBISnapshot], int]:
    """Candidate related PBIs: the latest committed version of every OTHER PBI
    in the same project, ordered by pbi_key. NOT relevance-ranked. Returns
    (snapshots capped at `limit`, total candidates before the cap)."""
    pid = _require(project_id, "project_id")
    excl_key = _require(exclude_pbi_key, "exclude_pbi_key")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")

    conn = _get_conn()
    total = conn.execute(
        """
        WITH latest_ver AS (
            SELECT v.pbi_id, MAX(v.version) AS max_ver
            FROM pbi_versions v
            JOIN pbis p ON p.id = v.pbi_id
            WHERE p.project_id = %s::uuid AND p.pbi_key <> %s
            GROUP BY v.pbi_id
        )
        SELECT COUNT(*) AS cnt FROM latest_ver
        """,
        (pid, excl_key),
    ).fetchone()["cnt"] or 0

    if limit == 0:
        return [], int(total)

    rows = conn.execute(
        """
        WITH latest_ver AS (
            SELECT v.pbi_id, MAX(v.version) AS max_ver
            FROM pbi_versions v
            JOIN pbis p ON p.id = v.pbi_id
            WHERE p.project_id = %s::uuid AND p.pbi_key <> %s
            GROUP BY v.pbi_id
        )
        SELECT
            p.pbi_key,
            v.id,
            v.pbi_id,
            v.version,
            v.proposal_id,
            v.title,
            v.user_story_narrative,
            v.persona,
            v.pbi_type,
            v.evidence_citations AS version_citations,
            v.committed_at,
            pr.source_version_id,
            pr.evidence_citations AS proposal_citations
        FROM pbi_versions v
        JOIN latest_ver lv
            ON lv.pbi_id = v.pbi_id AND lv.max_ver = v.version
        JOIN pbis p
            ON p.id = v.pbi_id
        LEFT JOIN proposals pr
            ON pr.id = v.proposal_id AND pr.project_id = %s::uuid
        WHERE p.project_id = %s::uuid
        ORDER BY p.pbi_key, v.id
        LIMIT %s
        """,
        (pid, excl_key, pid, pid, limit),
    ).fetchall()

    seen_keys = set()
    selected: List[Dict[str, Any]] = []
    for row in rows:
        k = row["pbi_key"]
        if k not in seen_keys:
            seen_keys.add(k)
            selected.append(row)

    ac_map = _load_acceptance_criteria(conn, [r["id"] for r in selected])
    snaps = [
        _to_snapshot(pid, r["pbi_key"], r, ac_map.get(str(r["id"]), []))
        for r in selected
    ]
    return snaps, int(total)