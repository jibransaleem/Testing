from __future__ import annotations

import re
from typing import List

from ingest import normalize_version

_VERSION_RE = re.compile(r"(?:\bversion\s*|\bv)(\d+(?:\.\d+)*)\b", re.I)

# word-boundary matching: "exchange" / "different" must NOT trigger history mode
_HISTORY_RE = re.compile(
    r"\b(changes?|changed|diff|differs?|difference|differences|compare[sd]?|comparison|"
    r"transition(?:ed)?|before|previous(?:ly)?|used to|updated?|history|historical|"
    r"older|original(?:ly)?|earlier|superseded|replaced|evolve[sd]?)\b",
    re.I,
)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class QueryRouter:
    def __init__(self, conn, strict_project_resolution: bool = True):
        """
        strict_project_resolution=True  (original behaviour): if several projects exist and
            the question names none, ask the user -- the caller's default is ignored.
        strict_project_resolution=False: fall back to the caller's default project instead.
        """
        self.conn = conn
        self.strict = strict_project_resolution

    # ------------------------------------------------------------------
    def parse_intent(self, query: str) -> dict:
        versions: List[str] = []
        for m in _VERSION_RE.finditer(query):
            v = normalize_version(m.group(1))
            if v not in versions:
                versions.append(v)

        is_comparative = bool(_HISTORY_RE.search(query))
        return {
            "target_versions": versions,                                   # [] / ['1'] / ['1','2']
            "target_version": versions[0] if len(versions) == 1 else None, # backward compat
            "is_comparative": is_comparative,
            "include_history": is_comparative or bool(versions),
        }

    # ------------------------------------------------------------------
    def resolve_target_project(self, query: str, default_project_id: str | None = None):
        """Returns (project_id, 'EXPLICIT_MATCH'|...) or (None, {"error": ..., ...})."""
        rows = self.conn.execute("SELECT id, name FROM projects ORDER BY name").fetchall()
        projects = [(r["id"], r["name"]) for r in rows]   # (id, name) tuples; rows are dicts
        if not projects:
            return None, {"error": "NO_PROJECTS"}

        q = f" {_norm(query)} "
        matched = []
        for pid, pname in projects:
            candidates = {_norm(pname), _norm(pid[5:] if pid.startswith("proj_") else pid)}
            if any(c and f" {c} " in q for c in candidates):
                matched.append((pid, pname))

        # "School Hub" inside "Smart School Hub": keep only the most specific match
        matched = [m for m in matched
                   if not any(m is not o and _norm(m[1]) in _norm(o[1]) and _norm(m[1]) != _norm(o[1])
                              for o in matched)]

        if len(matched) == 1:
            return matched[0][0], "EXPLICIT_MATCH"
        if len(matched) > 1:
            return None, {"error": "AMBIGUOUS_PROJECT", "matches": matched}

        if len(projects) == 1:
            return projects[0][0], "SINGLE_PROJECT_FALLBACK"

        if not self.strict and default_project_id and any(p[0] == default_project_id for p in projects):
            return default_project_id, "DEFAULT_FALLBACK"

        return None, {"error": "NO_PROJECT_SPECIFIED", "available_projects": projects}