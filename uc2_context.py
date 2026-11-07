"""
AGILIRO Product Owner Agent — UC2
uc2_context.py  —  builds the AlignmentContextPackage (read-only snapshot).

Strategic context (goal / vision / themes / requirements+decisions) is NOT
persisted anywhere in the current repository (the UC1 baseline keeps the
Product Goal and project_requirements in mock fixtures standing in for the
future Knowledge Agent). So the builder takes a pluggable `strategic_source`:
a callable project_id -> StrategicContext.

  * strategic_source=None  -> every strategic category is MISSING ("none").
  * mock_fixture_strategic_source -> reads UC1's mock fixtures for the goal
    and requirements only, tagged origin="mock_fixture" so it can never pass
    for persisted project state. It supplies NO vision and NO themes and
    never derives them from the goal.

Nothing here reasons about alignment, ranks relevance, or calls an LLM.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

import uc2_data
from uc2_contracts import (
    AlignmentContextPackage, ContextCategory, ContextSnapshot, MissingContextItem, PBIRef,
    ProductGoalContext, ProductVisionContext, ProjectRequirement, RequirementKind, StrategicTheme,
    compute_context_fingerprint, derive_available_categories,
)


@dataclass
class StrategicContext:
    """What a strategic-context source can supply. Absent = None / empty."""

    origin: str = "none"
    product_goal: Optional[str] = None
    product_goal_source: Optional[str] = None
    product_vision: Optional[str] = None
    product_vision_source: Optional[str] = None
    themes: List[StrategicTheme] = field(default_factory=list)
    requirements: List[ProjectRequirement] = field(default_factory=list)


StrategicSource = Callable[[str], StrategicContext]


def mock_fixture_strategic_source(project_id: str) -> StrategicContext:
    """Demo/test adapter over UC1's mock fixtures (goal + project_requirements
    only). Requirement ids are content-derived (sha1 of the text) because the
    fixtures carry plain strings with no identity."""
    from mock_data import MOCK_PROJECT_CONTEXT  # lazy: only this adapter needs the mock layer

    ctx = MOCK_PROJECT_CONTEXT.get(project_id)
    if not ctx:
        return StrategicContext(origin="mock_fixture")
    reqs = []
    for text in ctx.get("project_requirements") or []:
        if isinstance(text, str) and text.strip():
            rid = "req-" + hashlib.sha1(text.strip().encode()).hexdigest()[:10]
            reqs.append(ProjectRequirement(requirement_id=rid, kind=RequirementKind.REQUIREMENT,
                                           text=text.strip(), source="mock_fixture"))
    goal = ctx.get("product_goal")
    return StrategicContext(
        origin="mock_fixture", product_goal=goal if isinstance(goal, str) else None,
        product_goal_source="mock_fixture", requirements=reqs)


def _missing(strat: StrategicContext, present: set, *, related_total: int, evidence_note: str) -> List[MissingContextItem]:
    src = "no strategic context source configured" if strat.origin == "none" \
        else f"strategic source '{strat.origin}' provides no {{what}} for this project"
    reasons = {
        ContextCategory.PRODUCT_GOAL: src.format(what="Product Goal"),
        ContextCategory.PRODUCT_VISION: src.format(what="Product Vision"),
        ContextCategory.STRATEGIC_THEMES: src.format(what="strategic themes"),
        ContextCategory.REQUIREMENTS_DECISIONS: src.format(what="requirements or decisions"),
        ContextCategory.RELATED_PBIS: "no other committed PBIs exist in this project",
        ContextCategory.EVIDENCE: evidence_note,
    }
    return [MissingContextItem(category=c, reason=reasons[c]) for c in ContextCategory if c not in present]


async def build_alignment_context(
    project_id: str, target_pbi_key: str, *, target_version: Optional[int] = None,
    strategic_source: Optional[StrategicSource] = None,
    related_limit: int = uc2_data.DEFAULT_RELATED_LIMIT, now: Optional[datetime] = None,
) -> AlignmentContextPackage:
    """Build the deterministic snapshot for auditing `target_pbi_key` (at
    `target_version`, default latest) in `project_id`. Read-only. Raises
    uc2_data.TargetPBINotFoundError / TargetPBIVersionNotFoundError rather
    than substituting another PBI, version or project."""
    target, latest_version = await uc2_data.get_target_pbi(project_id, target_pbi_key, target_version)
    related, related_total = await uc2_data.list_related_pbis(project_id, target_pbi_key, limit=related_limit)
    strat = strategic_source(project_id) if strategic_source else StrategicContext()

    goal = ProductGoalContext(text=strat.product_goal, source=strat.product_goal_source or strat.origin) \
        if strat.product_goal and strat.product_goal.strip() else None
    vision = ProductVisionContext(text=strat.product_vision, source=strat.product_vision_source or strat.origin) \
        if strat.product_vision and strat.product_vision.strip() else None
    themes, reqs, evidence = list(strat.themes), list(strat.requirements), list(target.evidence_citations)

    present = set(derive_available_categories(
        product_goal=goal, product_vision=vision, strategic_themes=themes,
        requirements=reqs, related_pbis=related, evidence=evidence))
    ev_note = "the audited PBI version has no usable persisted evidence citations"
    if target.ignored_citation_count:
        ev_note += f" ({target.ignored_citation_count} malformed persisted citation(s) were ignored)"

    snapshot = ContextSnapshot(
        project_id=project_id, target_pbi_key=target.pbi_key, target_pbi_version=target.version,
        target_latest_version=latest_version, product_goal_source=goal.source if goal else None,
        product_vision_source=vision.source if vision else None,
        theme_ids=[t.theme_id for t in themes], requirement_ids=[r.requirement_id for r in reqs],
        related_pbis=[PBIRef(pbi_key=p.pbi_key, version=p.version) for p in related],
        strategic_context_origin=strat.origin,
        fingerprint=compute_context_fingerprint(
            project_id=project_id, target_pbi=target, target_latest_version=latest_version, product_goal=goal,
            product_vision=vision, strategic_themes=themes, requirements=reqs, related_pbis=related,
            evidence=evidence),
        built_at=(now or datetime.now(timezone.utc)).isoformat())

    return AlignmentContextPackage(
        project_id=project_id, target_pbi=target, target_latest_version=latest_version,
        product_goal=goal, product_vision=vision, strategic_themes=themes, requirements=reqs,
        related_pbis=related, related_pbis_total=related_total, evidence=evidence,
        available_context=[c for c in ContextCategory if c in present],
        missing_context=_missing(strat, present, related_total=related_total, evidence_note=ev_note),
        context_snapshot=snapshot)