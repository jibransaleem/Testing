from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

import uc.uc2_data as uc2_data
from uc.uc2_contracts import (
    AlignmentContextPackage,
    ContextCategory,
    ContextSnapshot,
    MissingContextItem,
    PBIRef,
    ProductGoalContext,
    ProductVisionContext,
    ProjectRequirement,
    StrategicTheme,
    compute_context_fingerprint,
    derive_available_categories,
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


def _missing(
    strat: StrategicContext,
    present: set,
    *,
    related_total: int,
    related_limit: int,
    evidence_note: str,
) -> List[MissingContextItem]:
    if strat.origin == "none":
        src = "no strategic context source configured"
    else:
        src = f"strategic source '{strat.origin}' provides no {{what}} for this project"

    related_reason = (
        "related PBIs omitted because related_limit=0"
        if related_limit == 0 and related_total > 0
        else "no other committed PBIs exist in this project"
    )

    reasons = {
        ContextCategory.PRODUCT_GOAL: src.format(what="Product Goal"),
        ContextCategory.PRODUCT_VISION: src.format(what="Product Vision"),
        ContextCategory.STRATEGIC_THEMES: src.format(what="strategic themes"),
        ContextCategory.REQUIREMENTS_DECISIONS: src.format(what="requirements or decisions"),
        ContextCategory.RELATED_PBIS: related_reason,
        ContextCategory.EVIDENCE: evidence_note,
    }
    return [
        MissingContextItem(category=c, reason=reasons[c])
        for c in ContextCategory
        if c not in present
    ]


async def build_alignment_context(
    project_id: str,
    target_pbi_key: str,
    *,
    target_version: Optional[int] = None,
    strategic_source: Optional[StrategicSource] = None,
    related_limit: int = uc2_data.DEFAULT_RELATED_LIMIT,
    now: Optional[datetime] = None,
) -> AlignmentContextPackage:
    """Build the deterministic snapshot for auditing `target_pbi_key` (at
    `target_version`, default latest) in `project_id`. Read-only. Raises
    uc2_data.TargetPBINotFoundError / TargetPBIVersionNotFoundError rather
    than substituting another PBI, version or project."""
    target, latest_version = await uc2_data.get_target_pbi(
        project_id, target_pbi_key, target_version
    )
    related, related_total = await uc2_data.list_related_pbis(
        project_id, target_pbi_key, limit=related_limit
    )
    strat = strategic_source(project_id) if strategic_source is not None else StrategicContext()

    goal = (
        ProductGoalContext(
            text=strat.product_goal.strip(),
            source=strat.product_goal_source or strat.origin,
        )
        if strat.product_goal and strat.product_goal.strip()
        else None
    )
    vision = (
        ProductVisionContext(
            text=strat.product_vision.strip(),
            source=strat.product_vision_source or strat.origin,
        )
        if strat.product_vision and strat.product_vision.strip()
        else None
    )
    themes = list(strat.themes)
    reqs = list(strat.requirements)
    evidence = list(target.evidence_citations)

    present = set(
        derive_available_categories(
            product_goal=goal,
            product_vision=vision,
            strategic_themes=themes,
            requirements=reqs,
            related_pbis=related,
            evidence=evidence,
        )
    )
    ev_note = "the audited PBI version has no usable persisted evidence citations"
    if target.ignored_citation_count:
        ev_note += f" ({target.ignored_citation_count} malformed persisted citation(s) were ignored)"

    snapshot = ContextSnapshot(
        project_id=project_id,
        target_pbi_key=target.pbi_key,
        target_pbi_version=target.version,
        target_latest_version=latest_version,
        product_goal_source=goal.source if goal else None,
        product_vision_source=vision.source if vision else None,
        theme_ids=[t.theme_id for t in themes],
        requirement_ids=[r.requirement_id for r in reqs],
        related_pbis=[PBIRef(pbi_key=p.pbi_key, version=p.version) for p in related],
        strategic_context_origin=strat.origin,
        fingerprint=compute_context_fingerprint(
            project_id=project_id,
            target_pbi=target,
            target_latest_version=latest_version,
            product_goal=goal,
            product_vision=vision,
            strategic_themes=themes,
            requirements=reqs,
            related_pbis=related,
            evidence=evidence,
        ),
        built_at=(now or datetime.now(timezone.utc)).isoformat(),
    )

    return AlignmentContextPackage(
        project_id=project_id,
        target_pbi=target,
        target_latest_version=latest_version,
        product_goal=goal,
        product_vision=vision,
        strategic_themes=themes,
        requirements=reqs,
        related_pbis=related,
        related_pbis_total=related_total,
        evidence=evidence,
        available_context=[c for c in ContextCategory if c in present],
        missing_context=_missing(
            strat,
            present,
            related_total=related_total,
            related_limit=related_limit,
            evidence_note=ev_note,
        ),
        context_snapshot=snapshot,
    )