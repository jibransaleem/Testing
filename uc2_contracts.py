"""
AGILIRO Product Owner Agent — UC2 (Backlog Alignment & Strategic Drift Audit)
uc2_contracts.py  —  Checkpoint 2: contracts + context boundary only.

Deliberately a separate module: contracts.py (UC1) is frozen, and UC1's
ContextPackageContract is transcript-centric (required raw_source_content /
source_version_id), which UC2 does not have. Nothing here imports UC1 code.

What is in this file
  * Report-side contracts (AlignmentStatus, AlignmentFinding, AlignmentCitation,
    AlignmentAuditReport) and the model-facing pre-grounding schema
    (RawAlignmentAssessment). They are DEFINED here but nothing in Checkpoint 2
    produces or validates them — reasoning, classification and evidence
    verification belong to later checkpoints.
  * AlignmentContextPackage: the deterministic, read-only SNAPSHOT of what UC2
    is allowed to know when auditing one PBI version. It is INPUT to
    reasoning, never a reasoning result.

Unknown means unknown (Invariant 6)
  Optional context that is unavailable is `None` / `[]` — never placeholder
  text — and is listed in `missing_context` with a reason. A validator
  enforces that `available_context` / `missing_context` always agree with the
  actual data, so absence can't silently become an assumption. No strategic
  context (vision, themes, goal) is ever inferred from anything else.

Human authority
  Report models use extra="forbid" and have no action fields (delete, merge,
  reorder, priority, ...). `advisory_notes` is text for the human PO only.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class AlignmentStatus(str, Enum):
    """Technical Design Specification §27."""

    ALIGNED = "ALIGNED"
    PARTIALLY_ALIGNED = "PARTIALLY_ALIGNED"
    POTENTIAL_DRIFT = "POTENTIAL_DRIFT"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    CONFLICT = "CONFLICT"


class CitationKind(str, Enum):
    """What a citation points at. Only SOURCE_SPAN carries character offsets."""

    SOURCE_SPAN = "SOURCE_SPAN"
    PRODUCT_GOAL = "PRODUCT_GOAL"
    REQUIREMENT = "REQUIREMENT"
    RELATED_PBI = "RELATED_PBI"
    DECISION = "DECISION"
    # Added in CP3 (additive): citable context that CP2 had no kind for.
    PRODUCT_VISION = "PRODUCT_VISION"
    STRATEGIC_THEME = "STRATEGIC_THEME"
    TARGET_PBI = "TARGET_PBI"   # the audited PBI's own text


class AlignmentDimension(str, Enum):
    """The strategic context a finding is about."""

    PRODUCT_GOAL = "PRODUCT_GOAL"
    PRODUCT_VISION = "PRODUCT_VISION"
    STRATEGIC_THEME = "STRATEGIC_THEME"
    RELATED_PBI = "RELATED_PBI"
    REQUIREMENT_DECISION = "REQUIREMENT_DECISION"


class ContextCategory(str, Enum):
    """Optional context categories tracked as available / missing."""

    PRODUCT_GOAL = "PRODUCT_GOAL"
    PRODUCT_VISION = "PRODUCT_VISION"
    STRATEGIC_THEMES = "STRATEGIC_THEMES"
    RELATED_PBIS = "RELATED_PBIS"
    REQUIREMENTS_DECISIONS = "REQUIREMENTS_DECISIONS"
    EVIDENCE = "EVIDENCE"


class EvidenceProvenance(str, Enum):
    """How far a citation can be trusted. Checkpoint 2 only ever produces
    PERSISTED_METADATA; VALIDATED and MODEL_ASSERTED exist so Checkpoint 4 can
    distinguish them without changing this contract."""

    PERSISTED_METADATA = "PERSISTED_METADATA"  # stored with a committed PBI; NOT re-verified against source text
    VALIDATED = "VALIDATED"                    # verified by deterministic checks (Checkpoint 4)
    MODEL_ASSERTED = "MODEL_ASSERTED"          # produced by the model, unverified (the safe default)


class RequirementKind(str, Enum):
    REQUIREMENT = "REQUIREMENT"
    DECISION = "DECISION"


# ---------------------------------------------------------------------------
# Citations (provenance-type aware; not transcript-only)
# ---------------------------------------------------------------------------
class AlignmentCitation(_Strict):
    kind: CitationKind
    reference_id: NonBlank = Field(description="Identity of the cited thing (source_version_id, pbi_key, requirement_id, ...).")
    reference_version: Optional[str] = Field(default=None, description="Version of the cited thing, where one exists.")
    verbatim_snippet: NonBlank
    char_start: Optional[int] = Field(default=None, ge=0)
    char_end: Optional[int] = Field(default=None, ge=0)
    target_field: Optional[str] = Field(default=None, description="PBI field the evidence was originally cited for, if known.")
    rationale: str = ""
    provenance: EvidenceProvenance = Field(
        default=EvidenceProvenance.MODEL_ASSERTED,
        description="Trust level. Defaults to the least trusted value; never assume VALIDATED.")

    @model_validator(mode="after")
    def _offsets_only_for_source_spans(self) -> "AlignmentCitation":
        has = self.char_start is not None or self.char_end is not None
        if self.kind == CitationKind.SOURCE_SPAN:
            if self.char_start is None or self.char_end is None:
                raise ValueError("SOURCE_SPAN citations require both char_start and char_end")
            if self.char_end < self.char_start:
                raise ValueError("char_end must be >= char_start")
        elif has:
            raise ValueError(f"{self.kind.value} citations must not carry character offsets")
        return self


# ---------------------------------------------------------------------------
# Context-package building blocks
# ---------------------------------------------------------------------------
class PBISnapshot(_Strict):
    """One persisted, committed PBI version exactly as stored. Absent stored
    fields stay None / [] — they are never filled in."""

    project_id: NonBlank
    pbi_key: NonBlank
    version: int = Field(ge=1)
    proposal_id: Optional[str] = None
    title: Optional[str] = None
    pbi_type: Optional[str] = None
    user_story_narrative: Optional[str] = None
    persona: Optional[str] = None
    acceptance_criteria: List[str] = Field(default_factory=list)
    source_version_id: Optional[str] = None
    committed_at: Optional[str] = None
    evidence_citations: List[AlignmentCitation] = Field(
        default_factory=list,
        description="Citations persisted with this PBI by UC1. NOT re-verified here (no source text is stored in the DB).")
    ignored_citation_count: int = Field(default=0, ge=0, description="Persisted citation entries too malformed to use.")


class ProductGoalContext(_Strict):
    text: NonBlank
    source: NonBlank


class ProductVisionContext(_Strict):
    text: NonBlank
    source: NonBlank


class StrategicTheme(_Strict):
    theme_id: NonBlank
    name: NonBlank
    description: Optional[str] = None
    source: NonBlank


class ProjectRequirement(_Strict):
    requirement_id: NonBlank
    kind: RequirementKind
    text: NonBlank
    source: NonBlank


class MissingContextItem(_Strict):
    category: ContextCategory
    reason: NonBlank


class PBIRef(_Strict):
    pbi_key: NonBlank
    version: int = Field(ge=1)


class ContextSnapshot(_Strict):
    """Identity record: 'this is the context PBI <key> v<n> was audited against'."""

    project_id: NonBlank
    target_pbi_key: NonBlank
    target_pbi_version: int = Field(ge=1)
    target_latest_version: int = Field(ge=1)
    product_goal_source: Optional[str] = None
    product_vision_source: Optional[str] = None
    theme_ids: List[str] = Field(default_factory=list)
    requirement_ids: List[str] = Field(default_factory=list)
    related_pbis: List[PBIRef] = Field(default_factory=list)
    strategic_context_origin: NonBlank = Field(description="e.g. 'mock_fixture' or 'none' — never hidden.")
    fingerprint: NonBlank = Field(description="sha256 over the context content (excludes built_at).")
    built_at: str = Field(default_factory=_utc_iso)


def compute_context_fingerprint(
    *, project_id: str, target_pbi: PBISnapshot, target_latest_version: int,
    product_goal: Optional[ProductGoalContext], product_vision: Optional[ProductVisionContext],
    strategic_themes: Sequence[StrategicTheme], requirements: Sequence[ProjectRequirement],
    related_pbis: Sequence[PBISnapshot], evidence: Sequence[AlignmentCitation],
) -> str:
    def d(x: Any) -> Any:
        return x.model_dump(mode="json") if x is not None else None

    payload = {
        "project_id": project_id, "target_pbi": d(target_pbi), "target_latest_version": target_latest_version,
        "product_goal": d(product_goal), "product_vision": d(product_vision),
        "strategic_themes": [d(t) for t in strategic_themes], "requirements": [d(r) for r in requirements],
        "related_pbis": [d(p) for p in related_pbis], "evidence": [d(e) for e in evidence],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def derive_available_categories(
    *, product_goal: Any, product_vision: Any, strategic_themes: Sequence[Any],
    requirements: Sequence[Any], related_pbis: Sequence[Any], evidence: Sequence[Any],
) -> List[ContextCategory]:
    """Single definition of 'available', shared by the builder and the validator."""
    flags = [
        (ContextCategory.PRODUCT_GOAL, product_goal is not None),
        (ContextCategory.PRODUCT_VISION, product_vision is not None),
        (ContextCategory.STRATEGIC_THEMES, bool(strategic_themes)),
        (ContextCategory.RELATED_PBIS, bool(related_pbis)),
        (ContextCategory.REQUIREMENTS_DECISIONS, bool(requirements)),
        (ContextCategory.EVIDENCE, bool(evidence)),
    ]
    return [c for c, ok in flags if ok]


class AlignmentContextPackage(_Strict):
    """Deterministic, read-only snapshot handed to UC2 reasoning (Checkpoint 3).

    * Optional categories are None / [] when unavailable and then listed in
      `missing_context` with a reason. Never placeholder text.
    * `related_pbis` = the CURRENT (latest) committed version of every OTHER
      PBI in the same project, ordered by pbi_key, capped at a limit.
      "Related" is NOT relevance-ranked at this stage; Checkpoint 3 decides
      relevance. `related_pbis_total` records the pre-cap count.
    * `evidence` = citations persisted with the target PBI version by UC1,
      tagged provenance=PERSISTED_METADATA. They are NOT validated evidence:
      no source text is stored in the DB to check them against.
    * Whether missing context makes an audit impossible (e.g. no Product Goal)
      is NOT decided here; that is the reasoning checkpoint's call.
    """

    project_id: NonBlank
    workflow_type: str = "UC2_ALIGNMENT_AUDIT"

    target_pbi: PBISnapshot
    target_latest_version: int = Field(ge=1)

    product_goal: Optional[ProductGoalContext] = None
    product_vision: Optional[ProductVisionContext] = None
    strategic_themes: List[StrategicTheme] = Field(default_factory=list)
    requirements: List[ProjectRequirement] = Field(default_factory=list)
    related_pbis: List[PBISnapshot] = Field(default_factory=list)
    related_pbis_total: int = Field(default=0, ge=0)
    evidence: List[AlignmentCitation] = Field(default_factory=list)

    available_context: List[ContextCategory] = Field(default_factory=list)
    missing_context: List[MissingContextItem] = Field(default_factory=list)
    context_snapshot: ContextSnapshot

    @property
    def audited_version_is_latest(self) -> bool:
        return self.target_pbi.version == self.target_latest_version

    @property
    def related_pbis_truncated(self) -> bool:
        return self.related_pbis_total > len(self.related_pbis)

    @model_validator(mode="after")
    def _invariants(self) -> "AlignmentContextPackage":
        t = self.target_pbi
        if t.project_id != self.project_id:
            raise ValueError("target_pbi belongs to a different project (project isolation)")
        if self.target_latest_version < t.version:
            raise ValueError("target_latest_version cannot be lower than the audited version")

        keys = [p.pbi_key for p in self.related_pbis]
        if any(p.project_id != self.project_id for p in self.related_pbis):
            raise ValueError("related PBI from a different project (project isolation)")
        if any(e.provenance != EvidenceProvenance.PERSISTED_METADATA for e in self.evidence):
            raise ValueError("context evidence must be PERSISTED_METADATA (validation happens in a later checkpoint)")
        if t.pbi_key in keys:
            raise ValueError("target PBI must not appear among related_pbis")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate pbi_key in related_pbis")
        if self.related_pbis_total < len(self.related_pbis):
            raise ValueError("related_pbis_total is smaller than the number of related PBIs supplied")

        expected = derive_available_categories(
            product_goal=self.product_goal, product_vision=self.product_vision,
            strategic_themes=self.strategic_themes, requirements=self.requirements,
            related_pbis=self.related_pbis, evidence=self.evidence)
        all_cats = set(ContextCategory)
        if set(self.available_context) != set(expected) or len(set(self.available_context)) != len(self.available_context):
            raise ValueError("available_context does not match the context actually present")
        missing_cats = [m.category for m in self.missing_context]
        if len(set(missing_cats)) != len(missing_cats) or set(missing_cats) != all_cats - set(expected):
            raise ValueError("missing_context must list exactly the unavailable categories, once each, with a reason")

        s = self.context_snapshot
        if (s.project_id, s.target_pbi_key, s.target_pbi_version, s.target_latest_version) != (
                self.project_id, t.pbi_key, t.version, self.target_latest_version):
            raise ValueError("context_snapshot identity does not match the package")
        if [(r.pbi_key, r.version) for r in s.related_pbis] != [(p.pbi_key, p.version) for p in self.related_pbis] \
                or s.theme_ids != [x.theme_id for x in self.strategic_themes] \
                or s.requirement_ids != [x.requirement_id for x in self.requirements]:
            raise ValueError("context_snapshot identifiers do not match the package")
        fp = compute_context_fingerprint(
            project_id=self.project_id, target_pbi=t, target_latest_version=self.target_latest_version,
            product_goal=self.product_goal, product_vision=self.product_vision,
            strategic_themes=self.strategic_themes, requirements=self.requirements,
            related_pbis=self.related_pbis, evidence=self.evidence)
        if s.fingerprint != fp:
            raise ValueError("context_snapshot fingerprint does not match package content")
        return self


# ---------------------------------------------------------------------------
# Report side (defined now; produced/validated in later checkpoints)
# ---------------------------------------------------------------------------
class AlignmentFinding(_Strict):
    dimension: AlignmentDimension
    subject_id: Optional[str] = None   # added in CP4 (additive): which theme/requirement/related PBI; None for goal/vision
    assessment: AlignmentStatus
    rationale: NonBlank
    citations: List[AlignmentCitation] = Field(default_factory=list)


class AlignmentConflict(_Strict):
    description: NonBlank
    citations: List[AlignmentCitation] = Field(default_factory=list)


class AlignmentAuditReport(_Strict):
    """Advisory result of one audit of one PBI version. Read-only: it cannot
    delete, reject, merge or reprioritise anything (no such fields exist)."""

    report_id: NonBlank
    project_id: NonBlank
    target_pbi_key: NonBlank
    audited_version: int = Field(ge=1)
    status: AlignmentStatus
    summary: NonBlank
    findings: List[AlignmentFinding] = Field(default_factory=list)
    conflicts: List[AlignmentConflict] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    advisory_notes: List[str] = Field(default_factory=list, description="Text for the human PO; never an instruction to the system.")
    context_fingerprint: NonBlank
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    created_at: str = Field(default_factory=_utc_iso)


class RawAlignmentCitation(_Strict):
    """Model-facing: the model supplies a reference and a snippet but NEVER offsets."""

    kind: CitationKind
    reference_id: str
    reference_version: Optional[str] = None
    verbatim_snippet: str
    rationale: str = ""


class RawAlignmentFinding(_Strict):
    dimension: AlignmentDimension
    assessment: AlignmentStatus
    rationale: str
    citation_candidates: List[RawAlignmentCitation] = Field(default_factory=list)
    # Added in CP3 (additive, defaulted): which subject of the dimension this is about
    # (None for goal/vision) and what is missing when the assessment is INSUFFICIENT_CONTEXT.
    subject_id: Optional[str] = None
    missing_information: List[str] = Field(default_factory=list)


class RawAlignmentAssessment(_Strict):
    """Untrusted model output, before grounding/validation (Checkpoint 4)."""

    status: AlignmentStatus
    summary: str
    findings: List[RawAlignmentFinding] = Field(default_factory=list)
    conflicts_noted: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)