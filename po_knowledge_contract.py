"""
po_knowledge_contract.py  --  PO Agent <-> Knowledge Agent communication contract (v1.0)

THIS FILE IS COMMUNICATION ONLY. Standard library, no other imports, no
database, no vector store, no LLM. Both agents copy this one file and speak
through it, so neither side can drift from the format.

Flow
----
    PO Agent                                      Knowledge Agent
    ContextRequest(...).to_dict()  ------------>  validate_request(dict)  -> retrieve
    ContextResponse.from_dict(dict) <-----------  response dict (same keys, always)

PO Agent side
    req  = ContextRequest(project_id="proj_x", query="login requirements")
    raw  = knowledge_agent.get_context(req.to_dict())     # transport = your choice
    resp = ContextResponse.from_dict(raw)                  # raises ContractError if malformed
    if resp.ok:
        for p in resp.passages:
            print(p.citation, p.text)
    elif resp.error:
        print(resp.error.code, resp.error.message)

Knowledge Agent side
    req = ContextRequest.from_dict(raw_request)            # raises InvalidRequest
    ... build a response dict ...
    validate_response(response_dict)                       # optional self-check

Two kinds of message
    1. CONTEXT   (ContextRequest / ContextResponse)  - ranked source passages with citations (evidence).
    2. PACKAGE   (PackageRequest / PackageResponse)  - the Knowledge-Agent-owned half of the PO Agent's
       context packages, in the PO models' EXACT field names:
         UC1 ContextPackageContract  <- knowledge.project_requirements (+ product_goal)
         UC2 AlignmentContextPackage <- knowledge.product_goal / product_vision / strategic_themes /
                                        requirements / available_context / missing_context
       The PO Agent keeps what it owns (transcript, backlog titles, source_version_id, PBI snapshots,
       related_pbis, evidence, fingerprint) and merges the two halves.

    3. ITEM      (ItemRequest / ItemResponse)        - the PO Agent WRITES an approved item (user story,
       meeting summary, action item, refinement note, ...) for a project; the Knowledge Agent stores it,
       versions it per item_id, and serves it back through the read messages.

Rules of the contract
    * project_id is explicit and authoritative; the query text never picks the project.
    * Every response has the same top-level keys, success or failure.
    * Unknown values are null, never missing.
    * Default retrieval = current (approved) knowledge only. Older versions or
      superseded content appear only if `versions` / `include_history` ask for it,
      and are flagged is_current=False.
    * Adding optional fields bumps the MINOR schema_version; breaking changes bump the MAJOR.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.2"   # 1.1 added PACKAGE messages; 1.2 adds ITEM (write) messages; older messages unchanged

# ----------------------------------------------------------------------------
# Limits and enumerations
# ----------------------------------------------------------------------------
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
MAX_QUERY_CHARS = 2000

STATUS_OK = "ok"                    # at least one verified passage returned
STATUS_NO_RESULTS = "no_results"    # valid request, nothing relevant found
STATUS_ERROR = "error"              # see response.error
STATUSES = (STATUS_OK, STATUS_NO_RESULTS, STATUS_ERROR)

ERR_INVALID_REQUEST = "INVALID_REQUEST"
ERR_UNKNOWN_PROJECT = "UNKNOWN_PROJECT"       # details.available_project_ids lists valid ids
ERR_RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
ERR_INTERNAL = "INTERNAL_ERROR"
ERR_ITEM_NOT_FOUND = "ITEM_NOT_FOUND"     # retire: nothing stored under that item_id
ERR_WRITE_FAILED = "WRITE_FAILED"           # nothing was stored (changes rolled back)
ERROR_CODES = (ERR_INVALID_REQUEST, ERR_UNKNOWN_PROJECT, ERR_RETRIEVAL_FAILED, ERR_INTERNAL,
               ERR_ITEM_NOT_FOUND, ERR_WRITE_FAILED)

WARN_VERSION_NOT_FOUND = "VERSION_NOT_FOUND"
WARN_DROPPED_UNVERIFIED = "DROPPED_UNVERIFIED_CHUNKS"
WARN_ANSWER_FAILED = "ANSWER_GENERATION_FAILED"
WARN_ANSWER_UNSUPPORTED = "ANSWER_NOT_SUPPORTED_BY_CONTEXT"
WARN_INDEX_OUT_OF_SYNC = "INDEX_OUT_OF_SYNC"   # stored in SQLite; vector index needs reconcile()
WARNING_CODES = (WARN_VERSION_NOT_FOUND, WARN_DROPPED_UNVERIFIED,
                 WARN_ANSWER_FAILED, WARN_ANSWER_UNSUPPORTED, WARN_INDEX_OUT_OF_SYNC)

# What a passage can be (value of source.doc_type / request.doc_types)
DOC_TYPES = (
    "srs", "glossary", "requirement", "user_story", "meeting_summary", "standup",
    "retro_feedback", "action_item", "decision", "approval_rationale",
    "refinement_note", "sync_record", "other",
)
APPROVAL_STATUSES = ("approved", "superseded", "draft", "rejected")
ANSWER_MODES = ("llm", "extractive")


class ContractError(ValueError):
    """A message does not follow the contract."""


class InvalidRequest(ContractError):
    """A request does not follow the contract."""


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def normalize_version(v: Any) -> str:
    """'v1.0' -> '1', '2.10' -> '2.10', '2.0.0' -> '2'. Same rule on both sides."""
    parts = [p for p in str(v).strip().lower().lstrip("v").split(".") if p != ""]
    try:
        nums = [int(p) for p in parts] or [1]
    except ValueError:
        return str(v)
    while len(nums) > 1 and nums[-1] == 0:
        nums.pop()
    return ".".join(str(n) for n in nums)


def _fail(path: str, msg: str):
    raise ContractError(f"{path}: {msg}")


def _obj(d: Any, path: str) -> dict:
    if not isinstance(d, dict):
        _fail(path, "expected an object")
    return d


def _field(d: dict, key: str, types, path: str, allowed=None):
    """Required key (may be null if None is in `types`)."""
    if key not in d:
        _fail(path, f"missing required field '{key}'")
    v = d[key]
    ok = isinstance(v, types)
    if isinstance(v, bool) and bool not in (types if isinstance(types, tuple) else (types,)):
        ok = False                       # bool is an int subclass; do not accept it as a number
    if not ok:
        _fail(f"{path}.{key}", f"wrong type {type(v).__name__}")
    if allowed is not None and v is not None and v not in allowed:
        _fail(f"{path}.{key}", f"'{v}' not in {list(allowed)}")
    return v


_S = (str,)
_NS = (str, type(None))
_NI = (int, type(None))


# ----------------------------------------------------------------------------
# REQUEST  (PO Agent -> Knowledge Agent)
# ----------------------------------------------------------------------------
@dataclass
class ContextRequest:
    project_id: str                                         # required, authoritative
    query: str                                              # required, what the PO Agent needs to know
    top_k: int = DEFAULT_TOP_K                              # 1..MAX_TOP_K
    versions: List[str] = field(default_factory=list)       # [] = current only; ["1"], ["1","2.1"]
    include_history: bool = False                           # also return superseded passages
    doc_types: Optional[List[str]] = None                   # restrict to these DOC_TYPES
    min_score: Optional[float] = None                       # 0..1 similarity floor; None = KA default
    generate_answer: bool = False                           # also return a synthesized answer
    context: Dict[str, Any] = field(default_factory=dict)   # request_id, requester, purpose, sprint_id

    _KEYS = ("project_id", "query", "top_k", "versions", "include_history",
             "doc_types", "min_score", "generate_answer", "context")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id, "query": self.query, "top_k": self.top_k,
            "versions": list(self.versions), "include_history": self.include_history,
            "doc_types": list(self.doc_types) if self.doc_types is not None else None,
            "min_score": self.min_score, "generate_answer": self.generate_answer,
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "ContextRequest":
        """Strict parse + validation. Raises InvalidRequest."""
        if isinstance(d, ContextRequest):
            return d
        if not isinstance(d, dict):
            raise InvalidRequest("Request must be an object.")
        unknown = set(d) - set(cls._KEYS)
        if unknown:
            raise InvalidRequest(f"Unknown request field(s): {sorted(unknown)}")

        pid = d.get("project_id")
        if not isinstance(pid, str) or not pid.strip():
            raise InvalidRequest("'project_id' is required and must be a non-empty string.")

        q = d.get("query")
        if not isinstance(q, str) or not q.strip():
            raise InvalidRequest("'query' is required and must be a non-empty string.")
        if len(q) > MAX_QUERY_CHARS:
            raise InvalidRequest(f"'query' exceeds {MAX_QUERY_CHARS} characters.")

        top_k = d.get("top_k", DEFAULT_TOP_K)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise InvalidRequest(f"'top_k' must be an integer between 1 and {MAX_TOP_K}.")

        versions = d.get("versions") or []
        if not isinstance(versions, list) or not all(
                isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in versions):
            raise InvalidRequest('\'versions\' must be a list of version strings, e.g. ["1", "2.1"].')
        versions = list(dict.fromkeys(normalize_version(v) for v in versions))

        include_history = d.get("include_history", False)
        if not isinstance(include_history, bool):
            raise InvalidRequest("'include_history' must be a boolean.")

        doc_types = d.get("doc_types")
        if doc_types is not None:
            if not isinstance(doc_types, list) or not doc_types or not all(isinstance(t, str) for t in doc_types):
                raise InvalidRequest("'doc_types' must be a non-empty list of strings.")
            bad = sorted(set(doc_types) - set(DOC_TYPES))
            if bad:
                raise InvalidRequest(f"Unknown doc_types {bad}. Allowed: {list(DOC_TYPES)}")

        min_score = d.get("min_score")
        if min_score is not None:
            if isinstance(min_score, bool) or not isinstance(min_score, (int, float)) or not 0.0 <= min_score <= 1.0:
                raise InvalidRequest("'min_score' must be a number between 0 and 1.")
            min_score = float(min_score)

        gen = d.get("generate_answer", False)
        if not isinstance(gen, bool):
            raise InvalidRequest("'generate_answer' must be a boolean.")

        ctx = d.get("context") or {}
        if not isinstance(ctx, dict):
            raise InvalidRequest("'context' must be an object.")

        return cls(pid.strip(), q.strip(), top_k, versions, include_history,
                   doc_types, min_score, gen, dict(ctx))


def validate_request(d: Any) -> ContextRequest:
    """Validate a raw request dict; returns the parsed request or raises InvalidRequest."""
    return ContextRequest.from_dict(d)


# ----------------------------------------------------------------------------
# RESPONSE  (Knowledge Agent -> PO Agent)
# ----------------------------------------------------------------------------
@dataclass
class ProjectRef:
    project_id: str
    name: str

    @classmethod
    def from_dict(cls, d, path="project"):
        d = _obj(d, path)
        return cls(_field(d, "project_id", _S, path), _field(d, "name", _S, path))


@dataclass
class VersionEntry:
    document_id: str
    version: str
    status: str            # 'current' | 'superseded'
    title: str

    @classmethod
    def from_dict(cls, d, path):
        d = _obj(d, path)
        return cls(_field(d, "document_id", _S, path), _field(d, "version", _S, path),
                   _field(d, "status", _S, path), _field(d, "title", _S, path))


@dataclass
class VersionInfo:
    requested: List[str]                 # versions the PO Agent asked for
    available: List[VersionEntry]        # every version of the project, oldest -> newest
    latest: Optional[str]                # current version
    returned: List[str]                  # versions that actually appear in passages
    include_history: bool

    @classmethod
    def from_dict(cls, d, path="versions"):
        d = _obj(d, path)
        avail = _field(d, "available", (list,), path)
        return cls(
            requested=[str(x) for x in _field(d, "requested", (list,), path)],
            available=[VersionEntry.from_dict(a, f"{path}.available[{i}]") for i, a in enumerate(avail)],
            latest=_field(d, "latest", _NS, path),
            returned=[str(x) for x in _field(d, "returned", (list,), path)],
            include_history=_field(d, "include_history", (bool,), path),
        )


@dataclass
class Source:
    """Where a passage came from. Non-document knowledge (decisions, stories) has null document_* fields."""
    project_id: str
    project_name: str
    document_id: Optional[str]
    document_title: Optional[str]
    document_version: Optional[str]
    document_status: Optional[str]
    doc_type: Optional[str]
    section: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    approval_status: str                 # one of APPROVAL_STATUSES
    decision_id: Optional[str]
    source_event: Optional[str]
    source_id: Optional[str]             # id of the originating row (requirement, decision, ...)

    @classmethod
    def from_dict(cls, d, path):
        d = _obj(d, path)
        return cls(
            project_id=_field(d, "project_id", _S, path),
            project_name=_field(d, "project_name", _S, path),
            document_id=_field(d, "document_id", _NS, path),
            document_title=_field(d, "document_title", _NS, path),
            document_version=_field(d, "document_version", _NS, path),
            document_status=_field(d, "document_status", _NS, path),
            doc_type=_field(d, "doc_type", _NS, path),
            section=_field(d, "section", _NS, path),
            page_start=_field(d, "page_start", _NI, path),
            page_end=_field(d, "page_end", _NI, path),
            approval_status=_field(d, "approval_status", _S, path, APPROVAL_STATUSES),
            decision_id=_field(d, "decision_id", _NS, path),
            source_event=_field(d, "source_event", _NS, path),
            source_id=_field(d, "source_id", _NS, path),
        )


@dataclass
class Passage:
    passage_id: str
    rank: int                            # 1 = most relevant
    text: str                            # the relevant text
    score: Optional[float]               # cosine similarity 0..1, higher = more relevant
    is_current: bool                     # False for superseded / older-version content
    version: Optional[str]               # document version, null for non-document knowledge
    citation: str                        # human-readable reference, ready to display
    source: Source

    @classmethod
    def from_dict(cls, d, path):
        d = _obj(d, path)
        score = _field(d, "score", (int, float, type(None)), path)
        return cls(
            passage_id=_field(d, "passage_id", _S, path),
            rank=_field(d, "rank", (int,), path),
            text=_field(d, "text", _S, path),
            score=None if score is None else float(score),
            is_current=_field(d, "is_current", (bool,), path),
            version=_field(d, "version", _NS, path),
            citation=_field(d, "citation", _S, path),
            source=Source.from_dict(_field(d, "source", (dict,), path), f"{path}.source"),
        )


@dataclass
class Answer:
    text: str
    mode: str                            # 'llm' | 'extractive'
    supported: bool                      # False -> passages did not explicitly answer the query
    citations: List[str]

    @classmethod
    def from_dict(cls, d, path="answer"):
        d = _obj(d, path)
        return cls(_field(d, "text", _S, path), _field(d, "mode", _S, path, ANSWER_MODES),
                   _field(d, "supported", (bool,), path),
                   [str(c) for c in _field(d, "citations", (list,), path)])


@dataclass
class Warning:
    code: str
    message: str

    @classmethod
    def from_dict(cls, d, path):
        d = _obj(d, path)
        return cls(_field(d, "code", _S, path), _field(d, "message", _S, path))


@dataclass
class ErrorInfo:
    code: str                            # one of ERROR_CODES
    message: str
    details: Dict[str, Any]

    @classmethod
    def from_dict(cls, d, path="error"):
        d = _obj(d, path)
        return cls(_field(d, "code", _S, path, ERROR_CODES), _field(d, "message", _S, path),
                   _field(d, "details", (dict,), path))


@dataclass
class ContextResponse:
    schema_version: str
    request_id: str
    status: str                          # one of STATUSES
    project: Optional[ProjectRef]        # null if the project could not be resolved
    query: str
    versions: VersionInfo
    passages: List[Passage]
    answer: Optional[Answer]             # null unless generate_answer was requested
    warnings: List[Warning]
    error: Optional[ErrorInfo]           # null unless status == 'error'
    context: Dict[str, Any]              # echo of the request context
    meta: Dict[str, Any]                 # top_k, min_score, returned, latency_ms, generated_at

    # convenience for the PO Agent
    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def citations(self) -> List[str]:
        return [p.citation for p in self.passages]

    @classmethod
    def from_dict(cls, d: Any) -> "ContextResponse":
        """Strict parse + validation of a raw response. Raises ContractError."""
        p = "response"
        d = _obj(d, p)
        major = str(_field(d, "schema_version", _S, p)).split(".")[0]
        if major != SCHEMA_VERSION.split(".")[0]:
            _fail(f"{p}.schema_version", f"unsupported major version '{d['schema_version']}' (expected {SCHEMA_VERSION})")

        status = _field(d, "status", _S, p, STATUSES)
        project = _field(d, "project", (dict, type(None)), p)
        answer = _field(d, "answer", (dict, type(None)), p)
        error = _field(d, "error", (dict, type(None)), p)
        passages = _field(d, "passages", (list,), p)
        warnings = _field(d, "warnings", (list,), p)

        if status == STATUS_ERROR and error is None:
            _fail(f"{p}.error", "status is 'error' but error is null")
        if status != STATUS_ERROR and error is not None:
            _fail(f"{p}.error", "error must be null unless status is 'error'")
        if status == STATUS_OK and not passages:
            _fail(f"{p}.passages", "status is 'ok' but there are no passages")

        return cls(
            schema_version=d["schema_version"],
            request_id=_field(d, "request_id", _S, p),
            status=status,
            project=ProjectRef.from_dict(project) if project is not None else None,
            query=_field(d, "query", _S, p),
            versions=VersionInfo.from_dict(_field(d, "versions", (dict,), p)),
            passages=[Passage.from_dict(x, f"{p}.passages[{i}]") for i, x in enumerate(passages)],
            answer=Answer.from_dict(answer) if answer is not None else None,
            warnings=[Warning.from_dict(w, f"{p}.warnings[{i}]") for i, w in enumerate(warnings)],
            error=ErrorInfo.from_dict(error) if error is not None else None,
            context=_field(d, "context", (dict,), p),
            meta=_field(d, "meta", (dict,), p),
        )


def validate_response(d: Any) -> ContextResponse:
    """Validate a raw response dict; returns the parsed response or raises ContractError."""
    return ContextResponse.from_dict(d)


# ----------------------------------------------------------------------------
# JSON Schema (for non-Python consumers / API docs / CI). Draft 2020-12.
# ----------------------------------------------------------------------------
_NSTR = {"type": ["string", "null"]}
_NINT = {"type": ["integer", "null"]}

CONTEXT_REQUEST_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "KnowledgeContextRequest",
    "type": "object",
    "additionalProperties": False,
    "required": ["project_id", "query"],
    "properties": {
        "project_id": {"type": "string", "minLength": 1},
        "query": {"type": "string", "minLength": 1, "maxLength": MAX_QUERY_CHARS},
        "top_k": {"type": "integer", "minimum": 1, "maximum": MAX_TOP_K, "default": DEFAULT_TOP_K},
        "versions": {"type": "array", "items": {"type": "string"}, "default": []},
        "include_history": {"type": "boolean", "default": False},
        "doc_types": {"type": ["array", "null"], "items": {"enum": list(DOC_TYPES)}, "minItems": 1},
        "min_score": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "generate_answer": {"type": "boolean", "default": False},
        "context": {
            "type": "object",
            "description": "Caller context, echoed back unchanged. Not used for retrieval.",
            "properties": {"request_id": {"type": "string"}, "requester": {"type": "string"},
                           "purpose": {"type": "string"}, "sprint_id": {"type": "string"}},
        },
    },
}

_SOURCE_SCHEMA = {
    "type": "object",
    "required": ["project_id", "project_name", "document_id", "document_title", "document_version",
                 "document_status", "doc_type", "section", "page_start", "page_end",
                 "approval_status", "decision_id", "source_event", "source_id"],
    "properties": {
        "project_id": {"type": "string"}, "project_name": {"type": "string"},
        "document_id": _NSTR, "document_title": _NSTR, "document_version": _NSTR,
        "document_status": _NSTR, "doc_type": _NSTR, "section": _NSTR,
        "page_start": _NINT, "page_end": _NINT,
        "approval_status": {"enum": list(APPROVAL_STATUSES)},
        "decision_id": _NSTR, "source_event": _NSTR, "source_id": _NSTR,
    },
}

CONTEXT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "KnowledgeContextResponse",
    "type": "object",
    "required": ["schema_version", "request_id", "status", "project", "query", "versions",
                 "passages", "answer", "warnings", "error", "context", "meta"],
    "properties": {
        "schema_version": {"type": "string"},
        "request_id": {"type": "string"},
        "status": {"enum": list(STATUSES)},
        "project": {"oneOf": [{"type": "null"}, {
            "type": "object", "required": ["project_id", "name"],
            "properties": {"project_id": {"type": "string"}, "name": {"type": "string"}}}]},
        "query": {"type": "string"},
        "versions": {
            "type": "object",
            "required": ["requested", "available", "latest", "returned", "include_history"],
            "properties": {
                "requested": {"type": "array", "items": {"type": "string"}},
                "available": {"type": "array", "items": {
                    "type": "object", "required": ["document_id", "version", "status", "title"],
                    "properties": {"document_id": {"type": "string"}, "version": {"type": "string"},
                                   "status": {"type": "string"}, "title": {"type": "string"}}}},
                "latest": _NSTR,
                "returned": {"type": "array", "items": {"type": "string"}},
                "include_history": {"type": "boolean"},
            },
        },
        "passages": {"type": "array", "items": {
            "type": "object",
            "required": ["passage_id", "rank", "text", "score", "is_current", "version", "citation", "source"],
            "properties": {
                "passage_id": {"type": "string"}, "rank": {"type": "integer", "minimum": 1},
                "text": {"type": "string"}, "score": {"type": ["number", "null"]},
                "is_current": {"type": "boolean"}, "version": _NSTR,
                "citation": {"type": "string"}, "source": _SOURCE_SCHEMA,
            },
        }},
        "answer": {"oneOf": [{"type": "null"}, {
            "type": "object", "required": ["text", "mode", "supported", "citations"],
            "properties": {"text": {"type": "string"}, "mode": {"enum": list(ANSWER_MODES)},
                           "supported": {"type": "boolean"},
                           "citations": {"type": "array", "items": {"type": "string"}}}}]},
        "warnings": {"type": "array", "items": {
            "type": "object", "required": ["code", "message"],
            "properties": {"code": {"type": "string"}, "message": {"type": "string"}}}},
        "error": {"oneOf": [{"type": "null"}, {
            "type": "object", "required": ["code", "message", "details"],
            "properties": {"code": {"enum": list(ERROR_CODES)}, "message": {"type": "string"},
                           "details": {"type": "object"}}}]},
        "context": {"type": "object"},
        "meta": {"type": "object",
                 "required": ["top_k", "min_score", "returned", "latency_ms", "generated_at"],
                 "properties": {"top_k": _NINT, "min_score": {"type": ["number", "null"]},
                                "returned": {"type": "integer"}, "latency_ms": {"type": "integer"},
                                "generated_at": {"type": "string"}}},
    },
}



# ============================================================================
# PACKAGE messages  (PO Agent context packages, Knowledge-Agent-owned half)
# ============================================================================
PKG_UC1 = "UC1_CONTEXT"          # feeds ContextPackageContract
PKG_UC2 = "UC2_ALIGNMENT"        # feeds AlignmentContextPackage
PACKAGE_TYPES = (PKG_UC1, PKG_UC2)

# Same spellings as the PO Agent's uc2_contracts.ContextCategory / RequirementKind.
# The Knowledge Agent only speaks about these four; RELATED_PBIS and EVIDENCE belong to the PO Agent.
KNOWLEDGE_CATEGORIES = ("PRODUCT_GOAL", "PRODUCT_VISION", "STRATEGIC_THEMES", "REQUIREMENTS_DECISIONS")
REQUIREMENT_KINDS = ("REQUIREMENT", "DECISION")
KNOWLEDGE_ORIGIN = "knowledge_agent"
DEFAULT_PACKAGE_TOP_K = 20
MAX_PACKAGE_TOP_K = MAX_TOP_K


@dataclass
class PackageRequest:
    project_id: str                                         # Knowledge Agent project id (authoritative)
    package_type: str                                       # PKG_UC1 | PKG_UC2
    focus_text: Optional[str] = None                        # what to be relevant to: UC1 = the signal/transcript text,
                                                            # UC2 = the audited PBI's title + story. None = generic.
    top_k: int = DEFAULT_PACKAGE_TOP_K                      # max requirements/decisions returned
    context: Dict[str, Any] = field(default_factory=dict)   # request_id, requester, ... echoed back

    _KEYS = ("project_id", "package_type", "focus_text", "top_k", "context")

    def to_dict(self) -> Dict[str, Any]:
        return {"project_id": self.project_id, "package_type": self.package_type,
                "focus_text": self.focus_text, "top_k": self.top_k, "context": dict(self.context)}

    @classmethod
    def from_dict(cls, d: Any) -> "PackageRequest":
        if isinstance(d, PackageRequest):
            return d
        if not isinstance(d, dict):
            raise InvalidRequest("Request must be an object.")
        unknown = set(d) - set(cls._KEYS)
        if unknown:
            raise InvalidRequest(f"Unknown request field(s): {sorted(unknown)}")
        pid = d.get("project_id")
        if not isinstance(pid, str) or not pid.strip():
            raise InvalidRequest("'project_id' is required and must be a non-empty string.")
        ptype = d.get("package_type")
        if ptype not in PACKAGE_TYPES:
            raise InvalidRequest(f"'package_type' must be one of {list(PACKAGE_TYPES)}.")
        focus = d.get("focus_text")
        if focus is not None:
            if not isinstance(focus, str):
                raise InvalidRequest("'focus_text' must be a string or null.")
            focus = focus.strip()[:MAX_QUERY_CHARS] or None
        top_k = d.get("top_k", DEFAULT_PACKAGE_TOP_K)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_PACKAGE_TOP_K:
            raise InvalidRequest(f"'top_k' must be an integer between 1 and {MAX_PACKAGE_TOP_K}.")
        ctx = d.get("context") or {}
        if not isinstance(ctx, dict):
            raise InvalidRequest("'context' must be an object.")
        return cls(pid.strip(), ptype, focus, top_k, dict(ctx))


def validate_package_request(d: Any) -> PackageRequest:
    return PackageRequest.from_dict(d)


@dataclass
class TextWithSource:
    """Same fields as the PO's ProductGoalContext / ProductVisionContext."""
    text: str
    source: str

    @classmethod
    def from_dict(cls, d, path):
        d = _obj(d, path)
        t, s = _field(d, "text", _S, path), _field(d, "source", _S, path)
        if not t.strip() or not s.strip():
            _fail(path, "text and source must be non-blank")
        return cls(t, s)


@dataclass
class Theme:
    """Same fields as the PO's StrategicTheme."""
    theme_id: str
    name: str
    description: Optional[str]
    source: str

    @classmethod
    def from_dict(cls, d, path):
        d = _obj(d, path)
        out = cls(_field(d, "theme_id", _S, path), _field(d, "name", _S, path),
                  _field(d, "description", _NS, path), _field(d, "source", _S, path))
        if not (out.theme_id.strip() and out.name.strip() and out.source.strip()):
            _fail(path, "theme_id, name and source must be non-blank")
        return out


@dataclass
class Requirement:
    """Same fields as the PO's ProjectRequirement."""
    requirement_id: str
    kind: str                     # REQUIREMENT_KINDS
    text: str
    source: str                   # human-readable citation of where it came from

    @classmethod
    def from_dict(cls, d, path):
        d = _obj(d, path)
        out = cls(_field(d, "requirement_id", _S, path), _field(d, "kind", _S, path, REQUIREMENT_KINDS),
                  _field(d, "text", _S, path), _field(d, "source", _S, path))
        if not (out.requirement_id.strip() and out.text.strip() and out.source.strip()):
            _fail(path, "requirement_id, text and source must be non-blank")
        return out


@dataclass
class MissingItem:
    """Same fields as the PO's MissingContextItem."""
    category: str                 # KNOWLEDGE_CATEGORIES
    reason: str

    @classmethod
    def from_dict(cls, d, path):
        d = _obj(d, path)
        out = cls(_field(d, "category", _S, path, KNOWLEDGE_CATEGORIES), _field(d, "reason", _S, path))
        if not out.reason.strip():
            _fail(path, "reason must be non-blank")
        return out


@dataclass
class Knowledge:
    """The Knowledge-Agent-owned half of a PO context package. Field names match the PO models."""
    origin: str                                              # always 'knowledge_agent'
    product_goal: Optional[TextWithSource]                   # null = Knowledge Agent holds none (never invented)
    product_vision: Optional[TextWithSource]
    strategic_themes: List[Theme]
    requirements: List[Requirement]                          # current requirements + decisions
    project_requirements: List[str]                          # UC1 form: exactly [r.text for r in requirements]
    available_context: List[str]                             # of KNOWLEDGE_CATEGORIES, those present
    missing_context: List[MissingItem]                       # of KNOWLEDGE_CATEGORIES, those absent + reason

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d, path="knowledge") -> "Knowledge":
        d = _obj(d, path)
        goal, vision = _field(d, "product_goal", (dict, type(None)), path), _field(d, "product_vision", (dict, type(None)), path)
        themes, reqs = _field(d, "strategic_themes", (list,), path), _field(d, "requirements", (list,), path)
        out = cls(
            origin=_field(d, "origin", _S, path, (KNOWLEDGE_ORIGIN,)),
            product_goal=TextWithSource.from_dict(goal, f"{path}.product_goal") if goal is not None else None,
            product_vision=TextWithSource.from_dict(vision, f"{path}.product_vision") if vision is not None else None,
            strategic_themes=[Theme.from_dict(t, f"{path}.strategic_themes[{i}]") for i, t in enumerate(themes)],
            requirements=[Requirement.from_dict(r, f"{path}.requirements[{i}]") for i, r in enumerate(reqs)],
            project_requirements=[str(x) for x in _field(d, "project_requirements", (list,), path)],
            available_context=[str(x) for x in _field(d, "available_context", (list,), path)],
            missing_context=[MissingItem.from_dict(m, f"{path}.missing_context[{i}]")
                             for i, m in enumerate(_field(d, "missing_context", (list,), path))],
        )
        if out.project_requirements != [r.text for r in out.requirements]:
            _fail(f"{path}.project_requirements", "must equal the texts of 'requirements', in order")
        if len({r.requirement_id for r in out.requirements}) != len(out.requirements):
            _fail(f"{path}.requirements", "duplicate requirement_id")
        if len({t.theme_id for t in out.strategic_themes}) != len(out.strategic_themes):
            _fail(f"{path}.strategic_themes", "duplicate theme_id")
        bad = [c for c in out.available_context if c not in KNOWLEDGE_CATEGORIES]
        if bad:
            _fail(f"{path}.available_context", f"unknown categories {bad}")
        missing = [m.category for m in out.missing_context]
        if set(out.available_context) & set(missing) or \
                set(out.available_context) | set(missing) != set(KNOWLEDGE_CATEGORIES) or \
                len(set(missing)) != len(missing) or len(set(out.available_context)) != len(out.available_context):
            _fail(path, "available_context and missing_context must split the four categories exactly once")
        present = {"PRODUCT_GOAL": out.product_goal is not None, "PRODUCT_VISION": out.product_vision is not None,
                   "STRATEGIC_THEMES": bool(out.strategic_themes), "REQUIREMENTS_DECISIONS": bool(out.requirements)}
        if {c for c, ok in present.items() if ok} != set(out.available_context):
            _fail(f"{path}.available_context", "does not match which categories actually have content")
        return out


@dataclass
class PackageResponse:
    schema_version: str
    request_id: str
    status: str                          # ok (>=1 category available) | no_results | error
    package_type: Optional[str]
    project: Optional[ProjectRef]
    knowledge: Knowledge                 # always present; empty with reasons when nothing was found
    warnings: List[Warning]
    error: Optional[ErrorInfo]
    context: Dict[str, Any]
    meta: Dict[str, Any]                 # top_k, returned, latency_ms, generated_at

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @classmethod
    def from_dict(cls, d: Any) -> "PackageResponse":
        p = "package_response"
        d = _obj(d, p)
        major = str(_field(d, "schema_version", _S, p)).split(".")[0]
        if major != SCHEMA_VERSION.split(".")[0]:
            _fail(f"{p}.schema_version", f"unsupported major version '{d['schema_version']}'")
        status = _field(d, "status", _S, p, STATUSES)
        error = _field(d, "error", (dict, type(None)), p)
        project = _field(d, "project", (dict, type(None)), p)
        if status == STATUS_ERROR and error is None:
            _fail(f"{p}.error", "status is 'error' but error is null")
        if status != STATUS_ERROR and error is not None:
            _fail(f"{p}.error", "error must be null unless status is 'error'")
        k = Knowledge.from_dict(_field(d, "knowledge", (dict,), p))
        if status == STATUS_OK and not k.available_context:
            _fail(f"{p}.knowledge", "status is 'ok' but no category has content")
        if status != STATUS_OK and k.available_context:
            _fail(f"{p}.status", "knowledge has content, status must be 'ok'")
        return cls(
            schema_version=d["schema_version"], request_id=_field(d, "request_id", _S, p), status=status,
            package_type=_field(d, "package_type", _NS, p, PACKAGE_TYPES),
            project=ProjectRef.from_dict(project) if project is not None else None, knowledge=k,
            warnings=[Warning.from_dict(w, f"{p}.warnings[{i}]") for i, w in enumerate(_field(d, "warnings", (list,), p))],
            error=ErrorInfo.from_dict(error) if error is not None else None,
            context=_field(d, "context", (dict,), p), meta=_field(d, "meta", (dict,), p))


def validate_package_response(d: Any) -> PackageResponse:
    return PackageResponse.from_dict(d)


# ---- JSON Schema for the package messages -------------------------------------------------------------------
_TWS = {"oneOf": [{"type": "null"}, {"type": "object", "required": ["text", "source"],
                                     "properties": {"text": {"type": "string", "minLength": 1},
                                                    "source": {"type": "string", "minLength": 1}}}]}

PACKAGE_REQUEST_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "KnowledgePackageRequest",
    "type": "object", "additionalProperties": False, "required": ["project_id", "package_type"],
    "properties": {
        "project_id": {"type": "string", "minLength": 1},
        "package_type": {"enum": list(PACKAGE_TYPES)},
        "focus_text": {"type": ["string", "null"], "maxLength": MAX_QUERY_CHARS},
        "top_k": {"type": "integer", "minimum": 1, "maximum": MAX_PACKAGE_TOP_K, "default": DEFAULT_PACKAGE_TOP_K},
        "context": {"type": "object"}}}

PACKAGE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "KnowledgePackageResponse",
    "type": "object",
    "required": ["schema_version", "request_id", "status", "package_type", "project", "knowledge",
                 "warnings", "error", "context", "meta"],
    "properties": {
        "schema_version": {"type": "string"}, "request_id": {"type": "string"},
        "status": {"enum": list(STATUSES)}, "package_type": {"enum": list(PACKAGE_TYPES) + [None]},
        "project": CONTEXT_RESPONSE_SCHEMA["properties"]["project"],
        "knowledge": {
            "type": "object",
            "required": ["origin", "product_goal", "product_vision", "strategic_themes", "requirements",
                         "project_requirements", "available_context", "missing_context"],
            "properties": {
                "origin": {"const": KNOWLEDGE_ORIGIN},
                "product_goal": _TWS, "product_vision": _TWS,
                "strategic_themes": {"type": "array", "items": {
                    "type": "object", "required": ["theme_id", "name", "description", "source"],
                    "properties": {"theme_id": {"type": "string"}, "name": {"type": "string"},
                                   "description": _NSTR, "source": {"type": "string"}}}},
                "requirements": {"type": "array", "items": {
                    "type": "object", "required": ["requirement_id", "kind", "text", "source"],
                    "properties": {"requirement_id": {"type": "string"}, "kind": {"enum": list(REQUIREMENT_KINDS)},
                                   "text": {"type": "string"}, "source": {"type": "string"}}}},
                "project_requirements": {"type": "array", "items": {"type": "string"}},
                "available_context": {"type": "array", "items": {"enum": list(KNOWLEDGE_CATEGORIES)}},
                "missing_context": {"type": "array", "items": {
                    "type": "object", "required": ["category", "reason"],
                    "properties": {"category": {"enum": list(KNOWLEDGE_CATEGORIES)}, "reason": {"type": "string"}}}}}},
        "warnings": CONTEXT_RESPONSE_SCHEMA["properties"]["warnings"],
        "error": CONTEXT_RESPONSE_SCHEMA["properties"]["error"],
        "context": {"type": "object"},
        "meta": {"type": "object", "required": ["top_k", "returned", "latency_ms", "generated_at"]}}}



# ============================================================================
# ITEM messages  (PO Agent -> Knowledge Agent: store approved project knowledge)
# ============================================================================
OP_UPSERT = "upsert"             # store a new item, or a new version of an existing item_id
OP_RETIRE = "retire"             # the item is no longer in force (dropped / replaced outside the KB)
OPERATIONS = (OP_UPSERT, OP_RETIRE)

# What the PO Agent may send (subset of DOC_TYPES; srs / glossary / requirement come from ingested documents).
ITEM_TYPES = ("user_story", "meeting_summary", "standup", "retro_feedback", "action_item",
              "decision", "approval_rationale", "refinement_note")
MAX_ITEM_TEXT_CHARS = 8000
MAX_ITEM_TITLE_CHARS = 200
_ITEM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,99}$")

ITEM_STORED = "stored"           # first version of this item_id
ITEM_UPDATED = "updated"         # new version stored; the previous one is now superseded (kept as history)
ITEM_UNCHANGED = "unchanged"     # identical content already current (safe to retry), or already retired
ITEM_RETIRED = "retired"         # current version marked superseded, no successor
ITEM_STATUSES = (ITEM_STORED, ITEM_UPDATED, ITEM_UNCHANGED, ITEM_RETIRED, STATUS_ERROR)


@dataclass
class ItemRequest:
    project_id: str                                         # Knowledge Agent project id
    item_type: str                                          # one of ITEM_TYPES
    item_id: str                                            # the PO's own id (PBI key, proposal id, ...); unique per project+type
    operation: str = OP_UPSERT
    text: Optional[str] = None                              # content to store (required for upsert)
    title: Optional[str] = None                             # short label, shown in citations
    approved: bool = False                                  # MUST be True for upsert: only human-approved items are stored
    context: Dict[str, Any] = field(default_factory=dict)   # request_id, requester, ... echoed back

    _KEYS = ("project_id", "item_type", "item_id", "operation", "text", "title", "approved", "context")

    def to_dict(self) -> Dict[str, Any]:
        return {"project_id": self.project_id, "item_type": self.item_type, "item_id": self.item_id,
                "operation": self.operation, "text": self.text, "title": self.title,
                "approved": self.approved, "context": dict(self.context)}

    @classmethod
    def from_dict(cls, d: Any) -> "ItemRequest":
        if isinstance(d, ItemRequest):
            return d
        if not isinstance(d, dict):
            raise InvalidRequest("Request must be an object.")
        unknown = set(d) - set(cls._KEYS)
        if unknown:
            raise InvalidRequest(f"Unknown request field(s): {sorted(unknown)}")
        pid = d.get("project_id")
        if not isinstance(pid, str) or not pid.strip():
            raise InvalidRequest("'project_id' is required and must be a non-empty string.")
        itype = d.get("item_type")
        if itype not in ITEM_TYPES:
            raise InvalidRequest(f"'item_type' must be one of {list(ITEM_TYPES)}.")
        iid = d.get("item_id")
        if not isinstance(iid, str) or not _ITEM_ID_RE.match(iid):
            raise InvalidRequest("'item_id' must be 1-100 characters: letters, digits and . _ : - (no spaces).")
        op = d.get("operation", OP_UPSERT)
        if op not in OPERATIONS:
            raise InvalidRequest(f"'operation' must be one of {list(OPERATIONS)}.")

        text, title = d.get("text"), d.get("title")
        approved = d.get("approved", False)
        if not isinstance(approved, bool):
            raise InvalidRequest("'approved' must be a boolean.")
        if title is not None:
            if not isinstance(title, str):
                raise InvalidRequest("'title' must be a string or null.")
            title = title.strip()[:MAX_ITEM_TITLE_CHARS] or None
        if op == OP_UPSERT:
            if not isinstance(text, str) or not text.strip():
                raise InvalidRequest("'text' is required for upsert and must be a non-empty string.")
            if len(text) > MAX_ITEM_TEXT_CHARS:
                raise InvalidRequest(f"'text' exceeds {MAX_ITEM_TEXT_CHARS} characters.")
            if approved is not True:
                raise InvalidRequest("Only human-approved items are stored: send approved=true after the "
                                     "Product Owner's approval (never drafts or proposals).")
            text = text.strip()
        else:
            if text is not None:
                raise InvalidRequest("'text' must not be sent with operation 'retire'.")

        ctx = d.get("context") or {}
        if not isinstance(ctx, dict):
            raise InvalidRequest("'context' must be an object.")
        return cls(pid.strip(), itype, iid, op, text, title, approved, dict(ctx))


def validate_item_request(d: Any) -> ItemRequest:
    return ItemRequest.from_dict(d)


@dataclass
class ItemRef:
    item_type: str
    item_id: str
    version: int                              # 1, 2, 3 ... per (project, item_type, item_id)
    chunk_id: str                             # id of this version in the Knowledge Agent
    previous_chunk_id: Optional[str]          # the version it superseded, if any
    approval_status: str                      # 'approved' (current) | 'superseded' (retired)

    @classmethod
    def from_dict(cls, d, path="item"):
        d = _obj(d, path)
        return cls(_field(d, "item_type", _S, path, ITEM_TYPES), _field(d, "item_id", _S, path),
                   _field(d, "version", (int,), path), _field(d, "chunk_id", _S, path),
                   _field(d, "previous_chunk_id", _NS, path),
                   _field(d, "approval_status", _S, path, ("approved", "superseded")))


@dataclass
class ItemResponse:
    schema_version: str
    request_id: str
    status: str                               # ITEM_STATUSES
    project: Optional[ProjectRef]
    item: Optional[ItemRef]                   # null only when status == 'error'
    warnings: List[Warning]
    error: Optional[ErrorInfo]
    context: Dict[str, Any]
    meta: Dict[str, Any]                      # latency_ms, generated_at

    @property
    def ok(self) -> bool:
        return self.status != STATUS_ERROR

    @classmethod
    def from_dict(cls, d: Any) -> "ItemResponse":
        p = "item_response"
        d = _obj(d, p)
        major = str(_field(d, "schema_version", _S, p)).split(".")[0]
        if major != SCHEMA_VERSION.split(".")[0]:
            _fail(f"{p}.schema_version", f"unsupported major version '{d['schema_version']}'")
        status = _field(d, "status", _S, p, ITEM_STATUSES)
        error = _field(d, "error", (dict, type(None)), p)
        item = _field(d, "item", (dict, type(None)), p)
        project = _field(d, "project", (dict, type(None)), p)
        if (status == STATUS_ERROR) != (error is not None):
            _fail(f"{p}.error", "error must be present exactly when status is 'error'")
        if (status == STATUS_ERROR) != (item is None):
            _fail(f"{p}.item", "item must be null exactly when status is 'error'")
        ref = ItemRef.from_dict(item) if item is not None else None
        if ref and ((status == ITEM_RETIRED) and ref.approval_status != "superseded"):
            _fail(f"{p}.item.approval_status", "a retired item must be 'superseded'")
        return cls(
            schema_version=d["schema_version"], request_id=_field(d, "request_id", _S, p), status=status,
            project=ProjectRef.from_dict(project) if project is not None else None, item=ref,
            warnings=[Warning.from_dict(w, f"{p}.warnings[{i}]") for i, w in enumerate(_field(d, "warnings", (list,), p))],
            error=ErrorInfo.from_dict(error) if error is not None else None,
            context=_field(d, "context", (dict,), p), meta=_field(d, "meta", (dict,), p))


def validate_item_response(d: Any) -> ItemResponse:
    return ItemResponse.from_dict(d)


ITEM_REQUEST_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "KnowledgeItemRequest",
    "type": "object", "additionalProperties": False, "required": ["project_id", "item_type", "item_id"],
    "properties": {
        "project_id": {"type": "string", "minLength": 1},
        "item_type": {"enum": list(ITEM_TYPES)},
        "item_id": {"type": "string", "pattern": _ITEM_ID_RE.pattern},
        "operation": {"enum": list(OPERATIONS), "default": OP_UPSERT},
        "text": {"type": ["string", "null"], "maxLength": MAX_ITEM_TEXT_CHARS},
        "title": {"type": ["string", "null"], "maxLength": MAX_ITEM_TITLE_CHARS},
        "approved": {"type": "boolean", "default": False, "description": "must be true for upsert"},
        "context": {"type": "object"}},
    "if": {"properties": {"operation": {"const": OP_RETIRE}}, "required": ["operation"]},
    "else": {"required": ["text", "approved"], "properties": {"approved": {"const": True}}}}

ITEM_RESPONSE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "KnowledgeItemResponse",
    "type": "object",
    "required": ["schema_version", "request_id", "status", "project", "item", "warnings", "error", "context", "meta"],
    "properties": {
        "schema_version": {"type": "string"}, "request_id": {"type": "string"},
        "status": {"enum": list(ITEM_STATUSES)},
        "project": CONTEXT_RESPONSE_SCHEMA["properties"]["project"],
        "item": {"oneOf": [{"type": "null"}, {
            "type": "object",
            "required": ["item_type", "item_id", "version", "chunk_id", "previous_chunk_id", "approval_status"],
            "properties": {"item_type": {"enum": list(ITEM_TYPES)}, "item_id": {"type": "string"},
                           "version": {"type": "integer", "minimum": 1}, "chunk_id": {"type": "string"},
                           "previous_chunk_id": _NSTR, "approval_status": {"enum": ["approved", "superseded"]}}}]},
        "warnings": CONTEXT_RESPONSE_SCHEMA["properties"]["warnings"],
        "error": CONTEXT_RESPONSE_SCHEMA["properties"]["error"],
        "context": {"type": "object"},
        "meta": {"type": "object", "required": ["latency_ms", "generated_at"]}}}


if __name__ == "__main__":
    # Self-check: python po_knowledge_contract.py
    import json
    req = ContextRequest(project_id="proj_demo", query="What are the login requirements?",
                         versions=["v1.0"], context={"requester": "po_agent", "purpose": "story_generation"})
    print("REQUEST ->", json.dumps(req.to_dict(), indent=2))
    sample = {
        "schema_version": SCHEMA_VERSION, "request_id": "req_demo", "status": STATUS_NO_RESULTS,
        "project": {"project_id": "proj_demo", "name": "Demo"}, "query": req.query,
        "versions": {"requested": ["1"], "available": [], "latest": None, "returned": [], "include_history": True},
        "passages": [], "answer": None, "warnings": [], "error": None, "context": {},
        "meta": {"top_k": 5, "min_score": None, "returned": 0, "latency_ms": 0, "generated_at": "now"},
    }
    print("RESPONSE parsed ->", validate_response(sample).status)