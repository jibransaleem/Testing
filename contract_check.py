"""
contract_check.py  --  checks the PO <-> Knowledge contract with dummy data. NO pytest needed.

Run:   python contract_check.py

Prints every step: what is sent, what comes back, and PASS / FAIL.
Dummy data only: no PDFs, no database, no embeddings, no API keys.
Needs the real po_knowledge_contract.py and knowledge_interface.py in the same folder.
Exit code is 0 if everything passed, 1 otherwise.
"""
from __future__ import annotations

import json
import sys
import traceback

try:
    import po_knowledge_contract as C
    from knowledge_interface import KnowledgeInterface
except Exception as e:  # noqa: BLE001
    sys.exit(f"Cannot import the contract / interface: {type(e).__name__}: {e}\n"
             "Make sure the REAL po_knowledge_contract.py is in this folder.")

PROJECT, NAME = "proj_demo", "Demo Project"
STORY = "As a user I want to reset my password so that I can regain access."
results = []          # (name, passed)


# ------------------------------------------------------------------ printing helpers
def show(label: str, obj, limit: int = 230) -> None:
    text = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    print(f"      {label}: {text[:limit]}{' ...' if len(text) > limit else ''}")


def check(name: str, fn) -> None:
    print(f"\n[{len(results) + 1}] {name}")
    try:
        fn()
    except AssertionError as e:
        print(f"      FAIL  {e or 'assertion failed'}")
        results.append((name, False))
    except Exception as e:  # noqa: BLE001
        print(f"      FAIL  unexpected {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        results.append((name, False))
    else:
        print("      PASS")
        results.append((name, True))


def must_reject(exc, fn, *args):
    """Calls fn(*args); passes only if it raises `exc`."""
    try:
        fn(*args)
    except exc as e:
        print(f"      rejected as expected -> {type(e).__name__}: {str(e)[:110]}")
        return
    raise AssertionError("was ACCEPTED but should have been rejected")


# ------------------------------------------------------------------ dummy messages
def ctx_request(**o):
    return {"project_id": PROJECT, "query": "which database does the system use", **o}


def passage(i=1):
    return {"passage_id": f"p{i}", "rank": i, "text": "The system uses PostgreSQL.", "score": 0.8,
            "is_current": True, "version": "1", "citation": f"Demo (v1), 3.1 Database, p.{i}",
            "source": {"project_id": PROJECT, "project_name": NAME, "document_id": "doc_demo_v1",
                       "document_title": "Demo", "document_version": "1", "document_status": "current",
                       "doc_type": "srs", "section": "3.1 Database", "page_start": i, "page_end": i,
                       "approval_status": "approved", "decision_id": None, "source_event": None,
                       "source_id": "doc_demo_v1"}}


def ctx_response(**o):
    b = {"schema_version": C.SCHEMA_VERSION, "request_id": "r1", "status": "ok",
         "project": {"project_id": PROJECT, "name": NAME}, "query": "q",
         "versions": {"requested": [], "available": [], "latest": "1", "returned": ["1"], "include_history": False},
         "passages": [passage()], "answer": None, "warnings": [], "error": None, "context": {},
         "meta": {"top_k": 5, "min_score": 0.0, "returned": 1, "latency_ms": 1, "generated_at": "t"}}
    b.update(o)
    return b


def pkg_request(**o):
    return {"project_id": PROJECT, "package_type": "UC1_CONTEXT", **o}


def pkg_response(**o):
    cats = [{"category": c, "reason": "none"} for c in ("PRODUCT_GOAL", "PRODUCT_VISION", "STRATEGIC_THEMES")]
    b = {"schema_version": C.SCHEMA_VERSION, "request_id": "r1", "status": "ok", "package_type": "UC1_CONTEXT",
         "project": {"project_id": PROJECT, "name": NAME},
         "knowledge": {"origin": "knowledge_agent", "product_goal": None, "product_vision": None,
                       "strategic_themes": [],
                       "requirements": [{"requirement_id": "r1", "kind": "REQUIREMENT", "text": "Admins only.", "source": "S1"}],
                       "project_requirements": ["Admins only."], "available_context": ["REQUIREMENTS_DECISIONS"],
                       "missing_context": cats},
         "warnings": [], "error": None, "context": {"request_id": "r1"},
         "meta": {"top_k": 20, "returned": 1, "latency_ms": 1, "generated_at": "t"}}
    b.update(o)
    return b


def item_request(**o):
    return {"project_id": PROJECT, "item_type": "user_story", "item_id": "PBI-1", "text": STORY, "approved": True, **o}


def item_response(**o):
    b = {"schema_version": C.SCHEMA_VERSION, "request_id": "r1", "status": "stored",
         "project": {"project_id": PROJECT, "name": NAME},
         "item": {"item_type": "user_story", "item_id": "PBI-1", "version": 1, "chunk_id": "x",
                  "previous_chunk_id": None, "approval_status": "approved"},
         "warnings": [], "error": None, "context": {}, "meta": {"latency_ms": 1, "generated_at": "t"}}
    b.update(o)
    return b


# ------------------------------------------------------------------ dummy agents
class GoodAgent:
    def get_context(self, r): return ctx_response()
    def get_package(self, r): return pkg_response()
    def put_item(self, r): return item_response()
    def list_projects(self): return [{"project_id": PROJECT, "name": NAME}]


class BrokenAgent:
    def get_context(self, r): raise RuntimeError("boom")
    def get_package(self, r): raise RuntimeError("boom")
    def put_item(self, r): raise RuntimeError("boom")


class SloppyAgent:
    def get_context(self, r): return {"status": "ok"}
    def get_package(self, r): return {"status": "ok"}
    def put_item(self, r): return {"status": "stored"}


# ------------------------------------------------------------------ 1. requests
print("=" * 78 + "\n1. REQUESTS  (what the PO Agent sends)\n" + "=" * 78)


def t_good_requests():
    for label, cls, req in (("context", C.ContextRequest, ctx_request()), ("package", C.PackageRequest, pkg_request()),
                            ("item", C.ItemRequest, item_request())):
        show(f"{label} request", req)
        parsed = cls.from_dict(req)
        assert parsed.project_id == PROJECT
        print(f"      accepted -> {cls.__name__}(project_id={parsed.project_id!r})")


check("valid context / package / item requests are accepted", t_good_requests)

for label, cls, bad_list in (
    ("context", C.ContextRequest, [({}, "empty"), (ctx_request(query="  "), "blank query"), (ctx_request(top_k=0), "top_k=0"),
                                   (ctx_request(top_k=99), "top_k=99"), (ctx_request(bogus=1), "unknown field"), ("text", "not a dict")]),
    ("package", C.PackageRequest, [({}, "empty"), (pkg_request(package_type="NOPE"), "bad package_type"),
                                   (pkg_request(top_k=0), "top_k=0"), (pkg_request(extra=1), "unknown field")]),
    ("item", C.ItemRequest, [(item_request(approved=False), "not approved"), (item_request(text=" "), "blank text"),
                             (item_request(item_type="srs"), "item_type not allowed"), (item_request(item_id="has space"), "bad item_id"),
                             (item_request(operation="delete"), "bad operation")]),
):
    for bad, why in bad_list:
        check(f"bad {label} request rejected: {why}",
              lambda cls=cls, bad=bad: (show("sent", bad), must_reject(C.InvalidRequest, cls.from_dict, bad)))

# ------------------------------------------------------------------ 2. responses
print("\n" + "=" * 78 + "\n2. RESPONSES  (what the Knowledge Agent sends back)\n" + "=" * 78)


def t_good_responses():
    r = C.ContextResponse.from_dict(ctx_response())
    show("context response", ctx_response())
    assert r.ok and r.passages[0].source.project_id == PROJECT
    print(f"      parsed -> ok={r.ok}, passages={len(r.passages)}, citation={r.citations[0]!r}")
    p = C.PackageResponse.from_dict(pkg_response())
    assert p.ok
    print(f"      package parsed -> ok={p.ok}, requirements={p.knowledge.project_requirements}")
    i = C.ItemResponse.from_dict(item_response())
    assert i.ok
    print(f"      item parsed -> status={i.status}")


check("valid responses are accepted", t_good_responses)

for label, cls, bad_list in (
    ("context", C.ContextResponse, [(ctx_response(status="error"), "status error but no error object"),
                                    (ctx_response(passages=[]), "status ok but no passages"),
                                    (ctx_response(schema_version="9.9"), "wrong schema_version"),
                                    (ctx_response(status="bogus"), "unknown status")]),
    ("package", C.PackageResponse, [(pkg_response(status="error"), "status error without error"),
                                    (pkg_response(status="no_results"), "no_results but has content")]),
    ("item", C.ItemResponse, [(item_response(status="error"), "status error without error"),
                              (item_response(item=None), "stored but no item")]),
):
    for bad, why in bad_list:
        check(f"inconsistent {label} response refused: {why}",
              lambda cls=cls, bad=bad: must_reject(C.ContractError, cls.from_dict, bad))

# ------------------------------------------------------------------ 3. interface
print("\n" + "=" * 78 + "\n3. KnowledgeInterface  (the front door, with dummy agents)\n" + "=" * 78)


def t_good_agent():
    ki = KnowledgeInterface(GoodAgent())
    for label, call, parse in (("handle", lambda: ki.handle(ctx_request()), C.ContextResponse.from_dict),
                               ("get_context", lambda: ki.get_context(ctx_request()), C.ContextResponse.from_dict),
                               ("get_package", lambda: ki.get_package(pkg_request()), C.PackageResponse.from_dict),
                               ("put_item", lambda: ki.put_item(item_request()), C.ItemResponse.from_dict)):
        out = call()
        show(f"{label} ->", out, 140)
        assert parse(out).status in ("ok", "stored")
    assert ki.projects() == [{"project_id": PROJECT, "name": NAME}]
    print(f"      projects() -> {ki.projects()}")


check("good agent: replies pass through the interface", t_good_agent)


def t_error_codes(agent_cls):
    def run():
        ki = KnowledgeInterface(agent_cls())
        a, b, c = ki.handle(ctx_request()), ki.get_package(pkg_request()), ki.put_item(item_request())
        for label, out in (("handle", a), ("get_package", b), ("put_item", c)):
            print(f"      {label} -> status={out['status']!r}, error.code={out['error']['code']!r}, "
                  f"message={out['error']['message'][:60]!r}")
        assert a["error"]["code"] == "INTERNAL_ERROR"
        assert C.PackageResponse.from_dict(b).error.code == "INTERNAL_ERROR"
        assert C.ItemResponse.from_dict(c).error.code == "INTERNAL_ERROR"
    return run


check("agent that CRASHES -> interface returns INTERNAL_ERROR, never raises", t_error_codes(BrokenAgent))
check("agent that returns a CONTRACT-BREAKING reply -> INTERNAL_ERROR", t_error_codes(SloppyAgent))


def t_request_id():
    ki = KnowledgeInterface(BrokenAgent())
    for label, call in (("handle", ki.handle), ("get_package", ki.get_package), ("put_item", ki.put_item)):
        out = call({"context": {"request_id": "keep-me"}})
        print(f"      {label}: request_id -> {out['request_id']!r}")
        assert out["request_id"] == "keep-me"


check("request_id is kept on errors", t_request_id)


def t_garbage():
    ki = KnowledgeInterface(BrokenAgent())
    for g in (None, "text", 42, [], {}):
        for label, call in (("handle", ki.handle), ("get_package", ki.get_package), ("put_item", ki.put_item)):
            assert call(g)["status"] == "error", (label, g)
        print(f"      input {g!r} -> all three calls returned status='error' (no exception)")


check("garbage input never raises", t_garbage)

# ------------------------------------------------------------------ 4. schemas
print("\n" + "=" * 78 + "\n4. JSON SCHEMAS  (needs: pip install jsonschema)\n" + "=" * 78)


def t_schemas():
    try:
        import jsonschema
    except ImportError:
        print("      SKIPPED (jsonschema not installed)")
        return
    for label, msg, schema in (("context request", ctx_request(), C.CONTEXT_REQUEST_SCHEMA),
                               ("context response", ctx_response(), C.CONTEXT_RESPONSE_SCHEMA),
                               ("package request", pkg_request(), C.PACKAGE_REQUEST_SCHEMA),
                               ("package response", pkg_response(), C.PACKAGE_RESPONSE_SCHEMA),
                               ("item request", item_request(), C.ITEM_REQUEST_SCHEMA),
                               ("item response", item_response(), C.ITEM_RESPONSE_SCHEMA)):
        jsonschema.validate(msg, schema)
        print(f"      {label}: valid against schema")
    must_reject(jsonschema.ValidationError, jsonschema.validate, ctx_request(top_k=99), C.CONTEXT_REQUEST_SCHEMA)


check("dummy messages match the JSON schemas", t_schemas)


# ------------------------------------------------------------------ 5. PO sends items to the KB
print("\n" + "=" * 78 + "\n5. PO AGENT SENDS ITEMS TO STORE  (meeting summary, notes, action item ...)\n" + "=" * 78)

PO_ITEMS = [
    ("meeting_summary", "MTG-1", "Sprint 4 planning", "Team committed to the login epic. Invoices moved to sprint 5."),
    ("standup", "SU-1", "Standup 12 Nov", "Ali finished OTP login. Sara is blocked on the Pusher key."),
    ("retro_feedback", "RT-1", "Sprint 3 retro", "Went well: fast reviews. Improve: testing started too late."),
    ("action_item", "ACT-1", "Get Pusher key", "Sara to request the Pusher API key from the client by Friday."),
    ("user_story", "PBI-12", "Reset password", STORY),
    ("decision", "DEC-1", "Hosting", "We will deploy with Docker on Linux using PostgreSQL."),
]


class DummyStoreAgent:
    """In-memory stand-in for the KB: remembers what the PO sent and answers like the real one."""
    def __init__(self):
        self.db = {}                                   # (project, type, id) -> list of texts (versions)

    def put_item(self, req):
        r = C.ItemRequest.from_dict(req)
        key = (r.project_id, r.item_type, r.item_id)
        if r.operation == "retire":
            status, ver, appr = ("retired", len(self.db[key]), "superseded") if key in self.db else (None, 0, "")
        else:
            versions = self.db.setdefault(key, [])
            if versions and versions[-1] == r.text:
                status = "unchanged"
            else:
                versions.append(r.text)
                status = "stored" if len(versions) == 1 else "updated"
            ver, appr = len(versions), "approved"
        return {"schema_version": C.SCHEMA_VERSION, "request_id": (r.context or {}).get("request_id", "r1"),
                "status": status, "project": {"project_id": PROJECT, "name": NAME},
                "item": {"item_type": r.item_type, "item_id": r.item_id, "version": ver, "chunk_id": f"{'__'.join(key)}__v{ver}",
                         "previous_chunk_id": None, "approval_status": appr},
                "warnings": [], "error": None, "context": r.context or {}, "meta": {"latency_ms": 1, "generated_at": "t"}}


store_agent = DummyStoreAgent()
store_ki = KnowledgeInterface(store_agent)


def make_send_check(item_type, item_id, title, text):
    def run():
        req = item_request(item_type=item_type, item_id=item_id, title=title, text=text,
                           context={"requester": "po_agent", "request_id": f"po-{item_id}"})
        print(f"      PO SENDS : {item_type} {item_id} \"{title}\" approved=True")
        resp = C.ItemResponse.from_dict(store_ki.put_item(req))
        print(f"      KB ANSWERS: {resp.status.upper()} v{resp.item.version} chunk={resp.item.chunk_id}")
        assert resp.status == "stored" and resp.item.version == 1
        assert (PROJECT, item_type, item_id) in store_agent.db
        print(f"      added to dummy KB -> {len(store_agent.db)} item(s) now stored")
    return run


for _t, _i, _title, _text in PO_ITEMS:
    check(f"PO sends a NEW {_t} -> stored", make_send_check(_t, _i, _title, _text))


def t_resend_same():
    r = C.ItemResponse.from_dict(store_ki.put_item(item_request(item_type="meeting_summary", item_id="MTG-1",
                                                                 title="Sprint 4 planning", text=PO_ITEMS[0][3])))
    print(f"      same meeting summary again -> {r.status.upper()} v{r.item.version}")
    assert r.status == "unchanged" and r.item.version == 1


def t_changed():
    r = C.ItemResponse.from_dict(store_ki.put_item(item_request(item_type="meeting_summary", item_id="MTG-1",
                                                                 text=PO_ITEMS[0][3] + " Retro on Friday.")))
    print(f"      changed meeting summary -> {r.status.upper()} v{r.item.version}")
    assert r.status == "updated" and r.item.version == 2


def t_retire():
    r = C.ItemResponse.from_dict(store_ki.put_item({"project_id": PROJECT, "item_type": "action_item", "item_id": "ACT-1",
                                                     "operation": "retire"}))
    print(f"      retire action item -> {r.status.upper()} approval_status={r.item.approval_status}")
    assert r.status == "retired" and r.item.approval_status == "superseded"


def t_unapproved():
    bad = item_request(item_type="action_item", item_id="ACT-9", approved=False)
    print(f"      PO SENDS : action_item ACT-9 approved=False")
    out = store_ki.put_item(bad)
    print(f"      KB ANSWERS: status={out['status']!r} error.code={out['error']['code']!r}")
    assert out["status"] == "error" and out["error"]["code"] in ("INVALID_REQUEST", "INTERNAL_ERROR")
    assert (PROJECT, "action_item", "ACT-9") not in store_agent.db
    print("      nothing was stored")


check("PO re-sends the SAME meeting summary -> unchanged (safe retry)", t_resend_same)
check("PO sends a CHANGED meeting summary -> updated (new version)", t_changed)
check("PO retires an action item -> retired", t_retire)
check("PO sends an UNAPPROVED item -> refused, nothing stored", t_unapproved)
check("every item type the contract allows can be sent", lambda: [
    (C.ItemRequest.from_dict(item_request(item_type=t, item_id=f"X-{t}", text=f"Sample {t}.")),
     print(f"      accepted item_type={t}"))
    for t in C.ITEM_TYPES])

# ------------------------------------------------------------------ summary
passed = sum(ok for _, ok in results)
print("\n" + "=" * 78)
print(f"RESULT: {passed}/{len(results)} checks passed")
for name, ok in results:
    if not ok:
        print(f"  FAILED: {name}")
print("=" * 78)
sys.exit(0 if passed == len(results) else 1)