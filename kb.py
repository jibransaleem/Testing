"""
kb.py - simple interface on top of a2a_client.py (keep both files in the same folder).

    import kb
    kb.ping()
    kb.ask("proj_project1", "What does the system do?")
    kb.add_meeting_summary("proj_project1", "MTG-001", "Sprint planning", "Decisions: ... Action items: ...")
    kb.retire_item("proj_project1", "meeting_summary", "MTG-001")
    kb.send("put_item", {...any JSON...})          # generic: which skill + the JSON

Terminal:
    python kb.py ping
    python kb.py ask proj_project1 "What does the system do?"
    python kb.py send put_item meeting.json

Set the address/key below or with env vars KB_URL / KB_KEY. Needs: pip install httpx
"""
import json
import os
import sys

from a2a_client import KnowledgeA2AClient, A2AError

KB_URL = os.getenv("KB_URL", "https://testing-eta-orpin.vercel.app")
KB_KEY = os.getenv("KB_KEY", "kbKey2026Fyp8a7d3f")       # do not commit a real key to Git

KBError = A2AError                                     # all problems raise this, with a plain message
_client = None


def _kb():
    global _client
    if _client is None:
        _client = KnowledgeA2AClient(KB_URL, KB_KEY)
    return _client


def send(skill, request=None):
    """Generic: skill = ping | get_context | get_package | put_item, request = plain dict (the JSON)."""
    return _kb().call(skill, request or {})


def ping():
    return send("ping")


def ask(project_id, question, top_k=None):
    """Returns the answer dict; the evidence is in result['passages']."""
    req = {"project_id": project_id, "query": question}
    if top_k:
        req["top_k"] = top_k
    return send("get_context", req)


def add_item(project_id, item_type, item_id, title, text, approved=True, **extra):
    """Store/update one item. Same item_id again = new version."""
    return send("put_item", {"project_id": project_id, "item_type": item_type, "item_id": item_id,
                             "title": title, "text": text, "approved": approved, **extra})


def add_meeting_summary(project_id, item_id, title, text, approved=True):
    return add_item(project_id, "meeting_summary", item_id, title, text, approved)


def retire_item(project_id, item_type, item_id):
    return send("put_item", {"project_id": project_id, "item_type": item_type,
                             "item_id": item_id, "operation": "retire"})


def _main(argv):
    try:
        if argv[:1] == ["ping"]:
            out = ping()
        elif argv[:1] == ["ask"] and len(argv) >= 3:
            out = ask(argv[1], " ".join(argv[2:]))
        elif argv[:1] == ["send"] and len(argv) >= 3:
            out = send(argv[1], json.load(open(argv[2], encoding="utf-8")))
        else:
            print(__doc__)
            return
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    except KBError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    _main(sys.argv[1:])