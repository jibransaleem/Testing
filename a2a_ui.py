"""a2a_ui.py - HTML of the log viewer (same as a2a_ui.html), served at /ui by a2a_server.py."""

UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowledge Agent - A2A logs</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --text:#1c1f24; --muted:#6b7280; --line:#e3e6ea;
          --ok:#127a3d; --okbg:#e4f5ea; --err:#b42318; --errbg:#fde8e6; --info:#1d4ed8; --infobg:#e5edff; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#15171a; --card:#1e2125; --text:#e8eaed; --muted:#9aa1ab; --line:#30353b;
            --ok:#5fd08a; --okbg:#173325; --err:#ff8a80; --errbg:#3b1d1b; --info:#8fb0ff; --infobg:#1b2744; }
  }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; background:var(--bg); color:var(--text); }
  header { padding:14px 20px; background:var(--card); border-bottom:1px solid var(--line);
           display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:space-between; }
  h1 { font-size:16px; margin:0; }
  .sub { color:var(--muted); font-size:12px; }
  .bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  input, select, button { font:inherit; padding:6px 10px; border:1px solid var(--line); border-radius:6px;
                          background:var(--card); color:var(--text); }
  button { cursor:pointer; }
  main { padding:16px 20px; max-width:1100px; margin:0 auto; }
  .stats { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }
  .stat { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 14px; }
  .stat b { display:block; font-size:18px; }
  .row { background:var(--card); border:1px solid var(--line); border-radius:8px; margin-bottom:8px; }
  .head { display:grid; grid-template-columns:185px 110px 110px 70px 1fr; gap:10px; padding:9px 12px;
          align-items:center; cursor:pointer; }
  .head:hover { background:var(--bg); }
  .badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; font-weight:600;
           background:var(--infobg); color:var(--info); }
  .badge.ok { background:var(--okbg); color:var(--ok); }
  .badge.err { background:var(--errbg); color:var(--err); }
  .muted { color:var(--muted); }
  .sum { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .detail { display:none; padding:0 12px 12px; border-top:1px solid var(--line); }
  .row.open .detail { display:block; }
  .cols { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:10px; }
  .cols h3 { margin:0 0 4px; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  pre { margin:0; padding:10px; background:var(--bg); border:1px solid var(--line); border-radius:6px;
        overflow:auto; max-height:380px; font:12px/1.4 ui-monospace,Menlo,Consolas,monospace; white-space:pre-wrap; word-break:break-word; }
  .empty { text-align:center; color:var(--muted); padding:40px 0; }
  .err-msg { color:var(--err); margin-top:8px; }
  @media (max-width:760px) { .head { grid-template-columns:1fr 1fr; } .cols { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <div>
    <h1>Knowledge Agent &middot; A2A logs</h1>
    <div class="sub" id="conn">not connected</div>
  </div>
  <div class="bar">
    <input id="url" placeholder="Server URL" size="28" title="KB server base URL">
    <input id="key" type="password" placeholder="API key (A2A_API_KEY)" size="22">
    <select id="skill">
      <option value="">all skills</option>
      <option>ping</option><option>get_context</option><option>get_package</option><option>put_item</option>
    </select>
    <select id="out">
      <option value="">all results</option><option value="ok">ok</option><option value="err">errors only</option>
    </select>
    <label class="sub"><input id="auto" type="checkbox" checked> auto-refresh</label>
    <button id="go">Refresh</button>
  </div>
</header>
<main>
  <div class="stats" id="stats"></div>
  <div id="list"><div class="empty">Enter the server URL and your key, then press Refresh.</div></div>
</main>
<script>
const $ = id => document.getElementById(id);
const GOOD = new Set(["ok","stored","unchanged","updated","retired","no_results","completed"]);
let timer = null, open = new Set(), last = [];

// default server: the page's own origin when served by the KB (/ui); otherwise localhost
const here = location.protocol.startsWith("http") ? location.origin : "http://localhost:8000";
$("url").value = localStorage.getItem("kb_url") || here;
$("key").value = localStorage.getItem("kb_key") || "";
const base = () => $("url").value.trim().replace(/\/+$/, "");

function isErr(l) { return !GOOD.has(l.outcome); }

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function summary(l) {
  const r = l.request || {};
  if (r.query) return r.query;
  if (r.item_id) return (r.operation || "upsert") + " " + (r.item_type || "") + " " + r.item_id;
  if (r.package_type) return r.package_type + (r.focus_text ? ": " + r.focus_text : "");
  return l.error || "";
}

function render() {
  const skill = $("skill").value, out = $("out").value;
  const rows = last.filter(l => (!skill || l.skill === skill) &&
    (!out || (out === "err" ? isErr(l) : !isErr(l))));

  const stats = $("stats"); stats.replaceChildren();
  [["Total", last.length], ["OK", last.filter(l => !isErr(l)).length], ["Errors", last.filter(isErr).length],
   ["Avg ms", last.length ? Math.round(last.reduce((a, l) => a + (l.latency_ms || 0), 0) / last.length) : 0]]
    .forEach(([k, v]) => { const s = el("div", "stat"); s.append(el("span", "muted", k), el("b", "", v)); stats.append(s); });

  const list = $("list"); list.replaceChildren();
  if (!rows.length) { list.append(el("div", "empty", "No requests yet.")); return; }

  rows.forEach(l => {
    const row = el("div", "row" + (open.has(l.id) ? " open" : ""));
    const head = el("div", "head");
    head.append(el("span", "muted", l.time));
    head.append(el("span", "badge", l.skill || l.method || "-"));
    head.append(el("span", "badge " + (isErr(l) ? "err" : "ok"), l.outcome));
    head.append(el("span", "muted", (l.latency_ms ?? "-") + " ms"));
    head.append(el("span", "sum", summary(l)));
    head.onclick = () => { open.has(l.id) ? open.delete(l.id) : open.add(l.id); row.classList.toggle("open"); };

    const d = el("div", "detail");
    d.append(el("div", "sub", "id " + l.id + " | from " + (l.client || "?") + " | method " + (l.method || "-") +
                " | http " + (l.http ?? 200)));
    if (l.error) d.append(el("div", "err-msg", l.error));
    const cols = el("div", "cols");
    const a = el("div"), b = el("div");
    a.append(el("h3", "", "Request (from PO agent)"), el("pre", "", JSON.stringify(l.request ?? null, null, 2)));
    b.append(el("h3", "", "Response (from KB)"), el("pre", "", JSON.stringify(l.response ?? null, null, 2)));
    cols.append(a, b); d.append(cols);
    row.append(head, d); list.append(row);
  });
}

async function load() {
  const key = $("key").value.trim();
  localStorage.setItem("kb_key", key);
  localStorage.setItem("kb_url", $("url").value.trim());
  try {
    const r = await fetch(base() + "/logs", { headers: key ? { Authorization: "Bearer " + key } : {} });
    if (r.status === 401) { $("conn").textContent = "wrong or missing API key"; return; }
    const data = await r.json();
    last = data.logs || [];
    $("conn").textContent = "connected | " + last.length + " logged request(s) | updated " + new Date().toLocaleTimeString();
    render();
  } catch (e) {
    $("conn").textContent = "cannot reach " + base() + " (is the server running? CORS?) - " + e;
  }
}

function schedule() {
  clearInterval(timer);
  if ($("auto").checked) timer = setInterval(load, 3000);
}

$("go").onclick = load;
$("auto").onchange = schedule;
$("skill").onchange = $("out").onchange = render;
schedule();
load();
</script>
</body>
</html>
"""