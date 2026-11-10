"""Vengram: a Vapi custom LLM that improves its own weights.
Public routes (through the tunnel), one secret token per project:
  POST /p/<token>/v1/chat/completions   OpenAI-compatible model endpoint for Vapi
  POST /p/<token>/webhook               Vapi server messages; forwarded to the project's own webhook at once
The dashboard and /api/* answer only on this computer.
The server opens its own ngrok tunnel unless --public-url names an address that already reaches it.
Usage: python server.py --model /path/to/model [--port 8790] [--public-url https://...]"""
import argparse, atexit, json, os, queue, re, secrets, shutil, signal, subprocess, sys, threading, time, traceback, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import brain
from brain import BenchError

HERE = os.path.dirname(os.path.abspath(__file__)); ARGS = None; JOBS = queue.Queue(); TURNS = {}; SEEN = set()
def config(): return brain.load("config.json", dict(public_url="", daily_edit_cap=10))
def projects(): return brain.load("projects.json", [])
def calls(): return brain.load("calls.json", [])
def project_by_token(token): return next((p for p in projects() if secrets.compare_digest(p["token"], token)), None)
def update_call(cid, **kw):
    with brain.STORE:
        rows = calls(); row = next((r for r in rows if r["id"] == cid), None)
        if row is None: row = dict(id=cid); rows.insert(0, row)
        row.update(kw); brain.save("calls.json", rows[:500])
def edits_today(): return sum(1 for r in calls() if r.get("edit_started", "").startswith(time.strftime("%Y-%m-%d")))

# ---------- public address ----------
def open_tunnel(port, tries=8):
    """Start ngrok for this port and return its public https address.
    After a restart, ngrok can report the old endpoint as online for some seconds, so that case is tried again."""
    for attempt in range(tries):
        try: return _open_tunnel(port)
        except RuntimeError as err:
            if "ERR_NGROK_334" not in str(err) or attempt == tries - 1: raise SystemExit(f"ngrok failed: {err}")
            time.sleep(5)
def _open_tunnel(port):
    exe = shutil.which("ngrok") or "/opt/homebrew/bin/ngrok"
    if not os.path.exists(exe): raise SystemExit("ngrok is not installed. Install it, or pass --public-url.")
    proc = subprocess.Popen([exe, "http", f"127.0.0.1:{port}", "--log=stdout", "--log-format=json", "--inspect=false"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    atexit.register(proc.terminate); deadline = time.time() + 30
    for line in proc.stdout:
        try: rec = json.loads(line)
        except json.JSONDecodeError: continue
        if rec.get("url", "").startswith("https://"):
            threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start(); return rec["url"]
        if rec.get("lvl") in ("eror", "crit"): proc.terminate(); raise RuntimeError(rec.get("err") or rec.get("msg"))
        if time.time() > deadline: break
    raise SystemExit("ngrok did not report a public address.")

# ---------- forwarding: the customer's own webhook gets every message at once ----------
def forward(url, raw, headers):
    def run():
        try:
            req = urllib.request.Request(url, data=raw, method="POST", headers={k: v for k, v in headers.items() if k.lower() in ("content-type", "user-agent", "x-vapi-signature")})
            with urllib.request.urlopen(req, timeout=10) as r: code = r.status
        except urllib.error.HTTPError as err: code = err.code
        except Exception as err: code = f"failed: {type(err).__name__}"
        brain.append("forward.jsonl", dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), url=url, status=code))
    threading.Thread(target=run, daemon=True).start()

# ---------- judging and editing ----------
def worker():
    while True:
        job = JOBS.get()
        try: handle(job)
        except Exception as err:
            traceback.print_exc(); update_call(job["id"], status="error", error=f"{type(err).__name__}: {err}")
def handle(job):
    p = next((x for x in projects() if x["id"] == job["project"]), None); turns = job["turns"]
    if p is None: return update_call(job["id"], status="project removed")
    if not turns: return update_call(job["id"], status="not judged", error="This server answered no turn of this call, so there is nothing to judge.")
    system = next((m.get("content") for m in turns[0]["messages"] if m.get("role") == "system"), "")
    update_call(job["id"], status="judging")
    g = brain.grade(system, job["transcript"], turns)
    u = g["update_model_instructions"]; update_call(job["id"], status="judged", grade=dict(passed=g["passed"], score=g["score"], summary=g["summary"]), update_model_instructions=u if u["needed"] else None)
    if g["passed"] and not u["needed"]: brain.remember_good(turns); return
    if not u["needed"]: return
    if not p.get("auto_improve", True): return update_call(job["id"], edit=dict(outcome="skipped", reason="Model updates are off for this project."))
    if edits_today() >= config()["daily_edit_cap"]: return update_call(job["id"], edit=dict(outcome="skipped", reason="The daily limit for model updates is reached."))
    if not (0 <= u["turn"] < len(turns)): return update_call(job["id"], edit=dict(outcome="skipped", reason="The instructions name no turn that this server answered."))
    update_call(job["id"], status="editing", edit_started=time.strftime("%Y-%m-%d %H:%M:%S")); log = []
    try: res = brain.edit_model(turns[u["turn"]], u, log)
    except BenchError as err: res = dict(outcome="no change", reason=str(err))
    update_call(job["id"], status="done", edit=res, steps=log)

# ---------- HTTP ----------
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
def tool_calls_in(text):
    out = []
    for raw in TOOL_CALL.findall(text):
        try: c = json.loads(raw)
        except json.JSONDecodeError: continue
        out.append(dict(id="call_" + secrets.token_hex(8), type="function", function=dict(name=c.get("name"), arguments=json.dumps(c.get("arguments") or {}))))
    return out

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def local(self):
        return self.client_address[0] == "127.0.0.1" and "X-Forwarded-For" not in self.headers and (self.headers.get("Host") or "").split(":")[0] in ("127.0.0.1", "localhost")
    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def body(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > 8_000_000: raise BenchError("Body too large.")
        return self.rfile.read(n)
    def do_GET(self):
        if not self.local(): return self.send(404, dict(error="Not found."))
        if self.path in ("/", "/index.html"): return self.send(200, open(os.path.join(HERE, "ui.html"), "rb").read(), "text/html; charset=utf-8")
        if self.path == "/logo.png": return self.send(200, open(os.path.join(HERE, "logo.png"), "rb").read(), "image/png")
        if self.path == "/api/state": return self.send(200, state())
        self.send(404, dict(error="Not found."))
    def do_POST(self):
        try:
            m = re.fullmatch(r"/p/([A-Za-z0-9_-]{20,})/(v1/chat/completions|chat/completions|webhook)", self.path.split("?")[0])
            if m:
                p = project_by_token(m.group(1))
                if p is None: return self.send(404, dict(error="Not found."))
                raw = self.body()
                return self.webhook(p, raw) if m.group(2) == "webhook" else self.chat(p, json.loads(raw or b"{}"))
            if not self.local(): return self.send(404, dict(error="Not found."))
            return self.send(200, admin(self.path, json.loads(self.body() or b"{}")))
        except BenchError as err: self.send(400, dict(error=str(err)))
        except Exception as err: traceback.print_exc(); self.send(500, dict(error=f"{type(err).__name__}: {err}"))
    def webhook(self, p, raw):
        if p.get("forward_url"): forward(p["forward_url"], raw, dict(self.headers))
        self.send(200, dict(ok=True))
        msg = (json.loads(raw or b"{}").get("message") or {})
        if msg.get("type") != "end-of-call-report": return
        call_id = (msg.get("call") or {}).get("id") or "unknown"; cid = f"{p['id']}:{call_id}"
        if cid in SEEN and call_id != "unknown": return
        SEEN.add(cid); turns = TURNS.pop((p["id"], call_id), None) or TURNS.pop((p["id"], "unknown"), None) or []
        art = msg.get("artifact") or {}
        update_call(cid, project=p["id"], call=call_id, time=time.strftime("%Y-%m-%d %H:%M:%S"), status="queued", ended=msg.get("endedReason"), turns=len(turns), transcript=(art.get("transcript") or "")[:20000])
        if p.get("judge", True): JOBS.put(dict(id=cid, project=p["id"], turns=turns, transcript=art.get("transcript")))
        else: update_call(cid, status="not judged", error="Judging is off for this project.")
    def chat(self, p, body):
        messages, tools = body.get("messages") or [], body.get("tools")
        call_id = (body.get("call") or (body.get("metadata") or {}).get("call") or {}).get("id") or "unknown"
        max_new = min(int(body.get("max_tokens") or 200), 400); rid = "chatcmpl-" + secrets.token_hex(8); now = int(time.time()); model = body.get("model") or "engram"
        pieces, stop, _ = brain.stream_live(messages, tools, max_new); stream = bool(body.get("stream")); full = ""; sent = 0; alive = True
        chunk = lambda delta, finish=None: json.dumps(dict(id=rid, object="chat.completion.chunk", created=now, model=model, choices=[dict(index=0, delta=delta, finish_reason=finish)]))
        def out(line):
            nonlocal alive
            if not alive: return
            try: self.wfile.write(f"data: {line}\n\n".encode()); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): alive = False; stop.flag.set()
        if stream:
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close"); self.end_headers(); self.close_connection = True
            out(chunk(dict(role="assistant", content="")))
        for piece in pieces:
            full += piece
            if not stream: continue
            cut = full.find("<tool_call>"); safe = len(full) if cut < 0 else cut
            if cut < 0:
                for k in range(1, min(11, len(full)) + 1):
                    if "<tool_call>".startswith(full[-k:]): safe = len(full) - k
            if safe > sent: out(chunk(dict(content=full[sent:safe]))); sent = safe
        calls_ = tool_calls_in(full); text = TOOL_CALL.sub("", full).strip()
        TURNS.setdefault((p["id"], call_id), []).append(dict(messages=messages, tools=tools, reply=full))
        if len(TURNS) > 200: TURNS.pop(next(iter(TURNS)))
        if not stream:
            msg = dict(role="assistant", content=text or None)
            if calls_: msg["tool_calls"] = calls_
            return self.send(200, dict(id=rid, object="chat.completion", created=now, model=model, choices=[dict(index=0, message=msg, finish_reason="tool_calls" if calls_ else "stop")]))
        if sent < len(text) and not calls_: out(chunk(dict(content=text[sent:])))
        if calls_: out(chunk(dict(tool_calls=[dict(index=i, id=c["id"], type="function", function=c["function"]) for i, c in enumerate(calls_)])))
        out(chunk({}, "tool_calls" if calls_ else "stop")); out("[DONE]")

# ---------- dashboard API ----------
def public(p):
    base = config()["public_url"].rstrip("/")
    return dict(p, llm_url=f"{base}/p/{p['token']}/v1" if base else "", webhook_url=f"{base}/p/{p['token']}/webhook" if base else "")
def state():
    vs = brain.load("versions.json", dict(active=0, versions=[]))
    return dict(config=config(), model=brain.LIVE.e.name, live_version=brain.LIVE.version, versions=vs, projects=[public(p) for p in projects()], calls=calls()[:100], edits_today=edits_today(), queue=JOBS.qsize())
def admin(path, b):
    with brain.STORE:
        ps = projects()
        if path == "/api/projects":
            name = (b.get("name") or "").strip()
            if not name: raise BenchError("Enter a project name.")
            p = dict(id=secrets.token_hex(4), token=secrets.token_urlsafe(24), name=name, forward_url=(b.get("forward_url") or "").strip(), judge=True, auto_improve=True, created=time.strftime("%Y-%m-%d %H:%M:%S"))
            ps.append(p); brain.save("projects.json", ps); return public(p)
        m = re.fullmatch(r"/api/projects/([0-9a-f]+)(/delete)?", path)
        if m:
            p = next((x for x in ps if x["id"] == m.group(1)), None)
            if p is None: raise BenchError("No such project.")
            if m.group(2): ps.remove(p); brain.save("projects.json", ps); return dict(ok=True)
            for k in ("name", "forward_url"):
                if k in b: p[k] = (b[k] or "").strip()
            for k in ("judge", "auto_improve"):
                if k in b: p[k] = bool(b[k])
            brain.save("projects.json", ps); return public(p)
        if path == "/api/config":
            c = config()
            if "daily_edit_cap" in b: c["daily_edit_cap"] = max(0, int(b["daily_edit_cap"]))
            brain.save("config.json", c); return c
    m = re.fullmatch(r"/api/versions/(\d+)/activate", path)
    if m: brain.activate(int(m.group(1))); return dict(ok=True)
    if path == "/api/try":
        msgs = b.get("messages") or []
        pieces, _, _ = brain.stream_live(msgs, None, 160); return dict(reply="".join(pieces).strip())
    raise BenchError("Unknown route.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--port", type=int, default=8790); ap.add_argument("--public-url"); ARGS = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    c = config(); c["public_url"] = (ARGS.public_url or open_tunnel(ARGS.port)).rstrip("/"); brain.save("config.json", c)
    brain.start(ARGS.model)
    threading.Thread(target=worker, daemon=True).start()
    print(f"Vengram: dashboard http://127.0.0.1:{ARGS.port}  public {config()['public_url']}  model {brain.LIVE.e.name}  version {brain.LIVE.version}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), H).serve_forever()
