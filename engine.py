"""Engram engine: trace a prompt through a Hugging Face transformer and edit its weights.

Every matrix is handled in (output, input) orientation, so W @ x is its forward map. GPT-2 Conv1D
weights are stored as (input, output) and are transposed on the way in and out.
Edits are stored as float32 deltas. The original values of touched tensors are kept for exact rollback.
"""
import base64, glob, hashlib, json, os, re, shutil, tempfile, time
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file, load_file

MAX_TOKENS, MAX_NEW, POOL_ROWS, POOL_COLS = 200, 24, 220, 440
HERE = os.path.dirname(os.path.abspath(__file__)); EXPORT_DIR = os.environ.get("ENGRAM_EXPORTS", os.path.join(HERE, "exports"))

class BenchError(ValueError): pass

def pick_device(name=None):
    if name: return name
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def find_layers(model):
    best, size, path = None, -1, None
    for n, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) > 1:
            s = sum(p.numel() for p in m.parameters())
            if s > size: best, size, path = m, s, n
    if best is None: raise BenchError("No repeated layer list found in this model.")
    return path, best

def find_final_norm(model, layers_path):
    found = None
    for n, m in model.named_modules():
        if "norm" in type(m).__name__.lower() and not n.startswith(layers_path + "."): found = m
    return found

def indices(text, n):
    """Parse '0:128,900' into sorted indices. Blank selects all."""
    if not str(text).strip(): return torch.arange(n)
    out = []
    for part in str(text).split(","):
        m = re.fullmatch(r"\s*(\d+)\s*(?::\s*(\d+)\s*)?", part)
        if not m: raise BenchError(f"Cannot read index range '{part}'. Use forms like 0:128,900.")
        a = int(m[1]); b = int(m[2]) if m[2] else a + 1
        if not (0 <= a < b <= n): raise BenchError(f"Range '{part.strip()}' is outside 0:{n}.")
        out.extend(range(a, b))
    return torch.tensor(sorted(set(out)))

def b64(t): return base64.b64encode(t.numpy().tobytes()).decode()
def to_bytes(t, top): return (t / max(float(top), 1e-12)).clamp(0, 1).mul(255).round().byte()
def pool2(t):
    r, c = min(POOL_ROWS, t.shape[0]), min(POOL_COLS, t.shape[1])
    return F.adaptive_avg_pool2d(t[None, None], (r, c))[0, 0]
def pool1(t, n): return F.adaptive_avg_pool1d(t[None, None], min(n, t.shape[0]))[0, 0]
def fro(t): return float(t.double().norm())   # float32 norms lose accuracy on large matrices
def q(t, p):
    t = t.flatten()
    return float(torch.quantile(t, p)) if t.numel() else 1.0

class Mat:
    def __init__(s, name, module, param, layer, kind, flip):
        s.name, s.module, s.param, s.layer, s.kind, s.flip = name, module, param, layer, kind, flip
        shape = module.weight.shape; s.rows, s.cols = (shape[1], shape[0]) if flip else (shape[0], shape[1])

class Engine:
    def __init__(s, model, tok, name, device):
        s.model, s.tok, s.name, s.device = model.eval(), tok, name, device
        s.layers_path, s.layers = find_layers(model); s.n_layers = len(s.layers)
        s.norm = find_final_norm(model, s.layers_path); s.unembed = model.get_output_embeddings()
        if s.unembed is None: raise BenchError("This model has no output embedding, so a layer trace is not possible.")
        s.params = dict(model.named_parameters()); pid = {id(p): n for n, p in s.params.items()}
        s.mats = {}
        prefix = s.layers_path + "."
        for n, m in model.named_modules():
            cls = type(m).__name__
            if (isinstance(m, nn.Linear) or cls == "Conv1D") and n.startswith(prefix):
                rest = n[len(prefix):].split(".", 1)
                s.mats[n] = Mat(n, m, pid[id(m.weight)], int(rest[0]), rest[1] if len(rest) > 1 else "linear", cls == "Conv1D")
        if not s.mats: raise BenchError("No linear matrices found inside the model layers.")
        s.kinds = list(dict.fromkeys(m.kind for m in s.mats.values()))
        s.hidden = s.unembed.weight.shape[1]
        s.orig, s.edits, s.pins, s.run, s.residual, s._next = {}, [], [], None, None, 1

    @classmethod
    def load(cls, path, device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not os.path.isdir(path): raise BenchError(f"Folder not found: {path}")
        device = pick_device(device); torch.set_num_threads(2)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device)
        return cls(model, tok, os.path.abspath(path), device)

    # ---------- info ----------
    def piece(s, i): return s.tok.decode([int(i)])
    def info(s):
        cfg = s.model.config
        heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
        return dict(model=s.name, device=s.device, layers=s.n_layers, kinds=s.kinds, hidden=s.hidden,
                    head_width=(s.hidden // heads) if heads else None, final_norm=type(s.norm).__name__ if s.norm else None,
                    params=sum(p.numel() for p in s.params.values()),
                    matrices=[dict(name=m.name, layer=m.layer, kind=m.kind, rows=m.rows, cols=m.cols, edited=any(m.param in e["deltas"] for e in s.edits if e["on"])) for m in s.mats.values()],
                    edits=[{k: e[k] for k in ("id", "label", "on", "stats")} for e in s.edits],
                    pins=[{k: p[k] for k in ("id", "group", "label")} for p in s.pins], residual=s.residual)

    def resolve(s, t):
        """A token is given by id, or by text. Exact vocabulary pieces win over tokenizing the text."""
        if isinstance(t, dict) and t.get("id") is not None: return int(t["id"])
        text = t.get("text") if isinstance(t, dict) else t
        if not text: return None
        v = s.tok.get_vocab().get(text)
        if v is not None: return v
        ids = s.tok.encode(text, add_special_tokens=False)
        if not ids: raise BenchError(f"Text '{text}' has no tokens.")
        return ids[0]

    # ---------- weights state ----------
    def _sync(s, edited=True):
        for n, o in s.orig.items():
            p = s.params[n]; v = o.clone()
            if edited:
                for e in s.edits:
                    if e["on"] and n in e["deltas"]: v = v + e["deltas"][n]
            p.data.copy_(v.to(p.device, p.dtype))

    def _W(s, m):
        W = m.module.weight.detach().float().cpu()
        return W.T if m.flip else W.clone()

    def _any_on(s): return any(e["on"] for e in s.edits)

    def _commit(s, deltas, label, stats, spec):
        for n in deltas:
            if n not in s.orig: s.orig[n] = s.params[n].detach().float().cpu().clone()
        s.edits.append(dict(id=s._next, label=label, on=True, stats=stats, spec=spec, deltas=deltas)); s._next += 1
        s.residual = None; s._sync(True)

    def _after_change(s):
        s._sync(True)
        s.residual = None if s._any_on() or not s.orig else max(float((s.params[n].detach().float().cpu() - o).abs().max()) for n, o in s.orig.items())

    def toggle(s, i, on):
        s._edit(i)["on"] = bool(on); s._after_change()
    def remove(s, i):
        e = s._edit(i); s.edits.remove(e); s._after_change()
    def _edit(s, i):
        for e in s.edits:
            if e["id"] == i: return e
        raise BenchError(f"No edit {i}.")

    # ---------- forward passes ----------
    def encode(s, prompt, chat):
        if chat:
            if not s.tok.chat_template: raise BenchError("This tokenizer has no chat template. Turn chat format off.")
            prompt = s.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = s.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        if ids.shape[1] == 0: raise BenchError("The prompt is empty.")
        if ids.shape[1] + MAX_NEW > MAX_TOKENS: raise BenchError(f"The prompt has {ids.shape[1]} tokens. The limit is {MAX_TOKENS - MAX_NEW}.")
        return ids.to(s.device)

    @torch.no_grad()
    def answer(s, ids):
        stop = {s.tok.eos_token_id, s.tok.pad_token_id} | set(getattr(s.model.generation_config, "eos_token_id", None) if isinstance(getattr(s.model.generation_config, "eos_token_id", None), list) else [])
        out = s.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None,
                               repetition_penalty=1.0, pad_token_id=s.tok.pad_token_id or s.tok.eos_token_id)[0, ids.shape[1]:].tolist()
        kept = []
        for t in out:
            if t in stop: break
            kept.append(t)
        return kept

    @torch.no_grad()
    def forward(s, ids, capture=False):
        resid, x, hs = [None] * s.n_layers, {}, []
        for L, layer in enumerate(s.layers):
            hs.append(layer.register_forward_hook(lambda m, i, o, L=L: resid.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0].detach())))
        if capture:
            for n, m in s.mats.items():
                hs.append(m.module.register_forward_pre_hook(lambda mod, i, n=n: x.__setitem__(n, i[0][0].detach().float().cpu()) if i[0].ndim == 3 else None))
        try: logits = s.model(ids).logits[0]
        finally:
            for h in hs: h.remove()
        return resid, x, logits

    @torch.no_grad()
    def lens_logits(s, h):
        return s.unembed(s.norm(h) if s.norm is not None else h)

    @torch.no_grad()
    def lens(s, resid):
        grid = []
        for h in resid:
            v, i = s.lens_logits(h).softmax(-1).topk(5)
            grid.append([[[s.piece(a), round(float(b), 4), int(a)] for a, b in zip(ii, vv)] for ii, vv in zip(i.tolist(), v.tolist())])
        return grid   # [layer][position][rank] = [piece, probability, id]

    def trace(s, prompt, chat):
        ids = s.encode(prompt, chat); n_prompt = ids.shape[1]
        on = s._any_on()
        s._sync(False); ans_b = s.answer(ids)
        full = torch.cat([ids, torch.tensor([ans_b], device=s.device, dtype=ids.dtype)], 1) if ans_b else ids
        rb, xb, _ = s.forward(full, capture=not on)
        s._sync(True); ans_e, re, xe = None, None, None
        if on:
            ans_e = s.answer(ids); re, xe, _ = s.forward(full, capture=True)
        x = xe if on else xb
        s.run = dict(prompt=prompt, chat=chat, ids=full, n_prompt=n_prompt, x=x,
                     base=[h.float().cpu() for h in rb], edited=[h.float().cpu() for h in re] if on else None)
        return s.trace_payload(ans_b, ans_e)

    def trace_payload(s, ans_b, ans_e):
        r = s.run; toks = [s.piece(t) for t in r["ids"][0].tolist()]
        activity = {k: [None] * s.n_layers for k in s.kinds}
        for n, x in r["x"].items():
            m = s.mats[n]; W = s._W(m)
            activity[m.kind][m.layer] = [round(v, 3) for v in (x @ W.T).norm(dim=-1).tolist()]
        lb = s.lens([h.to(s.device) for h in r["base"]]); le = s.lens([h.to(s.device) for h in r["edited"]]) if r["edited"] else None
        return dict(tokens=toks, n_prompt=r["n_prompt"], answer_base=s.tok.decode(ans_b), answer_edited=s.tok.decode(ans_e) if ans_e is not None else None,
                    base=lb, edited=le, activity=activity, kinds=s.kinds, info=s.info())

    def _need_run(s):
        if s.run is None: raise BenchError("Run a trace first.")
        return s.run

    @torch.no_grad()
    def track(s, target, foil, pos):
        r = s._need_run(); t = s.resolve(target); f = s.resolve(foil)
        if t is None: raise BenchError("Choose a target token.")
        def series(res):
            if res is None: return None
            out = []
            for h in res:
                lg = s.lens_logits(h[pos][None].to(s.device))[0]
                out.append(round(float(lg[t] - lg[f]), 4) if f is not None else round(float(lg.softmax(-1)[t]), 5))
            return out
        return dict(target=s.piece(t), target_id=t, foil=s.piece(f) if f is not None else None, foil_id=f, measure="logit difference" if f is not None else "probability",
                    base=series(r["base"]), edited=series(r["edited"]))

    # ---------- matrix view ----------
    def _mat(s, name):
        if name not in s.mats: raise BenchError(f"Unknown matrix: {name}")
        return s.mats[name]

    def matrix(s, name, pos):
        m = s._mat(name); W = s._W(m); wp = pool2(W.abs()); top = q(wp, 0.995)
        dW = None
        for e in s.edits:
            if e["on"] and m.param in e["deltas"]:
                d = e["deltas"][m.param]; d = d.T if m.flip else d
                dW = d.clone() if dW is None else dW + d
        out = dict(name=name, layer=m.layer, kind=m.kind, rows=m.rows, cols=m.cols, height=wp.shape[0], width=wp.shape[1], row_pool=m.rows / wp.shape[0], col_pool=m.cols / wp.shape[1],
                   weights=b64(to_bytes(wp, top)), frobenius=fro(W))
        if dW is not None:
            dp = pool2(dW.abs()); out["delta"] = b64(to_bytes(dp, q(dp, 0.995))); out["delta_frobenius"] = fro(dW)
        x = s.run["x"].get(name) if s.run else None
        if x is not None:
            v = x[pos]; lit = W.abs() * v.abs()[None]; lp = pool2(lit); out["lit"] = b64(to_bytes(lp, q(lp, 0.998)))
            xa = pool1(v.abs(), wp.shape[1]); ya = pool1((W @ v).abs(), wp.shape[0])
            out["x_strip"] = b64(to_bytes(xa, xa.max())); out["y_strip"] = b64(to_bytes(ya, ya.max()))
            out["contrib"] = float(v.norm()); signed = W * v[None]
            val, idx = signed.abs().flatten().topk(8); rc = [(int(i) // m.cols, int(i) % m.cols) for i in idx]
            out["top"] = [dict(row=a, col=b, w=float(W[a, b]), x=float(v[b]), wx=float(signed[a, b])) for a, b in rc]
            if dW is not None:
                dl = pool2(dW.abs() * v.abs()[None]); out["delta_lit"] = b64(to_bytes(dl, q(dl, 0.995)))
        return out

    # ---------- pins ----------
    def pin(s, group, pos):
        r = s._need_run()
        if group not in ("A", "B"): raise BenchError("A pin belongs to group A or group B.")
        n = r["ids"].shape[1]
        if not 0 <= pos < n: raise BenchError("That position is outside the trace.")
        s.pins.append(dict(id=s._next, group=group, label=f"{s.piece(r['ids'][0, pos])!r} @ {pos}", x={k: v[pos].clone() for k, v in r["x"].items()})); s._next += 1

    def unpin(s, i): s.pins = [p for p in s.pins if p["id"] != i]
    def clear_pins(s): s.pins = []

    def _contrast(s, name):
        A = [p["x"][name] for p in s.pins if p["group"] == "A"]; B = [p["x"][name] for p in s.pins if p["group"] == "B"]
        if not A or not B: raise BenchError("This edit needs at least one pin in group A and one in group B.")
        return torch.stack(A).mean(0), torch.stack(B).mean(0)

    def _token_direction(s, target, foil):
        t, f = s.resolve(target), s.resolve(foil)
        if t is None or f is None: raise BenchError("This edit needs a target token and a foil token.")
        E = s.unembed.weight.detach().float().cpu(); e = E[t] - E[f]
        g = getattr(s.norm, "weight", None)
        if g is not None and g.ndim == 1 and g.shape[0] == s.hidden: e = e * g.detach().float().cpu()
        return e / e.norm(), s.piece(t), s.piece(f)

    # ---------- edits ----------
    def apply(s, spec):
        m = s._mat(spec.get("module")); W = s._W(m); op = spec.get("op"); write = None
        if op == "scale":
            factor = float(spec["factor"]); R = indices(spec.get("rows", ""), m.rows); C = indices(spec.get("cols", ""), m.cols)
            delta = torch.zeros_like(W); delta[R[:, None], C] = W[R[:, None], C] * (factor - 1.0)
            label = f"Scale x{factor:g}, {len(R)} rows, {len(C)} cols"
        elif op == "rank_one":
            if m.rows != s.hidden: raise BenchError(f"A token write needs a matrix that outputs the {s.hidden}-wide residual. This one outputs {m.rows}.")
            alpha = float(spec["alpha"]); u, tn, fn = s._token_direction(spec.get("target"), spec.get("foil"))
            if spec.get("key") == "contrast":
                a, b = s._contrast(m.name); v = a - b
            else:
                r = s._need_run(); x = r["x"].get(m.name)
                if x is None: raise BenchError("No input was captured for this matrix.")
                v = x[int(spec["pos"])].clone()
            v = v / v.norm(); delta = alpha * torch.outer(u, v)
            if s.run and m.name in s.run["x"] and spec.get("pos") is not None: write = float(alpha * (v @ s.run["x"][m.name][int(spec["pos"])]))
            label = f"{tn!r} instead of {fn!r}, strength {alpha:g}"
        elif op == "remove_direction":
            strength = float(spec["strength"])
            if spec.get("source") == "tokens":
                d, tn, fn = s._token_direction(spec.get("target"), spec.get("foil")); what = f"{tn!r} minus {fn!r}"
                if m.rows != s.hidden: raise BenchError(f"A token direction needs a matrix that outputs the {s.hidden}-wide residual. This one outputs {m.rows}.")
            else:
                a, b = s._contrast(m.name); d = W @ (a - b); d = d / d.norm(); what = "A minus B"
            delta = -strength * torch.outer(d, d @ W); label = f"Remove direction {what}, strength {strength:g}"
        else: raise BenchError(f"Unknown edit type: {op}")
        nz = int((delta != 0).sum())
        if nz == 0: raise BenchError("This edit changes no entries.")
        stats = dict(module=m.name, frobenius=fro(delta), base_frobenius=fro(W), max_entry=float(delta.abs().max()), changed=nz, write_at_position=write)
        s._commit({m.param: (delta.T if m.flip else delta).contiguous()}, label, stats, {k: v for k, v in spec.items()})

    # ---------- save and load ----------
    def export(s, name):
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_") or time.strftime("edit_%Y%m%d_%H%M%S")
        total = {}
        for e in s.edits:
            if e["on"]:
                for k, d in e["deltas"].items(): total[k] = total[k] + d if k in total else d.clone()
        if not total: raise BenchError("No active edits to save.")
        os.makedirs(EXPORT_DIR, exist_ok=True); path = os.path.join(EXPORT_DIR, name + ".safetensors")
        save_file({k: v.contiguous() for k, v in total.items()}, path)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        meta = dict(model=s.name, sha256=h, tensors={k: list(v.shape) for k, v in total.items()}, edits=[dict(label=e["label"], stats=e["stats"], spec=e["spec"]) for e in s.edits if e["on"]])
        json.dump(meta, open(path[:-12] + ".json", "w"), indent=1)
        return dict(path=path, sha256=h)

    def export_model(s, name):
        """Write a complete Hugging Face model folder with the active edits built in."""
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_") or time.strftime("model_%Y%m%d_%H%M%S")
        if not any(e["on"] for e in s.edits): raise BenchError("No active edits to write.")
        os.makedirs(EXPORT_DIR, exist_ok=True); path = os.path.join(EXPORT_DIR, name)
        if os.path.exists(path): raise BenchError(f"This folder already exists: {path}")
        need = sum(p.numel() * p.element_size() for p in s.model.parameters())
        free = shutil.disk_usage(EXPORT_DIR).free
        if free < need * 1.1: raise BenchError(f"This model needs {need / 1e9:.1f} GB and the disk has {free / 1e9:.1f} GB free.")
        s._sync(True); tmp = tempfile.mkdtemp(dir=EXPORT_DIR, prefix=f".{name}.")
        try:
            s.model.save_pretrained(tmp, safe_serialization=True); s.tok.save_pretrained(tmp)
            checked = 0; live = s.model.state_dict()
            for file in glob.glob(os.path.join(tmp, "*.safetensors")):
                with safe_open(file, "pt") as f:
                    for k in f.keys():
                        if k in s.orig:
                            if not torch.equal(f.get_tensor(k), live[k].detach().cpu()): raise BenchError(f"The written weights differ from the edited weights: {k}")
                            checked += 1
            if checked != len(s.orig): raise BenchError(f"The written model has {checked} of the {len(s.orig)} edited matrices.")
            json.dump(dict(base_model=s.name, edits=[dict(label=e["label"], stats=e["stats"], spec=e["spec"]) for e in s.edits if e["on"]]), open(os.path.join(tmp, "engram_edits.json"), "w"), indent=1)
            os.chmod(tmp, 0o755); os.rename(tmp, path)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True); raise
        return dict(path=path, checked=checked, gigabytes=round(sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)) / 1e9, 2))

    def import_delta(s, path):
        if not os.path.isfile(path): raise BenchError(f"File not found: {path}")
        d = load_file(path)
        for k, v in d.items():
            if k not in s.params: raise BenchError(f"This delta names a tensor the model does not have: {k}")
            if s.params[k].shape != v.shape: raise BenchError(f"Shape mismatch for {k}.")
        size = sum(fro(v) ** 2 for v in d.values()) ** 0.5
        s._commit({k: v.float() for k, v in d.items()}, "Import " + os.path.basename(path), dict(module=", ".join(d), frobenius=size, base_frobenius=None, max_entry=max(float(v.abs().max()) for v in d.values()), changed=sum(int((v != 0).sum()) for v in d.values()), write_at_position=None), dict(op="import", path=path))
