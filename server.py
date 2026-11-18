"""Live dashboard + chat with the latest RL checkpoint.

  python server.py --run runs/F            # then open http://localhost:8765

- Dashboard reads runs/X/log.jsonl, eval.jsonl, samples.jsonl on every refresh.
- Chat model = 4-bit base (~1 GB, keeps training out of swap) + the run's LoRA adapters,
  hot-reloaded whenever adapters.safetensors changes. "Base" mode switches LoRA off.
- /api/compare runs a prompt on base and current models side by side.
Everything the model outputs is returned verbatim (think tags, special tokens, ...).
"""
import argparse, json, os, re, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import reasoning_gym
from common import SYSTEM, TASKS, EVAL_SEED, score, extract_answer

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/F")
ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct-4bit")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--no_system", action="store_true", help="compare prompts without the think system prompt")
ap.add_argument("--tools", action="store_true", help="execute <code> blocks in the sandbox (chat + compare)")
ap.add_argument("--template", default="", help="'r1zero' for base models: plain-text prompt, stop at </answer>")
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))

cfg = json.load(open(os.path.join(args.run, "config.json")))
M = {}  # model state, only touched by the worker thread


def _init_model():
    global mx, stream_generate, make_sampler
    import mlx.core as mx
    mx.set_memory_limit(int(1.5e9))
    mx.set_cache_limit(int(0.25e9))
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.lora import LoRALinear
    model, tok = load(args.model)
    model.freeze()
    linear_to_lora_layers(model, len(model.layers), {"rank": cfg["rank"], "scale": 2.0, "dropout": 0.0,
                          "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                   "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]})
    M["model"], M["tok"] = model, tok
    M["loras"] = [m for _, m in model.named_modules() if isinstance(m, LoRALinear)]
    M["scale"] = M["loras"][0].scale
adapter = {"mtime": None, "step": 0}
ADAPTER_PATH = os.path.join(args.run, "adapters.safetensors")
STATE_PATH = os.path.join(args.run, "state.json")


def maybe_reload():
    try:
        mt = os.path.getmtime(ADAPTER_PATH)
    except OSError:
        return
    if mt != adapter["mtime"]:
        time.sleep(0.5)  # let the trainer finish writing
        M["model"].load_weights(ADAPTER_PATH, strict=False)
        mx.eval(M["model"].parameters())
        adapter["mtime"] = mt
        try:
            adapter["step"] = json.load(open(STATE_PATH))["step"]
        except Exception:
            pass
        print(f"loaded adapters from step {adapter['step']}", flush=True)


def set_lora(on):
    for m in M["loras"]:
        m.scale = M["scale"] if on else 0.0


import queue
from types import SimpleNamespace
JOBS = queue.Queue()
JUDGE = None
READY = threading.Event()
_END = object()


def _worker():
    """All MLX work happens on this one thread (MLX streams are per-thread)."""
    _init_model()
    maybe_reload()
    READY.set()
    while True:
        job = JOBS.get()
        if job[0] == "call":                       # ("call", fn, use_adapter, out): run fn() on the MLX thread
            _, fn, use_adapter, out = job
            try:
                maybe_reload()
                set_lora(use_adapter and adapter["mtime"] is not None)
                out.put(fn())
            except Exception as e:
                import traceback; traceback.print_exc()
                out.put(e)
            finally:
                set_lora(True)
                out.put(_END)
            continue
        messages, use_adapter, temp, max_tokens, out = job
        try:
            maybe_reload()
            set_lora(use_adapter and adapter["mtime"] is not None)
            if args.template:
                from tools import generate_with_tools
                R1Z = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
                       "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
                       "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
                       "i.e., <think> reasoning process here </think> <answer> answer here </answer>.")
                if args.tools:   # same prompt the tools run trains with (Python + solvers + checkpoint/backtrack)
                    from tools import R1Z_TOOLS
                    R1Z = R1Z_TOOLS
                convo = R1Z
                for mm in messages:
                    if mm["role"] == "user":
                        convo += " User: " + mm["content"] + (" Assistant:" if mm is messages[-1] else "")
                    elif mm["role"] == "assistant":
                        convo += " Assistant: " + mm["content"]
                ids = M["tok"].encode(convo)
                r = generate_with_tools(M["model"], M["tok"], [ids], JUDGE, max_tokens=max_tokens, tools=args.tools,
                                        stop_answer=True, stop_user=True, backtrack=args.tools,
                                        sampler=make_sampler(temp=temp, top_p=0.95) if temp > 0 else None,
                                        completion_batch_size=1, prefill_batch_size=1)
                shown = r["log_texts"][0] if args.tools else r["texts"][0]     # full history incl. backtracked parts
                out.put(SimpleNamespace(text=shown, generation_tokens=sum(r["masks"][0]), generation_tps=0.0,
                                        finish_reason=("answer" if "</answer>" in r["texts"][0] else
                                                       ("length" if r["truncated"][0] else "stop"))
                                        + (f" · {r['calls'][0]} code runs · {r['backtracks'][0]} backtracks" if args.tools else "")))
                continue
            if args.tools:
                from tools import generate_with_tools
                ids = M["tok"].apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
                r = generate_with_tools(M["model"], M["tok"], [ids], JUDGE, max_tokens=max_tokens,
                                        sampler=make_sampler(temp=temp, top_p=0.95) if temp > 0 else None,
                                        completion_batch_size=1, prefill_batch_size=1)
                out.put(SimpleNamespace(text=r["texts"][0], generation_tokens=sum(r["masks"][0]),
                                        generation_tps=0.0, finish_reason=f"{r['calls'][0]} tool calls"))
                continue
            prompt = M["tok"].apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            sampler = make_sampler(temp=temp, top_p=0.95) if temp > 0 else None
            for r in stream_generate(M["model"], M["tok"], prompt, max_tokens=max_tokens, sampler=sampler):
                out.put(r)
        except Exception as e:
            out.put(e)
        finally:
            set_lora(True)
            out.put(_END)


def gen(messages, use_adapter=True, temp=0.7, max_tokens=512):
    out = queue.Queue()
    JOBS.put((messages, use_adapter, temp, max_tokens, out))
    while (r := out.get()) is not _END:
        if isinstance(r, Exception):
            raise r
        yield r


# recommended prompts: one HELD-OUT instance of every problem type (eval seed, never trained on)
import itertools as _it
CODE_SYSTEM = ("You are an expert Python programmer. Think step by step inside <think>...</think>, "
               "then give your complete solution in a single ```python code block.")
TRAIN_TASKS = [t for t in cfg.get("tasks", "").split(",") if t]
UNSEEN = [t for t in TASKS if t not in TRAIN_TASKS]
GROUPS = [("eval: trained", [t for t in TASKS if t in TRAIN_TASKS]),
          ("eval: never trained", UNSEEN),
          ("other training tasks", [t for t in TRAIN_TASKS if t not in TASKS])]
CODE_POOL = ([("kodcode", json.loads(l)) for l in _it.islice(open("data/code/kodcode_eval.jsonl"), 200)] +
             [("leetcode", json.loads(l)) for l in open("data/code/leetcode_eval.jsonl")])
PROMPTS, ENTRIES = [], []


def _add_rg(group, t, i=0, label=None):
    base, _, spec = t.partition("@")          # curriculum variants like "n_queens@n=4;max_remove=2"
    cfg = {k: int(v) if v.lstrip("-").isdigit() else v for k, v in (kv.split("=") for kv in spec.split(";") if kv)}
    ds = reasoning_gym.create_dataset(base, size=max(i + 1, 1), seed=EVAL_SEED, **cfg)
    e = ds[i]
    p = dict(id=len(PROMPTS), group=group, task=label or t, question=e["question"], answer=str(e["answer"]))
    PROMPTS.append(p)
    ENTRIES.append(("rg", ds, e))
    return p


def _code_question(r):
    return r["prompt"].strip() + (f"\n\nComplete this starter code:\n```python\n{r['starter'].strip()}\n```"
                                  if r["starter"] else "")


def _add_code(group, src, r, label=None):
    p = dict(id=len(PROMPTS), group=group, task=label or f"{src}: {r['difficulty']}", question=_code_question(r),
             answer="(graded by hidden unit tests)")
    PROMPTS.append(p)
    ENTRIES.append(("code", r))
    return p


for g, ts in GROUPS:
    for t in ts:
        _add_rg(g, t)
for diff in ("easy", "medium", "hard"):
    r = next((r for s, r in CODE_POOL if s == "kodcode" and r["difficulty"] == diff), None)
    if r:
        _add_code("code", "kodcode", r)
r = next((r for s, r in CODE_POOL if s == "leetcode"), None)
if r:
    _add_code("code", "leetcode", r)
PROMPTS.append(dict(id=len(PROMPTS), group="chat", task="chat", question="Explain in two sentences why the sky is blue.", answer=""))
ENTRIES.append(None)


def grade(entry, text):
    if entry is None:
        return None, None
    if entry[0] == "rg":
        return extract_answer(text), score(entry[1], entry[2], text) >= 1
    r = entry[1]
    body = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", body, re.S)
    if not blocks:
        ans = re.findall(r"<answer>(.*?)</answer>", body, re.S)
        if len(ans) == 1 and ans[0].strip():
            blocks = [re.sub(r"^\s*```(?:python|py)?\s*\n|```\s*$", "", ans[0].strip())]
    if len(blocks) != 1 or JUDGE is None:
        return "(no single code block)", False
    if r["mode"] == "tests":
        g = JUDGE.grade_tests(blocks[0], r["test_code"], prefix=r["prefix"], timeout=8)
        return f"{g['passed']}/{g['total']} tests", bool(g.get("all_pass"))
    g = JUDGE.grade(blocks[0], r["io"][:15], timeout=4, partial=False)
    return f"{g['passed']}/{g['total']} tests", g["score"] == 1.0


# ------------------------------------------------------------------ toy-testing helpers (run on the MLX thread)
R1Z_SYS = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
           "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
           "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
           "i.e., <think> reasoning process here </think> <answer> answer here </answer>.")


def call(fn, use_adapter=True):
    out = queue.Queue()
    JOBS.put(("call", fn, use_adapter, out))
    res = None
    while (r := out.get()) is not _END:
        if isinstance(r, Exception):
            raise r
        res = r
    return res


def r1z_prompt(messages):
    convo = R1Z_SYS
    for mm in messages:
        if mm["role"] == "user":
            convo += " User: " + mm["content"]
        elif mm["role"] == "assistant":
            convo += " Assistant:" + mm["content"]
    return convo + " Assistant:"


def batch_gen(strs, temp=0.7, max_tokens=640):
    from tools import generate_with_tools
    r = generate_with_tools(M["model"], M["tok"], [M["tok"].encode(x) for x in strs], None, max_tokens=max_tokens,
                            tools=False, stop_answer=True, stop_user=True,
                            sampler=make_sampler(temp=temp, top_p=0.95) if temp > 0 else None,
                            completion_batch_size=8, prefill_batch_size=4)
    return r["texts"]


def ntok(x):
    return len(M["tok"].encode(x, add_special_tokens=False))


def forced_think(prompt, min_tokens, cue, temp, max_tokens):
    """budget forcing: if the model writes </think> or <answer> before min_tokens, cut it, append cue, continue."""
    text, events = "", []
    for _ in range(12):
        t = batch_gen([prompt + text], temp, max(32, max_tokens - ntok(text)))[0]
        full = text + t
        hits = [full.find(mk, len(text) - 0) for mk in ("</think>", "<answer>") if full.find(mk, len(text)) >= 0]
        hit = min(hits) if hits else -1
        if hit >= 0 and ntok(full[:hit]) < min_tokens:
            cut = full[:hit].rstrip()
            events.append(dict(pos=len(cut), tokens=ntok(cut)))
            text = cut + cue
            continue
        return full, events
    return text, events


def answers_of(text):
    a = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    return a[-1].strip() if a else None


def cut_points(reasoning, fracs):
    out = []
    for f in fracs:
        target = int(len(reasoning) * f)
        b = [mm.end() for mm in re.finditer(r"\n+|(?<=[.!?:])\s", reasoning[:target])]
        pos = b[-1] if b and b[-1] > len(reasoning) * f * 0.5 else target
        if pos > 0 and (not out or pos > out[-1]):
            out.append(pos)
    return out


def read_jsonl(name):
    p = os.path.join(args.run, name)
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            b = open(os.path.join(HERE, "dashboard.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/api/data":
            samples = read_jsonl("samples.jsonl")
            self.send_json(dict(run=args.run, config=cfg, log=read_jsonl("log.jsonl"), eval=read_jsonl("eval.jsonl"),
                                samples=samples[-16:], adapter_step=adapter["step"], system="" if args.template else (__import__("tools").TOOL_SYSTEM if args.tools else ("" if args.no_system else SYSTEM)),
                                now=time.time()))
        elif self.path == "/api/results":
            # leaderboard from the paired big evals + narrow-family evals (results/*.json)
            import glob, statistics as st
            rows = []
            for f in sorted(glob.glob(os.path.join(HERE, "results", "evalbig_*.json"))):
                name = os.path.basename(f)[8:-5]
                try:
                    R = json.load(open(f))
                except Exception:
                    continue
                if not isinstance(R, dict) or "unseen" not in R:
                    continue
                u = R["unseen"]
                rows.append(dict(name=name, unseen_p1=st.mean(x["p1"] for x in u), unseen_maj=st.mean(x["maj"] for x in u),
                                 unseen_pk=st.mean(x["pk"] for x in u),
                                 **{k: st.mean(x["p1"] for x in R[k]) for k in ("gsm8k", "math500", "general") if k in R},
                                 tokens=st.mean(x["toks"] for x in u)))
            narrow = {}
            for f in sorted(glob.glob(os.path.join(HERE, "results", "narrow_*.json"))):
                name = os.path.basename(f)[7:-5]
                try:
                    R = json.load(open(f))
                except Exception:
                    continue
                if isinstance(R, list):
                    by = {}
                    for x in R:
                        by.setdefault(x["group"], []).append(x)
                    narrow[name] = {g: dict(p1=st.mean(x["p1"] for x in v), pk=st.mean(x["pk"] for x in v)) for g, v in by.items()}
            self.send_json(dict(big=rows, narrow=narrow))
        elif self.path == "/api/prompts":
            self.send_json(PROMPTS)
        elif self.path == "/api/random":
            # a real held-out test problem of ANY type: all reasoning tasks + code (equal chance per type)
            import random
            kinds = [t for _, ts in GROUPS for t in ts] + ["code"]
            k = random.choice(kinds)
            if k == "code":
                src, r = random.choice(CODE_POOL)
                p = _add_code("random", src, r, label=f"{src} ({r['difficulty']})")
            else:
                i = random.randrange(50)
                p = _add_rg("random", k, i, label=f"{k} #{i}")
            self.send_json(p)
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/chat":
            # Server-sent stream of raw text chunks
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            msgs = req.get("messages", [])
            if req.get("system"):
                msgs = [{"role": "system", "content": req["system"]}] + msgs
            last = None
            try:
                for last in gen(msgs, req.get("adapter", True), float(req.get("temp", 0.7)),
                                int(req.get("max_tokens", 512))):
                    self.wfile.write(b"data: " + json.dumps({"t": last.text}).encode() + b"\n\n")
                    self.wfile.flush()
                done = dict(done=True, step=adapter["step"] if req.get("adapter", True) else 0,
                            tokens=last.generation_tokens if last else 0,
                            tps=round(last.generation_tps, 1) if last else 0,
                            finish=last.finish_reason if last else None)
                self.wfile.write(b"data: " + json.dumps(done).encode() + b"\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/api/think":
            # chat with a minimum thinking budget (R1-Zero format)
            msgs, mn = req.get("messages", []), int(req.get("min_tokens", 256))
            cue, temp, mx_ = req.get("cue", " Wait,"), float(req.get("temp", 0.7)), int(req.get("max_tokens", 1024))
            text, ev = call(lambda: forced_think(r1z_prompt(msgs), mn, cue, temp, mx_), req.get("adapter", True))
            think_end = min([i for i in (text.find("</think>"), text.find("<answer>")) if i >= 0] or [len(text)])
            self.send_json(dict(text=text, events=ev, cue=cue, step=adapter["step"],
                                think_tokens=call(lambda: ntok(text[:think_end])), answer=answers_of(text)))
        elif self.path == "/api/vote":
            # N independent samples -> answer histogram (+ grading if it is a known test problem)
            n, temp = max(2, min(16, int(req.get("n", 8)))), float(req.get("temp", 0.8))
            q, ent, gold = req.get("question", ""), None, None
            if req.get("id") is not None:
                p = PROMPTS[int(req["id"])]; q, ent, gold = p["question"], ENTRIES[p["id"]], p["answer"]
            outs = call(lambda: batch_gen([r1z_prompt([{"role": "user", "content": q}])] * n, temp), req.get("adapter", True))
            rows = []
            for o in outs:
                a = answers_of(o)
                ok = grade(ent, o)[1] if ent is not None and ent[0] == "rg" else None
                rows.append(dict(answer=a, correct=ok, text=o))
            self.send_json(dict(question=q, gold=gold, rows=rows, step=adapter["step"]))
        elif self.path == "/api/branch":
            # V-curve on one answer: cut the reasoning at 25/50/75%, sample C continuations from each cut
            p = PROMPTS[int(req["id"])]; ent = ENTRIES[p["id"]]
            if ent is None or ent[0] != "rg":
                self.send_json({"error": "pick a reasoning problem (not code/chat)"}); return
            C, temp = max(2, min(8, int(req.get("c", 6)))), float(req.get("temp", 0.7))
            base = r1z_prompt([{"role": "user", "content": p["question"]}])
            want = req.get("want", "any")              # any | wrong | correct
            def job():
                tries = batch_gen([base] * 6, temp)
                oks = [bool(grade(ent, t)[1]) for t in tries]
                v0 = sum(oks) / len(oks)
                pick = [t for t, k in zip(tries, oks) if "<answer>" in t and (want == "any" or k == (want == "correct"))]
                trace = pick[0] if pick else tries[0]
                reasoning = trace[:trace.find("<answer>")] if "<answer>" in trace else trace
                cuts = cut_points(reasoning, [0.25, 0.5, 0.75])
                jobs = [(c, base + reasoning[:c]) for c in cuts for _ in range(C)]
                conts = batch_gen([j[1] for j in jobs], temp) if jobs else []
                res = []
                for k, c in enumerate(cuts):
                    cs = conts[k * C:(k + 1) * C]
                    cok = [bool(grade(ent, reasoning[:c] + x)[1]) for x in cs]
                    res.append(dict(pos=c, v=sum(cok) / C, conts=[dict(text=x, correct=o, answer=answers_of(reasoning[:c] + x)) for x, o in zip(cs, cok)]))
                return dict(trace=trace, reasoning_len=len(reasoning), v0=v0, final=bool(grade(ent, trace)[1]), cuts=res,
                            first_tries=[dict(answer=answers_of(t), correct=k) for t, k in zip(tries, oks)])
            out = call(job, req.get("adapter", True))
            out.update(question=p["question"], gold=p["answer"], task=p["task"], step=adapter["step"])
            self.send_json(out)
        elif self.path == "/api/compare":
            p = PROMPTS[int(req["id"])]
            ent = ENTRIES[p["id"]]
            msgs = [{"role": "user", "content": p["question"]}]
            if args.template:
                if ent is not None and ent[0] == "code":
                    msgs = [{"role": "user", "content": p["question"] + "\nThe answer must be the complete Python code (no explanation) inside the answer tags."}]
            elif ent is not None and ent[0] == "code":
                msgs = [{"role": "system", "content": CODE_SYSTEM}] + msgs     # same prompt as training
            elif args.tools:
                from tools import TOOL_SYSTEM
                msgs = [{"role": "system", "content": TOOL_SYSTEM}] + msgs
            elif not args.no_system:
                msgs = [{"role": "system", "content": SYSTEM}] + msgs
            out = {}
            for name, use in (("base", False), ("current", True)):
                text, last = "", None
                for last in gen(msgs, use, 0.0, 1024):
                    text += last.text
                ext, ok = grade(ent, text)
                out[name] = dict(text=text, tokens=last.generation_tokens if last else 0, extracted=ext, correct=ok)
            out["step"] = adapter["step"]
            self.send_json(out)
        else:
            self.send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    if args.tools or args.template:
        from sandbox.judge import Judge
        JUDGE = Judge(workers=2)
    threading.Thread(target=_worker, daemon=True).start()
    READY.wait()
    print(f"dashboard: http://localhost:{args.port}  (run {args.run})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
