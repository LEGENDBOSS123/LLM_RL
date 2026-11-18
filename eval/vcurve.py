import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""V-curve test: Monte-Carlo value of reasoning prefixes.

For each unseen problem: sample K answers (V(0) = their success rate). Take one correct and one wrong answer,
cut the reasoning (text before the first <answer>) at 25/50/75% (snapped back to a line/sentence boundary),
sample C continuations from each cut, V(cut) = fraction correct.
  correct traces: does V rise along the reasoning (prefix carries real progress) or stay flat (answer = re-draw)?
  wrong traces:   where does V drop (error location)? how often is V(cut) > 0 (rescuable)?
usage: python vcurve.py [model] [problems_per_task]
"""
import json, re, statistics, sys, random
import mlx.core as mx
mx.set_memory_limit(int(6e9))
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from common import normalize, rg_score
from tools import generate_with_tools
from diagnose_items import R1Z, UNSEEN, items

path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-1.7B-RL"
PER_TASK = int(sys.argv[2]) if len(sys.argv) > 2 else 5
K, C, CUTS = 8, 8, [0.25, 0.5, 0.75]
random.seed(0)

m, tok = load(path)
samp = make_sampler(temp=0.7, top_p=0.95)
probs = []
for t in UNSEEN:
    probs += [it for it in items if it[0] == t][:PER_TASK]
P = [R1Z + f" User: {e['question']} Assistant:" for _, _, e in probs]


def first_answer(text):
    a = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    return a[0].strip() if a else None


def ok(q, text):
    t, ds, e = probs[q]
    a = first_answer(text)
    return a is not None and rg_score(ds, e, normalize(a)) >= 1


def gen(strs):
    r = generate_with_tools(m, tok, [tok.encode(s) for s in strs], None, max_tokens=640, tools=False,
                            stop_answer=True, stop_user=True, sampler=samp, completion_batch_size=16)
    return r["texts"]


def cut_at(reasoning, frac):
    """snap back to the last newline / sentence end before frac of the reasoning"""
    target = int(len(reasoning) * frac)
    bounds = [mm.end() for mm in re.finditer(r"\n+|(?<=[.!?:])\s", reasoning[:target])]
    pos = bounds[-1] if bounds and bounds[-1] > len(reasoning) * frac * 0.5 else target
    return reasoning[:pos]


# round 1: K answers per problem
outs = gen([p for p in P for _ in range(K)])
print("round 1 done", flush=True)
traces = []   # (problem, kind, text, V0)
for q in range(len(probs)):
    o = outs[q * K:(q + 1) * K]
    oks = [ok(q, x) for x in o]
    v0 = statistics.mean(oks)
    good = [x for x, k in zip(o, oks) if k and "<answer>" in x]
    bad = [x for x, k in zip(o, oks) if not k and "<answer>" in x]
    if good:
        traces.append((q, "correct", random.choice(good), v0))
    if bad:
        traces.append((q, "wrong", random.choice(bad), v0))

# round 2: C continuations from each cut
jobs = []
for ti, (q, kind, text, v0) in enumerate(traces):
    reasoning = text[:text.find("<answer>")]
    for f in CUTS:
        pre = cut_at(reasoning, f)
        jobs += [(ti, f, pre)] * C
conts = gen([P[traces[ti][0]] + pre for ti, f, pre in jobs])
V = {}
for (ti, f, pre), c in zip(jobs, conts):
    V.setdefault((ti, f), []).append(ok(traces[ti][0], pre + c))

rows = []
for ti, (q, kind, text, v0) in enumerate(traces):
    vs = [statistics.mean(V[(ti, f)]) for f in CUTS]
    rows.append(dict(task=probs[q][0], q=q, kind=kind, V0=v0, V=dict(zip(map(str, CUTS), vs)), text=text,
                     prefixes={str(f): cut_at(text[:text.find("<answer>")], f) for f in CUTS}))

hdr = "  ".join(f"V({int(f*100)}%)" for f in CUTS)
print(f"\nmodel {path} | {len(probs)} problems, K={K} answers, C={C} continuations per cut\n")
for kind in ("correct", "wrong"):
    R = [r for r in rows if r["kind"] == kind]
    if not R:
        continue
    v0 = statistics.mean(r["V0"] for r in R)
    vs = [statistics.mean(r["V"][str(f)] for r in R) for f in CUTS]
    print(f"{kind:8} traces n={len(R):3} | V(0) {v0:.2f}  " + "  ".join(f"V({int(f*100)}%) {v:.2f}" for f, v in zip(CUTS, vs)))
    gains = [r["V"][str(CUTS[-1])] - r["V0"] for r in R]
    print(f"           prefix gain V(75%)-V(0): mean {statistics.mean(gains):+.2f} | "
          f"up >0.2: {sum(g > 0.2 for g in gains)}  flat: {sum(abs(g) <= 0.2 for g in gains)}  down <-0.2: {sum(g < -0.2 for g in gains)}")
    if kind == "wrong":
        resc = sum(any(r["V"][str(f)] > 0 for f in CUTS) for r in R)
        print(f"           rescuable (some cut has V>0): {resc}/{len(R)}")
        allfail = [r for r in R if r["V0"] == 0]
        if allfail:
            print(f"           problems with V(0)=0 (all {K} wrong): {len(allfail)}, rescued by a cut: "
                  f"{sum(any(r['V'][str(f)] > 0 for f in CUTS) for r in allfail)}")
        drops = []
        for r in R:   # largest drop between consecutive points 0 -> 25 -> 50 -> 75 -> answer(0)
            pts = [r["V0"]] + [r["V"][str(f)] for f in CUTS] + [0.0]
            d = [pts[i] - pts[i + 1] for i in range(len(pts) - 1)]
            drops.append(max(range(len(d)), key=lambda i: d[i]))
        names = ["0-25%", "25-50%", "50-75%", "75%-answer"]
        print("           biggest drop located in: " + ", ".join(f"{n} {drops.count(i)}" for i, n in enumerate(names)))
print("\nper task (correct traces V0 -> V25/50/75):")
for t in UNSEEN:
    R = [r for r in rows if r["task"] == t and r["kind"] == "correct"]
    if R:
        print(f"  {t:22} V0 {statistics.mean(r['V0'] for r in R):.2f} -> " +
              " ".join(f"{statistics.mean(r['V'][str(f)] for r in R):.2f}" for f in CUTS))
with open("results/vcurve_rows.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
