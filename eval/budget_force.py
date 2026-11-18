import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Test-time budget forcing (s1-style) on the unseen diagnostic set - no training.

Round 0 = normal generation (stop at </answer>) -> 'unforced' baseline.
Then for each minimum budget B in BUDGETS: any sample that answered before B generated tokens gets its
<answer>... cut off, ' Wait,' appended, and continues; repeated until it reaches B (or MAX_ROUNDS).
Scores the LAST answer. Reports pass@1 / maj@8 / pass@8, avg length, and answer flips (right->wrong, wrong->right).
usage: python budget_force.py [model] [B1,B2,...]
"""
import json, re, statistics, sys
import mlx.core as mx
mx.set_memory_limit(int(6e9))
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
import reasoning_gym
from common import EVAL_SEED, normalize, rg_score
from tools import generate_with_tools
from diagnose_items import R1Z, UNSEEN, K, items

path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-1.7B-RL"
BUDGETS = [int(b) for b in (sys.argv[2] if len(sys.argv) > 2 else "256,512").split(",")]
MAX_ROUNDS, MAX_TOTAL, CUE = 4, 1024, " Wait,"

m, tok = load(path)
samp = make_sampler(temp=0.7, top_p=0.95)
P = [R1Z + f" User: {e['question']} Assistant:" for _, _, e in items]
prompts = [p for p in P for _ in range(K)]
ntok = lambda s: len(tok.encode(s, add_special_tokens=False))


def answers(text):
    return [a.strip() for a in re.findall(r"<answer>(.*?)</answer>", text, re.S)]


def gen(prefixes, budgets):
    """continue each prompt+prefix; returns new text appended."""
    ids = [tok.encode(prompts[i] + prefixes[i]) for i in range(len(prefixes))]
    out = [""] * len(prefixes)
    groups = {}
    for i, b in enumerate(budgets):
        groups.setdefault(b, []).append(i)
    for b, idx in groups.items():
        r = generate_with_tools(m, tok, [ids[i] for i in idx], None, max_tokens=b, tools=False,
                                stop_answer=True, stop_user=True, sampler=samp, completion_batch_size=16)
        for i, t in zip(idx, r["texts"]):
            out[i] = t
    return out


cur = gen([""] * len(prompts), [640] * len(prompts))
states = {0: list(cur)}
for B in BUDGETS:
    for rnd in range(MAX_ROUNDS):
        need = [i for i, t in enumerate(cur) if answers(t) and ntok(t) < B and ntok(t) < MAX_TOTAL - 64]
        if not need:
            break
        pre = {i: cur[i][:cur[i].rfind("<answer>")].rstrip() + CUE for i in need}
        cont = gen([pre[i] for i in need], [min(640, MAX_TOTAL - ntok(pre[i])) for i in need])
        for i, c in zip(need, cont):
            cur[i] = pre[i] + c
        print(f"budget {B} round {rnd}: forced {len(need)}", flush=True)
    states[B] = list(cur)


def score(texts):
    per = {t: {"p1": [], "pk": [], "maj": []} for t in UNSEEN}
    oks = []
    for q, (t, ds, e) in enumerate(items):
        outs = texts[q * K:(q + 1) * K]
        ok = [bool(a := answers(o)) and rg_score(ds, e, normalize(a[-1])) >= 1 for o in outs]
        oks += ok
        per[t]["p1"].append(statistics.mean(ok)); per[t]["pk"].append(float(any(ok)))
        votes = [normalize(a[-1]) for o in outs if (a := answers(o))]
        top = max(set(votes), key=votes.count) if votes else None
        per[t]["maj"].append(float(top is not None and rg_score(ds, e, top) >= 1))
    agg = {k: statistics.mean(x for t in UNSEEN for x in per[t][k]) for k in ("p1", "maj", "pk")}
    return agg, per, oks


base_ok = None
for B, texts in states.items():
    agg, per, oks = score(texts)
    L = statistics.mean(ntok(t) for t in texts)
    line = f"{'unforced' if B == 0 else f'min {B} tokens':>16} | pass@1 {agg['p1']:.3f} | maj@{K} {agg['maj']:.3f} | pass@{K} {agg['pk']:.3f} | avg tokens {L:.0f}"
    if base_ok is None:
        base_ok = oks
    else:
        rw = sum(a and not b for a, b in zip(base_ok, oks)); wr = sum(b and not a for a, b in zip(base_ok, oks))
        line += f" | flips right->wrong {rw}, wrong->right {wr}"
    print(line)
    print("   " + "  ".join(f"{t[:12]} {statistics.mean(per[t]['p1']):.2f}" for t in UNSEEN))
with open("results/budget_force_samples.jsonl", "w") as f:
    for i in range(0, len(prompts), K * 3):
        f.write(json.dumps({"task": items[i // K][0], **{str(B): states[B][i] for B in states}}) + "\n")
