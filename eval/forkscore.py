import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Which token positions are real decision points ("forks")?

For each of Z's answers on unseen problems, score every reasoning position by:
  shannon  : entropy of the next-token distribution
  rao_emb  : Rao quadratic entropy  sum_ij p_i p_j (1 - cos(E_i, E_j))  over top-k, E = token embeddings
  rao_look : same, but distance between the model's hidden state AFTER appending each candidate (1-step look-ahead)
             (computed on the top-20% positions by shannon only)
  random   : baseline
Ground truth at a chosen position: force the top-2 candidate tokens, sample C continuations each,
  gap = |V(token1) - V(token2)|  (how much that single choice changes the chance of ending correct).
A better fork detector -> larger mean gap.
usage: python forkscore.py [model] [problems_per_task]
"""
import json, math, random, re, statistics, sys
import mlx.core as mx
mx.set_memory_limit(int(6e9))
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from common import normalize, rg_score
from tools import generate_with_tools
from diagnose_items import R1Z, UNSEEN, items

path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-1.7B-RL"
PER_TASK = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TOPK, C, NPOS = 5, 4, 2
random.seed(0)
model, tok = load(path)
samp = make_sampler(temp=0.7, top_p=0.95)
E = model.model.embed_tokens.weight.astype(mx.float32) if hasattr(model.model.embed_tokens, "weight") else None
probs = []
for t in UNSEEN:
    probs += [it for it in items if it[0] == t][:PER_TASK]
if len(sys.argv) > 3:
    probs = probs[:int(sys.argv[3])]
P = [tok.encode(R1Z + f" User: {e['question']} Assistant:") for _, _, e in probs]


def gen(id_lists):
    r = generate_with_tools(model, tok, id_lists, None, max_tokens=640, tools=False, stop_answer=True, stop_user=True,
                            sampler=samp, completion_batch_size=16)
    return r


def ok(q, text):
    t, ds, e = probs[q]
    a = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    return bool(a) and rg_score(ds, e, normalize(a[0].strip())) >= 1


def embed_rows(ids):
    if E is not None:
        return E[mx.array(ids)]
    return model.model.embed_tokens(mx.array(ids)).astype(mx.float32)   # quantized embedding fallback


def cosdist(V):
    V = V / mx.maximum(mx.linalg.norm(V, axis=-1, keepdims=True), 1e-6)
    return 1 - V @ V.T


r0 = gen(P)
print("answers generated", flush=True)
records = []
for q in range(len(probs)):
    comp = r0["token_ids"][q]
    text = r0["texts"][q]
    if "<answer>" not in text:
        continue
    # reasoning length in tokens (before <answer>)
    acc, ans_k = "", len(comp)
    for k, t_ in enumerate(comp):
        acc += tok.decode([t_])
        if "<answer>" in acc:
            ans_k = k
            break
    if ans_k < 12:
        continue
    ids = mx.array([P[q] + comp])
    logits = model(ids)[0].astype(mx.float32)                     # [T, V]
    base = len(P[q]) - 1                                          # logits[base + k] predicts comp[k]
    lp = logits[base:base + ans_k] - mx.logsumexp(logits[base:base + ans_k], axis=-1, keepdims=True)
    p = mx.exp(lp)
    H = -(p * lp).sum(-1)
    top = mx.argsort(-p, axis=-1)[:, :TOPK]
    ptop = mx.take_along_axis(p, top, axis=-1)
    mx.eval(H, top, ptop)
    H, top, ptop = H.tolist(), top.tolist(), ptop.tolist()
    rao_emb = []
    for k in range(ans_k):
        D = cosdist(embed_rows(top[k]))
        pv = mx.array(ptop[k])
        rao_emb.append((pv[:, None] * pv[None, :] * D).sum().item())
    # look-ahead Rao on the top-20% shannon positions (top-3 candidates)
    cand = sorted(range(1, ans_k), key=lambda k: -H[k])[:max(3, ans_k // 5)]
    rao_look = {}
    for k in cand:
        pre = P[q] + comp[:k]
        batch = mx.array([pre + [top[k][j]] for j in range(3)])
        h = model.model(batch)[:, -1, :].astype(mx.float32)
        D = cosdist(h)
        pv = mx.array(ptop[k][:3])
        rao_look[k] = (pv[:, None] * pv[None, :] * D).sum().item()
    picks = {
        "shannon": sorted(range(1, ans_k), key=lambda k: -H[k])[:NPOS],
        "rao_emb": sorted(range(1, ans_k), key=lambda k: -rao_emb[k])[:NPOS],
        "rao_look": sorted(rao_look, key=lambda k: -rao_look[k])[:NPOS],
        "random": random.sample(range(1, ans_k), min(NPOS, ans_k - 1)),
    }
    records.append(dict(q=q, comp=comp, picks=picks, top=top, ptop=ptop, H=H, rao_emb=rao_emb, rao_look=rao_look))
    mx.clear_cache()
print(f"scored {len(records)} traces", flush=True)

# ground truth: force top-2 tokens at each picked position, C continuations each
jobs, keyset = [], {}
for ri, r in enumerate(records):
    for pos in {k for ks in r["picks"].values() for k in ks}:
        for j in range(2):
            key = (ri, pos, j)
            keyset[key] = len(jobs)
            for _ in range(C):
                jobs.append((ri, pos, j))
res = gen([P[records[ri]["q"]] + records[ri]["comp"][:pos] + [records[ri]["top"][pos][j]] for ri, pos, j in jobs])
V = {}
for n, (ri, pos, j) in enumerate(jobs):
    r = records[ri]
    full = tok.decode(r["comp"][:pos] + [r["top"][pos][j]]) + res["texts"][n]
    V.setdefault((ri, pos, j), []).append(ok(r["q"], full))

gaps = {m: [] for m in ("shannon", "rao_emb", "rao_look", "random")}
examples = []
for ri, r in enumerate(records):
    for m, ks in r["picks"].items():
        for pos in ks:
            v1, v2 = statistics.mean(V[(ri, pos, 0)]), statistics.mean(V[(ri, pos, 1)])
            gaps[m].append(abs(v1 - v2))
            if m == "rao_look":
                examples.append((abs(v1 - v2), probs[r["q"]][0], tok.decode(r["comp"][max(0, pos - 12):pos]),
                                 tok.decode([r["top"][pos][0]]), tok.decode([r["top"][pos][1]]),
                                 r["ptop"][pos][0], r["ptop"][pos][1], v1, v2))

print(f"\nmodel {path} | {len(records)} traces | top-2 tokens forced, {C} continuations each\n")
print(f"{'method':10} {'mean |V1-V2|':>13} {'gap>=0.5':>9}   n")
for m, g in gaps.items():
    if g:
        print(f"{m:10} {statistics.mean(g):13.3f} {sum(x >= 0.5 for x in g) / len(g):9.2f}   {len(g)}")
print(f"\n(noise floor: two identical choices with C={C} samples differ by ~0.2-0.3 on average)")
# token pair types the embedding version merges/misses
print("\nlook-ahead forks (largest gaps):")
for ex in sorted(examples, reverse=True)[:10]:
    g, task, ctx, a, b, pa, pb, v1, v2 = ex
    print(f"  gap {g:.2f} {task:20} ...{ctx[-60:]!r} -> {a!r} ({pa:.2f}, V={v1:.2f}) vs {b!r} ({pb:.2f}, V={v2:.2f})")
with open("results/forkscore_rows.jsonl", "w") as f:
    for ri, r in enumerate(records):
        f.write(json.dumps(dict(task=probs[r["q"]][0], picks=r["picks"],
                                V={f"{p_}:{j}": statistics.mean(v) for (rj, p_, j), v in V.items() if rj == ri})) + "\n")
