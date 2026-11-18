import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Find positions in reasoning-heavy text where thinking could help predict the next N tokens.

Usage: python find_hard_positions.py MODEL SOURCE.jsonl OUT.jsonl [max_docs=500] [N=16]

For each doc (first CHUNK tokens): one forward pass gives per-token NLL. A candidate position p
must be at a "reasoning boundary" (sentence end, or right after a cue like 'therefore', '=', 'so').
Score = mean NLL of the next N tokens. We keep positions in the hard-but-not-noise band:
above the doc-pool median, below the 98th percentile (very high NLL is usually URLs, ids,
random numbers or broken LaTeX - unpredictable even with perfect reasoning).
Output rows: {id, pos, context, target, base_nll, cue} - context is the left text, target the
true next N tokens. The RL reward later is the NLL drop on target when a <think> is inserted.
"""
import sys, json, re, random, statistics
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

model_name, src, out_path = sys.argv[1:4]
MAX_DOCS = int(sys.argv[4]) if len(sys.argv) > 4 else 500
N = int(sys.argv[5]) if len(sys.argv) > 5 else 16
CHUNK, MIN_CTX, PER_DOC = 1024, 128, 8

CUES = re.compile(r"(therefore|thus|hence|so|because|since|which means|it follows|we get|"
                  r"implies|this gives|then|=|:)\s*$", re.I)

model, tok = load(f"models/{model_name}")
docs = [json.loads(l) for l in open(src)][:MAX_DOCS]
cands = []
for d in docs:
    ids = tok.encode(d["text"])[:CHUNK]
    if len(ids) < MIN_CTX + N + 1:
        continue
    x = mx.array(ids)[None]
    logits = model(x[:, :-1])
    nll = nn.losses.cross_entropy(logits, x[:, 1:], reduction="none")[0]   # nll[i] = -log p(ids[i+1])
    nll = nll.astype(mx.float32).tolist()
    # boundary positions p: predict ids[p : p+N] from ids[:p]
    for p in range(MIN_CTX, len(ids) - N):
        prev = tok.decode(ids[max(0, p - 6):p])
        cue = None
        m = CUES.search(prev)
        if m:
            cue = m.group(1).lower()
        elif re.search(r"[.!?]\s*$|\n\s*$", prev):
            cue = "sentence"
        if cue is None:
            continue
        cands.append(dict(id=d["id"], pos=p, cue=cue,
                          base_nll=statistics.mean(nll[p - 1:p - 1 + N]), _ids=ids))
    print(f"\r{d['id']}  candidates={len(cands)}", end="", file=sys.stderr)

vals = sorted(c["base_nll"] for c in cands)
lo, hi = vals[len(vals) // 2], vals[int(len(vals) * 0.98)]
random.seed(0)
by_doc = {}
for c in cands:
    if lo < c["base_nll"] < hi:
        by_doc.setdefault(c["id"], []).append(c)
rows = []
for doc_id, cs in by_doc.items():
    cs.sort(key=lambda c: -c["base_nll"])
    for c in cs[:PER_DOC]:                    # cap per doc so one doc can't dominate
        ids = c.pop("_ids")
        rows.append(dict(c, context=tok.decode(ids[:c["pos"]]), target=tok.decode(ids[c["pos"]:c["pos"] + N]),
                         base_nll=round(c["base_nll"], 3)))
with open(out_path, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
from collections import Counter
print(f"\n{len(cands)} candidates, NLL band ({lo:.2f}, {hi:.2f}) -> {len(rows)} positions written to {out_path}")
print("cues:", Counter(r["cue"] for r in rows).most_common(10))
