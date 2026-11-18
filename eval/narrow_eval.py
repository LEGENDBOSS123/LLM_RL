import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Held-out instances of the TRAINED task families (eval seed, never trained on), K=8 samples:
pass@1 / maj@8 / pass@8 per family. Used to see whether long narrow training "unlocks" a family.
usage: python narrow_eval.py run <model> <name> <task1,task2,...> [n_per_task]
       python narrow_eval.py compare <nameA> <nameB>
"""
import json, random, statistics, sys
import os
from evalbig import R1Z, R1Z_TOOLS, TOOLS, answer, boot_ci

K = 8


def run(path, name, tasks, n):
    import mlx.core as mx
    mx.set_memory_limit(int(6e9))
    import reasoning_gym
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler
    from common import EVAL_SEED, normalize, rg_score
    from tools import generate_with_tools
    m, tok = load(path)
    items = []
    for t in tasks:
        ds = reasoning_gym.create_dataset(t, size=n, seed=EVAL_SEED + 777)
        items += [(t, i, ds, e) for i, e in enumerate(ds)]
    judge = None
    if TOOLS:
        from sandbox.judge import Judge
        judge = Judge(workers=2)
    P = [tok.encode((R1Z_TOOLS if TOOLS else R1Z) + f" User: {e['question']} Assistant:") for _, _, _, e in items for _ in range(K)]
    r = generate_with_tools(m, tok, P, judge, max_tokens=1024 if TOOLS else 768, tools=TOOLS, stop_answer=True, stop_user=True,
                            sampler=make_sampler(temp=0.7, top_p=0.95), completion_batch_size=16, backtrack=TOOLS, max_calls=4)
    res = []
    for q, (t, i, ds, e) in enumerate(items):
        ans = [answer(o) for o in r["texts"][q * K:(q + 1) * K]]
        oks = [a is not None and rg_score(ds, e, normalize(a)) >= 1 for a in ans]
        votes = [normalize(a) for a in ans if a is not None]
        top = max(set(votes), key=votes.count) if votes else None
        res.append(dict(id=f"{t}/{i}", group=t, p1=statistics.mean(oks), pk=float(any(oks)),
                        maj=float(top is not None and rg_score(ds, e, top) >= 1)))
    json.dump(res, open(f"results/narrow_{name}.json", "w"))
    for t in tasks:
        R = [x for x in res if x["group"] == t]
        print(f"{name} {t:16} pass@1 {statistics.mean(x['p1'] for x in R):.3f} maj@8 {statistics.mean(x['maj'] for x in R):.3f} pass@8 {statistics.mean(x['pk'] for x in R):.3f}", flush=True)


def compare(a, b):
    A = {x["id"]: x for x in json.load(open(f"results/narrow_{a}.json"))}
    B = {x["id"]: x for x in json.load(open(f"results/narrow_{b}.json"))}
    ids = [i for i in A if i in B]
    groups = sorted({A[i]["group"] for i in ids})
    print(f"=== narrow families: {b} vs {a} (paired, 95% CI) ===")
    for g in groups + ["ALL"]:
        gi = [i for i in ids if g == "ALL" or A[i]["group"] == g]
        for met in ("p1", "maj", "pk"):
            d = [B[i][met] - A[i][met] for i in gi]
            lo, hi = boot_ci(d)
            print(f"  {g:16} {met:4} {statistics.mean(A[i][met] for i in gi):.3f} -> {statistics.mean(B[i][met] for i in gi):.3f}"
                  f"  diff {statistics.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]{'  *' if lo > 0 or hi < 0 else ''}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4].split(","), int(sys.argv[5]) if len(sys.argv) > 5 else 30)
    else:
        compare(sys.argv[2], sys.argv[3])
