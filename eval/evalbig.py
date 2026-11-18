import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Bigger, PAIRED evaluation: every model sees exactly the same problems; differences get bootstrap 95% CIs.

Sets (all never trained on):
  unseen   : 7 unseen reasoning-gym tasks x 40 problems, K=8 samples -> pass@1, maj@8, pass@8
  gsm8k    : first 150 test problems, K=2 -> pass@1
  math500  : first 100 test problems, K=2 -> pass@1 (math-verify)
  general  : 120 held-out general problems (science/econ/... MCQ, numeric), K=2 -> pass@1
Per-problem results cached in results/evalbig_<name>.json, so new models can be compared later without re-running old ones.
usage: python evalbig.py run  <model_path> <name>
       python evalbig.py compare <nameA> <nameB> [...]
"""
import json, os, random, re, statistics, sys, itertools, logging
for _n in ("math_verify", "math_verify.parser", "math_verify.grader"):
    logging.getLogger(_n).setLevel(logging.ERROR)

R1Z = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
       "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
       "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
       "i.e., <think> reasoning process here </think> <answer> answer here </answer>.")
UNSEEN = ["letter_counting", "family_relationships", "word_sorting", "number_sequence", "course_schedule", "circuit_logic", "word_ladder"]
N_UNSEEN, K_UNSEEN, K_OTHER = 40, 8, 2
TOOLS = os.environ.get("TOOLS") == "1"
from tools import R1Z_TOOLS
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def grade_gen(gold, at, pred):
    pred = (pred or "").strip()
    if not pred or len(pred) > 160:
        return False
    try:
        if at == "mcq":
            m = re.search(r"\b([A-E])\b", pred.upper()); return bool(m) and m.group(1) == gold.upper()
        if at == "bool":
            f = lambda x: {"yes": 1, "true": 1, "no": 0, "false": 0}.get(re.sub(r"[^a-z]", "", x.lower()), -1)
            return f(pred) == f(gold) != -1
        if at == "str":
            n = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
            return n(pred) == n(gold) or (len(n(gold)) > 3 and n(gold) in n(pred))
        g, p = _NUM.findall(gold.replace(",", "")), _NUM.findall(pred.replace(",", ""))
        if at == "num" and g and p and abs(float(g[0]) - float(p[0])) <= max(1e-6, 1e-2 * abs(float(g[0]))):
            return True
        from math_verify import parse, verify
        return bool(verify(parse("$" + gold + "$", parsing_timeout=None), parse("$" + pred + "$", parsing_timeout=None),
                           timeout_seconds=None))
    except Exception:
        return False


def answer(text):
    a = re.findall(r"<answer>(.*?)</answer>", text.split("</think>")[-1], re.S)
    return a[0].strip() if a else None


def boxed(s):
    k = s.rfind("\\boxed{")
    if k < 0:
        return None
    i, d, out = k + 7, 1, ""
    while i < len(s) and d:
        c = s[i]; d += (c == "{") - (c == "}")
        if d: out += c
        i += 1
    return out


def build_sets():
    import reasoning_gym
    from common import EVAL_SEED, normalize, rg_score
    sets = {"unseen": [], "gsm8k": [], "math500": [], "general": []}
    for t in UNSEEN:
        ds = reasoning_gym.create_dataset(t, size=N_UNSEEN, seed=EVAL_SEED + 4242)
        for i, e in enumerate(ds):
            sets["unseen"].append(dict(id=f"{t}/{i}", q=e["question"],
                                       check=(lambda a, ds=ds, e=e: rg_score(ds, e, normalize(a)) >= 1), group=t))
    for i, r in enumerate(itertools.islice(open("data/gsm8k_test.jsonl"), 150)):
        r = json.loads(r); g = r["answer"].split("####")[-1].strip().replace(",", "")
        sets["gsm8k"].append(dict(id=f"gsm/{i}", q=r["question"], check=(lambda a, g=g: grade_gen(g, "num", a)), group="gsm8k"))
    for i, r in enumerate(itertools.islice(open("data/math500_test.jsonl"), 100)):
        r = json.loads(r); g = r.get("answer") or boxed(r["solution"])
        sets["math500"].append(dict(id=f"math/{i}", q=r["problem"], check=(lambda a, g=g: grade_gen(g, "expr", a)),
                                    group=r.get("subject", "math")))
    rows = [json.loads(l) for l in open("data/general/eval.jsonl")]
    random.Random(0).shuffle(rows)
    for i, r in enumerate(rows[:120]):
        sets["general"].append(dict(id=f"gen/{r['id']}", q=r["question"],
                                    check=(lambda a, r=r: grade_gen(r["gold"], r["atype"], a)), group=r["source"]))
    return sets


def run(model_path, name):
    import mlx.core as mx
    mx.set_memory_limit(int(6e9))
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler
    from tools import generate_with_tools
    m, tok = load(model_path)
    samp = make_sampler(temp=0.7, top_p=0.95)
    out = {}
    judge = None
    if TOOLS:
        from sandbox.judge import Judge
        judge = Judge(workers=2)
    for sname, items in build_sets().items():
        K = K_UNSEEN if sname == "unseen" else K_OTHER
        prompts = [tok.encode((R1Z_TOOLS if TOOLS else R1Z) + f" User: {it['q']} Assistant:") for it in items for _ in range(K)]
        r = generate_with_tools(m, tok, prompts, judge, max_tokens=1024 if TOOLS else 768, tools=TOOLS, stop_answer=True,
                                stop_user=True, sampler=samp, completion_batch_size=16, backtrack=TOOLS, max_calls=4)
        res = []
        for n, it in enumerate(items):
            outs = r["texts"][n * K:(n + 1) * K]
            ans = [answer(o) for o in outs]
            oks = [a is not None and bool(it["check"](a)) for a in ans]
            votes = [a for a in ans if a is not None]
            top = max(set(votes), key=votes.count) if votes else None
            maj = top is not None and bool(it["check"](top))
            res.append(dict(id=it["id"], group=it["group"], p1=statistics.mean(oks), pk=float(any(oks)), maj=float(maj),
                            toks=statistics.mean(len(tok.encode(o)) for o in outs)))
        out[sname] = res
        print(f"{name} {sname:8} n={len(res):3} pass@1 {statistics.mean(x['p1'] for x in res):.3f}"
              + (f" maj@{K} {statistics.mean(x['maj'] for x in res):.3f} pass@{K} {statistics.mean(x['pk'] for x in res):.3f}" if K > 2 else "")
              + f" | avg tokens {statistics.mean(x['toks'] for x in res):.0f}", flush=True)
    json.dump(out, open(f"results/evalbig_{name}.json", "w"))


def boot_ci(diffs, n=4000):
    rng = random.Random(0)
    ms = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def compare(names):
    R = {n: json.load(open(f"results/evalbig_{n}.json")) for n in names}
    a = names[0]
    for b in names[1:]:
        print(f"\n=== {b} vs {a} (paired, same problems; 95% bootstrap CI of the difference) ===")
        for s in R[a]:
            A = {x["id"]: x for x in R[a][s]}; B = {x["id"]: x for x in R[b][s]}
            ids = [i for i in A if i in B]
            for met in (["p1", "maj", "pk"] if s == "unseen" else ["p1"]):
                d = [B[i][met] - A[i][met] for i in ids]
                lo, hi = boot_ci(d)
                sig = "  *" if lo > 0 or hi < 0 else ""
                print(f"  {s:8} {met:4} {a} {statistics.mean(A[i][met] for i in ids):.3f}  {b} {statistics.mean(B[i][met] for i in ids):.3f}"
                      f"  diff {statistics.mean(d):+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")
        if "unseen" in R[a]:
            print("  per unseen task pass@8:", ", ".join(
                f"{t} {statistics.mean(x['pk'] for x in R[a]['unseen'] if x['group'] == t):.2f}->"
                f"{statistics.mean(x['pk'] for x in R[b]['unseen'] if x['group'] == t):.2f}" for t in UNSEEN))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        compare(sys.argv[2:])
