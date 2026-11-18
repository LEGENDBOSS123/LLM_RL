import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Baseline coding eval.  Usage: python code_bench.py MODEL [n_kodcode=400] [k=4] [max_tokens=1024]
Eval sets: kodcode_eval (first n) + leetcode_eval (all 228). Graded in the sandbox.
Reports pass@1 (mean over k), pass@k, code-extraction rate, by source / difficulty / subset.
"""
import sys, json, time, math, re, statistics, itertools
from collections import defaultdict
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler
from sandbox.judge import Judge

name = sys.argv[1]
n_kc = int(sys.argv[2]) if len(sys.argv) > 2 else 400
k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 1024

SYSTEM = ("You are an expert Python programmer. Think step by step inside <think>...</think>, "
          "then give your complete solution in a single ```python code block.")

probs = [json.loads(l) for l in itertools.islice(open("data/code/kodcode_eval.jsonl"), n_kc)]
probs += [json.loads(l) for l in open("data/code/leetcode_eval.jsonl")]


def user_msg(p):
    s = p["prompt"].strip()
    if p["starter"]:
        s += f"\n\nComplete this starter code:\n```python\n{p['starter'].strip()}\n```"
    return s


def extract_code(text):
    body = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", body, re.S)
    return blocks[-1] if blocks else None


model, tok = load(f"models/{name}")
prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(p)}],
                                   add_generation_prompt=True, tokenize=True) for p in probs]
allp = [pp for pp in prompts for _ in range(k)]
print(f"{len(probs)} problems x {k} = {len(allp)} generations", flush=True)
t0 = time.time()
res = batch_generate(model, tok, allp, max_tokens=max_tokens, return_token_ids=True,
                     sampler=make_sampler(temp=0.7, top_p=0.95), verbose=True)
gen_s = time.time() - t0
ntok = sum(len(t) for t in res.token_ids)
print(f"\ngeneration done: {gen_s/60:.1f} min, {ntok/gen_s:.0f} tok/s", flush=True)

t1 = time.time()
codes = [extract_code(t) for t in res.texts]
jobs = [(i, probs[i // k], c) for i, c in enumerate(codes) if c]
scores = [0.0] * len(codes)
with Judge(workers=4) as j:
    from concurrent.futures import ThreadPoolExecutor
    def g(job):
        i, p, c = job
        if p["mode"] == "tests":
            r = j.grade_tests(c, p["test_code"], prefix=p["prefix"], timeout=8)
            return i, 1.0 if r.get("all_pass") else 0.0
        r = j.grade(c, p["io"][:20], timeout=4, partial=False)
        return i, r["score"]
    with ThreadPoolExecutor(4) as ex:
        for i, s in ex.map(g, jobs):
            scores[i] = s
grade_s = time.time() - t1

groups = defaultdict(lambda: defaultdict(list))
samples = []
for pi, p in enumerate(probs):
    sc = scores[pi * k:(pi + 1) * k]
    ex = [codes[pi * k + j] is not None for j in range(k)]
    lens = [len(x) for x in res.token_ids[pi * k:(pi + 1) * k]]
    for key in ("all", f"source={p['source']}", f"difficulty={p['difficulty']}",
                f"{p['source']}/{p['subset']}" if p["source"] == "kodcode" else None):
        if key is None:
            continue
        groups[key]["pass@1"].append(statistics.mean(sc))
        groups[key][f"pass@{k}"].append(float(max(sc) >= 1))
        groups[key]["has_code"].append(statistics.mean(ex))
        groups[key]["truncated"].append(statistics.mean(l >= max_tokens for l in lens))
        groups[key]["tokens"].append(statistics.mean(lens))
    samples.append(dict(id=p["id"], scores=sc, outs=res.texts[pi * k:(pi + 1) * k]))

summary = {g: {m: round(statistics.mean(v), 3) for m, v in d.items()} | {"n": len(d["pass@1"])}
           for g, d in groups.items()}
a = groups["all"]["pass@1"]
ci = 1.96 * statistics.stdev(a) / math.sqrt(len(a))
json.dump(dict(model=name, k=k, max_tokens=max_tokens, summary=summary, gen_min=gen_s / 60, grade_min=grade_s / 60,
               tok_s=ntok / gen_s), open(f"results/code_{name}.json", "w"), indent=2)
json.dump(samples, open(f"results/code_{name}.samples.json", "w"))
print(f"\n{name}: pass@1 {summary['all']['pass@1']:.3f} ±{ci:.3f}   (gen {gen_s/60:.0f} min, grading {grade_s/60:.1f} min)")
print(f"{'group':32}{'n':>5}" + "".join(f"{m:>11}" for m in summary["all"] if m != "n"))
for gname in sorted(summary, key=lambda x: (x != "all", x)):
    r = summary[gname]
    print(f"{gname:32}{r['n']:5}" + "".join(f"{r[m]:11.3f}" for m in r if m != "n"))
