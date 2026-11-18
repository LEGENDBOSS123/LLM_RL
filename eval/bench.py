import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Benchmark a model on held-out reasoning-gym problems.
Usage: python bench.py MODEL [n_per_task=50] [k=4] [max_tokens=512] [temp=0.7]
Reports pass@1 (mean over k samples), pass@k, maj@k, forced@1 (answer forcing), format rate, 95% CIs,
lenient accuracy + which extraction method matched, speed and memory.
"""
import sys, time, json, math, statistics
from collections import Counter, defaultdict
import mlx.core as mx
from mlx_lm import load, stream_generate, batch_generate
from mlx_lm.sample_utils import make_sampler
from common import *

name = sys.argv[1]
n_per_task = int(sys.argv[2]) if len(sys.argv) > 2 else 50
k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 512
temp = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7

model, tok = load(f"models/{name}")
probs = make_problems(n_per_task=n_per_task, seed=EVAL_SEED)

# --- single-stream speed ---
p = tok.apply_chat_template([{"role": "user", "content": "Explain how a binary search works."}],
                            add_generation_prompt=True, tokenize=False)
for _ in stream_generate(model, tok, p, max_tokens=8):
    pass
last = None
for last in stream_generate(model, tok, p, max_tokens=200):
    pass
single_tps = last.generation_tps

# --- k samples per problem, batched ---
prompts = [build_prompt(tok, e["question"]) for _, _, e in probs]
all_prompts = [pp for pp in prompts for _ in range(k)]
sampler = make_sampler(temp=temp, top_p=0.95) if temp > 0 else None
t0 = time.time()
res = batch_generate(model, tok, all_prompts, max_tokens=max_tokens, return_token_ids=True,
                     sampler=sampler, verbose=True)
dt = time.time() - t0
ntok = sum(len(t) for t in res.token_ids)
peak_gb = mx.get_peak_memory() / 1e9


def ci95(xs):
    m = statistics.mean(xs)
    se = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0
    return round(m, 4), round(1.96 * se, 4)


forced = answer_force(model, tok, all_prompts, res.texts)

rows = defaultdict(lambda: defaultdict(list))  # task -> metric -> per-problem values
methods = Counter()
samples = []
for i, (t, ds, e) in enumerate(probs):
    outs = res.texts[i * k:(i + 1) * k]
    lens = [len(x) for x in res.token_ids[i * k:(i + 1) * k]]
    strict = [score(ds, e, o) for o in outs]
    len_res = [score_lenient(ds, e, o) for o in outs]
    lenient = [s for s, _ in len_res]
    methods.update(m for s, m in len_res if s > 0)
    answers = [extract_answer(o) for o in outs]
    f_ans = forced[i * k:(i + 1) * k]
    f_scores = [rg_score(ds, e, a) for a in f_ans]
    voted = Counter(a for a in answers if a is not None).most_common(1)
    maj = rg_score(ds, e, voted[0][0]) if voted else 0.0
    r = rows[t]
    r["pass@1"].append(statistics.mean(strict))
    r[f"pass@{k}"].append(float(max(strict) >= 1.0))
    r[f"maj@{k}"].append(maj)
    r["forced@1"].append(statistics.mean(f_scores))
    r["lenient@1"].append(statistics.mean(lenient))
    r["format"].append(statistics.mean(a is not None for a in answers))
    r["truncated"].append(statistics.mean(l >= max_tokens for l in lens))
    r["tokens"].append(statistics.mean(lens))
    samples.append(dict(task=t, q=e["question"], gold=e["answer"], outs=outs,
                        strict=strict, lenient=len_res, extracted=answers,
                        forced=f_ans, forced_scores=f_scores))

metrics = list(next(iter(rows.values())).keys())
overall = {m: ci95([v for r in rows.values() for v in r[m]]) for m in metrics}
per_task = {t: {m: round(statistics.mean(r[m]), 3) for m in metrics} for t, r in rows.items()}
out = dict(model=name, n_problems=len(probs), k=k, temp=temp, max_tokens=max_tokens,
           overall=overall, per_task=per_task,
           lenient_match_methods=dict(methods),
           single_stream_tok_s=round(single_tps, 1), batch_tok_s=round(ntok / dt, 1),
           wall_s=round(dt, 1), peak_mem_gb=round(peak_gb, 2))
json.dump(out, open(f"results/{name}.json", "w"), indent=2)
json.dump(samples, open(f"results/{name}.samples.json", "w"), indent=1)

print(f"\n{name}  ({len(probs)} problems x {k} samples, temp {temp})")
for m, (v, ci) in overall.items():
    print(f"  {m:10} {v:8.3f}  ±{ci:.3f}")
print(f"  lenient matched by: {dict(methods)}")
print(f"  {'task':18}" + "".join(f"{m:>11}" for m in metrics))
for t, r in per_task.items():
    print(f"  {t:18}" + "".join(f"{r[m]:11.2f}" for m in metrics))
print(f"  speed: {single_tps:.0f} tok/s single, {ntok/dt:.0f} tok/s batched, {dt:.0f}s, peak {peak_gb:.1f}GB")
