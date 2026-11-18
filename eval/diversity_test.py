import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Base vs instruct output diversity on held-out problems (same family, same quantization)."""
import re, statistics, sys
import mlx.core as mx
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler
import reasoning_gym
from common import TASKS, EVAL_SEED, normalize, rg_score
mx.set_memory_limit(int(3e9))
R1Z = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
       "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
       "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
       "i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {q} Assistant:")
K, MAXT = 8, 384
items = []
for t in TASKS:
    ds = reasoning_gym.create_dataset(t, size=2, seed=EVAL_SEED + 99)
    items += [(ds, e) for e in ds]

def run(path, mode):
    m, tok = load(path)
    if mode == "base":
        prompts = [tok.encode(R1Z.format(q=e["question"])) for _, e in items]
    else:
        prompts = [tok.apply_chat_template([{"role": "user", "content": R1Z.split(' User:')[0] + "\n\n" + e["question"]}],
                                           add_generation_prompt=True, tokenize=True) for _, e in items]
    allp = [p for p in prompts for _ in range(K)]
    res = batch_generate(m, tok, allp, max_tokens=MAXT, return_token_ids=True, sampler=make_sampler(temp=1.0),
                         completion_batch_size=16)
    ents, distinct_open, ngram_div, acc, fmt = [], [], [], [], []
    for pi, (ds, e) in enumerate(items):
        outs = res.texts[pi * K:(pi + 1) * K]
        ids = res.token_ids[pi * K:(pi + 1) * K]
        outs = [o.split("</answer>")[0] + ("</answer>" if "</answer>" in o else "") for o in outs]
        distinct_open.append(len({tuple(x[:12]) for x in ids}))
        grams = [tuple(w) for o in outs for w in zip(*[o.split()[i:] for i in range(4)])]
        ngram_div.append(len(set(grams)) / max(len(grams), 1))
        for o in outs:
            a = re.findall(r"<answer>(.*?)</answer>", o, re.S)
            fmt.append(bool(a))
            acc.append(bool(a) and rg_score(ds, e, normalize(a[-1])) >= 1)
    # per-token entropy on a subset (teacher-forced over the sampled tokens)
    for k in range(0, len(allp), 16):
        seq = allp[k] + res.token_ids[k][:200]
        lg = m(mx.array(seq)[None])[0, len(allp[k]) - 1:-1].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, -1, keepdims=True)
        ents.append((-(mx.exp(lp) * lp).sum(-1)).mean().item())
    print(f"{mode:9} | entropy/token {statistics.mean(ents):.3f} | distinct openings (of {K}) {statistics.mean(distinct_open):.2f} "
          f"| 4-gram diversity {statistics.mean(ngram_div):.3f} | format {statistics.mean(fmt):.2f} | accuracy {statistics.mean(acc):.3f} "
          f"| avg tokens {statistics.mean(len(x) for x in res.token_ids):.0f}", flush=True)
    print("   sample:", repr(res.texts[0][:300]), flush=True)
    del m

run("models/Qwen3-1.7B-Base-4bit", "base")
run("models/Qwen3-1.7B-Instruct-4bit", "instruct")
