import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Context distillation data: sample WITH the think system prompt, keep well-formed + correct
outputs, and save them as (user question WITHOUT system prompt) -> output pairs.

  python sft_data.py [n_per_task=60] [k=4]  ->  data/sft/{train,valid}.jsonl
"""
import json, os, random, re, sys
from collections import Counter
import mlx.core as mx
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler
import reasoning_gym
from common import SYSTEM, TASKS, TRAIN_SEED, strip_think, normalize, rg_score

n_per_task = int(sys.argv[1]) if len(sys.argv) > 1 else 60
k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MAX_KEEP_PER_PROBLEM = 2
mx.set_memory_limit(int(7e9))
mx.set_cache_limit(int(1e9))
random.seed(0)

# training-seed problems only (never the eval seed); offset so SFT and RL problems differ
items = []
for t in TASKS:
    ds = reasoning_gym.create_dataset(t, size=n_per_task, seed=TRAIN_SEED + 500_000)
    items += [(t, ds, e) for e in ds]

model, tok = load("models/Qwen2.5-1.5B-Instruct-bf16")
prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": e["question"]}],
                                   add_generation_prompt=True, tokenize=True) for _, _, e in items]
allp = [p for p in prompts for _ in range(k)]
res = batch_generate(model, tok, allp, max_tokens=512, return_token_ids=True, verbose=True,
                     sampler=make_sampler(temp=0.7, top_p=0.95), completion_batch_size=16, prefill_batch_size=4)


def well_formed(text):
    """exactly: <think>...</think> then exactly one <answer>...</answer>, nothing else odd."""
    t = text.strip()
    if not t.startswith("<think>") or t.count("<think>") != 1 or t.count("</think>") != 1:
        return False
    after = t.split("</think>", 1)[1]
    return len(re.findall(r"<answer>.*?</answer>", after, re.S)) == 1 and "<think>" not in after


rows, stats = [], Counter()
for pi, (t, ds, e) in enumerate(items):
    kept = 0
    for j in range(k):
        i = pi * k + j
        text = res.texts[i]
        if len(res.token_ids[i]) >= 512:
            stats["truncated"] += 1; continue
        if not well_formed(text):
            stats["malformed"] += 1; continue
        ans = re.findall(r"<answer>(.*?)</answer>", strip_think(text), re.S)[-1]
        if rg_score(ds, e, normalize(ans)) < 1:
            stats["wrong"] += 1; continue
        stats["kept"] += 1
        if kept < MAX_KEEP_PER_PROBLEM:
            kept += 1
            rows.append(dict(task=t, messages=[{"role": "user", "content": e["question"]},
                                               {"role": "assistant", "content": text.strip()}]))

print("\n", dict(stats), "per task kept:", dict(Counter(r["task"] for r in rows)))
random.shuffle(rows)
os.makedirs("data/sft", exist_ok=True)
nv = max(20, len(rows) // 20)
for name, part in (("valid", rows[:nv]), ("train", rows[nv:])):
    with open(f"data/sft/{name}.jsonl", "w") as f:
        for r in part:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")
print(f"train {len(rows) - nv}  valid {nv}")
