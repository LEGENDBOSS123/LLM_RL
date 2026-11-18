import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Cold-start data for tool use (<code>...</code> -> sandbox -> <output>...</output>).

Samples the given model on training problems (27 reasoning-gym tasks + some KodCode), half of the samples
nudged with a short lead-in that opens a code block, half free. Keeps correct + well-formed answers,
balances ~60% tool / 40% no-tool, and stores each answer as SEGMENTS (text, trainable) so the sandbox
output can be masked out of the SFT loss.

  python sft_tools_data.py MODEL [n_per_task=12] [n_code=60]  ->  data/sft_tools/{train,valid}.jsonl
"""
import json, os, random, re, sys, itertools
from collections import Counter
import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
import reasoning_gym
from common import TRAIN_SEED, normalize, rg_score
from tools import generate_with_tools, TOOL_SYSTEM, TOOL_SYSTEM_CODE
from sandbox.judge import Judge

MODEL = sys.argv[1]
N_TASK = int(sys.argv[2]) if len(sys.argv) > 2 else 12
N_CODE = int(sys.argv[3]) if len(sys.argv) > 3 else 60
K = 4
mx.set_memory_limit(int(7e9))
mx.set_cache_limit(int(1e9))
random.seed(1)

TASKS = ("chain_sum,base_conversion,gcd,number_sorting,basic_arithmetic,lcm,prime_factorization,decimal_arithmetic,"
         "fraction_simplification,time_intervals,calendar_arithmetic,knights_knaves,zebra_puzzles,mini_sudoku,"
         "shortest_path,maze,tower_of_hanoi,cryptarithm,products,power_function,bitwise_arithmetic,rotate_matrix,"
         "syllogism,count_primes,polynomial_equations,spell_backward,decimal_chain_sum").split(",")
NUDGE_MATH = ["Let me compute this with code.", "I'll verify this using Python.", "Let me work this out with a quick script.",
              "To avoid arithmetic mistakes, I'll use code.", "Let me check this programmatically."]
NUDGE_CODE = ["Let me first test my approach with code.", "I'll write the function and try it on the example.",
              "Let me check the logic with a quick test."]
FMT = re.compile(r"\s*<think>.*?</think>\s*<answer>(?P<ans>.*?)</answer>\s*", re.S)

items = []
for t in TASKS:
    ds = reasoning_gym.create_dataset(t, size=N_TASK, seed=TRAIN_SEED + 700_000)   # disjoint from RL/eval seeds
    items += [dict(kind="math", task=t, ds=ds, e=e, q=e["question"]) for e in ds]
code_rows = [json.loads(l) for l in itertools.islice(open("data/code/kodcode_train.jsonl"), 200_000, 200_000 + 3000)]
code_rows = [r for r in code_rows if r["difficulty"] in ("easy", "medium")][:N_CODE]
for r in code_rows:
    q = r["prompt"].strip() + (f"\n\nComplete this starter code:\n```python\n{r['starter'].strip()}\n```" if r["starter"] else "")
    items.append(dict(kind="code", task="code_" + r["subset"], row=r, q=q))

model, tok = load(MODEL)
prompts, prefixes, owners = [], [], []
for ii, it in enumerate(items):
    sysmsg = TOOL_SYSTEM if it["kind"] == "math" else TOOL_SYSTEM_CODE
    base = tok.apply_chat_template([{"role": "system", "content": sysmsg}, {"role": "user", "content": it["q"]}],
                                   add_generation_prompt=True, tokenize=True)
    for k in range(K):
        if k < K // 2:
            pre = "<think>\n" + random.choice(NUDGE_MATH if it["kind"] == "math" else NUDGE_CODE) + "\n<code>\n"
        else:
            pre = ""
        prompts.append(base + tok.encode(pre, add_special_tokens=False))
        prefixes.append(pre)
        owners.append(ii)
print(f"{len(items)} problems x {K} = {len(prompts)} samples", flush=True)

with Judge(workers=3) as judge:
    res = generate_with_tools(model, tok, prompts, judge, max_tokens=1024, max_calls=4,
                              sampler=make_sampler(temp=0.8, top_p=0.95), completion_batch_size=16)
    keep, stats = [], Counter()
    for s, ii in enumerate(owners):
        it, pre = items[ii], prefixes[s]
        full = pre + res["texts"][s]
        if res["truncated"][s] or res["void"][s]:
            stats["truncated/void"] += 1; continue
        if it["kind"] == "math":
            m = FMT.fullmatch(full)
            if not m or full.count("<think>") != 1 or full.count("</think>") != 1 or full.count("<answer>") != 1:
                stats["malformed"] += 1; continue
            if rg_score(it["ds"], it["e"], normalize(m.group("ans"))) < 1:
                stats["wrong"] += 1; continue
        else:
            body = re.sub(r"<think>.*?</think>", "", full, flags=re.S)
            blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", body, re.S)
            if full.count("<think>") != 1 or full.count("</think>") != 1 or len(blocks) != 1:
                stats["malformed"] += 1; continue
            r = it["row"]
            g = judge.grade_tests(blocks[0], r["test_code"], prefix=r["prefix"], timeout=8)
            if not g.get("all_pass"):
                stats["wrong"] += 1; continue
        # segments: (text, trainable). prefix is trainable; sandbox output (mask 0) is not
        segs = [[pre, True]] if pre else []
        ids, msk = res["token_ids"][s], res["masks"][s]
        for flag, grp in itertools.groupby(zip(ids, msk), key=lambda x: x[1]):
            txt = tok.decode([i for i, _ in grp])
            if segs and segs[-1][1] == bool(flag):
                segs[-1][0] += txt
            else:
                segs.append([txt, bool(flag)])
        used = res["calls"][s] > 0
        stats["kept_tool" if used else "kept_notool"] += 1
        keep.append(dict(problem=ii, kind=it["kind"], task=it["task"], tool=used,
                         system=TOOL_SYSTEM if it["kind"] == "math" else TOOL_SYSTEM_CODE, user=it["q"], segments=segs))

print("\nstats:", dict(stats), flush=True)
# at most 2 per problem, balance ~60% tool / 40% no-tool
random.shuffle(keep)
per_prob, tool, notool = Counter(), [], []
for r in keep:
    if per_prob[r["problem"]] >= 2:
        continue
    per_prob[r["problem"]] += 1
    (tool if r["tool"] else notool).append(r)
notool = notool[: int(len(tool) * 0.67) + 1]
rows = tool + notool
random.shuffle(rows)
print(f"tool {len(tool)}  no-tool {len(notool)}  tasks {len(set(r['task'] for r in rows))}")
print("per task:", dict(Counter(r["task"] for r in rows)))
os.makedirs("data/sft_tools", exist_ok=True)
nv = max(16, len(rows) // 15)
for name, part in (("valid", rows[:nv]), ("train", rows[nv:])):
    with open(f"data/sft_tools/{name}.jsonl", "w") as f:
        for r in part:
            f.write(json.dumps(r) + "\n")
print(f"train {len(rows) - nv}  valid {nv}")
