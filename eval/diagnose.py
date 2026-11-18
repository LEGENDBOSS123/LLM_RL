import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
import json, re, statistics, itertools
import mlx.core as mx
mx.set_memory_limit(int(6e9))
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
import reasoning_gym
from common import EVAL_SEED, normalize, rg_score
from tools import generate_with_tools
R1Z = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
       "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
       "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
       "i.e., <think> reasoning process here </think> <answer> answer here </answer>.")
SHOT = " User: What is 15 + 27 * 2? Assistant: First multiply: 27 * 2 = 54. Then add: 15 + 54 = 69. <answer>69</answer>"
import sys
# never trained in runs Z or E
UNSEEN = ["letter_counting", "family_relationships", "word_sorting", "number_sequence", "course_schedule", "circuit_logic", "word_ladder"]
K, N = 8, 15
items = []
for t in UNSEEN:
    ds = reasoning_gym.create_dataset(t, size=N, seed=EVAL_SEED + 4242)
    items += [(t, ds, e) for e in ds]
gsm = [json.loads(l) for l in itertools.islice(open("data/gsm8k_test.jsonl"), 40)]

def answer(text):
    a = re.findall(r"<answer>(.*?)</answer>", text.split("</think>")[-1], re.S)
    return a[0].strip() if a else None

def gsm_ok(text, gold):
    a = answer(text)
    if a is None: return False
    nums = re.findall(r"-?\d+(?:\.\d+)?", a.replace(",", ""))
    g = float(gold.split("####")[-1].strip().replace(",", ""))
    return bool(nums) and abs(float(nums[-1]) - g) < 1e-6

def run(path, shot, label, modes=0):
    m, tok = load(path)
    samp = make_sampler(temp=0.7, top_p=0.95)
    pre = R1Z + (SHOT if shot else "")
    P = [tok.encode(pre + f" User: {e['question']} Assistant:") for _, _, e in items]
    if modes:   # cycle the K samples of each problem through the learned mode tags
        allP = [p + tok.encode(f" [Mode {(j % modes) + 1}]", add_special_tokens=False) for p in P for j in range(K)]
    else:
        allP = [p for p in P for _ in range(K)]
    r = generate_with_tools(m, tok, allP, None, max_tokens=640, tools=False,
                            stop_answer=True, sampler=samp, completion_batch_size=16)
    per = {t: {"p1": [], "pk": [], "maj": []} for t in UNSEEN}
    fmt = []
    for i, (t, ds, e) in enumerate(items):
        outs = r["texts"][i * K:(i + 1) * K]
        ok = [(a := answer(o)) is not None and rg_score(ds, e, normalize(a)) >= 1 for o in outs]
        fmt += [answer(o) is not None for o in outs]
        per[t]["p1"].append(statistics.mean(ok)); per[t]["pk"].append(float(any(ok)))
        votes = [normalize(a) for o in outs if (a := answer(o)) is not None]
        if votes:
            top = max(set(votes), key=votes.count)
            per[t]["maj"].append(float(rg_score(ds, e, top) >= 1))
        else:
            per[t]["maj"].append(0.0)
    G = [tok.encode(pre + f" User: {g['question']} Assistant:") for g in gsm]
    rg = generate_with_tools(m, tok, G, None, max_tokens=640, tools=False, stop_answer=True, sampler=samp, completion_batch_size=16)
    gacc = statistics.mean(gsm_ok(o, g["answer"]) for o, g in zip(rg["texts"], gsm))
    p1 = statistics.mean(x for t in UNSEEN for x in per[t]["p1"]); pk = statistics.mean(x for t in UNSEEN for x in per[t]["pk"])
    mj = statistics.mean(x for t in UNSEEN for x in per[t]["maj"])
    print(f"\n{label}\n  unseen pass@1 {p1:.3f} | maj@{K} {mj:.3f} | pass@{K} {pk:.3f} | answer-tag rate {statistics.mean(fmt):.2f} | GSM8K pass@1 {gacc:.3f}")
    for t in UNSEEN:
        print(f"    {t:22} pass@1 {statistics.mean(per[t]['p1']):.2f}  pass@{K} {statistics.mean(per[t]['pk']):.2f}")
    del m

# usage: python diagnose.py [model_path ...]   (default: base 1-shot + run-Z model)
paths = sys.argv[1:]
if not paths:
    run("models/Qwen3-1.7B-Base", True, "BASE + 1-shot format example (no training)")
    run("models/Qwen3-1.7B-RL", False, "RL model (step 180), zero-shot as trained")
else:
    modes = 0
    if "--modes" in paths:
        k = paths.index("--modes"); modes = int(paths[k + 1]); paths = paths[:k] + paths[k + 2:]
    for p in paths:
        run(p, False, p + " (plain sampling)")
        if modes:
            run(p, False, p + f" (cycling {modes} learned modes)", modes)
