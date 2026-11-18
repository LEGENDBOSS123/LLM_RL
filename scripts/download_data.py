import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
import json, os, reasoning_gym
from datasets import load_dataset
from common import TASKS, EVAL_SEED, TRAIN_SEED
D = "data"

def dump(rows, path):
    with open(os.path.join(D, path), "w") as f:
        for r in rows: f.write(json.dumps(r, default=str) + "\n")
    print(f"{path:35} {len(rows):6}")

# --- external datasets
import sys
for name, cfg, split, out in [
    ("openai/gsm8k", "main", "train", "gsm8k_train.jsonl"),
    ("openai/gsm8k", "main", "test", "gsm8k_test.jsonl"),
    ("HuggingFaceH4/MATH-500", None, "test", "math500_test.jsonl"),
    ("google-research-datasets/mbpp", "sanitized", "train", "mbpp_train.jsonl"),
    ("google-research-datasets/mbpp", "sanitized", "validation", "mbpp_val.jsonl"),
    ("google-research-datasets/mbpp", "sanitized", "test", "mbpp_test.jsonl"),
    ("openai/openai_humaneval", None, "test", "humaneval_test.jsonl"),
]:
    try:
        ds = load_dataset(name, cfg, split=split)
        dump([dict(r) for r in ds], out)
    except Exception as e:
        print("FAIL", name, split, repr(e)[:150])

# --- reasoning-gym fixed sets (entry re-creatable via task+seed+size+idx)
UNSEEN = ["propositional_logic", "family_relationships", "word_sorting", "prime_factorization", "lcm", "number_sequence"]
def rg(tasks, n, seed, out):
    rows = []
    for t in tasks:
        ds = reasoning_gym.create_dataset(t, size=n, seed=seed)
        for i, e in enumerate(ds):
            rows.append(dict(task=t, seed=seed, size=n, idx=i, question=e["question"], answer=e["answer"], metadata=e["metadata"]))
    dump(rows, out)

rg(TASKS, 2000, TRAIN_SEED, "rg_train.jsonl")
# hard guarantee: drop any train question identical to an eval question
ev = set()
for t in TASKS + UNSEEN:
    ev |= {e["question"] for e in reasoning_gym.create_dataset(t, size=50, seed=EVAL_SEED)}
rows = [json.loads(l) for l in open(f"{D}/rg_train.jsonl")]
dump([r for r in rows if r["question"] not in ev], "rg_train.jsonl")
rg(TASKS, 50, EVAL_SEED, "rg_eval.jsonl")
UNSEEN_ = ["propositional_logic", "family_relationships", "word_sorting", "prime_factorization", "lcm", "number_sequence"]
rg(UNSEEN, 50, EVAL_SEED, "rg_eval_unseen.jsonl")
