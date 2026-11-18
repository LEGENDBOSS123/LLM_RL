import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Build a large, diverse pool of human-written, auto-checkable problems -> data/general/{train,eval}.jsonl

Row: {id, source, cat, diff, question, gold, atype}
  atype: num (numeric, tolerance), expr (math-verify), mcq (letter), bool, str (short exact)
Sources: WebInstruct-verified (15 domains), NuminaMath-1.5, DeepScaleR, MATH (all subjects), SciQ, ARC-Challenge.
Excludes anything identical to MATH-500 / GSM8K test questions.
"""
import json, random, re
from collections import Counter
from datasets import load_dataset

random.seed(0)
OUT = "data/general"
banned = {json.loads(l)["problem"].strip() for l in open("data/math500_test.jsonl")}
banned |= {json.loads(l)["question"].strip() for l in open("data/gsm8k_test.jsonl")}
rows = []


def add(source, cat, diff, q, gold, atype):
    q, gold = q.strip(), str(gold).strip()
    if not q or not gold or q in banned or len(gold) > 40 or len(q) > 3000:
        return
    rows.append(dict(id=f"{source}/{len(rows)}", source=source, cat=cat, diff=diff, question=q, gold=gold, atype=atype))


# WebInstruct-verified: many domains
TYPEMAP = {"Float": "num", "Integer": "num", "Percentage": "num", "Fraction": "expr", "Expression": "expr",
           "Multiple Choice": "mcq", "Boolean": "bool", "String": "str"}
for r in load_dataset("TIGER-Lab/WebInstruct-verified", split="train"):
    at = TYPEMAP.get(r["answer_type"])
    if at is None or (at == "str" and len(r["answer"]) > 25):
        continue
    add("webinstruct", r["category"], r["difficulty"], r["question"], r["answer"], at)
print("webinstruct", len(rows))

# NuminaMath-1.5: checkable final answers only
n0 = len(rows)
nm = load_dataset("AI-MO/NuminaMath-1.5", split="train")
idx = list(range(len(nm)))
random.shuffle(idx)
for i in idx:
    r = nm[i]
    a = (r["answer"] or "").strip()
    if a.lower() in ("", "proof", "notfound", "none") or r["problem_is_valid"] != "Yes" or r["solution_is_valid"] != "Yes":
        continue
    if r["question_type"] not in ("math-word-problem",):
        continue
    add("numina", r["problem_type"] or "math", r["source"], r["problem"], a, "expr")
    if len(rows) - n0 >= 80000:
        break
print("numina", len(rows) - n0)

# DeepScaleR (hard competition math)
n0 = len(rows)
for r in load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train"):
    add("deepscaler", "competition", "hard", r["problem"], r["answer"], "expr")
print("deepscaler", len(rows) - n0)


# MATH train, all subjects (answer = last \boxed{})
def boxed(s):
    k = s.rfind("\\boxed{")
    if k < 0:
        return None
    i, depth, out = k + 7, 1, ""
    while i < len(s) and depth:
        c = s[i]
        depth += (c == "{") - (c == "}")
        if depth:
            out += c
        i += 1
    return out


n0 = len(rows)
for sub in ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]:
    for r in load_dataset("EleutherAI/hendrycks_math", sub, split="train"):
        b = boxed(r["solution"])
        if b:
            add("math", sub, r["level"], r["problem"], b, "expr")
print("math", len(rows) - n0)


# Science multiple choice
def mcq(q, options, correct_idx):
    letters = "ABCDE"[:len(options)]
    body = q + "\n" + "\n".join(f"{l}) {o}" for l, o in zip(letters, options)) + "\nAnswer with the letter of the correct option."
    return body, letters[correct_idx]


n0 = len(rows)
for r in load_dataset("allenai/sciq", split="train"):
    opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
    random.shuffle(opts)
    q, g = mcq(r["question"], opts, opts.index(r["correct_answer"]))
    add("sciq", "science", "school", q, g, "mcq")
for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train"):
    labels, texts = r["choices"]["label"], r["choices"]["text"]
    if r["answerKey"] not in labels:
        continue
    q, g = mcq(r["question"], texts, labels.index(r["answerKey"]))
    add("arc", "science", "challenge", q, g, "mcq")
print("science mcq", len(rows) - n0)

# dedupe + held-out eval (stratified by source)
seen, uniq = set(), []
for r in rows:
    k = r["question"][:300]
    if k not in seen:
        seen.add(k)
        uniq.append(r)
random.shuffle(uniq)
EVAL_N = {"webinstruct": 150, "numina": 60, "deepscaler": 30, "math": 40, "sciq": 30, "arc": 20}
ev, tr, cnt = [], [], Counter()
for r in uniq:
    if cnt[r["source"]] < EVAL_N.get(r["source"], 0):
        ev.append(r)
        cnt[r["source"]] += 1
    else:
        tr.append(r)
for name, part in (("train", tr), ("eval", ev)):
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in part:
            f.write(json.dumps(r) + "\n")
print("train", len(tr), dict(Counter(r["source"] for r in tr)))
print("eval", len(ev), dict(Counter(r["source"] for r in ev)))
print("train domains", Counter(r["cat"] for r in tr if r["source"] == "webinstruct").most_common(15))
