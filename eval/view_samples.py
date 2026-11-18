import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
import json, re, random, itertools
import mlx.core as mx
mx.set_memory_limit(int(5e9))
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
import reasoning_gym
from common import EVAL_SEED, normalize, rg_score
from tools import generate_with_tools
from sandbox.judge import Judge
R1Z = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
       "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
       "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
       "i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {q} Assistant:")
random.seed(3)
items = []
for t, tag in [("basic_arithmetic", "TRAINED"), ("base_conversion", "TRAINED"), ("countdown", "UNSEEN"), ("simple_equations", "UNSEEN"),
               ("letter_counting", "UNSEEN"), ("knights_knaves", "TRAINED"), ("zebra_puzzles", "TRAINED"), ("family_relationships", "UNSEEN")]:
    ds = reasoning_gym.create_dataset(t, size=50, seed=EVAL_SEED)
    e = ds[random.randrange(50)]
    items.append(dict(tag=tag, task=t, q=e["question"], ds=ds, e=e))
r = json.loads(next(itertools.islice(open("data/code/kodcode_eval.jsonl"), 5, 6)))
items.append(dict(tag="CODE", task=f"code ({r['difficulty']})", q=r["prompt"].strip() + "\nThe answer must be the complete Python code (no explanation) inside the answer tags.", row=r))
items.append(dict(tag="GENERAL", task="word problem", q="Sarah has 3 times as many apples as Tom. Together they have 48 apples. Sarah gives Tom 6 apples. How many apples does Tom have now?", gold="18"))
items.append(dict(tag="GENERAL", task="logic", q="If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Answer yes or no.", gold="yes"))
m, tok = load("models/Qwen3-1.7B-RL")
prompts = [tok.encode(R1Z.format(q=it["q"])) for it in items]
with Judge(workers=2) as j:
    res = generate_with_tools(m, tok, prompts, j, max_tokens=640, tools=False, stop_answer=True,
                              sampler=make_sampler(temp=0.7, top_p=0.95), completion_batch_size=12)
    for it, out in zip(items, res["texts"]):
        a = re.findall(r"<answer>(.*?)</answer>", out.split("</think>")[-1], re.S)
        ans = a[0].strip() if a else None
        if "e" in it:
            ok, gold = (ans is not None and rg_score(it["ds"], it["e"], normalize(ans)) >= 1), str(it["e"]["answer"])
        elif "row" in it:
            g = j.grade_tests(re.sub(r"^\s*```(?:python|py)?\s*\n|```\s*$", "", ans or ""), it["row"]["test_code"], prefix=it["row"]["prefix"], timeout=8) if ans else None
            ok, gold = bool(g and g.get("all_pass")), f"hidden tests ({g['passed']}/{g['total']} passed)" if g else "hidden tests"
        else:
            ok, gold = (ans is not None and ans.strip().lower().rstrip(".") == it["gold"]), it["gold"]
        print("=" * 100)
        print(f"[{it['tag']}] {it['task']}  ->  {'CORRECT' if ok else 'WRONG'}   (gold: {gold[:80]})")
        print("QUESTION:", it["q"][:700])
        print("MODEL OUTPUT (raw):")
        print(out)
