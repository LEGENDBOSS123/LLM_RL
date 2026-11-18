import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Show the model's raw outputs on held-out TEST problems (eval seed) with grading.

  python sample_test.py [run_dir=runs/T] [per_task=1] [model=<4bit copy of the run's base>]
Saves everything to results/test_samples_<run>_step<N>.jsonl and prints a readable summary.
"""
import json, os, re, sys, itertools, random
import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers
import reasoning_gym
from common import TASKS, EVAL_SEED, normalize, rg_score, strip_think
from tools import generate_with_tools, TOOL_SYSTEM
from sandbox.judge import Judge

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/T"
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 1
cfg = json.load(open(f"{RUN}/config.json"))
MODEL = sys.argv[3] if len(sys.argv) > 3 else cfg["model"] + "-4bit"
mx.set_memory_limit(int(2e9))
random.seed(7)
CODE_SYSTEM = ("You are an expert Python programmer. Think step by step inside <think>...</think>, "
               "then give your complete solution in a single ```python code block.")

model, tok = load(MODEL)
model.freeze()
linear_to_lora_layers(model, len(model.layers), {"rank": cfg["rank"], "scale": 2.0, "dropout": 0.0,
                      "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]})
model.load_weights(f"{RUN}/adapters.safetensors", strict=False)
step = json.load(open(f"{RUN}/state.json"))["step"]
train_tasks = cfg.get("tasks", "").split(",")

items = []
for t in TASKS:
    ds = reasoning_gym.create_dataset(t, size=50, seed=EVAL_SEED)
    for i in random.sample(range(50), PER):
        items.append(dict(kind="math", task=t, seen=t in train_tasks, ds=ds, e=ds[i], q=ds[i]["question"]))
for l in itertools.islice(open("data/code/kodcode_eval.jsonl"), 0, 2 * PER):
    r = json.loads(l)
    q = r["prompt"].strip() + (f"\n\nComplete this starter code:\n```python\n{r['starter'].strip()}\n```" if r["starter"] else "")
    items.append(dict(kind="code", task=f"code ({r['difficulty']})", seen=None, row=r, q=q))

prompts = [tok.apply_chat_template([{"role": "system", "content": TOOL_SYSTEM if it["kind"] == "math" else CODE_SYSTEM},
                                    {"role": "user", "content": it["q"]}], add_generation_prompt=True, tokenize=True)
           for it in items]
with Judge(workers=2) as j:
    res = generate_with_tools(model, tok, prompts, j, max_tokens=1024, sampler=make_sampler(temp=0.7, top_p=0.95),
                              completion_batch_size=8)
    out = []
    for it, text, calls, trunc in zip(items, res["texts"], res["calls"], res["truncated"]):
        if it["kind"] == "math":
            m = re.findall(r"<answer>(.*?)</answer>", strip_think(text), re.S)
            ans = normalize(m[-1]) if m else None
            ok = ans is not None and rg_score(it["ds"], it["e"], ans) >= 1
            gold = str(it["e"]["answer"])
        else:
            body = strip_think(text)
            blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", body, re.S)
            if not blocks:
                a = re.findall(r"<answer>(.*?)</answer>", body, re.S)
                blocks = [re.sub(r"^\s*```(?:python|py)?\s*\n|```\s*$", "", a[0].strip())] if len(a) == 1 else []
            r = it["row"]
            g = j.grade_tests(blocks[0], r["test_code"], prefix=r["prefix"], timeout=8) if len(blocks) == 1 else None
            ok = bool(g and g.get("all_pass"))
            ans = f"{g['passed']}/{g['total']} tests" if g else "(no single code block)"
            gold = "(hidden unit tests)"
        out.append(dict(task=it["task"], seen=it["seen"], question=it["q"], gold=gold, extracted=ans,
                        correct=ok, tool_calls=calls, truncated=trunc, output=text))

os.makedirs("results", exist_ok=True)
path = f"results/test_samples_{os.path.basename(RUN)}_step{step}.jsonl"
with open(path, "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(f"step {step} | {sum(o['correct'] for o in out)}/{len(out)} correct | saved {path}\n")
for o in out:
    tag = "TRAINED" if o["seen"] else ("UNSEEN" if o["seen"] is False else "CODE")
    print(f"=== [{tag}] {o['task']} | {'CORRECT' if o['correct'] else 'WRONG'} | gold: {o['gold'][:60]} | got: {str(o['extracted'])[:60]} | tool calls {o['tool_calls']}")
    print("Q:", o["question"][:300].replace("\n", " "))
    print(o["output"][:1200])
    print()
