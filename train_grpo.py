"""GRPO (Dr. GRPO variant) with LoRA on MLX.  Math/logic (reasoning-gym) + code (sandbox).

  python train_grpo.py --run runs/A --steps 1000            # condition A, RL only
  python train_grpo.py --run runs/A --resume                 # continue a run

Per step: sample P prompts by curriculum -> G rollouts each (temp 1.0) -> rewards ->
advantage = r - mean(group) (no std division) -> drop zero-variance groups ->
loss = -sum(A * logp) / (num_seqs * MAX_TOKENS) + beta * KL(policy || ref)   (token-level, k3)
Logs every step to runs/X/log.jsonl; samples, evals, adapters periodically.
"""
import argparse, itertools, json, math, os, random, re, statistics, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import grad_checkpoint

import reasoning_gym
from common import SYSTEM, TASKS, TRAIN_SEED, EVAL_SEED, strip_think, normalize, rg_score

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/A")
ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct-bf16")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--prompts", type=int, default=8)          # P
ap.add_argument("--group", type=int, default=8)            # G
ap.add_argument("--max_tokens", type=int, default=1024)
ap.add_argument("--lr", type=float, default=3e-6)
ap.add_argument("--beta", type=float, default=0.02)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--micro", type=int, default=2)            # sequences per backward pass
ap.add_argument("--code_frac", type=float, default=0.5)
ap.add_argument("--len_cost", type=float, default=0.0002)  # per think-token cost (-0.1 at 500 tokens)
ap.add_argument("--format_bonus", type=float, default=0.1)
ap.add_argument("--len_alpha", type=float, default=0.0, help="group-relative length shaping weight (Kimi k1.5 style)")
ap.add_argument("--len_gate", type=float, default=0.5, help="only shape length in groups with accuracy >= this (GRPO-lambda)")
ap.add_argument("--nothink_eval", action="store_true", help="also eval with forced empty think (CoT usefulness)")
ap.add_argument("--tools", action="store_true", help="model may run Python any time via <code>...</code>")
ap.add_argument("--max_calls", type=int, default=4)
ap.add_argument("--backtrack", action="store_true", help="tools: <checkpoint> / <backtrack>note</backtrack> context rewind")
ap.add_argument("--max_backtracks", type=int, default=3)
ap.add_argument("--tool_cost", type=float, default=0.0, help="reward cost per code call (correct answers only)")
ap.add_argument("--code_tools", action="store_true", help="also offer the tool on coding problems")
ap.add_argument("--explore_frac", type=float, default=0.0, help="fraction of rollouts that must keep reasoning after tool output")
ap.add_argument("--explore_k", type=int, default=24, help="tokens after a tool output during which closing is banned")
ap.add_argument("--temp", type=float, default=1.0, help="rollout sampling temperature (Skywork-OR1/POLARIS: >=1.0)")
ap.add_argument("--eval_only", default="", help="path to an adapters .safetensors: load it, run one eval, exit")
ap.add_argument("--eval_tag", type=int, default=-1, help="step number to record for --eval_only")
ap.add_argument("--gen_frac", type=float, default=0.0, help="fraction of prompts from the human-written general pool (data/general)")
ap.add_argument("--eval_gen_n", type=int, default=0, help="general-pool eval items (stratified, fixed)")
ap.add_argument("--max_entropy", type=float, default=0.0, help="guard: abort if batch entropy exceeds this for 2 consecutive steps")
ap.add_argument("--wreinforce", type=float, default=0.0,
                help="W-REINFORCE (Zhu et al. 2025): advantage = +lambda if correct else -1, no baseline, all groups kept")
ap.add_argument("--open_tree", type=int, default=0,
                help="tree over openings: per problem sample N openings (first sentence, <= --open_len tokens), "
                     "group/N continuations each; opening tokens get V(opening)-V(problem), rest get r-V(opening)")
ap.add_argument("--open_len", type=int, default=48)
ap.add_argument("--open_max_v", type=float, default=1.0,
                help="apply opening credit only to problems with group accuracy <= this (stuck problems); "
                     "others get plain r - mean advantages")
ap.add_argument("--progress", type=int, default=0,
                help="Monte-Carlo progress credit: branch up to N traces/step, cut reasoning at --pr_cuts, "
                     "--pr_conts continuations per cut -> V(prefix); token advantage = V(next cut) - V(this cut)")
ap.add_argument("--pr_cuts", default="0.33,0.67")
ap.add_argument("--pr_conts", type=int, default=4)
ap.add_argument("--pr_train_conts", type=int, default=1, help="also train on continuations (adv = r - V(cut))")
ap.add_argument("--passk", type=int, default=0,
                help="Pass@k training (Chen et al. 2025, 2508.10751): analytic group advantage of the pass@k reward, "
                     "A_pos=1-Rbar, A_neg=1-Rbar-C(n_neg-1,k-1)/C(n-1,k-1); wrong answers barely penalised -> keeps exploration")
ap.add_argument("--modes", type=int, default=0, help="K learned thinking modes: prompt tag ' [Mode z]' after 'Assistant:'")
ap.add_argument("--mode_gamma", type=float, default=0.5, help="ensemble bonus: correct where the OTHER modes are wrong")
ap.add_argument("--mode_delta", type=float, default=0.1, help="bonus for a wrong answer that differs from other modes' wrong answers")
ap.add_argument("--minibatches", type=int, default=1, help="DAPO/PPO: optimizer steps per rollout batch (ratio-clipped)")
ap.add_argument("--clip_low", type=float, default=0.2)
ap.add_argument("--clip_high", type=float, default=0.28, help="clip-higher (DAPO)")
ap.add_argument("--max_tokens_cap", type=int, default=0, help="staged budget: raise max_tokens up to this cap")
ap.add_argument("--stage_trunc", type=float, default=0.08, help="raise budget when mean truncation over 20 steps exceeds this")
ap.add_argument("--stage_min_steps", type=int, default=40, help="min steps between budget raises")
ap.add_argument("--div_alpha", type=float, default=0.0, help="bonus for CORRECT rollouts that are semantically unusual within their group")
ap.add_argument("--plan_sem", action="store_true", help="plan diversity by embedding distance (not word overlap)")
ap.add_argument("--ent_target", type=float, default=0.0, help="adaptive entropy floor: if entropy EMA < target, add entropy bonus")
ap.add_argument("--ent_coef", type=float, default=0.01)
ap.add_argument("--hard_len_alpha", type=float, default=0.0, help="among CORRECT answers in low-accuracy groups, prefer shorter (insight)")
ap.add_argument("--self_seed", action="store_true", help="model proposes its own strategy plans; closed-loop exploration")
ap.add_argument("--plans", type=int, default=4, help="self-proposed plans per problem")
ap.add_argument("--plan_frac", type=float, default=0.5, help="initial fraction of rollouts conditioned on a plan")
ap.add_argument("--plan_frac_end", type=float, default=0.1)
ap.add_argument("--plan_fade_steps", type=int, default=300)
ap.add_argument("--plan_max", type=int, default=48)
ap.add_argument("--plan_temp", type=float, default=1.2)
ap.add_argument("--plan_weight", type=float, default=1.0, help="loss weight of the plan-proposer objective")
ap.add_argument("--plan_div", type=float, default=0.1, help="bonus for plans that differ from the other plans of the same problem")
ap.add_argument("--lte", action="store_true", help="when all rollouts fail, retry conditioned on the model's own failed answer")
ap.add_argument("--lte_n", type=int, default=4)
ap.add_argument("--is_clip", type=float, default=2.0, help="clip for importance weights of conditioned rollouts")
ap.add_argument("--last_answer", action="store_true", help="grade the LAST <answer>; don't stop at the first one (self-correction allowed)")
ap.add_argument("--algo", default="grpo", choices=["grpo", "ppo"], help="ppo = learned critic (LoRA critic + value head)")
ap.add_argument("--critic_lr", type=float, default=5e-5)
ap.add_argument("--critic_warmup", type=int, default=10, help="PPO: steps of value pre-training with no policy update")
ap.add_argument("--gae_alpha", type=float, default=0.05, help="PPO: length-adaptive policy lambda = 1 - 1/(alpha*L) (VAPO)")
ap.add_argument("--ppo_len_cost", type=float, default=1e-4, help="PPO reward = correct - c * model_tokens")
ap.add_argument("--temp_ladder", default="", help="'lo,hi': each group's samples get temperatures spread over [lo,hi]")
ap.add_argument("--top_p", type=float, default=1.0)
ap.add_argument("--template", default="", help="'r1zero': plain-text R1-Zero prompt for BASE models (no chat template)")
ap.add_argument("--overlong_buffer", type=int, default=0, help="DAPO soft overlong penalty: last N tokens before the limit")
ap.add_argument("--overlong_penalty", type=float, default=0.2, help="max penalty reached at the limit")
ap.add_argument("--judge_workers", type=int, default=3)
ap.add_argument("--eval_every", type=int, default=100)
ap.add_argument("--save_every", type=int, default=25)
ap.add_argument("--sample_every", type=int, default=25)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--skip_eval0", action="store_true")
ap.add_argument("--stratified", action="store_true", help="each step contains every train task at least once")
ap.add_argument("--no_system", action="store_true", help="math prompts without the think system prompt (post-SFT models)")
ap.add_argument("--mem_limit_gb", type=float, default=7.0, help="hard cap on MLX memory (keeps macOS responsive)")
ap.add_argument("--cache_limit_gb", type=float, default=1.0)
ap.add_argument("--gen_batch", type=int, default=16, help="max sequences decoded concurrently")
ap.add_argument("--tasks", default=",".join(TASKS), help="comma-separated reasoning-gym tasks to train on")
ap.add_argument("--eval_math_n", type=int, default=16, help="eval problems per task (all 8 tasks)")
ap.add_argument("--eval_code_n", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
TRAIN_TASKS = [t for t in args.tasks.split(",") if t]
FORMAT_BONUS = args.format_bonus
os.makedirs(args.run, exist_ok=True)
mx.set_memory_limit(int(args.mem_limit_gb * 1e9))
mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
random.seed(args.seed)
mx.random.seed(args.seed)

CODE_SYSTEM = ("You are an expert Python programmer. Think step by step inside <think>...</think>, "
               "then give your complete solution in a single ```python code block.")

# ------------------------------------------------------------------ behaviours (logged, never rewarded)
BEHAVIOURS = {
    "alternatives": re.compile(r"\b(alternatively|another (approach|way|method|idea)|other (approach|option)|"
                               r"option \d|approach \d|we could also|or we could|instead,? (we|let's|i))\b", re.I),
    "verification": re.compile(r"\b(let me (check|verify|double[- ]?check|test)|let's (check|verify)|verify|"
                               r"double[- ]check|sanity check|checking)\b", re.I),
    "backtracking": re.compile(r"\b(wait|hmm|actually,|that's (wrong|incorrect|not right)|mistake|"
                               r"let me (redo|rethink|reconsider|try again)|i was wrong)\b", re.I),
    "subgoals": re.compile(r"\b(first,|step \d|next,|then,|finally,|sub-?problem|break (it|this) down)\b", re.I),
}


def post_tool_stats(texts):
    """words of reasoning after the LAST tool output (inside think), and retry-after-error rate."""
    words, errs, retries = [], 0, 0
    for o in texts:
        if "</output>" not in o:
            continue
        tail = o[o.rfind("</output>") + len("</output>"):]
        words.append(len(tail.split("</think>")[0].split()))
        outs = re.findall(r"<output>(.*?)</output>", o, re.S)
        for k, out in enumerate(outs):
            if "[error]" in out or "[timeout]" in out or "Error" in out:
                errs += 1
                retries += k + 1 < len(outs)          # another code block came after the error
    return (round(statistics.mean(words), 1) if words else None), (round(retries / errs, 3) if errs else None)


def think_text(t):
    return " ".join(re.findall(r"<think>(.*?)(?:</think>|$)", t, re.S))


# ------------------------------------------------------------------ data
class MathSource:
    def __init__(self):
        self.ds, self.idx = {}, defaultdict(list)
        for l in open("data/rg_train.jsonl"):
            r = json.loads(l)
            self.idx[r["task"]].append(r["idx"])
        for t in TRAIN_TASKS:
            base, _, spec = t.partition("@")          # "n_queens@n=4;max_remove=2" -> easier curriculum variant
            cfg = {k: int(v) if v.lstrip("-").isdigit() else v for k, v in (kv.split("=") for kv in spec.split(";") if kv)}
            self.ds[t] = reasoning_gym.create_dataset(base, size=2000, seed=TRAIN_SEED, **cfg)
            if t not in self.idx:                       # task not in rg_train.jsonl (never evaluated -> no leak)
                self.idx[t] = list(range(2000))

    def sample(self, task):
        i = random.choice(self.idx[task])
        e = self.ds[task][i]
        return dict(bucket=task, kind="math", question=e["question"], entry=e, ds=self.ds[task])


class GenSource:
    """Human-written problems (WebInstruct 15 domains, NuminaMath, DeepScaleR, MATH, SciQ, ARC). Bucketed by source/domain."""
    def __init__(self, path="data/general/train.jsonl"):
        self.b = defaultdict(list)
        for l in open(path):
            r = json.loads(l)
            key = f"gen_{r['source']}_{r['cat']}" if r["source"] == "webinstruct" else f"gen_{r['source']}"
            self.b[re.sub(r"[^A-Za-z0-9_]", "", key.replace(" ", "_"))].append(r)
        print("general buckets:", {k: len(v) for k, v in sorted(self.b.items())}, flush=True)

    def sample(self, bucket):
        r = random.choice(self.b[bucket])
        return dict(bucket=bucket, kind="gen", question=r["question"], row=r)


import logging as _logging
for _n in ("math_verify", "math_verify.parser", "math_verify.grader"):
    _logging.getLogger(_n).setLevel(_logging.ERROR)
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def grade_gen(row, pred):
    pred = pred.strip()
    if not pred or len(pred) > 160:
        return False
    gold, at = row["gold"], row["atype"]
    try:
        if at == "mcq":
            m = re.search(r"\b([A-E])\b", pred.upper())
            return bool(m) and m.group(1) == gold.upper()
        if at == "bool":
            f = lambda x: {"yes": 1, "true": 1, "no": 0, "false": 0}.get(re.sub(r"[^a-z]", "", x.lower()), -1)
            return f(pred) == f(gold) != -1
        if at == "str":
            n = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
            return n(pred) == n(gold) or (len(n(gold)) > 3 and n(gold) in n(pred))
        if at == "num":
            g, p = _NUM.findall(gold.replace(",", "")), _NUM.findall(pred.replace(",", ""))
            if g and p:
                gv, pv = float(g[0]), float(p[0])
                if abs(gv - pv) <= max(1e-6, 1e-2 * abs(gv)):
                    return True
        from math_verify import parse, verify
        return bool(verify(parse("$" + gold + "$", parsing_timeout=None), parse("$" + pred + "$", parsing_timeout=None),
                           timeout_seconds=None))
    except Exception:
        return False


def gen_reward(item, text, truncated, p_task=0.5, think_tokens=0):
    if truncated and not (args.last_answer and "</answer>" in text):
        return 0.0, "truncated", False
    tags = re.findall(r"<answer>(.*?)</answer>", text.split("</think>")[-1], re.S)
    if not tags or (len(tags) > 1 and not args.last_answer):
        return 0.0, "no_answer", False
    ok = grade_gen(item["row"], tags[-1])
    return (1.0 + (1.0 - p_task)) if ok else 0.0, ("ok" if ok else "wrong") + ("_rev" if len(tags) > 1 else ""), ok


def any_reward(judge, item, text, truncated, p_task=0.5, think_tokens=0):
    if item["kind"] == "math":
        return math_reward(item, text, truncated, p_task, think_tokens)
    if item["kind"] == "gen":
        return gen_reward(item, text, truncated, p_task, think_tokens)
    return code_reward(judge, item, text, truncated)


class CodeSource:
    """Reservoir-sample a manageable pool from kodcode/leetcode/taco training files."""
    def __init__(self, pool=30000):
        self.b = defaultdict(list)
        rng = random.Random(1)
        for path, keep in [("data/code/kodcode_train.jsonl", pool), ("data/code/leetcode_train.jsonl", 10**9),
                           ("data/code/taco_train.jsonl", 10**9)]:
            res = []
            for n, l in enumerate(open(path)):
                if len(res) < keep:
                    res.append(l)
                else:
                    j = rng.randint(0, n)
                    if j < keep:
                        res[j] = l
            for l in res:
                r = json.loads(l)
                if r["mode"] == "stdio" and not r["io"]:
                    continue
                self.b[f"code_{r['source']}_{r['difficulty']}"].append(r)
        print("code buckets:", {k: len(v) for k, v in sorted(self.b.items())}, flush=True)

    def sample(self, bucket):
        r = random.choice(self.b[bucket])
        return dict(bucket=bucket, kind="code", question=self.msg(r), row=r)

    @staticmethod
    def msg(r):
        s = r["prompt"].strip()
        if r["starter"]:
            s += f"\n\nComplete this starter code:\n```python\n{r['starter'].strip()}\n```"
        return s


class Curriculum:
    """Per-bucket EMA pass rate; sample buckets with weight p(1-p) (+ floor) -> focus on 'learnable'."""
    def __init__(self, buckets, state=None):
        self.p = state or {b: 0.3 for b in buckets}
        for b in buckets:
            self.p.setdefault(b, 0.3)

    def weights(self, buckets):
        return [max(self.p[b] * (1 - self.p[b]), 0.02) for b in buckets]

    def pick(self, buckets):
        return random.choices(buckets, weights=self.weights(buckets))[0]

    def update(self, b, rate, a=0.1):
        self.p[b] = (1 - a) * self.p[b] + a * rate


# ------------------------------------------------------------------ rewards
FORMAT_RE = re.compile(r"\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<ans>.*?)</answer>\s*", re.S)


def format_ok(text):
    """Exactly one closed think block, then exactly one closed answer block, nothing else."""
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return None
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None
    return FORMAT_RE.fullmatch(text)


def math_reward(item, text, truncated, p_task=0.5, think_tokens=0):
    """Returns (reward, why, correct).
    0 if truncated or malformed; else 0.1 format bonus + score * (1 + (1 - p_task)) - len_cost * think_tokens,
    floored at 0. p_task = running pass rate of this task (harder task -> correct answer worth more)."""
    if truncated and not (args.last_answer and "</answer>" in text):
        return 0.0, "truncated", False
    m = format_ok(text)
    if args.last_answer:
        tags = re.findall(r"<answer>(.*?)</answer>", text.split("</think>")[-1], re.S)
        if not tags:
            return 0.0, "no_answer", False
        s = rg_score(item["ds"], item["entry"], normalize(tags[-1]))
        return max(s * (1.0 + (1.0 - p_task)), 0.0), ("ok" if s >= 1 else "wrong") + ("_rev" if len(tags) > 1 else ""), s >= 1
    if args.template and not m:
        # R1-Zero style: correctness needs exactly one <answer>; the 0.1 format bonus needs the full structure
        after = text.split("</think>")[-1]          # unclosed <think> must not swallow the answer
        tags = re.findall(r"<answer>(.*?)</answer>", after, re.S)
        if len(tags) != 1:
            return 0.0, "no_answer", False
        s = rg_score(item["ds"], item["entry"], normalize(tags[0]))
        return max(s * (1.0 + (1.0 - p_task)), 0.0), "ok_noformat" if s >= 1 else "wrong_noformat", s >= 1
    if not m:
        return 0.0, "malformed", False
    s = rg_score(item["ds"], item["entry"], normalize(m.group("ans")))
    r = FORMAT_BONUS + s * (1.0 + (1.0 - p_task)) - args.len_cost * think_tokens
    return max(r, 0.0), "ok" if s >= 1 else "wrong", s >= 1


def code_reward(judge, item, text, truncated):
    if truncated and not (args.last_answer and ("</answer>" in text or text.count("```") >= 2)):
        return 0.0, "truncated", False
    body = strip_think(text)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", body, re.S)
    if not blocks:
        # also accept the solution inside a single <answer>...</answer> (fences stripped)
        ans = re.findall(r"<answer>(.*?)</answer>", body, re.S)
        if len(ans) == 1 and ans[0].strip():
            blocks = [re.sub(r"^\s*```(?:python|py)?\s*\n|```\s*$", "", ans[0].strip())]
    if args.last_answer and len(blocks) > 1:
        blocks = blocks[-1:]
    if len(blocks) != 1:
        return 0.0, f"code_blocks={len(blocks)}", False
    r = item["row"]
    if r["mode"] == "tests":
        g = judge.grade_tests(blocks[0], r["test_code"], prefix=r["prefix"], timeout=8)
        ok = bool(g.get("all_pass"))
        why = "ok" if ok else ",".join(sorted(set(map(str, g["verdicts"]))))[:60]
    else:
        g = judge.grade(blocks[0], r["io"][:15], timeout=4, partial=False)
        ok = g["score"] == 1.0
        why = "ok" if ok else ",".join(sorted(set(g["verdicts"])))[:60]
    clean = text.count("<think>") == text.count("</think>") >= 1
    return ((1.0 + (0.1 if clean else 0.0)) if ok else 0.0), why, ok


# ------------------------------------------------------------------ model
model, tok = load(args.model)
model.freeze()
linear_to_lora_layers(model, len(model.layers), {"rank": args.rank, "scale": 2.0, "dropout": 0.0,
                      "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]})
grad_checkpoint(model.layers[0])
from mlx_lm.tuner.lora import LoRALinear
LORA_MODULES = [m for _, m in model.named_modules() if isinstance(m, LoRALinear)]
LORA_SCALE = LORA_MODULES[0].scale


def set_lora(on: bool):
    """Reference policy = same weights with LoRA switched off (saves a 3 GB model copy)."""
    for m in LORA_MODULES:
        m.scale = LORA_SCALE if on else 0.0
if args.algo == "ppo":
    import math as _math
    # Critic = SAME frozen base + its own LoRA (lora_a_c/lora_b_c) + scalar value head.
    for m in LORA_MODULES:
        d_in, r = m.lora_a.shape
        m.lora_a_c = mx.random.uniform(low=-1 / _math.sqrt(d_in), high=1 / _math.sqrt(d_in), shape=(d_in, r))
        m.lora_b_c = mx.zeros(m.lora_b.shape)
        m.mode = "policy"

    def _dual_call(self, x):
        y = self.linear(x)
        if getattr(self, "mode", "policy") == "critic":
            a, b = self.lora_a_c, self.lora_b_c
        else:
            a, b = self.lora_a, self.lora_b
        return y + (self.scale * ((x @ a) @ b)).astype(x.dtype)
    LoRALinear.__call__ = _dual_call
    hidden = model.args.hidden_size
    model.value_head = nn.Linear(hidden, 1)
    model.value_head.weight = mx.zeros((1, hidden))
    model.value_head.bias = mx.array([-1.7])        # sigmoid(-1.7) ~ 0.15 = base model's rough P(correct)


def set_mode(mode):
    for m in LORA_MODULES:
        m.mode = mode


def is_critic_key(k):
    return "lora_a_c" in k or "lora_b_c" in k or k.startswith("value_head")


def split_grads(g):
    from mlx.utils import tree_unflatten as _tu
    flat = tree_flatten(g)
    return (_tu([(k, v) for k, v in flat if not is_critic_key(k)]),
            _tu([(k, v) for k, v in flat if is_critic_key(k)]))


n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
print(f"trainable params: {n_train/1e6:.1f}M", flush=True)
opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
copt = optim.AdamW(learning_rate=args.critic_lr, weight_decay=0.0) if args.algo == "ppo" else None

state_path = f"{args.run}/state.json"
start_step, cur_state = 0, None
if args.resume and os.path.exists(state_path):
    st = json.load(open(state_path))
    model.load_weights(f"{args.run}/adapters.safetensors", strict=False)
    if os.path.exists(f"{args.run}/optimizer.safetensors"):
        from mlx.utils import tree_unflatten
        opt.state = tree_unflatten(list(mx.load(f"{args.run}/optimizer.safetensors").items()))
        opt.learning_rate = args.lr          # saved state carries the old LR; the CLI value wins
        print("restored optimizer state", flush=True)
        if copt is not None and os.path.exists(f"{args.run}/optimizer_critic.safetensors"):
            copt.state = tree_unflatten(list(mx.load(f"{args.run}/optimizer_critic.safetensors").items()))
            copt.learning_rate = args.critic_lr
            print("restored critic optimizer state", flush=True)
    else:
        print("WARNING: no optimizer state found - resuming with fresh Adam moments", flush=True)
    start_step, cur_state = st["step"], st["curriculum"]
    if st.get("max_tokens") and args.max_tokens_cap:
        args.max_tokens = max(args.max_tokens, st["max_tokens"])
    print(f"resumed at step {start_step}", flush=True)
json.dump(vars(args), open(f"{args.run}/config.json", "w"), indent=2)


def seq_logps(m, ids, plen):
    """ids: [B, T] padded; returns per-token logp of tokens[1:] -> [B, T-1]."""
    logits = m(ids[:, :-1]).astype(mx.float32)
    return -nn.losses.cross_entropy(logits, ids[:, 1:], reduction="none")


def loss_fn(m, ids, mask, adv, ref_lp, norm, beh=None, cond=None, ent_c=0.0):
    logits = m(ids[:, :-1]).astype(mx.float32)
    lp = -nn.losses.cross_entropy(logits, ids[:, 1:], reduction="none")
    a = adv if adv.ndim == 2 else adv[:, None]
    if beh is not None:
        # conditioned rollouts were sampled from pi(.|x, plan) but are trained under pi(.|x):
        # token-level importance weight pi(y|x)/pi(y|x,plan), clipped and stop-gradient (Guide-style)
        w = mx.stop_gradient(mx.clip(mx.exp(lp - beh), 0.0, args.is_clip))
        a = a * mx.where(cond[:, None] > 0, w, mx.ones_like(w))
    pg = -(a * lp * mask).sum()
    if ent_c > 0:
        lsm = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        ent = -(mx.exp(lsm) * lsm).sum(-1)
        pg = pg - ent_c * (ent * mask).sum()
    d = ref_lp - lp
    kl = ((mx.exp(d) - d - 1) * mask).sum()
    return (pg + args.beta * kl) / norm, kl / mx.maximum(mask.sum(), 1)


loss_and_grad = nn.value_and_grad(model, loss_fn)


def ppo_loss_fn(m, ids, mask, adv, old, norm, ref=None):
    """DAPO/PPO token-level clipped surrogate with clip-higher (+ optional k3 KL to the reference policy).
    Returns (loss, (clip_fraction, kl_per_token))."""
    lp = seq_logps(m, ids, None)
    ratio = mx.exp(lp - old)
    a = adv if adv.ndim == 2 else adv[:, None]
    obj = mx.minimum(ratio * a, mx.clip(ratio, 1 - args.clip_low, 1 + args.clip_high) * a)
    clipped = ((ratio < 1 - args.clip_low) | (ratio > 1 + args.clip_high)).astype(mx.float32)
    loss = -(obj * mask).sum()
    kl = mx.array(0.0)
    if ref is not None:
        d = ref - lp
        klt = (mx.exp(d) - d - 1) * mask
        loss = loss + args.beta * klt.sum()
        kl = klt.sum() / mx.maximum(mask.sum(), 1)
    return loss / norm, ((clipped * mask).sum() / mx.maximum(mask.sum(), 1), kl)


ppo_loss_and_grad = nn.value_and_grad(model, ppo_loss_fn)


def embed(seqs, starts, micro=4):
    """L2-normalised mean last-layer hidden state over tokens [start:] of each sequence (no grad)."""
    out = []
    for k in range(0, len(seqs), micro):
        bs, st = seqs[k:k + micro], starts[k:k + micro]
        T = max(len(x) for x in bs)
        ids = mx.array([x + [0] * (T - len(x)) for x in bs])
        h = mx.stop_gradient(model.model(ids)).astype(mx.float32)
        m = mx.array([[1.0 if s_ <= j < len(x) else 0.0 for j in range(T)] for x, s_ in zip(bs, st)])
        e = (h * m[:, :, None]).sum(1) / mx.maximum(m.sum(1, keepdims=True), 1)
        e = e / mx.maximum(mx.linalg.norm(e, axis=-1, keepdims=True), 1e-6)
        mx.eval(e)
        out += e.tolist()
        del h
    mx.clear_cache()
    return out


def novelty(vecs):
    """1 - mean cosine to the others (vectors already normalised)."""
    n = len(vecs)
    res = []
    for i in range(n):
        sims = [sum(a * b for a, b in zip(vecs[i], vecs[j])) for j in range(n) if j != i]
        res.append(1 - (sum(sims) / len(sims) if sims else 1.0))
    return res


ENT_EMA = [None]
HIGH_ENT = [0]
TRUNC_HIST, STAGE = [], [0]


def critic_logits(m, ids):
    """Critic logit of P(correct | prefix up to position j) -> [B, T-1] (float32)."""
    set_mode("critic")
    h = m.model(ids[:, :-1]).astype(mx.float32)
    set_mode("policy")
    return m.value_head(h).squeeze(-1)


def critic_values(m, ids):
    return mx.sigmoid(critic_logits(m, ids))


def critic_loss_fn(m, ids, mask, target):
    """BCE on correct (0/1): the critic only ever sees correctness, never the length term."""
    z = critic_logits(m, ids)
    bce = mx.logaddexp(0.0, z) - target * z          # = -[t log s(z) + (1-t) log(1 - s(z))]
    return (bce * mask).sum() / mx.maximum(mask.sum(), 1)


critic_loss_and_grad = nn.value_and_grad(model, critic_loss_fn) if args.algo == "ppo" else None


def pad_batch(seqs, plens, tmasks=None):
    """mask[i, j] = 1 if token j+1 is a MODEL-generated completion token (tool output and prompt excluded)."""
    T = max(len(s) for s in seqs)
    ids = mx.array([s + [tok.pad_token_id or 0] * (T - len(s)) for s in seqs])
    rows = []
    for i, s in enumerate(seqs):
        tm = tmasks[i] if tmasks is not None else [1] * (len(s) - plens[i])
        row = [0.0] * (T - 1)
        for k, m in enumerate(tm):
            j = plens[i] - 1 + k
            if j < T - 1:
                row[j] = float(m)
        rows.append(row)
    return ids, mx.array(rows)


from types import SimpleNamespace
from tools import generate_with_tools, TOOL_SYSTEM, TOOL_SYSTEM_CODE


R1ZERO = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
          "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
          "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
          "i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {q} Assistant:")


R1ZERO_TOOLS = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
                "The assistant first thinks about the reasoning process and then provides the user with the answer "
                "inside <answer> </answer> tags. While thinking, the assistant can run Python by writing <code> ... </code>; "
                "whatever the code prints is shown inside <output> </output>. Python has math, itertools, sympy, z3 and "
                "networkx available. Each code block runs on its own.{bt} User: {q} Assistant:")
BT_TEXT = (" The assistant can also write <checkpoint> to mark a point in its reasoning, and <backtrack> what went "
           "wrong </backtrack> to discard everything after the last checkpoint and continue from there; the note stays visible.")


def chat_ids(system, q):
    if args.template == "r1zero" and args.tools and system != CODE_SYSTEM:
        return tok.encode(R1ZERO_TOOLS.format(q=q, bt=BT_TEXT if args.backtrack else ""))
    if args.template == "r1zero":
        if system == CODE_SYSTEM:
            q = q + "\nThe answer must be the complete Python code (no explanation) inside the answer tags."
        return tok.encode(R1ZERO.format(q=q))
    if args.tools:
        if system == SYSTEM:
            system = TOOL_SYSTEM
        elif args.code_tools:
            system = TOOL_SYSTEM_CODE
    msgs = [{"role": "user", "content": q}]
    if system and not (args.no_system and system == SYSTEM and not args.tools):
        msgs = [{"role": "system", "content": system}] + msgs
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)


def rollout(judge, prompts, sampler, explore_frac=0.0, samplers=None):
    """-> SimpleNamespace(texts, token_ids, masks, truncated, void, calls, explore)"""
    if args.tools or args.template:
        ex = [random.random() < explore_frac for _ in prompts]
        r = generate_with_tools(model, tok, prompts, judge, max_tokens=args.max_tokens, max_calls=args.max_calls,
                                sampler=sampler, completion_batch_size=args.gen_batch, prefill_batch_size=4,
                                explore=ex, explore_k=args.explore_k, tools=args.tools,
                                stop_answer=bool(args.template) and not args.last_answer,
                                stop_user=bool(args.template), samplers=samplers,
                                backtrack=args.backtrack, max_backtracks=args.max_backtracks)
        return SimpleNamespace(texts=r["texts"], token_ids=r["token_ids"], masks=r["masks"],
                               truncated=r["truncated"], void=r["void"], calls=r["calls"], explore=r["explore"],
                               segments=r.get("segments"), log_texts=r.get("log_texts"), backtracks=r.get("backtracks"))
    r = batch_generate(model, tok, prompts, max_tokens=args.max_tokens, return_token_ids=True, sampler=sampler,
                       completion_batch_size=args.gen_batch, prefill_batch_size=4)
    n = len(prompts)
    return SimpleNamespace(texts=r.texts, token_ids=r.token_ids, masks=[[1] * len(t) for t in r.token_ids],
                           truncated=[len(t) >= args.max_tokens for t in r.token_ids], void=[False] * n, calls=[0] * n,
                           explore=[False] * n)


# ------------------------------------------------------------------ eval
def evaluate(judge, step, base=False):
    model.eval()
    if base:
        set_lora(False)
    out = {}
    # math: 16 per task from the held-out eval seed
    items = []
    for t in TASKS:
        ds = reasoning_gym.create_dataset(t, size=args.eval_math_n, seed=EVAL_SEED)
        items += [dict(bucket=t, kind="math", question=e["question"], entry=e, ds=ds) for e in ds]
    code_rows = [json.loads(l) for l in itertools.islice(open("data/code/kodcode_eval.jsonl"), args.eval_code_n)]
    items += [dict(bucket="code_eval", kind="code", question=CodeSource.msg(r), row=r) for r in code_rows]  # may be empty
    if args.eval_gen_n:
        ge = [json.loads(l) for l in open("data/general/eval.jsonl")]
        by = defaultdict(list)
        for r in ge:
            by[r["source"]].append(r)
        take = []
        for src_, rs_ in sorted(by.items()):
            take += rs_[:max(1, round(args.eval_gen_n * len(rs_) / len(ge)))]
        items += [dict(bucket="gen_eval", kind="gen", question=r["question"], row=r) for r in take]
    prompts = [chat_ids(CODE_SYSTEM if it["kind"] == "code" else SYSTEM, it["question"]) for it in items]
    res = rollout(judge, prompts, make_sampler(temp=0.7, top_p=0.95))  # same sampler for every model/eval
    with ThreadPoolExecutor(4) as ex:
        rw = list(ex.map(lambda a: any_reward(judge, a[0], a[1], a[2]), zip(items, res.texts, res.truncated)))
    per = defaultdict(list)
    for it, (_, _, ok) in zip(items, rw):
        per[it["kind"]].append(float(ok))
        if it["kind"] == "gen":
            per["gen_" + it["row"]["source"]].append(float(ok))
        per[it["bucket"]].append(float(ok))
    if args.nothink_eval:
        mi = [k for k, it in enumerate(items) if it["kind"] == "math"]
        forced = [prompts[k] + tok.encode("<think>\n</think>\n<answer>", add_special_tokens=False) for k in mi]
        r2 = batch_generate(model, tok, forced, max_tokens=48, completion_batch_size=args.gen_batch, prefill_batch_size=4)
        nt = [float(rg_score(items[k]["ds"], items[k]["entry"], normalize(t.split("</answer>")[0])) >= 1)
              for k, t in zip(mi, r2.texts)]
        per["nothink"] = nt
    out = {k: round(statistics.mean(v), 4) for k, v in per.items()}
    if "nothink" in out:
        out["cot_gain"] = round(out["math"] - out["nothink"], 4)
    out.setdefault("code", None)
    _tt = [x for t in TRAIN_TASKS if t in TASKS for x in per.get(t, [])]
    out["train_tasks_acc"] = round(statistics.mean(_tt), 4) if _tt else None
    out["heldout_tasks_acc"] = (round(statistics.mean(x for t in TASKS if t not in TRAIN_TASKS for x in per[t]), 4)
                                if any(t not in TRAIN_TASKS for t in TASKS) else None)
    thinks = [think_text(t) for t in res.texts]
    out["behaviours"] = {k: round(statistics.mean(bool(p.search(th)) for th in thinks), 4) for k, p in BEHAVIOURS.items()}
    out["format"] = round(statistics.mean(bool(format_ok(t)) for it, t in zip(items, res.texts) if it["kind"] == "math"), 4)
    out["avg_tokens"] = round(statistics.mean(len(t) for t in res.token_ids), 1)
    out["tool_rate"] = round(statistics.mean(c > 0 for c in res.calls), 4)
    if base:
        set_lora(True)
    model.train()
    mx.clear_cache()
    with open(f"{args.run}/eval_samples_step{step}{'_base' if base else ''}.jsonl", "w") as f:
        for it, text, (r, why, ok), calls in zip(items, res.texts, rw, res.calls):
            f.write(json.dumps(dict(task=it["bucket"], question=it["question"][:3000], output=text,
                                    correct=ok, why=why, tool_calls=calls)) + "\n")
    out["n_per_task"] = args.eval_math_n
    out["base"] = base
    with open(f"{args.run}/eval.jsonl", "a") as f:
        f.write(json.dumps(dict(step=step, **out)) + "\n")
    print(f"[eval step {step}] math {out['math']:.3f} (train tasks {out['train_tasks_acc']}, held-out tasks "
          f"{out['heldout_tasks_acc']})  code {out['code']}  format {out['format']}  tokens {out['avg_tokens']}", flush=True)


# ------------------------------------------------------------------ main loop
from sandbox.judge import Judge

math_src = MathSource()
code_src = CodeSource() if args.code_frac > 0 else None
gen_src = GenSource() if (args.gen_frac > 0) else None
gen_buckets = sorted(gen_src.b) if gen_src else []
code_buckets = sorted(code_src.b) if code_src else []
cur = Curriculum(TRAIN_TASKS + code_buckets + gen_buckets, cur_state)
sampler = make_sampler(temp=args.temp, top_p=args.top_p)
LADDER = [float(x) for x in args.temp_ladder.split(",")] if args.temp_ladder else None
model.train()

import contextlib
uses_code = args.code_frac > 0 or args.eval_code_n > 0 or args.tools
with (Judge(workers=args.judge_workers) if uses_code else contextlib.nullcontext(None)) as judge, ThreadPoolExecutor(8) as pool:
    if args.eval_only:
        model.load_weights(args.eval_only, strict=False)
        print(f"eval-only: {args.eval_only}", flush=True)
        evaluate(judge, args.eval_tag)
        sys.exit(0)
    if start_step == 0 and not args.skip_eval0:
        evaluate(judge, 0)
    elif start_step > 0:
        prev = [json.loads(l) for l in open(f"{args.run}/eval.jsonl")] if os.path.exists(f"{args.run}/eval.jsonl") else []
        if not any(e.get("n_per_task") == args.eval_math_n and ("nothink" in e or not args.nothink_eval) for e in prev):
            print(f"eval size changed -> re-baselining at {args.eval_math_n}/task", flush=True)
            evaluate(judge, 0, base=True)        # starting model (LoRA off) at the new eval size
            evaluate(judge, start_step)          # current checkpoint at the new eval size
    for step in range(start_step + 1, args.steps + 1):
        t0 = time.time()
        items = []
        draws = [random.random() for _ in range(args.prompts)]
        n_code = sum(d < args.code_frac for d in draws)
        n_gen = sum(args.code_frac <= d < args.code_frac + args.gen_frac for d in draws)
        n_math = args.prompts - n_code - n_gen
        # stratified: every task once per step (shuffled), remaining slots by curriculum weight
        strat = []
        if args.stratified:
            task_pool = list(TRAIN_TASKS)
            while task_pool and len(strat) < n_math:
                t = random.choices(task_pool, weights=cur.weights(task_pool))[0]
                strat.append(t)
                task_pool.remove(t)
        for t in strat + [cur.pick(TRAIN_TASKS) for _ in range(n_math - len(strat))]:
            items.append(math_src.sample(t))
        for _ in range(n_code):
            items.append(code_src.sample(cur.pick(code_buckets)))
        for _ in range(n_gen):
            items.append(gen_src.sample(cur.pick(gen_buckets)))
        prompts = [chat_ids(CODE_SYSTEM if it["kind"] == "code" else SYSTEM, it["question"]) for it in items]
        allp = [p for p in prompts for _ in range(args.group)]
        cond_of = [None] * len(allp)          # (plan_index) for conditioned rollouts
        mode_of = [None] * len(allp)
        if args.modes:
            for gi in range(len(items)):
                for j in range(args.group):
                    z = j % args.modes
                    i = gi * args.group + j
                    allp[i] = prompts[gi] + tok.encode(f" [Mode {z + 1}]", add_special_tokens=False)
                    mode_of[i] = z
        plans = []                            # per item: list of (plan_text, plan_prompt_ids, plan_token_ids)
        model.eval()
        if args.self_seed:
            frac = args.plan_frac + (args.plan_frac_end - args.plan_frac) * min(1.0, (step - 1) / max(args.plan_fade_steps, 1))
            n_cond = int(round(args.group * frac))
            plan_prompts = [tok.encode(R1ZERO.split(" User:")[0] + " User: " + it["question"] +
                                       "\nBefore solving, briefly describe a strategy for approaching this problem. Assistant: Strategy:")
                            for it in items]
            pr = generate_with_tools(model, tok, [p for p in plan_prompts for _ in range(args.plans)], None,
                                     max_tokens=args.plan_max, tools=False, stop_answer=False, stop_user=True,
                                     sampler=make_sampler(temp=args.plan_temp, top_p=0.95),
                                     completion_batch_size=args.gen_batch, prefill_batch_size=4)
            for gi, it in enumerate(items):
                pl = []
                for k in range(args.plans):
                    j = gi * args.plans + k
                    ids_ = pr["token_ids"][j]
                    txt = tok.decode(ids_)
                    cut = txt.split("\n")[0].strip()
                    keep = tok.encode(" " + cut, add_special_tokens=False) if cut else []
                    pl.append((cut, plan_prompts[gi], keep))
                plans.append(pl)
                sysm = CODE_SYSTEM if it["kind"] == "code" else SYSTEM
                for j in range(args.group - n_cond, args.group):
                    k = (j - (args.group - n_cond)) % args.plans
                    if not pl[k][0]:
                        continue
                    allp[gi * args.group + j] = chat_ids(sysm, it["question"] + "\nA possible approach: " + pl[k][0])
                    cond_of[gi * args.group + j] = k

        temps, samplers = None, None
        if LADDER:
            lo, hi = LADDER
            temps = []
            for _ in items:
                g = [lo + (hi - lo) * j / max(args.group - 1, 1) for j in range(args.group)]
                random.shuffle(g)
                temps += g
            samplers = [make_sampler(temp=x, top_p=args.top_p) for x in temps]
        open_k = None
        if args.open_tree:
            G_, O_ = args.group, args.open_tree
            M_ = G_ // O_
            from tools import generate_with_tools as _gwt
            ro = _gwt(model, tok, [allp[gi * G_] for gi in range(len(items)) for _ in range(O_)], judge,
                      max_tokens=args.open_len, tools=False, stop_answer=True, stop_user=True,
                      sampler=make_sampler(temp=1.0, top_p=args.top_p), completion_batch_size=args.gen_batch, prefill_batch_size=4)
            opens = []
            for ids_ in ro["token_ids"]:
                pieces = [tok.decode([t]) for t in ids_]
                acc, ans_k = "", len(ids_)
                for k, pc in enumerate(pieces):
                    acc += pc
                    if "<answer>" in acc or "User:" in acc:
                        ans_k = k
                        break
                bnd = [k + 1 for k in range(ans_k) if k + 1 >= 6 and ("\n" in pieces[k] or pieces[k].rstrip().endswith((".", ":", "?", "!")))]
                cut = bnd[-1] if bnd else max(1, min(ans_k, len(ids_)))
                opens.append(list(ids_[:cut]))
            open_k = [len(opens[(i // G_) * O_ + (i % G_) // M_]) for i in range(len(allp))]
            cp = [allp[i] + opens[(i // G_) * O_ + (i % G_) // M_] for i in range(len(allp))]
            rc = rollout(judge, cp, sampler, 0.0, samplers)
            res = SimpleNamespace(
                texts=[tok.decode(opens[(i // G_) * O_ + (i % G_) // M_]) + rc.texts[i] for i in range(len(allp))],
                token_ids=[opens[(i // G_) * O_ + (i % G_) // M_] + rc.token_ids[i] for i in range(len(allp))],
                masks=[[1] * open_k[i] + rc.masks[i] for i in range(len(allp))],
                truncated=rc.truncated, void=rc.void, calls=rc.calls, explore=rc.explore)
        else:
            res = rollout(judge, allp, sampler, args.explore_frac, samplers)
        model.train()
        mx.clear_cache()
        t_gen = time.time() - t0

        think_toks = [sum(m) for m in res.masks]          # model-written tokens only (tool output excluded)
        jobs = [(items[i // args.group], res.texts[i], res.truncated[i],
                 cur.p[items[i // args.group]["bucket"]], think_toks[i]) for i in range(len(allp))]
        rw = list(pool.map(lambda a: any_reward(judge, *a), jobs))
        rewards = [r for r, _, _ in rw]
        correct = [ok for _, _, ok in rw]
        if args.tool_cost > 0:
            rewards = [r - (args.tool_cost * res.calls[i] if correct[i] else 0.0) for i, r in enumerate(rewards)]
        lens = [sum(m) for m in res.masks]
        shaped_groups = 0
        if args.len_alpha > 0:
            for gi in range(len(items)):
                sl = slice(gi * args.group, (gi + 1) * args.group)
                cs, Ls = correct[sl], think_toks[sl]
                if statistics.mean(map(float, cs)) < args.len_gate or max(Ls) == min(Ls):
                    continue          # low accuracy -> pure correctness reward (avoid length-collapse)
                shaped_groups += 1
                lo, hi = min(Ls), max(Ls)
                for j in range(args.group):
                    i = gi * args.group + j
                    lam = 0.5 - (think_toks[i] - lo) / (hi - lo)
                    if rw[i][1] in ("truncated", "malformed"):
                        continue
                    rewards[i] = max(0.0, rewards[i] + args.len_alpha * (lam if correct[i] else min(0.0, lam)))
        t_rew = time.time() - t0 - t_gen

        div_stats = {}
        if args.div_alpha > 0 or args.hard_len_alpha > 0:
            G = args.group
            todo = []
            for gi in range(len(items)):
                cs = [i for i in range(gi * G, (gi + 1) * G) if correct[i]]
                if len(cs) >= 2:
                    todo.append(cs)
            if args.div_alpha > 0 and todo:
                flat = [i for cs in todo for i in cs]
                E = embed([allp[i] + res.token_ids[i] for i in flat], [len(allp[i]) for i in flat])
                emap = dict(zip(flat, E))
                nov_all = []
                for cs in todo:
                    nov = novelty([emap[i] for i in cs])
                    order = sorted(range(len(cs)), key=lambda q: nov[q])
                    for rank, q in enumerate(order):        # rank-based: least novel -0.5a ... most novel +0.5a
                        rewards[cs[q]] += args.div_alpha * (rank / max(1, len(cs) - 1) - 0.5)
                    nov_all += nov
                div_stats["novelty_correct"] = round(statistics.mean(nov_all), 4)
            if args.hard_len_alpha > 0:
                for cs in todo:
                    gi = cs[0] // G
                    acc = sum(correct[gi * G:(gi + 1) * G]) / G
                    if acc > 0.5:
                        continue
                    Ls = [lens[i] for i in cs]
                    lo, hi = min(Ls), max(Ls)
                    if hi == lo:
                        continue
                    for i in cs:                             # shorter correct solution on a hard problem -> bonus
                        rewards[i] += args.hard_len_alpha * (0.5 - (lens[i] - lo) / (hi - lo))
            div_stats["groups_with_2plus_correct"] = len(todo)

        n_overlong = 0
        if args.overlong_buffer > 0:
            start = args.max_tokens - args.overlong_buffer
            for i in range(len(rewards)):
                if not res.truncated[i] and lens[i] > start:
                    n_overlong += 1
                    rewards[i] = max(0.0, rewards[i] - args.overlong_penalty * (lens[i] - start) / args.overlong_buffer)

        extra = [0.0] * len(allp)                  # additive advantage terms (mode ensemble bonuses)
        mode_stats = {}
        if args.modes:
            G, K = args.group, args.modes
            bon, mode_acc = [], defaultdict(list)
            for gi in range(len(items)):
                idx = list(range(gi * G, (gi + 1) * G))
                acc = {z: statistics.mean(float(correct[i]) for i in idx if mode_of[i] == z) for z in range(K)}
                for z, a in acc.items():
                    mode_acc[z].append(a)
                ans = {}
                for i in idx:
                    t_ = re.findall(r"<answer>(.*?)</answer>", res.texts[i], re.S)
                    ans[i] = normalize(t_[-1]) if t_ else None
                for i in idx:
                    z = mode_of[i]
                    others = [acc[o] for o in range(K) if o != z]
                    if correct[i]:
                        # correct where the other modes mostly fail -> big contribution to the cross-mode vote
                        extra[i] += args.mode_gamma * (1.0 - statistics.mean(others))
                    elif ans[i] is not None:
                        # wrong, but not the SAME wrong answer the other modes give -> errors stay decorrelated
                        other_wrong = [ans[j] for j in idx if mode_of[j] != z and not correct[j] and ans[j] is not None]
                        if other_wrong:
                            common = max(set(other_wrong), key=other_wrong.count)
                            if ans[i] != common:
                                extra[i] += args.mode_delta
                    bon.append(extra[i])
            mode_stats = dict(mode_acc={f"m{z + 1}": round(statistics.mean(v), 3) for z, v in mode_acc.items()},
                              mode_bonus=round(statistics.mean(bon), 4))

        # ------------------------------------------------------------ Monte-Carlo progress credit (branching)
        branch = {}                                 # rollout idx -> (token advantage list, [(prefix_ids, cont list)])
        pr_stats = {}
        if args.open_tree:
            G_, O_ = args.group, args.open_tree
            M_ = G_ // O_
            spreads = []
            for gi in range(len(items)):
                cg = [float(bool(correct[gi * G_ + j])) for j in range(G_)]
                v0 = statistics.mean(cg)
                vo = [statistics.mean(cg[o * M_:(o + 1) * M_]) for o in range(O_)]
                spreads.append(max(vo) - min(vo))
                if v0 in (0.0, 1.0) or v0 > args.open_max_v:
                    continue
                for j in range(G_):
                    i = gi * G_ + j
                    L = len(res.token_ids[i]); k = open_k[i]; v = vo[j // M_]
                    branch[i] = ([v - v0] * k + [cg[j] - v] * (L - k), [], [])
            pr_stats = dict(open_spread=round(statistics.mean(spreads), 3),
                            open_len=round(statistics.mean(open_k), 1))
        if args.progress and not args.open_tree:
            G_ = args.group
            corr_g = [[bool(correct[gi * G_ + j]) for j in range(G_)] for gi in range(len(items))]
            def cands(pred):
                return [i for i in range(len(allp)) if pred(i // G_) and not res.void[i] and not res.truncated[i]
                        and "<answer>" in res.texts[i] and all(mm == 1 for mm in res.masks[i])]
            mixed = cands(lambda g: 0 < sum(corr_g[g]) < G_)
            allfail = [i for i in cands(lambda g: sum(corr_g[g]) == 0)]
            random.shuffle(mixed); random.shuffle(allfail)
            half = args.progress // 2
            pick = allfail[:half]
            pick += mixed[:args.progress - len(pick)]
            pick += allfail[half:half + args.progress - len(pick)]
            cut_fr = [float(x) for x in args.pr_cuts.split(",")]
            plan = []                                   # (rollout idx, [cut token positions])
            for i in pick:
                ids_ = res.token_ids[i]
                pieces = [tok.decode([t]) for t in ids_]
                acc, ans_k = "", len(ids_)
                for k, pc in enumerate(pieces):
                    acc += pc
                    if "<answer>" in acc:
                        ans_k = k
                        break
                bounds = [k + 1 for k in range(ans_k) if "\n" in pieces[k] or pieces[k].rstrip().endswith((".", ":", "!", "?"))]
                cuts = []
                for f in cut_fr:
                    tgt = int(ans_k * f)
                    b = [k for k in bounds if tgt * 0.5 < k <= tgt]
                    k = b[-1] if b else tgt
                    if 0 < k < ans_k and (not cuts or k > cuts[-1]):
                        cuts.append(k)
                if cuts:
                    plan.append((i, cuts))
            jobs_p = [(i, k) for i, cuts in plan for k in cuts for _ in range(args.pr_conts)]
            if jobs_p:
                model.eval()
                cres = rollout(judge, [allp[i] + res.token_ids[i][:k] for i, k in jobs_p], sampler, 0.0, None)
                model.train()
                mx.clear_cache()
                cjobs = [(items[i // G_], tok.decode(res.token_ids[i][:k]) + cres.texts[n], cres.truncated[n],
                          cur.p[items[i // G_]["bucket"]], k + sum(cres.masks[n])) for n, (i, k) in enumerate(jobs_p)]
                cok = [bool(o) for _, _, o in pool.map(lambda a: any_reward(judge, *a), cjobs)]
                n = 0
                gains, rescued = [], 0
                for i, cuts in plan:
                    vs, conts = [], []
                    for k in cuts:
                        oks = cok[n:n + args.pr_conts]
                        conts.append((k, [(cres.token_ids[n + c], cres.masks[n + c], oks[c], cres.void[n + c])
                                          for c in range(args.pr_conts)]))
                        vs.append(statistics.mean(float(o) for o in oks))
                        n += args.pr_conts
                    gi = i // G_
                    v0 = statistics.mean(float(c) for c in corr_g[gi])
                    pts = [0] + cuts + [len(res.token_ids[i])]
                    vals = [v0] + vs + [float(bool(correct[i]))]
                    tadv = []
                    for s_ in range(len(pts) - 1):
                        tadv += [vals[s_ + 1] - vals[s_]] * (pts[s_ + 1] - pts[s_])
                    branch[i] = (tadv, conts, vs)
                    gains.append(vs[-1] - v0)
                    rescued += int(v0 == 0 and any(v > 0 for v in vs))
                pr_stats = dict(pr_traces=len(plan), pr_conts=len(jobs_p), pr_cont_acc=round(statistics.mean(cok), 3),
                                pr_gain=round(statistics.mean(gains), 3), pr_rescued=rescued,
                                pr_allfail=sum(1 for i, _ in plan if sum(corr_g[i // G_]) == 0))

        # advantages, curriculum update, zero-variance filtering
        seqs, plens, advs, tmasks = [], [], [], []
        src = []                                   # rollout index (into allp) of each training sequence
        n_void = 0
        kept_groups = 0
        for gi, it in enumerate(items):
            rs = rewards[gi * args.group:(gi + 1) * args.group]
            cur.update(it["bucket"], statistics.mean(float(c) for c in correct[gi * args.group:(gi + 1) * args.group]))
            m = statistics.mean(rs)
            if args.algo == "grpo" and not args.wreinforce and max(rs) - min(rs) < 1e-6 and not args.modes:
                continue
            if args.passk:
                cg = [bool(correct[gi * args.group + j]) for j in range(args.group)]
                n_, c_ = len(cg), sum(cg)
                if c_ in (0, n_):
                    continue
                k_ = min(args.passk, n_)
                rbar = 1 - math.comb(n_ - c_, k_) / math.comb(n_, k_)
                a_pos = 1 - rbar
                a_neg = 1 - rbar - math.comb(n_ - c_ - 1, k_ - 1) / math.comb(n_ - 1, k_ - 1)
            kept_groups += 1
            for j in range(args.group):
                i = gi * args.group + j
                if res.void[i]:                      # SimpleTIR: keep in the group mean, drop from the gradient
                    n_void += 1
                    continue
                pre = allp[i] if args.modes else prompts[gi]     # modes: train conditioned on the mode tag
                seqs.append(pre + res.token_ids[i])
                plens.append(len(pre))
                n_before = len(advs)
                if i in branch:     # Monte-Carlo progress credit (shaping offset added uniformly)
                    off = (rs[j] - float(bool(correct[i]))) + extra[i]
                    advs.append([a + off for a in branch[i][0]])
                elif args.passk:      # shaping terms (overlong etc.) kept as offset from binary correctness
                    advs.append((a_pos if correct[i] else a_neg) + (rs[j] - float(bool(correct[i]))) + extra[i])
                elif args.wreinforce:
                    advs.append((args.wreinforce if correct[i] else -1.0) + extra[i])
                else:
                    advs.append(rs[j] - m + extra[i])
                tmasks.append(res.masks[i])
                src.append(i)
                sg = getattr(res, "segments", None)
                if args.backtrack and sg and len(sg[i]) > 1:   # rewound rollout: one training sequence per context
                    a_ = advs.pop(); seqs.pop(); plens.pop(); tmasks.pop(); src.pop()
                    a_ = a_ if not isinstance(a_, list) else statistics.mean(a_)
                    for ctx_, g_, m_ in sg[i]:
                        if not g_:
                            continue
                        seqs.append(list(ctx_) + list(g_)); plens.append(len(ctx_))
                        advs.append(a_); tmasks.append(list(m_)); src.append(i)

        if branch:
            in_seqs = set(src)
            for i, (tadv, conts, vs) in branch.items():
                if i not in in_seqs and any(abs(a_) > 1e-9 for a_ in tadv):   # all-fail group was filtered -> add the branched trace itself
                    seqs.append(allp[i] + res.token_ids[i]); plens.append(len(allp[i]))
                    advs.append(list(tadv)); tmasks.append(res.masks[i]); src.append(i)
                if args.pr_train_conts:
                    for (k, cl), v in zip(conts, vs):
                        if all(o == cl[0][2] for _, _, o, _ in cl):
                            continue                 # zero-variance at this cut: no signal
                        for cid, cm, o, vd in cl:
                            if vd:
                                continue
                            pre = allp[i] + res.token_ids[i][:k]
                            seqs.append(pre + cid); plens.append(len(pre))
                            advs.append(float(o) - v); tmasks.append(cm); src.append(i)

        # ------------------------------------------------------------ self-seed / LTE extras
        beh_prompt = [allp[i] if cond_of[i] is not None else None for i in src]   # conditioning prompt per seq
        seed_stats = {}
        if args.self_seed or args.lte:
            G = args.group
            # (a) learn from own failures: all rollouts wrong -> retry conditioned on the model's own wrong answer
            lte_added = lte_ok = 0
            if args.lte:
                fail_items = [gi for gi in range(len(items)) if not any(correct[gi * G:(gi + 1) * G])]
                rp, rmeta = [], []
                for gi in fail_items:
                    it = items[gi]
                    prev = None
                    for i in range(gi * G, (gi + 1) * G):
                        a_ = re.findall(r"<answer>(.*?)</answer>", res.texts[i], re.S)
                        if a_:
                            prev = a_[-1].strip()[:200]
                            break
                    note = (f"\nA previous attempt answered: {prev}. That answer is wrong." if prev
                            else "\nA previous attempt failed to reach an answer.")
                    cp = chat_ids(CODE_SYSTEM if it["kind"] == "code" else SYSTEM, it["question"] + note)
                    for _ in range(args.lte_n):
                        rp.append(cp)
                        rmeta.append(gi)
                if rp:
                    rr = rollout(judge, rp, sampler)
                    rjobs = [(items[gi], rr.texts[k], rr.truncated[k], cur.p[items[gi]["bucket"]], sum(rr.masks[k]))
                             for k, gi in enumerate(rmeta)]
                    rrw = list(pool.map(lambda a: any_reward(judge, *a), rjobs))
                    for gi in set(rmeta):
                        ks = [k for k, g in enumerate(rmeta) if g == gi]
                        rs_ = [rrw[k][0] for k in ks] + rewards[gi * G:(gi + 1) * G]
                        mu = statistics.mean(rs_)
                        if max(rs_) - min(rs_) < 1e-6:
                            continue
                        for k in ks:
                            seqs.append(prompts[gi] + rr.token_ids[k])
                            plens.append(len(prompts[gi]))
                            advs.append(rrw[k][0] - mu)
                            tmasks.append(rr.masks[k])
                            src.append(None)
                            beh_prompt.append(rp[k])
                            lte_added += 1
                            lte_ok += int(rrw[k][2])
            # (b) plan proposer: reward = uplift of its plan over plain rollouts (+ diversity bonus)
            plan_seqs = 0
            uplifts, distinct = [], []
            if args.self_seed:
                for gi in range(len(items)):
                    pl = plans[gi]
                    plain = [rewards[gi * G + j] for j in range(G) if cond_of[gi * G + j] is None]
                    base = statistics.mean(plain) if plain else statistics.mean(rewards[gi * G:(gi + 1) * G])
                    words = [set(p[0].lower().split()) for p in pl]
                    pem = None
                    if args.plan_sem and sum(1 for p in pl if p[2]) >= 2:
                        pidx = [k for k, p in enumerate(pl) if p[2]]
                        pv = embed([pl[k][1] + pl[k][2] for k in pidx], [len(pl[k][1]) for k in pidx])
                        pnov = novelty(pv)
                        pem = dict(zip(pidx, pnov))
                    distinct.append(len({p[0] for p in pl if p[0]}) / max(1, len(pl)))
                    R = {}
                    for k, p in enumerate(pl):
                        rk = [rewards[gi * G + j] for j in range(G) if cond_of[gi * G + j] == k]
                        if not rk or not p[2]:
                            continue
                        up = statistics.mean(rk) - base
                        if pem is not None:
                            R[k] = up + args.plan_div * pem.get(k, 0.0) * 10      # embedding novelty is small -> scale
                        else:
                            others = [len(words[k] & words[o]) / max(1, len(words[k] | words[o])) for o in range(len(pl)) if o != k]
                            R[k] = up + args.plan_div * (1 - (max(others) if others else 0))
                        uplifts.append(up)
                    if len(R) < 2:
                        continue
                    mu = statistics.mean(R.values())
                    if max(R.values()) - min(R.values()) < 1e-6:
                        continue
                    for k, rv in R.items():
                        seqs.append(pl[k][1] + pl[k][2])
                        plens.append(len(pl[k][1]))
                        advs.append(args.plan_weight * (rv - mu))
                        tmasks.append([1] * len(pl[k][2]))
                        src.append(None)
                        beh_prompt.append(None)
                        plan_seqs += 1
            # (c) behaviour log-probs of conditioned sequences under their conditioning prompt (no grad)
            beh_rows = [None] * len(seqs)
            cidx = [n for n in range(len(seqs)) if beh_prompt[n] is not None]
            for k in range(0, len(cidx), args.micro):
                chunk = cidx[k:k + args.micro]
                bseq = [beh_prompt[n] + seqs[n][plens[n]:] for n in chunk]
                T = max(len(b) for b in bseq)
                bids = mx.array([b + [0] * (T - len(b)) for b in bseq])
                blp = mx.stop_gradient(seq_logps(model, bids, None))
                mx.eval(blp)
                blp = blp.tolist()
                for n, row in zip(chunk, blp):
                    st = len(beh_prompt[n]) - 1
                    beh_rows[n] = row[st:st + len(seqs[n]) - plens[n]]
            mx.clear_cache()
            cond_correct = [float(correct[i]) for i in range(len(allp)) if cond_of[i] is not None]
            plain_correct = [float(correct[i]) for i in range(len(allp)) if cond_of[i] is None]
            seed_stats = dict(
                plain_acc=round(statistics.mean(plain_correct), 3) if plain_correct else None,
                cond_acc=round(statistics.mean(cond_correct), 3) if cond_correct else None,
                plan_uplift=round(statistics.mean(uplifts), 4) if uplifts else None,
                plan_distinct=round(statistics.mean(distinct), 3) if distinct else None,
                plan_seqs=plan_seqs, lte_added=lte_added, lte_correct=lte_ok,
                plan_examples=[p[0][:120] for p in (plans[0] if plans else [])])
        else:
            beh_rows = [None] * len(seqs)
        rev = [t for t in res.texts if t.count("<answer>") > 1]
        seed_stats["revise_rate"] = round(len(rev) / max(1, len(res.texts)), 3)

        ppo_stats = {}
        if args.algo == "ppo" and seqs:
            keep_idx = [i for i in range(len(rewards)) if not res.void[i]]   # same order as seqs
            seq_correct = [float(correct[i]) for i in keep_idx]
            # length term: -c * (L - mean L of the group), known exactly -> no critic needed
            grp_mean_len = {}
            for gi in range(len(items)):
                Ls = [lens[gi * args.group + j] for j in range(args.group)]
                grp_mean_len[gi] = statistics.mean(Ls)
            seq_len_adv = [-args.ppo_len_cost * (lens[i] - grp_mean_len[i // args.group]) for i in keep_idx]
            seq_rewards = seq_correct
            all_vals, all_rows = [], []
            for k in range(0, len(seqs), args.micro):
                ids, mask = pad_batch(seqs[k:k + args.micro], plens[k:k + args.micro], tmasks[k:k + args.micro])
                v = mx.stop_gradient(critic_values(model, ids))
                mx.eval(v)
                all_vals += [row for row in v.tolist()]
                all_rows += [row for row in mask.tolist()]
            tok_adv, tok_tgt, v_pred, v_tgt = [], [], [], []
            for vrow, mrow, R, la in zip(all_vals, all_rows, seq_rewards, seq_len_adv):
                pos = [j for j, mm in enumerate(mrow) if mm > 0]
                n = len(pos)
                adv_row, tgt_row = [0.0] * len(mrow), [0.0] * len(mrow)
                if n:
                    lam = min(0.995, max(0.9, 1 - 1 / (args.gae_alpha * n)))
                    vs = [vrow[j] for j in pos]
                    gae = 0.0
                    for kk in range(n - 1, -1, -1):
                        nxt = vs[kk + 1] if kk + 1 < n else 0.0
                        r = R if kk == n - 1 else 0.0
                        delta = r + nxt - vs[kk]
                        gae = delta + lam * gae
                        adv_row[pos[kk]] = gae + la                      # correctness credit (GAE) + length term
                        tgt_row[pos[kk]] = R                             # critic target: correct (0/1)
                    v_pred += vs
                    v_tgt += [R] * n
                tok_adv.append(adv_row)
                tok_tgt.append(tgt_row)
            flat = [a for row, mrow in zip(tok_adv, all_rows) for a, mm in zip(row, mrow) if mm > 0]
            mu = statistics.mean(flat) if flat else 0.0
            sd = (statistics.pstdev(flat) if len(flat) > 1 else 1.0) or 1.0
            tok_adv = [[(a - mu) / sd if mm > 0 else 0.0 for a, mm in zip(row, mrow)] for row, mrow in zip(tok_adv, all_rows)]
            var_t = statistics.pvariance(v_tgt) if len(v_tgt) > 1 else 0.0
            ev = 1 - statistics.pvariance([t - p for t, p in zip(v_tgt, v_pred)]) / var_t if var_t > 1e-8 else None
            # critic update (value regression to final reward)
            cgrads, vloss = None, 0.0
            for k in range(0, len(seqs), args.micro):
                ids, mask = pad_batch(seqs[k:k + args.micro], plens[k:k + args.micro], tmasks[k:k + args.micro])
                T1 = ids.shape[1] - 1
                tgt = mx.array([row + [0.0] * (T1 - len(row)) for row in tok_tgt[k:k + args.micro]])
                l, g = critic_loss_and_grad(model, ids, mask, tgt)
                _, gc = split_grads(g)
                cgrads = gc if cgrads is None else tree_map(lambda a, b: a + b, cgrads, gc)
                mx.eval(cgrads, l)
                vloss += l.item() / max(1, (len(seqs) + args.micro - 1) // args.micro)
            model.update(copt.apply_gradients(cgrads, model), strict=False)
            mx.eval(model.parameters(), copt.state)
            del cgrads
            ppo_stats = dict(value_loss=round(vloss, 5), explained_var=None if ev is None else round(ev, 4),
                             ppo_reward=round(statistics.mean(c - args.ppo_len_cost * lens[i] for c, i in zip(seq_correct, keep_idx)), 4),
                             v_mean=round(statistics.mean(v_pred), 4) if v_pred else None, adv_raw_std=round(sd, 4))
            advs = tok_adv          # per-token advantages for the policy update

        loss_v = kl_v = 0.0
        entropy = None
        grad_norm, cosines = None, []
        policy_frozen = args.algo == "ppo" and step <= args.critic_warmup
        mb_done = False
        if seqs and args.minibatches > 1 and args.algo == "grpo" and not args.self_seed:
            # behaviour log-probs once (no grad), then several clipped optimizer steps on shuffled mini-batches
            old_rows, ent_vals, ref_rows = [], [], []
            for k in range(0, len(seqs), args.micro):
                ids, mask = pad_batch(seqs[k:k + args.micro], plens[k:k + args.micro], tmasks[k:k + args.micro])
                if args.beta > 0:                              # reference = starting weights (LoRA off)
                    set_lora(False)
                    rl_ = mx.stop_gradient(seq_logps(model, ids, None))
                    mx.eval(rl_)
                    set_lora(True)
                    for r_, sq in zip(rl_.tolist(), seqs[k:k + args.micro]):
                        ref_rows.append(r_[:len(sq) - 1])
                    del rl_
                lg_ = mx.stop_gradient(model(ids[:, :-1])).astype(mx.float32)
                lsm = lg_ - mx.logsumexp(lg_, axis=-1, keepdims=True)
                lp_ = mx.take_along_axis(lsm, ids[:, 1:, None], axis=-1).squeeze(-1)
                ent_ = -(mx.exp(lsm) * lsm).sum(-1)
                ent_vals.append(((ent_ * mask).sum() / mx.maximum(mask.sum(), 1)).item())
                mx.eval(lp_)
                for r_, sq in zip(lp_.tolist(), seqs[k:k + args.micro]):
                    old_rows.append(r_[:len(sq) - 1])
                del lg_, lsm, ent_
            mx.clear_cache()
            entropy = statistics.mean(ent_vals)              # full-batch entropy
            def adv_rows(sel, T1):
                rows_ = []
                for i in sel:
                    a_ = advs[i]
                    if isinstance(a_, list):
                        r_ = [0.0] * T1
                        for k_, v_ in enumerate(a_):
                            j_ = plens[i] - 1 + k_
                            if j_ < T1:
                                r_[j_] = v_
                    else:
                        r_ = [a_] * T1
                    rows_.append(r_)
                return mx.array(rows_)
            order = list(range(len(seqs)))
            random.shuffle(order)
            mbs = [order[i::args.minibatches] for i in range(args.minibatches)]
            clipf, gnorms = [], []
            for mb in mbs:
                if not mb:
                    continue
                norm = len(mb) * args.max_tokens
                grads = None
                for k in range(0, len(mb), args.micro):
                    sel = mb[k:k + args.micro]
                    ids, mask = pad_batch([seqs[i] for i in sel], [plens[i] for i in sel], [tmasks[i] for i in sel])
                    T1 = ids.shape[1] - 1
                    old = mx.array([old_rows[i] + [0.0] * (T1 - len(old_rows[i])) for i in sel])
                    ref = (mx.array([ref_rows[i] + [0.0] * (T1 - len(ref_rows[i])) for i in sel]) if ref_rows else None)
                    (l, (cf, klv)), g = ppo_loss_and_grad(model, ids, mask, adv_rows(sel, T1), old, norm, ref)
                    grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
                    mx.eval(grads, l, cf, klv)
                    loss_v += l.item() / len(mbs)
                    clipf.append(cf.item())
                    kl_v += klv.item() / max(1, len(mbs) * ((len(mb) + args.micro - 1) // args.micro))
                gnorms.append(mx.sqrt(sum((v.astype(mx.float32) ** 2).sum() for _, v in tree_flatten(grads))).item())
                model.update(opt.apply_gradients(grads, model), strict=False)
                mx.eval(model.parameters(), opt.state)
                del grads
                mx.clear_cache()
            grad_norm = statistics.mean(gnorms) if gnorms else None
            seed_stats["clip_frac"] = round(statistics.mean(clipf), 4) if clipf else None
            mb_done = True
        if seqs and not policy_frozen and not mb_done:
            ids0, mask0 = pad_batch(seqs[:args.micro], plens[:args.micro], tmasks[:args.micro])
            lg0 = mx.stop_gradient(model(ids0[:, :-1])).astype(mx.float32)
            lp0 = lg0 - mx.logsumexp(lg0, axis=-1, keepdims=True)
            ent0 = -(mx.exp(lp0) * lp0).sum(-1)
            entropy = ((ent0 * mask0).sum() / mx.maximum(mask0.sum(), 1)).item()
            ENT_EMA[0] = entropy if ENT_EMA[0] is None else 0.8 * ENT_EMA[0] + 0.2 * entropy
            del lg0, lp0, ent0
            mx.clear_cache()
            norm = len(seqs) * args.max_tokens
            grads = None
            prev_g, cosines = None, []
            ent_c = args.ent_coef if (args.ent_target > 0 and ENT_EMA[0] is not None and ENT_EMA[0] < args.ent_target) else 0.0
            div_stats["ent_bonus_on"] = ent_c > 0
            for k in range(0, len(seqs), args.micro):
                ids, mask = pad_batch(seqs[k:k + args.micro], plens[k:k + args.micro], tmasks[k:k + args.micro])
                if args.beta > 0:
                    set_lora(False)
                    ref_lp = mx.stop_gradient(seq_logps(model, ids, None))
                    mx.eval(ref_lp)
                    set_lora(True)
                else:
                    ref_lp = mx.zeros((ids.shape[0], ids.shape[1] - 1))
                if args.algo == "ppo":
                    T1 = ids.shape[1] - 1
                    adv_in = mx.array([row + [0.0] * (T1 - len(row)) for row in advs[k:k + args.micro]])
                else:
                    adv_in = mx.array(advs[k:k + args.micro])
                if any(b is not None for b in beh_rows[k:k + args.micro]):
                    T1 = ids.shape[1] - 1
                    bm, cf = [], []
                    for n in range(k, min(k + args.micro, len(seqs))):
                        row = [0.0] * T1
                        if beh_rows[n] is not None:
                            for q, v in enumerate(beh_rows[n]):
                                j = plens[n] - 1 + q
                                if j < T1:
                                    row[j] = v
                        bm.append(row)
                        cf.append(1.0 if beh_rows[n] is not None else 0.0)
                    (l, kl), g = loss_and_grad(model, ids, mask, adv_in, ref_lp, norm, mx.array(bm), mx.array(cf), ent_c)
                else:
                    (l, kl), g = loss_and_grad(model, ids, mask, adv_in, ref_lp, norm, None, None, ent_c)
                if args.algo == "ppo":
                    g, _ = split_grads(g)
                gflat = mx.concatenate([v.reshape(-1).astype(mx.float32) for _, v in tree_flatten(g)])
                if prev_g is not None:
                    cos = (gflat * prev_g).sum() / (mx.linalg.norm(gflat) * mx.linalg.norm(prev_g) + 1e-12)
                    cosines.append(cos.item())
                prev_g = gflat
                grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
                mx.eval(grads, l, kl)
                loss_v += l.item()
                kl_v += kl.item() * len(seqs[k:k + args.micro]) / len(seqs)
            grad_norm = mx.sqrt(sum((v.astype(mx.float32) ** 2).sum() for _, v in tree_flatten(grads))).item()
            model.update(opt.apply_gradients(grads, model), strict=False)
            mx.eval(model.parameters(), opt.state)
            del grads
        mx.clear_cache()
        t_step = time.time() - t0
        peak = mx.get_peak_memory() / 1e9

        # logging
        thinks = [think_text(t) for t in res.texts]
        beh = {k: statistics.mean(bool(p.search(th)) for th in thinks) for k, p in BEHAVIOURS.items()}
        by_kind = defaultdict(list)
        for it_i, c in enumerate(correct):
            by_kind[items[it_i // args.group]["kind"]].append(float(c))
        log = dict(step=step, t=round(t_step, 1), peak_gb=round(mx.get_peak_memory() / 1e9, 2), t_gen=round(t_gen, 1), t_reward=round(t_rew, 1),
                   reward=round(statistics.mean(rewards), 4),
                   acc_math=round(statistics.mean(by_kind["math"]), 3) if by_kind["math"] else None,
                   acc_code=round(statistics.mean(by_kind["code"]), 3) if by_kind["code"] else None,
                   kept_groups=kept_groups, loss=round(loss_v, 5), kl=round(kl_v, 5) if args.beta > 0 else None,
                   tokens=round(statistics.mean(lens), 1), truncated=round(statistics.mean(l >= args.max_tokens for l in lens), 3),
                   think_rate=round(statistics.mean("<think>" in t for t in res.texts), 3),
                   format_rate=round(statistics.mean(bool(format_ok(t)) for t in res.texts), 3),
                   think_tokens=round(statistics.mean(think_toks), 1),
                   difficulty={b: round(1 - p, 3) for b, p in cur.p.items()},
                   acc_by_task={b: round(statistics.mean(float(c) for j, c in enumerate(correct)
                                                         if items[j // args.group]["bucket"] == b), 3)
                                for b in sorted({it["bucket"] for it in items})},
                   lr=args.lr, shaped_groups=shaped_groups, void=n_void, entropy=entropy, temp=args.temp,
                   overlong=n_overlong, grad_norm=grad_norm, algo=args.algo, policy_frozen=policy_frozen, **ppo_stats, **seed_stats, **div_stats, **mode_stats, **pr_stats,
                   acc_by_temp=({f"{b:.1f}": round(statistics.mean(float(correct[i]) for i in range(len(correct))
                                                                    if round(temps[i], 1) == b), 3)
                                 for b in sorted({round(x, 1) for x in temps})} if temps else None),
                   answer_rate=round(statistics.mean("<answer>" in x for x in res.texts), 3),
                   acc_explore=(round(statistics.mean(float(correct[i]) for i in range(len(correct))
                                                      if res.explore[i] and res.calls[i] > 0), 3)
                                if any(res.explore[i] and res.calls[i] > 0 for i in range(len(correct))) else None),
                   acc_tool_noexplore=(round(statistics.mean(float(correct[i]) for i in range(len(correct))
                                                             if not res.explore[i] and res.calls[i] > 0), 3)
                                       if any(not res.explore[i] and res.calls[i] > 0 for i in range(len(correct))) else None),
                   grad_cos=round(statistics.mean(cosines), 4) if cosines else None,
                   tool_rate=round(statistics.mean(c > 0 for c in res.calls), 3),
                   post_tool_words=post_tool_stats(res.texts)[0], retry_after_error=post_tool_stats(res.texts)[1],
                   tool_calls=round(statistics.mean(res.calls), 3),
                   backtrack_rate=(round(statistics.mean(b > 0 for b in res.backtracks), 3) if getattr(res, "backtracks", None) else None),
                   acc_with_tool=(round(statistics.mean(float(correct[i]) for i in range(len(correct)) if res.calls[i] > 0), 3)
                                  if any(res.calls) else None),
                   acc_without_tool=(round(statistics.mean(float(correct[i]) for i in range(len(correct)) if res.calls[i] == 0), 3)
                                     if not all(res.calls) else None),
                   behaviours={k: round(v, 3) for k, v in beh.items()},
                   buckets=[it["bucket"] for it in items])
        with open(f"{args.run}/log.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")
        print(f"step {step:5d} | {t_step:5.0f}s (gen {t_gen:.0f}) | R {log['reward']:.3f} | math {log['acc_math']} "
              f"code {log['acc_code']} | kept {kept_groups}/{args.prompts} | kl {kl_v:.4f} | tok {log['tokens']:.0f} | " +
              (f"OPEN spread {pr_stats.get('open_spread')} len {pr_stats.get('open_len')} | " if "open_spread" in pr_stats else "") + (f"BRANCH traces {pr_stats.get('pr_traces')} contacc {pr_stats.get('pr_cont_acc')} gain {pr_stats.get('pr_gain')} rescued {pr_stats.get('pr_rescued')}/{pr_stats.get('pr_allfail')} seqs {len(seqs)} | " if pr_stats.get("pr_traces") is not None else "") +
              f"plain {seed_stats.get('plain_acc')} cond {seed_stats.get('cond_acc')} uplift {seed_stats.get('plan_uplift')} lte {seed_stats.get('lte_correct')}/{seed_stats.get('lte_added')} rev {seed_stats.get('revise_rate')} | ent {entropy if entropy is None else round(entropy, 3)} | mem {peak:.1f}GB | " + " ".join(f"{k[:4]} {v:.2f}" for k, v in beh.items()), flush=True)

        mx.reset_peak_memory()
        if args.max_entropy and entropy is not None:
            HIGH_ENT[0] = HIGH_ENT[0] + 1 if entropy > args.max_entropy else 0
            if HIGH_ENT[0] >= 2:
                print(f"=== ENTROPY GUARD: entropy {entropy:.2f} > {args.max_entropy} for 2 steps at step {step} - aborting (last good checkpoint kept) ===", flush=True)
                sys.exit(3)
        if step % args.sample_every == 0:
            with open(f"{args.run}/samples.jsonl", "a") as f:
                for i in random.sample(range(len(allp)), 8):
                    f.write(json.dumps(dict(step=step, bucket=items[i // args.group]["bucket"],
                                            question=items[i // args.group]["question"][:2000],
                                            output=(res.log_texts[i] if getattr(res, "log_texts", None) else res.texts[i]),
                                            reward=rewards[i], why=rw[i][1], tool_calls=res.calls[i],
                                            backtracks=(res.backtracks[i] if getattr(res, "backtracks", None) else 0))) + "\n")
        if args.max_tokens_cap and args.max_tokens < args.max_tokens_cap:
            TRUNC_HIST.append(log["truncated"])
            if (len(TRUNC_HIST) >= 20 and step - STAGE[0] >= args.stage_min_steps
                    and statistics.mean(TRUNC_HIST[-20:]) > args.stage_trunc):
                new = min(args.max_tokens_cap, {768: 1024, 1024: 1536}.get(args.max_tokens, int(args.max_tokens * 1.5)))
                print(f"=== staged budget: max_tokens {args.max_tokens} -> {new} (truncation {statistics.mean(TRUNC_HIST[-20:]):.3f}) ===", flush=True)
                args.max_tokens = new
                STAGE[0] = step
                TRUNC_HIST.clear()
        if step % args.save_every == 0 or step == args.steps:
            mx.save_safetensors(f"{args.run}/adapters.safetensors", dict(tree_flatten(model.trainable_parameters())))
            mx.save_safetensors(f"{args.run}/optimizer.safetensors", dict(tree_flatten(opt.state)))
            if copt is not None:
                mx.save_safetensors(f"{args.run}/optimizer_critic.safetensors", dict(tree_flatten(copt.state)))
            json.dump(dict(step=step, curriculum=cur.p, max_tokens=args.max_tokens), open(state_path, "w"))
        if step % args.eval_every == 0:
            # keep a snapshot per eval so a degraded run can be rolled back to its best checkpoint
            os.makedirs(f"{args.run}/snapshots", exist_ok=True)
            mx.save_safetensors(f"{args.run}/snapshots/adapters_step{step}.safetensors",
                                dict(tree_flatten(model.trainable_parameters())))
            evaluate(judge, step)
