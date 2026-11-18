# LLM Reasoning via RL (local, small models)

Reinforcement learning to make a small LLM (Qwen3-1.7B) reason better, trained entirely on one
Apple Silicon laptop (M1 Pro, 16 GB) with [MLX](https://github.com/ml-explore/mlx). No SFT, no distillation:
the model only learns from its own rollouts and a verifiable reward.

## Results

All numbers are paired comparisons on the same never-trained problems (7 unseen reasoning-gym tasks × 40 problems × 8 samples, MATH-500, GSM8K, general questions).

| Model | How it was trained | Unseen maj@8 | Unseen pass@8 | MATH-500 |
|---|---|---|---|---|
| Base | Qwen3-1.7B-Base, untrained | – | – | – |
| **Z** | zero-RL: GRPO from the base model | 24.6 | 55.7 | 54.5 |
| **B** | + Monte-Carlo progress credit (branch mid-reasoning) | 28.2 | 55.4 | 62.0* |
| **O3** | + opening credit on unsolved problems only | **31.8*** | 55.7 | **61.5*** |

\* significant vs Z (paired bootstrap 95% CI). Zero-RL itself took held-out logic/math accuracy from 14% → 64%.

Key findings:
- **Baseline CoT was non-causal.** Continuing from 75% of a *correct* chain-of-thought succeeded no more often than starting fresh (V-curve, `eval/vcurve.py`).
- **Progress credit** (advantage = how much a reasoning segment changes the Monte-Carlo success probability) is safe and gives a small real gain.
- **Opening credit everywhere collapses diversity** (pass@8 −9 pts); restricting it to problems the model is stuck on keeps the gain without the collapse.
- Token-level fork detection (entropy, Rao quadratic entropy) did **not** beat random positions.
- Tool use (Python + z3 sandbox) and a checkpoint/backtrack action: the model learns to call code more, but accuracy with tools did not improve within 40 steps.

## How it works

`train_grpo.py` is a single-file trainer:
- **Algorithm:** GRPO with Dr. GRPO normalisation, DAPO clip-higher and minibatches, optional KL to the start model.
- **Model:** LoRA (rank 32) on all attention and MLP projections. The reference model is the same weights with LoRA switched off, so only one copy fits in memory.
- **Prompt:** R1-Zero plain-text template; reward = correct `<answer>`.
- **Data:** [reasoning-gym](https://github.com/open-thought/reasoning-gym) tasks plus about 300k general verifiable problems.
- **Credit options:** `--progress` (branch rollouts at reasoning prefixes) and `--open_tree` / `--open_max_v` (credit openings on stuck problems).
- **Tools:** `--tools --backtrack` adds a sandboxed Python executor (Podman) and a `<checkpoint>` / `<backtrack>` context-rewind action.

## Layout

```
train_grpo.py    trainer (all algorithms behind flags)
tools.py         generation with tool calls + backtracking
common.py        prompts, answer extraction, grading
fuse_run.py      merge a run's LoRA into a standalone model
server.py        live dashboard + chat   (dashboard.html)
chat.py          terminal chat UI;  think_chat.py = chat with a minimum thinking budget
eval/            evaluations: evalbig (paired), narrow_eval, vcurve, forkscore, diagnose, ...
scripts/         dataset download / build
pipelines/       the exact shell pipelines used for each run (B, O3, G, T, ...)
sandbox/         Podman image + judge for running model-written code
results/         eval outputs (small JSON / txt)
archive/         abandoned experiments (SFT, RLP)
FUTURE_FEATURES.md  research notes and ideas
```
`models/`, `data/`, `runs/` and `logs/` are created locally and not committed.

## Recreate

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# models (MLX format)
python -m mlx_lm convert --hf-path Qwen/Qwen3-1.7B-Base --mlx-path models/Qwen3-1.7B-Base
python -m mlx_lm convert --hf-path Qwen/Qwen3-1.7B-Base --mlx-path models/Qwen3-1.7B-Base-4bit -q   # for the dashboard

# data
python scripts/download_data.py
python scripts/build_general.py

# optional: code sandbox (needed for --tools and code tasks)
podman machine init --memory 1536 && podman machine start
podman build -t llmrl-sandbox -f sandbox/Containerfile sandbox
```

Train (run everything from the project root):
```bash
# zero-RL from the base model (run Z)
python train_grpo.py --run runs/Z --model models/Qwen3-1.7B-Base --template r1zero --algo grpo \
  --tasks countdown,chain_sum,... --steps 180 --lr 1e-5

# the best recipe (O3) and the others: see pipelines/pipeline_B.sh and pipelines/pipeline_O3.sh
zsh pipelines/pipeline_O3.sh > logs/pipeline_O3.log 2>&1
```

Evaluate, watch, chat:
```bash
python fuse_run.py runs/O3 models/Qwen3-1.7B-O3                    # merge LoRA
python eval/evalbig.py run models/Qwen3-1.7B-O3 O3                 # paired big eval
python eval/evalbig.py compare Z O3                                # CIs vs another model
python server.py --run runs/O3 --template r1zero --model models/Qwen3-1.7B-Base-4bit   # http://localhost:8765
python think_chat.py --model models/Qwen3-1.7B-O3 --min 256        # chat, forced 256+ thinking tokens
```

Memory tips for 16 GB: keep `--gen_batch` at 16 or less and `--mem_limit_gb` around 6. Run one training job at a time, and stop the Podman VM when it isn't needed.
