#!/bin/zsh
# Context-distillation SFT: data/sft -> models/Qwen2.5-1.5B-sft (bf16) + models/Qwen2.5-1.5B-sft-4bit
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

# 1. strict re-filter with the exact RL format check (one <think>..</think> then one <answer>..</answer>, nothing else)
python - <<'EOF'
import json, re
FMT = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.S)
def ok(t):
    return (t.count("<think>") == 1 and t.count("</think>") == 1 and t.count("<answer>") == 1
            and t.count("</answer>") == 1 and FMT.fullmatch(t) is not None)
for split in ("train", "valid"):
    rows = [json.loads(l) for l in open(f"data/sft/{split}.jsonl")]
    keep = [r for r in rows if ok(r["messages"][-1]["content"])]
    with open(f"data/sft/{split}.jsonl", "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    print(f"{split}: {len(rows)} -> {len(keep)} after strict format filter")
EOF

N=$(wc -l < data/sft/train.jsonl)
ITERS=$(( N * 3 / 4 ))          # ~3 epochs (iters count micro-batches of 4)
[ $ITERS -lt 60 ] && ITERS=60
echo "SFT: $N examples, $ITERS iters"

# 2. LoRA SFT (prompt masked: only the assistant output is learned)
rm -rf adapters/sft
PYTHONUNBUFFERED=1 nice -n 10 python -m mlx_lm.lora --model models/Qwen2.5-1.5B-Instruct-bf16 --train --data data/sft \
  --fine-tune-type lora --num-layers -1 --batch-size 4 --grad-accumulation-steps 2 --iters $ITERS \
  --learning-rate 2e-5 --mask-prompt --grad-checkpoint --max-seq-length 1024 \
  --steps-per-report 10 --steps-per-eval 24 --val-batches 10 --save-every 1000 --adapter-path adapters/sft \
  2>&1 | tee data/sft_lora.log

# 2b. abort if training diverged
python - <<'PYEOF'
import re, sys
log = re.sub(r"\x1b\[[0-9;]*m", "", open("data/sft_lora.log").read())
vals = [float(v) for v in re.findall(r"val ([0-9.]+)", log)]
print("val losses:", vals)
if not vals or vals[-1] > 1.0 or vals[-1] > 2 * min(vals):
    sys.exit("SFT diverged - not fusing")
PYEOF

# 3. merge into a new base model, plus a 4-bit copy for the dashboard chat
rm -rf models/Qwen2.5-1.5B-sft models/Qwen2.5-1.5B-sft-4bit
python -m mlx_lm.fuse --model models/Qwen2.5-1.5B-Instruct-bf16 --adapter-path adapters/sft --save-path models/Qwen2.5-1.5B-sft
python -m mlx_lm.convert --hf-path models/Qwen2.5-1.5B-sft --mlx-path models/Qwen2.5-1.5B-sft-4bit -q --q-bits 4
# 4. sanity: model must produce <think>..</think><answer>..</answer> with no system prompt
python - <<'PYEOF'
from mlx_lm import load, generate
m, t = load("models/Qwen2.5-1.5B-sft")
for q in ["Calculate 17 * 23 - 45.", "Sort these numbers in ascending order: 5, -2, 9, 0", "Hi! How are you today?"]:
    p = t.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
    print("Q:", q, "\n", generate(m, t, p, max_tokens=300), "\n----")
PYEOF
echo SFT_DONE
