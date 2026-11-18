#!/bin/zsh
cd /Users/vedantarora/Terminal/LLM_RL
log() { echo "[$(date +%H:%M)] $*"; }
until grep -q "^\[eval step 180\]" runs/Z/train.out; do sleep 15; done
log "$(grep '^\[eval step 180\]' runs/Z/train.out)"
pkill -f "train_grpo.py --run runs/Z"; pkill -f "server.py --run"; sleep 5
source .venv/bin/activate
mkdir -p runs/Zeval
COMMON=(--model models/Qwen3-1.7B-Base --template r1zero --algo grpo --code_frac 0.25 --tasks chain_sum,base_conversion,gcd,number_sorting,basic_arithmetic --eval_math_n 25 --eval_code_n 50 --max_tokens 640 --gen_batch 16 --mem_limit_gb 7 --judge_workers 2)
for s in 120 180; do
  log "precise eval of step $s"
  nice -n 10 python train_grpo.py --run runs/Zeval "${COMMON[@]}" --eval_only runs/Z/snapshots/adapters_step$s.safetensors --eval_tag $s 2>&1 | grep -E "^\[eval|Traceback|Error"
done
BEST=$(python3 -c "
import json
E={e['step']:e for e in map(json.loads, open('runs/Zeval/eval.jsonl')) if e['step'] in (120,180)}
print(max(E, key=lambda s: E[s]['math'] + E[s]['code']))")
log "best checkpoint: step $BEST"
rm -rf models/Qwen3-1.7B-RL models/Qwen3-1.7B-RL-4bit
python fuse_run.py runs/Z models/Qwen3-1.7B-RL runs/Z/snapshots/adapters_step$BEST.safetensors
python -m mlx_lm convert --hf-path models/Qwen3-1.7B-RL --mlx-path models/Qwen3-1.7B-RL-4bit -q --q-bits 4 2>&1 | tail -1
mkdir -p runs/Zfinal && cp runs/Z/config.json runs/Zfinal/ && cp runs/Z/eval.jsonl runs/Z/log.jsonl runs/Zfinal/ 2>/dev/null
nohup nice -n 5 python server.py --run runs/Zfinal --template r1zero --model models/Qwen3-1.7B-RL-4bit > runs/Zfinal/server.out 2>&1 &
log "FINALIZE_DONE best=$BEST"
