#!/bin/zsh
# step-40 eval -> stop S -> fuse RL weights -> tool traces -> masked SFT -> 4bit -> phase 3 RL (tools + code)
set -e
cd /Users/vedantarora/Terminal/LLM_RL
source .venv/bin/activate
log() { echo "[$(date +%H:%M)] $*"; }
until grep -q "^\[eval step 40\]" runs/S/train.out; do sleep 15; done
log "eval 40 done: $(grep '^\[eval step 40\]' runs/S/train.out)"
pkill -f "train_grpo.py --run runs/S" || true; sleep 5
# RL step 40 degraded held-out accuracy (56.5% -> 45.5% at lr 2e-5) -> build on the clean SFT model instead
BASE=models/Qwen2.5-1.5B-sft
log "generating tool traces"
nice -n 10 python archive/sft_tools_data.py $BASE 12 60 2>&1 | grep -v "Finished processing"
log "masked SFT"
rm -rf models/Qwen2.5-1.5B-tools models/Qwen2.5-1.5B-tools-4bit
nice -n 10 python archive/sft_masked.py $BASE data/sft_tools models/Qwen2.5-1.5B-tools 2 5e-5
python -m mlx_lm convert --hf-path models/Qwen2.5-1.5B-tools --mlx-path models/Qwen2.5-1.5B-tools-4bit -q --q-bits 4 2>&1 | tail -1
log "starting phase 3 (runs/T)"
TASKS=chain_sum,base_conversion,gcd,number_sorting,basic_arithmetic,lcm,prime_factorization,decimal_arithmetic,fraction_simplification,time_intervals,calendar_arithmetic,knights_knaves,zebra_puzzles,mini_sudoku,shortest_path,maze,tower_of_hanoi,cryptarithm,products,power_function,bitwise_arithmetic,rotate_matrix,syllogism,count_primes,polynomial_equations,spell_backward,decimal_chain_sum
rm -rf runs/T && mkdir -p runs/T
pkill -f "server.py --run" || true
nohup caffeinate -i nice -n 10 python train_grpo.py --run runs/T --model models/Qwen2.5-1.5B-tools --tools \
  --steps 2000 --lr 1e-5 --beta 0.001 --len_cost 0 --len_alpha 0.1 --len_gate 0.5 --max_tokens 1024 \
  --prompts 4 --group 16 --stratified --nothink_eval --tasks $TASKS --code_frac 0.25 \
  --eval_every 20 --eval_math_n 25 --eval_code_n 50 --save_every 10 --sample_every 10 > runs/T/train.out 2>&1 &
sleep 30
nohup nice -n 5 python server.py --run runs/T --tools --model models/Qwen2.5-1.5B-tools-4bit > runs/T/server.out 2>&1 &
log "PIPELINE_DONE"
