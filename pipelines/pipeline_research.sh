#!/bin/zsh
# Research pipeline: baseline diag -> run W (W-REINFORCE) -> diag -> run M (W-REINFORCE + learned modes) -> diag
cd /Users/vedantarora/Terminal/LLM_RL
source .venv/bin/activate
log() { echo "[$(date +%H:%M)] $*"; }
TASKS=countdown,simple_equations,puzzle24,n_queens,futoshiki,kakurasu,survo,jugs,cryptarithm,mini_sudoku,knights_knaves,zebra_puzzles,polynomial_equations,shortest_path,maze,tower_of_hanoi,chain_sum,base_conversion,gcd,lcm,prime_factorization,fraction_simplification,calendar_arithmetic,time_intervals,decimal_arithmetic,products,bitwise_arithmetic,count_primes,syllogism,rotate_matrix,spell_backward,basic_arithmetic,number_sorting
COMMON=(--model models/Qwen3-1.7B-RL --template r1zero --algo grpo --minibatches 2 --clip_low 0.2 --clip_high 0.28 --steps 60 --lr 3e-6 --beta 0.02 --max_entropy 2.0 --len_alpha 0 --max_tokens 768 --prompts 6 --group 16 --stratified --tasks $TASKS --code_frac 0.2 --gen_frac 0.4 --eval_every 30 --eval_math_n 10 --eval_code_n 20 --eval_gen_n 120 --save_every 10 --sample_every 10 --gen_batch 16 --mem_limit_gb 7 --judge_workers 2 --top_p 0.95 --temp_ladder 0.9,1.2 --overlong_buffer 96 --overlong_penalty 0.2)

sleep 1
log "STAGE 1: baseline already in results/diag_Z.txt (skipped)"

log "STAGE 2: run W2 resumed from step 10 with KL 0.02 (entropy drifted to 1.2 at KL 0.005)"
python3 - <<'PY'
import json
st = json.load(open("runs/W/state.json"))["step"]
for f in ("runs/W/log.jsonl", "runs/W/samples.jsonl"):
    rows = [l for l in open(f) if json.loads(l)["step"] <= st]
    open(f, "w").writelines(rows)
PY
echo "=== resume step 10, KL 0.005 -> 0.02 ===" >> runs/W/train.out
python train_grpo.py --run runs/W --resume --wreinforce 0.1 "${COMMON[@]}" >> runs/W/train.out 2>&1 || log "run W exited non-zero"
log "STAGE 3: fuse + diagnose W"
rm -rf models/Qwen3-1.7B-W && python fuse_run.py runs/W models/Qwen3-1.7B-W 2>&1 | tail -1
python eval/diagnose.py models/Qwen3-1.7B-W 2>&1 | grep -vE "zoxide|Please ensure|If the issue|ajeetdsouza|_ZO_DOCTOR|^$" | tee results/diag_W.txt

log "STAGE 4: run M (W-REINFORCE + 4 learned modes + ensemble-contribution reward)"
rm -rf runs/M && mkdir -p runs/M
python train_grpo.py --run runs/M --wreinforce 0.1 --modes 4 --mode_gamma 0.5 --mode_delta 0.1 "${COMMON[@]}" > runs/M/train.out 2>&1 || log "run M exited non-zero"
log "STAGE 5: fuse + diagnose M (plain vs cycling modes)"
rm -rf models/Qwen3-1.7B-M && python fuse_run.py runs/M models/Qwen3-1.7B-M 2>&1 | tail -1
python eval/diagnose.py models/Qwen3-1.7B-M --modes 4 2>&1 | grep -vE "zoxide|Please ensure|If the issue|ajeetdsouza|_ZO_DOCTOR|^$" | tee results/diag_M.txt
log "PIPELINE_DONE"
