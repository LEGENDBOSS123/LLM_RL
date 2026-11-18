#!/bin/zsh
cd /Users/vedantarora/Terminal/LLM_RL
source .venv/bin/activate
log() { echo "[$(date +%H:%M)] $*"; }
TASKS=countdown,simple_equations,puzzle24,n_queens,futoshiki,kakurasu,survo,jugs,cryptarithm,mini_sudoku,knights_knaves,zebra_puzzles,polynomial_equations,shortest_path,maze,tower_of_hanoi,chain_sum,base_conversion,gcd,lcm,prime_factorization,fraction_simplification,calendar_arithmetic,time_intervals,decimal_arithmetic,products,bitwise_arithmetic,count_primes,syllogism,rotate_matrix,spell_backward,basic_arithmetic,number_sorting
log "STAGE M: run M = proven GRPO recipe (run F settings) + 4 learned modes + ensemble-contribution reward"
rm -rf runs/M && mkdir -p runs/M
python train_grpo.py --run runs/M --modes 4 --mode_gamma 0.5 --mode_delta 0.1 \
  --model models/Qwen3-1.7B-RL --template r1zero --algo grpo --minibatches 4 --clip_low 0.2 --clip_high 0.28 \
  --steps 60 --lr 1e-5 --beta 0 --max_entropy 2.0 --len_alpha 0 --max_tokens 768 --prompts 6 --group 16 --stratified \
  --tasks $TASKS --code_frac 0.2 --gen_frac 0.4 --eval_every 30 --eval_math_n 10 --eval_code_n 20 --eval_gen_n 120 \
  --save_every 10 --sample_every 10 --gen_batch 16 --mem_limit_gb 7 --judge_workers 2 --top_p 0.95 --temp_ladder 0.9,1.2 \
  --overlong_buffer 96 --overlong_penalty 0.2 > runs/M/train.out 2>&1 || log "run M exited non-zero"
log "STAGE M-diag: fuse + diagnose M (plain vs cycling 4 modes)"
rm -rf models/Qwen3-1.7B-M && python fuse_run.py runs/M models/Qwen3-1.7B-M 2>&1 | tail -1
python eval/diagnose.py models/Qwen3-1.7B-M --modes 4 2>&1 | grep -vE "zoxide|Please ensure|If the issue|ajeetdsouza|_ZO_DOCTOR|^$" | tee results/diag_M.txt
log "PIPELINE_DONE"
