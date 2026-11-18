#!/bin/zsh
# Run P: pass@k training (Chen et al. 2025) on top of run Z - reward "any of k correct" to keep exploration.
cd /Users/vedantarora/Terminal/LLM_RL
source .venv/bin/activate
log() { echo "[$(date +%H:%M)] $*"; }
TASKS=countdown,simple_equations,puzzle24,n_queens,futoshiki,kakurasu,survo,jugs,cryptarithm,mini_sudoku,knights_knaves,zebra_puzzles,polynomial_equations,shortest_path,maze,tower_of_hanoi,chain_sum,base_conversion,gcd,lcm,prime_factorization,fraction_simplification,calendar_arithmetic,time_intervals,decimal_arithmetic,products,bitwise_arithmetic,count_primes,syllogism,rotate_matrix,spell_backward,basic_arithmetic,number_sorting
log "STAGE P: pass@4 advantage GRPO from Qwen3-1.7B-RL"
rm -rf runs/P && mkdir -p runs/P
ARGS=(--run runs/P --passk 4 --model models/Qwen3-1.7B-RL --template r1zero --algo grpo --minibatches 4
  --clip_low 0.2 --clip_high 0.28 --steps 60 --lr 1e-5 --beta 0 --max_entropy 2.0 --len_alpha 0 --max_tokens 768
  --prompts 6 --group 16 --stratified --tasks $TASKS --code_frac 0.2 --gen_frac 0.4 --eval_every 30
  --eval_math_n 10 --eval_code_n 20 --eval_gen_n 120 --save_every 5 --sample_every 10 --gen_batch 12
  --mem_limit_gb 6 --judge_workers 2 --top_p 0.95 --temp_ladder 0.9,1.2 --overlong_buffer 96 --overlong_penalty 0.2)
python train_grpo.py $ARGS > runs/P/train.out 2>&1
rc=$?
for try in 1 2 3; do
  [[ $rc -eq 0 || $rc -eq 3 ]] && break
  log "run P exited $rc - resuming (try $try)"
  python train_grpo.py $ARGS --resume --skip_eval0 >> runs/P/train.out 2>&1
  rc=$?
done
log "run P finished rc=$rc"
log "STAGE P-diag: fuse + diagnose"
rm -rf models/Qwen3-1.7B-P && python fuse_run.py runs/P models/Qwen3-1.7B-P 2>&1 | tail -1
python eval/diagnose.py models/Qwen3-1.7B-P 2>&1 | grep -vE "zoxide|Please ensure|If the issue|ajeetdsouza|_ZO_DOCTOR|^$" | tee results/diag_P.txt
log "PIPELINE_DONE"
