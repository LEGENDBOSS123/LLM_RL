#!/bin/zsh
# Run O: tree over openings, continuing from run B's model.
# Why: B's V-curve shows most wrong answers go wrong in the first 25% (the opening approach), and the fork test's
# largest single-token forks were opening phrases. Per problem: 4 different openings (first sentence) x 4 continuations;
# opening tokens get V(opening) - V(problem), the rest get r - V(opening). No extra generation vs plain GRPO.
cd /Users/vedantarora/Terminal/LLM_RL
source .venv/bin/activate
log() { echo "[$(date +%H:%M)] $*"; }
F="zoxide|Please ensure|If the issue|ajeetdsouza|_ZO_DOCTOR|^$"

until grep -q PIPELINE_DONE pipeline_B2.log; do sleep 60; done
TASKS=countdown,simple_equations,puzzle24,n_queens,futoshiki,kakurasu,survo,jugs,cryptarithm,mini_sudoku,knights_knaves,zebra_puzzles,polynomial_equations,shortest_path,maze,tower_of_hanoi,chain_sum,base_conversion,gcd,lcm,prime_factorization,fraction_simplification,calendar_arithmetic,time_intervals,decimal_arithmetic,products,bitwise_arithmetic,count_primes,syllogism,rotate_matrix,spell_backward,basic_arithmetic,number_sorting
log "STAGE O3: opening credit only on stuck problems (acc<=0.3), lr 5e-6, KL 0.01 from Qwen3-1.7B-B"
rm -rf runs/O3 && mkdir -p runs/O3
ARGS=(--run runs/O3 --open_tree 4 --open_len 48 --open_max_v 0.3
  --model models/Qwen3-1.7B-B --template r1zero --algo grpo --minibatches 4 --clip_low 0.2 --clip_high 0.28
  --skip_eval0 --steps 60 --lr 5e-6 --beta 0.01 --max_entropy 2.0 --len_alpha 0 --max_tokens 768 --prompts 6 --group 16 --stratified
  --tasks $TASKS --code_frac 0 --gen_frac 0.3 --eval_every 30 --eval_math_n 10 --eval_code_n 0 --eval_gen_n 120
  --save_every 5 --sample_every 10 --gen_batch 12 --mem_limit_gb 6 --judge_workers 2 --top_p 0.95 --temp_ladder 0.9,1.2
  --overlong_buffer 96 --overlong_penalty 0.2)
python train_grpo.py $ARGS > runs/O3/train.out 2>&1
rc=$?
for try in 1 2 3; do
  [[ $rc -eq 0 || $rc -eq 3 ]] && break
  log "run O exited $rc - resuming (try $try)"
  python train_grpo.py $ARGS --resume --skip_eval0 >> runs/O3/train.out 2>&1
  rc=$?
done
log "run O finished rc=$rc"
rm -rf models/Qwen3-1.7B-O3 && python fuse_run.py runs/O3 models/Qwen3-1.7B-O3 2>&1 | tail -1
log "STAGE O-eval: big paired eval + V-curve"
python eval/evalbig.py run models/Qwen3-1.7B-O3 O3 2>&1 | grep -vE "$F"
python eval/evalbig.py compare Z B O3 2>&1 | grep -vE "$F" | tee results/evalbig_Z_B_O3.txt
cp results/vcurve_rows.jsonl results/vcurve_rows_B2.jsonl
python eval/vcurve.py models/Qwen3-1.7B-O3 5 2>&1 | grep -vE "$F" | tee results/vcurve_O3.txt
log "PIPELINE_DONE"
