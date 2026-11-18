#!/bin/zsh
# Run O: tree over openings, continuing from run B's model.
# Why: B's V-curve shows most wrong answers go wrong in the first 25% (the opening approach), and the fork test's
# largest single-token forks were opening phrases. Per problem: 4 different openings (first sentence) x 4 continuations;
# opening tokens get V(opening) - V(problem), the rest get r - V(opening). No extra generation vs plain GRPO.
cd /Users/vedantarora/Terminal/LLM_RL
source .venv/bin/activate
log() { echo "[$(date +%H:%M)] $*"; }
F="zoxide|Please ensure|If the issue|ajeetdsouza|_ZO_DOCTOR|^$"
until grep -q EVALBIG_DONE queue_evalbig.log; do sleep 60; done
TASKS=countdown,simple_equations,puzzle24,n_queens,futoshiki,kakurasu,survo,jugs,cryptarithm,mini_sudoku,knights_knaves,zebra_puzzles,polynomial_equations,shortest_path,maze,tower_of_hanoi,chain_sum,base_conversion,gcd,lcm,prime_factorization,fraction_simplification,calendar_arithmetic,time_intervals,decimal_arithmetic,products,bitwise_arithmetic,count_primes,syllogism,rotate_matrix,spell_backward,basic_arithmetic,number_sorting
log "STAGE O: opening-tree credit from Qwen3-1.7B-B"
rm -rf runs/O && mkdir -p runs/O
ARGS=(--run runs/O --open_tree 4 --open_len 48
  --model models/Qwen3-1.7B-B --template r1zero --algo grpo --minibatches 4 --clip_low 0.2 --clip_high 0.28
  --steps 60 --lr 1e-5 --beta 0 --max_entropy 2.0 --len_alpha 0 --max_tokens 768 --prompts 6 --group 16 --stratified
  --tasks $TASKS --code_frac 0 --gen_frac 0.3 --eval_every 30 --eval_math_n 10 --eval_code_n 0 --eval_gen_n 120
  --save_every 5 --sample_every 10 --gen_batch 12 --mem_limit_gb 6 --judge_workers 2 --top_p 0.95 --temp_ladder 0.9,1.2
  --overlong_buffer 96 --overlong_penalty 0.2)
python train_grpo.py $ARGS > runs/O/train.out 2>&1
rc=$?
for try in 1 2 3; do
  [[ $rc -eq 0 || $rc -eq 3 ]] && break
  log "run O exited $rc - resuming (try $try)"
  python train_grpo.py $ARGS --resume --skip_eval0 >> runs/O/train.out 2>&1
  rc=$?
done
log "run O finished rc=$rc"
rm -rf models/Qwen3-1.7B-O && python fuse_run.py runs/O models/Qwen3-1.7B-O 2>&1 | tail -1
log "STAGE O-eval: big paired eval + V-curve"
python eval/evalbig.py run models/Qwen3-1.7B-O O 2>&1 | grep -vE "$F"
python eval/evalbig.py compare Z B O 2>&1 | grep -vE "$F" | tee results/evalbig_Z_B_O.txt
cp results/vcurve_rows.jsonl results/vcurve_rows_B.jsonl
python eval/vcurve.py models/Qwen3-1.7B-O 5 2>&1 | grep -vE "$F" | tee results/vcurve_O.txt
log "PIPELINE_DONE"
