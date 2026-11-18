#!/bin/zsh
# Run B: Monte-Carlo progress credit ("branching") on top of run Z.
#  - 6 problems x 12 answers; 12 traces/step branched at 33%/67% of their reasoning, 4 continuations per cut
#  - token advantage = V(next cut) - V(this cut)  (telescopes to the usual r - mean, but credits the segment that made progress)
#  - continuations also trained (adv = r - V(cut)); all-fail groups can be rescued from good prefixes
# After training: fuse -> diagnose (pass@1 / maj@8 / pass@8 on unseen) -> V-curve (did reasoning become causal?)
cd /Users/vedantarora/Terminal/LLM_RL
source .venv/bin/activate
log() { echo "[$(date +%H:%M)] $*"; }
TASKS=countdown,simple_equations,puzzle24,n_queens,futoshiki,kakurasu,survo,jugs,cryptarithm,mini_sudoku,knights_knaves,zebra_puzzles,polynomial_equations,shortest_path,maze,tower_of_hanoi,chain_sum,base_conversion,gcd,lcm,prime_factorization,fraction_simplification,calendar_arithmetic,time_intervals,decimal_arithmetic,products,bitwise_arithmetic,count_primes,syllogism,rotate_matrix,spell_backward,basic_arithmetic,number_sorting
until grep -q PIPELINE_DONE pipeline_O2.log; do sleep 60; done
log "STAGE B2: progress credit continued from B (6 conts per cut)"
rm -rf runs/B2 && mkdir -p runs/B2
ARGS=(--run runs/B2 --progress 12 --pr_cuts 0.33,0.67 --pr_conts 6 --pr_train_conts 1
  --model models/Qwen3-1.7B-B --template r1zero --algo grpo --minibatches 4 --clip_low 0.2 --clip_high 0.28
  --skip_eval0 --steps 60 --lr 1e-5 --beta 0 --max_entropy 2.0 --len_alpha 0 --max_tokens 768 --prompts 6 --group 12 --stratified
  --tasks $TASKS --code_frac 0 --gen_frac 0.3 --eval_every 30 --eval_math_n 10 --eval_code_n 0 --eval_gen_n 120
  --save_every 5 --sample_every 10 --gen_batch 12 --mem_limit_gb 6 --judge_workers 2 --top_p 0.95 --temp_ladder 0.9,1.2
  --overlong_buffer 96 --overlong_penalty 0.2)
python train_grpo.py $ARGS > runs/B2/train.out 2>&1
rc=$?
for try in 1 2 3; do
  [[ $rc -eq 0 || $rc -eq 3 ]] && break
  log "run B2 exited $rc - resuming (try $try)"
  python train_grpo.py $ARGS --resume --skip_eval0 >> runs/B2/train.out 2>&1
  rc=$?
done
log "run B2 finished rc=$rc"
F="zoxide|Please ensure|If the issue|ajeetdsouza|_ZO_DOCTOR|^$"
rm -rf models/Qwen3-1.7B-B2 && python fuse_run.py runs/B2 models/Qwen3-1.7B-B2 2>&1 | tail -1
python eval/evalbig.py run models/Qwen3-1.7B-B2 B2 2>&1 | grep -vE "$F"
python eval/evalbig.py compare Z B B2 2>&1 | grep -vE "$F" | tee results/evalbig_Z_B_B2.txt
python eval/vcurve.py models/Qwen3-1.7B-B2 5 2>&1 | grep -vE "$F" | tee results/vcurve_B2.txt
log "PIPELINE_DONE"
