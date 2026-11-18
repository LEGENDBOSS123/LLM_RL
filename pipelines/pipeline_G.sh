#!/bin/zsh
# Run G: NARROW + LONG ("grokking" setup from the RL-grokking / ProRL papers) with B's progress-credit branching.
# 4 hard search/constraint families where B is weak (~13-18% train acc): countdown, puzzle24, n_queens, mini_sudoku.
# Question: does long focused RL unlock a family (pass@8 jumps), and does anything transfer to unseen tasks / MATH?
cd /Users/vedantarora/Terminal/LLM_RL
source .venv/bin/activate
log() { echo "[$(date +%H:%M)] $*"; }
F="zoxide|Please ensure|If the issue|ajeetdsouza|_ZO_DOCTOR|^$"
NT=countdown,puzzle24,n_queens,mini_sudoku
TT="countdown,puzzle24,mini_sudoku,n_queens,n_queens@n=4;max_remove=2,n_queens@n=6;max_remove=3"
log "STAGE G: narrow long training from O3 (progress credit + n_queens easy->hard ladder)"
rm -rf runs/G && mkdir -p runs/G
ARGS=(--run runs/G --progress 8 --pr_cuts 0.33,0.67 --pr_conts 4 --pr_train_conts 1
  --model models/Qwen3-1.7B-O3 --template r1zero --algo grpo --minibatches 4 --clip_low 0.2 --clip_high 0.28
  --skip_eval0 --steps 70 --lr 1e-5 --beta 0 --max_entropy 2.0 --len_alpha 0 --max_tokens 768 --prompts 6 --group 16 --stratified
  --tasks "$TT" --code_frac 0 --gen_frac 0 --eval_every 35 --eval_math_n 10 --eval_code_n 0 --eval_gen_n 120
  --save_every 5 --sample_every 10 --gen_batch 12 --mem_limit_gb 6 --judge_workers 2 --top_p 0.95 --temp_ladder 0.9,1.2
  --overlong_buffer 96 --overlong_penalty 0.2)
python train_grpo.py $ARGS > runs/G/train.out 2>&1
rc=$?
for try in 1 2 3; do
  [[ $rc -eq 0 || $rc -eq 3 ]] && break
  log "run G exited $rc - resuming (try $try)"
  python train_grpo.py $ARGS --resume --skip_eval0 >> runs/G/train.out 2>&1
  rc=$?
done
log "run G finished rc=$rc"
rm -rf models/Qwen3-1.7B-G && python fuse_run.py runs/G models/Qwen3-1.7B-G 2>&1 | tail -1
log "STAGE G-eval"
python eval/narrow_eval.py run models/Qwen3-1.7B-G G $NT 30 2>&1 | grep -vE "$F"
python eval/narrow_eval.py compare O3 G 2>&1 | grep -vE "$F" | tee results/narrow_O3_vs_G.txt
python eval/evalbig.py run models/Qwen3-1.7B-G G 2>&1 | grep -vE "$F"
python eval/evalbig.py compare Z B O3 G 2>&1 | grep -vE "$F" | tee results/evalbig_Z_B_O3_G.txt
log "PIPELINE_DONE"
