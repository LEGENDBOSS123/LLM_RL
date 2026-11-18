#!/bin/zsh
# Run T: RL with TOOLS (Python sandbox + sympy/z3/networkx) and FORK/BACKTRACK (<checkpoint>/<backtrack>), from O3.
# Stages: smoke test -> no-training tool baseline (O3 + tools) -> RL (60 steps) -> narrow + big evals (with tools).
cd /Users/vedantarora/Terminal/LLM_RL
source .venv/bin/activate
log() { echo "[$(date +%H:%M)] $*"; }
F="zoxide|Please ensure|If the issue|ajeetdsouza|_ZO_DOCTOR|^$"
NT2=countdown,puzzle24,n_queens,mini_sudoku,futoshiki,zebra_puzzles
TT="countdown,puzzle24,n_queens,n_queens@n=6;max_remove=3,mini_sudoku,futoshiki,zebra_puzzles"
until grep -q PIPELINE_DONE pipeline_G.log; do sleep 60; done
podman machine start >/dev/null 2>&1

log "STAGE T-smoke: 2 tiny steps with tools + backtrack"
rm -rf runs/T_smoke
python train_grpo.py --run runs/T_smoke --tools --backtrack --max_calls 4 --tool_cost 0.02 --model models/Qwen3-1.7B-O3 \
  --template r1zero --algo grpo --minibatches 2 --steps 2 --lr 5e-6 --beta 0.01 --max_tokens 512 --prompts 2 --group 4 \
  --tasks countdown,puzzle24 --code_frac 0 --gen_frac 0 --eval_every 1000 --skip_eval0 --save_every 1000 --sample_every 1 \
  --gen_batch 8 --mem_limit_gb 6 --judge_workers 2 --top_p 0.95 > runs/T_smoke.out 2>&1
if ! grep -q "^step     2" runs/T_smoke.out; then log "SMOKE FAILED"; tail -20 runs/T_smoke.out; log "PIPELINE_DONE"; exit 1; fi
grep "^step" runs/T_smoke.out | cut -c1-200

log "STAGE T0: tool baseline without training (O3 + tools)"
TOOLS=1 python eval/narrow_eval.py run models/Qwen3-1.7B-O3 O3tools $NT2 30 2>&1 | grep -vE "$F"

log "STAGE T: RL with tools + backtrack from O3"
rm -rf runs/T && mkdir -p runs/T
ARGS=(--run runs/T --tools --backtrack --max_backtracks 3 --max_calls 4 --tool_cost 0.02
  --model models/Qwen3-1.7B-O3 --template r1zero --algo grpo --minibatches 4 --clip_low 0.2 --clip_high 0.28
  --skip_eval0 --steps 60 --lr 5e-6 --beta 0.01 --max_entropy 2.0 --len_alpha 0 --max_tokens 1024 --prompts 8 --group 8 --stratified
  --tasks "$TT" --code_frac 0 --gen_frac 0.2 --eval_every 30 --eval_math_n 10 --eval_code_n 0 --eval_gen_n 120
  --save_every 5 --sample_every 5 --gen_batch 12 --mem_limit_gb 6 --judge_workers 2 --top_p 0.95 --temp_ladder 0.9,1.2
  --overlong_buffer 128 --overlong_penalty 0.2)
python train_grpo.py $ARGS > runs/T/train.out 2>&1
rc=$?
for try in 1 2 3; do
  [[ $rc -eq 0 || $rc -eq 3 ]] && break
  log "run T exited $rc - resuming (try $try)"
  python train_grpo.py $ARGS --resume --skip_eval0 >> runs/T/train.out 2>&1
  rc=$?
done
log "run T finished rc=$rc"
rm -rf models/Qwen3-1.7B-T && python fuse_run.py runs/T models/Qwen3-1.7B-T 2>&1 | tail -1
log "STAGE T-eval"
TOOLS=1 python eval/narrow_eval.py run models/Qwen3-1.7B-T Ttools $NT2 30 2>&1 | grep -vE "$F"
python eval/narrow_eval.py compare O3tools Ttools 2>&1 | grep -vE "$F" | tee results/narrow_O3tools_vs_Ttools.txt
TOOLS=1 python eval/evalbig.py run models/Qwen3-1.7B-T Ttools 2>&1 | grep -vE "$F"
python eval/evalbig.py compare Z O3 Ttools 2>&1 | grep -vE "$F" | tee results/evalbig_Z_O3_Ttools.txt
log "PIPELINE_DONE"
