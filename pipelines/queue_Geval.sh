#!/bin/zsh
cd /Users/vedantarora/Terminal/LLM_RL; source .venv/bin/activate
F="zoxide|Please ensure|If the issue|ajeetdsouza|_ZO_DOCTOR|^$"
until grep -q PIPELINE_DONE pipeline_T.log; do sleep 60; done
echo "[$(date +%H:%M)] G (step 45) fuse + narrow eval"
rm -rf models/Qwen3-1.7B-G && python fuse_run.py runs/G models/Qwen3-1.7B-G 2>&1 | tail -1
python eval/narrow_eval.py run models/Qwen3-1.7B-G G countdown,puzzle24,n_queens,mini_sudoku 30 2>&1 | grep -vE "$F"
python eval/narrow_eval.py compare O3 G 2>&1 | grep -vE "$F" | tee results/narrow_O3_vs_G.txt
echo "[$(date +%H:%M)] GEVAL_DONE"
