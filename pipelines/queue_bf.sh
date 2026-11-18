#!/bin/zsh
cd /Users/vedantarora/Terminal/LLM_RL; source .venv/bin/activate
until grep -q PIPELINE_DONE pipeline_P.log; do sleep 60; done
echo "[$(date +%H:%M)] budget forcing test on Z"
python eval/budget_force.py models/Qwen3-1.7B-RL 256,512 2>&1 | grep -vE "zoxide|Please ensure|If the issue|ajeetdsouza|_ZO_DOCTOR|^$" | tee results/budget_force_Z.txt
echo "[$(date +%H:%M)] BF_DONE"
