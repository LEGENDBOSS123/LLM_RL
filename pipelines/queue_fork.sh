#!/bin/zsh
cd /Users/vedantarora/Terminal/LLM_RL; source .venv/bin/activate
until grep -q PIPELINE_DONE pipeline_B.log; do sleep 60; done
echo "[$(date +%H:%M)] fork-score test (Rao vs Shannon vs random) on Z"
python eval/forkscore.py models/Qwen3-1.7B-RL 4 2>&1 | grep -vE "zoxide|Please ensure|If the issue|ajeetdsouza|_ZO_DOCTOR|^$" | tee results/forkscore_Z.txt
echo "[$(date +%H:%M)] FORK_DONE"
