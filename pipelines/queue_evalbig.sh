#!/bin/zsh
cd /Users/vedantarora/Terminal/LLM_RL; source .venv/bin/activate
F="zoxide|Please ensure|If the issue|ajeetdsouza|_ZO_DOCTOR|^$"
until grep -q FORK_DONE queue_fork.log; do sleep 60; done
echo "[$(date +%H:%M)] big paired eval: Z"
python eval/evalbig.py run models/Qwen3-1.7B-RL Z 2>&1 | grep -vE "$F"
echo "[$(date +%H:%M)] big paired eval: B"
python eval/evalbig.py run models/Qwen3-1.7B-B B 2>&1 | grep -vE "$F"
python eval/evalbig.py compare Z B 2>&1 | grep -vE "$F" | tee results/evalbig_Z_vs_B.txt
echo "[$(date +%H:%M)] EVALBIG_DONE"
