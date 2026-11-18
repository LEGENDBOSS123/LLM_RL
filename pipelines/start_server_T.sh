#!/bin/zsh
cd /Users/vedantarora/Terminal/LLM_RL; source .venv/bin/activate
until [ -f runs/T/config.json ]; do sleep 30; done
pkill -f "[s]erver.py"; sleep 2
exec nice -n 5 python server.py --run runs/T --tools --template r1zero --model models/Qwen3-1.7B-O3-4bit
