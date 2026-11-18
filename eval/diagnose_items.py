import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Shared unseen diagnostic set (same as diagnose.py)."""
import reasoning_gym
from common import EVAL_SEED
R1Z = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
       "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
       "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
       "i.e., <think> reasoning process here </think> <answer> answer here </answer>.")
UNSEEN = ["letter_counting", "family_relationships", "word_sorting", "number_sequence", "course_schedule", "circuit_logic", "word_ladder"]
K, N = 8, 15
items = []
for t in UNSEEN:
    ds = reasoning_gym.create_dataset(t, size=N, seed=EVAL_SEED + 4242)
    items += [(t, ds, e) for e in ds]
