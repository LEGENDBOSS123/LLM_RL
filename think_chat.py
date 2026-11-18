"""Chat where the model must think for at least N tokens (s1-style budget forcing).

Whenever the model tries to stop thinking (writes </think> or <answer>) before N thinking tokens,
that marker is cut off, a cue (default " Wait,") is appended, and it keeps going.
Everything is shown raw: system prompt, think tags, forced cues (yellow), token counts.

usage:
  python think_chat.py                                  # run-Z model, R1-Zero format, min 256
  python think_chat.py --min 512 --cue " Hmm,"
  python think_chat.py --model models/Qwen3-1.7B-Instruct-4bit --format chat
commands in chat: /min N   /cue TEXT   /reset   /quit
"""
import argparse, sys
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/Qwen3-1.7B-RL-4bit")
ap.add_argument("--format", default="r1zero", choices=["r1zero", "chat"])
ap.add_argument("--min", type=int, default=256, help="minimum thinking tokens")
ap.add_argument("--cue", default=" Wait,")
ap.add_argument("--max_tokens", type=int, default=2048)
ap.add_argument("--temp", type=float, default=0.7)
args = ap.parse_args()
mx.set_memory_limit(int(5e9))

R1Z = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
       "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
       "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
       "i.e., <think> reasoning process here </think> <answer> answer here </answer>.")
DIM, YEL, CYA, GRN, RST = "\033[2m", "\033[1;33m", "\033[36m", "\033[32m", "\033[0m"
MARKERS = ["</think>", "<answer>"]

print(f"{DIM}loading {args.model} ...{RST}")
model, tok = load(args.model)
sampler = make_sampler(temp=args.temp, top_p=0.95)
history = []          # (user, assistant_text)


def build_prompt(user):
    if args.format == "r1zero":
        s = R1Z
        for u, a in history:
            s += f" User: {u} Assistant:{a}"
        return s + f" User: {user} Assistant:"
    msgs = []
    for u, a in history:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def ntok(s):
    return len(tok.encode(s, add_special_tokens=False))


def reply(user):
    prompt = build_prompt(user)
    print(f"{DIM}{prompt}{RST}", end="", flush=True)         # raw prompt incl. system prompt
    text, forced = "", 0
    while True:
        restart = False
        for r in stream_generate(model, tok, prompt + text, max_tokens=args.max_tokens - ntok(text), sampler=sampler):
            text += r.text
            print(r.text, end="", flush=True)
            hit = min((text.find(m) for m in MARKERS if m in text), default=-1)
            if hit >= 0 and ntok(text[:hit]) < args.min:            # tried to stop thinking too early
                cut = text[:hit]
                print(f"{YEL}  ✂[{ntok(cut)} < {args.min} tokens]{args.cue}{RST}", end="", flush=True)
                text, forced, restart = cut.rstrip() + args.cue, forced + 1, True
                break
            if args.format == "r1zero" and ("</answer>" in text or "\nUser:" in text or " User:" in text[-8:]):
                text = text.split("\nUser:")[0]
                break
        if not restart:
            break
    think_end = min((text.find(m) for m in MARKERS if m in text), default=len(text))
    print(f"\n{CYA}[thinking tokens {ntok(text[:think_end])} | total {ntok(text)} | forced continues {forced}]{RST}\n")
    history.append((user, text))


print(f"{GRN}min thinking tokens = {args.min}, cue = {args.cue!r}, format = {args.format}. /min N  /cue TEXT  /reset  /quit{RST}\n")
while True:
    try:
        u = input(f"{GRN}you> {RST}").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not u:
        continue
    if u == "/quit":
        break
    if u == "/reset":
        history.clear(); print("history cleared"); continue
    if u.startswith("/min "):
        args.min = int(u.split()[1]); print(f"min = {args.min}"); continue
    if u.startswith("/cue "):
        args.cue = " " + u[5:].strip(); print(f"cue = {args.cue!r}"); continue
    reply(u)
