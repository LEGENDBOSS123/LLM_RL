import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Measure info gain of untrained base-model thoughts under different thought-channel formats (no training)."""
import sys, statistics, random
sys.argv = ["x", "--run", "runs/Rprobe", "--eval_every", "100000"]
import importlib.util
spec = importlib.util.spec_from_file_location("R", "train_rlp.py")
src = open("train_rlp.py").read().split("sampler = make_sampler(temp=args.temp")[0]   # definitions only
ns = {"__name__": "rlp_probe"}
exec(compile(src, "train_rlp.py", "exec"), ns)
tok, model, statistics = ns["tok"], ns["model"], ns["statistics"]
random.seed(5)
pos = ns["pick_positions"](ns["EVAL_DOCS"], 32)
G, RESUME = 4, 32
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import BatchGenerator
import mlx.core as mx

def run(opener, closer, resume, label, think_max=64):
    OPEN = tok.encode(opener, add_special_tokens=False)
    ctxs = [p["ctx"] for p in pos for _ in range(G)]
    gen = BatchGenerator(model, stop_tokens=ns["STOP"], sampler=make_sampler(temp=0.8, top_p=0.95), completion_batch_size=32, prefill_batch_size=8)
    uids = gen.insert([c + OPEN for c in ctxs], [think_max] * len(ctxs))
    idx = {u: i for i, u in enumerate(uids)}; toks = [[] for _ in ctxs]; fin = set()
    while len(fin) < len(ctxs):
        for r in gen.next_generated():
            toks[idx[r.uid]].append(r.token)
            if r.finish_reason is not None: fin.add(idx[r.uid])
    gen.close()
    CL = tok.encode(closer, add_special_tokens=False)
    gains = []
    for k in range(0, len(ctxs), 4):
        seqs, starts = [], []
        for i in range(k, min(k + 4, len(ctxs))):
            c, t, tg = ctxs[i], toks[i], pos[i // G]["tgt"]
            rs = c[-resume:] if resume else []
            s = c + OPEN + t + CL + rs
            seqs.append(s + tg); starts.append(len(s))
        lp = ns["token_logps"](seqs); mx.eval(lp); lp = lp.tolist()
        for i, (row, s) in enumerate(zip(lp, starts)):
            gi = (k + i) // G
            gains.append(statistics.mean(row[s - 1:s - 1 + args_target]) - pos[gi]["base_lp"])
    best = [max(gains[g * G:(g + 1) * G]) for g in range(len(pos))]
    print(f"{label:58} mean gain {statistics.mean(gains):+.3f} | best-of-{G} gain {statistics.mean(best):+.3f} | positions where best>0: {statistics.mean(b > 0 for b in best):.2f}")
    ex = tok.decode(toks[0][:60]); print("    e.g. thought:", repr(ex[:160]))

args_target = ns["args"].target
run(" <think>\n", "\n</think>\n", 0, "A: bare <think>, no resume")
run(" <think>\n", "\n</think>\n", RESUME, "B: bare <think> + resume last 32 ctx tokens")
run("\n\n[Note: let me think about what comes next. ", "]\n\n", RESUME, "C: '[Note: let me think about what comes next.' + resume")
run(" (Thinking step by step about this: ", ")\n", RESUME, "D: '(Thinking step by step about this:' + resume")
