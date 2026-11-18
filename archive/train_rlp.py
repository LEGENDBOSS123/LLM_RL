import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""RLP-style next-token reasoning RL (NVIDIA RLP / RPT family) on plain text.

Per step: pick P positions in real documents where the frozen base finds the next N tokens hard-but-learnable,
sample G thoughts "<think> ... </think>" after the context, and reward each thought by how much it raises the
likelihood of the REAL next N tokens:

    reward = mean_t [ log p_policy(x_t | ctx, thought) ]  -  mean_t [ log p_base(x_t | ctx) ]   (- c * thought_len)

Advantage = reward - mean over the G thoughts of that position (GRPO). Only the thought tokens get gradient;
the document text is never trained on directly. Unlimited data, no templates, no distillation.

  python train_rlp.py --run runs/R --steps 400
"""
import argparse, json, os, random, statistics, time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.lora import LoRALinear
from tools import suffix_stop_sequences

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/R")
ap.add_argument("--model", default="models/Qwen3-1.7B-Base")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--positions", type=int, default=8)          # P
ap.add_argument("--group", type=int, default=8)              # G thoughts per position
ap.add_argument("--ctx", type=int, default=256)              # max context tokens before the position
ap.add_argument("--target", type=int, default=16)            # N real tokens scored after the thought
ap.add_argument("--think_max", type=int, default=64)
ap.add_argument("--think_cost", type=float, default=0.0)     # thoughts are fixed-length (forced close)
ap.add_argument("--nll_band", default="1.0,5.0", help="keep positions whose base NLL (mean over target) is in this band")
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--micro", type=int, default=4)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--gen_batch", type=int, default=32)
ap.add_argument("--eval_every", type=int, default=25)
ap.add_argument("--eval_n", type=int, default=96)
ap.add_argument("--save_every", type=int, default=10)
ap.add_argument("--sample_every", type=int, default=10)
ap.add_argument("--mem_limit_gb", type=float, default=7.0)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.run, exist_ok=True)
random.seed(args.seed)
mx.random.seed(args.seed)
mx.set_memory_limit(int(args.mem_limit_gb * 1e9))
mx.set_cache_limit(int(1e9))
LO, HI = [float(x) for x in args.nll_band.split(",")]

# ------------------------------------------------------------------ data (eval docs held out)
DOCS = []
for f in ("data/text/openwebmath.jsonl", "data/text/fineweb_edu.jsonl"):
    for l in open(f):
        DOCS.append(json.loads(l)["text"])
rng = random.Random(123)
rng.shuffle(DOCS)
EVAL_DOCS, TRAIN_DOCS = DOCS[:400], DOCS[400:]
print(f"docs: train {len(TRAIN_DOCS)}  eval {len(EVAL_DOCS)}", flush=True)

# ------------------------------------------------------------------ model
model, tok = load(args.model)
model.freeze()
linear_to_lora_layers(model, len(model.layers), {"rank": args.rank, "scale": 2.0, "dropout": 0.0,
                      "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]})
grad_checkpoint(model.layers[0])
LORAS = [m for _, m in model.named_modules() if isinstance(m, LoRALinear)]


def set_lora(on):
    for m in LORAS:
        m.scale = 2.0 if on else 0.0


opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
start = 0
if args.resume and os.path.exists(f"{args.run}/state.json"):
    model.load_weights(f"{args.run}/adapters.safetensors", strict=False)
    if os.path.exists(f"{args.run}/optimizer.safetensors"):
        opt.state = tree_unflatten(list(mx.load(f"{args.run}/optimizer.safetensors").items()))
        opt.learning_rate = args.lr
    start = json.load(open(f"{args.run}/state.json"))["step"]
    print("resumed at", start, flush=True)
json.dump(vars(args), open(f"{args.run}/config.json", "w"), indent=2)

# Thought channel (probed: best info gain for an untrained base): a bracketed note, then the text RESUMES with
# the last RESUME context tokens so the thought is a true aside and the target follows naturally.
OPEN = tok.encode("\n\n[Note: let me think about what comes next. ", add_special_tokens=False)
CLOSE = tok.encode("]\n\n", add_special_tokens=False)
RESUME = 32
STOP = [[t] for t in tok.eos_token_ids] + suffix_stop_sequences(tok, "</", "think")


def token_logps(ids_batch):
    """ids_batch: list of token lists (padded here). Returns [B, T-1] logp of token j+1, float32."""
    T = max(len(s) for s in ids_batch)
    ids = mx.array([s + [0] * (T - len(s)) for s in ids_batch])
    logits = model(ids[:, :-1]).astype(mx.float32)
    return -nn.losses.cross_entropy(logits, ids[:, 1:], reduction="none")


def pick_positions(docs_pool, n, max_tries=6):
    """Hard-but-learnable positions: base (LoRA off) NLL over the next N tokens within [LO, HI]."""
    out, tries = [], 0
    set_lora(False)
    while len(out) < n and tries < max_tries:
        tries += 1
        chunks = []
        for _ in range(n * 2):
            ids = tok.encode(random.choice(docs_pool)[:6000])
            L = args.ctx + args.target
            if len(ids) < 96 + args.target:
                continue
            off = random.randint(0, max(0, len(ids) - L))
            chunks.append(ids[off:off + L])
        if not chunks:
            continue
        lp = []
        for k in range(0, len(chunks), 8):                 # small batches: full-vocab logits are huge
            part = mx.stop_gradient(token_logps(chunks[k:k + 8]))
            mx.eval(part)
            lp += part.tolist()
            del part
        mx.clear_cache()
        for c, row in zip(chunks, lp):
            cands = []
            for p in range(64, len(c) - args.target):
                nll = -statistics.mean(row[p - 1:p - 1 + args.target])
                if LO <= nll <= HI:
                    cands.append((p, nll))
            if cands:
                p, nll = random.choice(cands)
                out.append(dict(ctx=c[:p], tgt=c[p:p + args.target], base_lp=-nll))
            if len(out) >= n:
                break
    set_lora(True)
    return out


def sample_thoughts(contexts, sampler):
    """Generate a thought after each context; returns (thought_ids incl. close, train-mask, closed?)."""
    gen = BatchGenerator(model, stop_tokens=STOP, sampler=sampler, completion_batch_size=args.gen_batch, prefill_batch_size=8)
    uids = gen.insert([c + OPEN for c in contexts], [args.think_max] * len(contexts))
    idx = {u: i for i, u in enumerate(uids)}
    toks = [[] for _ in contexts]
    fin = {}
    while len(fin) < len(contexts):
        for r in gen.next_generated():
            i = idx[r.uid]
            toks[i].append(r.token)
            if r.finish_reason is not None:
                fin[i] = r.finish_reason
    gen.close()
    out = []
    for i, t in enumerate(toks):
        if t and t[-1] in tok.eos_token_ids:
            t = t[:-1]
        out.append((t + CLOSE, [1] * len(t) + [0] * len(CLOSE), fin[i] != "length"))
    return out


def score(items):
    """items: list of (ctx, thought, tgt). Returns mean logp of tgt under the policy (LoRA on), no grad."""
    res = []
    for k in range(0, len(items), args.micro):
        batch = items[k:k + args.micro]
        seqs = [c + OPEN + th + c[-RESUME:] + tg for c, th, tg in batch]
        lp = mx.stop_gradient(token_logps(seqs))
        mx.eval(lp)
        lp = lp.tolist()
        for (c, th, tg), row in zip(batch, lp):
            s = len(c) + len(OPEN) + len(th) + len(c[-RESUME:])
            res.append(statistics.mean(row[s - 1:s - 1 + len(tg)]))
    return res


def loss_fn(m, ids, mask, adv, norm):
    logits = m(ids[:, :-1]).astype(mx.float32)
    lp = -nn.losses.cross_entropy(logits, ids[:, 1:], reduction="none")
    return -(adv[:, None] * lp * mask).sum() / norm


lg = nn.value_and_grad(model, loss_fn)


def evaluate(step):
    """Held-out: info gain of a single greedy-ish thought vs no thought, on fixed eval positions."""
    st = random.getstate()
    random.seed(999)
    pos = pick_positions(EVAL_DOCS, args.eval_n)
    random.setstate(st)
    th = sample_thoughts([p["ctx"] for p in pos], make_sampler(temp=0.3, top_p=0.95))
    with_lp = score([(p["ctx"], t[0], p["tgt"]) for p, t in zip(pos, th)])
    gains = [w - p["base_lp"] for w, p in zip(with_lp, pos)]
    lens = [sum(t[1]) for t in th]
    out = dict(step=step, info_gain=round(statistics.mean(gains), 4),
               frac_positive=round(statistics.mean(g > 0 for g in gains), 3),
               thought_len=round(statistics.mean(lens), 1), closed=round(statistics.mean(t[2] for t in th), 3),
               base_nll=round(-statistics.mean(p["base_lp"] for p in pos), 4))
    with open(f"{args.run}/eval.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
    with open(f"{args.run}/eval_samples_step{step}.jsonl", "w") as f:
        for p, t, g in list(zip(pos, th, gains))[:24]:
            f.write(json.dumps(dict(context=tok.decode(p["ctx"][-120:]), thought=tok.decode(t[0]),
                                    target=tok.decode(p["tgt"]), gain=round(g, 4))) + "\n")
    print(f"[eval step {step}] info_gain {out['info_gain']:+.4f} nats/token | positive {out['frac_positive']:.2f} | "
          f"thought {out['thought_len']} tok | closed {out['closed']}", flush=True)
    mx.clear_cache()


sampler = make_sampler(temp=args.temp, top_p=args.top_p)
if start == 0:
    evaluate(0)
for step in range(start + 1, args.steps + 1):
    t0 = time.time()
    pos = pick_positions(TRAIN_DOCS, args.positions)
    ctxs = [p["ctx"] for p in pos for _ in range(args.group)]
    th = sample_thoughts(ctxs, sampler)
    t_gen = time.time() - t0
    items = [(ctxs[i], th[i][0], pos[i // args.group]["tgt"]) for i in range(len(ctxs))]
    with_lp = score(items)
    lens = [sum(t[1]) for t in th]
    gains = [with_lp[i] - pos[i // args.group]["base_lp"] for i in range(len(items))]
    rewards = [g - args.think_cost * L for g, L in zip(gains, lens)]
    advs = []
    for gi in range(len(pos)):
        rs = rewards[gi * args.group:(gi + 1) * args.group]
        m = statistics.mean(rs)
        advs += [r - m for r in rs]
    # policy gradient on thought tokens only
    seqs, masks = [], []
    for (c, thought, tg), (_, tmask, _) in zip(items, th):
        s = c + OPEN + thought
        seqs.append(s)
        masks.append([0.0] * (len(c) + len(OPEN) - 1) + [float(x) for x in tmask])
    norm = len(seqs) * args.think_max
    grads, loss_v = None, 0.0
    for k in range(0, len(seqs), args.micro):
        bs, bm = seqs[k:k + args.micro], masks[k:k + args.micro]
        T = max(len(s) for s in bs)
        ids = mx.array([s + [0] * (T - len(s)) for s in bs])
        mask = mx.array([m + [0.0] * (T - 1 - len(m)) for m in bm])
        l, g = lg(model, ids, mask, mx.array(advs[k:k + args.micro]), norm)
        grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
        mx.eval(grads, l)
        loss_v += l.item()
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)
    del grads
    mx.clear_cache()
    dt = time.time() - t0
    log = dict(step=step, t=round(dt, 1), t_gen=round(t_gen, 1), info_gain=round(statistics.mean(gains), 4),
               reward=round(statistics.mean(rewards), 4), thought_len=round(statistics.mean(lens), 1),
               closed=round(statistics.mean(t[2] for t in th), 3), empty=round(statistics.mean(L <= 2 for L in lens), 3),
               adv_std=round(statistics.pstdev(advs), 4), base_nll=round(-statistics.mean(p["base_lp"] for p in pos), 3),
               loss=round(loss_v, 5), peak_gb=round(mx.get_peak_memory() / 1e9, 2))
    mx.reset_peak_memory()
    with open(f"{args.run}/log.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")
    print(f"step {step:4d} | {dt:4.0f}s (gen {t_gen:.0f}) | gain {log['info_gain']:+.4f} | thought {log['thought_len']:5.1f} tok "
          f"| closed {log['closed']:.2f} | empty {log['empty']:.2f} | adv_std {log['adv_std']:.4f} | mem {log['peak_gb']}GB", flush=True)
    if step % args.sample_every == 0:
        with open(f"{args.run}/samples.jsonl", "a") as f:
            for i in random.sample(range(len(items)), 6):
                f.write(json.dumps(dict(step=step, context=tok.decode(items[i][0][-120:]), thought=tok.decode(th[i][0]),
                                        target=tok.decode(items[i][2]), gain=round(gains[i], 4))) + "\n")
    if step % args.save_every == 0 or step == args.steps:
        mx.save_safetensors(f"{args.run}/adapters.safetensors", dict(tree_flatten(model.trainable_parameters())))
        mx.save_safetensors(f"{args.run}/optimizer.safetensors", dict(tree_flatten(opt.state)))
        json.dump(dict(step=step), open(f"{args.run}/state.json", "w"))
    if step % args.eval_every == 0:
        os.makedirs(f"{args.run}/snapshots", exist_ok=True)
        mx.save_safetensors(f"{args.run}/snapshots/adapters_step{step}.safetensors", dict(tree_flatten(model.trainable_parameters())))
        evaluate(step)
