import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""LoRA SFT with per-segment loss masks (sandbox output never trained on), then fuse.

  python sft_masked.py BASE_MODEL DATA_DIR OUT_MODEL [epochs=2] [lr=5e-5]
"""
import json, math, random, sys, time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import grad_checkpoint

BASE, DATA, OUT = sys.argv[1:4]
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
LR = float(sys.argv[5]) if len(sys.argv) > 5 else 5e-5
BATCH, RANK, MAXLEN = 4, 16, 1600
mx.set_memory_limit(int(7e9))
mx.set_cache_limit(int(1e9))
random.seed(0)

model, tok = load(BASE)
model.freeze()
KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
linear_to_lora_layers(model, len(model.layers), {"rank": RANK, "scale": 2.0, "dropout": 0.0, "keys": KEYS})
grad_checkpoint(model.layers[0])


def encode(r):
    ids = tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                                  add_generation_prompt=True, tokenize=True)
    mask = [0] * len(ids)
    for txt, trainable in r["segments"]:
        t = tok.encode(txt, add_special_tokens=False)
        ids += t
        mask += [1 if trainable else 0] * len(t)
    end = tok.encode("<|im_end|>", add_special_tokens=False)   # learn to stop
    ids += end
    mask += [1] * len(end)
    return ids[:MAXLEN], mask[:MAXLEN]


train = [encode(json.loads(l)) for l in open(f"{DATA}/train.jsonl")]
valid = [encode(json.loads(l)) for l in open(f"{DATA}/valid.jsonl")]
print(f"train {len(train)}  valid {len(valid)}  trainable tokens {sum(sum(m) for _, m in train)}", flush=True)


def batchify(rows):
    T = max(len(i) for i, _ in rows)
    ids = mx.array([i + [0] * (T - len(i)) for i, _ in rows])
    m = mx.array([m[1:] + [0] * (T - len(m)) for _, m in rows], dtype=mx.float32)   # mask aligned to targets
    return ids, m


def loss_fn(model, ids, m):
    logits = model(ids[:, :-1]).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, ids[:, 1:], reduction="none")
    return (ce * m).sum() / mx.maximum(m.sum(), 1)


lg = nn.value_and_grad(model, loss_fn)
steps = EPOCHS * math.ceil(len(train) / BATCH)
sched = optim.cosine_decay(LR, steps, LR * 0.1)
opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)


def val_loss():
    tot = n = 0.0
    for k in range(0, len(valid), BATCH):
        ids, m = batchify(valid[k:k + BATCH])
        tot += loss_fn(model, ids, m).item() * m.sum().item()
        n += m.sum().item()
    return tot / max(n, 1)


best = v0 = val_loss()
print(f"val loss start {v0:.4f}", flush=True)
step, t0 = 0, time.time()
for ep in range(EPOCHS):
    random.shuffle(train)
    for k in range(0, len(train), BATCH):
        ids, m = batchify(train[k:k + BATCH])
        l, g = lg(model, ids, m)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state, l)
        step += 1
        if step % 10 == 0:
            print(f"step {step}/{steps} loss {l.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    v = val_loss()
    print(f"epoch {ep + 1} val loss {v:.4f}", flush=True)
    if v > 2 * v0 or not math.isfinite(v):
        sys.exit("diverged - not saving")

# fuse LoRA into the weights and save
from mlx_lm.utils import save
fused = [(n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")]
from mlx.utils import tree_unflatten
model.update_modules(tree_unflatten(fused))
import shutil, os, glob
save(OUT, BASE, model, tok, config=json.load(open(f"{BASE}/config.json")), donate_model=False)
print("saved", OUT, flush=True)
