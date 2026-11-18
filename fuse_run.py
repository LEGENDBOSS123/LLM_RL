"""Merge an RL run's LoRA adapters into its base model.  python fuse_run.py runs/S OUT_MODEL"""
import json, sys
from mlx.utils import tree_unflatten
from mlx_lm import load
from mlx_lm.utils import save
from mlx_lm.tuner.utils import linear_to_lora_layers
run, out = sys.argv[1:3]
adapter = sys.argv[3] if len(sys.argv) > 3 else f"{run}/adapters.safetensors"
cfg = json.load(open(f"{run}/config.json"))
model, tok = load(cfg["model"])
model.freeze()
linear_to_lora_layers(model, len(model.layers), {"rank": cfg["rank"], "scale": 2.0, "dropout": 0.0,
                      "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]})
model.load_weights(adapter, strict=False)
fused = [(n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")]
model.update_modules(tree_unflatten(fused))
save(out, cfg["model"], model, tok, json.load(open(f"{cfg['model']}/config.json")), donate_model=False)
print("fused", adapter, "->", out)
