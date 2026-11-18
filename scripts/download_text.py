import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Stream ~N MB slices of reasoning-heavy pretraining text into data/text/."""
import json, os, sys
from datasets import load_dataset
MB = int(sys.argv[1]) if len(sys.argv) > 1 else 75
os.makedirs("data/text", exist_ok=True)
SOURCES = [
    ("open-web-math/open-web-math", None, "openwebmath.jsonl"),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "fineweb_edu.jsonl"),
]
for name, cfg, out in SOURCES:
    ds = load_dataset(name, cfg, split="train", streaming=True)
    n = size = 0
    with open(f"data/text/{out}", "w") as f:
        for r in ds:
            t = r["text"]
            if len(t) < 1500:          # skip tiny pages: little room for multi-step reasoning
                continue
            f.write(json.dumps({"id": f"{out[:-6]}-{n}", "text": t, "url": r.get("url")}) + "\n")
            n += 1; size += len(t)
            if size > MB * 1e6:
                break
    print(f"{out:22} {n:6} docs  {size/1e6:.0f} MB")
