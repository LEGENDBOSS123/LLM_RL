import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Run reference solutions on a sample of each unified file -> conversion sanity check."""
import json, random, sys, itertools
from collections import Counter
from sandbox.judge import Judge
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
random.seed(1)
def sample(path, n):
    rows = [json.loads(l) for l in itertools.islice(open(path), 20000)]
    rows = [r for r in rows if r.get("ref")]
    return random.sample(rows, min(n, len(rows)))
with Judge(workers=4) as j:
    for f in ["kodcode_train", "taco_train", "leetcode_train", "leetcode_eval"]:
        ok, why = 0, Counter()
        rs = sample(f"data/code/{f}.jsonl", N)
        for r in rs:
            if r["mode"] == "tests":
                g = j.grade_tests(r["ref"], r["test_code"], prefix=r["prefix"], timeout=8)
                good = g.get("all_pass")
            else:
                g = j.grade(r["ref"], r["io"][:10], timeout=4, partial=False)
                good = g["score"] == 1.0
            ok += bool(good)
            if not good: why.update(set(map(str, g["verdicts"])))
        print(f"{f:18} reference pass {ok}/{len(rs)}   failures: {why.most_common(5)}", flush=True)
