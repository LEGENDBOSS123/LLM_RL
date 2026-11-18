import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root on path
"""Convert downloaded coding datasets into one unified JSONL format in data/code/.

Unified row:
  id, source, subset, difficulty (easy|medium|hard|unknown), tags [..], date,
  prompt      : problem statement shown to the model
  starter     : starter code (may be "")
  mode        : "tests"  -> grade with Judge.grade_tests(prefix + code, test_code)
                "stdio"  -> grade with Judge.grade(code, io)
  prefix      : code prepended to the model's solution (helper imports/classes), not scanned
  test_code   : (tests mode) python with test_* functions importing from `solution`
  io          : (stdio mode) [[stdin, expected_stdout], ...]
  ref         : one reference solution (for validating the conversion only, never shown)
"""
import ast, re, base64, glob, json, os, pickle, random, zlib, io as _io
from collections import Counter
import pyarrow.parquet as pq

OUT = "data/code"
HEAVY = re.compile(r"^\s*(?:from|import)\s+(torch|torchvision|sklearn|matplotlib|seaborn|flask|PIL|requests|cv2|"
                   r"bs4|psutil|aiohttp|tensorflow|keras|django|selenium|boto3|nltk|transformers)\b", re.M)
NEEDS_PYTEST_RUNNER = re.compile(r"pytest\.fixture|parametrize|def test_\w+\(\s*\w|capsys|monkeypatch|tmp_path")


def usable(*srcs):
    return not any(HEAVY.search(s or "") for s in srcs)
random.seed(0)


def lit(s, default=None):
    if s is None:
        return default
    if not isinstance(s, str):
        return s
    for f in (json.loads, ast.literal_eval):
        try:
            return f(s)
        except Exception:
            pass
    return default


def diff_norm(d):
    d = (d or "").lower()
    if d in ("easy", "introductory"):
        return "easy"
    if d in ("medium", "interview", "medium_hard"):
        return "medium"
    if d in ("hard", "very_hard", "competition"):
        return "hard"
    return "unknown"


def write(rows, name):
    with open(f"{OUT}/{name}", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name:28} {len(rows):7}   modes={dict(Counter(r['mode'] for r in rows))}  "
          f"difficulty={dict(Counter(r['difficulty'] for r in rows))}")


# ---------------------------------------------------------------- KodCode (pytest-style)
def kodcode():
    rows = []
    for f in sorted(glob.glob(f"{OUT}/kodcode/data/train-*.parquet")):
        t = pq.read_table(f, columns=["question_id", "subset", "question", "solution", "test",
                                      "test_info", "gpt_difficulty", "gpt_pass_percentage", "style"])
        for r in t.to_pylist():
            if "from solution import" not in (r["test"] or ""):
                continue
            if not usable(r["test"], r["solution"]) or NEEDS_PYTEST_RUNNER.search(r["test"]):
                continue
            info = lit(r["test_info"], []) or []
            decl = "\n".join(i.get("function_declaration", "") for i in info if isinstance(i, dict))
            prompt = r["question"]
            if decl and r["style"] != "complete":
                prompt += f"\n\nYour solution must define:\n{decl}"
            rows.append(dict(id=f"kodcode/{r['question_id']}", source="kodcode", subset=r["subset"],
                             difficulty=diff_norm(r["gpt_difficulty"]), tags=[r["subset"]], date=None,
                             prompt=prompt, starter="", mode="tests", prefix="", test_code=r["test"],
                             io=None, ref=r["solution"],
                             gpt4o_pass_rate=float(r["gpt_pass_percentage"] or 0)))
    return rows


# ---------------------------------------------------------------- TACO-verified
def taco():
    rows = []
    for r in json.load(open(f"{OUT}/taco_verified/taco_verified.json")):
        iot = lit(r["input_output"], {}) or {}
        ins, outs = iot.get("inputs") or [], iot.get("outputs") or []
        sols = lit(r["solutions"], []) or []
        if not ins or len(ins) != len(outs):
            continue
        base = dict(id=f"taco/{r['id']}", source="taco", subset=r.get("source"),
                    difficulty=diff_norm(r["difficulty"]), tags=lit(r.get("skill_types"), []) or [],
                    date=r.get("date"), prompt=r["question"], starter=r.get("starter_code") or "",
                    prefix="", ref=sols[0] if sols else None)
        fn = iot.get("fn_name")
        if fn:  # call-based -> generate a pytest-style file
            lines = ["import solution as _s", "def _call(*a):",
                     f"    f = getattr(_s, {fn!r}, None) or getattr(_s.Solution(), {fn!r})", "    return f(*a)"]
            for i, (a, o) in enumerate(list(zip(ins, outs))[:30]):
                exp = o[0] if isinstance(o, list) and len(o) == 1 else o
                lines.append(f"def test_{i}():\n    r = _call(*{a!r})\n    assert r == {exp!r} or [r] == {exp!r}")
            rows.append(dict(base, mode="tests", test_code="\n".join(lines), io=None))
        else:
            io = [[i if isinstance(i, str) else "\n".join(map(str, i)),
                   o if isinstance(o, str) else "\n".join(map(str, o))] for i, o in zip(ins, outs)][:30]
            rows.append(dict(base, mode="stdio", test_code=None, io=io))
    return rows


# ---------------------------------------------------------------- LeetCodeDataset
def leetcode(split):
    rows = []
    for l in open(f"{OUT}/leetcode/LeetCodeDataset-{split}.jsonl"):
        r = json.loads(l)
        test = r["test"] + f"\n\nfrom solution import *\ndef test_all():\n    check({r['entry_point']})\n"
        rows.append(dict(id=f"leetcode/{r['task_id']}", source="leetcode", subset=split,
                         difficulty=diff_norm(r["difficulty"]), tags=lit(r["tags"], []) or [],
                         date=r.get("estimated_date"), prompt=r["problem_description"],
                         starter=r["starter_code"], mode="tests",
                         prefix=r["prompt"], test_code=test, io=None, ref=r["completion"]))
    return rows


# ---------------------------------------------------------------- LiveCodeBench (eval)
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):  # only plain data allowed, never code
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def lcb_private(s):
    try:
        return json.loads(s)
    except Exception:
        raw = zlib.decompress(base64.b64decode(s))
        return json.loads(SafeUnpickler(_io.BytesIO(raw)).load())


def livecodebench():
    rows, skipped = [], Counter()
    for f in sorted(glob.glob(f"{OUT}/livecodebench/test*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            try:
                tests = json.loads(r["public_test_cases"]) + lcb_private(r["private_test_cases"])
            except Exception:
                skipped["decode"] += 1
                continue
            if any(t.get("testtype") != "stdin" for t in tests):
                skipped["functional"] += 1   # LeetCode-style functional tests: TODO
                continue
            rows.append(dict(id=f"lcb/{r['question_id']}", source="livecodebench", subset=r["platform"],
                             difficulty=diff_norm(r["difficulty"]), tags=[r["platform"]],
                             date=r["contest_date"][:10], prompt=r["question_content"], starter="",
                             mode="stdio", prefix="", test_code=None,
                             io=[[t["input"], t["output"]] for t in tests][:40], ref=None))
    print("livecodebench skipped:", dict(skipped))
    return rows


if __name__ == "__main__":
    kc = kodcode()
    random.shuffle(kc)
    kc_eval, kc_train = kc[:1000], kc[1000:]
    write(kc_train, "kodcode_train.jsonl")
    write(kc_eval, "kodcode_eval.jsonl")
    tc = taco()
    write(tc, "taco_train.jsonl")
    write(leetcode("train"), "leetcode_train.jsonl")
    write(leetcode("test"), "leetcode_eval.jsonl")
    write(livecodebench(), "livecodebench_eval.jsonl")
