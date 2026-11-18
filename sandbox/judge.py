"""Host-side judge: a pool of locked-down, long-lived sandbox containers.

    from sandbox.judge import Judge
    with Judge(workers=4) as j:
        r = j.run("print(input()[::-1])", stdin="abc")          # raw execution
        g = j.grade(code, tests=[("1 2\n", "3"), ...])          # hidden-test grading

Grading happens HERE on the host: the sandbox never sees expected outputs.
"""
import json, queue, re, subprocess, threading, uuid
from concurrent.futures import ThreadPoolExecutor

IMAGE = "llmrl-sandbox"
PODMAN_ARGS = [
    "--rm", "-i",
    "--network", "none",
    "--read-only",
    "--tmpfs", "/tmp:rw,size=64m,mode=1777",
    "--memory", "400m", "--memory-swap", "400m",
    "--cpus", "1",
    "--pids-limit", "64",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--user", "10001:10001",
]


HARNESS = r'''
import sys, os, json, types, builtins
def _mk():
    n = sys.stdin.readline().strip()
    w, fl = sys.stdout.write, sys.stdout.flush
    def report(s):
        w("\n@@RESULT-" + n + "@@" + s + "\n"); fl()
    return report
_report = _mk()
del _mk
sys.stdin = open(os.devnull)
_SAFE = (int, float, complex, str, bytes, bool, type(None), list, tuple, dict, set, frozenset)

def _check_types(v, depth=0):
    """Reject objects with custom __eq__ (the 'always-equal' reward hack)."""
    if depth > 50:
        return True
    if isinstance(v, (list, tuple, set, frozenset)):
        return all(_check_types(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return all(_check_types(k, depth + 1) and _check_types(x, depth + 1) for k, x in v.items())
    if type(v).__eq__ in (object.__eq__,) or type(v) in _SAFE:
        return True
    return type(v).__module__ in ("builtins", "collections", "decimal", "fractions", "datetime", "numpy", "re")

open("{module}.py", "w").write({sol})
sys.path.insert(0, os.getcwd())
results = {{}}
try:
    import {module} as _sol
    for _n in dir(_sol):
        _f = getattr(_sol, _n)
        if callable(_f) and getattr(_f, "__module__", None) == "{module}" and isinstance(_f, types.FunctionType):
            def _wrap(f):
                def g(*a, **k):
                    r = f(*a, **k)
                    if not _check_types(r):
                        raise TypeError("non-builtin return type rejected")
                    return r
                g.__wrapped__ = f
                return g
            setattr(_sol, _n, _wrap(_f))
except BaseException as e:
    results["import"] = "import_error:" + type(e).__name__
else:
    ns = {{"__name__": "tests"}}
    try:
        exec(compile({tests}, "tests.py", "exec"), ns)
        tests = [(k, v) for k, v in ns.items() if k.startswith("test") and callable(v)]
        if not tests:
            results["module_asserts"] = "pass"
        for k, v in tests:
            try:
                v()
                results[k] = "pass"
            except AssertionError:
                results[k] = "fail"
            except BaseException as e:
                results[k] = "error:" + type(e).__name__
    except AssertionError:
        results["module_asserts"] = "fail"
    except BaseException as e:
        results["module"] = "error:" + type(e).__name__
_report(json.dumps(results))
os._exit(0)
'''


# Code that pokes at the harness / interpreter internals gets zero reward and is flagged.
import re as _re
SUSPICIOUS = _re.compile(r"__main__|__closure__|sys\.modules|\bgc\b|\binspect\b|_report|os\._exit|"
                         r"__builtins__|builtins|sys\.stdout|__eq__|ctypes|@@RESULT", _re.S)


class Worker:
    def __init__(self):
        self.name = f"llmrl-sbx-{uuid.uuid4().hex[:8]}"
        self.p = subprocess.Popen(["podman", "run", "--name", self.name, *PODMAN_ARGS, IMAGE],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1)

    def call(self, req, hard_timeout):
        res = {}
        def rd():
            res["line"] = self.p.stdout.readline()
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        t = threading.Thread(target=rd, daemon=True)
        t.start()
        t.join(hard_timeout)
        if t.is_alive() or not res.get("line"):
            raise RuntimeError("worker hung or died")
        return json.loads(res["line"])

    def kill(self):
        subprocess.run(["podman", "rm", "-f", "-t", "0", self.name], capture_output=True)
        try:
            self.p.kill()
        except Exception:
            pass


class Judge:
    def __init__(self, workers=4):
        self.pool = queue.Queue()
        for _ in range(workers):
            self.pool.put(Worker())
        self.ex = ThreadPoolExecutor(workers)

    def run(self, code, stdin="", timeout=5.0, mem_mb=256):
        req = dict(id=uuid.uuid4().hex, code=code, stdin=stdin, timeout=timeout, mem_mb=mem_mb)
        w = self.pool.get()
        try:
            return w.call(req, hard_timeout=timeout + 15)
        except Exception as e:
            w.kill()          # misbehaving container -> replace it
            w = Worker()
            return dict(status="internal", stdout="", stderr=repr(e), exit=None, time=timeout)
        finally:
            self.pool.put(w)

    def run_many(self, reqs):
        """reqs: list of dicts with run() kwargs. Parallel over the pool."""
        return list(self.ex.map(lambda r: self.run(**r), reqs))

    def grade(self, code, tests, timeout=2.0, partial=True):
        """tests: [(stdin, expected_stdout)]. Output compared on host (whitespace-normalized).
        Returns dict(score, passed, total, verdicts)."""
        results = self.run_many([dict(code=code, stdin=i, timeout=timeout) for i, _ in tests])
        verdicts = []
        for (_, exp), r in zip(tests, results):
            if r["status"] != "ok":
                verdicts.append(r["status"])
            elif r["stdout"].split() == str(exp).split():
                verdicts.append("pass")
            else:
                verdicts.append("wrong")
        passed = verdicts.count("pass")
        score = passed / len(tests) if partial else float(passed == len(tests))
        return dict(score=score, passed=passed, total=len(tests), verdicts=verdicts)

    def grade_tests(self, code, test_code, timeout=6.0, module="solution", prefix=""):
        """Function/pytest-style grading. `test_code` contains `def test_*():` functions
        (and/or bare asserts at top level) that `from solution import ...`.
        Each test runs in-process; results are reported on a line tagged with a random
        nonce that is read from stdin before the model's code is imported, so the
        solution can't simply print a fake 'all passed' line.
        Known residual hack: returning objects whose __eq__ is always True. We reject
        results of non-builtin types inside the asserts' == via a guard (see HARNESS)."""
        hit = SUSPICIOUS.search(re.sub(r"if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", "", code))
        if hit:
            return dict(score=0.0, passed=0, total=0, verdicts=["flagged:" + hit.group(0)], all_pass=False)
        nonce = uuid.uuid4().hex
        prog = HARNESS.format(sol=repr(prefix + "\n" + code), tests=repr(test_code), module=module)
        r = self.run(prog, stdin=nonce + "\n", timeout=timeout, mem_mb=512)
        tag = f"@@RESULT-{nonce}@@"
        line = next((l for l in r["stdout"].splitlines() if l.startswith(tag)), None)
        if line is None:
            return dict(score=0.0, passed=0, total=0, verdicts=[r["status"]], stderr=r["stderr"][-500:])
        res = json.loads(line[len(tag):])
        total = len(res)
        passed = sum(v == "pass" for v in res.values())
        return dict(score=passed / total if total else 0.0, passed=passed, total=total,
                    verdicts=list(res.values()), all_pass=total > 0 and passed == total)

    def close(self):
        while not self.pool.empty():
            self.pool.get().kill()
        self.ex.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
