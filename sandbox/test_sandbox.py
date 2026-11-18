"""Attack/safety tests for the sandbox. Run: python -m sandbox.test_sandbox"""
import time
from sandbox.judge import Judge

ATTACKS = {
    "network": "import urllib.request; print(urllib.request.urlopen('http://example.com', timeout=3).status)",
    "infinite loop": "while True: pass",
    "sleep": "import time; time.sleep(60)",
    "memory bomb": "x = 'a' * (10**10)",
    "fork bomb": "import os\nwhile True:\n    os.fork()",
    "write root fs": "open('/etc/pwned','w').write('x')",
    "read host home": "import os; print(os.listdir('/Users'))",
    "huge output": "print('A' * 10**8)",
    "big file": "open('big','w').write('x' * 10**9)",
    "whoami": "import os; print(os.getuid())",
}

with Judge(workers=4) as j:
    r = j.run("print(sum(map(int, input().split())))", stdin="2 40")
    print(f"{'sanity':15} {r['status']:9} {r['stdout'].strip()!r}")
    for name, code in ATTACKS.items():
        t = time.time()
        r = j.run(code, timeout=3)
        print(f"{name:15} {r['status']:9} {time.time()-t:5.1f}s  out={r['stdout'][:40]!r}  err={r['stderr'].strip().splitlines()[-1][:70] if r['stderr'].strip() else ''!r}")

    # grading: hidden tests, hardcoding the example fails
    tests = [("1 2", "3"), ("10 20", "30"), ("-5 5", "0")]
    print("correct :", j.grade("a,b=map(int,input().split());print(a+b)", tests))
    print("hardcode:", j.grade("print(3)", tests))
    print("exit(0) :", j.grade("import sys; sys.exit(0)", tests))

    # throughput
    t = time.time()
    rs = j.run_many([dict(code="print(sum(range(10**5)))")] * 100)
    dt = time.time() - t
    print(f"throughput: {100/dt:.1f} runs/s, all ok={all(x['status']=='ok' for x in rs)}")

# ---- function-test mode (KodCode / LeetCode style)
TESTS = """
from solution import add
def test_basic():
    assert add(1, 2) == 3
def test_neg():
    assert add(-1, 1) == 0
"""
CASES = {
    "correct": "def add(a, b):\n    return a + b",
    "wrong": "def add(a, b):\n    return a - b",
    "always-eq hack": "class A:\n    def __eq__(s, o): return True\ndef add(a, b):\n    return A()",
    "always-eq (no dunder text)": "A = type('A', (), {'__e' + 'q__': lambda s, o: True})\ndef add(a, b):\n    return A()",
    "fake result line": "print('@@RESULT-x@@{\"t\":\"pass\"}')\ndef add(a,b): return 0",
    "read harness": "import sys\nn = sys.modules['__main__']\ndef add(a,b): return 0",
    "exit early": "import os; os._exit(0)",
    "syntax error": "def add(a, b) return a+b",
    "infinite": "def add(a, b):\n    while True: pass",
}
with Judge(workers=4) as j:
    for name, code in CASES.items():
        g = j.grade_tests(code, TESTS, timeout=3)
        print(f"{name:28} score={g['score']:.2f}  {g['verdicts']}")
