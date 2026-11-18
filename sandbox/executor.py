"""Runs INSIDE the sandbox container. Reads one JSON request per line on stdin:
  {"id":..., "code": str, "stdin": str, "timeout": float, "mem_mb": int}
Runs code in a fresh, rlimited python subprocess with a scratch dir, writes one JSON
result line: {"id", "status": ok|timeout|error|oom, "stdout", "stderr", "exit", "time"}.
"""
import json, os, resource, shutil, signal, subprocess, sys, tempfile, time

MAX_OUT = 64 * 1024


def limits(mem_mb, cpu_s):
    def f():
        os.setsid()
        b = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return f


def run(req):
    timeout = float(req.get("timeout", 5))
    mem = int(req.get("mem_mb", 256))
    d = tempfile.mkdtemp(dir="/tmp")
    try:
        path = os.path.join(d, "main.py")
        with open(path, "w") as f:
            f.write(req["code"])
        t0 = time.time()
        p = subprocess.Popen([sys.executable, "-I", "-S", path] if req.get("no_site") else [sys.executable, "-I", path],
                             cwd=d, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             preexec_fn=limits(mem, int(timeout) + 1), env={"PATH": "/usr/local/bin:/usr/bin", "HOME": d})
        try:
            out, err = p.communicate(req.get("stdin", "").encode(), timeout=timeout)
            status = "ok" if p.returncode == 0 else ("oom" if b"MemoryError" in err else "error")
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, err = p.communicate()
            status = "timeout"
        return dict(status=status, exit=p.returncode, time=round(time.time() - t0, 3),
                    stdout=out[:MAX_OUT].decode(errors="replace"), stderr=err[-4096:].decode(errors="replace"))
    finally:
        shutil.rmtree(d, ignore_errors=True)


for line in sys.stdin:
    try:
        req = json.loads(line)
        res = run(req)
    except Exception as e:
        req, res = (req if isinstance(req, dict) else {}), dict(status="internal", stdout="", stderr=repr(e), exit=None, time=0)
    res["id"] = req.get("id")
    sys.stdout.write(json.dumps(res) + "\n")
    sys.stdout.flush()
