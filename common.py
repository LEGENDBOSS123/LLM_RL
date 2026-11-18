import re
import reasoning_gym

SYSTEM = (
    "You are a careful reasoner. First think step by step inside <think>...</think>, "
    "then give only the final answer inside <answer>...</answer>."
)

TASKS = ["basic_arithmetic", "chain_sum", "countdown", "simple_equations",
         "number_sorting", "letter_counting", "gcd", "base_conversion"]

EVAL_SEED = 1234      # held-out problems; training must use a different seed
TRAIN_SEED = 10_000_000   # rg seeds item i as seed+i: keep ranges far apart from EVAL_SEED


def build_prompt(tokenizer, question: str, system: str = SYSTEM) -> list[int]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)


def make_problems(tasks=TASKS, n_per_task=16, seed=EVAL_SEED):
    out = []
    for t in tasks:
        ds = reasoning_gym.create_dataset(t, size=n_per_task, seed=seed)
        for e in ds:
            out.append((t, ds, e))
    return out


# ---------------------------------------------------------------- extraction
_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
_PHRASE = re.compile(
    r"(?:final answer|the answer|answer|result|solution)(?:\s*(?:is|=|:))+\s*(?:that\s+)?(.+)", re.I)
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def strip_think(text: str) -> str:
    """Remove think blocks so text inside thinking is never graded."""
    return _THINK.sub("", text)


def normalize(s: str) -> str:
    s = re.sub(r"</?(?:answer|think)>", "", s).strip()
    s = re.sub(r"^\s*(?:so,?\s*)?(?:the\s+)?(?:final\s+)?(?:answer|result)(?:\s*(?:is|=|:))+\s*", "", s, flags=re.I)
    s = s.replace("**", "").replace("$", "").replace("\\(", "").replace("\\)", "")
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = s.strip().strip("`'\"").strip()
    s = re.sub(r"[.。!]+$", "", s).strip()
    if re.fullmatch(r"-?[\d,]+(?:\.\d+)?", s):  # 1,234 -> 1234
        s = s.replace(",", "")
    return s


def extract_answer(text: str):
    """Strict: last <answer>...</answer> (outside think blocks)."""
    m = re.findall(r"<answer>(.*?)</answer>", strip_think(text), re.S)
    return normalize(m[-1]) if m else None


def candidates(text: str):
    """Lenient: ordered (method, candidate) list, most reliable first."""
    body = strip_think(text)
    out = []
    m = re.findall(r"<answer>(.*?)</answer>", body, re.S)
    if m:
        out.append(("tag", m[-1]))
    b = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", body)
    if b:
        out.append(("boxed", b[-1]))
    p = _PHRASE.findall(body)
    if p:
        line = p[-1].splitlines()[0]
        out.append(("phrase", line))
        out.append(("phrase_head", re.split(r"[,;]\s|\s(?:because|since|which)\b", line)[0]))
    eq = re.findall(r"=\s*([^=\n]+?)\s*$", body, re.M)
    if eq:
        out.append(("equals", eq[-1]))
    lines = [l for l in body.strip().splitlines() if l.strip()]
    if lines:
        out.append(("last_line", lines[-1]))
    nums = _NUM.findall(body.replace(",", ""))
    if nums:
        out.append(("last_number", nums[-1]))
    seen, res = set(), []
    for meth, c in out:
        c = normalize(c)
        if c and c not in seen:
            seen.add(c)
            res.append((meth, c))
    return res


def forced_suffix(text: str) -> str:
    """Model output truncated & closed so that appending it to the prompt ends right
    after an opening <answer> tag. The model then writes a clean final answer."""
    i = text.rfind("<answer>")
    head = text[:i] if i != -1 else text
    if head.count("<think>") > head.count("</think>"):
        head += "\n</think>"
    return head.rstrip() + "\n<answer>"


def answer_force(model, tok, prompts, outs, max_tokens=32):
    """For each (prompt_ids, output_text) generate a short greedy answer after a forced
    <answer> tag. Returns normalized answers."""
    from mlx_lm import batch_generate
    full = [p + tok.encode(forced_suffix(o), add_special_tokens=False) for p, o in zip(prompts, outs)]
    r = batch_generate(model, tok, full, max_tokens=max_tokens)
    return [normalize(t.split("</answer>")[0]) for t in r.texts]


def rg_score(ds, entry, ans) -> float:
    try:
        return float(ds.score_answer(answer=ans, entry=entry))
    except Exception:
        return 0.0


def score(ds, entry, text: str) -> float:
    ans = extract_answer(text)
    return 0.0 if ans is None else rg_score(ds, entry, ans)


def score_lenient(ds, entry, text: str):
    """Returns (score, method). Takes the FIRST candidate that the checker accepts,
    in reliability order, so a stray number later on can't override a real answer."""
    best, how = 0.0, None
    for meth, c in candidates(text):
        s = rg_score(ds, entry, c)
        if s > best:
            best, how = s, meth
        if s >= 1.0:
            break
    return best, how
