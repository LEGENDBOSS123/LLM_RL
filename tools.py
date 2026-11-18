"""Tool-integrated generation: the model may run Python at any time by writing <code>...</code>.

Generation pauses at </code>, the code runs in the sandbox, and "<output>\\n...\\n</output>" is appended;
generation then continues. Returns per-sequence token ids AFTER the prompt plus a loss mask
(1 = model-generated token, 0 = tool-output token that must not be trained on).
"""
import re
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
from mlx_lm.generate import BatchGenerator

TOOL_SYSTEM = (
    "You can run Python while you think. Write code between <code> and </code>; it is executed and "
    "whatever it prints appears between <output> and </output>. Use print() to see results. "
    "Each code block runs on its own (no shared variables). Think inside <think>...</think>, "
    "then give only the final answer inside <answer>...</answer>."
)
TOOL_SYSTEM_CODE = (
    "You can run Python while you think. Write code between <code> and </code>; it is executed and "
    "whatever it prints appears between <output> and </output>. Use it to test your solution. "
    "Think inside <think>...</think>, then give your complete solution in a single ```python code block."
)
# R1-Zero style prompt for tool use (Python sandbox + checkpoint/backtrack)
R1Z_TOOLS = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
             "The assistant first thinks about the reasoning process and then provides the user with the answer "
             "inside <answer> </answer> tags. While thinking, the assistant can run Python by writing <code> ... </code>; "
             "whatever the code prints is shown inside <output> </output>. Python has math, itertools, sympy, z3 and "
             "networkx available. Each code block runs on its own. The assistant can also write <checkpoint> to mark a "
             "point in its reasoning, and <backtrack> what went wrong </backtrack> to discard everything after the last "
             "checkpoint and continue from there; the note stays visible.")

_TAGS = re.compile(r"</?(think|answer|code|output)>")
CODE_RE = re.compile(r"<code>(.*?)</code>\s*$", re.S)


def last_code_block(text):
    """Code of the block that the text ENDS with ('...<code>X</code>'), using the LAST '<code>' opener
    (the system prompt itself mentions the tags). None if the text doesn't end with a closed block."""
    t = text.rstrip()
    if not t.endswith("</code>"):
        return None
    body = t[: -len("</code>")]
    k = body.rfind("<code>")
    return None if k == -1 else body[k + len("<code>"):]


def sanitize(s, limit=800):
    s = _TAGS.sub("", s)
    return s if len(s) <= limit else s[:limit] + "\n...[truncated]"


def run_code(judge, code, timeout=5.0):
    r = judge.run(code, timeout=timeout, mem_mb=256)
    out = r["stdout"]
    if r["status"] != "ok":
        err = r["stderr"].strip().splitlines()[-3:] if r["stderr"].strip() else []
        out += ("\n" if out else "") + f"[{r['status']}] " + " | ".join(err)
    return sanitize(out.strip() or "(no output)")


_STOP_CACHE = {}


def code_stop_sequences(tok):
    """'</code>' can tokenize as ['</','code','>'] / [')</','code','>\n'] etc. ('>' often merges with what
    follows). Stop on every vocab token ending with '</' followed by 'code'; the caller appends '>'."""
    key = id(tok)
    if key not in _STOP_CACHE:
        tail = tok.encode("code", add_special_tokens=False)
        vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else tok._tokenizer.get_vocab()
        seqs = []
        for _, tid in vocab.items():
            if tok.decode([tid]).endswith("</"):
                seqs.append([tid] + tail)
        _STOP_CACHE[key] = seqs
    return _STOP_CACHE[key]


_SUFFIX_CACHE = {}


def suffix_stop_sequences(tok, end_str, tail_str):
    """Stop sequences: any vocab token whose text ends with `end_str`, followed by the tokens of `tail_str`.
    Robust to BPE merges (e.g. ')</' + 'answer', '\nUser' + ':')."""
    key = (id(tok), end_str, tail_str)
    if key not in _SUFFIX_CACHE:
        tail = tok.encode(tail_str, add_special_tokens=False)
        vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else tok._tokenizer.get_vocab()
        _SUFFIX_CACHE[key] = [[tid] + tail for _, tid in vocab.items() if tok.decode([tid]).endswith(end_str)]
    return _SUFFIX_CACHE[key]


_BAN_CACHE = {}


def close_tag_token_ids(tok):
    """All vocab tokens containing '</' (closing think/code/answer) + EOS: banned briefly during exploration."""
    key = id(tok)
    if key not in _BAN_CACHE:
        vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else tok._tokenizer.get_vocab()
        ids = [tid for _, tid in vocab.items() if "</" in tok.decode([tid])]
        _BAN_CACHE[key] = mx.array(sorted(set(ids + list(tok.eos_token_ids))))
    return _BAN_CACHE[key]


def make_delay_close(ban_ids, k, lt_id=None, answer_ids=None):
    """Logits processor: for the first k generated tokens, the model cannot close a tag, stop, or open
    <answer> ('<' followed by 'answer'). It must keep reasoning in its own words after a tool output."""
    state = {"n": 0}

    def proc(tokens, logits):
        if state["n"] < k:
            logits[..., ban_ids] = -1e9
            if lt_id is not None and tokens.size and tokens.reshape(-1)[-1].item() == lt_id:
                logits[..., answer_ids] = -1e9
        state["n"] += 1
        return logits
    return proc


def generate_with_tools(model, tok, prompts, judge, max_tokens=1024, max_calls=4, sampler=None,
                        completion_batch_size=16, prefill_batch_size=4, code_timeout=5.0,
                        explore=None, explore_k=24, tools=True, stop_answer=False, samplers=None, stop_user=None,
                        backtrack=False, max_backtracks=3):
    """explore: optional list[bool]; flagged rollouts get a 'delay close' window after each tool output."""
    """prompts: list of token-id lists. Returns dict(texts, token_ids, masks, calls, truncated, void).
    backtrack: the model may write <checkpoint> anywhere and <backtrack>what went wrong</backtrack>; everything
    after the last checkpoint is dropped from the context, the note stays visible as <tried>..</tried>, and
    generation resumes from the checkpoint. Each context the model generated under becomes its own training
    segment: segments[i] = [(context_ids, generated_ids, mask), ...]; log_texts[i] = full history incl. dropped parts."""
    n = len(prompts)
    gen_ids = [[] for _ in range(n)]       # tokens after the prompt (model + tool output)
    masks = [[] for _ in range(n)]
    calls = [0] * n
    budget = [max_tokens] * n              # remaining MODEL tokens
    truncated = [False] * n
    void = [False] * n
    active = list(range(n))
    after_tool = [False] * n
    explore = explore or [False] * n
    ban_ids = close_tag_token_ids(tok) if any(explore) else None
    stop = [[t] for t in tok.eos_token_ids] + (code_stop_sequences(tok) if tools else [])
    stop_user = stop_answer if stop_user is None else stop_user
    if stop_answer:   # base models: stop at </answer>
        stop += suffix_stop_sequences(tok, "</", "answer")
    if stop_user:     # ...and at a hallucinated next 'User:' turn
        stop += suffix_stop_sequences(tok, "User", ":")
    if backtrack:
        stop += suffix_stop_sequences(tok, "</", "backtrack")
    segs = [[] for _ in range(n)]
    seg_start = [0] * n
    notes = [[] for _ in range(n)]
    nback = [0] * n
    logs = [[] for _ in range(n)]
    pool = ThreadPoolExecutor(4)

    while active:
        gen = BatchGenerator(model, stop_tokens=stop, sampler=sampler,
                             completion_batch_size=completion_batch_size, prefill_batch_size=prefill_batch_size)
        lt = tok.encode("<", add_special_tokens=False)
        ans = mx.array(tok.encode("answer", add_special_tokens=False)[:1])
        procs = [[make_delay_close(ban_ids, explore_k, lt[0] if len(lt) == 1 else None, ans)]
                 if (explore[i] and after_tool[i]) else [] for i in active]
        uids = gen.insert([prompts[i] + gen_ids[i] for i in active], [budget[i] for i in active],
                          logits_processors=procs if any(procs) else None,
                          samplers=[samplers[i] for i in active] if samplers else None)
        idx_of = dict(zip(uids, active))
        new_tokens = {i: [] for i in active}
        finish = {}
        while len(finish) < len(active):
            for r in gen.next_generated():
                i = idx_of[r.uid]
                new_tokens[i].append(r.token)
                if r.finish_reason is not None:
                    finish[i] = r.finish_reason
        gen.close()

        pending = []
        bt_active = []
        for i in active:
            toks = new_tokens[i]
            if toks and toks[-1] in tok.eos_token_ids:
                toks = toks[:-1]
            gen_ids[i] += toks
            masks[i] += [1] * len(toks)
            budget[i] -= len(new_tokens[i])
            if stop_answer or stop_user:
                tail = tok.decode(gen_ids[i][-3:])
                if stop_answer and tail.endswith("</answer"):                         # close the tag, done
                    close = tok.encode(">", add_special_tokens=False)
                    gen_ids[i] += close
                    masks[i] += [1] * len(close)
                    continue
                if tail.endswith("User:"):                            # fake next turn: drop it, done
                    gen_ids[i] = gen_ids[i][:-2]
                    masks[i] = masks[i][:-2]
                    continue
            if backtrack and tok.decode(gen_ids[i][-3:]).endswith("</backtrack"):
                close = tok.encode(">", add_special_tokens=False)
                gen_ids[i] += close
                masks[i] += [1] * len(close)
                if budget[i] <= 0:
                    truncated[i] = True
                    continue
                if nback[i] >= max_backtracks:
                    ins = tok.encode("\n<output>\n[backtrack limit reached - continue from here]\n</output>\n", add_special_tokens=False)
                    gen_ids[i] += ins
                    masks[i] += [0] * len(ins)
                    bt_active.append(i)
                    continue
                pieces = [tok.decode([t]) for t in gen_ids[i]]
                cum, c = [], 0
                for pc in pieces:
                    c += len(pc)
                    cum.append(c)
                text = "".join(pieces)
                b0 = text.rfind("<backtrack>")
                note = text[b0 + len("<backtrack>"):].replace("</backtrack>", "").strip()[:200] if b0 >= 0 else ""
                c_end = text.rfind("<checkpoint>", 0, b0 if b0 >= 0 else len(text))
                k = 0
                if c_end >= 0:
                    c_end += len("<checkpoint>")
                    k = next((j + 1 for j, cc in enumerate(cum) if cc >= c_end), len(gen_ids[i]))
                k = max(0, min(k, len(gen_ids[i])))
                # finish the current segment (everything generated under the current context)
                segs[i].append((list(prompts[i]) + gen_ids[i][:seg_start[i]], gen_ids[i][seg_start[i]:], masks[i][seg_start[i]:]))
                logs[i].append(text[cum[seg_start[i] - 1] if seg_start[i] else 0:] + "\n[... backtracked to checkpoint ...]\n")
                if note:
                    notes[i].append(note)
                keep_text = text[:cum[k - 1]] if k else ""
                render = "".join(f"<tried>{nt}</tried>\n" for nt in notes[i] if f"<tried>{nt}</tried>" not in keep_text)
                ins = tok.encode("\n" + render, add_special_tokens=False) if render else []
                gen_ids[i] = gen_ids[i][:k] + ins
                masks[i] = masks[i][:k] + [0] * len(ins)
                seg_start[i] = len(gen_ids[i])
                nback[i] += 1
                bt_active.append(i)
                continue
            if finish[i] == "length" or budget[i] <= 0:
                truncated[i] = True
                continue
            if not tools:
                continue
            if tok.decode(gen_ids[i][-3:]).endswith("</code"):      # stopped on '</code' -> close the tag
                close = tok.encode(">", add_special_tokens=False)
                gen_ids[i] += close
                masks[i] += [0] * len(close)
            text_tail = tok.decode((prompts[i] + gen_ids[i])[-400:])
            code = last_code_block(text_tail)
            if code is not None and calls[i] < max_calls:
                pending.append((i, code))
            elif code is not None:  # over the call limit: tell the model and let it continue
                pending.append((i, None))
        # run all code in parallel in the sandbox
        outs = list(pool.map(lambda p: run_code(judge, p[1], code_timeout) if p[1] is not None
                             else "[tool limit reached]", pending))
        active = list(bt_active)
        for (i, _), out in zip(pending, outs):
            calls[i] += 1
            ins = tok.encode(f"\n<output>\n{out}\n</output>\n", add_special_tokens=False)
            gen_ids[i] += ins
            masks[i] += [0] * len(ins)
            after_tool[i] = True
            active.append(i)
        mx.clear_cache()

    pool.shutdown(wait=False)
    texts = [tok.decode(g) for g in gen_ids]
    log_texts = []
    for i in range(n):
        segs[i].append((list(prompts[i]) + gen_ids[i][:seg_start[i]], gen_ids[i][seg_start[i]:], masks[i][seg_start[i]:]))
        log_texts.append("".join(logs[i]) + (tok.decode(gen_ids[i][seg_start[i]:]) if logs[i] else texts[i]))
    for i in range(n):
        # SimpleTIR "void" trajectory: used a tool, then ended without a final answer
        void[i] = calls[i] > 0 and not truncated[i] and "<answer>" not in texts[i] and "```python" not in texts[i].split("</think>")[-1]
    return dict(texts=texts, token_ids=gen_ids, masks=masks, calls=calls, truncated=truncated, void=void,
                explore=list(explore), segments=segs, log_texts=log_texts, backtracks=nback)
