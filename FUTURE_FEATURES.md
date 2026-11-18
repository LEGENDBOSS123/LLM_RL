# Future Features & Experiments

Ideas collected during brainstorming, not yet implemented. Roughly ordered by priority within each section.

## Current baseline (for reference)
- Model: Qwen2.5-1.5B-Instruct bf16, M1 Pro 16 GB
- Math/logic (reasoning-gym, 8 tasks, 400 problems × 4): strict pass@1 17.3%, pass@4 38.8%, lenient 48.4%, format 34%
- Code smoke test: pass@1 ~5% overall (KodCode ~22%, LeetCode-eval ~4%)
- Reasoning behaviours in baseline math outputs (1,600 samples): alternatives 0.19%, verification 0.75%,
  backtracking 0%, "edge case" 0%

---

## 0. Research notes (Nov 2025 literature pass)
- **LoRA LR ≈ 10× full-FT LR**, also for RL; LoRA on MLP matters most; rank barely matters for RL (Thinking Machines, "LoRA Without Regret").
- **Flat length penalties can collapse accuracy** → gate length shaping on group accuracy (GRPO-λ, Dai et al. 2505.18086);
  Kimi k1.5 / ShorterBetter use group-relative length rewards. *(implemented: --len_alpha/--len_gate)*
- **Adaptive curriculum > fixed difficulty**; reasoning-gym RL transfers to MATH/GSM8K/BBH (Reasoning Gym 2505.24760).
  TODO: use reasoning-gym difficulty configs to ramp per-task difficulty.
- **Binary correctness rewards** (DeepScaleR, Open-RS, SimpleRL). DeepScaleR grows context 8K→16K→24K in stages.
- **DAPO**: dynamic sampling (done), token-level loss (done via Dr. GRPO norm), clip-higher, soft overlong punishment
  (TODO: ramp penalty near the token limit instead of hard 0).
- **Spurious rewards (2506.10947)**: Qwen2.5 improves even with *random* rewards → trust held-out/unseen tasks;
  TODO: random-reward control run to measure how much gain is real.
- **Yue et al. 2504.13837**: RL raises pass@1 but can shrink pass@k at large k. **ProRL**: long + diverse training + KL with
  periodic reference resets expands it. TODO: pass@k eval; periodic reference reset (merge LoRA into base).
- **Entropy collapse** is the main long-run failure → TODO: log policy entropy.
- **Gandhi et al. 2503.01307**: backtracking emerges via RL only if the base already shows it; priming (even with wrong
  answers) fixes it. Our 1.5B shows 0% backtracking → RL-only emergence may fail; priming is the fallback.

## 1. Sparse-reward / exploration tricks (from self-play experience)
| Feature | Why | Cost |
|---|---|---|
| **Bigger groups, fewer prompts** (P=4 × G=16 instead of 8 × 8) | Rare winning behaviours show up more often inside a group; same compute | Free |
| **Adaptive G** (G=16–32 for buckets with pass rate < 15%) | At 5% pass rate, G=8 has 66% chance of zero successes; G=32 only 19% | More generation on hard buckets |
| **Self-imitation buffer** | Replay rare successes (pass rate < 25%) in later batches so they aren't washed out | Free |
| **Per-problem EMA baseline** | Blend running per-problem reward with group mean → lower advantage variance | Free |
| **EMA reference policy** | Slowly-tracking KL anchor instead of fixed base model (target-network style) | Small |
| **EMA / Polyak weights for eval** | Smoother, usually better checkpoints | Small |
| **Entropy logging + clip-higher (DAPO)** | Detect entropy collapse; keep exploration alive | Free |
| **Temperature 1.1 test** | Makes rare tokens ("Wait", "let me check") likelier; risk of gibberish | Free |
| **Reverse curriculum / hints** | For 0% buckets, prepend X% of a reference solution, shrink X as pass rate rises (R3-style) | Small |
| **VinePPO-style values** | Monte-Carlo continuations from intermediate states → per-step credit | Expensive (generation-bound) |
| **Multiple PPO epochs per batch** with importance clipping | Reuse expensive rollouts | Medium |

Decision rule: if behaviour rates are flat by ~step 300, try G=32 or temp 1.1 before concluding "RL alone can't do it at 1.5B".

### Tree search over opening think tokens (TreeRL / TreeRPO / segment-GRPO style)
- Per problem: K openings (first N≈32 think tokens, or branch at high-entropy tokens) × M continuations each
  (e.g. 4 problems × 4 × 4 = 64 rollouts, same compute as 8×8).
- Opening value = mean reward of its continuations; opening tokens get A = V(opening) − mean V;
  continuation tokens get A = r − mean(siblings).
- Why: the opening is where strategy/brainstorming lives; tree gives low-variance credit to it.
- Watch for: generic "safe" openings that don't commit.
- Plan: run T vs run S from the same checkpoint, equal compute, compare per-hour progress + behaviour rates.

### Answer-fixer (user idea, 2026-11-14) - next experiment after diversity
- Train the model (same weights, second turn) to take (question, its own wrong answer) -> corrected answer; reward = fixed
  answer correct. Use wrong answers from groups with 10-90% pass rate (fixable), NOT all-fail groups (LTE on all-fail
  groups got 0/60). Gives a test-time draft -> fix -> fix loop. SCoRe-style; watch for "always change the answer".

## 2. Thinking-format experiments
- **Condition B – free-placement thinking**: `<think>` blocks allowed anywhere, any number of times; grading
  ignores think content; reward = correctness − λ·think_tokens (λ ≈ 0.0002). Compare against A at equal compute.
- **`<check>` blocks**: explicit verification blocks; bonus if a check catches an error and the answer is revised.
- **Budgeted thinking**: fixed token budget; reward spending more on hard problems, less on easy ones.
- **Interleaved reasoning**: think → partial answer → think → partial answer, rewarding correct intermediate answers.
- **`<note>` scratchpad**: persistent working memory across turns of multi-step tasks.

## 3. Brainstorming / cognitive-behaviour seeding (only if RL-alone fails)
- Short SFT on a few hundred traces that list 2–3 approaches, evaluate, then commit (generated by 3B or an API
  model, keep correct ones). Gandhi et al. 2025: priming behaviours works even with incorrect answers.
- Experiment: RL-only vs seeded-then-RL, measuring accuracy and behaviour frequencies.
- Consider 3B model if emergence is the main goal (much likelier to show it).

## 4. Process / dense rewards
- **Intermediate-value rewards** from reasoning-gym (chain_sum partial sums, base_conversion steps, etc.).
- **Potential-based shaping** to keep the optimal policy unchanged.

## 5. Next-token reasoning RL on plain text (RLP / RPT / Quiet-STaR style)
- Data downloaded: `data/text/openwebmath.jsonl` (7,455 docs, 80 MB), `data/text/fineweb_edu.jsonl` (12,759 docs, 78 MB).
- `find_hard_positions.py` works, but 81% of picks are weak "sentence end" positions that are unpredictable rather than reasoning-dependent.
- TODO: **learnability filter**: keep positions where 3B NLL ≪ 1.5B NLL (RHO-loss idea); drop sentence-end cue,
  up-weight strong cues (therefore / so / since / =).
- Reward: information gain = log p(next N tokens | thought) − log p(next N tokens | no thought); N ≈ 8–32.
- Optional thinking with a per-token cost, so the model learns *when* to think.
- Watch for: thoughts that just paraphrase the upcoming text.
- Other sources: Proof-Pile-2/arXiv, ProofWiki, StackExchange, Gutenberg (philosophy, detective fiction), case law.

## 6. Latent / "neuralese" reasoning
- MLX Qwen2 accepts `input_embeddings` (`mlx_lm/models/qwen2.py:142`), so it's feasible without changing the model.
- Stage 1: **Soft thinking** (training-free, probability-weighted embedding mixture) + logit-lens view in TUI.
- Stage 2: **Coconut curriculum**, replacing CoT steps with latent vectors stage by stage.
- Stage 3: RL with Gaussian/Gumbel noise on latents.
- Expect wins on search tasks (countdown, path finding, zebra puzzles), not arithmetic.

## 7. Self-play
- **Absolute Zero-style**: same model proposes coding problems and solves them; sandbox verifies;
  proposer rewarded for problems neither trivial nor impossible. Unlimited data.

## 8. Tool-integrated reasoning
- Model writes ```python inside thinking → executed in sandbox → output pasted back → keeps reasoning.
  Big expected win on arithmetic-heavy tasks (basic_arithmetic baseline 19% strict).

## 9. Chat / general reasoning
- Phase 2: IFEval-style checkable constraints (chat-shaped, verifiable by code).
- Multi-turn tasks with hidden goals (ask clarifying question, then answer).
- Phase 3: LLM-judge / rubric rewards for open chat (3B judge or API); high reward-hacking risk.
- Always: chat-sanity eval prompts to detect regressions from math/code RL.

## 10. Evaluation upgrades
- Run `forced@1` (answer forcing) on a full eval; replace regex-lenient as headline diagnostic.
- `math-verify` for GSM8K / MATH symbolic equivalence.
- Logprob-of-gold-answer as a smooth training curve.
- Regression suite: small MMLU + IFEval + BBH.
- Unseen-task eval (`data/rg_eval_unseen.jsonl`) every eval step.

## 11. Data / infra
- Pre-filter code problems whose reference solution fails (~30–60 min sandbox pass).
- Support LiveCodeBench "functional" tests (444 problems skipped).
- Trim `livecodebench_eval.jsonl` (7.8 GB) to ≤ 40 tests per problem.
- Add APPS / CodeContests if more competitive-programming volume is needed.
- Use KodCode `gpt4o_pass_rate` for curriculum ordering.
- Optional: 4-bit rollouts + bf16 training (~25% faster, slight off-policy mismatch).
- TUI: logit-lens view for latent thinking; load trained LoRA adapters for chatting.
