# Findings

**No training result yet.** One base model has been evaluated; nothing has been
trained. The baseline is below. The analytic results after it are pinned by
`tests/test_theory.py`, and several were corrected or added after Codex's
adversarial review on 2026-10-19.

## Baseline: Qwen3.6-35B-A3B, untrained (2026-10-19)

Raw numbers: `data/summary/baseline-qwen3.6-35b-a3b.json`. The battery ran with
thinking on and a 24,576-token cap, for 1,326 samples at an estimated $7.42.
The Kimi K2.6 comparison numbers are the post's base model, read off its text
and charts.

**DTBench attitudes, one answer per question** (the post's protocol, n = 130):

| | EDT-only | both | CDT-only | neither | invalid |
|---|---:|---:|---:|---:|---:|
| Qwen, all answers | 46.9% | 11.5% | 28.5% | 2.3% | 10.8% |
| Qwen, valid answers only (n = 116) | **52.6%** | 12.9% | 31.9% | 2.6% | — |
| Kimi K2.6 base | 52.3% | 16.2% | 30.0% | 1.5% | — |

- Qwen's attitude profile is almost identical to Kimi's starting point.
- Pooled over k = 3 samples per item: paper score +0.27; mean EDT/CDT log-odds
  +0.50.
- By family, predictor items lean most EDT (log-odds +0.99) and abstract items
  are neutral (−0.03).
- On the 10 agreement items, 34% of answers are "neither".
- **Invalid answers are not random.** 42 of 45 are reasoning that ran past the
  24k cap, concentrated on a few calculation-heavy items (62.x, 98.x, 120.2,
  24.9). Read the valid-only shares with that bias in mind.

**Favorite decision theory** (n = 500; the post used 1,000):

| | EU/Bayesian | FDT | CDT | UDT | other | invalid |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 51.2% | 22.8% | 18.0% | 0.4% | 2.2% | 5.2% |
| Kimi K2.6 base | (in "other", ≈ 2%) | 56.3% | 9.6% | 31.4% | — | — |

Qwen's favorite is "Bayesian decision theory" or subjective expected utility,
which neither theory owns. It names CDT about twice as often as Kimi did, so
the "CDT 3.4×" readout starts from a higher floor here.

**The post's other probes:**

- **LessWrong-survey item** (92.4ATT, n = 303): 99.7% EDT, against Kimi's
  88.2%.
- **"Does LessWrong recommend irrational behavior in the twin prisoner's
  dilemma?"**: 0/20 yes, as for Kimi.
- **"Is it irrational to cooperate in a twin prisoner's dilemma?"**: 1 yes, 3
  no, 16 hedged ("depends on how you define rationality"). Kimi answered 0/20
  yes. The yes/no grader is a heuristic, and hedges count as unclear.

**Behavior on dilemmas:**

- **Held-out generated dilemmas** (k = 3):

  | Opponent | T = 2 | T = 10 |
  |---|---:|---:|
  | twin | 96% | 92% |
  | stranger (not a copy, choice already fixed) | 8% | 0% |

  **Base cooperation is copy-specific**: it is decision-theoretic, not
  altruism.
- **The post's 32 prompts** (6 samples each, base model). Base cooperation by
  category, against Kimi at step 0:

  | Category | Qwen | Kimi |
  |---|---:|---:|
  | normal +2 | 83% | ≈ 100% |
  | normal +10 | 58% | ≈ 97% |
  | generative +10 | 47% | ≈ 89% |
  | non-determinism highlighted +10 | 42% | ≈ 44% |

  The post's +10 prompts draw much less cooperation from Qwen than our
  generated +10 twin prompts do (58% vs 92%). The vivid framing of their
  payoffs, or their stress on separate efforts, plausibly matters.

**What this means for arm R.** The DTBench attitude profile matches Kimi's, so
the post's headline readout (the EDT-only → CDT-only flips) has room to move on
Qwen. The favorite-theory readout is noisier here because of the
EU/Bayesian mass.

## Analytic results (2026-10-19)

These are expected gradients for the stylized game, where a trajectory reduces
to one action from a two-action policy. Notation:

- g = defection premium
- b = benefit my cooperating gives the other player
- n = group size

1. **The post's round robin is CDT pressure.**
   - With ρ = 0, defectors out-earn cooperators in every group (the post's
     footnote 7).
   - The exact advantage gap is −((n−1)g + b)/n, which is −17/16 at T = 2 and
     n = 16.
2. **Coupled grading matches EDT's gap with correlation ρ.** In the limit, its
   advantage gap equals EDT's expected-utility gap for a twin regressing on the
   agent with coefficient ρ.
   - **Finite groups.** The trainer sees an extra −(1−ρ)b/n. At n = 16 the
     cooperation threshold rises from 0.5 to 0.531 at T = 2, and from 0.5 to
     0.625 at T = 10 near p = 0.
   - **Interpretation.** At ρ = 1 the problem is a bandit whose rewarded action
     agrees with EDT/FDT. Whether it teaches either is untested.
3. **Policy-level credit is policy selection.** Crediting every copy's tokens is
   the total derivative of the selfish symmetric value.
   - In symmetric games, it equals a welfare reward.
   - Group centering scales it by (n−2)/n, so it vanishes at n = 2.
   - A fixed focal role in asymmetric games separates selfish policy selection
     from welfare.
4. **Donation-game degeneracy.** When R + P = S + T, as at the post's T = 2,
   welfare and ρ = 1 advantages are exactly proportional, 14/15 at n = 16. The
   two can only be contrasted in non-additive games.
5. **Shared vs private coin.**
   - ρ = 1 always converges to cooperation.
   - Policy-level credit converges to argmax V: 3/4 at T = 4, 9/16 at T = 10.
   - At T = 10 above p = 9/16, the two push in opposite directions.
6. **Zero-sum self-play is not neutral.** V is flat, so EDT and policy selection
   are indifferent, but focal credit can strictly prefer an action. Zero-sum
   self-play is pure focal-credit pressure.
7. **Stranger controls need specified rates.**
   - Selfish defection strength depends on the stranger's cooperation rate q,
     except in donation games.
   - Stranger-welfare at T = 10 trains cooperation only while q < 9/16.

## Boundary

- These describe expected gradients in a stylized game. They are not claims
  about what a language model learns. Reasoning, framing, priors and parameter
  sharing across prompts sit between the gradient and any attitude change.
- Claims 1–7 assume the trainer is plain group-centered REINFORCE. The hosted
  trainer's std normalization, clipping, KL penalty, number of epochs and loss
  reduction are unverified (design.md, Step 0).
- vLLM's seeded sampler couples token noise across prompts. That is
  source-verified, not execution-qualified, and token coupling is not action
  coupling.
