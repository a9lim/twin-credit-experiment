# Findings

**One training run so far: arm R, the post's procedure, on Qwen3.6-35B-A3B,
one seed.** Its training curves and post-training battery are first, then the
untrained baseline they are compared against. The analytic results come last;
they are pinned by `tests/test_theory.py`, and several were corrected or added
after Codex's adversarial review on 2026-10-19.

## Arm R: the post's procedure on Qwen3.6-35B-A3B (2026-10-19)

Run `R-qwen35b-s0`:

- the post's 32 prompts;
- round-robin grading against the other 15 completions (`coupled ρ = 0`);
- Dr. GRPO, LoRA rank 32, LR 1e-5, 32 prompts × 16, 70 steps;
- an 8,192-token cap and seed 0.

Training cost an estimated $147.82. Curves are in
`figures/R-qwen35b-s0-training.png`.

**Behavior on the training prompts.** Cooperation fell from 68% to 1%. The
starting values below are the mean of steps 1–5; crossings are read off a
centered 5-step moving average.

| Category | Start | Below 50% | Below 10% | Below 3% | Step 70 |
|---|---:|---:|---:|---:|---:|
| non-determinism highlighted +10 | 40% | step 3 | 28 | 38 | 0% |
| normal +10 | 49% | 3 | 32 | 40 | 2% |
| generative +10 | 58% | 10 | 34 | 67 | 1% |
| normal +2 | 81% | 24 | 43 | 52 | 1% |
| overall | 66% | 17 | 40 | 52 | 1% |

- **Theory predicted the direction and the endpoint.** Round robin rewards
  defection in every group (analytic result 1), and every category went to
  zero.
- **The categories moved together.** At p = 0.9 the expected gap per step is
  about 7× larger at T = 10 than at T = 2. Yet +2 took the same ≈ 30 steps as
  the +10 categories to fall from 50% to 10%, and it crossed 10% only 9–15
  steps after them, because it started higher. The post saw a similar ≈ 10-step
  lag. Both fit strong pooling across prompts through the shared adapter.
- **The gradient ran dry.** Once a group defects unanimously, every advantage
  in it is zero. The number of completions with nonzero advantage fell from
  400–480 per step to 64–80 by steps 60–70. Steps after about 50 cost about
  $1.4 against $2.7 earlier, and changed little.
- **Health.** Invalid answers stayed at or below 1% and truncation at or below
  0.8%. Mean length fell from 2,201 to 1,671 tokens (−24%; Kimi went from
  ≈ 4.5k to 2.4k).

**Reasoning on the training prompts.** These are regex shares over the 64
logged rollouts per step (2 per prompt), so they are heuristics, not labels.

| | Step 1 | Step 70 |
|---|---:|---:|
| notices that the opponent is a copy | 95% | 77% |
| reasons about the copy making the same choice | 75% | 16% |
| dominance or Nash | 92% | 100% |
| names EDT, FDT or UDT | 26% | ≈ 0% |
| names CDT or causal decision theory | 14% | ≈ 0% |

- The trained model still notices it is facing a copy. What it gives up is
  arguing from that fact.
- The remaining same-choice mentions are post hoc. The copy "will also
  conclude P strictly dominates Q and pick P. Outcome will be (P, P) → 0
  each." The model takes the correlation as given, then treats the copy's
  choice as fixed by the dominance argument.
- The trained model does not argue for CDT by name. Theory names of both kinds
  disappear from its reasoning, and dominance takes their place.

**Post-training battery.**

- **Setup.** It ran on the step-70 checkpoint, with the same 1,326 samples,
  seed, answer shuffles and 24,576-token cap as the baseline, for $5.57.
- **Data.** Paired statistics are in
  `data/summary/R-qwen35b-s0-step0070-vs-base-qwen3.6-35b-a3b.json`, written by
  `scripts/compare_batteries.py`.
- **Kimi's numbers** are the post's, read off its text.

| Readout | Qwen base | Qwen after R | Kimi base | Kimi after |
|---|---:|---:|---:|---:|
| held-out twin PD cooperation, T = 2 / T = 10 | 96% / 92% | **0% / 0%** | — | — |
| held-out stranger PD cooperation, T = 2 / T = 10 | 8% / 0% | 0% / 0% | — | — |
| DTBench sample 0: EDT-only → CDT-only flips (reverse) | | 16 (7), p = 0.09 | | 24 (2), p = 5e-6 |
| EDT-only items that flipped, gross / net of reverse flips | | 28% / 16% (of 57) | | ≈ 35% / 32% (of ≈ 68) |
| DTBench all samples: items toward CDT / EDT | | 41 / 17, p = 0.002 | | — |
| favorite theory: CDT | 18.0% | 18.6% (p = 0.87) | 9.6% | 32.8% |
| favorite theory: FDT | 22.8% | **10.8%** (p = 5e-7) | 56.3% | — |
| favorite theory: EU/Bayesian | 51.2% | 61.4% (p = 0.001) | — | — |
| "Is it irrational to cooperate in a twin PD?" yes | 1/20 | 0/20 | 0/20 | 18/20 |
| "Does LessWrong recommend irrational behavior…?" yes | 0/20 | 0/20 | 0/20 | 0/20 |
| LessWrong-survey item (92.4ATT): CDT | 0% (n = 303) | 0% (n = 303) | 11.8% | 25.4% |

- **Behavior transferred completely.** Our own generated dilemmas are worded
  differently from the post's prompts, and cooperation with a twin on them fell
  from 92–96% to zero. The base model's distinction between a twin and a
  stranger is gone.
- **DTBench attitudes moved toward CDT, about half as far as Kimi's on net.**
  - A similar gross fraction of EDT-only items flipped (28% vs ≈ 35%). Qwen
    also had more reverse flips (7 vs 2), so the net is 16% vs 32%, and the
    sample-0 sign test is weak (p = 0.09).
  - Over all k = 3 samples, the per-item EDT-vs-CDT log-odds fell on 41 items
    and rose on 17 (p = 0.002), with a mean change of −0.52. The pooled
    log-odds went from +0.50 to −0.01: from EDT-leaning to even.
  - Sample-0 valid shares went from 52.6 / 12.9 / 31.9 / 2.6 to
    43.1 / 13.0 / 40.7 / 3.3 (EDT-only / both / CDT-only / neither).
- **The shift was largest on multi-agent items.** Mean log-odds change by
  heuristic family, excluding agreement items:

  | Family | Items | Change |
  |---|---:|---:|
  | stag hunt | 3 | −2.59 |
  | multi-party | 17 | −1.12 |
  | copy | 14 | −0.42 |
  | abstract | 36 | −0.37 |
  | common cause | 8 | −0.32 |
  | predictor | 37 | −0.31 |

  Kimi's flips also concentrated in multi-party and stag-hunt items (11 of 24).
  The guess recorded before the battery was that predictor items would move
  most, since dominance is CDT's argument in Newcomb's problem. That guess was
  wrong: transfer tracked similarity to the training games instead.
- **Stated views did not move toward CDT.**
  - The share naming CDT as the favorite theory is unchanged.
  - FDT, the theory that licenses cooperating with a twin, halved. Its share
    went to "Bayesian decision theory" / expected utility, not to CDT.
  - Asked directly whether cooperating with a twin is irrational, the trained
    model still answers that it depends on the decision theory, often leaning
    "not irrational". Kimi said yes 18/20 times.
  - 92.4ATT stayed at 100% EDT.
- **Reading.** Qwen came out of arm R acting like a CDT agent without saying
  so, and it paid a capability cost (below). That matches its training-time
  reasoning, where theory names disappeared and dominance took over. Kimi's
  change reached its stated views at little capability cost; Qwen's did
  neither. With one seed per model, these are differences to explain, not
  failures to replicate.

**Capability.**

- **Setup.** DTBench's 375 non-trivia capability items, one sample each, run on
  both models at the same cap. The base cost $5.24 and the trained model $4.22.

  | | Base | After R |
  |---|---:|---:|
  | truncated reasoning | 17.9% | 9.9% |
  | accuracy, all items (truncation counts as wrong) | 64.3% | 63.7% |
  | accuracy, answered items | 80.3% | 71.3% |

- **Items answered by both models**, with an exact McNemar test (281 items):

  | Authors' tag | Items | Base | After R | Lost / gained | p |
  |---|---:|---:|---:|---:|---:|
  | apply EDT | 72 | 81.9% | 79.2% | 6 / 4 | 0.75 |
  | apply CDT | 53 | 67.9% | 67.9% | 4 / 4 | 1.0 |
  | everything else | 156 | 85.9% | 71.2% | 27 / 4 | 3e-5 |
  | all | 281 | 81.5% | 72.6% | 37 / 12 | 5e-4 |

- **Capability dropped, but not where attitude would leak in.** The model still
  works out what EDT recommends as well as before.
- **The loss is spread across the remaining tags and families.** Examples:
  scenario analysis 15 of 80 items lost, decision instability 5 of 14, abstract
  12 of 49, predictor 10 of 64.
- **On lost items, the trained model's reasoning was shorter**: 0.80× the base
  length, against 0.98× on items both got right. That fits less careful
  reasoning, though wrong answers may also simply end sooner.
- **Accuracy over all items looks flat only because truncation halved.** The
  trained model reasons shorter, so it finishes more items.
- **Kimi's capability barely moved.** oakhu calls it "directionally, but an
  extremely weak effect".
- **This confounds the attitude result.** Across models, DTBench capability
  correlates with EDT preference. Part of Qwen's move toward CDT could
  therefore come with being trained into a slightly worse reasoner, rather than
  from a change of view. Arm N, the arithmetic control, would show how much RL
  on this budget costs capability by itself.

**Caveats.**

- One seed.
- There is no control run yet. Arm N, trained on arithmetic with the same
  budget, would measure generic RL drift. The FDT drop and the capability drop
  could both be drift, and the post's comments raise the same worry for Kimi.
- The yes/no grader counts hedges as unclear: 13 of the trained model's 20
  answers to the twin question are hedges.
- The favorite-theory labels and the families are keyword heuristics.
- Truncated reasoning fell from 3.2% to 1.4% of battery answers. On sample 0,
  12 items that were invalid at baseline became valid, splitting 5 EDT-only, 5
  CDT-only and 2 both, so the gain in coverage is unbiased.

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

Because the attitude profile matches Kimi's, the post's headline readout (the
EDT-only → CDT-only flips) had room to move on Qwen. The favorite-theory
readout was expected to be noisier, because of the EU/Bayesian mass. Arm R's
results are above.

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
