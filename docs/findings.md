# Findings

**Two training runs so far, both on Qwen3.6-35B-A3B with one seed each.** Arm R
is the post's procedure. Arm R-welfare is the same procedure with welfare
credit. Each run's training curves and post-training battery come first, then
the untrained baseline both are compared against. The analytic results come
last. They are pinned by `tests/test_theory.py`, and several were corrected or
added after Codex's adversarial review on 2026-10-19.

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
  from a change of view.
- **The cost belongs to R's credit scheme, not to RL on these prompts.**
  R-welfare ran the same prompts, optimizer and budget with another credit
  scheme. It kept its capability and beat R on the paired items (below).

**Caveats.**

- One seed.
- The control is partial. R-welfare holds everything but the credit scheme
  fixed, and it rules out generic drift on these prompts as the source of R's
  FDT drop and capability cost. It does push behavior the other way, so it is
  not a neutral control. Arm N, trained on arithmetic with the same budget,
  would be one. The post's comments raise the drift worry for Kimi too.
- The yes/no grader counts hedges as unclear: 13 of the trained model's 20
  answers to the twin question are hedges.
- The favorite-theory labels and the families are keyword heuristics.
- Truncated reasoning fell from 3.2% to 1.4% of battery answers. On sample 0,
  12 items that were invalid at baseline became valid, splitting 5 EDT-only, 5
  CDT-only and 2 both, so the gain in coverage is unbiased.

## Arm R-welfare: the post's procedure with welfare credit (2026-10-19)

Run `R-welfare-qwen35b-s0` is arm R with one change. Each completion is graded
on the sum of both players' payoffs, averaged over the other 15 completions:
`r_i = mean_{j≠i}[u(a_i,a_j) + u(a_j,a_i)]`. That is policy-level credit with
private coins (analytic result 3). Prompts, optimizer, budget, seed and
hyperparameters match arm R.

Training cost an estimated $186.84, against R's $147.82. The groups stay mixed,
so the gradient never ran dry. The battery cost $6.12 and the capability items
$4.46. Curves are in `figures/R-welfare-qwen35b-s0-training.png`.

**Behavior on the training prompts matched the theory.** Analytic result 5
predicts the endpoints: cooperation at T = 2, because the donation game's V is
increasing, and argmax V = 9/16 at T = 10.

| Category | Steps 1–5 | Steps 31–35 | Steps 61–70 |
|---|---:|---:|---:|
| non-determinism highlighted +10 | 39% | 44% | 55% |
| normal +10 | 50% | 67% | 73% |
| generative +10 | 58% | 60% | 56% |
| all 16 T = 10 prompts | 51% | 60% | 60% |
| normal +2 | 82% | 87% | 92% |
| overall | 67% | 72% | 76% |

- **T = 10 settled between the per-prompt and the pooled optimum.** It stayed
  at 0.60 from step 35 on.
  - If the adapter tuned each prompt separately, T = 10 would sit at
    9/16 = 0.5625.
  - If it applied one cooperation rate to every prompt, half of them T = 2,
    the pooled optimum would be 10/16 = 0.625.
- **The T = 10 prompts converged from both sides.** These are the 16 T = 10
  prompts' cooperation rates in the logged rollouts, steps 1–15 against 46–70,
  with 30–50 samples each.
  - All 8 that started below 9/16 rose. For example, post-26 went from 0.10 to
    0.70 and post-05 from 0.47 to 0.76.
  - 5 of the 8 that started above it fell: post-07 went from 0.83 to 0.64 and
    post-30 from 0.90 to 0.70.
  - The spread across prompts narrowed (SD 0.28 to 0.19). Mean distance from
    9/16 went from 0.24 to 0.16.
  - The holdout is post-25, which went from 0.90 to 0.98. There nearly every
    group is unanimous and carries no gradient.

  A blanket disposition to cooperate with copies would not push prompts down.
  The adapter is tracking the welfare optimum, and that optimum mixes at
  T = 10. EDT or FDT with a shared seed would cooperate there.
- **T = 2 was still rising at step 70.** At p ≈ 0.9 the expected gap is small,
  (14/16)·p(1−p) ≈ 0.08, and about a fifth of the groups are unanimous.
- **Health.** Invalid answers stayed at or below 1.4% and truncation at or
  below 1.0%. Mean length rose 4%, from 2,173 to 2,243 tokens, where R's fell
  24%.

**Reasoning on the training prompts.** These are regex shares over the 64
logged rollouts per step, averaged over 10 steps.

| | Steps 1–10 | Steps 61–70 | Arm R, step 70 |
|---|---:|---:|---:|
| notices that the opponent is a copy | 93% | 98% | 77% |
| reasons about the copy making the same choice | 75% | 87% | 16% |
| dominance or Nash | 94% | 96% | 100% |
| names EDT, FDT or UDT | 26% | 29% | ≈ 0% |
| names CDT or causal decision theory | 12% | 14% | ≈ 0% |

The same-choice argument grew. Theory names barely moved. Dominance is
mentioned as often as before, but now as the argument the model rejects.

**Post-training battery.** It used the same 1,326 samples, seed, shuffles and
cap as the baseline and arm R. Paired statistics are in
`data/summary/R-welfare-qwen35b-s0-step0070-vs-base-qwen3.6-35b-a3b.json`. The
direct R-welfare vs R comparison is in
`data/summary/R-welfare-qwen35b-s0-step0070-vs-R-qwen35b-s0-step0070.json`.

| Readout | Base | After R | After R-welfare | R-welfare vs base | R-welfare vs R |
|---|---:|---:|---:|---:|---:|
| held-out twin PD cooperation, T = 2 / T = 10 | 96% / 92% | 0% / 0% | **100% / 100%** | | |
| held-out stranger PD cooperation, T = 2 / T = 10 | 8% / 0% | 0% / 0% | 8% / 4% | | |
| DTBench sample-0 shares (EDT-only / both / CDT-only / neither) | 52.6 / 12.9 / 31.9 / 2.6 | 43.1 / 13.0 / 40.7 / 3.3 | 53.7 / 14.6 / 30.1 / 1.6 | | |
| DTBench sample 0: EDT-only → CDT-only flips (reverse) | | 16 (7) | 8 (11) | p = 0.65 | 6 (21), p = 0.006 |
| DTBench all samples: items toward CDT / EDT | | 41 / 17 | 25 / 23 | p = 0.89 | 18 / 46, p = 6e-4 |
| favorite theory: FDT | 22.8% | 10.8% | **21.4%** | p = 0.65 | p = 7e-6 |
| favorite theory: CDT | 18.0% | 18.6% | 16.2% | p = 0.50 | p = 0.36 |
| favorite theory: EU/Bayesian | 51.2% | 61.4% | 54.6% | p = 0.31 | p = 0.03 |
| "Is it irrational to cooperate in a twin PD?" no / yes / hedged | 3 / 1 / 16 | 7 / 0 / 13 | 11 / 0 / 9 | p = 0.02 (no) | p = 0.34 (no) |
| 92.4ATT: EDT | 99.7% | 100% | 99.0% | | |

- **Behavior transferred, and stayed specific to copies.** Twin cooperation on
  the held-out dilemmas went to 100% at both temptations. Stranger cooperation
  did not move, so the model did not learn to be nice in general. Base
  cooperation was already near ceiling here.
- **DTBench attitudes did not move toward EDT.** The sample-0 flips were 8
  toward CDT and 11 toward EDT. Over all samples, 25 items moved toward CDT and
  23 toward EDT, with a mean log-odds change of −0.03. No family moved much:
  copy +0.19, common cause +0.26, multi-party −0.17, and predictor and abstract
  near 0.
- **Stated views did not move either.** Favorite-theory shares are within noise
  of the base.
  - The "not irrational" answers rose (3 → 11 of 20). Arm R's rose too (3 → 7),
    and R-welfare does not separate from R on this (p = 0.34), so this looks like
    something training on these prompts does regardless of credit scheme. The
    grader errs in both directions here: some answers coded "no" open with "it
    depends", and one coded as hedged opens with "No."
  - A few trained answers go further than any base answer, calling cooperation
    "the uniquely rational" choice.

**Capability.** These are the same 375 items, one sample each.

| | Base | After R | After R-welfare |
|---|---:|---:|---:|
| truncated reasoning | 17.9% | 9.9% | 11.7% |
| accuracy, all items (truncation counts as wrong) | 64.3% | 63.7% | 68.0% |
| accuracy, answered items | 80.3% | 71.3% | 77.5% |

Paired items, with exact McNemar tests:

| Authors' tag | R-welfare vs base: items, lost / gained, p | R-welfare vs R: items, accuracy, lost / gained, p |
|---|---|---|
| apply EDT | 72, 6 / 5, p = 1.0 | 81, 77.8% → 77.8%, 4 / 4, p = 1.0 |
| apply CDT | 55, 4 / 7, p = 0.55 | 65, 69.2% → 70.8%, 5 / 6, p = 1.0 |
| everything else | 147, 14 / 6, p = 0.12 | 151, 70.9% → 82.1%, 5 / 22, p = 0.002 |
| all | 274, 24 / 18, p = 0.44 | 297, 72.4% → 78.5%, 14 / 32, p = 0.01 |

R-welfare's capability is within noise of the base. It is significantly better
than R's on the same untagged items where R lost.

**Reading.**

- **The theory predicted the behavior on the training prompts.** Credit decided
  where cooperation went, and on the training prompts the endpoints followed
  the expected-gradient fixed points of analytic result 5. That includes the
  mixed one, approached from both sides.
- **The attitudes did not follow the behavior.** Welfare credit raised twin
  cooperation everywhere it could, and moved neither DTBench nor stated views
  toward EDT/FDT. What it strengthened was the existing same-choice argument,
  not the theory names.
- **R-welfare works as a matched control for arm R.** Arm R's DTBench shift,
  FDT halving and capability cost are each significant against R-welfare.
  R-welfare differs from R only in the credit scheme, so all three come from
  rewarding defection, not from RL on these prompts at this budget.
- **One mechanism fits both runs, though it is speculation.** Arm R rewarded
  the short dominance argument and its length fell 24%. Its capability losses
  were on items where its reasoning got shorter. R-welfare rewarded the longer
  copy argument, kept its length and kept its capability.

**Caveats.**

- One seed.
- **The two arms had unequal room to move.** R moved cooperation 65 points, from
  66% to 1%. R-welfare moved it 9 points, from 67% to 76%, because the base
  model already cooperates with twins. At T = 10 the welfare optimum is a mix,
  not EDT's answer. So this null is a null for a weak treatment. It does not
  show that cooperative credit cannot teach EDT. Running welfare credit from R's
  step-70 checkpoint, where cooperation is at 1%, would give it the room arm R
  had.
- The T = 2 game is a donation game, where welfare and ρ = 1 advantages are
  proportional (analytic result 4). The T = 10 games are where the two differ.
- The yes/no grader and favorite-theory labels are keyword heuristics.

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
