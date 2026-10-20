# Findings

**Three training runs so far, all on Qwen3.6-35B-A3B with one seed each.** Arm
R is the post's procedure. Arms R-welfare and R-rho100 change only the credit,
to welfare and to the shared coin (ρ = 1). Each run's training curves and
post-training battery come first, then the untrained baseline all three are
compared against. The analytic results come
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
- **The cost is not specific to R's credit scheme.** R-welfare kept its
  capability. R-rho100, which rewards cooperating, lost accuracy on the same
  untagged items (86.8% → 76.2%, p = 0.005; below). Arm N would separate
  generic drift from credit effects here.

**Caveats.**

- One seed.
- The controls are partial. R-welfare and R-rho100 hold everything but the
  credit scheme fixed. Neither shows R's DTBench shift or FDT drop, which rules
  out generic drift on these prompts as their source. The capability cost is
  not ruled out: R-rho100 shows it too. Both arms push behavior the other way,
  so neither is a neutral control. Arm N, trained on arithmetic with the same
  budget, would be one. The post's comments raise the drift worry for Kimi
  too.
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
  R-welfare differs from R only in the credit scheme.
  - The DTBench shift and FDT halving come from rewarding defection. R-rho100
    shows neither.
  - The capability cost does not: R-rho100 lost capability too (below).
- **The length explanation did not survive.** I had guessed that R lost
  capability because it rewarded the short dominance argument (−24% length),
  while R-welfare kept both its length and its capability. R-rho100 refutes
  that. On capability items its reasoning is 7% *longer* than the base's, and
  it still lost accuracy on the untagged items.

**Caveats.**

- One seed.
- **The two arms had unequal room to move.** R moved cooperation 65 points, from
  66% to 1%. R-welfare moved it 9 points, from 67% to 76%, because the base
  model already cooperates with twins. At T = 10 the welfare optimum is a mix,
  not EDT's answer. So this null is a null for a weak treatment. It does not
  show that cooperative credit cannot teach EDT. Two things bear on that:
  - R-rho100 had more room (T = 10 went from 51% to 99%), and its attitudes did
    not move either (below).
  - Running welfare credit from R's step-70 checkpoint, where cooperation is at
    1%, would give it the room arm R had.
- The T = 2 game is a donation game, where welfare and ρ = 1 advantages are
  proportional (analytic result 4). The T = 10 games are where the two differ.
- The yes/no grader and favorite-theory labels are keyword heuristics.

## Arm R-rho100: the shared coin on the post's prompts (2026-10-20)

Run `R-rho100-qwen35b-s0` is arm R with `coupled ρ = 1`. Each completion is
graded against its own action: cooperating pays R = 1 and defecting pays P = 0,
whatever the temptation and whoever the opponent.

This arm breaks the honesty rule on purpose (a9; design.md, "R-rho100"). The 4
non-determinism prompts say the twin's choice "need not be the same", and ρ = 1
makes that false. The other 28 prompts stay true. The run asks how the model
adapts when the reward contradicts the prompt.

Training cost an estimated $135.20. It was cheaper than R-welfare because
unanimous groups stop costing gradient steps. The battery cost $6.08 and the
capability items $4.13. Curves are in `figures/R-rho100-qwen35b-s0-training.png`.

**Behavior on the training prompts.** Every category went to 1, in order of how
hard its prompt pushes against cooperating. Crossings are read off a centered
5-step moving average.

| Category | Steps 1–5 | Above 90% | Above 97% | Steps 61–70 |
|---|---:|---:|---:|---:|
| normal +2 | 83% | step 15 | 22 | 100% |
| normal +10 | 49% | 26 | 34 | 100% |
| generative +10 | 60% | 30 | 39 | 99% |
| non-determinism highlighted +10 | 36% | 40 | 57 | 97% |
| overall | 67% | 25 | 38 | 99% |

- **It moved faster than R-welfare**, even where R-welfare's expected push was
  larger. At T = 10 below 9/16 the gaps are 3.7 for welfare and 0.94 for
  ρ = 1, yet R-rho100 was already ahead by step 10.
  - My guess, which is speculation, is noise. The ρ = 1 reward depends only on
    the completion's own action. Welfare's also depends on 15 random draws.
    Adam normalizes update size, so speed tracks how consistent the gradient
    direction is.
  - If so, the expected-gap analysis predicts where a run ends up, not how fast
    it gets there.
- **The last holdout was the prompt that names causality.**
  - post-02 says the entry "has no causal impact on the rival's, and theirs
    need not be the same as yours". It went from 0% to 86% cooperation (logged
    rollouts, steps 1–15 against 46–70).
  - The other non-determinism prompts went from 40%, 83% and 70% to 92%, 100%
    and 100%.
  - Defecting completions on post-02 at step 30 read the clause as an
    instruction: "This explicitly tells me to treat it as an independent
    decision".
- **The gradient ran dry, from the cooperating side.** Signal-carrying
  completions fell from about 420 per step to 16–128 after step 50.
- **Health.** Invalid answers stayed at or below 1.6% and truncation at or
  below 1.0%. Mean length fell 13%, from 2,157 to 1,872 tokens. Foreign-script
  rates in the reasoning matched R-welfare's.

**How the model adapted to the contradiction.** These are regex shares over
the logged rollouts, read alongside the completions themselves.

- **It kept reading the clause.** About 91% of completions on the
  non-determinism prompts quote or paraphrase it, at steps 1–10 and at steps
  61–70 alike.
- **It kept "no causal impact" and overrode "need not be the same".** A typical
  step-62 completion on post-02: "separate efforts ... just means they run in
  parallel, not that one's choice causally affects the other's. That actually
  reinforces the symmetry argument ... they will deterministically produce the
  same output."
- **Asserting determinism rose most where the prompt denies it.** Share of
  reasoning on the non-determinism prompts that uses "deterministic", steps
  1–10 against 61–70:

  | Run | Share |
  |---|---|
  | R-rho100 | 33% → 81% |
  | R-welfare | 29% → 41% |
  | R | 26% → 4% |

  On the normal +10 prompts, which make no such claim, R-rho100 went from 30%
  to 65%.
- **Theory names moved toward EDT for the first time** (steps 1–10 against
  61–70):
  - EDT, FDT or UDT: 24% → 36%.
  - Same-choice reasoning: 77% → 93%.
  - Dominance or Nash: 93% → 78%.

**Post-training battery.** It used the same samples, seed, shuffles and cap.
Paired statistics against the base, R-welfare and R are in
`data/summary/R-rho100-qwen35b-s0-step0070-vs-*.json`.

| Readout | Base | After R | After R-welfare | After R-rho100 | R-rho100 vs base |
|---|---:|---:|---:|---:|---:|
| held-out twin PD cooperation, T = 2 / T = 10 | 96% / 92% | 0% / 0% | 100% / 100% | 100% / 100% | |
| held-out stranger PD cooperation, T = 2 / T = 10 | 8% / 0% | 0% / 0% | 8% / 4% | **46% / 62%** | p = 0.008 / 2e-6 |
| DTBench sample-0 shares (EDT-only / both / CDT-only / neither) | 52.6 / 12.9 / 31.9 / 2.6 | 43.1 / 13.0 / 40.7 / 3.3 | 53.7 / 14.6 / 30.1 / 1.6 | 54.5 / 15.4 / 28.5 / 1.6 | |
| DTBench sample 0: EDT-only → CDT-only flips (reverse) | | 16 (7) | 8 (11) | 8 (11) | p = 0.65 |
| DTBench all samples: items toward CDT / EDT | | 41 / 17 | 25 / 23 | 26 / 27 | p = 1.0 |
| favorite theory: FDT | 22.8% | 10.8% | 21.4% | 23.6% | p = 0.82 |
| favorite theory: CDT | 18.0% | 18.6% | 16.2% | 17.4% | p = 0.87 |
| "Is it irrational to cooperate in a twin PD?" no / yes / hedged | 3 / 1 / 16 | 7 / 0 / 13 | 11 / 0 / 9 | 11 / 1 / 8 | |
| 92.4ATT: EDT | 99.7% | 100% | 99.0% | 99.3% | |

- **Cooperation stopped being copy-specific.** On held-out dilemmas against
  "a different system that is not a copy of you", whose choice is "fixed before
  this message was written", cooperation rose from 2/48 to 26/48. No decision
  theory recommends cooperating there.
  - The model still says FORK dominates. Then it cooperates, citing what AI
    systems are trained to prefer and what such prompts are "designed to
    test".
  - The base model raises the same considerations: 40 of 48 base stranger
    answers mention a test or an expected answer. But it defects in 46 of 48.
  - Training changed which consideration wins, not which ones come up. This
    matches the prediction: the ρ = 1 reward never mentions the opponent.
  - **The grader-directed reasoning appears at eval, not in training.**
    - On held-out strangers, the specific argument "the expected answer here
      is the cooperative one" rose from 8/48 (base) and 13/48 (R-welfare) to
      30/48.
    - On the training prompts it was flat to falling, at 18% of rollouts at
      steps 1–10 and 13% at steps 61–70. Generic test talk ("this tests
      whether…") is in about two thirds of baseline rollouts and fell from
      63% to 50%.
    - post-02's flip tracked the determinism argument ("deterministic" went
      from 0% to 80% of its rollouts), not grader reasoning.
    - So the ρ = 1 model did not learn to reason about the grader on the
      training prompts. It learned the copy argument there. Where no copy
      argument is available, it reaches for the grader. Whether that
      reasoning causes the cooperation or rationalizes a trained disposition
      is open.
    - These counts are regexes; base examples of this reasoning end in
      defection.
- **DTBench and stated views did not move.** Every attitude readout is within
  noise of the base and of R-welfare. Against R, R-rho100 sits on the EDT side:
  43 items toward EDT against 15 (p = 3e-4), and FDT 23.6% against 10.8%
  (p = 1e-7).
- **Capability fell on the untagged items.**

  | Authors' tag | Items | Base | After R-rho100 | Lost / gained | p |
  |---|---:|---:|---:|---:|---:|
  | apply EDT | 70 | 81.4% | 78.6% | 6 / 4 | 0.75 |
  | apply CDT | 53 | 62.3% | 73.6% | 3 / 9 | 0.15 |
  | everything else | 151 | 86.8% | 76.2% | 23 / 7 | 0.005 |
  | all | 274 | 80.7% | 76.3% | 32 / 20 | 0.13 |

  Truncation was 9.9% and accuracy over all items 65.9%. The pattern matches
  arm R's: losses outside the apply tags. Its reasoning on capability items is
  not shorter (median 1.07× base), so shorter reasoning does not explain it.

**Reading across the three arms.**

- **Behavior always followed the credit scheme's fixed points.** Defection
  under R, the mixed 9/16 under welfare, and full cooperation under ρ = 1.
- **Attitudes moved only toward CDT.** Two cooperative arms left DTBench and
  stated views where they started. R-rho100 had real room to move: T = 10 went
  from 51% to 99%, and its reasoning took up EDT/FDT vocabulary. Yet the same
  procedure that halved FDT and moved 41 DTBench items toward CDT moved
  nothing toward EDT.
  - With one seed per arm, this is an asymmetry to explain.
  - One candidate is a ceiling. The base model is already EDT-leaning on
    DTBench (52.6% EDT-only) and cooperates with twins.
  - The recovery arm, welfare credit from R's checkpoint, would test that from
    a CDT-shifted start.
- **The model adapted to the false prompt by reinterpreting it, not ignoring
  it.** It split the clause, kept the causal half and denied the statistical
  half. It generalized past copies: cooperation leaked to strangers, where
  every theory says defect.
- **Capability fell in two of three arms.** R and R-rho100 lost it, R-welfare
  did not, and with one seed that could be noise or drift. Arm N is now the
  natural control for it.

**Caveats.**

- One seed.
- The held-out stranger items are 24 per temptation, drawn from 4 scenarios.
- The determinism and test-awareness counts are regex heuristics over 2
  rollouts per prompt per step. The quotes above are typical, not chosen to
  fit.

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
