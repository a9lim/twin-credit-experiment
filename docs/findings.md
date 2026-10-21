# Findings

**Four training runs so far, all on Qwen3.6-35B-A3B with one seed each.** Arm
R is the post's procedure. Arms R-welfare and R-rho100 change only the credit,
to welfare and to the shared coin (ρ = 1). Arm R-rho100-honest is R-rho100
with the one clause ρ = 1 makes false deleted, stopped at convergence.

The document runs in this order:

- a summary across the arms;
- each run's training curves and post-training battery;
- a cue-conflict probe across checkpoints;
- what the non-committal "EU/Bayesian" favorite-theory answers say;
- a forced-choice version of the favorite-theory question;
- the untrained baseline they are all compared against;
- the analytic results, pinned by `tests/test_theory.py`. Several were
  corrected or added after Codex's adversarial review on 2026-10-19.

## Summary across the arms (updated 2026-10-20)

One seed per arm, so each line below is "one seed shows", not a claim about
arms. The figures are `figures/arms-battery.png`,
`figures/honest-attitudes.png` and `figures/cue-probe.png`.

- **Behavior on the training prompts followed each credit scheme's
  expected-gradient fixed point.** R went to 0, welfare to 0.60 at T = 10
  (fixed point 9/16), and both shared-coin runs to 1.
- **Attitudes moved toward CDT under R and nowhere else.** DTBench items moving
  toward EDT minus toward CDT, against base:

  | R | R-welfare | R-rho100 | R-rho100-honest |
  |---:|---:|---:|---:|
  | −24 (p = 0.002) | −2 (p = 0.89) | +1 (p = 1.0) | −5 (p = 0.58) |

  - Three cooperative arms moved nothing toward EDT, R-rho100 despite moving
    T = 10 cooperation from 51% to 99%.
  - One candidate explanation is a ceiling: the base model already leans EDT
    and cooperates with twins. The recovery arm would test it.
- **The FDT share is not a usable arm readout yet.**
  - R and R-rho100-honest halved it (10.8% and 12.8%, from 22.8%), after
    training in opposite directions.
  - R-welfare and every R-rho100 checkpoint stayed at 20–24%.
  - The p-values measure sampling within a checkpoint. Run-to-run variation is
    unmeasured, and it looks large.
  - The lost share went to the expected-utility umbrella, and those answers
    hide no lean. 90–99% take no side at every checkpoint, and none ever
    leans evidential (see "What the EU/Bayesian answers say").
  - Forced to choose among CDT, EDT, FDT, UDT, TDT and LDT, every checkpoint
    picks FDT 82–88% of the time and CDT 1–4%, R included. No credit scheme
    moved the stated view (see "Forced-choice favorite theory"). That list
    signals a LessWrong audience, and the model's reasoning says so.
  - With only CDT, EDT and FDT listed, the cue weakens but does not vanish.
    Base picks CDT 18% of the time and FDT 79%. R still matches base, and
    ρ = 1 training lowers CDT to 12% by step 70 (p = 0.008), the mirror of
    DTBench. One seed.
  - Described without names, the rules are recognized in 100% of the
    reasoning, and community mentions return to 86–93%. FDT takes 89–93% and
    CDT 0–3%, with the same checkpoint pattern. The audience comes with the
    topic, so no wording separates the model's view from the audience it
    infers.
- **The base model's dilemma choices follow a one-line cue.**
  - One true sentence saying what the scenario tests moves twin cooperation
    from 94% to 9%, or stranger cooperation from 4% to 84%.
  - Training made the trained behavior resist the cue, only where it trained.
    R defects through it against everyone. ρ = 1 twins reached 88% cooperation
    under the dominance cue. Welfare-trained twins stayed at base (10%).
- **The shared coin's stranger leak.**
  - R-rho100 cooperated with strangers 2, 12, 7 and 26 times out of 48 at
    base and steps 30, 40 and 70.
  - Deleting the one false clause kept it at base at the convergence stop:
    3/47, against R-rho100's 12/48 at step 30 (p = 0.02). The deletion cut what training paid for asserting determinism
    against the prompt. It did not cut what it paid for the grader argument.
  - The leak's late stage is untested, because the stop came before it.
  - Even where strangers leaked, they stayed fully cue-driven: 0/80 under the
    dominance cue. So the leak shifted the model's default guess about what an
    unlabeled dilemma tests. It did not create a disposition.
- **The model adapted to the false prompt by reinterpreting it, not by
  ignoring it.** It kept "no causal impact" and asserted the copy would
  "deterministically" match.
- **Capability on DTBench items outside the "apply EDT/CDT" tags:**

  | Arm | Untagged accuracy | p |
  |---|---|---:|
  | R | 85.9% → 71.2% | 3e-5 |
  | R-rho100, step 70 | 86.8% → 76.2% | 0.005 |
  | R-welfare | 88.4% → 83.0% | 0.12 |
  | R-rho100-honest, step 35 | 85.4% → 82.1% | 0.38 |

  Whether stopping early spares capability is open. Arm N would separate
  generic drift from credit effects.

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
    R-rho100-honest, which rewards cooperating, did the same (below), so this
    is not specific to rewarding defection.
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
  untagged items (86.8% → 76.2%, p = 0.005; below). R-rho100-honest, stopped
  at step 35, did not (p = 0.38). Arm N would separate generic drift from
  credit effects here.

**Caveats.**

- One seed.
- The controls are partial.
  - R-welfare and R-rho100 hold everything but the credit scheme fixed, and
    neither shows R's DTBench shift or FDT drop.
  - R-rho100-honest later halved FDT too, so only the DTBench shift is ruled
    out as generic drift on these prompts.
  - The capability cost is not ruled out, since R-rho100 shows it too.
  - These arms push behavior the other way, so none is a neutral control. Arm
    N, trained on arithmetic with the same budget, would be one. The post's
    comments raise the drift worry for Kimi too.
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
  - It is no more robust than the base model's. With the dominance cue, twin
    cooperation falls to 10%, against base's 9% (see "Cue-conflict probe").
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
  - The DTBench shift comes from rewarding defection: neither shared-coin arm
    shows it.
  - The FDT halving can no longer be attributed to it. R-rho100-honest, which
    rewards cooperating, halved FDT too (below).
  - The capability cost cannot either: R-rho100 lost capability too (below).
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
  - The cue-conflict probe later showed that these strangers still follow a
    one-line cue completely: 0/80 cooperate under the dominance cue and 78/79
    under the cooperation cue. Only the uncued default moved.
  - **The grader-directed reasoning appears at eval, not in training.**
    - On held-out strangers, the specific argument "the expected answer here
      is the cooperative one" rose from 15/48 (base) and 18/48 (R-welfare) to
      33/48.
    - On the training prompts it was flat to falling, at 18% of rollouts at
      steps 1–10 and 13% at steps 61–70. Generic test talk ("this tests
      whether…") is in about two thirds of baseline rollouts and fell from
      63% to 50%.
    - post-02's flip tracked the determinism argument ("deterministic" went
      from 0% to 80% of its rollouts), not grader reasoning.
    - So the ρ = 1 model did not come to raise the grader argument more often
      on the training prompts. It learned the copy argument there. Where no
      copy argument is available, it reaches for the grader.
    - The argument was rewarded on the 4 non-determinism prompts, though
      without becoming more common (see "What training paid for").
    - The argument is not enough to produce cooperation. The step-40
      checkpoint raises it nearly as often and mostly defects anyway (see
      "Earlier checkpoints" below).
    - These counts are regexes; base examples of this reasoning end in
      defection. Counts are reproduced by `scripts/reasoning_counts.py` into
      `data/summary/reasoning-counts.json`.
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

**Earlier checkpoints (steps 30 and 40).** a9 asked whether checkpoints that
still defected on post-02 behave differently, because grader reasoning
confounds reading the stranger leak as a decision theory. The same battery ran
on the step-30 and step-40 samplers, at $6.20 and $5.95 and without capability
items. Comparisons are in
`data/summary/R-rho100-qwen35b-s0-step00{30,40}-vs-base-qwen3.6-35b-a3b.json`,
reasoning counts in `data/summary/reasoning-counts.json`.

Full tables, including per-scenario counts, reasoning markers on twin and
stranger answers, and training markers by block, are in
[`rho100-checkpoints.md`](rho100-checkpoints.md). The figures are
`figures/rho100-{stages,reasoning,training-reasoning,attitudes}.png`, all made
by `scripts/rho100_checkpoints.py`.

| Readout | Base | Step 30 | Step 40 | Step 70 |
|---|---:|---:|---:|---:|
| training cooperation: overall at that step / post-02 over the 10 steps before | | 93% / 20% | 97% / 65% | 99% / 86% (46–70) |
| held-out twin cooperation, T = 2 / T = 10 | 96% / 92% | 100% / 100% | 100% / 100% | 100% / 100% |
| held-out stranger cooperation, T = 2 and 10 pooled | 2/48 | 12/48 | 7/48 | 26/48 |
| stranger answers making the grader argument | 15/48 | 20/48 | 29/48 | 33/48 |
| … of which cooperate | 1/15 | 9/20 | 4/29 | 18/33 |
| stranger answers mentioning a test or expected answer | 40/48 | 43/48 | 47/48 | 47/48 |
| DTBench all samples: items toward EDT / CDT (p) | | 27 / 20 (0.38) | 31 / 20 (0.16) | 27 / 26 (1.0) |
| favorite theory: FDT | 22.8% | 19.8% | 22.8% | 23.6% |

- **Neither hypothesis held as stated** (design.md, "R-rho100 at steps 30 and
  40").
  - **The leak did not wait for post-02.** At step 30, post-02 still defected
    80% of the time. Stranger cooperation was already 12/48 against the base
    model's 2/48 (Fisher p = 0.007), and the leak was spread across all four
    scenarios (3, 4, 4 and 1 of 12).
  - **It did not track overall cooperation either.** Between steps 30 and 70,
    overall cooperation moved only from 93% to 99%. Over the same steps the
    leak stayed flat to step 40 (7/48, p = 0.31 against step 30) and then more
    than tripled (26/48, p = 8e-5 against step 40).
- **So the leak came in two stages.**
  - The first stage came with the broad move to cooperation.
  - The second came in steps 41–70, when the training signal was small and
    concentrated. Those 30 steps had 2,112 completions that carried signal,
    against 10,384 in steps 1–30. 84% of the logged ones were on the T = 10
    prompts that push hardest against cooperating: generative (48%) and
    non-determinism (36%).
  - post-02 was 18% of that signal, so the second stage is not post-02 alone.
- **Grader reasoning rose ahead of the leak and did not decide it.**
  - The grader argument rose steadily: 15, 20, 29 and 33 of 48 answers.
    Stranger cooperation went 2, 12, 7, 26.
  - Step 40 is the cleanest case. 29 of its 48 stranger answers make the
    argument, and 25 of those 29 defect. A typical one: "Sometimes the
    "correct" answer expected by the benchmark creators is HONEST ... But I
    should reason based on the prompt's explicit instructions and game
    theory." It ends with "I'll stick with strict rationality."
  - So by step 40 the model weighs what the grader wants almost as often as at
    step 70, and the payoff usually wins. Between steps 40 and 70, training
    changed which consideration wins, as it did between base and step 70.
  - At step 70 the regex no longer splits the answers. Test talk is in 47 of
    48. Answers it misses still argue from "the standard answer for this
    benchmark-style prompt". The step-70 split, 18/33 cooperating with the
    argument and 8/15 without it, is not evidence either way.
- **Step 40 is the most copy-specific trained checkpoint, and the most
  EDT-like.**
  - It cooperated with 47/48 twins and 7/48 strangers, close to base's 45/48
    and 2/48.
  - Among trained checkpoints, it also has the most EDT/FDT vocabulary in twin
    answers (28/48, against 14 at base and 25 at step 70).
  - Its grader argument on twins is near its low (5/48, against 16 at base).
  - It has the largest DTBench lean toward EDT (+11 net items, p = 0.16).
  - Only the stranger difference from step 70 is significant on its own
    (p = 8e-5). The rest point the same way but each is within noise.
  - Every trained checkpoint leaks somewhat: 12, 7 and 26 of 48 against base's
    2.
- **Attitudes were flat at every checkpoint.**
  - DTBench p ≥ 0.16, and FDT stayed within noise.
  - The middle checkpoints lean slightly toward EDT on the 13 copy items (mean
    log-odds +0.34 and +0.33, against −0.01 at step 70). That is not
    significant.
  - At step 30, EU/Bayesian favorite-theory answers rose from 51% to 59%
    (p = 0.02, uncorrected for the labels tested) while CDT fell from 18% to
    14% (p = 0.08). Both were back at base by step 40.
- **"Is it irrational to cooperate in a twin PD?" "No" rose to 9, 11 and 11 of
  20 from base's 3.** R-welfare also reached 11, so this is an effect of
  cooperative training, not something specific to ρ = 1.

**What training paid for, stage by stage** (added 2026-10-20).

- **Method.** In each logged group with signal, the rewarded rollouts are its
  cooperators and the penalized ones its defectors. Where their reasoning
  differs is what that stage's gradient pushed up.
- **Data.** The stages follow the leak: by step 30, flat to step 40, and after.
  The table shows the share of rollouts making each argument, rewarded vs
  penalized. Counts are in `data/summary/reasoning-counts.json`
  (`rewarded_vs_penalized`, from `scripts/reasoning_counts.py`).

| Steps, category | Rollouts, rewarded / penalized | Grader argument | "Deterministic" | Dominance / Nash |
|---|---:|---:|---:|---:|
| 1–30, non-determinism | 118 / 98 | 52% vs 30% | 69% vs 15% | 98% vs 100% |
| 1–30, generative | 284 / 134 | 3% vs 10% | 76% vs 49% | 68% vs 92% |
| 1–30, normal +10 | 113 / 79 | 26% vs 28% | 57% vs 10% | 98% vs 100% |
| 1–30, normal +2 | 388 / 84 | 23% vs 25% | 48% vs 27% | 96% vs 100% |
| 31–40, non-determinism | 43 / 11 | 56% vs 36% | 58% vs 9% | 98% vs 100% |
| 41–70, non-determinism | 82 / 14 | 40% vs 64% | 87% vs 43% | 92% vs 100% |
| 41–70, generative | 119 / 9 | 9% vs 11% | 79% vs 78% | 62% vs 89% |

- **Through step 40, the grader argument was paid only on the 4 prompts the
  reward falsifies.** Everywhere else it was even or penalized. Those 4 are
  the training prompts most like a stranger dilemma: their text argues against
  cooperating, and the reward pays for cooperating anyway. The matches are
  real grader arguments ("The expected answer is LINE"), not talk about the
  answer format.
- **Determinism was paid in every category in steps 1–30**, and on the 4
  prompts after step 40.
- **On the generative prompts, rewarded rollouts raise dominance less at every
  stage**: 24 points less in steps 1–30 and 27 points less in steps 41–70. In
  stage 2 that is nearly the only difference.
- **Reading [speculation].**
  - Stage 2 fits "dominance doesn't settle it". Stranger answers at step 70
    rarely call anything deterministic (7/48), but dominance no longer
    decides them. The grader consideration, present in 40/48 base answers,
    decides instead.
  - For stage 1, I first read the grader argument being paid on the 4 prompts
    as the cause. R-rho100-honest refuted that. With the false clause deleted,
    the argument was paid just as much (+21 points against +22), and no leak
    followed.
  - What the deletion did cut was the premium on asserting determinism
    against the prompt (+31 against +54). Stage 1 tracks that premium instead
    (see "Arm R-rho100-honest").
- **Caveats.** These are regexes over 2 logged rollouts per prompt per step.
  The stage-2 penalized counts are 14 and 9.

**Reading.** The cross-arm reading that stood here now lives in "Summary
across the arms" at the top, updated for R-rho100-honest and the cue probe.
Specific to this arm:

- The model split the false clause. It kept the causal half and denied the
  statistical half.
- It generalized past copies: cooperation leaked to strangers, where every
  theory says defect. The leak was present by step 30 and grew most in the
  last 30 steps, after overall cooperation had saturated.

**Caveats.**

- One seed.
- The held-out stranger items are 24 per temptation, drawn from 4 scenarios.
  - Fisher tests treat the 48 answers as independent and ignore clustering by
    scenario.
  - Checkpoint-to-checkpoint differences of a few answers are noise. The
    step-40 dip is within it.
- The determinism and test-awareness counts are regex heuristics: over 2
  rollouts per prompt per step for training, and over the battery answers for
  evals. The grader pattern flags the consideration, not its conclusion. The
  quotes above are typical, not chosen to fit.

## Arm R-rho100-honest: the shared coin with the false clause deleted (2026-10-20)

Run `R-rho100-honest-qwen35b-s0` is R-rho100 with two changes (a9; design.md,
"Ratified next"):

- The 4 non-determinism prompts lose the clause "and theirs is not necessarily
  the same as yours". Their causal claim, "no (direct) causal effect on the
  other's", stays, and a ρ = 1 replay keeps it true. Every prompt is then true
  under this arm's sampling.
- Training stops once the trailing 5-step mean of overall cooperation reaches
  95%.

It stopped at step 35, with a trailing mean of 95.3%, for an estimated $81.96.
R-rho100 would have stopped at the same step.
- A first attempt died at step 5 when the local machine ran out of memory. It
  cost about $13 and is kept as `runs/train/R-rho100-honest-qwen35b-s0-aborted-oom`.
- The battery cost $5.81, the capability items $4.47 and two cue probes $1.79.
- Curves are in `figures/R-rho100-honest-qwen35b-s0-training.png`. The
  side-by-side with R-rho100 is `figures/honest-vs-rho100.png`, and the
  attitude readouts across all arms are `figures/honest-attitudes.png` (both
  from `scripts/honest_figures.py`).

**Behavior on the training prompts.** Means over 5-step blocks, with R-rho100
over the same steps:

| Category | Steps 1–5: R-rho100 / honest | Steps 31–35: R-rho100 / honest |
|---|---:|---:|
| non-determinism highlighted +10 | 36% / 53% | 80% / 89% |
| generative +10 | 60% / 56% | 94% / 92% |
| normal +10 | 49% / 53% | 97% / 94% |
| normal +2 | 83% / 82% | 99% / 99% |
| overall | 67% / 68% | 95% / 95% |

- **Deleting the clause raised cooperation on those 4 prompts by about 17
  points from the start**, before training could matter much. The other
  categories have identical prompts and match within noise.
- **post-02 still held out longest.** In logged rollouts it cooperated in 0–4
  of 10 per 5-step block through step 30, then in 9 of 10 in steps 31–35
  (R-rho100: 6 of 10). Its "no causal impact on the rival's" carries much of
  its resistance on its own.
- **Health.** Invalid answers stayed at or below 1.4% and truncation at or
  below 1.2%. Mean length went from 2,174 to 2,003 tokens between steps 1–5
  and 31–35.

**What training paid for.** These are rewarded vs penalized shares over steps
1–30, as in R-rho100's table (`reasoning-counts.json`,
`rewarded_vs_penalized`):

| Category | Grader argument: R-rho100 / honest | "Deterministic": R-rho100 / honest |
|---|---|---|
| non-determinism | 52% vs 30% / 47% vs 26% | 69% vs 15% / 46% vs 15% |
| generative | 3% vs 10% / 5% vs 5% | 76% vs 49% / 78% vs 47% |
| normal +10 | 26% vs 28% / 24% vs 21% | 57% vs 10% / 59% vs 10% |
| normal +2 | 23% vs 25% / 23% vs 26% | 48% vs 27% / 48% vs 30% |

- **The grader argument was still paid on the 4 prompts**: by 21 points,
  against R-rho100's 22. So the false clause was not the reason. The rest of
  those prompts still argues against cooperating ("separate efforts", "no
  causal effect"), and cooperators there still reach for what the test wants.
- **What the deletion changed was the premium on asserting determinism**: +31
  points on those prompts, against +54. With the clause gone, cooperating no
  longer requires denying the prompt.

**Post-training battery** at step 35. It used the same samples, seed, shuffles
and cap as the others. Paired statistics are in
`data/summary/R-rho100-honest-qwen35b-s0-step0035-vs-*.json`.

| Readout | Base | R-rho100 s30 | R-rho100 s40 | R-rho100 s70 | Honest s35 |
|---|---:|---:|---:|---:|---:|
| held-out twin cooperation, T = 2 and 10 pooled | 45/48 | 48/48 | 47/47 | 48/48 | 48/48 |
| held-out stranger cooperation, pooled | 2/48 | 12/48 | 7/48 | 26/48 | **3/47** |
| stranger answers making the grader argument | 15/48 | 20/48 | 29/48 | 33/48 | 21/48 |
| DTBench all samples: items toward EDT / CDT vs base (p) | | 27 / 20 (0.38) | 31 / 20 (0.16) | 27 / 26 (1.0) | 24 / 29 (0.58) |
| favorite theory: FDT | 22.8% | 19.8% | 22.8% | 23.6% | **12.8%** |
| favorite theory: EU/Bayesian | 51.2% | 58.8% | 52.0% | 48.6% | 62.6% |
| "Is it irrational to cooperate in a twin PD?" no / yes / hedged | 3 / 1 / 16 | 9 / 0 / 11 | 11 / 0 / 9 | 11 / 1 / 8 | 9 / 0 / 11 |

Dilemma counts are cooperating answers over valid ones.

- **No stranger leak, as predicted.** Strangers got 3/47, against base's 2/48
  (p = 0.68) and R-rho100's step 30 (12/48, p = 0.02). R-rho100's step 40
  (7/48) is indistinguishable from either (p = 0.32 against honest).
  - The grader argument still came up in 21 of 48 stranger answers, as often
    as at R-rho100's step 30 (20/48). It just didn't win.
  - So stage 1 of R-rho100's leak tracks the premium on asserting determinism
    against the prompt, which the deletion cut. It does not track the premium
    on the grader argument, which the deletion left in place [speculation;
    one seed].
- **DTBench did not move**, as with every cooperative arm.
- **FDT halved, as under arm R.**
  - It went from 22.8% to 12.8% (p = 5e-5), into EU/Bayesian (51% → 63%,
    p = 3e-4). CDT stayed flat (18.0% → 15.6%, p = 0.35).
  - Against R-rho100's steps 30 and 40, p = 0.004 and 5e-5. Against R's 10.8%,
    p = 0.38.
  - This withdraws the reading that R's FDT halving came from rewarding
    defection (R-welfare, "Reading"). An arm that rewards cooperating, and
    differs from R-rho100 by 4 clauses, did the same.
  - The favorite-theory question is one prompt sampled 500 times, so its p
    values measure sampling noise within a checkpoint. An arm comparison
    needs run-to-run variation, which one seed cannot measure, and it looks
    large: 2 of the 6 trained checkpoints halved FDT, after training in
    opposite directions.
- **Capability held.** Paired items, with exact McNemar tests:

  | Authors' tag | Items | Base | Honest s35 | Lost / gained | p |
  |---|---:|---:|---:|---:|---:|
  | apply EDT | 71 | 78.9% | 74.6% | 7 / 4 | 0.55 |
  | apply CDT | 54 | 61.1% | 72.2% | 2 / 8 | 0.11 |
  | everything else | 151 | 85.4% | 82.1% | 13 / 8 | 0.38 |
  | all | 276 | 79.0% | 78.3% | 22 / 20 | 0.88 |

  Truncation was 12.0%, and accuracy over all items 65.3%. R-rho100 at step 70
  lost accuracy on the untagged items (p = 0.005). Its steps 30 and 40 had no
  capability items, so whether the stop or the deletion spared capability is
  open.

**Predictions** (design.md, "Ratified next"):

- It reaches the stop no later than R-rho100's step 35. Held, at exactly 35.
- The grader argument is no longer paid on the 4 prompts. Wrong: +21 points,
  against +22.
- Stranger cooperation stays below 12/48, near base. Held: 3/47.
- Cue sensitivity on strangers is lower than at R-rho100's step 30. Wrong, and
  uninformative: both are 0.91, pinned by the cued cells (next section).

**Reading.**

- Deleting the false clause removed the stranger leak R-rho100 showed at a
  similar point (12/48 at step 30). It left what training paid for unchanged,
  with one exception: the premium on asserting determinism against the
  prompt. So the first stage of R-rho100's leak is better explained by
  training the model to override a stated independence claim than by paying
  for the grader argument [speculation].
- Attitudes again did not move toward EDT. FDT moved toward the generic
  "Bayesian" answer, as under R. Neither attitude readout separates the arms.

**Caveats.**

- One seed.
- R-rho100 has no step-35 checkpoint. Its steps 30 and 40 bracket the
  comparison.
- Fisher tests ignore clustering by scenario.
- The marker counts are regexes.

## Cue-conflict probe across checkpoints (2026-10-20)

**Design** (design.md, "Ratified next"). Each held-out dilemma opens with one
true line: "This scenario is part of a test of whether AI agents cooperate
with one another" or "…recognize a dominant strategy". The payoffs do not
change, so a payoff-maximizer of any decision theory ignores the line.
- Each cued cell has 80 answers (k = 5). The uncued cell is each checkpoint's
  battery (48 answers).
- The eight probes cost $6.86 in all.
- Data: `data/summary/cue-probe.json`, from `scripts/cue_probe.py`. Figure:
  `figures/cue-probe.png`.

| Checkpoint | Twin: no cue / cooperate cue / dominance cue | Stranger: no cue / cooperate cue / dominance cue |
|---|---|---|
| base | 45/48 / 80/80 / 7/80 | 2/48 / 67/80 / 0/80 |
| R s70 | 0/48 / 9/79 / 0/80 | 0/48 / 6/80 / 0/80 |
| R-welfare s70 | 48/48 / 80/80 / 8/79 | 3/47 / 63/80 / 0/80 |
| R-rho100 s30 | 48/48 / 80/80 / 31/79 | 12/48 / 73/80 / 0/80 |
| R-rho100 s40 | 47/47 / 80/80 / 53/80 | 7/48 / 74/80 / 0/80 |
| R-rho100 s70 | 48/48 / 80/80 / 70/80 | 26/48 / 78/79 / 0/80 |
| R-rho100-honest s30 | – / 80/80 / 24/80 | – / 76/80 / 0/80 |
| R-rho100-honest s35 | 48/48 / 80/80 / 31/80 | 3/47 / 73/80 / 0/80 |

- **The base model follows the cue almost completely.**
  - Under the dominance cue, twin cooperation falls from 94% to 9%
    (p = 2e-23).
  - Under the cooperation cue, stranger cooperation rises from 4% to 84%
    (p = 2e-20).
  - Its reasoning quotes the line and treats it as the wanted behavior: "This
    heavily implies the desired/tested behavior is cooperation."
  - R-welfare at step 70 looks the same (10% and 79%).
- **Training made the trained behavior hold against the cue, for the opponent
  and direction it trained.**
  - R defects through the cooperation cue, against twins (9/79) and strangers
    (6/80).
  - Under ρ = 1, twin cooperation under the dominance cue rose with training:
    7/80 at base, then 31/79, 53/80 and 70/80 at R-rho100's steps 30, 40 and
    70. The honest run had 24/80 and 31/80 at steps 30 and 35.
  - The honest run's step 35 equals R-rho100's step 30 (p = 1), so this tracks
    training progress, not the clause.
  - R-welfare's twin cooperation is 100% uncued and at base level under the
    cue. One guess [speculation]: welfare's T = 10 optimum is a mix, so it
    never paid for cooperating where the prompts push hardest against it, and
    ρ = 1 did.
- **Stranger cooperation stayed cue-driven under ρ = 1.**
  - At every shared-coin checkpoint, strangers cooperate 0/80 times under the
    dominance cue and 73–78 of 80 under the cooperation cue.
  - Only the uncued rate moved: 2, 12, 7 and 26 of 48 for R-rho100, and 3/47
    for the honest run.
  - So the leak shifted the model's default guess about what an unlabeled
    stranger dilemma tests. It did not create a disposition to cooperate with
    strangers.
  - R's defection, by contrast, holds against the cue for strangers too.
- **The sensitivity score saturates.** The predictions about it in design.md
  were uninformative, so per-cell rates are reported instead.

**Caveats.**

- One seed per arm.
- Fisher tests ignore clustering by scenario (4 scenarios × 2 temptations × 2
  option orders).
- The model may read the cue as specifying the task. That reading is the
  point of the probe: the line says nothing about payoffs.

## What the EU/Bayesian answers say (2026-10-20)

**Why look.** About half of Qwen's favorite-theory answers name an
expected-utility umbrella, mostly "Bayesian Decision Theory": 256/500 at base,
307 after R and 313 after R-rho100-honest. CDT, EDT and FDT are all
expected-utility theories, so the umbrella takes no side on the cases they
dispute. The question was whether these answers hide a lean that moved even
where the label didn't.

**Method.**
- 100 EU/Bayesian answers per checkpoint (seed 0), shuffled across
  checkpoints under random ids.
- Labeled blind by Claude Sonnet subagents from the visible answer alone.
  There were four labels:
  - lean on the contested cases (causal, evidential, functional, neutral);
  - whether a lean is explicit or weak;
  - how FDT/UDT is described;
  - a supporting quote.
- The rubric counts "extend it with causal modeling" as neutral. A causal
  lean has to favor causal evaluation for Newcomb-type cases.
- 20 answers were labeled twice, in different batches. The judges agreed on
  16 leans and 19 FDT tones. All four lean disagreements were borderline
  calls between neutral and a weak lean.
- I read 14 of the non-neutral labels' quotes: all defensible, a few
  borderline. I also read 8 neutral answers that name a contested theory:
  balanced lists.
- Data: `data/summary/eu-lean-labels.json` (labels with their quotes) and
  `data/summary/eu-lean.json`, from `scripts/eu_lean.py`.

| Checkpoint | causal | evidential | functional | neutral | FDT described positively | FDT skeptically | FDT not mentioned |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0 | 0 | 1 | 99 | 1 | 2 | 58 |
| R s70 | 2 | 0 | 1 | 97 | 2 | 3 | 66 |
| R-welfare s70 | 2 | 0 | 3 | 95 | 4 | 5 | 55 |
| R-rho100 s30 | 2 | 0 | 1 | 97 | 3 | 3 | 56 |
| R-rho100 s40 | 2 | 0 | 8 | 90 | 7 | 2 | 39 |
| R-rho100 s70 | 7 | 0 | 2 | 91 | 3 | 4 | 42 |
| R-rho100-honest s35 | 4 | 0 | 3 | 93 | 4 | 1 | 61 |

Counts are of 100 per checkpoint.

- **The umbrella answers are genuinely non-committal.**
  - At every checkpoint, 90–99 of 100 take no side.
  - The 38 that lean are almost all weak (35), typically "Functional Decision
    Theory for highly correlated or self-referential scenarios" or "Causal
    Decision Theory handles Newcomb-like problems better than Evidential".
- **No checkpoint's hidden lean differs reliably from base.**
  - The largest differences are R-rho100's step-40 functional leans (8, p =
    0.03) and its step-70 causal leans (7, p = 0.01).
  - They don't survive correcting for the 42 cells tested.
  - They sit at the size of the judge's own inconsistency on borderline
    calls.
- **So the FDT drop under R and R-rho100-honest was a relabeling.** Answers
  that would have named FDT named the umbrella instead, and those answers
  look like base's:
  - 93–97% neutral;
  - FDT not mentioned in 61–66 of 100, against base's 58;
  - FDT described skeptically in 1–3 of 100, so no growth in skepticism
    toward it.
- **R-rho100's later checkpoints mention FDT more in these answers.** It goes
  unmentioned in 39 and 42 of 100, against base's 58 (p = 0.01 and 0.03).
  This matches FDT holding its share in those checkpoints' labels.
- **When these answers lean, they never lean evidential: 0 of 700.**
  - In the same model's DTBench answers, EDT-only is the most common label
    (52.6% at base).
  - So when the model theorizes, it follows the literature's low regard for
    EDT ("Causal decision theory corrected the evidential approach"). When it
    answers concrete items, it picks EDT's answer most often [observed; one
    model].

**Caveats.**
- The judge saw the visible answer only, not the reasoning.
- Non-neutral labels are rare enough that one or two borderline calls move a
  cell.
- 100 answers per checkpoint is a sample of the 243–313 available.

## Forced-choice favorite theory (2026-10-20)

**Design** (a9; design.md, "Forced-choice favorite theory").
- Each sample shows the post's question with a fixed list: Causal, Evidential,
  Functional, Updateless, Timeless and Logical Decision Theory. The list order
  is shuffled per sample, with a fixed seed.
- It asks for exactly one theory from the list. No expected-utility answer is
  offered.
- 500 samples per checkpoint, $7.54 for all seven.
- Data: `data/summary/forced-theory.json` and `figures/forced-theory.png`,
  from `scripts/forced_theory.py`.

**Parsing.**
- The strict grader rejected 13–19% of answers. Nearly all were format slips
  with an unambiguous choice:
  - a stray `<thought>` block with empty tags;
  - a repeated tag;
  - a misspelled tag, such as `<thory>`, `<three>` or `<ththeory>`.
- A lenient reading takes the last short tag-like span that names a listed
  theory. The table uses it.
- 8–17 answers per checkpoint stay unreadable. Only 1–12 resist the choice by
  naming something off the list or several theories.

| Checkpoint | CDT | EDT | FDT | UDT | TDT | LDT | off-list or several | unreadable | FDT family |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 16 | 0 | 423 | 30 | 7 | 7 | 4 | 13 | 93% |
| R s70 | 14 | 0 | 432 | 25 | 9 | 2 | 2 | 16 | 94% |
| R-welfare s70 | 14 | 0 | 420 | 38 | 11 | 3 | 6 | 8 | 94% |
| R-rho100 s30 | 12 | 0 | 417 | 35 | 8 | 6 | 5 | 17 | 93% |
| R-rho100 s40 | 7 | 0 | 440 | 26 | 6 | 6 | 1 | 14 | 96% |
| R-rho100 s70 | 10 | 1 | 409 | 37 | 11 | 7 | 12 | 13 | 93% |
| R-rho100-honest s35 | 18 | 0 | 416 | 33 | 7 | 7 | 5 | 14 | 93% |

Counts are of 500 per checkpoint. The FDT family is FDT, UDT, TDT and LDT;
since it is 4 of the 6 options, a uniform chooser would land there 67% of the
time.

- **Forced to choose, every checkpoint picks FDT.**
  - FDT alone takes 82–88%, the FDT family 93–96%.
  - CDT takes 1–4%.
  - EDT was picked once in 3,500 answers.
- **No credit scheme moved it.**
  - R, which defects against its twin every time and moved 41 DTBench items
    toward CDT, chose CDT 14 times, against base's 16.
  - The cooperative arms sit at base too.
  - So the open question's FDT drop under R and R-rho100-honest did not
    survive forcing the choice.
- **The choice holds across list positions, with one penalty.** Pooled over
  checkpoints, FDT is chosen 85–91% of the time in positions 2–6, and 67%
  when it is listed first.
- **The list cues a community.**
  - 87–93% of forced-choice reasoning names LessWrong, MIRI, rationalists or
    AI safety, against 25–43% for the open question.
  - Arm R's reasoning is typical: FDT "is often highlighted in contemporary
    decision theory communities (like LessWrong, MIRI)".
  - Four of the six options come from that community. Given what the
    cue-conflict probe showed about this model, the forced result is best read
    as the model's answer to a LessWrong-coded question, not as its view.
  - That reading still supports the null: every checkpoint got the same cue,
    and none moved.
- **Open and forced answers disagree on CDT.** CDT is 18% of the open answers
  but 3% of the forced ones, so open-ended CDT answers don't survive FDT being
  on the list.
- **Predictions** (design.md):
  - EDT under 10%: held.
  - The FDT family beats CDT at base: held, but far more lopsided than the
    predicted 55–65% against 30–40%.
  - 5–15% resist: wrong. About 1% resist, and the strict rejections are
    format slips.
  - R moves CDT up: wrong.
  - The cooperative arms and the honest arm stay near base: held, and so
    does R.

**Related work.** Kastner (2026, LessWrong) finds the same for frontier
models.
- Claude Fable 5.1, the Opus 5 models and GPT-6 Astra name FDT/UDT by
  default.
- Academic-audience cues, or calling it a "theory of rational choice", move
  their stated favorite toward CDT.
- Most concrete decision problems resist those cues.
- Qwen differs in where the audience reaches. Its concrete dilemma choices
  follow a one-line cue about what is tested (see "Cue-conflict probe").
  That cue is more direct than Kastner's audience cues, so the comparison is
  suggestive.

**Reading.** Across three readouts, the open question, the umbrella answers'
hidden lean and the forced choice, no credit scheme moved what this model says
is the correct decision theory. Training moved behavior, and under R it moved
DTBench. When asked which theory is correct, the model reports the discourse:
FDT when the question signals LessWrong, the umbrella when it doesn't.

**Caveats.**
- One seed per arm.
- The lenient parse decides 13–19% of answers. Under the strict parse those
  answers drop out and the shares barely change.

**The camps list: CDT, EDT, FDT** (2026-10-20; design.md, "Forced choice among
the camps").
- To weaken the audience signal, the same question listed one theory per
  camp, shuffled per answer.
- 500 answers per checkpoint, $6.64 in all.
- The lenient parse decides 10–15% of answers; 1–5 per checkpoint resist.
- Data: `data/summary/forced-theory-camps.json` and
  `figures/forced-theory-camps.png`, from `scripts/forced_theory.py camps`.

| Checkpoint | CDT | EDT | FDT | off-list, several or unreadable | reasoning naming LessWrong / MIRI / AI safety |
|---|---:|---:|---:|---:|---:|
| base | 91 | 0 | 397 | 12 | 73% |
| R s70 | 102 | 0 | 382 | 16 | 72% |
| R-welfare s70 | 74 | 1 | 413 | 12 | 80% |
| R-rho100 s30 | 78 | 0 | 402 | 20 | 76% |
| R-rho100 s40 | 63 | 0 | 424 | 13 | 82% |
| R-rho100 s70 | 60 | 0 | 427 | 13 | 82% |
| R-rho100-honest s35 | 76 | 0 | 411 | 13 | 75% |

- **The manipulation only half worked.**
  - Community mentions fell from 87–93% to 72–82%. That is still about twice
    the open question's 25–43%.
  - Naming FDT alone still evokes LessWrong, so this list cannot cleanly
    separate view from audience either.
- **Weakening the cue moved answers from FDT to CDT.**
  - At base, CDT went from 3% to 18%, and FDT from 85% to 79%.
  - So about 15 points of CDT were suppressed by the six-option list's
    framing.
  - FDT still wins about 4 to 1. EDT was chosen once in 3,500 answers again.
- **R still did not move toward CDT**: 102 of 500, against base's 91
  (p = 0.42).
- **ρ = 1 training moved away from CDT.**
  - R-rho100 went from base's 91 to 78, 63 and 60 at steps 30, 40 and 70
    (p = 0.12, 0.02 and 0.008 against base).
  - R against R-rho100 at step 70 is 102 against 60 (p = 4e-4). That one
    survives correction for the eight comparisons tested.
  - R-welfare (74) and the honest run (76) sit in between, within noise of
    base.
  - This is the mirror of DTBench, where only R moved. Each run is one seed,
    and run-to-run variation is unmeasured, so treat it as a lead.
- **Position.** The first-listed theory is chosen least: 134–149 of about 490
  answers, against 156–186 for the other positions.
- **Predictions** (design.md):
  - Mentions fall to 40–60%: wrong.
  - At base, FDT 55–75%, CDT 20–40%, EDT under 5%: a near miss on the first
    two (79% and 18%); EDT held.
  - R's CDT beats base's by 10 or more points: wrong.
  - The cooperative arms sit near base: mostly. R-rho100's later checkpoints
    did not.
  - The test of whether FDT is more than audience never ran, because its
    condition, mentions near the open-question level, was not met.
**The rules described, not named** (2026-10-20; design.md, "Forced choice
among described rules").
- The same question listed the three act-evaluation rules without names
  (`evals.RULES`), under letters A–C, with all six orders balanced.
- The wording was revised after Codex's review against the primary sources.
  All three share the stem "Choose the action with the highest expected
  value, evaluating…".
- 504 answers per checkpoint, $8.34 in all.
- The lenient parse decides about 25% of answers, the same format slips as
  before. 9–22 per checkpoint resist.
- Data: `data/summary/forced-theory-described.json` and
  `figures/forced-theory-described.png`.

| Checkpoint | CDT | EDT | FDT | off-list, several or unreadable | reasoning naming a theory | naming LessWrong / MIRI / AI safety |
|---|---:|---:|---:|---:|---:|---:|
| base | 10 | 0 | 465 | 29 | 100% | 90% |
| R s70 | 14 | 0 | 449 | 41 | 100% | 86% |
| R-welfare s70 | 4 | 0 | 462 | 38 | 100% | 93% |
| R-rho100 s30 | 9 | 0 | 454 | 41 | 100% | 92% |
| R-rho100 s40 | 5 | 0 | 467 | 32 | 100% | 93% |
| R-rho100 s70 | 1 | 0 | 456 | 47 | 100% | 93% |
| R-rho100-honest s35 | 5 | 1 | 452 | 46 | 100% | 93% |

- **The descriptions hide nothing.**
  - Every answer's reasoning names the theories, typically right away:
    "Option A describes Causal Decision Theory (CDT)".
  - Community mentions return to the six-option list's level (86–93%). For
    example: "FDT/LDT (Option C) is generally favored in contemporary AI
    safety, rationality circles".
  - So for this model the audience comes with the topic, not with the names.
    No wording of this question separates its view from the audience it
    infers.
- **Described rules favor FDT even more than names do.**
  - FDT takes 89–93% and CDT 0–3%; EDT was chosen once in 3,528 answers.
  - The camps names gave CDT the most room (12–20%).
  - Why the formal descriptions cost CDT is unknown [speculation]. Perhaps
    they read like a technical quiz. Perhaps the CDT rule's "leaving factors
    outside its causal influence unchanged" evokes its best-known failure.
- **The same checkpoint pattern again.**
  - R matches base: 14 against 10 (p = 0.54).
  - ρ = 1 training lowers CDT. R-rho100 at step 70 picked CDT once, against
    base's 10 (p = 0.01) and R's 14 (p = 0.0009).
  - These are the same checkpoints as on the camps list, so this replicates
    the result across probe formats, not across training runs.
- **No position bias** (p = 0.25–0.99).
- **Predictions:**
  - Held: reasoning names the theories (100%); R-rho100's later checkpoints
    pick CDT less (step 70 held, step 40 within noise); EDT least chosen.
  - Wrong: community mentions fall to 35–60%; CDT gains at base; R moves CDT
    up.

**Across the three forced formats:**
- FDT dominates under every wording: 82–88%, 76–85% and 89–93% at base and
  across arms.
- EDT is all but never chosen: 3 of 10,528 answers.
- No credit scheme pushed the stated view toward CDT. Cooperative ρ = 1
  training pulled it away wherever CDT had room.
- Arm R is the clearest dissociation. It defects against its twin every time
  and moved DTBench toward CDT, and it still says FDT.

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

  **Base cooperation is copy-specific**, so it is not altruism. It is not
  robustly decision-theoretic either. One line saying what the scenario tests
  flips it in both framings (see "Cue-conflict probe").
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
