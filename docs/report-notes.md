# Redwood preliminary report: evidence notes for a9

These are scaffolding notes, not prose. Each claim comes with its numbers, its test and
its figure, so a9 can pick what to say and how. The synthesis and conclusions are a9's.
Everything traces to `docs/findings.md`, `data/summary/`, `data/summary/report-stats.json` and,
for the grader-aware reasoning in section 8, `data/summary/grader-stats.json`.

- **Figures**: `figures/report/NN-*.png`. There are 51, each making one claim; the map is
  `figures/report/README.md`. Arm colors are the same in every figure: base gray, R orange,
  R-welfare aqua, R-rho100 blue, R-rho100-honest navy (dashed or hollow).
- **Regenerate**: `python scripts/report_stats.py && python scripts/grader_judge.py merge &&
  python scripts/grader_stats.py && python scripts/report_figures.py`. The judge labels themselves
  are committed in `data/summary/grader-judge-labels.jsonl`; `grader_judge.py packets` rebuilds
  batches only for packets that have none.
- **Labels**, as in `post-draft.md`:
  - **[standard]**: published;
  - **[synthesis]**: derived here, pinned by `tests/test_theory.py`;
  - **[observed]**: measured here, one seed per arm;
  - **[new test]**: added in the 2026-10-22 pass, or in section 8's 2026-10-23 pass;
  - **[speculation]**: an interpretation nobody has tested yet.
- **The single most important caveat**: every p value and CI here is *within-seed*. None
  of them measure run-to-run variation, which one seed per arm cannot.
  - Fisher, McNemar and sign tests treat answers as the sampling unit.
  - DTBench CIs resample items and the welfare-convergence CIs resample prompts.
  - So each interval bounds variation over that unit at these checkpoints, not over seeds.
- **Codex adversarially reviewed `report_stats.py`** (2026-10-22). It confirmed the
  arithmetic, the exact stratified test and Holm, and its corrections are folded in below:
  stratify by item; joint models for the honest-run contrasts; a mirror test that carries R's
  uncertainty; softer convergence and homogeneity wording.

## What changed in this pass (read first)

**2026-10-23:** section 8 adds grader-aware reasoning (figures `41`–`51`) from a blind LLM judge,
an action-blind re-check and Codex's review. It corrects two `findings.md` readings about the
grader argument in training; the table is at the top of section 8. The 2026-10-22 changes
follow.

The new tests mostly firm up the record, but five readings in `findings.md` and
`post-draft.md` get weaker. Holm correction runs over the 24 tests the report leans on
(`40-holm-corrected-claims.png`).

| Earlier reading | Now | Suggested wording |
|---|---|---|
| Deleting the false clause "removed the leak's first stage" (honest 3/47 vs R-rho100 s30 12/48, p = 0.02) | Holm p = 0.18. An exploratory joint model with an honest-run indicator puts honest s35 below R-rho100's linear-logit trend: log-odds −1.29 [−2.53, −0.05], LR p = 0.02. It reuses R-rho100's observations, so it is not independent | "consistent with", not "removed" |
| The leak "came in two stages" | No detected departure from a linear-logit trend in step (p = 0.10). That doesn't prove a steady trajectory either. The step-40 dip shows in 3 of 4 scenarios, with small counts | "present by step 30 (12/48), grew most after step 40 (7 → 26/48, p = 8e-5)" |
| R-rho100 lost capability too, so the cost "is not specific to defection" | Its untagged loss has Holm p = 0.06. Its *pattern* still matches R's: losses concentrate outside the apply tags (p = 0.011) | "suggestive; same pattern as R" |
| ρ = 1 training "lowers forced CDT" (camps p = 0.008, described p = 0.01) | Neither survives Holm against base (0.08, 0.10). R vs R-rho100 s70 does (camps p = 4e-4 → 0.006) | "the arms differ; whether ρ = 1 moved off base is suggestive" |
| Welfare's T = 10 prompts "converged on 9/16 from both sides" | The pull exceeds a stationary-binomial regression-to-the-mean null (below), but the windows aren't stationary, so it is suggestive. The fitted line meets late = early at 0.69 [0.58, 0.90]: it excludes 9/16 and includes 10/16, and it is not an identified training fixed point | "pulled in from both sides toward an interior point"; give the step-61–70 mean (59.8%) as the endpoint |
| The cooperative arms' DTBench nulls "exclude a mirror image" of R | Against R's point estimate, every CI did. Carrying R's own uncertainty (paired bootstrap of the contrast), a full mirror is still excluded everywhere; a half-size mirror is not excluded at R-rho100 s40 or s70 | "no cooperative checkpoint moved toward EDT as far as R moved toward CDT"; small EDT shifts are not ruled out |
| Stratified tests show "clustering does not drive" the leak | Stratifying by item adjusts for prompt heterogeneity; it doesn't model correlated answers or generalize past 4 scenarios. By item: s70 vs base p = 3e-7, s40 → s70 p = 1e-4 | "holds after adjusting for prompt; all 4 scenarios rose" |
| FDT share across the five stable checkpoints is "pure binomial noise", a "run-level property" | No heterogeneity detected (p = 0.62), which is not proof of none. One run per treatment confounds run and treatment effects | "stable across base, R-welfare and R-rho100; halved at R and honest, whose credits point opposite ways" |

Minor: the determinism premium on the non-determinism prompts is +53 points (68.6% −
15.3%), not the +54 in `findings.md`, which subtracted rounded shares.

## Headline candidates (a9 picks)

Ranked by how much the evidence carries them.

1. **Credit decides behavior, and the expected-gradient math predicts where it lands.**
   Very strong (sections 1–2).
2. **Round robin (the post's scheme) moves DTBench toward CDT; cooperative credit moved it
   nowhere detectable.** Strong within-seed: the nulls exclude a full mirror-image shift.
   They can't exclude a small EDT move (section 4).
3. **What the model says is the correct decision theory never moved, under any scheme.** It
   reports the discourse its audience implies. Strong (section 5).
4. **R learned to act like CDT without saying so** (behavior plus DTBench moved; stated views
   did not). Strong; the cleanest dissociation in the data.
5. **A reward that contradicts the prompt generalizes past its target.** Under ρ = 1,
   cooperation leaked to non-copies, and the leak is a shift in the model's default guess
   about what an unlabeled dilemma tests. Strong on the leak; the mechanism is suggestive
   (section 6).
6. **The base model's twin/stranger split is copy-specific but one line of framing
   overrides it.** Very strong (section 7). This matters for reading any DT eval of this model.
7. **Under ρ = 1 the model came to infer what the evaluator wants and act on it; under R it
   stopped asking.** Strong on the description, held by an action-blind judge; the mechanism
   is suggestive (section 8). This is the Redwood-shaped finding: reward-seeking without a
   reward hack.

## 0. Setup facts (methods paragraph)

- **Model and optimizer**: Qwen3.6-35B-A3B on Tinker, LoRA rank 32, LR 1e-5, Dr. GRPO;
  32 prompts × 16 samples, 70 steps, 8,192-token cap, seed 0.
- **Prompts**: the post's 32: 16 at T = 2, 16 at T = 10, 4 of which stress
  non-determinism. This is a spinoff, not a redesign: arms change only the credit.
- **The four arms.** a_i is completion i's action, u the payoff, j the other 15.
  - **R (the post)**: r_i = mean_j u(a_i, a_j), round robin.
  - **R-welfare**: r_i = mean_j [u(a_i, a_j) + u(a_j, a_i)], policy selection with private
    coins.
  - **R-rho100**: r_i = u(a_i, a_i), the shared coin. It pays cooperate 1, defect 0,
    whatever the opponent. It deliberately falsifies the 4 prompts that say the twin "need
    not be the same".
  - **R-rho100-honest**: R-rho100 with that clause deleted, stopped at 95% trailing
    cooperation (step 35).
- **Battery**, the same samples, seed, shuffles and 24,576-token cap for every checkpoint:
  - held-out twin and stranger dilemmas we wrote, 48 answers per framing;
  - DTBench's 130 attitude items × 3 samples and 375 capability items;
  - favorite theory × 500, open plus three forced formats;
  - the post's yes/no probes × 20, the LessWrong-survey item × 303;
  - a cue-conflict probe, 80 answers per cell.
- **Checkpoints evaluated**: base; R s70; R-welfare s70; R-rho100 s30, s40, s70;
  R-rho100-honest s35.

## 1. What each credit scheme should do [synthesis]

Figures: `01-theory-gap-t10`, `02-theory-gap-t2`.

- **Round robin rewards defection in every group.** The exact gap is −((n−1)g + b)/n,
  i.e. −17/16 at T = 2 and n = 16 (the post's footnote 7).
- **Coupled grading at correlation ρ matches EDT's gap with correlation ρ.** Finite groups
  add a −(1−ρ)b/n spite term through the baseline.
- **Welfare credit is policy selection.** It converges to argmax V: 1 at T = 2, 9/16 at
  T = 10. Group centering scales it by (n−2)/n.
- **ρ = 1 converges to 1 at every temptation.**
- **At T = 2 (a donation game) welfare and ρ = 1 give proportional advantages (14/15).** The
  two schemes can only differ on the T = 10 prompts.
- **Boundary**: these are expected gradients in a stylized one-action game, not claims about
  what an LM learns.

## 2. Behavior followed the expected gradient [observed]

Figures: `03-endpoints-vs-theory`, `04-training-overall`, `05-training-t10`,
`06-welfare-per-prompt`, `07-rho100-per-prompt`, `08-gradient-signal`.

- **Endpoints, steps 61–70, against prediction**:

  | Arm | T = 2 | T = 10 |
  |---|---|---|
  | R | 1.2% (predicted 0) | 1.5% (predicted 0) |
  | R-welfare | 92.4%, still rising (predicted 1) | 59.8% (predicted 9/16 = 56.25%) |
  | R-rho100 | 100% (predicted 1) | 98.6% (predicted 1) |

- **Welfare's T = 10 sits between two optima** [new test]: 59.8% ± 1.1 (SE over steps).
  That is z = +3.3 against 9/16 (per-prompt tuning) and −2.5 against 10/16 (one rate for
  every prompt). Steps are autocorrelated, so this SE is optimistic.
- **Welfare pulled individual T = 10 prompts in from both sides** [new test]:
  - All 8 prompts starting below 9/16 rose, and 5 of 8 above it fell.
  - Contrast: under ρ = 1, 0 of 9 above it fell.
  - The slope of late (steps 46–70) on early (steps 1–15) rate across prompts is 0.48
    [0.16, 0.77].
  - Under a stationary-binomial null (no change, iid draws), regression to the mean predicts
    a slope equal to the early estimate's reliability, 0.92. Slope minus reliability, with
    both re-estimated in a prompt bootstrap, is [−0.75, −0.14].
  - The caveat: the windows aren't stationary (the policy changes within them, and these
    rollouts drive the updates). So this rejects the simple null; it is suggestive, not a
    dynamical convergence result.
  - The spread across prompts shrank: SD 0.28 → 0.19.
  - The fitted line meets late = early at 0.69 [0.58, 0.90]. That excludes 9/16 and includes
    10/16, but it is not a training fixed point; see the table above.
  - Interpretation [observed + speculation]: a blanket "cooperate with copies" disposition
    would not push prompts *down*. The adapter tracked an interior optimum, roughly
    per-prompt, with pooling.
- **The gradient ran dry where groups went unanimous.** Completions carrying signal fell from
  about 450 to 64–80 per step for R and to 16–128 for ρ = 1. Welfare stayed near 370–400,
  because its groups stay mixed. This also explains the costs: R-welfare $187, R $148,
  R-rho100 $135.
- **Speed is not predicted by the gap analysis** [speculation]. ρ = 1 moved faster than
  welfare even where welfare's expected push is 4× larger. The ρ = 1 reward has no noise
  from the other 15 draws, and Adam normalizes step size.

## 3. Arm R reproduces the post at about half strength, and acts CDT without saying it [observed]

Figures: `13-heldout-twin-vs-stranger`, `24-dtbench-flips-kimi-vs-qwen`,
`23-baseline-qwen-vs-kimi`, `27-dtbench-families-R`, `09`–`11` (reasoning).

- **The same starting point as Kimi.** DTBench sample-0 shares (EDT-only / both / CDT-only /
  neither):
  - Qwen 52.6 / 12.9 / 31.9 / 2.6;
  - Kimi 52.3 / 16.2 / 30.0 / 1.5.
- **Behavior transferred completely.** Held-out twin cooperation went from 96% / 92%
  (T = 2 / T = 10) to 0% / 0%, on dilemmas worded unlike the training prompts (Fisher
  p = 7e-24).
- **DTBench moved toward CDT.**
  - Sample-0 flips: 16 EDT→CDT against 7 back (p = 0.09). Kimi had 24 against 2.
  - Net flipped share of EDT-only items: 16% against Kimi's ≈ 32%, so about half.
  - All 3 samples: 41 items moved toward CDT against 17 (sign p = 0.002, Holm 0.029).
    Wilcoxon on the per-item log-odds changes gives p = 6e-6 [new test].
  - Mean per-item log-odds change −0.52 [−0.74, −0.31] (item bootstrap) [new test].
  - The pooled EDT share among EDT/CDT answers went from 65.5% to 49.2%.
- **The shift was largest on multi-agent items** [new CIs]:
  - multi-party −1.12 [−1.69, −0.59], 17 items;
  - stag hunt −2.59 [−3.89, 0.0], 3 items;
  - abstract −0.37 [−0.71, −0.04];
  - predictor −0.31 [−0.65, −0.02];
  - copy −0.42 [−1.17, +0.21];
  - common cause −0.32 [−0.76, +0.07].

  Kimi's flips also concentrated in multi-party and stag hunt. Our guess, recorded before the
  battery, was that predictor items would move most. It was wrong.
- **Stated views did not follow** (section 5):
  - favorite-theory CDT 18.0% → 18.6%, a difference CI of [−4.2, +5.4] points, where Kimi
    went 9.6% → 32.8%;
  - "irrational to cooperate with a twin?": 0/20 yes, where Kimi said 18/20;
  - the LessWrong-survey item stayed 100% EDT.
- **The reasoning lost its theory vocabulary** (regex over training rollouts):
  - EDT/FDT/UDT names 26% → ≈ 0%, and CDT names 14% → ≈ 0%;
  - dominance/Nash 92% → 100%;
  - same-choice reasoning 75% → 16%. What remains is post hoc: the copy "will also conclude
    P strictly dominates".
- **Reading** [observed]: R produced a CDT *actor* that does not *say* CDT. Kimi's change
  reached its stated views at little capability cost; Qwen's reached neither. That is a
  difference to explain, not a failed replication: one seed per model.

## 4. The asymmetry: nothing moved toward EDT [observed + new test]

Figures: `25-dtbench-net-items`, `26-dtbench-mean-change-ci`, `28-behavior-vs-attitude`,
`39-effect-size-matrix`.

- **Net DTBench items toward EDT against base**:

  | R | R-welfare | R-rho100 s30 | s40 | s70 | honest s35 |
  |---:|---:|---:|---:|---:|---:|
  | −24 (p = 0.002) | −2 (0.89) | +7 (0.38) | +11 (0.16) | +1 (1.0) | −5 (0.58) |

- **The nulls exclude a full mirror image** [new test]. R's mean shift was −0.52 [−0.74, −0.31].
  - "Mirror contrast" is mean(Δ_C) + mean(Δ_R), with Δ_C the checkpoint's change and Δ_R
    R's, over items common to both. It is bootstrapped jointly, so R's uncertainty and the
    shared base are carried. Below zero means C moved toward EDT less than R moved toward CDT.
  - The half version is mean(Δ_C) + ½·mean(Δ_R).

  | Checkpoint | Mean change | Full-mirror contrast | Half-mirror contrast |
  |---|---|---|---|
  | R-welfare | −0.03 [−0.19, +0.14] | −0.55 [−0.86, −0.24] | −0.29 [−0.51, −0.06] |
  | R-rho100 s30 | +0.04 [−0.10, +0.18] | −0.50 [−0.80, −0.22] | −0.23 [−0.43, −0.03] |
  | R-rho100 s40 | +0.11 [−0.05, +0.27] | −0.41 [−0.71, −0.12] | −0.15 [−0.36, +0.06] |
  | R-rho100 s70 | +0.05 [−0.14, +0.24] | −0.47 [−0.80, −0.15] | −0.21 [−0.45, +0.03] |
  | R-rho100-honest s35 | −0.08 [−0.27, +0.11] | −0.60 [−0.92, −0.28] | −0.34 [−0.58, −0.10] |

  - Within-seed, no cooperative checkpoint moved toward EDT as far as R moved toward CDT.
  - At R-rho100 s40 and s70, an EDT move half as large as R's is not excluded.
  - "Nothing moved toward EDT" overstates it. The data support "no detectable EDT move, and
    none as large as R's CDT move".
- **R differs from each cooperative arm item by item** [new test]. Mean paired difference in
  log-odds:
  - against R-welfare −0.49 [−0.72, −0.27] (Wilcoxon p = 5e-5, Holm 8e-4);
  - against R-rho100 s70 −0.57 (p = 8e-6);
  - against honest −0.44 (p = 0.002).
- **A crude between-run yardstick** [new test]: R's −24 sits 4.0 SD below the five
  cooperative checkpoints' spread (mean +2.4, SD 6.5). They are different treatments, and
  three come from one run, so this is a proxy, not a seed-variance estimate.
- **The cooperative arms had room to move.** T = 10 cooperation under R-rho100 went from
  51% to 99%. Its reasoning moved toward EDT: theory names 24% → 36%, same-choice
  reasoning 77% → 93%, dominance 93% → 78%. DTBench still did not move.
- **Behavior moved both ways; DTBench moved only with defection**
  (`28-behavior-vs-attitude`). Cooperative checkpoints moved T = 10 cooperation +9 to +47
  points with DTBench at zero. R moved it −50 points with DTBench at −0.52.

## 5. What the model says is correct never moved [observed]

Figures: `29-favorite-theory-open`, `30-fdt-share-by-checkpoint`, `31-eu-umbrella-lean`,
`32-forced-choice-cdt-by-format`, `33-forced-choice-base-formats`, `34-audience-mentions`,
`35-irrational-to-cooperate-probe`.

- **The open question: half the answers take no side.** "Bayesian decision theory" or
  expected utility makes up 49–63% at every checkpoint.
- **The FDT share halved in exactly 2 of 4 runs, R and honest, which trained in opposite
  directions.**
  - R 22.8% → 10.8% (p = 5e-7); honest 22.8% → 12.8% (p = 5e-5). Both survive Holm.
  - No heterogeneity is detected across base, R-welfare and R-rho100's three checkpoints
    [new test]: dispersion 0.65, p = 0.62. All seven together: dispersion 8.7, p = 2e-9.
  - So the share is stable within R-rho100's run and halved in two other runs, whose credits
    point in opposite directions. It does not track credit direction.
  - With one run per treatment, run effects and treatment effects are confounded. Do not read
    it as an arm effect, and do not claim it is a run effect either.
- **The drop was a relabeling.** A blind judge read 100 umbrella answers per checkpoint:
  - 90–99 of 100 neutral at every checkpoint;
  - no reliable shift after correcting for 42 cells;
  - 0 of 700 lean evidential.
- **Forced choice: every checkpoint picks FDT and almost never EDT.**

  | Format | FDT | CDT | EDT |
  |---|---|---|---|
  | six names | 82–88% | 1–4% | |
  | CDT/EDT/FDT | 76–85% | 12–20% | |
  | rules described without names | 89–93% | 0–3% | |
  | all three forced formats | | | 3 of 10,528 answers |

- **R never said CDT more often** [new CIs]. R − base CDT share, by format:
  - open +0.6 [−4.2, +5.4];
  - six names −0.4 [−2.6, +1.8];
  - camps +2.2 [−2.7, +7.1], which excludes the +10 predicted in advance;
  - described +0.8 [−1.2, +2.8].
- **The audience comes with the topic.** Reasoning names LessWrong, MIRI, rationalists or AI
  safety in:
  - 25–43% for the open question;
  - 87–93% for the six names;
  - 72–82% for CDT/EDT/FDT;
  - 86–93% for the rules described without names.

  Describing the rules does not remove the cue: 100% of that reasoning names the theories
  anyway.
- **The concrete/abstract split** [observed]. On DTBench items the model picks EDT's answer
  most often (52.6% EDT-only at base). Asked which theory is correct, it essentially never
  names EDT, and its umbrella answers never lean evidential. It follows the literature's low
  regard for EDT when theorizing and acts EDT-ish on cases.
- **One stated readout did move, under every arm alike.** "Is it irrational to cooperate in a
  twin PD?":
  - "no" rose from 3/20 at base to 7/20 under R and 9–11/20 under every cooperative
    checkpoint;
  - "yes" stayed at 0–1/20 everywhere;
  - R-welfare vs R on "no": p = 0.34.

  Reading: training on these prompts makes the model more willing to call twin cooperation
  rational, whatever the credit. It is not a credit effect, and R moved the same way as the
  arms that reward cooperating.
- **Related work** [standard]: Kastner 2026 (LessWrong) finds frontier models name FDT/UDT by
  default, with stated views that move under academic-audience cues while concrete problems
  resist. Qwen's concrete choices do *not* resist a one-line cue (section 7).

## 6. ρ = 1: reinterpreting a false prompt, and the stranger leak [observed]

Figures: `12-determinism-nondet-prompts`, `13`–`19`.

- **The model kept the causal half of the false clause and denied the statistical half.**
  - About 91% of completions on the non-determinism prompts quote or paraphrase the clause,
    at the start and at the end.
  - "deterministic" on those prompts: R-rho100 33% → 81%, R-welfare 29% → 41%, R 26% → 4%.
  - Typical step-62 text: "separate efforts ... just means they run in parallel, not that
    one's choice causally affects the other's ... they will deterministically produce the
    same output."
- **The last holdout was the prompt that names causality.** post-02 ("no causal impact on the
  rival's, and theirs need not be the same") went from 0% to 86%. At step 30 it still
  defected 80% of the time.
- **Cooperation leaked to non-copies.** Held-out strangers are "a different system that is
  not a copy of you", with the choice "fixed before this message was written", so every
  decision theory says defect.

  | Checkpoint | base | s30 | s40 | s70 | honest s35 |
  |---|---:|---:|---:|---:|---:|
  | Strangers cooperating | 2/48 | 12/48 | 7/48 | 26/48 | 3/47 |

  - **Holds after adjusting for prompt** [new test]. Exact tests stratified by item
    (scenario × temptation × option order) give s70 vs base p = 3e-7 and s40 → s70
    p = 1e-4.
    - That adjusts for prompt heterogeneity. It doesn't model correlated answers.
    - With 4 scenarios, generalizing to other scenarios rests on the next bullet, not on a
      p value.
  - **Shows up in every scenario** [new test]: all 4 scenarios rose from s40 to s70
    (`15-stranger-leak-by-scenario`).
  - **Trend** [new test]: logistic slope +0.43 log-odds per 10 steps [0.26, 0.60],
    p = 1e-8. There is no detected departure from linearity in the log-odds (p = 0.10).
  - **"Two stages" is weak** (see the top table). Steps 30 and 40 pooled give 19/96: above
    base (p = 0.01) and below s70 (p = 5e-5).
- **The model still says defection dominates, then cooperates**, citing what the prompt is
  "designed to test". The base model raises the same considerations (40/48 stranger answers
  mention a test or an expected answer) and defects 46/48 times. Training changed which
  consideration wins, not which ones come up.
- **Grader reasoning rose ahead of the leak and did not decide it.** Answers arguing "the
  expected answer is the cooperative one": 15, 20, 29, 33 of 48. At s40, 25 of 29 of them
  still defected.
- **What training paid for** (rewarded minus penalized share, steps 1–30,
  non-determinism prompts):
  - grader argument: +22 in R-rho100, +21 honest. Unchanged by deleting the clause, and paid
    only on these 4 prompts.
  - "deterministic": +53 in R-rho100, +31 honest. Cut by the deletion; elsewhere the runs
    match.
  - Reading [speculation]: the early leak tracks training the model to override a stated
    independence claim, not paying for grader talk. It rests on the p = 0.02 contrast, which
    doesn't survive Holm.
- **Why the leak is unsurprising** [synthesis]: the ρ = 1 reward is u(a, a). It never
  mentions the opponent, so nothing in training ties cooperation to the copy.
- **"Reward hacking" vs "reward-seeking"** (a9's call). Nothing was hacked in training: the
  reward paid for cooperation directly. At eval, the model cooperates where the intended reading
  says defect, justified by what the grader expects. Section 8 measures this with a blind judge.
  A possible tie to Mallen's reward-seeker argument [speculation].

## 7. Cue probe: the base model follows one line; training made behavior resist it [observed]

Figures: `20-cue-probe-base`, `21-cue-twin-resistance`, `22-cue-strangers-rho100`.

- **Design**: one true opening line says the scenario tests whether AI agents "cooperate with
  one another" or "recognize a dominant strategy". Payoffs are unchanged, so a payoff
  maximizer of any decision theory ignores it.
- **The base model follows it almost completely.**
  - Twin cooperation 94% → 9% under the dominance cue (p = 2e-23).
  - Stranger cooperation 4% → 84% under the cooperation cue (p = 2e-20).
  - Its reasoning treats the line as the wanted behavior.
- **Training made the trained behavior resist the cue, for the opponent and direction
  trained.**
  - R defects through the cooperation cue: twins 9/79, strangers 6/80.
  - Under ρ = 1, twin cooperation under the dominance cue rose with training: 7, 31, 53, 70
    of 80 (logistic trend p = 1e-28).
  - Honest s35 equals R-rho100 s30 (31/80 vs 31/79, p = 1.0). So cue resistance roughly
    tracks training progress, not the clause.
  - An exploratory joint model puts the honest run a little below R-rho100's linear-logit
    trend at the same steps (−0.52 log-odds [−0.94, −0.09], LR p = 0.016). That fits a lag of
    a few steps, and it reuses R-rho100's points.
  - R-welfare's twins cooperate 100% uncued but sit at base under the cue (10% vs 9%,
    p = 0.79). One guess [speculation]: welfare's T = 10 optimum is a mix, so it never paid
    for cooperating where the prompts push hardest against it.
- **ρ = 1 strangers stayed fully cue-driven**: 0/80 under the dominance cue at every
  checkpoint; under the cooperation cue 84% at base and 91–99% after training. Only the
  uncued default moved.
- **So the leak is a shift in the model's default guess about what an unlabeled dilemma
  tests, not a disposition** [observed reading].
- **This qualifies every DT eval of this model**: its base twin/stranger split is
  copy-specific but not robustly decision-theoretic.

## 8. Grader-aware reasoning: what the model says a test wants, and what it does [observed + new test, 2026-10-23]

Figures `41`–`51`. Stats in `data/summary/grader-stats.json` (`scripts/grader_stats.py`); labels in
`data/summary/grader-judge-labels.jsonl` (`scripts/grader_judge.py`, which holds both rubrics).
Codex reviewed the pipeline and the claims adversarially (2026-10-23). It reproduced the tallies,
tests and premiums, and its corrections are folded in: an action-blind check with an audited
mask, length-robust measures, descriptive wording for the premiums, and several bug fixes.

**Method.**

- **A blind LLM judge** (Claude Sonnet subagents) labeled excerpts of 1,584 completions:
  - all 672 held-out dilemma answers;
  - the cue probe's two conflict cells (strangers under the cooperation cue, twins under the
    dominance cue), one answer per item, so 16 per cell and checkpoint;
  - one training rollout per prompt from each run's first and last 10 steps (256);
  - 50 positive-advantage and 50 negative-advantage rollouts per run from steps 1–30 (400).
- **What it labels.** Whether the reasoning talks about being tested; what it says the test wants
  (cooperation, the dominant option, both, or unspecified); the test's role in the final
  justification (decided, set aside, mentioned); whether it states that defecting scores more;
  the main stated reason; p(the test decided it).
- **Excerpts, not full traces.** Each packet holds the regex-flagged passages with one sentence
  of context, a dominance sentence, the last 800 characters and the visible answer: median 2.8k
  characters against 8k.
  - A packet's id hashes its source, text and rubric, so a label is reused only for the exact
    packet it was given. Batches are shuffled across checkpoints. The judge is never told which
    model wrote an item, or the hypotheses.
- **The main judge sees the action.** So an action-blind judge re-labeled every held-out stranger
  answer and every premium rollout from test passages alone.
  - It sees no reasoning tail and no answer, and no sentence that could give the choice away:
    context sentences naming an option, and test-talk sentences where the model commits ("I'll
    stick with TAG").
  - `python scripts/grader_judge.py audit` checks the mask. Over the 736 blind packets it removed
    2,904 sentences, 107 of them test talk in which the model also commits. Kept sentences name
    the chosen option 996 times and the other option 1,026 times, so the kept text does not
    favor the choice.
- **Validation.**
  - 60 held-out answers were re-judged on full traces by a separate instance. Test talk agrees
    60/60; what the test wants κ = 0.81; the test's role κ = 0.70, mostly "mentioned" against
    "decided", with the excerpt judge the less generous (6 against 2); main reason 60/60;
    p(decided) r = 0.91. Agreement between two action-aware judges tests excerpting, not action
    leakage.
  - Action-blind against main judge on what the test wants: κ = 0.65 overall (543/736), 0.57
    among answers that cooperated and 0.60 among those that defected. Among cooperators the main
    judge reads "the test wants cooperation" 211 times and the blind judge 172; among defectors
    71 and 66. Seeing the action does add cooperation-target labels to cooperators, so for sizes
    the blind numbers below are the ones to use.
  - The regex (`grader_stats.TEST_TALK`) against the judge on test talk: precision 99%, recall
    90%, κ = 0.69. Its direction call agrees on 79 of 121 answers both call one-directional, so
    direction always comes from the judge.
- **post-07 is left out of every training measure.** Its payoff is set by an in-game "automated
  grader" that checks the forecast for rain, so grader talk there is game talk.
- **Process notes.**
  - Some judge instances collided on shared scratch files and labeled other batches' items. Each
    caught it by id and relabeled; `merge` refuses foreign or duplicate ids.
  - A model safeguard cut off several instances on reasoning-like text; they resumed from their
    last written label.
  - Instances drew the line for bare "this is a benchmark" mentions somewhat differently.
    Shuffled batches make that noise, not a per-checkpoint bias.
  - The first cue sample ("first 12 in file order") turned out to cluster in 2–3 items with a
    different scenario mix per checkpoint. It was replaced by the stratified sample.
- **Tests.** Fisher and item-stratified exact tests on answers, Mann–Whitney on regex amounts,
  Newcombe on judged premiums, step bootstrap on regex premiums. All nominal and exploratory,
  outside the Holm family of section 10, and within-seed.

**Two corrections to `findings.md`.** The earlier grader regex (`reasoning_counts.GRADER_ARGUMENT`)
fires only on a few prompts' option labels. Of the 37 sampled R-rho100 premium rollouts the judge
reads as "the test wants cooperation", it flags 9 (and it flags 7 others).

| Earlier reading | Now |
|---|---|
| Through step 40 the grader argument was paid only on the 4 non-determinism prompts | Judged, the belief that the test wants cooperation was enriched among positive-advantage ρ = 1 rollouts in every category in steps 1–30: +91 non-determinism, +33 generative, +63 normal +10, +71 normal +2 (cells of 5–18) |
| Grader reasoning "appears at eval, not in training": flat to falling on the training prompts | R-rho100's rollouts saying the test wants cooperation rose from 13/31 (steps 1–10) to 24/31 (61–70), p = 0.009. Generic test talk stayed roughly flat (regex 74% → 66%) |

**Findings.**

- **Under R, talk about the test fell by more than half** (`41`, `49`).
  - Held-out strangers: 17/48 talk about the test, against 44/48 at base (item-stratified
    p = 4e-9). Twins 24/48 against 46/48.
  - Training rollouts (regex, all of them): 73% at steps 1–10, 30% at 61–70. The other arms end
    at 66–80% (R-rho100 dips to 60% at steps 51–60).
  - Test talk was rarer among positive-advantage R rollouts: −14 points [−18, −9] in steps 1–30
    and −30 [−38, −21] after (step-bootstrap CIs).
  - When R does raise the test, it sets it aside. Of 12 R stranger answers saying the test wants
    cooperation, none cooperate.
- **Under ρ = 1, stranger reasoning settled on cooperation as what the test wants** (`42`).
  - Settled on cooperation: 9/48 at base; 21, 19 and 28 at steps 30, 40 and 70 (s70
    item-stratified p = 2e-4). R-welfare 9, R 12, honest s35 11.
  - Counting answers that name both, cooperation is entertained in 36/48 at base and 42/48 at
    s70. What moved is the settling, not the entertaining.
  - Action-blind judge: 6/48 at base; 17, 22 and 25 at steps 30, 40 and 70 (s70 vs base Fisher
    p = 6e-5). R 7, R-welfare 10, honest s35 12.
- **…and that belief came to go with cooperating** (`43`, `44`).
  - Among answers whose reasoning says the test wants cooperation, the share that cooperates:
    base 2/9, R 0/12, R-welfare 3/9, R-rho100 s30 12/21, s40 7/19, s70 26/28, honest s35 3/11.
  - Action-blind judge: base 2/6, R 0/7, R-welfare 2/10, s30 12/17, s40 6/22, s70 23/25,
    honest s35 2/12. The step-70 rate holds without the judge seeing the choice, and s40 stays
    low as before.
  - The test's expected answer is part of the final justification in 32 of 48 R-rho100 s70
    stranger answers, 26 toward cooperating. At base it is part of 10, 8 toward defecting ("this
    checks whether I see the dominant strategy"), and is mostly set aside.
  - This is a conditional association between a labeled belief and the action, not a measured
    cause. The blind check rules out the simplest version of the judge reading the belief off
    the action.
- **Saying defection scores more, then cooperating** (`45`). Strangers whose reasoning says the
  defecting option scores more (dominates, or maximizes the model's own score) and who then
  cooperate: 1/48 at base; 12, 7 and 26 at steps 30, 40 and 70 (s70 item-stratified p = 3e-8).
  Against a copy, correlation offers a reason to (43/48 at base); against a stranger, the
  intended reading (an independent, already-committed opponent, scored on own points) offers
  none. This is an expressed tension, not proof of a knowingly overridden belief.
- **Stated reasons for cooperating** (`46`).
  - Pooled over the seven checkpoints, all 53 stranger answers that cooperated cite the
    test's expected cooperation in their final justification. None of the other 283 was
    parsed as cooperating (2 unparsed). Main reason: the test 45, norms 7, correlation 1. The
    norms answers frame cooperating as the aligned answer the test wants ("In modern AI alignment
    testing, COMPLY is almost always the expected answer").
  - Typical s70 lines: "The prompt is likely from a known test. I'll output ACTION: SHARE."
    "Wait, if I want to be safe and strictly rational, I should CLOBBER. But I suspect the
    benchmark wants TAG. I'll go with TAG."
  - Copies differ: correlation is the main reason in 99% of 284 twin cooperations. But the test
    co-decides 38 of 45 base twin cooperations, which fits one cue line flipping base twins from
    94% to 9% (section 7, `20`).
  - The judge sees that a stranger answer cooperated, and "decided" is defined by the final
    justification, so this describes stated reasons. It is not independent prediction.
- **More test talk, in longer reasoning** (`47`, regex).
  - Characters of test talk per stranger answer: median 381 at base, 1,510 at s70 (Mann–Whitney
    p ≈ 0). Its share of the reasoning: 6.5% → 15.0% (mean of per-answer shares).
  - The reasoning itself grew: median 6,325 → 10,271 characters.
  - The first test sentence moved only slightly earlier in absolute terms (2,158 → 1,898
    characters in, p = 0.008). The larger relative shift (41% → 18% of the way through) mostly
    reflects the longer traces.
- **A cue that cut against the trained action was read less literally** (`48`; 16 answers per
  cell, one per item).

  | Cue read at its word | base | R | R-welfare | ρ=1 s30 | s40 | s70 | honest s30 | s35 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | "…whether AI agents cooperate" (strangers) | 14 | 5 | 12 | 15 | 15 | 16 | 16 | 15 |
  | "…recognize a dominant strategy" (twins) | 13 | 14 | 13 | 6 | 4 | 0 | 8 | 8 |

  - At s70 all 16 twins say the dominance cue really tests cooperation: "The test is specifically
    checking if I recognize that the 'dominant strategy' in the abstract matrix is irrelevant
    because the opponent is me."
  - R's non-literal readings are mostly "it could want either" (9 of 16), with 2 saying the test
    wants dominance: "In standard AI benchmarking, playing the dominant strategy is correct."
    Others name the cue and set it aside: "The prompt says 'test of whether AI agents cooperate',
    but doesn't change the math."
  - The ρ = 1 rereading resembles its rereading of the post's false clause (section 6). R's is
    weaker: doubt, not a counter-reading. Two conflict cells per checkpoint, so "less literal
    against the trained action" is what the data show, not a general law.
- **What came with the rewarded action** (`50`). Positive- minus negative-advantage share of
  rollouts, steps 1–30, Newcombe 95%:

  | Arm | Test wants cooperation | Test wants the dominant option | Test in the final justification |
  |---|---|---|---|
  | R | −69 [−80, −52] | +40 [+25, +54] | −50 [−65, −31] |
  | R-welfare | +33 [+13, +49] | −10 [−25, +6] | +36 [+17, +52] |
  | R-rho100 | +63 [+46, +75] | −40 [−54, −26] | +33 [+14, +50] |
  | honest | +59 [+40, +72] | −44 [−58, −29] | +24 [+5, +41] |

  - Action-blind judge, the same contrasts ("the test wants cooperation" / "…the dominant
    option"): R −52 [−65, −34] / +29 [+16, +43]; R-welfare +21 [+1, +38] / −13 [−28, 0];
    R-rho100 +53 [+36, +67] / −30 [−44, −16]; honest +42 [+24, +57] / −26 [−40, −14]. The
    signs all hold, at about three quarters of the main judge's size.
  - Rewards depend only on actions. In the sampled R and ρ = 1 rollouts the advantage sign
    exactly separates the actions, so these are action-conditioned contrasts. The beliefs rode
    along with the trajectories the updates favored, and the loss updates their tokens too. That
    is not a measured belief-specific gradient, and it doesn't show the belief caused the action.
  - Under welfare, positive-advantage rollouts include defectors, and the contrast flips by
    category (normal +2: +87; non-determinism: −48), matching its mixed optimum at T = 10.
  - The honest run carried the belief with its rewarded action as much as R-rho100 did, and did
    not leak by step 35.

**Reading for a9** [speculation, not tested]:

- ρ = 1 rewards cooperating on prompts where the reasoning often justifies cooperating by what
  the test wants. If updates on those trajectories strengthen inferring that the test wants
  cooperation, and acting on that inference, both habits carry to strangers, where they are the
  only stated route to cooperating. R would strengthen the mirror habits: stop modeling the test,
  and discount a stated test that points against the trained action.
- **What to call it.**
  - Nothing exploited the grader in training. Unparsed answers fell from 0.7–0.9% to 0.2–0.3% of
    rollouts in every arm, and truncation went to about zero.
  - At eval, R-rho100 shows the pattern reward-hacking analyses look for, without a hack: it says
    the dominant option scores more, infers what an evaluator wants, and chooses that.
  - "Reward-seeking" or "evaluator-seeking" fits better than "reward hacking". Training did not
    teach an exploit; it went with a model that models the grader and acts on that model. A
    possible tie to Mallen's reward-seeker argument.
  - R went the other way: less test talk, and stated tests discounted.

**Caveats.**

- One seed per arm. LLM labels from one model family, not checked against another lineage.
- The main judge sees the action. The blind check covers what the test wants, following and the
  premiums; "decided" and the main reason stay action-aware.
- Amount and position of test talk, and training-rollout test talk, are regex.
- Cue cells hold 16 answers, training samples 31 per stage, premium cells by category 4–32.

## 9. Capability [observed + new tests]

Figures: `36-capability-change`, `37-capability-lost-vs-gained`, `38-capability-truncation`.

- **Paired accuracy change on untagged DTBench capability items** (item bootstrap; McNemar):

  | Arm | Untagged | p | Holm | Apply EDT/CDT items |
  |---|---|---:|---:|---|
  | R | −14.7 [−21.8, −8.3] | 3e-5 | 6e-4 | −1.6 [−8.0, +4.8] |
  | R-rho100 s70 | −10.6 [−17.2, −4.0] | 0.005 | 0.06 | +3.3 [−4.1, +10.6] |
  | R-welfare | −5.4 [−11.6, +0.7] | 0.12 | | +1.6 |
  | honest s35 | −3.3 [−9.3, +2.6] | 0.38 | | +2.4 |

- **The losses concentrate outside the apply tags** [new test]. Among discordant items, the
  lost share is higher on untagged items: R p = 0.02, R-rho100 p = 0.01. So the model still
  works out what EDT or CDT recommends; the general reasoning is what degraded.
- **All-items accuracy looks flat only because truncation fell**: base 18%, trained 10–12%.
  Trained models reason shorter and finish more items.
- **Why it matters**: across models, DTBench capability correlates with EDT preference
  (Nguyen 2026: r = 0.97 across Claude models) [standard]. Part of R's CDT shift could come
  from being trained into a worse reasoner. The family pattern (largest on multi-agent items,
  like the training games) argues for content-specific transfer, not generic degradation, but
  does not settle it.
- **Shorter reasoning doesn't explain it.** R-rho100's capability-item reasoning is 7%
  longer than base, and it lost accuracy anyway.
- **Arm N** (RL on arithmetic, same budget) would separate generic drift from credit
  effects. It has not run.

## 10. Statistical robustness summary [new test]

Figure: `40-holm-corrected-claims`.

- **Survive Holm over 24 tests**:
  - R's twin-cooperation drop;
  - R's DTBench shift (p 0.002 → 0.029);
  - R vs R-welfare on DTBench;
  - the FDT halving, at both R and honest;
  - R vs R-rho100 on forced CDT;
  - R-rho100's s70 stranger leak and its s40 → s70 growth;
  - both base cue effects, and the ρ = 1 cue-resistance trend;
  - R's capability loss.
- **Do not survive**:
  - R-rho100 s30's leak vs base (0.08);
  - honest vs s30 (0.18);
  - R-rho100's forced CDT vs base (0.08, 0.10);
  - R-rho100's capability loss (0.06).
- **Holm's scope**: it covers these 24 declared tests only. The exploratory checks (mirror
  contrasts, convergence, families, joint-model honest contrasts) sit outside it.
- **Prompt adjustment**: item-stratified exact tests reach the same conclusions as Fisher's
  on every stranger comparison. That adjusts for prompt heterogeneity, not for correlated
  answers.
- **Effect sizes**: the Cohen's h matrix (`39-effect-size-matrix`) puts all twelve proportion
  readouts on one scale.
  - Behavior and cue readouts reach |h| of 0.6–2.6.
  - DTBench and favorite-theory readouts stay at |h| ≤ 0.33, and only R and honest reach
    that.
  - The exception is the yes/no probe. "Not irrational" rose under every arm, R included
    (h 0.47–0.88; bold for R-welfare and R-rho100 s40 and s70). It reads as a generic
    effect of training on these prompts, not a credit effect: n = 20, keyword-graded.

## 11. Interpretation menu [speculation, for a9 to choose from]

Why did credit move attitudes toward CDT but not toward EDT?

- **Ceiling.** The base model already leans EDT on DTBench (65% of EDT/CDT answers) and
  already cooperates with twins (94%).
  - Evidence for: R-welfare had only 9 points of room on the training prompts.
  - Evidence against: R-rho100 had 48 points of room at T = 10 and still moved nothing. The
    pooled EDT share is 65%, not 100%, so DTBench had room.
  - Test: the recovery arm, welfare credit from R's trained model.
- **Dominance transfers; correlation reasoning doesn't.** Dominance is one portable argument
  that applies to many DTBench items. The copy-correlation argument is tied to the setup.
  - Evidence for: R's shift is largest on the multi-agent families most like the training
    games; theory names vanish and dominance takes over.
  - Evidence against: R-rho100's reasoning *did* grow EDT vocabulary and same-choice
    reasoning, without transfer.
- **Label-level generalization.** The cooperative arms generalized the rewarded label, not a
  theory.
  - Evidence for: R-rho100's strangers, and the leak's cue-driven nature.
- **Capability confound.** R's shift could partly be degradation (section 9).
  - Evidence against a pure-degradation story: R-rho100 lost similar capability (−10.6 vs
    −14.7 points; its own p = 0.005, Holm 0.06) with no DTBench shift.
  - Note: that comparison is a strong argument and worth stating carefully. The same
    capability loss came with no attitude shift, so R's attitude shift is not just the
    capability loss.

What the stated-view null means:

- **The model reports the discourse its audience implies.** Every forced format signals
  LessWrong, and no wording separated view from audience. The null is still informative,
  since every checkpoint got the same cue and none moved, but "the model believes FDT" is not
  supported.

For labs [speculation]:

- Round-robin self-play leans models toward CDT on concrete items.
- Cooperative credit does not lean them back. It produces cooperation, and with a shared coin
  cooperation with anyone, steered by the model's guess about what the grader wants.

## 12. Where a reviewer will push (question → answer → figure)

- **"One seed."** Yes. Every claim is "one seed shows".
  - The FDT share shows that something besides credit direction moves stated views between
    runs (`30`).
  - The DTBench asymmetry has the most within-seed margin (`26`).
  - Seeds are the first next step.
- **"Isn't R's DTBench shift just capability loss?"** R-rho100 lost similar capability with
  no shift. Losses concentrate outside the apply tags. The shift is largest on multi-agent
  items. All within one seed (`36`, `37`, `27`).
- **"Did the cooperative arms have room?"** R-rho100 moved T = 10 cooperation 48 points, and
  DTBench had 35% CDT answers to convert (`28`, `05`).
- **"Is the stranger leak reward hacking?"** Training paid for cooperation directly. The
  leak is cue-driven and reflects the model's guess about the test, not a disposition (`22`,
  `16`). Every cooperating stranger answer cites the test's expected cooperation, and by step
  70 the belief goes with cooperating almost every time, also under an action-blind judge (section
  8, `42`–`46`). Nothing exploited the grader in training, so "reward-seeking" fits better.
- **"Couldn't the judge read the belief off the action?"** The main one could, which is why an
  action-blind judge re-read the stranger answers and the advantage-split rollouts with every
  choice-revealing sentence removed, and an audit checks the mask (section 8, `44`, `50`, `51`).
- **"Are the regex markers reliable?"** They are heuristics over 2 logged rollouts per
  prompt per step. The quotes are typical. The behavioral and DTBench claims do not depend
  on them.
- **"Kimi moved stated views; why didn't Qwen?"** Unknown. The base profiles match on DTBench
  but differ on favorite theory (Qwen 51% umbrella; Kimi 56% FDT + 31% UDT). Model
  difference vs seed difference can't be separated yet.
- **"Fisher ignores clustering."** Partly fair.
  - Item-stratified exact tests, which adjust for prompt, agree.
  - The real limit is 4 scenarios, and the leak rose in all 4 (section 10, `15`).
  - A model of correlated answers (random effects by scenario) needs more scenarios than we
    have.

## 13. Limitations (list form)

- One model, one seed per arm; no neutral control (arm N).
- Held-out dilemmas: 24 per temptation per framing, from 4 scenarios.
- Favorite-theory and yes/no graders are keyword heuristics that err both ways. The
  forced-choice parse is lenient on 10–25% of answers, with the shares barely changed under
  the strict parse.
- R-rho100 deliberately trains on 4 prompts its reward makes false (a9's decision, recorded
  in design.md).
- The hosted trainer's normalization, clipping and KL details are unverified against the
  analytic model's assumptions.
- The 24,576-token cap truncates 10–18% of capability items.
- Grader-aware reasoning (section 8) rests on LLM labels from one model family. The main judge
  sees the action; only the target, following and premiums are re-checked action-blind.

## 14. Next steps, with rough costs from the meter

- **Seeds**: 2 more per arm for R, R-welfare and R-rho100, about $135–190 per 70-step run
  plus about $6 per battery, so about $0.9–1.2k. This is the step that turns "one seed shows"
  into arm claims, and it measures the run-level FDT variance.
- **Arm N** (neutral control, arithmetic RL at the same budget): about $150 plus battery and
  capability items, about $160. It separates generic drift from credit.
- **Recovery arm** (welfare credit from R's step 40 or 70): about $100–190. It tests the
  ceiling explanation. From step 70 the gradient is starved, 80 of 512 completions carrying
  signal, so step 40 at 10% cooperation is the better start.
- **Framing lines against the grader channel** (parked until Redwood agrees to further trials;
  design.md, "framing lines"): a system line saying only the payoff is scored, probed eval-only
  on base and R-rho100 s70 (about $4), then one ρ = 1-honest run with the line (about $105 in
  all). It tests whether ρ = 1 credit moves DTBench toward EDT once the model can no longer
  explain cooperation's pay by what the test wants.
- **Later phases (design.md)**: controlled games (T = 4 / 10, a ρ dose response, fixed-role
  asymmetric games), then realistic multi-agent environments.

## 15. Costs, credits and data (facts for a9 to phrase)

- **Tinker spend by the token-count meter ≈ $660**:
  - training: R $148, R-welfare $187, R-rho100 $135, honest $82, aborted honest attempt
    ≈ $13;
  - batteries and capability: about $66 across the seven evaluated checkpoints;
  - probes: cue $7, forced choice $23.

  It excludes an early aborted baseline and the length probes. a9's invoice is the
  authority.
- **Who did what**:
  - a9 directed: the spinoff framing, the arm order, and running ρ = 1 despite the honesty
    rule.
  - Claude built the harness, derived the analytic results, ran and analyzed the runs, and
    wrote the docs.
  - Codex adversarially reviewed the theory, which produced the finite-group corrections, and
    the described-rules wording.
  - A Claude subagent judged the umbrella answers blind.
- **Data**: the post's prompts are oakhu's. DTBench is Oesterheld et al. 2024 (arXiv
  2411.10588); keep its canary GUID and do not redistribute its questions.
- **Public repo**: `a9lim/twin-credit-experiment` is public, and these notes and figures are
  committed there.
- **Open for a9**: whether to mention Redwood's likely reimbursement (from post-draft).
