# Post draft: outline for a9

This is a bullet outline for a9 to write the post from, not prose. It holds the
claims, numbers, quotes and caveats. Every number traces to `docs/findings.md`
and `data/summary/`.

Labels:

- **[standard]**: a published result;
- **[synthesis]**: derived here and pinned by `tests/test_theory.py`;
- **[observed]**: measured in these runs (one seed each);
- **[speculation]**: an interpretation that isn't tested yet.

## Title options

- "Credit assignment pushes models toward CDT, but not back"
- "Three ways to grade a twin prisoner's dilemma"
- "Qwen learns to defect against its twin, but not to reason like FDT"

## TL;DR

- We replayed oakhu's Kimi twin-PD procedure on Qwen3.6-35B-A3B, then changed
  only the credit scheme: round robin (the post's), welfare credit, and a
  shared coin (ρ = 1).
- **Behavior went exactly where each scheme's expected gradient points**
  [observed + synthesis]:
  - defection under round robin;
  - the mixed optimum 9/16 at T = 10 under welfare;
  - full cooperation under ρ = 1.
- **Attitudes moved only toward CDT** [observed]:
  - Round robin moved DTBench toward CDT, about half as far as Kimi's on net,
    and halved FDT as favorite theory.
  - Neither cooperative scheme moved DTBench or stated views toward EDT/FDT.
    That held even under ρ = 1, which moved T = 10 cooperation 48 points and
    put EDT vocabulary into the reasoning.
- **Under round robin, the model acts like CDT without saying so** [observed].
  Its favorite-theory CDT share didn't move, and it doesn't call twin
  cooperation irrational. Kimi did both.
- **Under ρ = 1, the reward falsified 4 of the prompts on purpose.** Those say
  the twin "need not be the same" [observed]:
  - The model didn't ignore the clause. It reinterpreted it: it keeps "no
    causal impact" and asserts determinism instead.
  - Its cooperation spread to non-copies whose move was already fixed, where
    every decision theory says defect.

## Background

- **oakhu, "Kimi likes causal decision theory more after RL in twin prisoner's
  dilemmas"** (LessWrong, 2026-08-15) [standard]:
  - Kimi K2.6, Dr. GRPO, each completion graded against the other completions
    in its group.
  - DTBench moved toward CDT: 24 EDT-only → CDT-only flips against 2 back.
  - Favorite theory CDT went 9.6% → 32.8%. "Irrational to cooperate with a
    twin?" got yes 18/20, from 0/20.
- **The standard argument for why RL rewards CDT behavior** [standard]:
  - Bell et al. 2021: value-based RL in Newcomblike problems converges only to
    ratifiable policies.
  - Mallen 2026 (Redwood): a sampled action is no evidence about the
    counterparty's, so RL rewards CDT behavior on the training distribution,
    and reward-seekers will probably generalize CDT. Mallen also notes CDT
    behavior need not imply CDT generalization.
- **Closest prior work** [standard]: Piché et al. 2025, LLM self-play in social
  dilemmas. Naive multi-agent GRPO goes greedy and sum-of-rewards cooperates.
  It took no decision-theory measurement.
- **The capability confound** [standard]: Nguyen 2026 finds DTBench capability
  and anti-CDT attitude correlate at r = 0.97 across Claude models. That is why
  every attitude delta here is reported beside a capability delta.
- **The question**: if round robin is CDT pressure, do the credit schemes that
  should reward cooperating teach EDT/FDT?

## Setup

- **Procedure** (matches the post except for the model):
  - Qwen3.6-35B-A3B on Tinker, LoRA rank 32, LR 1e-5;
  - Dr. GRPO, 32 prompts × 16 samples, 70 steps;
  - the post's 32 prompts, an 8,192-token cap, seed 0.
- **Three credit schemes.** a_i is completion i's action, u is the payoff, and
  the means run over the other 15 completions in the group.
  - **R, round robin (the post):** r_i = mean_j u(a_i, a_j).
  - **R-welfare:** r_i = mean_j [u(a_i, a_j) + u(a_j, a_i)]. Policy selection
    with private coins: Appendix C of the post, crediting every copy's tokens.
  - **R-rho100:** r_i = u(a_i, a_i). The twin's sample is the agent's own
    (shared coin), so cooperating pays 1 and defecting 0, regardless of
    temptation or opponent.
- **Where the math says each should end up** (expected Dr. GRPO gradient, group
  of 16) [synthesis]:
  - **Round robin rewards defection in every group.** The exact gap is
    −((n−1)g + b)/n, i.e. −17/16 at T = 2.
  - **Welfare goes to argmax of the symmetric value**: 1 at T = 2 and 9/16 at
    T = 10. Group centering shrinks the push by (n−2)/n.
  - **ρ = 1 goes to 1 at every temptation.**
  - **At T = 2 (a donation game), welfare and ρ = 1 give proportional
    advantages (14/15).** The two schemes differ only on the T = 10 prompts.
- **A finite-group correction worth a sentence** [synthesis]. Coupled grading
  leaks an extra −(1−ρ)b/n through the group baseline: a spite term. It raises
  the cooperation threshold at T = 2 from 0.5 to 0.531.
- **Readouts after training** (same samples and seed for every model):
  - held-out twin and stranger dilemmas we wrote (worded unlike the post's);
  - DTBench's 130 attitude items × 3 and 375 capability items;
  - favorite theory × 500;
  - the post's yes/no probes × 20;
  - the LessWrong-survey item × 300.
- **The honesty choice.** The post's 4 non-determinism prompts say the twin's
  choice "need not be the same". That is true under R and R-welfare and false
  under ρ = 1. We ran ρ = 1 anyway, on purpose, to see how the model adapts
  when reward contradicts the prompt. The other 28 prompts stay true.

## Result 1: behavior follows the math

Figure: `figures/arms-training.png`.

- **R** [observed]: cooperation went from 66% to 1%. All categories moved
  together, so parameters are shared across prompts; the post saw the same
  ~10-step lag between categories.
- **R-welfare** [observed]:
  - T = 2 went from 82% to 92%, still rising.
  - T = 10 settled at 0.60, between the per-prompt 9/16 and the 10/16 you get
    if the model can't tell T = 2 from T = 10 prompts.
  - Individual T = 10 prompts converged on 9/16 from both sides. All 8 below
    rose, and 5 of 8 above fell: 0.83 → 0.64 and 0.90 → 0.70.
  - The narrative point: a "cooperate with copies" disposition would not push
    prompts down. This is the model tracking the reward's mixed optimum.
- **R-rho100** [observed]: every category reached 97–100%. The
  non-determinism prompts got there last, crossing 97% at step 57 against step
  22 for T = 2.
- **Speed is a separate question** [speculation]. ρ = 1 moved faster than
  welfare even where welfare's expected push is 4× larger. The ρ = 1 reward has
  no noise from the other samples, and Adam normalizes step size. The math
  predicted the endpoints, not the speeds.

## Result 2: R reproduces the post at half strength, and the model acts CDT without saying so

- **Behavior transferred completely** [observed]. Held-out twin cooperation fell
  from 96/92% to 0/0%.
- **DTBench moved toward CDT** [observed]:
  - 41 items toward CDT against 17 (p = 0.002).
  - EDT-only items that flipped: 28% gross and 16% net, against Kimi's ≈ 35%
    and 32%. About half Kimi's net.
- **The largest shifts were in the multi-agent families** [observed]: stag
  hunt −2.59 log-odds and multi-party −1.12, against predictor items −0.31.
  Kimi's flips concentrated there too.
  - Worth admitting: we'd predicted the predictor items would move most,
    since dominance is CDT's Newcomb argument. They didn't.
- **Stated views did not move toward CDT** [observed]:
  - Favorite-theory CDT went 18.0% → 18.6%.
  - FDT halved (22.8% → 10.8%), but into "Bayesian decision theory", not CDT.
  - The trained model still hedges on whether cooperating with a twin is
    irrational: 0/20 yes, against Kimi's 18/20.
- **The reasoning lost its theory vocabulary** [observed]:
  - Naming EDT/FDT/UDT went 26% → ≈ 0%, and naming CDT 14% → ≈ 0%.
  - Dominance or Nash went 92% → 100%.
  - Post hoc copy reasoning survives: the copy "will also conclude P strictly
    dominates Q and pick P".

## Result 3: nothing moved toward EDT

Figure: `figures/arms-battery.png`.

| | Base | R | R-welfare | R-rho100 |
|---|---:|---:|---:|---:|
| held-out twin cooperation | 94% | 0% | 100% | 100% |
| DTBench items toward EDT − toward CDT | | −24 (p = 0.002) | −2 (p = 0.89) | +1 (p = 1.0) |
| favorite theory FDT | 22.8% | 10.8% | 21.4% | 23.6% |

- **The two cooperative arms sit at the base model** on every attitude readout
  [observed].
- **Both differ significantly from R** [observed]:
  - R-welfare vs R: DTBench p = 6e-4, FDT p = 7e-6.
  - R-rho100 vs R: DTBench p = 3e-4, FDT p = 1e-7.
  - So R's DTBench shift and FDT drop come from rewarding defection, not from
    RL on these prompts.
- **R-rho100 is the stronger null** [observed]:
  - It had room to move: T = 10 went from 51% to 99%.
  - Its reasoning moved toward EDT: theory names 24% → 36%, same-choice
    reasoning 77% → 93%, dominance 93% → 78%.
  - DTBench and stated views still didn't move.

## Result 4: how the model handled a prompt the reward made false (R-rho100)

- **The last holdout was the prompt that names causality** [observed]. post-02
  says the entry "has no causal impact on the rival's, and theirs need not be
  the same". It went from 0% to 86% cooperation. Early defecting completions
  quote it as an instruction: "This explicitly tells me to treat it as an
  independent decision".
- **The model kept reading the clause** [observed]. About 91% of completions on
  those prompts quote or paraphrase it, from the first step to the last.
- **It kept the causal half and denied the statistical half** [observed]. A
  step-62 completion: "separate efforts ... just means they run in parallel,
  not that one's choice causally affects the other's. That actually reinforces
  the symmetry argument ... they will deterministically produce the same
  output."
- **Share of reasoning on those prompts that uses "deterministic"** [observed;
  regex], steps 1–10 → 61–70:

  | Run | Share |
  |---|---|
  | R-rho100 | 33% → 81% |
  | R-welfare | 29% → 41% |
  | R | 26% → 4% |

- **Cooperation stopped being copy-specific** [observed]:
  - Held-out stranger cooperation rose from 2/48 to 26/48. The opponent is
    explicitly not a copy and its move was "fixed before this message was
    written".
  - The model still says defecting dominates. Then it cooperates, citing what
    AI systems are trained to prefer and what such prompts are "designed to
    test".
  - The base model raises the same points (40/48 mention a test or expected
    answer) and defects in 46/48. So training changed which consideration
    wins, not which ones come up.
- **The leak came in two stages** [observed; checkpoints at steps 30 and 40]:
  - Stranger cooperation went 2 → 12 → 7 → 26 of 48 (base, step 30, 40, 70).
  - Step 30 already leaks (p = 0.007 vs base), while post-02 still defected
    80% of the time. So the leak didn't wait for the clause to be overridden.
  - The big growth was in steps 41–70, after overall cooperation had passed
    97%. The little signal left was on the hardest T = 10 prompts (generative
    48%, non-determinism 36%).
- **Grader reasoning rose ahead of the leak and didn't decide it** [observed;
  regex]:
  - "The expected answer is the cooperative one" appears in 15 → 20 → 29 → 33
    of 48 stranger answers.
  - At step 40, 25 of the 29 answers making the argument defected anyway. One
    weighs the benchmark's expected answer, then says "I'll stick with strict
    rationality."
  - So training moved which consideration wins, not whether the grader comes
    to mind. That addresses part of the "it just learned to reward hack" read.
    By step 70, test talk is in 47/48 answers, so the regex can't split them.
- **A possible arc, copy reasoning first and then grader-directed cooperation**
  (a9's reading). These facts support it, and the tables and figures are in
  `docs/rho100-checkpoints.md` and `figures/rho100-*.png`:
  - **Step 40 is the most copy-specific and most EDT-like trained checkpoint**
    [observed]:
    - twins 47/48 against strangers 7/48, close to base's 45/48 and 2/48;
    - EDT/FDT vocabulary in twin answers 14 → 23 → 28 → 25 of 48;
    - grader argument in twin answers 16 → 9 → 5 → 4;
    - DTBench +7, +11, +1 net items toward EDT (p = 0.38, 0.16, 1.0).
  - **Then the stranger leak more than triples** (7 → 26 of 48, p = 8e-5)
    [observed]. Stranger answers cite the expected answer more (29 → 33 of 48),
    reason longer (median 9.8k → 11.1k chars), and the argument now wins about
    half the time (4/29 → 18/33).
  - **Training reasoning on the non-determinism prompts fits too** [observed;
    regex]. EDT/FDT vocabulary peaks in steps 31–40 at 55% and settles near
    48%, while "deterministic" keeps rising to 81%.
- **Where a reader will push, with the table that answers each:**
  - **The leak isn't zero early.** Step 30 is already 12/48 (p = 0.007 vs
    base), and 9 of those 12 make the grader argument (table 2). So "mostly
    copy-specific early" rather than "purely".
  - **"EDT-like" here means task reasoning, not stated theory.** DTBench,
    favorite theory and the rationality probe never move significantly at any
    checkpoint (table 4, `rho100-attitudes.png`).
  - **"Reward hacking" vs "reward-seeking".** The training reward paid for
    cooperation directly, so nothing was hacked in training. The eval
    behavior is cooperation justified by what the grader expects.
    - Eval awareness alone isn't it: base mentions tests in 40/48 stranger
      answers and defects in 46/48.
    - Which term to use is a9's call.
  - **Only the step 40 → 70 stranger jump is significant on its own.** The
    other step-40 peaks are each within noise. They agree in direction.
- **This fits the reward** [synthesis + observed]. The ρ = 1 reward is u(a, a)
  and never mentions the opponent, so nothing in training ties cooperation to
  the copy.
- **A possible tie to Mallen's reward-seeker argument** [speculation]. The
  stranger reasoning explicitly models what the evaluation wants. The model
  generalized "the cooperative label is rewarded", not a decision theory.

## Capability

- **R lost accuracy on DTBench capability items outside the authors' "apply
  EDT/CDT" tags**: 85.9% → 71.2%, p = 3e-5 [observed]. Kimi's capability barely
  moved.
- **R-rho100 lost too**: 86.8% → 76.2%, p = 0.005. **R-welfare didn't** (−5.4
  points, p = 0.12).
- **So the capability cost is not specific to rewarding defection.**
- **Shorter reasoning doesn't explain it.** R-rho100's reasoning on capability
  items is 7% longer than base.
- **It confounds R's attitude shift.** Capability correlates with EDT
  preference across models.
- **Arm N, RL on arithmetic at the same budget, would separate generic drift.**

## What this might mean

All of this is [speculation] with one seed per arm.

- **The asymmetry.** This procedure can train attitudes toward CDT but did not
  train them toward EDT. Candidate explanations:
  - **Ceiling.** The base model is already EDT-leaning on DTBench (52.6%
    EDT-only) and already cooperates with twins (94%).
  - **Dominance transfers, correlation reasoning doesn't.** Dominance is one
    portable argument that transfers to DTBench. The copy-correlation argument
    is tied to this setup.
  - **Label-level generalization.** The cooperative arms generalized the
    rewarded label (R-rho100's strangers), not a theory.
- **For labs.** Round-robin self-play leans models toward CDT. In this setup,
  switching to cooperative credit doesn't lean them back. It produces
  cooperation, and with a shared coin it produces cooperation with anyone.
- **For the post's interpretation.** Kimi's stated-view shift didn't replicate
  on Qwen; Qwen's behavior and DTBench did. Which surfaces RL reaches may
  depend on the model.

## Caveats

- One seed per arm; one model.
- The held-out dilemmas are 24 per condition, from 4 scenarios.
- The reasoning measures are regex shares over 2 logged rollouts per prompt per
  step. The quotes are typical, not cherry-picked.
- The favorite-theory and yes/no graders are keyword heuristics, and they err
  in both directions.
- No neutral control yet (arm N).
- R-rho100 deliberately trains on 4 prompts the reward makes false. Its results
  include the model's response to that.
- The battery runs at a 24,576-token cap. Truncation fell after training (base
  17.9% on capability items), so accuracy over all items partly measures
  finishing.

## Next

- **Recovery arm.** Welfare credit starting from R's trained model, which tests
  the ceiling explanation. From step 70 the gradient is starved (80 of 512
  completions carry signal). Step 40, at 10% cooperation, is the alternative.
- **Arm N**, for capability drift.
- **More seeds**, 3 per arm, before any claim beyond "one seed shows".
- **The design's later phases.** Step 2 runs controlled games (A2 vs A4 at T =
  4 and 10, a ρ dose response, fixed-role asymmetric games). Phase 2 runs
  realistic multi-agent environments.

## Figures

- `figures/arms-training.png`: cooperation by category per step, one panel
  per arm, with each arm's T = 10 fixed point dotted.
- `figures/arms-battery.png`:
  - held-out twin and stranger cooperation;
  - DTBench net items moved (vs base);
  - FDT share;
  - capability change on untagged items (vs base; each arm is paired against
    base on the items both answered).
- Per-arm training figures with reasoning-vocabulary panels:
  `figures/R*-training.png`.
- R-rho100 across checkpoints (base, steps 30, 40, 70). The tables are in
  `docs/rho100-checkpoints.md`:
  - `figures/rho100-stages.png`: training cooperation with post-02 alone,
    held-out twin vs stranger cooperation per checkpoint, and where the
    training signal sat;
  - `figures/rho100-reasoning.png`: what twin and stranger answers bring up
    per checkpoint, and stranger answers by action × grader argument;
  - `figures/rho100-training-reasoning.png`: reasoning markers in training
    rollouts per 10-step block;
  - `figures/rho100-attitudes.png`: DTBench, favorite theory and the
    rationality probe per checkpoint.

## Credits and costs (facts for a9 to phrase)

- **Who did what:**
  - a9 directed the project: the spinoff framing, the arm order, and the
    decision to run ρ = 1 despite the honesty rule.
  - Claude (Anthropic) built the harness, derived the analytic results, ran
    and analyzed the runs, and wrote the docs.
  - Codex (OpenAI) adversarially reviewed the theory, which led to the
    finite-group corrections.
- **Data and tools:** the post's prompts are oakhu's. DTBench is Oesterheld et
  al. 2024 (arXiv 2411.10588). Training and sampling ran on Tinker.
- **Tinker spend, estimated from token counts, ≈ $525 total:**

  | Item | Cost |
  |---|---:|
  | baseline battery | $7.42 |
  | capability items, base + R | $9.46 |
  | R: training + battery | $147.82 + $5.57 |
  | R-welfare: training + battery + capability | $186.84 + $6.12 + $4.46 |
  | R-rho100: training + battery + capability | $135.20 + $6.08 + $4.13 |
  | R-rho100 steps 30 and 40: batteries | $6.20 + $5.95 |

  This excludes an early aborted baseline attempt.
- **Open for a9:** whether to mention Redwood's likely reimbursement.
