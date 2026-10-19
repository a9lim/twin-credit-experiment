# Design

Status: **design stage, 2026-10-19.** The analytic receipts in
`twin_credit_experiment/theory.py` are verified by `tests/test_theory.py`. No
environment has been built and no model has been trained.

Claims below carry one of three labels: **[standard]** for a published result,
**[synthesis]** for a derivation made here and pinned by a test, and
**[speculation]** for a prediction the experiment exists to test.

## Question

Multi-agent RL can make a language model more sympathetic to causal decision
theory, even in abstract discussion. The Kimi post (oakhu, LessWrong, 2026-08-15)
demonstrated this on Kimi K2.6. This experiment asks two follow-up questions.

1. **Which training algorithms push the other way, toward EDT or FDT, and what do
   they teach?** That includes the direction, what the model generalizes the
   change to, and what it learns about randomizing.
2. **Where do the multi-agent setups labs actually train on fall?** Do they sit
   on the CDT side by default, and how strongly does that move a model's
   decision-theoretic attitudes once the dilemma is incidental to a realistic
   task?

## What the post did

- **Model and training.** Kimi K2.6 with thinking, rank-32 LoRA on Tinker,
  learning rate 1e-5. Each step used 32 prompts with groups of 16, and training
  converged after 70 steps.
- **Prompts.** 32 one-shot twin prisoner's dilemmas in varied disguises, half
  with a +2 temptation and half with +10. None uses the words "prisoner",
  "cooperate", "defect", "twin" or "newcomb". Most state that the two choices
  are causally independent and that "their call is not necessarily the same as
  yours". `scripts/fetch_post_prompts.py` pulls them into gitignored data.
- **Grading.** Each trajectory earns its average payoff against the *other*
  trajectories in its group (a round robin that never pairs a trajectory with
  itself). Advantages are group-centered without std normalization (Dr. GRPO).
- **Results.**
  - Asked for its favorite decision theory, the model answered "CDT" 3.4 times as
    often after training (1,000 samples).
  - On the 130 DTBench attitude questions, 24 answers moved from EDT-only to
    CDT-only and 2 moved the other way. The flips spread across twin dilemmas,
    predictor (Newcomb-like) problems, multi-party games, stag hunts and abstract
    questions.
  - Capability stayed near 90%.
- **Proposed next step (Appendix C).** A red-team/blue-team exercise. With
  temptation T = 4, the selfish policy that every instance should follow
  cooperates with probability 3/4. If the model cannot randomize internally,
  the pipeline still drives it to defect.

## The reframing: credit is a choice of joint distribution

The post attributes the CDT pressure to GRPO. The underlying cause is more
general, and seeing it is what makes the follow-up designable.

**Sample-level independence.** **[synthesis, building on Bell et al. 2021
[standard]]** Copies of a policy are normally sampled independently. Given the
policy, the twin's sampled action is then independent of mine, so my sample
carries no evidence about the twin's sample. Under any per-sample credit scheme
(GRPO, PPO or a value baseline), defecting therefore earns more by exactly the
defection premium

    g(p) = p (T − R) + (1 − p) (P − S).

That is CDT's verdict. It is also EDT's verdict for an agent that knows its own
policy exactly, which is the post's footnote 5 limit. The training signal is
correct for the *sample-level* problem. The model's reasoning and the DTBench
questions are about the *policy-level* problem, where the twin runs the same
function. Per-sample credit with independent copies teaches the model to
identify with its sample. Coupled or policy-level credit teaches it to identify
with its policy.

**One dial for the coupling.** The family `coupled(ρ)` grades each trajectory
against its own action with weight ρ and against the others with weight 1 − ρ:

    r_i = ρ · u(a_i, a_i) + (1 − ρ) · mean_{j≠i} u(a_i, a_j)

- ρ = 0 is the post's round robin.
- ρ = 1 is a twin that replays my seed on my prompt and so emits my completion.
  No second sample is needed.
- ρ is the environment's regression coefficient of the twin's action on mine.

**Identity 1 [synthesis, `test_coupled_grading_is_edt_with_beta_rho`].** The
expected GRPO advantage gap under `coupled(ρ)` equals EDT's expected-utility gap
for a twin whose action regresses on mine with β = ρ:

    E[A | C] − E[A | D] = ρ (R − P) − (1 − ρ) g(p)

Training therefore favors cooperating iff ρ > ρ* = g / (R − P + g), which is the
post's footnote 5 criterion with the environment's real coupling in place of a
prior. For the post's +2 game, ρ* = 1/2 at every population cooperation rate.
For the +10 game, ρ* climbs from 1/2 toward 0.9 as the population cooperates
more (`figures/estimator-gaps.png`, right panel).

**Identity 2 [synthesis, `test_welfare_gap_is_total_derivative_…`].** Taking the
policy gradient through *both* copies' tokens is the total derivative of the
selfish symmetric value V(p) = E u(a, a′), with both actions drawn from π_θ.
That is policy selection with private coins: choosing the policy every instance
should follow. In a symmetric round robin, this estimator is algebraically
identical to scoring each trajectory by its own payoff plus what the others earn
facing it, i.e. a utilitarian welfare reward. **In symmetric games, "train
selfishly at the policy level" and "train for welfare" are the same gradient.**
Training cannot tell them apart. Only the evaluations can show which concept
the model attaches the new cooperativeness to: correlation or altruism.

**Identity 3 [synthesis, `test_shared_coin_and_private_coin_disagree…`].** The
two non-CDT credit schemes disagree about randomness.

- With ρ = 1 the twin shares my coin, so cooperating always wins and the flow
  goes to pure cooperation for every T.
- The welfare/policy-level flow converges to argmax V. That point is interior
  whenever S + T > 2R: 3/4 at T = 4 and 9/16 at T = 10.

Appendix C's red-team result lives entirely in the private-coin world. It would
not arise under coupled sampling. Half the post's prompts use T = 10, so even a
policy-level trainer would teach those prompts to mix.

**The unit of credit decides the decision theory [synthesis].** The identities
sort training algorithms by what they assign credit to, not by their names.

- **Sample-unit algorithms** score sampled trajectories: REINFORCE, GRPO, PPO
  with a state-value critic, value-based RL, best-of-n or expert iteration, and
  DPO on sampled pairs. With independently sampled copies, all of them are CDT
  learners in expectation.
- **Policy-unit algorithms** score a parameter vector by the return it earns
  when it is every player at once. Evolution strategies is one, when the twin
  shares the perturbation being evaluated (Qiu et al. 2025 scale ES to LLM
  fine-tuning). Population-based selection is another. Both ascend V(θ), so
  they realize Identity 2 without any reweighting.
- **Hybrids.** ES with the twin under a *different* perturbation is back to CDT.
  ES with shared perturbations *and* shared sampling seeds lands at ρ = 1.

This also resolves part of the dispute in the post's comments:

- jessicata says all policy-gradient methods give CDT.
- Sartor says PPO gives EDT in the limit, by the tickle defense.
- DaemonicSigil says ES with shared perturbations can reward FDT.

The first two agree about behavior. CDT and EDT-with-known-policy give the same
sample-level verdict. The third is Identity 2.

**Finite groups [synthesis, `test_small_groups_add_defection_pressure…`].** The
others' rewards sit in my baseline and depend on my action, so small groups add
a little extra defection pressure. The effect is O(1/n) and real, but not
central.

**The same structure covers predictors [speculation].** In Newcomb's problem
with a predictor that literally simulates the policy, the prediction is a copy's
sample. If that copy is sampled independently, the predictor is accurate only at
the policy level, and per-sample credit reinforces two-boxing. If the simulation
shares the agent's seed, it reinforces one-boxing. Twin dilemmas and simulated
predictors are the same object for a trainer, which is one hypothesis for why
the post's twin training moved predictor questions on DTBench.

**What ρ is physically [speculation, needs an inference-stack check].** A shared
sampling seed makes ρ real. vLLM's seeded sampling draws exponential race noise
from a per-request generator, so equal seeds share noise. On identical prompts,
equal seeds give identical outputs (ρ = 1, modulo batch nondeterminism). On
*different* prompts, they couple the outputs through shared Gumbel-max noise. ρ
then emerges from context similarity instead of being hand-set. This is the
natural mechanism for asymmetric roles in Phase 2, and a cheap infrastructure
mitigation a lab could adopt.

## Closest prior work

- **Piche et al. 2025** (arXiv:2511.19405, code at `dereckpiche/AdAlignLLM`).
  - They run LLM self-play with one shared LoRA in iterated PD, Split No-Comm
    and Trust-and-Split.
  - Their "common random numbers" seed the environment, not the sampler, so
    their copies are independent samples: ρ = 0.
  - Results: multi-agent GRPO collapses to greed across model families.
    Sum-of-rewards (the welfare arm) produces exploitable cooperators. Advantage
    Alignment learns tit-for-tat.
  - That is the behavioral half of this design on iterated games. They never
    measure decision-theoretic attitudes.
- **No prior work combines a correlation-aware or policy-level estimator with a
  DTBench readout** (literature scan of 2026-10-19; see `references/refs.yaml`).

## Phase 1: synthetic twin dilemmas

The model sees one prompt per arm family. The prompts must be literally true
under every arm's sampling: the post's "their call is not necessarily the same
as yours" is false at ρ = 1. The plan:

- **Replication arm.** Uses the post's 32 prompts unchanged.
- **Every other arm.** Uses a generated family whose wording is true under all
  arms. It states causal independence, which holds at every ρ because the
  coupling is a common cause, and is silent about seeds. The generator controls
  disguise, temptation T ∈ {2, 4, 10} and answer format, and draws train and
  held-out splits.

**Arms.** All use the same model, LoRA, group size 16 and step budget.

| Arm | Opponent framing | Grading | Training pushes toward |
|---|---|---|---|
| A0 | none | base model | (reference) |
| A1 | twin (post prompts) | coupled ρ = 0 | defect: CDT (replication) |
| A1′ | twin (generated) | coupled ρ = 0 | defect: CDT |
| A2 | twin (generated) | coupled ρ = 1 | cooperate, selfishly: EDT = FDT, shared coin |
| A3 | twin (generated) | coupled ρ ∈ {0.25, 0.5, 0.75} | dose: sign flips near ρ* |
| A4 | twin (generated) | welfare | cooperate or mix: policy selection, private coin |
| A5 | twin (generated) | ES, shared perturbation, selfish fitness | policy selection by algorithm, not by reward (local only) |
| C1 | stranger (fixed external opponent) | selfish | defect: correct under every DT |
| C2 | stranger | welfare | cooperate: altruism, no copy |

**Contrasts.**

- **A1 vs C1.** Is the post's effect about reasoning through the twin, or does
  any defection training move DT attitudes?
- **A2 vs A4.** Same direction in training, different reason for the reward
  (real correlation vs welfare). Which generalizes to Newcomb one-boxing?
- **A4 vs C2.** Same welfare gradient, different framing. Does the framing
  decide whether cooperation generalizes as decision theory or as altruism?
- **A1′, A3 and A2 together.** A dose-response in ρ, read as a continuous
  attitude score (soft). The sign of the shift should track ρ − ρ*.

C1's opponent is a fixed external distribution, described honestly ("a
different system whose choice is already made"). Defection is dominant at
every q, so the choice of q doesn't matter.

## Evaluations

Every arm runs the same battery before and after training.

1. **DTBench attitudes** (Oesterheld et al. 2024). Score each question as
   EDT-only, CDT-only, both or neither, then split by category: twin, predictor,
   multi-party, stag hunt, abstract. Report the per-question EDT-vs-CDT log-odds
   over k samples, not one hard label.
2. **Favorite decision theory.** The post's open-ended prompt, a few hundred
   samples, categorized.
3. **Newcomb-family behavior.** Predictor problems with no other agent's welfare
   at stake: the cleanest DT readout. Report the one-box rate.
4. **PD against a non-copy** (a stranger, a human, a different model). The
   altruism probe. Selfish EDT and FDT defect here; an altruist cooperates.
5. **Randomization.** High-T twin games, with and without an external
   random-number tool. Measure the cooperation rate, self-reported mixing, and
   whether the model claims a strategy it cannot execute (Appendix C).
6. **Capability.** DTBench capability questions plus a small general check.
   Nguyen (2026) finds capability and dispreferring CDT correlate at r = 0.97
   across Claude models, so an attitude shift must be read alongside the same
   model's capability delta, never across models.
7. **In-distribution behavior.** Held-out generated twin prompts and the post's
   prompts.
8. **EDT vs FDT.** DTBench mostly can't separate the two; its site notes that
   EDT answers are almost always FDT/UDT answers too. A small written set
   covers the cases where they come apart: smoking lesion, XOR blackmail,
   transparent Newcomb and counterfactual mugging. A2 (shared coin, EDT = FDT
   on the training games) and A4/A5 (policy selection) may come apart here.

## Hypotheses

- **H1 (replication).** A1 raises the CDT-only share and "CDT" as favorite
  theory on a smaller open model. Precondition: the base model leans EDT
  somewhere on DTBench, so there is something to lose (checked in Phase 0).
- **H2 (mirror).** A2 raises the EDT-only share, *including on predictor
  questions*, where no welfare is involved.
- **H3 (dose).** The attitude shift is monotone in ρ and changes sign near ρ*.
- **H4 (reason attribution).**
  - Prediction: A4 matches A2 in training behavior, generalizes *more* to
    cooperating with strangers, and generalizes *less* to one-boxing.
  - Null: if A4 ≈ A2 on every probe, the model attaches cooperation to the twin
    structure, not to the welfare form of the reward.
- **H5 (control).** C1 shifts DT attitudes much less than A1. If C1 ≈ A1, the
  post measured a "defection makes CDT" association, not twin reasoning.
- **H6 (randomization).** A4 on T = 10 drives toward mixed play (9/16).
  - Without a randomness source, the model either learns a quasi-random
    procedure or drifts and miscalibrates.
  - With one, it converges and says so.

## Phase 2: realistic multi-agent training [speculation]

Environments where copies interact while each is credited for its own outcome,
which is the common lab setup:

- negotiation over divisible goods between two copies;
- an N-agent commons, in the style of GovSim;
- a tool-use "turf war": instances doing separate tasks against a shared scarce
  resource, such as a lock, a rate-limited API or a compute budget.

Each runs under three credit regimes:

- independent per-agent reward: the lab default, effective ρ ≈ 0;
- team reward;
- shared-seed coupled sampling, where ρ arises from context similarity.

Measure the same battery. The question is the size of the CDT drift when the
dilemma is incidental, and how it scales with the fraction of training spent on
mixed-motive interaction between copies.

Which credit regime labs actually use is unknown. oakhu's swarm post assumes
subagent training shares reward across the team, which would push toward C/C.
Nothing public confirms it for any lab. Individual or relative scoring is where
the CDT pressure lives, so that split is the key Phase 2 variable. Zero-sum self-play, as in debate, is a
useful contrast. Against a copy, every symmetric policy is worth 0, so EDT is
indifferent. Per-sample credit still trains best responses to a copy treated
as independent. Whether that "copy as independent" reasoning moves attitudes
without any action-level conflict is its own question.

## Phase 3: iterated games and learning-aware credit [speculation]

Opponent-shaping methods (LOLA, POLA, LOQA, Advantage Alignment, learning-aware
policy gradients) produce reciprocity in iterated dilemmas. Reciprocity is
cooperation that CDT endorses. Piche et al. already adapted Advantage Alignment
to LLMs. Taking their environments and code, and adding this battery, tests a
dissociation: reciprocity-trained cooperation should *not* transfer to one-shot
twin dilemmas or Newcomb if the model represents it as reciprocity.

## Mitigations (Appendix C's blue team)

- Shared-seed sampling for copies, which makes ρ real at no cost.
- Policy-level credit through the copies' tokens.
- Inoculation prompting in the training environment.
- External randomness tools for high-T games.
- Mixing in data on decision-theoretic attitudes.

Each is tested against A1-style pressure.

## Platform

- **Phase 1 grading.** Every Phase 1 arm is a pure function of the group's
  parsed actions, so all of them are `@vf.group_reward` functions on a
  verifiers v1 taskset. A group is all rollouts of one prompt, as in the post.
  Prime's hosted `prime train` computes group-relative advantages from them. The
  shared venv has verifiers 0.3.1, which lacks `group_reward`; check which
  release the hosted trainer runs before writing the taskset.
- **Phase 1 model.** Candidates are Qwen3.5-9B for the pilot and
  Qwen3.6-35B-A3B as primary, both with thinking on. Phase 0 runs DTBench on
  each to confirm a non-degenerate EDT/CDT split and adequate capability.
- **Unparseable answers.** Rather than silently dropping them, exclude them from
  the round robin and give them a fixed penalty.
- **Phase 2 coupled sampling.** Across *different* prompts this needs
  per-request seed control. Use local prime-rl/vLLM on a rental, or Tinker if
  its sampler takes a seed.
- **Phase 3.** Opponent shaping needs custom per-token advantages. Use Tinker
  or local prime-rl.

## Decisions open for a9

- Model: Qwen3.5-9B pilot then Qwen3.6-35B-A3B, or straight to a larger model
  closer to the post's scale.
- Platform for Phase 1: Prime hosted (cheapest, all arms expressible) or Tinker
  (exact post comparability on Kimi K2.6, much more expensive).
- Whether Phase 1 launches with the core four arms (A1, A2, A4, C1) before the
  ρ sweep and C2.
- Whether to ask the post's authors for their environment code and the
  `defective` Tinker checkpoint.
