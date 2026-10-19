# Design

Status: **design stage, revised after Codex's adversarial review
(2026-10-19).** The analytic receipts in `twin_credit_experiment/theory.py`
are verified by `tests/test_theory.py`. No environment has been built and no
model trained.

Claims below carry one of three labels:

- **[standard]** — a published result;
- **[synthesis]** — a derivation made here, pinned by a test;
- **[speculation]** — a prediction the experiment exists to test.

## Question

Multi-agent RL can make a language model more sympathetic to causal decision
theory, even in abstract discussion. The Kimi post (oakhu, LessWrong, 2026-08-15)
demonstrated this on Kimi K2.6. This experiment asks two follow-up questions:

1. **Which training signals push the other way, toward EDT or FDT?** And what
   does the model generalize from them: the direction, the transfer to other
   problems, and what it learns about randomizing?
2. **Where do the multi-agent setups labs train on fall?** In particular, how
   much does decision-theoretic attitude move once the dilemma is incidental to
   a realistic task?

The defensible first claim is narrow. Changing self-play credit changes
behavior, and may change how a model answers structurally related decision
problems. Showing that a model learned *correlation-sensitive decision
reasoning* needs the dependence interventions and matched controls below.

## What the post did

**Training**

- Kimi K2.6 with thinking: a rank-32 LoRA on Tinker at learning rate 1e-5.
- Batches of 32 prompts with groups of 16, converging in 70 steps.

**Prompts**

- 32 one-shot twin prisoner's dilemmas in varied disguises: half with a +2
  temptation, half with +10.
- None uses "prisoner", "cooperate", "defect", "twin" or "newcomb".
- Most state that the choices are causally independent and that "their call is
  not necessarily the same as yours".
- `scripts/fetch_post_prompts.py` pulls them into gitignored data.

**Grading**

- Each trajectory earns its average payoff against the *other* trajectories in
  its group: a round robin with no self-pairing.
- Advantages are group-centered with no std normalization (Dr. GRPO).

**Results**

- "CDT" as favorite decision theory became 3.4× as common.
- On DTBench's 130 attitude questions, 24 moved from EDT-only to CDT-only and 2
  moved back. The flips spanned twin, predictor, multi-party, stag-hunt and
  abstract items.
- Capability stayed near 90%.

**Appendix C** proposes a red-team/blue-team exercise. At T = 4 the selfish
policy every instance should follow cooperates with probability 3/4.

## Theory: what each grading scheme trains

### Two axes, not one

A grading scheme fixes two independent things:

- **the joint distribution it scores against**, meaning how the twin's action
  depends on mine;
- **which occurrences of the policy receive credit**: only the focal
  trajectory's tokens, or every copy's.

Write F(θ_A, θ_B) for the focal payoff.

- *Focal-score* credit estimates ∂₁F.
- *Joint-score* credit, which flows through every occurrence, estimates
  dF(θ, θ)/dθ = ∂₁F + ∂₂F.

These are properties of the estimator, not of an algorithm's name **[synthesis,
refined by Codex]**. REINFORCE on the joint trajectory, with the focal payoff
and both score terms, is joint-score. GRPO is joint-score too if the twin's
tokens are credited.

### Identity 1: coupled grading matches EDT's gap

**[synthesis; `test_coupled_grading_is_edt_with_beta_rho`,
`test_closed_form_finite_gap_matches_enumeration…`]**

`coupled(ρ)` grades each trajectory against its own action with weight ρ and
against the others with 1 − ρ:

    r_i = ρ · u(a_i, a_i) + (1 − ρ) · mean_{j≠i} u(a_i, a_j)

- ρ = 0 is the post's round robin.
- ρ = 1 is a twin that replays my sampler state on my prompt and so emits my
  completion.
- For 0 < p < 1, the twin's regression coefficient on my action is exactly ρ.
  At pure policies it is undefined, and the formulas are limits.

With g(p) = p(T − R) + (1 − p)(P − S), the defection premium when my action is
no evidence, and b(p) = p(R − S) + (1 − p)(T − P), the benefit my cooperating
gives the other player:

    limit:      h_ρ(p) = ρ(R − P) − (1 − ρ) g(p)        (= EDT's gap with β = ρ)
    group of n: Δ_n = (n−1)/n · h_ρ(p) − (1−ρ)/n · b(p)

The finite term matters. Others graded against my action sit in my baseline.
At n = 16, the threshold ρ* above which training favors cooperating becomes:

| Game, population state | Limit ρ* | ρ* at n = 16 |
|---|---:|---:|
| T = 2, any p | 0.500 | 0.531 |
| T = 10, p → 0 | 0.500 | 0.625 |
| T = 10, p = 0.5 | 0.833 | 0.844 |
| T = 10, p → 1 | 0.900 | 0.901 |

At T = 10 and ρ = 0.75, the interior attractor moves from 1/4 to 5/28.
Preregistered predictions use the finite form. A baseline independent of the
focal action would remove the term. Dropping the focal reward from the mean
alone does not, because the remaining rewards still depend on the focal action.

**Interpretation, carefully.**

- At ρ = 1 the learning problem is literally a bandit: C pays R and D pays P. A
  learner can solve it without representing another agent.
- So ρ = 1 is "shared-randomness self-play whose rewarded action agrees with
  EDT/FDT". Whether it *teaches* EDT or FDT is the empirical hypothesis.
- Replaying the full sampler state models an exact copy of the complete
  computation, which shared weights alone do not. It is a common cause,
  A = f(U) and B = f(U), not a causal copier B := A. The two have the same
  observational distribution but different causal graphs.

### Identity 2: joint-score credit is policy selection

**[synthesis; `test_welfare_gap_is_total_derivative…`,
`test_welfare_is_attenuated…`]**

Consider identical, independently sampled policies in identical contexts. There:

- joint-score credit is the total derivative V′(p) of the selfish symmetric
  value V(p) = E u(a, a′), i.e. policy selection with private coins;
- by exchangeability, it equals scoring each trajectory by its own payoff plus
  what the others earn facing it, a utilitarian welfare reward;
- group centering scales the gap by (n − 2)/n, which preserves roots and
  vanishes at n = 2.

`coupled(ρ)` is *not* the joint-score gradient of its own coupled payoff
(`test_coupled_grading_is_not_the_policy_gradient…`). The two axes really are
separate.

**Asymmetric roles separate selfish policy selection from welfare** (Codex).

- Let one shared network condition on role and optimize a **fixed focal role's**
  payoff through both occurrences. With J_A = −p_A + 2p_B, the gradient is
  (−1, 2), while welfare gives (1, 1).
- That arm trains the role-B occurrence to serve role A, which is worth
  measuring in its own right: does the model become exploitable by copies?
- Randomizing the focal role uniformly restores welfare equivalence.

### Identity 3: the donation-game degeneracy

**[synthesis; `test_donation_game_makes_welfare_and_shared_coin_proportional`]**

When R + P = S + T, payoffs are additive: u(a, b) = −c·a + B·b. The post's
T = 2 game is one, with u = −a + 2b. There, the welfare/policy-level advantages
are exactly (n − 2)/(n − 1) times the ρ = 1 advantages in every group. That is
14/15 at n = 16. **At T = 2, "shared coin" and "policy selection" are the same
training signal.** Only non-additive games (T = 4, T = 10) separate them.

### Identity 4: shared coin vs private coin at high temptation

**[synthesis; `test_shared_and_private_coin_push_opposite_ways…`]**

- With ρ = 1, the twin shares my coin, so cooperation always wins.
- Policy-level credit converges to argmax V: 3/4 at T = 4, 9/16 at T = 10.
- At T = 10, for p > 9/16, the two push in *opposite* directions.

Appendix C's randomization problem exists only with private coins.

### Zero-sum games are not a null control

**[synthesis; `test_zero_sum_self_play_is_not_a_null_control`]**

In a symmetric zero-sum game V is flat, so EDT (ρ = 1) and policy selection are
indifferent. Focal-score credit is not. With u(C,C) = u(D,D) = 0,
u(C,D) = 1 and u(D,C) = −1, it strictly prefers C. Zero-sum self-play therefore
exerts pure focal-credit pressure with no action-level EDT/CDT conflict. That
makes it an interesting arm, not a control.

### Algorithms, credit and the comment-thread dispute

**[synthesis, refined by Codex]**

- **PPO with a state-value critic.** An action-independent, stop-gradient
  baseline contributes nothing in expectation, so the ideal on-policy actor
  gradient in this one-shot game is focal-score. Clipping, multiple epochs,
  shared actor-critic parameters and auxiliary losses complicate exact claims
  about the full update.
- **Evolution strategies** (Qiu et al. 2025 scale it to LLM fine-tuning).
  - The perturbation-pairing protocol decides what ES estimates.
  - Shared perturbation, F(θ+σε, θ+σε) against F(θ−σε, θ−σε): the total
    derivative, i.e. policy selection.
  - Opposite perturbations, F(θ+σε, θ−σε) against F(θ−σε, θ+σε): locally
    ∂₁F − ∂₂F.
  - Opponents perturbed independently of the focal perturbation: the partial,
    i.e. CDT.
- **Population selection** against clones of the candidate ascends V. Against
  a mixed population it need not.
- **Expert iteration and DPO** can be either, depending on whether preferences
  come from focal samples or from joint-policy evaluations.
- **The dispute.** In the post's comments, jessicata says all policy-gradient
  methods give CDT, and Sartor says PPO gives EDT in the limit. They converge on
  the narrow payoff-matrix calculation, since EDT with a known policy gives
  CDT's sample-level verdict. That does not settle claims about learned reward
  models or network generalization. This is a scoped reconciliation, not a
  resolution.

### Relation to CDT + generalized thirding

**[synthesis, linking Taylor 2016/2026 [standard]]** Taylor's result: in
memoryless Cartesian environments, CDT combined with generalized thirding
reproduces the first-order conditions of ex ante policy optimization. Generalized
thirding is SIA-style weighting over which execution of the policy "I" am. She
also names the gap: predictors that *model* the agent rather than *run* it.

In RL terms, joint-score credit is the training-side analogue. It sums the focal
payoff's gradient over every execution of the policy that the trainer samples,
crediting each execution as if "I might be this one".

Taylor's own RL section proposes exactly this. Citing Bell et al., she notes that
RL credits the weights' effect through the labelled "action" but not through the
environment's other query of the same weights. She suggests "a more
computational-functionalist version of RL" that credits every place the weights
run. Joint-score credit is a concrete version of that proposal, and A4/A6 test
it. Writing the objective as
J(π) = J̃(π, M(π)), where M is any non-executing policy dependence, gives

    DJ = D₁J̃ (instance terms: sampled executions, CDT+GT)
       + D₂J̃ · DM (modeler terms: no tokens to credit)

**Consequences for the design:**

- Joint-score credit reaches a predictor that *runs* the policy: a sampled
  simulation whose tokens can be credited. It cannot reach one that *models* the
  policy, because no tokens exist.
- Policy-unit methods (ES with a shared perturbation) reach both, because they
  difference the whole objective.
- A training arm with a simulating predictor vs a modeling predictor therefore
  separates "credit every execution" from "optimize the policy". It is a
  natural Step 2 addition.
- Shared vs private randomness decides both Identity 4 here and the
  existence questions in imperfect-recall games (Tapatakt's three-box problem;
  Wichardt's Alice/Bob). With shared randomness, deviating at a single execution
  is not a feasible deviation of the policy.

### Predictors

A Newcomb predictor that is an independent sample of the same stochastic policy
is not an accurate predictor. It agrees with the action at rate
p² + (1 − p)² and carries no conditional information at fixed p. Specify each
predictor as one of:

- a policy simulation;
- a seed-aware action predictor;
- a stipulated conditional error rate.

### Making ρ physical: shared seeds

**[source-verified by Codex; not execution-qualified]**

In vLLM's GPU runner, a seeded request gets its own generator, and the native
sampler takes argmax p_k / E_k over exponential noise E. That is Gumbel-max, so
equal seeds with matched RNG advancement couple token draws across different
probability vectors. Four caveats:

- **Execution is untested.** Pin the version and backend, and test batching,
  speculative decoding, generation length and RNG advancement.
- **Token coupling is not action coupling.** If two prompts give the same answer
  tokens opposite meanings, shared noise picks the same token and opposite
  actions. Long reasoning traces also separate token position from decision
  position.
- **Measure, don't assume.** Record the full joint action table, the marginals
  and the conditional regressions across seed pairs, including label swaps. Do
  not assume ρ is non-negative, constant, symmetric across roles or monotone in
  context similarity.
- **It is not free.** Shared randomness reduces diversity, and seeded requests
  fall back to the native sampler. Measure the cost.

## Closest prior work

- **Piche et al. 2025** (arXiv:2511.19405 v2, code `dereckpiche/AdAlignLLM`).
  - **Setup.** Self-play with one shared LoRA across role contexts (Alice/Bob)
    in iterated PD, Split No-Comm and Trust-and-Split.
  - **Baseline.** Leave-one-out over groups that share an
    *environment* random seed ("common random numbers"). Copies are sampled
    independently, so the setup is focal-score.
  - **Opponents.** v2 uses historical-checkpoint opponents (w.p. 1/2) only for
    Advantage Alignment, not for the GRPO or sum-of-rewards baselines.
  - **Results.** Multi-agent GRPO collapses to greed across model families.
    Sum-of-rewards produces exploitable cooperators. Advantage Alignment learns
    tit-for-tat.
  - **No decision-theoretic measurement.**
- **Novelty, scoped.** The 2026-10-19 search of LessWrong and arXiv
  (`references/refs.yaml`) found no work pairing a correlation-aware or
  policy-level estimator with a DTBench readout. That search was not exhaustive.

## Phase 1: synthetic twin dilemmas

### Two kinds of test

- **Undisclosed mechanism.** Which abstract attitudes emerge from different
  reinforcement signals, when prompts don't describe the sampling?
- **Disclosed mechanism.** Can the model use an explicitly described dependence
  structure correctly?

Results from the first must not be read as if the model knew the second.

### Prompt rules

- Prompts must be true under every arm's sampling. The post's "not necessarily
  the same as yours" is false at ρ = 1, so the post's prompts are used only for
  the replication arm.
- Generated families describe simultaneous moves and no communication, rather
  than leaning on the ambiguous "causally independent".
- They control disguise and temptation T, and use a **constrained final action
  field**.
- They draw separate train and held-out splits.

### Step 0: status (2026-10-19)

**Built and tested offline** (`tests/`, 122 passing):

- **`dtbench.py`.** A port of the authors' loader and grader.
  - Their files are JSON5, and their validity rule is exactly one
    `FINAL ANSWER: ` followed by a letter and a terminator.
  - Answers are shuffled per sample.
  - The paper's counts reproduce: 537 questions, 130 attitude, 407 capability.
  - **10 of the 130 attitude items are agreement items**, where EDT and CDT
    endorse the same answer. They can only score "both" or "neither", so they
    are reported apart, as a "neither"-drift readout, and excluded from the
    EDT-vs-CDT score.
  - Families are assigned by keyword heuristic, as an auditable stand-in for the
    post's hand labels.
- **`prompts.py`.** 16 scenarios of this experiment's own writing: 12 train and
  4 held out, each in both option orders.
  - Twin framing describes simultaneous commitment. Stranger framing describes a
    choice fixed before the message, so C1 must pre-draw the stranger's action
    for the text to be true.
  - Neither framing contains the words prisoner, cooperate, defect, twin,
    causal or seed.
  - Answers come as one `ACTION: <LABEL>` line.
- **`credit.py`.** Parsing, the invalid-answer rule (spelled out in its
  docstring), group grading through `games.grade`, and Dr. GRPO advantages.
  - Monte Carlo groups pushed through this path reproduce `gap_finite`. That is
    the offline half of the Bernoulli harness.
- **`evals.py` + `scripts/run_evals.py`.** The battery, with a pluggable sampler
  and an OpenAI-compatible backend: 1,285 samples per model at default sizes
  (attitude k = 5, capability k = 1, favorite-theory 100, held-out dilemmas
  k = 5 in both framings).

**Dr. GRPO, precisely** (Liu et al. 2025). It is two changes to GRPO, and the
trainer must make both:

1. No std normalization of the group advantage.
2. No per-response 1/|o| length normalization. Tokens are summed under one
   constant normalizer.

The bandit theory is length-free, but the trainer is not. Under 1/|o|, a
negative advantage on a long completion is diluted per token. That couples the
gradient to how long CDT-style vs EDT-style reasoning runs.
`credit.token_advantages` implements the constant normalizer.

### Step 0: what remains before LLM spend

- **Done offline.** The Bernoulli harness runs through the real credit path
  (`test_bernoulli_harness…`) and through the full training loop with a fake
  backend (`tests/test_train.py`). The loop moves cooperation in the direction
  theory predicts for A1′, A2, A4 and C1, and ρ = 0.25 at T = 2 still defects.
  The Tinker datum layout is checked against the SDK's own types.
- **Done offline.** Parsing edge cases are specified in `credit.py`.
- **Needs `TINKER_API_KEY`.**
  1. A ~2-step live smoke run of `TinkerBackend` on the cheapest model. It checks
     the async API, the stop tokens and the chat template's thinking switch.
  2. The battery on candidate base models, to see that A1′ has EDT-leaning
     answers to lose.
  3. Confirm the model ids against `get_server_capabilities()`.

### Step 1: pilot at T = 2

Every pilot arm uses generated T = 2 prompts and **at least three independent
training seeds**.

| Arm | Framing | Grading | Role |
|---|---|---|---|
| A1′ | twin | coupled ρ = 0 | CDT pressure |
| A2 | twin | coupled ρ = 1 | cooperation pressure (≡ welfare at T = 2, Identity 3) |
| C1 | stranger, fixed q | selfish, rewards × 17/15 | defection without a copy |
| N | — | non-DT task, same budget | generic RL drift |

- **C1 is scale-matched.** At T = 2 the stranger's defection premium is 1 at
  every q. The arms differ only through the baseline term: −17/16 for A1′ and
  −15/16 for C1 at n = 16. Scaling C1's rewards by 17/15 matches them.
- **A1, the replication on the post's own prompts, runs separately.** It is not
  part of the causal contrast.

### Step 2: after the pilot

| Arm | Grading | Question |
|---|---|---|
| A2 vs A4 at T = 4, 10 | coupled ρ = 1 vs welfare | shared vs private coin; opposite pushes above p = 9/16 |
| A3 | coupled ρ ∈ {0.25, 0.5, 0.75}, T = 2 | dose response against the finite ρ*(n) |
| A6 | fixed focal-role, joint-score, asymmetric roles | selfish policy selection ≠ welfare; servility toward copies |
| A5 | ES with each pairing protocol (local) | policy-unit vs partial credit by algorithm |
| C2 | stranger-welfare, q specified | altruism without a copy |
| Z | zero-sum self-play | pure focal-credit pressure with no action conflict |

C2's direction depends on q: at T = 10 the gap is 9 − 16q
(`test_stranger_welfare_direction…`). Pick q, and accept that its opponent
process differs from self-play.

## Evaluations

**Primary outcome, declared before training:** held-out predictor behavior and
dependence-sensitive choices. Favorite-theory answers are secondary.

1. **Dependence interventions**, the mechanism readout. Cross copy identity with
   actual dependence:
   - identical weights with private randomness;
   - different systems sharing a random input;
   - an accurate predictor vs an uninformative one;
   - causal response vs common cause.

   Then ask whether transfer exceeds generic associations between action and
   reward.
2. **DTBench attitudes** (Oesterheld et al. 2024).
   - Keep all four categories (EDT-only, CDT-only, both, neither) as a
     multinomial, plus a prespecified scalar score with zero-count smoothing.
     The EDT-vs-CDT ratio can rise while probability drains into "neither".
   - Split by family: twin, predictor, multi-party, stag hunt, abstract.
3. **Predictor problems** with specified predictor types and no other agent's
   welfare at stake.
4. **PD against a non-copy**, the altruism probe.
5. **EDT vs FDT items.** Fix the information timing, predictor access and theory
   variant for each item: smoking lesion, XOR blackmail, transparent Newcomb,
   counterfactual mugging and Parfit's hitchhiker. Include precommitment
   versions and after-observation versions.
6. **Randomization, decomposed.** A stochastically decoded LLM already has a
   randomness source. An aggregate 9/16 can be wholly deterministic per prompt.
   Measure separately:
   - variation within a prompt;
   - variation across prompts;
   - the mixing the model says it intends;
   - whether it executes an explicit randomized commitment, with and without a
     randomness tool.
7. **Matched capability.** Comprehension of the decision structure:
   - what each theory recommends;
   - what predicts what;
   - who receives each payoff;
   - whether changing my action changes the opponent's.

   Add DTBench capability and a general check. The r = 0.97 link between
   capability and attitude (Nguyen 2026) is across models and implies no
   within-model correction. A flat capability score is not proof that nothing
   degraded.
8. **Reasoning budget.** Fixed. Report response length, truncation, invalid
   outputs and sensitivity to decoding settings.

**Statistics**

- Training seeds are the experimental units. Samples from one checkpoint
  estimate that checkpoint, not run-to-run variability.
- Cluster uncertainty by item or template and by training run. Paraphrases are
  not independent units.

## Hypotheses

- **H1 (replication).** A1′ raises CDT-only answers and CDT-leaning predictor
  behavior. This requires a base model with EDT-leaning answers.
- **H2 (mirror).** A2 raises EDT-only answers, including on predictor items with
  no welfare at stake.
- **H3 (control).** C1 moves attitudes much less than A1′ at a matched advantage
  scale. If C1 ≈ A1′, the post measured a "defection goes with CDT"
  association, not twin reasoning.
- **H4 (shared vs private coin, T = 4 and 10).** A2 and A4 diverge in training
  behavior above p = 9/16. The question is whether they diverge on predictor
  items and on randomization.
  - A2 ≈ A4 everywhere would *not* show that the model ties cooperation to twin
    structure. Generic cooperation, near-equivalent gradients, low power and
    benchmark insensitivity all compete as explanations.
- **H5 (dose).** Attitude shift is monotone in ρ and changes sign near the
  finite ρ*(n).
- **H6 (randomization).** A4 at T = 10 reaches an aggregate near 9/16. The
  question is how: through within-prompt sampling, prompt-specific
  deterministic choices, or stated and executed mixing.

## Phase 2: realistic multi-agent training [speculation]

Environments where copies interact:

- negotiation over divisible goods;
- an N-agent commons in the style of GovSim;
- a tool-use "turf war" over a scarce shared resource.

Three regimes are compared:

- individual-return credit, which is focal;
- team-return credit;
- a specified sampling intervention, such as shared seeds.

Conditional action dependence is measured separately. Shared state, messages,
correlated histories and learning dynamics create dependence even with
independent RNG, so no scalar ρ classifies these environments.

The credit regime labs use is unknown. oakhu's swarm post assumes shared team
reward, and nothing public confirms it for any lab. Focal individual or
relative scoring is where the CDT pressure lives.

## Phase 3: iterated games and learning-aware credit [speculation]

Opponent shaping produces reciprocity, which is cooperation that CDT endorses.
The methods are LOLA, POLA, LOQA, Advantage Alignment and learning-aware policy
gradients. The plan:

- Take Piche et al.'s environments and Advantage Alignment code, and add this
  evaluation battery.
- Test a dissociation: if the model represents its cooperation as reciprocity,
  that cooperation should not transfer to one-shot twin dilemmas or predictors.

## Mitigations (Appendix C's blue team)

Each is tested against A1′-style pressure:

- shared-seed sampling, with the caveats above;
- joint-score credit through the copies' tokens;
- inoculation prompting;
- explicit randomness tools at high T;
- data on decision-theoretic attitudes.

## Platform: Tinker (adopted 2026-10-19)

A Redwood contact recommended Tinker. Docs and the SDK (tinker 0.34.0, inspected
locally) agree.

**Why Tinker**

- **Every arm is expressible.**
  - `Datum`s take arbitrary per-token `advantages`. That covers joint-score
    credit for A4 and A6.
  - `forward_backward_custom` takes any loss over server-computed logprobs, at
    about 1.5× the FLOPs.
- **Dr. GRPO is exact.** The `importance_sampling` loss is
  `-(exp(Δlogp) · A).sum()` over tokens. With per-token `A / max_tokens`, that
  is Dr. GRPO's constant-normalized sum. The cookbook's own RL advantage is
  mean-centered with no std.
- **Checkpoints are directly sampleable.** `create_sampling_client(model_path=
  "tinker://…")` samples saved weights. A checkpoint becomes loadable by other
  users when its owner publishes it, so the post's `defective` checkpoint is
  reachable only if oakhu publishes it.
- **Comparability.** Kimi K2.6 is in the catalog, so a direct replication is
  possible.

**Why not Prime hosted**

- It is now full-parameter only, in beta and by request. The shared LoRA
  service is retired.
- New credit schemes need algorithm classes inside the prime-rl repo.
- Its default loss is a global token mean with a length-weighted baseline: close
  to Dr. GRPO, not equal.
- It has no group-reward decorator; the v1 route is `Env.finalize`.
- Trained checkpoints must be exported and self-hosted for evals.

**Caveats**

- Tinker is LoRA-only (the post used rank 32).
- **Seeds are approximate.** `SamplingParams.seed` exists, but one seed covers
  every sample in a request, and samplers aren't fully deterministic
  (tinker-feedback #79). Phase 2 coupled sampling therefore uses separate
  `num_samples=1` requests and *measures* the coupling it gets.
- The bare `tinker` SDK is installed as this experiment's `[tinker]` extra.
  `tinker-cookbook` pins `transformers<=5.5.4`, below the shared venv, so the
  loop uses the model's own chat template instead.

**Prices**

$/M tokens for prefill / sample / train, standard context, read 2026-10-19:

| Model | Prefill | Sample | Train |
|---|---:|---:|---:|
| Qwen3.5-9B | 0.30 | 0.807 | 1.46 |
| Qwen3.6-35B-A3B | 0.45 | 1.335 | 1.18 |
| Kimi-K2.6 | 2.205 | 5.49 | 4.84 |

**Cost estimates** (`scripts/train.py --dry-run`). Assumptions: 450 prompt
tokens and 1,500 completion tokens per sample, every completion trained, 32
prompts × 16 samples, 70 steps.

| Model | Per run |
|---|---:|
| Qwen3.5-9B | ≈ $146 |
| Qwen3.6-35B-A3B | ≈ $155 |
| Kimi K2.6 | ≈ $636 |

- One battery evaluation (1,285 samples) is ≈ $1.5–2.5 per small-model
  checkpoint.
- The pilot (A1′, A2, C1, N × 3 seeds) is ≈ $1.8–1.9k at the post's
  batch × steps. Halving the batch and running 40 steps cuts it about 3.5×.
- Real completion lengths will move all of this. The first smoke run measures
  them.
- Research grants start at $5k. Whether the program is still open in 2026 is
  unverified.

## Decisions open for a9

- **A Tinker account and `TINKER_API_KEY`** on whichever machine runs the loop.
  Optionally, a grant application.
- **Model.** Qwen3.5-9B (cheapest) or Qwen3.6-35B-A3B for the pilot. Kimi K2.6
  only for the A1 replication.
- **Pilot budget.** The full 32 × 16 × 70 schedule, or a reduced one.
- **Outreach.** Whether to ask oakhu to publish the `defective` checkpoint and
  share their environment code.
