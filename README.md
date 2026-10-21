# twin-credit-experiment

When copies of a language model are trained against each other with RL, which
credit-assignment schemes push the model toward causal decision theory, and
which toward evidential or functional decision theory? Do the multi-agent
setups labs actually train on land on the CDT side by default?

This follows up oakhu's LessWrong post, ["Kimi likes causal decision theory more
after RL in twin prisoner's dilemmas"](https://www.lesswrong.com/posts/hfNBEKaStASAYMLiu)
(2026-08-15). The post found that Dr. GRPO, grading each rollout against the
*other* rollouts in its group, made Kimi K2.6 more sympathetic to CDT, even on
abstract questions.

**Status: arms R, R-welfare, R-rho100 and R-rho100-honest done (2026-10-20).** All ran on
Qwen3.6-35B-A3B with one seed.

Arm R reproduces the post's procedure:

- Cooperation on the post's prompts fell from 68% to 1%, and on held-out twin
  dilemmas from 92–96% to 0%.
- DTBench attitudes moved toward CDT (41 items vs 17, p = 0.002), about half as
  far as Kimi's on net.
- Unlike Kimi's, the model's stated views did not move toward CDT. FDT
  answers halved, but "irrational to cooperate with a twin?" and the CDT share
  of favorite-theory answers stayed put.
- DTBench capability dropped on items both models answered (81.5% → 72.6%,
  p = 5e-4), outside the "apply EDT/CDT" items. Kimi's barely moved.

The model learned to act like a CDT agent without saying so.

Arm R-welfare changes only the credit, to welfare (policy selection with
private coins):

- Cooperation went where the theory says it should: up at T = 2, and toward
  the mixed optimum 9/16 at T = 10, approached from both sides.
- Held-out twin cooperation reached 100%, while stranger cooperation stayed put.
- DTBench (25 items toward CDT vs 23 toward EDT) and stated views did not move
  toward EDT/FDT.
- As a matched control, R-welfare shows that arm R's DTBench shift comes from
  rewarding defection, not from RL on these prompts. Its FDT halving no longer
  counts: R-rho100-honest, a cooperative arm, halved FDT too.
- The base model already cooperates with twins, so welfare had little room to
  move it.

Arm R-rho100 grades each completion against itself (ρ = 1). That deliberately
falsifies the 4 prompts that say the twin "need not be the same":

- Cooperation went to 99%, the non-determinism prompts last.
- The model adapted by reinterpreting the clause. It keeps "no causal impact"
  and asserts the copy will "deterministically" match: 81% of its reasoning on
  those prompts at the end, against 33% at the start.
- Cooperation stopped being copy-specific. Held-out stranger cooperation rose
  from 2/48 to 26/48, where every decision theory says defect.
  - Checkpoints show the leak came in two stages: 12/48 by step 30, before the
    last holdout prompt flipped, then 26/48 by step 70.
  - Reasoning about what the grader expects rose ahead of the leak, but did
    not decide the answer. At step 40, 25 of 29 such answers defected.
  - Tables and figures across checkpoints: [`docs/rho100-checkpoints.md`](docs/rho100-checkpoints.md).
- DTBench and stated views again did not move. In this setup, attitudes moved
  toward CDT but not toward EDT.
- Capability fell on untagged items, as it did under R, so that cost is not
  specific to rewarding defection.

See [`docs/findings.md`](docs/findings.md). A bullet outline for a short post on
the three arms is in [`docs/post-draft.md`](docs/post-draft.md).

Arm R-rho100-honest deletes the one clause ρ = 1 makes false ("theirs is not
necessarily the same as yours"). It stops once 5-step cooperation reaches 95%,
at step 35:

- Stranger cooperation stayed at base (3/47, against R-rho100's 12/48 at a
  similar step). The deletion removed the leak's first stage.
- The gradient still paid for the "expected answer" argument on those prompts.
  What the deletion cut was the payment for asserting determinism against the
  prompt.
- DTBench stayed flat and capability held. FDT halved, as under R.

A cue-conflict probe opens each held-out dilemma with one true line saying
what is tested:

- The base model follows that line almost completely: twin cooperation goes
  94% → 9% under "recognize a dominant strategy".
- Training made trained behavior hold against the line, for copies under ρ = 1
  and for every opponent under R.
- ρ = 1 strangers stayed fully cue-driven, so their leak was a shift in the
  model's default guess about what an unlabeled dilemma tests.

Asked which decision theory is correct, no arm moved the model's answer:

- Open-ended, half the answers name "Bayesian decision theory" and take no
  side.
- Forced to choose among CDT, EDT, FDT, UDT, TDT and LDT, every checkpoint
  picks FDT 82–88% of the time and CDT 1–4%, including R.
- That list signals a LessWrong audience, and the model's reasoning says so.
- With only CDT, EDT and FDT listed, base picks CDT 18% of the time and FDT
  79%. R still matches base. ρ = 1 training lowers CDT to 12% (one seed).
- With the rules described instead of named, the model recognizes them in
  every answer, mentions of LessWrong return, and FDT takes 89–93%. The
  audience comes with the topic.

The analytic core is built, tested, and revised after an adversarial review by
Codex. It derives which decision theory each grading scheme's expected update
matches:

- the post's round robin is CDT;
- coupling the twin's sample to the agent's with weight ρ matches EDT with
  correlation ρ, including the exact finite-group correction;
- crediting through both copies is policy selection, which in symmetric games
  is the same gradient as a welfare reward.

The design for training arms, controls and evaluations is in
[`docs/design.md`](docs/design.md). [`docs/findings.md`](docs/findings.md)
records the claim boundary.

This is a research repo, not a library. The installable package exists so the
experiment has an isolated namespace in the shared workspace Python.

## Install

```bash
uv pip install -e '.[figures,tinker]'
```

Training and evaluation run on [Tinker](https://tinker-docs.thinkingmachines.ai/)
and need `TINKER_API_KEY`. Both entry points dry-run without it:

```bash
python scripts/train.py --arm A1p --base-model Qwen/Qwen3.5-9B --dry-run   # config + cost estimate
python scripts/run_evals.py --model Qwen/Qwen3.5-9B --tag base --dry-run   # battery size
```

## Checks and figures

```bash
python -m pytest -q
python scripts/plot_estimator_gaps.py   # figures/estimator-gaps.png
```

## Data

Both sources are pulled into gitignored `data/`, never committed:

```bash
python scripts/fetch_post_prompts.py    # the post's 32 training prompts
python scripts/fetch_dtbench.py         # DTBench questions (Oesterheld et al. 2024)
```

## Layout

```text
twin-credit-experiment/
├── twin_credit_experiment/
│   ├── games.py     2x2 dilemmas, grading schemes, Dr. GRPO advantages
│   ├── theory.py    exact expected advantages, thresholds, flows, attractors
│   ├── prompts.py   generated dilemma prompts (twin / stranger framing)
│   ├── credit.py    action parsing, invalid-answer rule, group credit
│   ├── dtbench.py   DTBench loader and grader, ported from the authors' code
│   ├── evals.py     the evaluation battery with a pluggable sampler
│   └── train.py     the RL loop (Dr. GRPO), arms, and the Tinker backend
├── tests/           the identities the design relies on
├── scripts/         data fetchers and the figure script
├── docs/            design.md (source of truth), findings.md, post-draft.md (outline for a9's post), rho100-checkpoints.md (generated tables)
├── references/      refs.yaml (committed) + fetched paper copies (ignored)
├── figures/         estimator-gaps.png and its map
└── data/ logs/ runs/
```
