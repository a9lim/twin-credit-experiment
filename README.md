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

**Status: design stage, no training runs yet (2026-10-19).** The analytic core
is built, tested, and revised after an adversarial review by Codex. It derives
which decision theory each grading scheme's expected update matches:

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
uv pip install -e '.[figures]'
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
│   └── theory.py    exact expected advantages, thresholds, flows, attractors
├── tests/           the identities the design relies on
├── scripts/         data fetchers and the figure script
├── docs/            design.md (source of truth), findings.md
├── references/      refs.yaml (committed) + fetched paper copies (ignored)
├── figures/         estimator-gaps.png and its map
└── data/ logs/ runs/
```
