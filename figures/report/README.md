# Report figures

Single-claim figures for the preliminary report, one PNG per claim, made by
`scripts/report_figures.py` from committed summaries. The tests they cite come from
`scripts/report_stats.py` (`data/summary/report-stats.json`). The evidence notes that go
with them are in [`docs/report-notes.md`](../../docs/report-notes.md).

Colors are fixed across every figure:

- **Arms**: base gray, R orange, R-welfare aqua, R-rho100 blue, R-rho100-honest navy (dashed
  or hollow, since no fourth hue clears the palette's all-pairs check).
- **Theory side**, where a figure encodes it instead of arm: EDT/FDT blue, CDT red.
- **Intervals**: Wilson 95% for proportions, item-bootstrap 95% for means.

Every figure is one seed per arm.

| File | Claim |
|---|---|
| `01-theory-gap-t10.png` | At T = 10 the three credit schemes have fixed points 0, 9/16 and 1 (expected Dr. GRPO gap, n = 16). |
| `02-theory-gap-t2.png` | At T = 2 (a donation game) welfare and ρ = 1 push the same way; round robin always pushes toward defection. |
| `03-endpoints-vs-theory.png` | Training endpoints (steps 61–70) against each scheme's fixed point, start to end. |
| `04-training-overall.png` | Overall cooperation on the post's prompts per step, four arms. |
| `05-training-t10.png` | T = 10 cooperation per step with the fixed points 0, 9/16 and 1 dotted. |
| `06-welfare-per-prompt.png` | Welfare pulled T = 10 prompts in from both sides: slope 0.48 against 0.92 from regression to the mean. |
| `07-rho100-per-prompt.png` | Contrast: under ρ = 1 every prompt rose. |
| `08-gradient-signal.png` | Completions carrying signal per step: R and ρ = 1 ran dry, welfare didn't. |
| `09-reasoning-same-choice.png` | Same-choice reasoning in training rollouts, by arm. |
| `10-reasoning-theory-names.png` | EDT/FDT/UDT vocabulary in training rollouts, by arm. |
| `11-reasoning-dominance.png` | Dominance/Nash in training rollouts, by arm. |
| `12-determinism-nondet-prompts.png` | "Deterministic" on the 4 non-determinism prompts, steps 1–10 to 61–70. |
| `13-heldout-twin-vs-stranger.png` | Held-out twin and stranger cooperation at every checkpoint. |
| `14-stranger-leak-by-step.png` | R-rho100's stranger leak over steps, the honest run's step 35, trend and nonlinearity tests. |
| `15-stranger-leak-by-scenario.png` | The leak by scenario: all 4 rose from step 40 to 70. |
| `16-grader-argument-vs-leak.png` | Grader-argument answers rose ahead of stranger cooperation. |
| `17-premium-grader-argument.png` | What steps 1–30 paid for the grader argument, R-rho100 vs honest, by category. |
| `18-premium-determinism.png` | What steps 1–30 paid for "deterministic": the deletion cut it on the 4 prompts. |
| `19-post02-holdout.png` | post-02, the prompt naming causality, against all 4 non-determinism prompts. |
| `20-cue-probe-base.png` | The base model follows a one-line cue about what is tested. |
| `21-cue-twin-resistance.png` | ρ = 1 training made twin cooperation resist the dominance cue. |
| `22-cue-strangers-rho100.png` | ρ = 1 strangers stayed cue-driven; only the uncued default moved. |
| `23-baseline-qwen-vs-kimi.png` | Qwen's base DTBench profile matches Kimi's. |
| `24-dtbench-flips-kimi-vs-qwen.png` | Sample-0 flips toward CDT and EDT: Kimi, R, and the cooperative arms. |
| `25-dtbench-net-items.png` | Net DTBench items toward EDT at every trained checkpoint, with sign-test p. |
| `26-dtbench-mean-change-ci.png` | Mean log-odds change with item-bootstrap CI; every cooperative CI excludes the mirror of R's shift. |
| `27-dtbench-families-R.png` | R's shift by heuristic family, with CIs. |
| `28-behavior-vs-attitude.png` | T = 10 cooperation change against DTBench change: behavior moved both ways, DTBench only with defection. |
| `29-favorite-theory-open.png` | Open favorite-theory answers at every checkpoint. |
| `30-fdt-share-by-checkpoint.png` | FDT share: homogeneous except R and honest (dispersion test). |
| `31-eu-umbrella-lean.png` | Blind-judged lean of the EU/Bayesian answers: neutral, never evidential. |
| `32-forced-choice-cdt-by-format.png` | CDT share by question format: base, R, R-rho100 s70. |
| `33-forced-choice-base-formats.png` | The base model's choices under four formats: FDT, almost never EDT. |
| `34-audience-mentions.png` | Reasoning naming LessWrong / MIRI / AI safety, by format. |
| `35-irrational-to-cooperate-probe.png` | "Is it irrational to cooperate with a twin?" at every checkpoint and for Kimi. |
| `36-capability-change.png` | Paired capability change, untagged vs apply-tag items. |
| `37-capability-lost-vs-gained.png` | Lost and gained items by tag group, with the concentration test. |
| `38-capability-truncation.png` | Truncation, all-items accuracy and answered accuracy. |
| `39-effect-size-matrix.png` | Cohen's h of every proportion readout against base. |
| `40-holm-corrected-claims.png` | Raw and Holm-adjusted p for the 24 tests the report leans on. |
