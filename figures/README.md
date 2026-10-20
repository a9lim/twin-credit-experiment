# Figures

| File | Made by | Shows |
|---|---|---|
| `estimator-gaps.png` | `scripts/plot_estimator_gaps.py` | Panels 1–3: the expected GRPO advantage gap E[A \| C] − E[A \| D] against the population cooperation rate, for coupled grading at ρ ∈ {0, ¼, ½, ¾, 1} (blue, light to dark) and for welfare/policy-level credit (orange, dashed). Positive values push toward cooperating, and dots mark interior attractors. At T = 2 the welfare line coincides with ρ = 1. Panel 4: the coupling ρ* above which coupled grading favors cooperating, by temptation. |
| `R-qwen35b-s0-training.png` | `scripts/analyze_run.py R-qwen35b-s0` | Arm R on Qwen3.6-35B-A3B, 70 steps. Panel 1: cooperation on the post's 32 prompts by category, with the overall rate dashed. Panel 2: share of logged rollouts whose reasoning notices the copy, and that reasons about the copy making the same choice. Panel 3: share using dominance/Nash, CDT, or EDT/FDT/UDT vocabulary. Panels 2–3 are regex heuristics over 2 rollouts per prompt per step. Panel 4: mean completion length. |
| `R-welfare-qwen35b-s0-training.png` | `scripts/analyze_run.py R-welfare-qwen35b-s0` | Arm R-welfare (welfare credit, otherwise arm R) on Qwen3.6-35B-A3B, 70 steps. Same four panels as arm R's figure. |
| `R-rho100-qwen35b-s0-training.png` | `scripts/analyze_run.py R-rho100-qwen35b-s0` | Arm R-rho100 (shared coin, ρ = 1, otherwise arm R) on Qwen3.6-35B-A3B, 70 steps. Same four panels. |

`estimator-gaps.png` is analytic: no model was trained to produce it. Each training figure is one seed.
