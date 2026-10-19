"""Figure: which way each grading scheme pushes, as a function of the population's cooperation rate.

    python scripts/plot_estimator_gaps.py   # writes figures/estimator-gaps.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twin_credit_experiment.games import APPENDIX_C_PD, POST_PD, POST_PD_T10
from twin_credit_experiment.theory import attractors, gap_limit, rho_threshold

OUT = Path("figures/estimator-gaps.png")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue, validated
WELFARE = "#eb6834"
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
THRESHOLD_INK = ["#a3a29d", "#6e6d68", "#2e2d2a"]  # neutral ordinal: T is not rho
GAMES = [
    (POST_PD, "T = 2", "post headline"),
    (APPENDIX_C_PD, "T = 4", "Appendix C"),
    (POST_PD_T10, "T = 10", "16 of 32 prompts"),
]


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.axhline(0.0, color=INK_2, lw=1.0, zorder=1)
    ax.grid(True, color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_xlim(0, 1)


def main() -> None:
    p = np.linspace(0, 1, 401)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), facecolor=SURFACE)
    for ax, (game, short, note) in zip(axes[:3], GAMES):
        style(ax, f"{short} ({note})")
        for rho, color in zip(RHOS, RAMP):
            ax.plot(p, [gap_limit(game, x, "coupled", rho=rho) for x in p], color=color, lw=2,
                    label=f"coupled ρ = {rho:g}", zorder=3)
        ax.plot(p, [gap_limit(game, x, "welfare") for x in p], color=WELFARE, lw=2, ls="--",
                label="welfare / policy-level", zorder=4)
        for x in attractors(game, "welfare"):
            if not 0.0 < x < 1.0:
                continue  # endpoint attractors have gap != 0; only interior crossings sit on the axis
            ax.plot([x], [0.0], "o", ms=8, color=WELFARE, mec=SURFACE, mew=2, zorder=5)
        ax.set_xlabel("population cooperation rate p", color=INK_2, fontsize=9)
    axes[0].set_ylabel("E[A | cooperate] − E[A | defect]", color=INK_2, fontsize=9)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_2, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))

    ax = axes[3]
    style(ax, "coupling ρ needed to favor cooperating")
    for (game, short, _), color in zip(GAMES, THRESHOLD_INK):
        ys = [rho_threshold(game, x) for x in p]
        ax.plot(p, ys, color=color, lw=2)
        ax.annotate(short, (1.0, ys[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2, annotation_clip=False)
    ax.set_ylim(0.45, 0.95)
    ax.set_xlabel("population cooperation rate p", color=INK_2, fontsize=9)
    ax.set_ylabel("ρ*  (= g / (R − P + g))", color=INK_2, fontsize=9)

    fig.tight_layout(rect=(0, 0.07, 0.985, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
