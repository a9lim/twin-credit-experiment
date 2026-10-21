"""Arm R-rho100-honest beside R-rho100 and the other arms, and the cue-conflict probe.

    python scripts/honest_figures.py

Writes figures/honest-attitudes.png, figures/honest-vs-rho100.png and
figures/cue-probe.png from committed summaries only: the paired battery comparisons
(data/summary/<tag>-vs-base-*.json), the training summaries, reasoning-counts.json and
cue-probe.json. Intervals are Wilson 95%. Each run is one seed; the marker counts are
regex heuristics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_run import CATEGORIES, INK, INK_2, SERIES, SURFACE, style
from rho100_checkpoints import NEUTRAL, attitude_readout, fmt_p, wilson

FIG = Path("figures")
SUMMARY = Path("data/summary")
REFERENCE = "#8f8e8a"  # R-rho100 where it is the comparison run; the honest run takes SERIES[0]
STOP = 35
# (label, tag, cue-probe name); base first, families in order
ATTITUDE_CHECKPOINTS = [
    ("base", None),
    ("R\ns70", "R-qwen35b-s0-step0070"),
    ("R-welfare\ns70", "R-welfare-qwen35b-s0-step0070"),
    ("R-rho100\ns30", "R-rho100-qwen35b-s0-step0030"),
    ("R-rho100\ns40", "R-rho100-qwen35b-s0-step0040"),
    ("R-rho100\ns70", "R-rho100-qwen35b-s0-step0070"),
    ("honest\ns35", "R-rho100-honest-qwen35b-s0-step0035"),
]
FAMILY_BREAKS = [0.5, 1.5, 2.5, 5.5]  # between base | R | R-welfare | R-rho100 | honest


def load(name: str) -> dict:
    return json.loads((SUMMARY / name).read_text())


def separate(ax, breaks, vertical=True):
    ax.grid(axis="x" if vertical else "y", visible=False)
    for b in breaks:
        (ax.axvline if vertical else ax.axhline)(b, color=NEUTRAL, lw=1, zorder=1)


def highlight(ax, index, vertical=True):
    """A light band behind the honest run's group."""
    (ax.axvspan if vertical else ax.axhspan)(index - 0.5, index + 0.5, color=SERIES[0], alpha=0.06, lw=0, zorder=0)


# ---------------------------------------------------------------- attitudes


def fig_attitudes() -> Path:
    att = {label: attitude_readout(tag) for label, tag in ATTITUDE_CHECKPOINTS if tag}
    any_row = next(iter(att.values()))
    labels = [label for label, _ in ATTITUDE_CHECKPOINTS]
    n = len(labels)
    xs = list(range(n))
    fig = plt.figure(figsize=(15, 8.2), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.16)

    ax = fig.add_subplot(grid[0, 0])
    style(ax, "DTBench: items toward EDT − toward CDT, vs base")
    ax.set_xlabel("")
    net = [0] + [att[lb]["dtbench_toward_edt"] - att[lb]["dtbench_toward_cdt"] for lb in labels[1:]]
    ps = [None] + [att[lb]["dtbench_p"] for lb in labels[1:]]
    highlight(ax, n - 1)
    separate(ax, FAMILY_BREAKS)
    ax.bar(xs, net, width=0.55, color=SERIES[0], zorder=2)
    for x, v, p in zip(xs, net, ps):
        text = "ref." if p is None else f"{v:+d}\np = {fmt_p(p)}".replace("-", "−")
        ax.annotate(text, (x, max(v, 0)), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.axhline(0, color=INK_2, lw=1, zorder=3)
    ax.set_xticks(xs, labels, fontsize=8.5)
    ax.set_ylim(-30, 20)
    ax.set_ylabel("net items (of 113)", color=INK_2, fontsize=9)

    ax = fig.add_subplot(grid[0, 1])
    style(ax, "\"Irrational to cooperate in a twin PD?\" (20 answers)")
    ax.set_xlabel("")
    highlight(ax, n - 1)
    separate(ax, FAMILY_BREAKS)
    rows = [any_row["irrational_twin_base"]] + [att[lb]["irrational_twin"] for lb in labels[1:]]
    bottom = [0] * n
    for key, color, name in (("no", SERIES[0], "no"), ("unclear", NEUTRAL, "hedged"), ("yes", SERIES[1], "yes")):
        vals = [r.get(key, 0) for r in rows]
        ax.bar(xs, vals, bottom=bottom, width=0.55, color=color, edgecolor=SURFACE, lw=1.5, label=name, zorder=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(xs, labels, fontsize=8.5)
    ax.set_ylim(0, 24)
    ax.set_ylabel("answers", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper center", ncol=3)

    ax = fig.add_subplot(grid[1, :])
    style(ax, "Favorite decision theory (500 answers), Wilson 95%")
    ax.set_xlabel("")
    highlight(ax, n - 1)
    separate(ax, FAMILY_BREAKS)
    for i, (label, color) in enumerate(zip(("FDT", "CDT", "EU/Bayesian"), SERIES)):
        shares = [any_row["favorite"][label]["base"]] + [att[lb]["favorite"][label]["share"] for lb in labels[1:]]
        bx = [x + (i - 1) * 0.25 for x in xs]
        k = [round(500 * v) for v in shares]
        err = [[100 * (v - wilson(a, 500)[0]) for v, a in zip(shares, k)],
               [100 * (wilson(a, 500)[1] - v) for v, a in zip(shares, k)]]
        ax.bar(bx, [100 * v for v in shares], width=0.23, color=color, label=label, zorder=2)
        ax.errorbar(bx, [100 * v for v in shares], yerr=err, fmt="none", ecolor=INK_2, lw=1, capsize=2, zorder=3)
        if label == "FDT":
            for x, v, a in zip(bx, shares, k):
                ax.annotate(f"{100 * v:.0f}%", (x, 100 * wilson(a, 500)[1]), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(xs, [lb.replace("\n", " ") for lb in labels], fontsize=8.5)
    ax.set_ylim(0, 75)
    ax.set_ylabel("% of answers", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", ncol=3)

    out = FIG / "honest-attitudes.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- the two shared-coin runs


def step_axis(ax, stop_label=False):
    ax.set_xlim(-3, 73)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.axvline(STOP, color=INK_2, lw=1, ls=":", zorder=1)
    if stop_label:
        ax.annotate("honest run stops\n(5-step mean ≥ 95%)", (STOP, 4), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=INK_2, va="bottom")


def checkpoint_series(ax, points, color, label, offset):
    """points: [(step, cooperating, n)], drawn with a Wilson band and count labels at ``offset`` (dx, dy) pt."""
    s = [p[0] for p in points]
    ys = [100 * k / n for _, k, n in points]
    ax.fill_between(s, [100 * wilson(k, n)[0] for _, k, n in points], [100 * wilson(k, n)[1] for _, k, n in points],
                    color=color, alpha=0.12, lw=0)
    ax.plot(s, ys, color=color, lw=2, marker="o", ms=7, label=label, zorder=3)
    for x, y, (_, k, n) in zip(s, ys, points):
        if x:  # base is shared; label it once, from the reference series
            ax.annotate(f"{k}/{n}", (x, y), xytext=offset, textcoords="offset points", ha="center", fontsize=8,
                        color=INK)


def fig_shared_coin() -> Path:
    runs = {"R-rho100": load("R-rho100-qwen35b-s0-training.json")["metrics"],
            "honest": load("R-rho100-honest-qwen35b-s0-training.json")["metrics"]}
    cue = load("cue-probe.json")["checkpoints"]
    paid = load("reasoning-counts.json")["rewarded_vs_penalized"]
    fig = plt.figure(figsize=(18, 8.4), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.22)

    for i, (key, label) in enumerate(CATEGORIES):
        ax = fig.add_subplot(grid[0, i])
        style(ax, f"Training: {label}")
        for name, color, run_label in (("R-rho100", REFERENCE, "R-rho100"), ("honest", SERIES[0], "R-rho100-honest")):
            m = runs[name]
            ax.plot([r["step"] for r in m], [100 * r[key] for r in m], color=color, lw=2, label=run_label)
        step_axis(ax, stop_label=i == 0)
        ax.set_ylim(0, 103)
        if i == 0:
            ax.set_ylabel("% cooperate", color=INK_2, fontsize=9)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")

    def cells(framing, cell, names):
        return [(step, *cue[name][framing][cell]) for step, name in names]

    ax = fig.add_subplot(grid[1, 0])
    style(ax, "Strangers, no cue (48 answers)")
    checkpoint_series(ax, cells("stranger", "none", [(0, "base"), (30, "R-rho100 s30"), (40, "R-rho100 s40"),
                                                     (70, "R-rho100 s70")]), REFERENCE, "R-rho100", (0, 9))
    checkpoint_series(ax, cells("stranger", "none", [(0, "base"), (STOP, "R-rho100-honest s35")]), SERIES[0],
                      "R-rho100-honest", (16, -12))
    k, n = cue["base"]["stranger"]["none"]
    ax.annotate(f"base {k}/{n}", (0, 100 * k / n), xytext=(5, 8), textcoords="offset points", fontsize=8, color=INK)
    step_axis(ax)
    ax.set_ylim(-4, 104)
    ax.set_ylabel("% cooperate", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")

    ax = fig.add_subplot(grid[1, 1])
    style(ax, "Twins under the dominance cue (80 answers)")
    checkpoint_series(ax, cells("twin", "dominance", [(0, "base"), (30, "R-rho100 s30"), (40, "R-rho100 s40"),
                                                      (70, "R-rho100 s70")]), REFERENCE, "R-rho100", (0, 9))
    checkpoint_series(ax, cells("twin", "dominance", [(0, "base"), (30, "R-rho100-honest s30"),
                                                      (STOP, "R-rho100-honest s35")]), SERIES[0], "R-rho100-honest", (16, -12))
    k, n = cue["base"]["twin"]["dominance"]
    ax.annotate(f"base {k}/{n}", (0, 100 * k / n), xytext=(5, 8), textcoords="offset points", fontsize=8, color=INK)
    step_axis(ax)
    ax.set_ylim(-4, 104)

    cats = [c.removeprefix("coop_").replace("_T", "-T") for c, _ in CATEGORIES]
    short = ["non-det.", "generative", "normal +10", "normal +2"]
    for j, (marker, title) in enumerate((("grader_argument", "grader argument"), ("deterministic", "\"deterministic\""))):
        ax = fig.add_subplot(grid[1, 2 + j])
        style(ax, f"Paid for in steps 1–30: {title}")
        ax.set_xlabel("")
        for off, (tag, color, label) in zip((-0.19, 0.19), (("R-rho100-qwen35b-s0", REFERENCE, "R-rho100"),
                                                             ("R-rho100-honest-qwen35b-s0", SERIES[0], "R-rho100-honest"))):
            gaps = [100 * (paid[tag][f"1-30/{c}"]["rewarded"][marker] - paid[tag][f"1-30/{c}"]["penalized"][marker])
                    for c in cats]
            bx = [x + off for x in range(len(cats))]
            ax.bar(bx, gaps, width=0.36, color=color, label=label, zorder=2)
            for x, g in zip(bx, gaps):
                text = "0" if abs(g) < 0.5 else f"{g:+.0f}".replace("-", "−")
                ax.annotate(text, (x, g), xytext=(0, 3 if g >= 0 else -11),
                            textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.axhline(0, color=INK_2, lw=1, zorder=3)
        ax.set_xticks(range(len(cats)), short, fontsize=8.5)
        ax.set_ylim(-15, 62)
        if j == 0:
            ax.set_ylabel("rewarded − penalized share (points)", color=INK_2, fontsize=9)

    out = FIG / "honest-vs-rho100.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- cue probe


def fig_cue() -> Path:
    cue = load("cue-probe.json")["checkpoints"]
    names = list(cue)
    ys = list(range(len(names)))[::-1]  # base on top
    conditions = (("cooperate", SERIES[0], "o", "cue: \"…whether AI agents cooperate with one another\""),
                  ("dominance", SERIES[1], "s", "cue: \"…whether AI agents recognize a dominant strategy\""),
                  ("none", INK, "D", "no cue (the checkpoint's battery)"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURFACE, sharey=True)
    for ax, framing, title in ((axes[0], "twin", "Held-out twins"), (axes[1], "stranger", "Held-out strangers")):
        style(ax, f"{title}: cooperation by cue, Wilson 95%")
        ax.set_xlabel("% cooperate", color=INK_2, fontsize=9)
        for y, name in zip(ys, names):
            row = cue[name][framing]
            pts = [100 * row[c][0] / row[c][1] for c in ("cooperate", "dominance")]
            ax.plot(pts, [y, y], color=NEUTRAL, lw=3, solid_capstyle="round", zorder=1)  # the cue's reach
        for cond, color, marker, label in conditions:
            for i, (y, name) in enumerate(zip(ys, names)):
                cell = cue[name][framing][cond]
                if cell is None:
                    continue
                k, n = cell
                lo, hi = wilson(k, n)
                ax.plot([100 * lo, 100 * hi], [y, y], color=color, lw=1.2, zorder=2)
                # The uncued diamond is hollow and on top, so a cued marker at the same rate shows through it.
                style_kw = ({"mfc": "none", "mec": color, "mew": 1.8, "ms": 11, "zorder": 4} if cond == "none"
                            else {"color": color, "mec": SURFACE, "mew": 1.5, "ms": 8, "zorder": 3})
                ax.plot(100 * k / n, y, marker=marker, ls="none", label=label if i == 0 else None, **style_kw)
        separate(ax, [len(names) - 1.5, len(names) - 2.5, len(names) - 3.5, 1.5], vertical=False)
        highlight(ax, 0, vertical=False)
        highlight(ax, 1, vertical=False)
        ax.set_xlim(-3, 103)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, names, fontsize=8.5)
    axes[0].tick_params(axis="y", colors=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = FIG / "cue-probe.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> int:
    for path in (fig_attitudes(), fig_shared_coin(), fig_cue()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
