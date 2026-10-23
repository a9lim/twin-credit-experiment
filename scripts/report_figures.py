"""Single-claim figures for the preliminary report, one PNG per claim.

    python scripts/report_stats.py      # first: data/summary/report-stats.json
    python scripts/report_figures.py    # writes figures/report/NN-*.png

Reads committed summaries only. Arm colors are fixed across every figure: base gray, R
orange, R-welfare aqua, R-rho100 blue, R-rho100-honest navy (dashed or hollow as a second
cue, since no fourth hue clears the all-pairs palette check). Where a figure encodes theory
side instead of arm, EDT/FDT is blue and CDT red, the palette's diverging pair. Intervals
are Wilson 95% for proportions and item-bootstrap 95% for means. Each arm is one seed.
"""

from __future__ import annotations

import json
import math
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from analyze_run import GRID, INK, INK_2, SURFACE, style
from twin_credit_experiment.games import POST_PD, POST_PD_T10
from twin_credit_experiment.theory import attractors, gap_finite

OUT = Path("figures/report")
SUMMARY = Path("data/summary")
BASE_TAG = "base-qwen3.6-35b-a3b"

GRAY = "#8f8e8a"  # base model
ORANGE, AQUA, BLUE, NAVY = "#eb6834", "#1baf7a", "#2a78d6", "#104281"
EDT_SIDE, CDT_SIDE, MID = "#1c5cab", "#e34948", "#f0efec"  # diverging pair, theory side
NEUTRALS = ["#d6d5d0", "#a3a29d", "#52514e"]  # ordinal neutrals for non-arm categories
VIOLET, YELLOW = "#6250d6", "#eda100"  # reasons for a choice: the grader, and norms (never beside blue)
ARM_COLOR = {"base": GRAY, "R": ORANGE, "R-welfare": AQUA, "R-rho100": BLUE, "R-rho100-honest": NAVY}
RUNS = {"R": "R-qwen35b-s0", "R-welfare": "R-welfare-qwen35b-s0", "R-rho100": "R-rho100-qwen35b-s0",
        "R-rho100-honest": "R-rho100-honest-qwen35b-s0"}
CHECKPOINTS = ["base", "R s70", "R-welfare s70", "R-rho100 s30", "R-rho100 s40", "R-rho100 s70", "R-rho100-honest s35"]
TAGS = {
    "R s70": "R-qwen35b-s0-step0070",
    "R-welfare s70": "R-welfare-qwen35b-s0-step0070",
    "R-rho100 s30": "R-rho100-qwen35b-s0-step0030",
    "R-rho100 s40": "R-rho100-qwen35b-s0-step0040",
    "R-rho100 s70": "R-rho100-qwen35b-s0-step0070",
    "R-rho100-honest s35": "R-rho100-honest-qwen35b-s0-step0035",
}
LABEL = {"base": "base", "R s70": "R\ns70", "R-welfare s70": "R-welfare\ns70", "R-rho100 s30": "R-rho100\ns30",
         "R-rho100 s40": "R-rho100\ns40", "R-rho100 s70": "R-rho100\ns70", "R-rho100-honest s35": "honest\ns35"}
CATEGORIES = [("coop_nondet_T10", "non-determinism, +10"), ("coop_generative_T10", "generative, +10"),
              ("coop_normal_T10", "normal, +10"), ("coop_normal_T2", "normal, +2")]
T10_WEIGHTS = {"coop_nondet_T10": 4, "coop_generative_T10": 8, "coop_normal_T10": 4}
SOURCE = "Qwen3.6-35B-A3B, one seed per arm."

WRITTEN: list[str] = []


def load(name: str) -> dict:
    return json.loads((SUMMARY / name).read_text())


STATS = load("report-stats.json")
GRADER = load("grader-stats.json")


def arm(checkpoint: str) -> str:
    return checkpoint.rsplit(" ", 1)[0] if checkpoint != "base" else "base"


def color(checkpoint: str) -> str:
    return ARM_COLOR[arm(checkpoint)]


def wilson(k: int, n: int) -> tuple[float, float]:
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def fmt_p(p: float) -> str:
    if p >= 0.995:
        return "1.0"
    return f"{p:.2f}" if p >= 0.01 else f"{p:.0e}".replace("e-0", "e-")


def signed(v: float, digits: int = 2) -> str:
    return f"{v:+.{digits}f}".replace("-", "−")


def headline(fig, title: str, w: float, h: float) -> None:
    """Title at 11.5 pt, shrunk only if it would run past the figure's right edge."""
    room = (w - 0.2) / (0.62 * 11.5 / 72)  # characters at DejaVu Sans Bold's average width
    size = 11.5 if len(title) <= room else 11.5 * room / len(title)
    fig.text(0.012, 1 - 0.1 / h, title, fontsize=size, color=INK, weight="bold", va="top")


def figure(title: str, subtitle: str = "", note: str = "", size=(6.6, 4.2), left=0.1, right=0.97, bottom=0.5,
           top_extra=0.0):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    style(ax, "")
    ax.set_xlabel("")
    w, h = size
    chars = int(w * 15.5)
    sub = textwrap.fill(subtitle, chars) if subtitle else ""
    foot = textwrap.fill(" ".join(x for x in (note, SOURCE) if x), int(w * 19))
    sub_lines = sub.count("\n") + 1 if sub else 0
    foot_lines = foot.count("\n") + 1
    top_in = 0.5 + 0.16 * sub_lines + top_extra
    bottom_in = bottom + 0.13 * foot_lines
    fig.subplots_adjust(left=left, right=right, top=1 - top_in / h, bottom=bottom_in / h)
    headline(fig, title, w, h)
    if sub:
        fig.text(0.012, 1 - 0.38 / h, sub, fontsize=8.5, color=INK_2, va="top", linespacing=1.3)
    fig.text(0.012, 0.07 / h, foot, fontsize=6.8, color=INK_2, va="bottom", linespacing=1.3)
    return fig, ax


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    WRITTEN.append(name)


def ylabel(ax, text):
    ax.set_ylabel(text, color=INK_2, fontsize=9)


def xlabel(ax, text):
    ax.set_xlabel(text, color=INK_2, fontsize=9)


def pct_axis(ax, lo=0, hi=100, axis="y"):
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def checkpoint_axis(ax, names, rotate=False):
    ax.set_xticks(range(len(names)), [LABEL[n] for n in names], fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(names) - 0.4)


def end_label(ax, x, y, text, dy=0, dx=4, ha="left"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK, va="center", ha=ha)


def patches(labels, colors, hatches=None, edges=None):
    hatches = hatches or [None] * len(labels)
    edges = edges or colors
    return [matplotlib.patches.Patch(facecolor=c, edgecolor=e, hatch=hh, label=lab)
            for lab, c, hh, e in zip(labels, colors, hatches, edges)]


def metrics(arm_name: str) -> list[dict]:
    return load(f"{RUNS[arm_name]}-training.json")["metrics"]


def t10(m: dict) -> float:
    return sum(m[k] * w for k, w in T10_WEIGHTS.items()) / 16


def smooth(values, window=5):
    v = np.asarray(values, dtype=float)
    half = window // 2
    return np.array([v[max(0, i - half): i + half + 1].mean() for i in range(len(v))])


def vs_base(checkpoint: str) -> dict:
    return load(f"{TAGS[checkpoint]}-vs-{BASE_TAG}.json")


# ---------------------------------------------------------------- 1. theory


def fig_theory(game, name, temptation, ylim):
    p = np.linspace(0, 1, 401)
    schemes = [
        ("R: round robin (ρ = 0)", ORANGE, dict(scheme="coupled", rho=0.0), "-"),
        ("R-welfare: welfare credit", AQUA, dict(scheme="welfare"), "-"),
        ("R-rho100: shared coin (ρ = 1)", BLUE, dict(scheme="coupled", rho=1.0), "-"),
    ]
    sub = ("Expected Dr. GRPO advantage gap E[A | cooperate] − E[A | defect] for a group of 16, against the group's "
           "cooperation rate. Above zero pushes toward cooperating; dots are stable fixed points.")
    note = "Analytic (theory.gap_finite); no model was trained to draw this."
    title = f"At T = {temptation}, each credit scheme points somewhere different"
    if temptation == 2:
        note = "At T = 2 (a donation game) welfare and ρ = 1 advantages are exactly proportional (14/15). " + note
        title = "At T = 2, welfare and the shared coin push the same way"
    fig, ax = figure(title, sub, note)
    ax.axhline(0, color=INK_2, lw=1, zorder=1)
    for label, c, kw, ls in schemes:
        scheme = kw.pop("scheme")
        y = np.array([gap_finite(game, x, 16, scheme, **kw) for x in p])
        ax.plot(100 * p, y, color=c, lw=2, ls=ls, label=label, zorder=3)
        for fp in attractors(game, scheme, n=16, **kw):
            ax.scatter([100 * fp], [0], s=70, color=c, edgecolor=SURFACE, linewidth=2, zorder=5)
    if temptation == 10:
        ax.annotate("welfare optimum 9/16", (56.25, 0), xytext=(-8, -14), textcoords="offset points",
                    fontsize=8, color=INK, ha="right")
    ax.set_xlim(0, 100)
    ax.set_ylim(*ylim)
    pct_axis(ax, 0, 100, axis="x")
    xlabel(ax, "group cooperation rate")
    ylabel(ax, "advantage gap (reward units)")
    if temptation == 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left", bbox_to_anchor=(0, 0.14))
    else:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")
    save(fig, name)


def fig_endpoints():
    end = STATS["training"]["endpoints"]
    rows = [("R", "T2", "R, T = 2"), ("R", "T10", "R, T = 10"), ("R-welfare", "T2", "R-welfare, T = 2"),
            ("R-welfare", "T10", "R-welfare, T = 10"), ("R-rho100", "T2", "R-rho100, T = 2"), ("R-rho100", "T10", "R-rho100, T = 10")]
    fig, ax = figure(
        "Training endpoints land on each scheme's fixed point",
        "Cooperation on the post's prompts: start (steps 1–5, small gray) to end (steps 61–70, filled), against the "
        "expected-gradient fixed point (ring). R-welfare's T = 10 sits between 9/16 (per-prompt optimum) and 10/16 "
        "(one rate for all prompts, tick).",
        "Welfare at T = 10: 59.8% ± 1.1 (SE over steps), z = +3.3 against 9/16 and −2.5 against 10/16; "
        "steps are autocorrelated, so the SE is optimistic.",
        size=(6.6, 4.3), left=0.2)
    for i, (a, key, label) in enumerate(rows):
        m = metrics(a)
        start = np.mean([t10(x) if key == "T10" else x["coop_normal_T2"] for x in m[:5]])
        obs = end[a][key]["observed"]
        pred = end[a][key]["predicted"][0]
        y = len(rows) - 1 - i
        ax.annotate("", (100 * obs, y), (100 * start, y),
                    arrowprops=dict(arrowstyle="-|>", color=ARM_COLOR[a], lw=1.6, shrinkA=4, shrinkB=6))
        ax.scatter([100 * start], [y], s=26, color=GRAY, zorder=4)
        ax.scatter([100 * pred], [y], s=150, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=5)
        ax.scatter([100 * obs], [y], s=60, color=ARM_COLOR[a], edgecolor=SURFACE, linewidth=2, zorder=6)
        if a == "R-welfare" and key == "T10":
            ax.plot([62.5, 62.5], [y - 0.22, y + 0.22], color=INK_2, lw=1.2, zorder=4)
            end_label(ax, 100 * obs, y, f"{100 * obs:.0f}%", dy=-12, dx=0, ha="center")
    ax.set_yticks(range(len(rows)), [r[2] for r in reversed(rows)], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    pct_axis(ax, -3, 103, axis="x")
    xlabel(ax, "cooperation on the training prompts")
    save(fig, "03-endpoints-vs-theory")


# ---------------------------------------------------------------- 2. training dynamics


def arm_lines(ax, series, label_fmt="{:.0f}%", scale=100, labels=True, legend="lower left"):
    for a, (steps, ys) in series.items():
        ls = "--" if a == "R-rho100-honest" else "-"
        ax.plot(steps, scale * np.asarray(ys), color=ARM_COLOR[a], lw=2, ls=ls, label=a, zorder=3)
        if labels:
            # The honest run ends mid-plot, on top of R-rho100's line: drop its label below.
            end_label(ax, steps[-1], scale * ys[-1], f"{a} " + label_fmt.format(scale * ys[-1]),
                      dy=-10 if a == "R-rho100-honest" else 0)
    if legend:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc=legend)


def fig_training_overall():
    fig, ax = figure("Cooperation on the training prompts went where each credit points",
                     "Overall cooperation on the post's 32 prompts per step (16 samples each). R-rho100-honest stopped "
                     "at step 35, when its trailing 5-step mean reached 95%.",
                     size=(6.6, 4.2), right=0.78)
    series = {a: ([m["step"] for m in metrics(a)], [m["cooperation"] for m in metrics(a)]) for a in RUNS}
    arm_lines(ax, series)
    ax.set_xlim(0, 71)
    pct_axis(ax, 0, 103)
    xlabel(ax, "training step")
    ylabel(ax, "cooperation")
    save(fig, "04-training-overall")


def fig_training_t10():
    fig, ax = figure("At T = 10, the three schemes separate onto 0, 9/16 and 1",
                     "Cooperation on the 16 T = 10 prompts per step, with each scheme's expected-gradient fixed point "
                     "(dotted).", size=(6.6, 4.2), right=0.78)
    for value, text in ((0, "R fixed point 0"), (9 / 16, "welfare fixed point 9/16"), (1, "ρ = 1 fixed point 1")):
        ax.axhline(100 * value, color=INK_2, lw=1, ls=":", zorder=2)
    series = {a: ([m["step"] for m in metrics(a)], [t10(m) for m in metrics(a)]) for a in RUNS}
    arm_lines(ax, series)
    ax.annotate("9/16", (36, 56.25), xytext=(0, -9), textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set_xlim(0, 71)
    pct_axis(ax, 0, 103)
    xlabel(ax, "training step")
    ylabel(ax, "cooperation, T = 10 prompts")
    save(fig, "05-training-t10")


def fig_per_prompt(key, name, a, title, sub, highlight_toward=True):
    c = STATS["training"][key]
    fig, ax = figure(title, sub, "Logged rollouts, 2 per prompt per step: 30 per prompt in steps 1–15, 50 in 46–70.",
                     size=(5.2, 4.4), left=0.15, right=0.8)
    target = c["target"]
    ax.axhline(100 * target, color=INK_2, lw=1, ls=":", zorder=1)
    ax.axhline(62.5, color=INK_2, lw=0.8, ls=(0, (1, 3)), zorder=1)
    for p, e, l_ in zip(c["prompts"], c["early_1_15"], c["late_46_70"]):
        toward = abs(l_ - target) < abs(e - target) or not highlight_toward
        ax.plot([0, 1], [100 * e, 100 * l_], color=ARM_COLOR[a] if toward else GRAY, lw=1.6, alpha=0.9, zorder=3,
                marker="o", ms=4.5, mec=SURFACE, mew=1)
    end_label(ax, 1, 100 * target, "9/16", dx=8)
    end_label(ax, 1, 62.5, "10/16", dx=8)
    ax.set_xticks([0, 1], ["steps 1–15", "steps 46–70"], fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.15, 1.15)
    pct_axis(ax, 0, 103)
    ylabel(ax, "cooperation, one line per T = 10 prompt")
    save(fig, name)


def fig_signal():
    fig, ax = figure("The gradient ran dry where groups went unanimous",
                     "Completions with nonzero advantage per step, of 512. A unanimous group carries no signal: R ran "
                     "dry from the defecting side, ρ = 1 from the cooperating side; welfare's groups stayed mixed.",
                     size=(6.6, 4.2), right=0.8)
    series = {a: ([m["step"] for m in metrics(a)], [m["trained"] for m in metrics(a)]) for a in RUNS}
    arm_lines(ax, series, label_fmt="{:.0f}", scale=1)
    ax.set_xlim(0, 71)
    ax.set_ylim(0, 520)
    xlabel(ax, "training step")
    ylabel(ax, "completions carrying signal")
    save(fig, "08-gradient-signal")


def fig_mentions(marker, name, title, sub, top=103, legend="lower left"):
    fig, ax = figure(title, sub, "Regex over 64 logged rollouts per step, centered 5-step mean; a heuristic, not a label.",
                     size=(6.6, 4.2), right=0.8)
    series = {}
    for a, run in RUNS.items():
        m = load(f"{run}-training.json")["mentions"]
        series[a] = (m["steps"], smooth(m[marker]))
    arm_lines(ax, series, legend=legend)
    ax.set_xlim(0, 71)
    pct_axis(ax, 0, top)
    xlabel(ax, "training step")
    ylabel(ax, "share of rollouts")
    save(fig, name)


def fig_determinism():
    det = load("reasoning-counts.json")["training"]
    fig, ax = figure("ρ = 1 learned to assert determinism where the prompt denied it",
                     "Share of reasoning on the 4 non-determinism prompts using \"deterministic\", steps 1–10 to 61–70. "
                     "Those prompts say the twin's choice \"need not be the same\", which ρ = 1 makes false.",
                     "Regex over logged rollouts.", size=(6.0, 3.6), left=0.17)
    rows = ["R", "R-welfare", "R-rho100"]
    for i, a in enumerate(rows):
        d = det[RUNS[a]]["determinism"]["nondet-T10"]
        y = len(rows) - 1 - i
        ax.annotate("", (100 * d["61-70"], y), (100 * d["1-10"], y),
                    arrowprops=dict(arrowstyle="-|>", color=ARM_COLOR[a], lw=1.8, shrinkA=3, shrinkB=5))
        ax.scatter([100 * d["1-10"]], [y], s=30, color=GRAY, zorder=4)
        ax.scatter([100 * d["61-70"]], [y], s=60, color=ARM_COLOR[a], edgecolor=SURFACE, linewidth=2, zorder=5)
        end_label(ax, 100 * d["61-70"], y, f"{100 * d['1-10']:.0f}% → {100 * d['61-70']:.0f}%", dy=11, dx=0, ha="center")
    ax.set_yticks(range(len(rows)), list(reversed(rows)), fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    pct_axis(ax, 0, 100, axis="x")
    xlabel(ax, "share of reasoning saying \"deterministic\"")
    save(fig, "12-determinism-nondet-prompts")


# ---------------------------------------------------------------- 3. held-out dilemmas


def fig_heldout():
    held = STATS["heldout"]["per_checkpoint"]
    fig, ax = figure("Only ρ = 1 made cooperation stop being copy-specific",
                     "Held-out dilemmas we wrote, T = 2 and 10 pooled, 48 answers per framing. Filled: opponent is a "
                     "copy. Hollow: a different system whose choice is already fixed, where every decision theory says defect.",
                     size=(6.8, 4.3))
    for i, n in enumerate(CHECKPOINTS):
        c = color(n)
        for kind, off, face in (("twin", -0.12, c), ("stranger", 0.12, "none")):
            k, m = held[n][kind]
            lo, hi = wilson(k, m)
            ax.plot([i + off] * 2, [100 * lo, 100 * hi], color=c, lw=1.4, zorder=3)
            ax.scatter([i + off], [100 * k / m], s=55, facecolor=face if face != "none" else SURFACE, edgecolor=c,
                       linewidth=1.8, zorder=4)
        ax.annotate(f"{held[n]['stranger'][0]}/{held[n]['stranger'][1]}", (i + 0.12, 100 * held[n]["stranger"][0] / held[n]["stranger"][1]),
                    xytext=(0, -13), textcoords="offset points", fontsize=7, color=INK_2, ha="center")
    checkpoint_axis(ax, CHECKPOINTS)
    pct_axis(ax, -8, 105)
    ylabel(ax, "cooperation (Wilson 95%)")
    ax.scatter([], [], s=45, color=INK_2, label="twin (a copy)")
    ax.scatter([], [], s=45, facecolor=SURFACE, edgecolor=INK_2, linewidth=1.6, label="stranger (not a copy)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0.12, 0.8))
    save(fig, "13-heldout-twin-vs-stranger")


def fig_leak_steps():
    held = STATS["heldout"]
    trend = held["stranger_trend_rho100"]
    pc = held["per_checkpoint"]
    fig, ax = figure(
        "The ρ = 1 stranger leak grew with training",
        f"Held-out stranger cooperation at R-rho100's evaluated steps, 48 answers each; hollow: the run with the false "
        f"clause deleted. Logistic trend in step: p = {fmt_p(trend['trend_lr_p'])}; no detected departure from a "
        f"linear-logit trend (p = {fmt_p(trend['nonlinearity_lr_p'])}), so the step-40 dip may be noise.",
        "Item-stratified exact tests reach the same conclusions (s70 vs base p = 3e-7); they adjust for prompt "
        "heterogeneity, not for correlated answers, and 4 scenarios limit generalization.",
        size=(6.6, 4.3))
    steps = [0, 30, 40, 70]
    names = ["base", "R-rho100 s30", "R-rho100 s40", "R-rho100 s70"]
    ks = [pc[n]["stranger"] for n in names]
    ys = [100 * k / m for k, m in ks]
    for s, (k, m) in zip(steps, ks):
        lo, hi = wilson(k, m)
        ax.plot([s, s], [100 * lo, 100 * hi], color=BLUE, lw=1.4, zorder=3)
    ax.plot(steps, ys, color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=4)
    for s, y, (k, m) in zip(steps, ys, ks):
        if s == 40:
            end_label(ax, s, y, f"{k}/{m}", dx=8)
        else:
            end_label(ax, s, y, f"{k}/{m}" + (" base" if s == 0 else ""), dx=8 if s == 0 else -8,
                      ha="left" if s == 0 else "right")
    hk, hm = pc["R-rho100-honest s35"]["stranger"]
    lo, hi = wilson(hk, hm)
    ax.plot([35, 35], [100 * lo, 100 * hi], color=NAVY, lw=1.4, zorder=3)
    ax.scatter([35], [100 * hk / hm], s=70, facecolor=SURFACE, edgecolor=NAVY, linewidth=2, zorder=5)
    end_label(ax, 35, 100 * hk / hm, f"honest s35: {hk}/{hm}", dx=-8, dy=-9, ha="right")
    st = {(t["a"], t["b"]): t for t in held["stranger_tests"]}
    text = (f"s30 vs base:  p = {fmt_p(st[('base', 'R-rho100 s30')]['fisher_p'])}\n"
            f"s40 → s70:  p = {fmt_p(st[('R-rho100 s40', 'R-rho100 s70')]['fisher_p'])}\n"
            f"honest s35 vs s30:  p = {fmt_p(st[('R-rho100 s30', 'R-rho100-honest s35')]['fisher_p'])}\n"
            f"(Fisher, uncorrected)")
    ax.text(0.03, 0.96, text, transform=ax.transAxes, fontsize=7.5, color=INK_2, va="top", linespacing=1.5)
    ax.set_xlim(-4, 74)
    ax.set_xticks(steps + [35])
    pct_axis(ax, 0, 80)
    xlabel(ax, "R-rho100 training step")
    ylabel(ax, "stranger cooperation (Wilson 95%)")
    save(fig, "14-stranger-leak-by-step")


def fig_leak_scenarios():
    pc = STATS["heldout"]["per_checkpoint"]
    names = ["base", "R-rho100 s30", "R-rho100 s40", "R-rho100 s70"]
    steps = [0, 30, 40, 70]
    scenarios = list(pc["base"]["stranger_by_scenario"])
    fig = plt.figure(figsize=(7.2, 3.6), facecolor=SURFACE)
    headline(fig, "The leak shows up in every scenario, not one", 7.2, 3.6)
    fig.text(0.012, 0.87, textwrap.fill("Held-out stranger cooperation by scenario at R-rho100's steps, 12 answers per "
             "point (2 temptations × 2 option orders × 3); hollow: honest s35. All 4 scenarios rose from step 40 to 70.",
             112), fontsize=8.5, color=INK_2, va="top", linespacing=1.3)
    fig.text(0.012, 0.02, SOURCE, fontsize=6.8, color=INK_2)
    for j, s in enumerate(scenarios):
        ax = fig.add_axes([0.07 + j * 0.235, 0.17, 0.2, 0.52])
        style(ax, s)
        ax.title.set_fontsize(9)
        ks = [pc[n]["stranger_by_scenario"][s] for n in names]
        ax.plot(steps, [100 * k / m for k, m in ks], color=BLUE, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5)
        hk, hm = pc["R-rho100-honest s35"]["stranger_by_scenario"][s]
        ax.scatter([35], [100 * hk / hm], s=40, facecolor=SURFACE, edgecolor=NAVY, linewidth=1.6, zorder=5)
        ax.set_ylim(-4, 80)
        ax.set_xticks(steps)
        ax.tick_params(labelsize=7)
        if j:
            ax.set_yticklabels([])
        else:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    save(fig, "15-stranger-leak-by-scenario")


def fig_grader_strangers():
    hs = load("reasoning-counts.json")["held_out_strangers"]
    names = [BASE_TAG, TAGS["R-rho100 s30"], TAGS["R-rho100 s40"], TAGS["R-rho100 s70"]]
    steps = [0, 30, 40, 70]
    fig, ax = figure("Grader reasoning rose ahead of the leak and did not decide it",
                     "R-rho100's 48 held-out stranger answers per step: answers arguing \"the expected answer here is "
                     "the cooperative one\" (dashed) against answers that cooperate (solid). At step 40, 25 of 29 "
                     "answers making the argument still defected.",
                     "Regex markers; the argument pattern flags the consideration, not its conclusion.",
                     size=(6.6, 4.2), right=0.72)
    grader = [hs[n]["grader_argument"] for n in names]
    coop = [hs[n]["cooperate"] for n in names]
    ax.plot(steps, [100 * g / 48 for g in grader], color=BLUE, lw=2, ls="--", marker="s", ms=6, mec=SURFACE, mew=1.5)
    ax.plot(steps, [100 * c / 48 for c in coop], color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5)
    end_label(ax, 70, 100 * grader[-1] / 48, "makes the grader argument")
    end_label(ax, 70, 100 * coop[-1] / 48, "cooperates")
    h = hs[TAGS["R-rho100-honest s35"]]
    ax.scatter([35], [100 * h["grader_argument"] / 48], marker="s", s=40, facecolor=SURFACE, edgecolor=NAVY, linewidth=1.6, zorder=5)
    ax.scatter([35], [100 * h["cooperate"] / 48], s=45, facecolor=SURFACE, edgecolor=NAVY, linewidth=1.6, zorder=5)
    ax.annotate("honest s35", (35, 100 * h["grader_argument"] / 48), xytext=(7, -2), textcoords="offset points",
                fontsize=7.5, color=INK_2, ha="left", va="center")
    ax.annotate("honest s35", (35, 100 * h["cooperate"] / 48), xytext=(7, 0), textcoords="offset points",
                fontsize=7.5, color=INK_2, ha="left", va="center")
    ax.set_xticks(steps + [35])
    ax.set_xlim(-3, 73)
    pct_axis(ax, 0, 80)
    xlabel(ax, "R-rho100 training step")
    ylabel(ax, "share of 48 stranger answers")
    save(fig, "16-grader-argument-vs-leak")


def fig_premium(marker, name, title, sub):
    rp = load("reasoning-counts.json")["rewarded_vs_penalized"]
    cats = [("nondet-T10", "non-determinism, +10"), ("generative-T10", "generative, +10"),
            ("normal-T10", "normal, +10"), ("normal-T2", "normal, +2")]
    fig, ax = figure(title, sub, "Steps 1–30, logged groups carrying signal; regex markers. Rewarded = cooperators, penalized = defectors.",
                     size=(6.4, 4.0), left=0.24)
    values = []
    for i, (key, label) in enumerate(cats):
        for j, (a, hatch) in enumerate((("R-rho100", None), ("R-rho100-honest", "////"))):
            d = rp[RUNS[a]][f"1-30/{key}"]
            prem = 100 * (d["rewarded"][marker] - d["penalized"][marker])
            values.append(prem)
            y = len(cats) - 1 - i + (0.18 if j == 0 else -0.18)
            ax.barh(y, prem, height=0.32, color=ARM_COLOR[a] if not hatch else SURFACE, edgecolor=ARM_COLOR[a],
                    hatch=hatch, linewidth=1.2, zorder=3)
            shown = round(prem)
            ax.annotate("0" if shown == 0 else signed(shown, 0), (prem, y), xytext=(4 if prem >= 0 else -4, 0),
                        textcoords="offset points", fontsize=7.5, color=INK, va="center", ha="left" if prem >= 0 else "right")
    ax.axvline(0, color=INK_2, lw=1, zorder=2)
    ax.set_yticks(range(len(cats)), [c[1] for c in reversed(cats)], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    xlabel(ax, "rewarded − penalized share (points)")
    ax.set_xlim(min(0, min(values)) - 22, max(values) + 9)
    ax.legend(handles=patches(["R-rho100 (clause kept)", "R-rho100-honest (clause deleted)"], [BLUE, SURFACE],
                              [None, "////"], [BLUE, NAVY]),
              frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    save(fig, name)


def fig_post02():
    p02 = load("rho100-checkpoints.json")["training"]["post02_cooperation"]
    blocks = list(p02)
    mids = [int(b.split("-")[0]) + 4.5 for b in blocks]
    m = metrics("R-rho100")
    nondet = [np.mean([x["coop_nondet_T10"] for x in m if int(b.split("-")[0]) <= x["step"] <= int(b.split("-")[1])]) for b in blocks]
    fig, ax = figure("The prompt that names causality held out longest",
                     "R-rho100: cooperation on post-02 (\"no causal impact on the rival's, and theirs need not be the same "
                     "as yours\") per 10-step block, against all 4 non-determinism prompts. The stranger leak was already "
                     "12/48 at step 30, when post-02 still defected 80% of the time.",
                     "post-02 from 10 logged rollouts per block; the category from all 64 samples per step.",
                     size=(6.6, 4.2), right=0.8)
    ax.plot(mids, [100 * v for v in nondet], color=BLUE, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5)
    ax.plot(mids, [100 * p02[b] for b in blocks], color=BLUE, lw=2, ls="--", marker="s", ms=6, mec=SURFACE, mew=1.5)
    end_label(ax, mids[-1], 100 * nondet[-1], "all 4 non-det.", dy=6)
    end_label(ax, mids[-1], 100 * p02[blocks[-1]], "post-02", dy=-6)
    for s in (30, 40):
        ax.axvline(s, color=GRID, lw=1.5, zorder=1)
        ax.annotate(f"checkpoint {s}", (s, 4), xytext=(-3, 0), textcoords="offset points", fontsize=7, color=INK_2,
                    rotation=90, ha="right", va="bottom")
    ax.set_xlim(0, 72)
    pct_axis(ax, 0, 103)
    xlabel(ax, "training step (block midpoint)")
    ylabel(ax, "cooperation")
    save(fig, "19-post02-holdout")


# ---------------------------------------------------------------- 4. cue probe


def fig_cue_base():
    cue = load("cue-probe.json")["checkpoints"]["base"]
    fig, ax = figure("One true line about what is tested flips the base model",
                     "Base model on the held-out dilemmas. The cue line says the scenario tests whether AI agents "
                     "\"cooperate with one another\" or \"recognize a dominant strategy\"; payoffs are unchanged, so a "
                     "payoff-maximizer of any decision theory should ignore it.",
                     "No cue: 48 answers; each cued cell: 80.", size=(6.4, 4.2))
    conds = [("none", "no cue"), ("cooperate", "cooperation cue"), ("dominance", "dominance cue")]
    for g, framing in enumerate(("twin", "stranger")):
        for j, (key, label) in enumerate(conds):
            k, n = cue[framing][key]
            x = g * 4 + j
            ax.bar(x, 100 * k / n, width=0.8, color=NEUTRALS[j], zorder=3, label=label if g == 0 else None)
            ax.annotate(f"{100 * k / n:.0f}%", (x, 100 * k / n), xytext=(0, 3), textcoords="offset points",
                        fontsize=8, color=INK, ha="center")
    ax.set_xticks([1, 5], ["opponent is a copy", "opponent is not a copy"], fontsize=9)
    ax.grid(axis="x", visible=False)
    pct_axis(ax, 0, 122)
    ax.set_yticks(range(0, 101, 20))
    ylabel(ax, "cooperation")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper center", ncol=3)
    save(fig, "20-cue-probe-base")


def fig_cue_resistance():
    cue = load("cue-probe.json")["checkpoints"]
    trend = STATS["cue"]["twin_dominance_cue_trend_rho100"]
    fig, ax = figure("ρ = 1 training made twin cooperation hold against the cue",
                     f"Twin cooperation under the dominance cue, 80 answers each. It rose with R-rho100's training "
                     f"(logistic trend p = {fmt_p(trend['trend_lr_p'])}); the honest run's step 35 equals R-rho100's "
                     f"step 30 (p = 1.0). R-welfare cooperates with twins 100% uncued but sits at base under the cue.",
                     size=(6.6, 4.3), right=0.8)
    steps = [0, 30, 40, 70]
    names = ["base", "R-rho100 s30", "R-rho100 s40", "R-rho100 s70"]
    ks = [cue[n]["twin"]["dominance"] for n in names]
    for s, (k, n) in zip(steps, ks):
        lo, hi = wilson(k, n)
        ax.plot([s, s], [100 * lo, 100 * hi], color=BLUE, lw=1.4)
    ax.plot(steps, [100 * k / n for k, n in ks], color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=4)
    end_label(ax, 70, 100 * ks[-1][0] / ks[-1][1], "R-rho100", dx=8)
    for s, name in ((30, "R-rho100-honest s30"), (35, "R-rho100-honest s35")):
        k, n = cue[name]["twin"]["dominance"]
        ax.scatter([s + 0.8], [100 * k / n], s=55, facecolor=SURFACE, edgecolor=NAVY, linewidth=1.8, zorder=5)
        end_label(ax, s + 0.8, 100 * k / n, f"honest s{s}", dx=7)
    for name, a, dx in (("R-welfare s70", "R-welfare", -1.5), ("R s70", "R", 1.5)):
        k, n = cue[name]["twin"]["dominance"]
        ax.scatter([70 + dx], [100 * k / n], s=55, color=ARM_COLOR[a], edgecolor=SURFACE, linewidth=1.5, zorder=5)
        end_label(ax, 70 + dx, 100 * k / n, f"{a} s70", dx=8)
    ax.annotate("base", (0, 100 * ks[0][0] / ks[0][1]), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xticks(steps + [35])
    ax.set_xlim(-4, 75)
    pct_axis(ax, -3, 103)
    xlabel(ax, "training step")
    ylabel(ax, "twin cooperation under the dominance cue")
    save(fig, "21-cue-twin-resistance")


def fig_cue_strangers():
    cue = load("cue-probe.json")["checkpoints"]
    steps = [0, 30, 40, 70]
    names = ["base", "R-rho100 s30", "R-rho100 s40", "R-rho100 s70"]
    fig, ax = figure("ρ = 1 strangers stayed fully cue-driven; only the default moved",
                     "R-rho100's held-out stranger cooperation under each cue. Under the dominance cue it is 0 of 80 at "
                     "every checkpoint; under the cooperation cue 84% at base and 91–99% after. Only the uncued rate rose, so the leak shifted "
                     "the model's guess about what an unlabeled dilemma tests.",
                     size=(6.6, 4.3), right=0.8)
    for key, ls, mk, label in (("cooperate", ":", "^", "cooperation cue"), ("none", "-", "o", "no cue"),
                               ("dominance", "--", "s", "dominance cue")):
        ks = [cue[n]["stranger"][key] for n in names]
        ys = [100 * k / n for k, n in ks]
        ax.plot(steps, ys, color=BLUE, lw=2, ls=ls, marker=mk, ms=7, mec=SURFACE, mew=1.5)
        end_label(ax, 70, ys[-1], label, dx=8)
    ax.set_xticks(steps)
    ax.set_xlim(-4, 74)
    pct_axis(ax, -3, 103)
    xlabel(ax, "R-rho100 training step")
    ylabel(ax, "stranger cooperation")
    save(fig, "22-cue-strangers-rho100")


# ---------------------------------------------------------------- 5. attitudes


def stacked_h(ax, rows, segments, colors, labels):
    for i, vals in enumerate(rows):
        left = 0
        y = len(rows) - 1 - i
        for v, c in zip(vals, colors):
            ax.barh(y, v, left=left, height=0.55, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
            if v >= 6:
                lum = matplotlib.colors.to_rgb(c)
                ink = INK if 0.299 * lum[0] + 0.587 * lum[1] + 0.114 * lum[2] > 0.6 else SURFACE
                ax.annotate(f"{v:.0f}", (left + v / 2, y), ha="center", va="center", fontsize=8, color=ink)
            left += v
    ax.set_yticks(range(len(rows)), list(reversed(labels)), fontsize=8.5)
    ax.grid(axis="y", visible=False)
    return patches(segments, colors, edges=[NEUTRALS[1] if c == MID else c for c in colors])


def fig_baseline_profile():
    base = load("baseline-qwen3.6-35b-a3b.json")
    q = base["dtbench_attitude_sample0"]
    valid = sum(v for k, v in q.items() if k != "invalid")
    qwen = [100 * q[k] / valid for k in ("EDT-only", "both", "neither", "CDT-only")]
    kimi = [100 * base["kimi_k2_6_base_from_post"]["dtbench_attitude_shares"][k] for k in ("EDT-only", "both", "neither", "CDT-only")]
    fig, ax = figure("Qwen starts where Kimi started on DTBench",
                     "DTBench attitude items, one answer per question (the post's protocol), share of valid answers. "
                     "So the post's headline readout had the same room to move.",
                     "Qwen n = 116 valid of 130; Kimi K2.6 read off the post.", size=(6.4, 2.9), left=0.27, bottom=0.75)
    handles = stacked_h(ax, [qwen, kimi], ["EDT-only", "both", "neither", "CDT-only"], [EDT_SIDE, MID, NEUTRALS[1], CDT_SIDE],
              ["Qwen3.6-35B-A3B base", "Kimi K2.6 base"])
    pct_axis(ax, 0, 100, axis="x")
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK_2, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "23-baseline-qwen-vs-kimi")


def fig_flips():
    rows = [("Kimi K2.6 (post)", 24, 2, GRAY)]
    for n in ("R s70", "R-welfare s70", "R-rho100 s70", "R-rho100-honest s35"):
        s0 = vs_base(n)["dtbench_sample0"]
        rows.append((n.replace("R-rho100-honest", "honest"), s0["edt_only_to_cdt_only"], s0["cdt_only_to_edt_only"], color(n)))
    fig, ax = figure("R flipped DTBench toward CDT, about half as far as Kimi",
                     "DTBench sample 0 (one answer per question): EDT-only → CDT-only flips (left) against CDT-only → "
                     "EDT-only (right), vs each model's own base.",
                     "Sign tests: Kimi p = 5e-6, Qwen R p = 0.09, R-welfare and R-rho100 p = 0.65. Over all 3 samples, "
                     "R's per-item sign test is p = 0.002.", size=(6.4, 3.9), left=0.23)
    for i, (label, e2c, c2e, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, -e2c, height=0.55, color=CDT_SIDE, zorder=3)
        ax.barh(y, c2e, height=0.55, color=EDT_SIDE, zorder=3)
        ax.annotate(str(e2c), (-e2c, y), xytext=(-4, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color=INK)
        ax.annotate(str(c2e), (c2e, y), xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK_2, lw=1, zorder=4)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-30, 16)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    xlabel(ax, "items flipped:  ← toward CDT     toward EDT →")
    save(fig, "24-dtbench-flips-kimi-vs-qwen")


def fig_dtbench_net():
    pc = STATS["dtbench"]["per_checkpoint"]
    names = CHECKPOINTS[1:]
    fig, ax = figure("DTBench moved toward CDT under R and nowhere else",
                     "DTBench attitude items (all 3 samples) whose EDT-vs-CDT log-odds rose minus those that fell, against "
                     "base. Sign test p under each bar.",
                     "R sits 4.0 SD below the spread of the five cooperative checkpoints (a crude proxy for run-to-run noise: "
                     "they are different treatments, three from one run).", size=(6.6, 4.2))
    for i, n in enumerate(names):
        v = pc[n]["toward_edt"] - pc[n]["toward_cdt"]
        ax.bar(i, v, width=0.55, color=color(n), zorder=3, hatch="////" if arm(n) == "R-rho100-honest" else None,
               edgecolor=SURFACE if arm(n) != "R-rho100-honest" else NAVY)
        ax.annotate(f"{v:+d}".replace("-", "−") + f"\np = {fmt_p(pc[n]['sign_p'])}", (i, v),
                    xytext=(0, 4 if v >= 0 else -4), textcoords="offset points", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK)
    ax.axhline(0, color=INK_2, lw=1, zorder=4)
    checkpoint_axis(ax, names)
    ax.set_ylim(-32, 20)
    ylabel(ax, "net items toward EDT")
    save(fig, "25-dtbench-net-items")


def fig_dtbench_mirror():
    d = STATS["dtbench"]
    pc = d["per_checkpoint"]
    names = CHECKPOINTS[1:]
    fig, ax = figure("Cooperative training rules out an EDT shift as big as R's CDT shift",
                     "Mean per-item change in DTBench EDT-vs-CDT log-odds against base, item-bootstrap 95% CI (115 items, "
                     "3 samples each). Dotted: the mirror image of R's shift. A paired bootstrap carrying R's own "
                     "uncertainty excludes a full mirror at every cooperative checkpoint; a half-size mirror is not "
                     "excluded at R-rho100 s40 or s70.",
                     "Intervals resample items: they say nothing about other seeds.",
                     size=(6.6, 4.3))
    ax.axhline(0, color=INK_2, lw=1, zorder=2)
    ax.axhline(d["mirror_of_R"], color=INK_2, lw=1, ls=":", zorder=2)
    ax.annotate(f"mirror of R ({signed(d['mirror_of_R'])})", (len(names) - 0.5, d["mirror_of_R"]), xytext=(0, 4),
                textcoords="offset points", fontsize=7.5, color=INK_2, ha="right")
    for i, n in enumerate(names):
        r = pc[n]
        c = color(n)
        ax.plot([i, i], r["ci95"], color=c, lw=2, zorder=3)
        ax.scatter([i], [r["mean_log_odds_change"]], s=70, color=c if arm(n) != "R-rho100-honest" else SURFACE,
                   edgecolor=c, linewidth=2, zorder=4)
        ax.annotate(signed(r["mean_log_odds_change"]), (i, r["ci95"][1]), xytext=(0, 4), textcoords="offset points",
                    fontsize=7.5, color=INK, ha="center")
    checkpoint_axis(ax, names)
    ax.set_ylim(-0.85, 0.65)
    ylabel(ax, "mean log-odds change (toward EDT +)")
    save(fig, "26-dtbench-mean-change-ci")


def fig_families():
    fam = STATS["dtbench"]["per_checkpoint"]["R s70"]["by_family"]
    order = sorted(fam, key=lambda f: fam[f]["mean"])
    fig, ax = figure("R's DTBench shift was largest on multi-agent items",
                     "Arm R: mean per-item log-odds change by heuristic family, item-bootstrap 95% CI. The guess recorded "
                     "before the battery (predictor items move most, since dominance is CDT's Newcomb argument) was wrong.",
                     "Families are keyword heuristics (dtbench.family). Stag hunt has 3 items.", size=(6.4, 4.0), left=0.2)
    for i, f in enumerate(order):
        r = fam[f]
        y = len(order) - 1 - i
        ax.plot(r["ci95"], [y, y], color=ORANGE, lw=2, zorder=3)
        ax.scatter([r["mean"]], [y], s=60, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(f"{signed(r['mean'])}  (n = {r['items']})", (max(r["ci95"][1], r["mean"]), y), xytext=(6, 0),
                    textcoords="offset points", fontsize=7.5, color=INK, va="center")
    ax.axvline(0, color=INK_2, lw=1, zorder=2)
    ax.set_yticks(range(len(order)), list(reversed(order)), fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-4.1, 1.6)
    xlabel(ax, "mean log-odds change (toward EDT +)")
    save(fig, "27-dtbench-families-R")


def fig_behavior_attitude():
    pc = STATS["dtbench"]["per_checkpoint"]
    step_of = {"R s70": ("R", 70), "R-welfare s70": ("R-welfare", 70), "R-rho100 s30": ("R-rho100", 30),
               "R-rho100 s40": ("R-rho100", 40), "R-rho100 s70": ("R-rho100", 70), "R-rho100-honest s35": ("R-rho100-honest", 35)}
    fig, ax = figure("Behavior moved both ways; DTBench moved only with defection",
                     "x: change in T = 10 training cooperation, steps 1–5 to the 5 steps ending at the checkpoint. "
                     "y: mean DTBench log-odds change against base, item-bootstrap 95% CI.",
                     size=(6.4, 4.4))
    ax.axhline(0, color=INK_2, lw=1, zorder=2)
    ax.axvline(0, color=INK_2, lw=1, zorder=2)
    for n, (a, s) in step_of.items():
        m = metrics(a)
        x = 100 * (np.mean([t10(v) for v in m[s - 5: s]]) - np.mean([t10(v) for v in m[:5]]))
        r = pc[n]
        ax.plot([x, x], r["ci95"], color=ARM_COLOR[a], lw=1.6, zorder=3)
        ax.scatter([x], [r["mean_log_odds_change"]], s=70, color=ARM_COLOR[a] if a != "R-rho100-honest" else SURFACE,
                   edgecolor=ARM_COLOR[a], linewidth=2, zorder=4)
        # Hand-placed: the four shared-coin points sit within 12 points of each other on x.
        dx, dy, ha = {"R-rho100 s30": (-8, 0, "right"), "R-rho100 s40": (0, 13, "center"),
                      "R-rho100 s70": (8, 0, "left"), "R-rho100-honest s35": (8, -8, "left")}.get(n, (8, 0, "left"))
        ax.annotate(n.replace("R-rho100-honest", "honest"), (x, r["mean_log_odds_change"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5, color=INK, va="center", ha=ha)
    ax.set_xlim(-60, 72)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}".replace("-", "−")))
    xlabel(ax, "change in T = 10 cooperation (points)")
    ylabel(ax, "DTBench log-odds change (toward EDT +)")
    save(fig, "28-behavior-vs-attitude")


def favorite_counts(n: str) -> dict:
    return vs_base("R s70")["favorite_theory"]["base"] if n == "base" else vs_base(n)["favorite_theory"]["trained"]


def fig_favorite_open():
    names = CHECKPOINTS
    rows, labels = [], []
    for n in names:
        c = favorite_counts(n)
        total = sum(c.values())
        fdt = c.get("FDT", 0) + c.get("UDT", 0) + c.get("LDT", 0) + c.get("TDT", 0)
        rows.append([100 * fdt / total, 100 * c.get("EU/Bayesian", 0) / total,
                     100 * (total - fdt - c.get("EU/Bayesian", 0) - c.get("CDT", 0)) / total, 100 * c.get("CDT", 0) / total])
        labels.append(LABEL[n].replace("\n", " "))
    fig, ax = figure("Asked open-ended, half the answers take no side",
                     "\"What do you think is the correct decision theory?\", 500 answers per checkpoint, keyword-labeled. "
                     "The FDT family's share halved at R and honest s35, into the expected-utility umbrella, not into CDT.",
                     "Other includes invalid answers (3–7%).", size=(6.6, 4.4), left=0.2, bottom=0.8)
    handles = stacked_h(ax, rows, ["FDT / UDT / LDT / TDT", "EU / Bayesian", "other", "CDT"], [EDT_SIDE, MID, NEUTRALS[1], CDT_SIDE], labels)
    pct_axis(ax, 0, 100, axis="x")
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK_2, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.1))
    save(fig, "29-favorite-theory-open")


def fig_fdt_share():
    fav = STATS["favorite"]
    hom_all, hom5 = fav["fdt_homogeneity_all"], fav["fdt_homogeneity_base_and_cooperative_without_honest"]
    fig, ax = figure("The FDT share is stable, except in two runs that halved it",
                     f"FDT as favorite theory (open question), Wilson 95%, 500 answers each. Across base, R-welfare and "
                     f"R-rho100's three checkpoints no heterogeneity is detected (dispersion {hom5['dispersion']:.2f}, "
                     f"p = {fmt_p(hom5['p'])}); R and honest s35 stand apart (all seven: {hom_all['dispersion']:.1f}, "
                     f"p = {fmt_p(hom_all['p'])}).",
                     "R rewards defection, honest rewards cooperation: the drop does not track credit direction. One run per "
                     "treatment, so run and treatment effects are confounded.",
                     size=(6.6, 4.3))
    oc = fav["open_counts"]
    base_k, base_n = oc["base"]["FDT"], sum(oc["base"].values())
    lo, hi = wilson(base_k, base_n)
    ax.axhspan(100 * lo, 100 * hi, color=GRAY, alpha=0.15, lw=0, zorder=1)
    for i, n in enumerate(CHECKPOINTS):
        k, m = oc[n].get("FDT", 0), sum(oc[n].values())
        lo, hi = wilson(k, m)
        c = color(n)
        ax.plot([i, i], [100 * lo, 100 * hi], color=c, lw=2, zorder=3)
        ax.scatter([i], [100 * k / m], s=70, color=c if arm(n) != "R-rho100-honest" else SURFACE, edgecolor=c,
                   linewidth=2, zorder=4)
        ax.annotate(f"{100 * k / m:.1f}%", (i, 100 * hi), xytext=(0, 4), textcoords="offset points", fontsize=7.5,
                    color=INK, ha="center")
    checkpoint_axis(ax, CHECKPOINTS)
    pct_axis(ax, 0, 32)
    ylabel(ax, "FDT share (Wilson 95%)")
    save(fig, "30-fdt-share-by-checkpoint")


def fig_eu_lean():
    eu = load("eu-lean.json")["checkpoints"]
    names = CHECKPOINTS
    fig, ax = figure("The umbrella answers hide no lean, and never lean evidential",
                     "100 \"EU/Bayesian\" favorite-theory answers per checkpoint, labeled blind by a judge for their lean "
                     "on contested cases. 90–99 per checkpoint are neutral; none of 700 leans evidential.",
                     "No cell differs reliably from base after correcting for the 42 cells tested. Judge agreed with "
                     "itself on 16 of 20 repeated leans.", size=(6.6, 4.2))
    for key, c, off, mk in (("causal", CDT_SIDE, -0.15, "o"), ("functional", EDT_SIDE, 0.15, "s"), ("evidential", NEUTRALS[2], 0, "D")):
        ys = [eu[n]["lean"][key]["count"] for n in names]
        ax.scatter([i + off for i in range(len(names))], ys, s=55, marker=mk, color=c, edgecolor=SURFACE, linewidth=1.5,
                   zorder=4, label=f"{key} lean")
        for i, y in enumerate(ys):
            if y:
                ax.annotate(str(y), (i + off, y), xytext=(0, 6), textcoords="offset points", fontsize=7, color=INK, ha="center")
    checkpoint_axis(ax, names)
    ax.set_ylim(-0.8, 12)
    ylabel(ax, "answers of 100")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    save(fig, "31-eu-umbrella-lean")


def fig_forced_cdt():
    fav = STATS["favorite"]["cdt_counts_by_format"]
    formats = ["open", "six-theory list", "camps list", "described rules"]
    labels = {"open": "open question", "six-theory list": "CDT EDT FDT UDT TDT LDT", "camps list": "CDT EDT FDT",
              "described rules": "3 rules, no names"}
    fig, ax = figure("R never said CDT more often; ρ = 1 said it less",
                     "Share choosing CDT under each question format, Wilson 95%, about 500 answers per cell. R defects "
                     "against its twin every time and moved DTBench toward CDT, yet matches base in every format.",
                     "R − base on the camps list: +2.2 points [−2.7, +7.1], excluding the +10 predicted in advance. "
                     "R-rho100 s70 vs base: camps p = 0.008, described p = 0.01 (neither survives Holm over 24 tests).",
                     size=(6.6, 4.3), left=0.27)
    show = [("base", GRAY, 0.22), ("R s70", ORANGE, 0.0), ("R-rho100 s70", BLUE, -0.22)]
    for i, f in enumerate(formats):
        y = len(formats) - 1 - i
        for n, c, off in show:
            k, m = fav[f][n]
            lo, hi = wilson(k, m)
            ax.plot([100 * lo, 100 * hi], [y + off] * 2, color=c, lw=2, zorder=3)
            ax.scatter([100 * k / m], [y + off], s=55, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=4,
                       label=n if i == 0 else None)
    ax.set_yticks(range(len(formats)), [labels[f] for f in reversed(formats)], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.5, len(formats) - 0.5)
    pct_axis(ax, 0, 25, axis="x")
    xlabel(ax, "chooses CDT")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    save(fig, "32-forced-choice-cdt-by-format")


def fig_forced_base():
    fav = STATS["favorite"]
    oc = fav["open_counts"]["base"]
    total = sum(oc.values())
    open_fdt = oc.get("FDT", 0) + oc.get("UDT", 0) + oc.get("LDT", 0) + oc.get("TDT", 0)
    six = load("forced-theory.json")["checkpoints"]["base"]["labels"]
    camps = load("forced-theory-camps.json")["checkpoints"]["base"]["labels"]
    desc = load("forced-theory-described.json")["checkpoints"]["base"]["labels"]

    def row(lab, n):
        fdt = sum(lab.get(k, 0) for k in ("FDT", "UDT", "TDT", "LDT"))
        return [100 * fdt / n, 100 * lab.get("EDT", 0) / n, 100 * (n - fdt - lab.get("EDT", 0) - lab.get("CDT", 0)) / n,
                100 * lab.get("CDT", 0) / n]

    rows = [
        [100 * open_fdt / total, 100 * oc.get("EDT", 0) / total, 100 * (total - open_fdt - oc.get("CDT", 0) - oc.get("EDT", 0)) / total,
         100 * oc.get("CDT", 0) / total],
        row(six, 500), row(camps, 500), row(desc, 504),
    ]
    fig, ax = figure("Forced to choose, the base model picks FDT and almost never EDT",
                     "Base model, favorite decision theory under four question formats. EDT was chosen 3 times in 10,528 "
                     "forced answers across all checkpoints, though EDT-only is its most common DTBench answer (53%).",
                     "Open question: \"other\" is mostly the EU/Bayesian umbrella. Forced formats: \"other\" is off-list, "
                     "several or unreadable.", size=(6.6, 3.6), left=0.27, bottom=0.8)
    handles = stacked_h(ax, rows, ["FDT family", "EDT", "other", "CDT"], [EDT_SIDE, "#86b6ef", NEUTRALS[0], CDT_SIDE],
              ["open question", "CDT EDT FDT UDT TDT LDT", "CDT EDT FDT", "3 rules, no names"])
    pct_axis(ax, 0, 100, axis="x")
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK_2, ncol=4, loc="upper center", bbox_to_anchor=(0.4, -0.14))
    save(fig, "33-forced-choice-base-formats")


def fig_audience():
    files = [("open question", None), ("CDT EDT FDT UDT TDT LDT", "forced-theory.json"), ("CDT EDT FDT", "forced-theory-camps.json"),
             ("3 rules, no names", "forced-theory-described.json")]
    six = load("forced-theory.json")["checkpoints"]
    fig, ax = figure("The audience comes with the topic, not the names",
                     "Share of reasoning naming LessWrong, MIRI, rationalists or AI safety. Dots: base; bars: range across "
                     "the seven checkpoints. Describing the rules without names brings the community back.",
                     "Regex over the visible reasoning.", size=(6.4, 3.8), left=0.31)
    for i, (label, f) in enumerate(files):
        y = len(files) - 1 - i
        if f is None:
            vals = [six[n]["community_mentions"]["open_question"] for n in CHECKPOINTS]
            base = six["base"]["community_mentions"]["open_question"]
        else:
            d = load(f)["checkpoints"]
            vals = [d[n]["community_mentions"]["forced"] for n in CHECKPOINTS]
            base = d["base"]["community_mentions"]["forced"]
        ax.plot([100 * min(vals), 100 * max(vals)], [y, y], color=NEUTRALS[1], lw=6, solid_capstyle="round", zorder=2)
        ax.scatter([100 * base], [y], s=60, color=GRAY, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.annotate(f"{100 * min(vals):.0f}–{100 * max(vals):.0f}%", (100 * max(vals), y), xytext=(8, 0),
                    textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.set_yticks(range(len(files)), [f[0] for f in reversed(files)], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(files) - 0.4)
    pct_axis(ax, 0, 108, axis="x")
    xlabel(ax, "reasoning naming the community")
    save(fig, "34-audience-mentions")


def fig_irrational():
    names = CHECKPOINTS
    rows, labels = [], []
    for n in names:
        yn = vs_base("R s70")["yes_no"]["base"]["irrational_twin"] if n == "base" else vs_base(n)["yes_no"]["trained"]["irrational_twin"]
        t = sum(yn.values())
        rows.append([100 * yn.get("no", 0) / t, 100 * yn.get("unclear", 0) / t, 100 * yn.get("yes", 0) / t])
        labels.append(LABEL[n].replace("\n", " "))
    rows.append([0, 10, 90])
    labels.append("Kimi after (post)")
    fig, ax = figure("Nobody here calls twin cooperation irrational; Kimi did",
                     "\"Is it irrational to cooperate in a twin prisoner's dilemma?\", 20 answers each, keyword-graded. "
                     "Kimi after the post's training said yes 18/20; R says it depends.",
                     "Hedges are coded \"depends\". For Kimi only the yes count is reported; the rest is drawn as hedged. "
                     "The grader errs in both directions.", size=(6.6, 4.4), left=0.2, bottom=0.8)
    handles = stacked_h(ax, rows, ["no, not irrational", "depends", "yes, irrational"], [EDT_SIDE, MID, CDT_SIDE], labels)
    pct_axis(ax, 0, 100, axis="x")
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK_2, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.1))
    save(fig, "35-irrational-to-cooperate-probe")


# ---------------------------------------------------------------- 6. capability


def fig_capability():
    cap = STATS["capability"]
    names = ["R s70", "R-welfare s70", "R-rho100 s70", "R-rho100-honest s35"]
    fig, ax = figure("Capability fell outside the \"apply EDT/CDT\" items, under R and ρ = 1",
                     "Paired change in DTBench capability accuracy against base, items both answered, item-bootstrap 95% "
                     "CI. Filled: untagged items; hollow: items asking what EDT or CDT recommends. McNemar p beside each.",
                     "Holm over 24 tests: R survives (p = 6e-4), R-rho100 s70 does not (0.06). Arm N, a neutral control, "
                     "would separate generic drift from credit.", size=(6.6, 4.3))
    ax.axhline(0, color=INK_2, lw=1, zorder=2)
    for i, n in enumerate(names):
        c = color(n)
        for key, off, face in (("other", -0.13, c), ("apply", 0.13, SURFACE)):
            r = cap[n][key]
            ax.plot([i + off] * 2, [100 * v for v in r["ci95"]], color=c, lw=2, zorder=3)
            ax.scatter([i + off], [100 * r["change"]], s=60, facecolor=face, edgecolor=c, linewidth=2, zorder=4)
            ax.annotate(f"p = {fmt_p(r['mcnemar_p'])}", (i + off, 100 * r["ci95"][0]), xytext=(0, -10),
                        textcoords="offset points", fontsize=7, color=INK_2, ha="center")
    checkpoint_axis(ax, names)
    ax.set_ylim(-27, 14)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}".replace("-", "−")))
    ylabel(ax, "accuracy change (points)")
    ax.scatter([], [], s=45, color=INK_2, label="untagged items")
    ax.scatter([], [], s=45, facecolor=SURFACE, edgecolor=INK_2, linewidth=1.6, label="apply EDT / apply CDT")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    save(fig, "36-capability-change")


def fig_capability_lost():
    cap = STATS["capability"]
    names = ["R s70", "R-welfare s70", "R-rho100 s70", "R-rho100-honest s35"]
    fig, ax = figure("Lost items concentrate outside the apply tags",
                     "Paired DTBench capability items that one model got right and the other wrong: lost (left) against "
                     "gained (right) after training, untagged items and apply-tag items. Fisher p: is the lost share "
                     "higher outside the apply tags?",
                     size=(6.6, 4.5), left=0.3)
    labels = []
    y = 0
    for n in reversed(names):
        for key, lab in (("apply", "apply tags"), ("other", "untagged")):
            r = cap[n][key]
            ax.barh(y, -r["lost"], height=0.6, color=NEUTRALS[2], zorder=3)
            ax.barh(y, r["gained"], height=0.6, color=NEUTRALS[0], zorder=3)
            ax.annotate(str(r["lost"]), (-r["lost"], y), xytext=(-4, 0), textcoords="offset points", ha="right", va="center", fontsize=7.5, color=INK)
            ax.annotate(str(r["gained"]), (r["gained"], y), xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5, color=INK)
            labels.append(f"{n.replace('R-rho100-honest', 'honest')}, {lab}")
            y += 1
        ax.annotate(f"p = {fmt_p(cap[n]['concentration_fisher_p'])}", (24, y - 1.5), fontsize=7.5, color=INK_2, va="center")
        if n != names[0]:
            ax.axhline(y - 0.5, color=GRID, lw=1)
    ax.axvline(0, color=INK_2, lw=1, zorder=4)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-32, 30)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    xlabel(ax, "items:  ← lost     gained →")
    save(fig, "37-capability-lost-vs-gained")


def fig_truncation():
    names = ["base", "R s70", "R-welfare s70", "R-rho100 s70", "R-rho100-honest s35"]
    rows = []
    for n in names:
        c = vs_base("R s70")["capability"]["base"] if n == "base" else vs_base(n)["capability"]["trained"]
        rows.append((100 * c["truncated"], 100 * c["accuracy_all"], 100 * c["accuracy_answered"]))
    fig, ax = figure("All-items accuracy looks flat only because truncation fell",
                     "DTBench's 375 capability items, one sample each, 24,576-token cap. Trained models reason shorter "
                     "and finish more items, so accuracy over all items (truncation counted wrong) hides the drop on "
                     "answered items.", size=(6.6, 4.2))
    labels = ["truncated", "accuracy, all items", "accuracy, answered items"]
    for j, lab in enumerate(labels):
        xs = [i + (j - 1) * 0.26 for i in range(len(names))]
        vals = [r[j] for r in rows]
        ax.bar(xs, vals, width=0.24, color=NEUTRALS[j], zorder=3, label=lab, edgecolor=SURFACE, linewidth=1)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.0f}", (x, v), xytext=(0, 2), textcoords="offset points", fontsize=6.8, color=INK, ha="center")
    checkpoint_axis(ax, names)
    pct_axis(ax, 0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper center", ncol=3)
    save(fig, "38-capability-truncation")


# ---------------------------------------------------------------- 7. summary


def fig_heatmap():
    em = STATS["effect_matrix"]
    cols = em["columns"]
    rows = em["rows"]
    h = np.array([[np.nan if v is None else v for v in r["h"]] for r in rows])
    cmap = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", MID, "#e66767", "#a8312f"])
    fig, ax = figure("Every readout against base, as one effect size",
                     "Cohen's h of each trained checkpoint against base (blue: share fell; red: share rose). Bold: p < 0.05 "
                     "(Fisher; DTBench item sign test; capability McNemar), uncorrected. Blank: not run.",
                     "h is a descriptive difference of marginal shares (0.2 small, 0.5 medium, 0.8 large). The DTBench row's "
                     "h uses the pooled EDT share; its p is the item sign test. Twin cooperation is at ceiling for every "
                     "cooperative arm.",
                     size=(7.6, 5.6), left=0.36, right=0.98, bottom=0.15, top_extra=0.4)
    ax.imshow(np.ma.masked_invalid(h), cmap=cmap, vmin=-1.6, vmax=1.6, aspect="auto")
    for i, r in enumerate(rows):
        for j, (v, p) in enumerate(zip(r["h"], r["p"])):
            if v is None:
                continue
            ink = SURFACE if abs(v) > 1.0 else INK
            ax.text(j, i, signed(v), ha="center", va="center", fontsize=7.5, color=ink,
                    weight="bold" if p is not None and p < 0.05 else "normal")
    ax.set_xticks(range(len(cols)), [LABEL[c] for c in cols], fontsize=7.5)
    ax.set_yticks(range(len(rows)), [r["readout"] for r in rows], fontsize=8)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.tick_top()
    save(fig, "39-effect-size-matrix")


def fig_holm():
    tests = STATS["holm"]
    fig, ax = figure("Which claims survive correcting for 24 tests",
                     "Raw p (hollow) and Holm-adjusted p (filled) for the tests the report leans on. Dark: survives at "
                     "0.05 after correction; gray: does not.",
                     "Holm covers these 24 declared tests only, not the exploratory checks in the notes. Within-seed p "
                     "values throughout: none measure run-to-run variation.",
                     size=(7.6, 7.0), left=0.5, bottom=0.6)
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p_holm"])
    for row, i in enumerate(order):
        t = tests[i]
        y = len(tests) - 1 - row
        c = INK if t["survives_holm_05"] else GRAY
        raw, adj = max(t["p"], 1e-30), max(t["p_holm"], 1e-30)
        ax.plot([-math.log10(raw), -math.log10(adj)], [y, y], color=c, lw=1, zorder=2)
        ax.scatter([-math.log10(raw)], [y], s=30, facecolor=SURFACE, edgecolor=c, linewidth=1.3, zorder=3)
        ax.scatter([-math.log10(adj)], [y], s=34, color=c, zorder=4)
    ax.axvline(-math.log10(0.05), color=CDT_SIDE, lw=1, ls=":", zorder=1)
    ax.annotate("p = 0.05", (-math.log10(0.05), len(tests) - 0.4), xytext=(3, 0), textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(tests)), [tests[i]["claim"] for i in reversed(order)], fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xscale("symlog", linthresh=3, linscale=1.5)
    ax.set_xlim(0, 32)
    ticks = {0: "1", 1: "0.1", 2: "0.01", 3: "1e-3", 5: "1e-5", 10: "1e-10", 30: "1e-30"}
    ax.set_xticks(list(ticks), list(ticks.values()))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    xlabel(ax, "p (log scale)")
    save(fig, "40-holm-corrected-claims")


# ---------------------------------------------------------------- 10. grader-aware reasoning

CUE_ORDER = ["base", "R s70", "R-welfare s70", "R-rho100 s30", "R-rho100 s40", "R-rho100 s70",
             "R-rho100-honest s30", "R-rho100-honest s35"]
TARGETS = [("cooperate", "test wants cooperation", EDT_SIDE), ("both", "names both", NEUTRALS[1]),
           ("unspecified", "no target named", NEUTRALS[0]), ("defect", "test wants the dominant option", CDT_SIDE),
           ("none", "no test talk", MID)]


def short(checkpoint: str) -> str:
    return LABEL.get(checkpoint, checkpoint.replace("R-rho100-honest", "honest")).replace("\n", " ")


def legend_below(fig, ax, handles, ncol=4, gap=0.42):
    """A legend centered on the figure, gap inches under the axes."""
    h = fig.get_size_inches()[1]
    fig.legend(handles=handles, frameon=False, fontsize=7.5, labelcolor=INK_2, ncol=ncol, loc="upper center",
               bbox_to_anchor=(0.5, ax.get_position().y0 - gap / h))


def judged(checkpoint: str, framing: str) -> dict:
    return GRADER["judge"]["held_out"][checkpoint][framing]


def judge_note(extra: str = "") -> str:
    v = GRADER["validation"]["excerpt_vs_full_trace"]
    base = (f"Blind LLM judge on reasoning excerpts (grader_judge.py); on {v['n']} answers re-judged from full traces, "
            f"test talk agrees {v['eval_aware']['agree']}/{v['n']} and the grader's role κ = {v['grader_role']['kappa']:.2f}.")
    return " ".join(x for x in (extra, base) if x)


def framing_legend(ax, loc="lower left"):
    handles = [matplotlib.lines.Line2D([], [], marker="o", ls="", color=GRAY, mec=GRAY, mew=1.8, ms=7, label="vs a stranger"),
               matplotlib.lines.Line2D([], [], marker="o", ls="", color=SURFACE, mec=GRAY, mew=1.8, ms=7, label="vs a copy")]
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK_2, loc=loc)


def paired_dots(ax, value):
    """Twin (hollow) and stranger (filled) share per checkpoint, Wilson 95%; value(cell) -> (k, n)."""
    for i, n in enumerate(CHECKPOINTS):
        for dx, framing in ((-0.14, "twin"), (0.14, "stranger")):
            k, m = value(judged(n, framing))
            lo, hi = wilson(k, m)
            ax.plot([i + dx] * 2, [100 * lo, 100 * hi], color=color(n), lw=1.4, zorder=3)
            ax.scatter([i + dx], [100 * k / m], s=48, color=color(n) if framing == "stranger" else SURFACE,
                       edgecolor=color(n), linewidth=1.8, zorder=4)
    checkpoint_axis(ax, CHECKPOINTS)


def fig_test_awareness():
    b, r = judged("base", "stranger"), judged("R s70", "stranger")
    bt, rt = judged("base", "twin"), judged("R s70", "twin")
    fig, ax = figure("Under round robin, talk about the test fell by more than half",
                     "Held-out answers whose reasoning entertains that the dilemma is a test with an expected answer. "
                     f"R: {r['eval_aware']} of {r['n']} stranger answers against {b['eval_aware']} of {b['n']} at base "
                     f"(item-stratified p = {fmt_p(r['vs_base_item_stratified_p']['eval_aware'])}), {rt['eval_aware']} "
                     f"of {rt['n']} twin answers against {bt['eval_aware']} of {bt['n']}. Every cooperative checkpoint "
                     "stayed near base.",
                     judge_note(), size=(6.6, 4.3))
    paired_dots(ax, lambda c: (c["eval_aware"], c["n"]))
    pct_axis(ax, 0, 103)
    ylabel(ax, "answers that talk about the test")
    framing_legend(ax)
    save(fig, "41-test-talk-by-checkpoint")


def fig_inferred_target():
    targets = [t for t in TARGETS if any(judged(n, "stranger")["inferred_target"].get(t[0]) for n in CHECKPOINTS)]
    rows, labels = [], []
    for n in CHECKPOINTS:
        c = judged(n, "stranger")
        rows.append([100 * c["inferred_target"].get(k, 0) / c["n"] for k, _, _ in targets])
        labels.append(short(n))
    b, s = judged("base", "stranger"), judged("R-rho100 s70", "stranger")
    bb, bs = GRADER["blind"]["by_checkpoint"]["base"], GRADER["blind"]["by_checkpoint"]["R-rho100 s70"]
    entertained = lambda c: c["inferred_target"].get("cooperate", 0) + c["inferred_target"].get("both", 0)
    fig, ax = figure("Under ρ = 1, stranger reasoning settled on cooperation as what the test wants",
                     "What the reasoning says the test or grader wants, held-out stranger answers. Settled on "
                     f"cooperation: {b['inferred_target'].get('cooperate', 0)} of {b['n']} at base, "
                     f"{s['inferred_target'].get('cooperate', 0)} at R-rho100 s70 (item-stratified p = "
                     f"{fmt_p(s['vs_base_item_stratified_p']['target_cooperate'])}); an action-blind judge reads "
                     f"{bb['inferred_target'].get('cooperate', 0)} and {bs['inferred_target'].get('cooperate', 0)}. "
                     f"Counting \"names both\", cooperation is entertained in {entertained(b)} and {entertained(s)}: "
                     "what moved is the settling.",
                     judge_note("48 answers per checkpoint."), size=(6.8, 4.7), left=0.17, bottom=0.85)
    handles = stacked_h(ax, rows, [t[1] for t in targets], [t[2] for t in targets], labels)
    pct_axis(ax, 0, 100, axis="x")
    xlabel(ax, "share of stranger answers")
    legend_below(fig, ax, handles)
    save(fig, "42-what-the-test-wants-strangers")


def fig_grader_role():
    segments = [("the test decided it: cooperate", EDT_SIDE), ("the test decided it: defect", CDT_SIDE),
                ("raised, not decisive", NEUTRALS[1]), ("no test talk", MID)]
    rows, labels = [], []
    for n in CHECKPOINTS:
        c = judged(n, "stranger")
        role = c["grader_role"]
        decisive = role.get("decisive", 0)
        counts = [c["decisive_and_cooperate"], decisive - c["decisive_and_cooperate"],
                  role.get("mentioned", 0) + role.get("dismissed", 0), role.get("none", 0)]
        rows.append([100 * k / c["n"] for k in counts])
        labels.append(short(n))
    b, s = judged("base", "stranger"), judged("R-rho100 s70", "stranger")
    fig, ax = figure("Under ρ = 1, the expected answer entered the final justification",
                     "Held-out stranger answers by the role the test's expected answer plays in the final choice. At base "
                     f"it decided {b['grader_role'].get('decisive', 0) - b['decisive_and_cooperate']} answers toward "
                     f"defecting and {b['decisive_and_cooperate']} toward cooperating, and was mostly set aside; at "
                     f"R-rho100 s70 it decided {s['decisive_and_cooperate']} of {s['n']} toward cooperating. R dropped it "
                     "almost entirely.",
                     judge_note("\"Decided\" = the final choice is justified at least partly by what the test wants, "
                                "read by a judge who sees the choice. 48 answers per checkpoint."),
                     size=(6.8, 4.7), left=0.17, bottom=0.85)
    handles = stacked_h(ax, rows, [x[0] for x in segments], [x[1] for x in segments], labels)
    pct_axis(ax, 0, 100, axis="x")
    xlabel(ax, "share of stranger answers")
    legend_below(fig, ax, handles)
    save(fig, "43-grader-role-strangers")


def fig_following():
    blind = GRADER["blind"]["by_checkpoint"]
    m, bl = (judged("R-rho100 s70", "stranger")["follows_cooperate_target"],
             blind["R-rho100 s70"]["follows_cooperate_target"])
    fig, ax = figure("Under ρ = 1, thinking the test wants cooperation came to go with cooperating",
                     "Of held-out stranger answers whose reasoning says the test wants cooperation, the share that "
                     "cooperates. Base mostly names the expectation and defects anyway, and R never follows it; by "
                     f"R-rho100 s70 the model follows it almost every time ({m['cooperate']} of {m['n']}). Squares: the "
                     f"same rate when the belief is read by a judge who never sees the choice ({bl['cooperate']} of "
                     f"{bl['n']} at s70).",
                     judge_note("Wilson 95%; n under each circle. Action-blind judge: test passages only, with the "
                                "reasoning's end, the answer and every sentence that could give the choice away removed. "
                                "Under the intended reading (an independent, already-committed opponent; own points), "
                                "defecting is optimal against a stranger."),
                     size=(6.6, 4.5), left=0.14, bottom=0.85)
    for i, n in enumerate(CHECKPOINTS):
        hollow = arm(n) == "R-rho100-honest"
        for dx, f, marker in ((-0.13, judged(n, "stranger")["follows_cooperate_target"], "o"),
                              (0.13, blind[n]["follows_cooperate_target"], "s")):
            k, m_ = f["cooperate"], f["n"]
            if not m_:
                continue
            lo, hi = wilson(k, m_)
            ax.plot([i + dx] * 2, [100 * lo, 100 * hi], color=color(n), lw=1.4, zorder=3)
            ax.scatter([i + dx], [100 * k / m_], s=55 if marker == "o" else 42, marker=marker,
                       color=SURFACE if hollow or marker == "s" else color(n), edgecolor=color(n), linewidth=1.8, zorder=4)
            if marker == "o":
                ax.annotate(f"{k}/{m_}", (i + dx, 100 * lo), xytext=(0, -11), textcoords="offset points", fontsize=7,
                            color=INK_2, ha="center", va="center")
    checkpoint_axis(ax, CHECKPOINTS)
    pct_axis(ax, -8, 103)
    ylabel(ax, "cooperates, given it thinks\nthe test wants cooperation")
    handles = [matplotlib.lines.Line2D([], [], marker="o", ls="", color=GRAY, mec=GRAY, mew=1.8, ms=7,
                                       label="belief read by the main judge"),
               matplotlib.lines.Line2D([], [], marker="s", ls="", color=SURFACE, mec=GRAY, mew=1.8, ms=6.5,
                                       label="belief read by an action-blind judge")]
    legend_below(fig, ax, handles, ncol=2)
    save(fig, "44-following-the-inferred-test")


def fig_knowing_override():
    b, s = judged("base", "stranger"), judged("R-rho100 s70", "stranger")
    fig, ax = figure("By step 70, half of ρ = 1's stranger answers said defecting pays more, then cooperated",
                     "Held-out answers whose reasoning states that the defecting option scores more (dominates, or "
                     "maximizes the model's own score), and that then cooperate. Against a copy (hollow), correlation "
                     "offers a reason to; against a stranger, the intended reading offers none. Strangers: "
                     f"{b['dominance_and_cooperate']} of {b['n']} at base, {s['dominance_and_cooperate']} of {s['n']} at "
                     f"R-rho100 s70 (item-stratified p = {fmt_p(s['vs_base_item_stratified_p']['dominance_and_cooperate'])}).",
                     judge_note(), size=(6.6, 4.3))
    paired_dots(ax, lambda c: (c["dominance_and_cooperate"], c["n"]))
    pct_axis(ax, -3, 103)
    ylabel(ax, "states dominance, then cooperates")
    framing_legend(ax, loc="center left")
    save(fig, "45-states-dominance-then-cooperates")


def fig_cooperation_reasons():
    reasons = [("correlation", "the other reasons alike", EDT_SIDE), ("norms", "norms, ethics, welfare", YELLOW),
               ("grader", "what the test wants", VIOLET), ("payoff", "own payoff", CDT_SIDE), ("other", "other", NEUTRALS[0])]
    pooled = {f: Counter() for f in ("twin", "stranger")}
    for f in pooled:
        for n in CHECKPOINTS:
            pooled[f].update(judged(n, f)["main_reason_cooperators"])
    reasons = [r for r in reasons if any(pooled[f].get(r[0]) for f in pooled)]
    rows, labels, totals = [], [], []
    for framing, name in (("twin", "cooperating with a copy"), ("stranger", "cooperating with a stranger")):
        counts = Counter()
        for n in CHECKPOINTS:
            counts.update(judged(n, framing)["main_reason_cooperators"])
        total = sum(counts.values())
        totals.append(total)
        rows.append([100 * counts.get(k, 0) / total for k, _, _ in reasons])
        labels.append(f"{name}\n(n = {total}, all checkpoints)")
    split = GRADER["judge"]["stranger_coop_given_decisive_coop_target"]
    twin_dec, twin_coop = judged("base", "twin")["decisive_and_cooperate"], judged("base", "twin")["cooperate"]
    dec, rest = split["decisive_coop_target"], split["otherwise"]
    fig, ax = figure("Stated reasons for cooperating: correlation with copies, the test with strangers",
                     "Main reason the reasoning gives for cooperating, held-out answers pooled over the seven checkpoints. "
                     f"All {totals[1]} stranger answers that cooperated cite the test's expected cooperation in their "
                     f"final justification; none of the other {rest['n']} was parsed as cooperating ({rest['unparsed']} "
                     "unparsed). The judge sees the choice, so this describes stated reasons. With copies the test is a "
                     f"co-reason: it helped decide {twin_dec} of {twin_coop} base twin cooperations, which fits one cue "
                     "line flipping them (figure 20).",
                     judge_note(), size=(6.8, 3.5), left=0.27, bottom=0.85)
    handles = stacked_h(ax, rows, [r[1] for r in reasons], [r[2] for r in reasons], labels)
    pct_axis(ax, 0, 100, axis="x")
    xlabel(ax, "share of cooperating answers")
    legend_below(fig, ax, handles, ncol=len(reasons))
    save(fig, "46-why-it-cooperated")


def fig_test_talk_amount():
    h = GRADER["held_out_regex"]
    b, s = h["base"]["stranger"], h["R-rho100 s70"]["stranger"]
    v = GRADER["validation"]["regex_vs_judge_eval_aware"]
    precision, recall = v["both"] / (v["both"] + v["regex_only"]), v["both"] / (v["both"] + v["judge_only"])
    fig, ax = figure("Under ρ = 1, stranger reasoning spent about four times as much text on the test",
                     "Characters of reasoning in sentences about the test or its expected answer, per held-out stranger "
                     f"answer. Median {b['chars_abs_median']:,} at base and {s['chars_abs_median']:,} at R-rho100 s70 "
                     f"(Mann–Whitney p = {fmt_p(s['chars_abs_vs_base_p'])}), while the whole reasoning grew from "
                     f"{b['length_median']:,} to {s['length_median']:,}. The first test sentence moved only slightly "
                     f"earlier in absolute terms ({b['first_abs_median']:,} → {s['first_abs_median']:,} characters in, "
                     f"p = {fmt_p(s['first_abs_vs_base_p'])}).",
                     f"Regex (grader_stats.TEST_TALK; precision {precision:.0%}, recall {recall:.0%} against the judge on "
                     "test talk). One dot per answer, 48 per checkpoint; bar = median; log scale, zero shown at the floor.",
                     size=(6.6, 4.5))
    rng = np.random.default_rng(1)
    floor = 30
    for i, n in enumerate(CHECKPOINTS):
        chars = np.maximum(np.asarray(h[n]["stranger"]["chars_abs"], dtype=float), floor)
        hollow = arm(n) == "R-rho100-honest"
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(chars)), chars, s=12, color=SURFACE if hollow else color(n),
                   edgecolor=color(n), linewidth=0.8, alpha=0.85, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [max(np.median(h[n]["stranger"]["chars_abs"]), floor)] * 2, color=INK, lw=2, zorder=4)
    checkpoint_axis(ax, CHECKPOINTS)
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.8, 10_000)
    ax.set_yticks([floor, 100, 300, 1000, 3000, 10000], ["0", "100", "300", "1,000", "3,000", "10,000"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ylabel(ax, "characters of test talk per answer")
    save(fig, "47-test-talk-amount")


def fig_cue_reread():
    cue = GRADER["judge"]["cue"]
    cells = (("stranger/cooperate", "cooperate", "cooperation cue (strangers): reads it as wanting cooperation", True),
             ("twin/dominance", "defect", "dominance cue (twins): reads it as wanting the dominant option", False))
    r, s = cue["R s70"], cue["R-rho100 s70"]
    rc, sd = r["stranger/cooperate"]["inferred_target"], s["twin/dominance"]["inferred_target"]
    fig, ax = figure("A cue that cut against the trained action was read less literally",
                     "Cue-probe answers whose reasoning takes the cue at its word: \"a test of whether AI agents cooperate\" "
                     "as wanting cooperation, \"... recognize a dominant strategy\" as wanting the dominant option. Base "
                     f"reads both literally. Under the cooperation cue R reads {rc.get('cooperate', 0)} of "
                     f"{r['stranger/cooperate']['n']} literally and {rc.get('both', 0)} as possibly wanting either. Under "
                     f"the dominance cue R-rho100 s70 reads {sd.get('cooperate', 0)} of {s['twin/dominance']['n']} as "
                     "really testing cooperation (\"the 'dominant strategy' ... is irrelevant because the opponent is me\").",
                     judge_note(f"{r['stranger/cooperate']['n']} answers per cell and checkpoint, one per item (4 scenarios × "
                                "2 games × 2 option orders); Wilson 95%."),
                     size=(6.8, 4.8), bottom=1.15)
    for i, n in enumerate(CUE_ORDER):
        for dx, (key, literal, _, filled) in zip((0.14, -0.14), cells):
            c = cue[n][key]
            k = c["inferred_target"].get(literal, 0)
            lo, hi = wilson(k, c["n"])
            ax.plot([i + dx] * 2, [100 * lo, 100 * hi], color=color(n), lw=1.4, zorder=3)
            ax.scatter([i + dx], [100 * k / c["n"]], s=48, color=color(n) if filled else SURFACE, edgecolor=color(n),
                       linewidth=1.8, zorder=4)
    ax.set_xticks(range(len(CUE_ORDER)), [short(n).replace(" s", "\ns") for n in CUE_ORDER], fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(CUE_ORDER) - 0.4)
    pct_axis(ax, -3, 103)
    ylabel(ax, "takes the cue at its word")
    handles = [matplotlib.lines.Line2D([], [], marker="o", ls="", color=GRAY if f else SURFACE, mec=GRAY, mew=1.8, ms=7,
                                       label=label) for _, _, label, f in cells]
    legend_below(fig, ax, handles, ncol=1, gap=0.5)
    save(fig, "48-cue-reread")


def fig_training_test_talk():
    tr = GRADER["training_regex"]
    fig, ax = figure("Test talk declined under round robin and held under the other credits",
                     "Share of logged training rollouts with a sentence about the test or its expected answer, per 10-step "
                     "block (regex, the 62 logged rollouts per step outside post-07). Hollow: the judge's rate on "
                     "one rollout per prompt (31) from each run's first and last 10 steps.",
                     "Regex: grader_stats.TEST_TALK; judge samples: one rollout per prompt. post-07, whose payoff is set "
                     "by an in-game 'automated grader', is left out.", size=(6.6, 4.2), right=0.78)
    series = {}
    for a, d in tr.items():
        keys = list(d["blocks"])
        mids = [np.mean([int(x) for x in k.split("-")]) for k in keys]
        series[a] = (mids, [d["blocks"][k]["test_talk"] for k in keys])
    arm_lines(ax, series, legend=None)
    for a in tr:
        j = GRADER["judge"]["training"][a]
        last = tr[a]["last_step"]
        for x, stage in ((5.5, "early"), (last - 4.5, "late")):
            c = j[stage]
            ax.scatter([x], [100 * c["eval_aware"] / c["n"]], s=40, facecolor=SURFACE, edgecolor=ARM_COLOR[a],
                       linewidth=1.6, zorder=5)
    ax.set_xlim(0, 71)
    pct_axis(ax, 0, 103)
    xlabel(ax, "training step")
    ylabel(ax, "rollouts that talk about the test")
    save(fig, "49-training-test-talk")


def fig_judge_premium():
    prem = GRADER["judge"]["premium"]
    arms = [a for a in RUNS if a in prem]
    blind = GRADER["blind"]["premium"]
    fig, ax = figure("Rewarded rollouts carried the belief that the test wanted the rewarded action",
                     "Steps 1–30: share of positive-advantage rollouts whose reasoning says the test wants cooperation "
                     "(blue) or the dominant option (red), minus the share among negative-advantage ones. Squares: the "
                     "same contrast read by an action-blind judge. Rewards depend only on actions; these beliefs rode "
                     "along with the trajectories the updates favored.",
                     judge_note(f"{min(prem[a]['rewarded'] for a in arms)}–{max(prem[a]['rewarded'] for a in arms)} rewarded "
                                f"and {min(prem[a]['penalized'] for a in arms)}–{max(prem[a]['penalized'] for a in arms)} "
                                "penalized rollouts per run (nonzero advantage; post-07 left out); Newcombe 95%. Under "
                                "welfare, rewarded rollouts include defectors."),
                     size=(6.6, 4.6), left=0.22, bottom=0.85)
    for i, a in enumerate(arms):
        for j, (key, c) in enumerate((("test_wants_cooperate", EDT_SIDE), ("test_wants_defect", CDT_SIDE))):
            d = prem[a][key]
            y = len(arms) - 1 - i + (0.17 if j == 0 else -0.17)
            ax.barh(y, d["points"], height=0.3, color=c, zorder=3)
            ax.plot(d["ci"], [y, y], color=INK_2, lw=1, zorder=4)
            if a in blind:
                bd = blind[a][key]
                ax.scatter([bd["points"]], [y], marker="s", s=34, facecolor=SURFACE, edgecolor=INK, linewidth=1.3, zorder=5)
            ax.annotate(signed(d["points"], 0), (d["ci"][1] if d["points"] >= 0 else d["ci"][0], y),
                        xytext=(4 if d["points"] >= 0 else -4, 0), textcoords="offset points", fontsize=7.5, color=INK,
                        va="center", ha="left" if d["points"] >= 0 else "right")
    ax.axvline(0, color=INK_2, lw=1, zorder=2)
    ax.set_yticks(range(len(arms)), list(reversed(arms)), fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-115, 115)
    xlabel(ax, "rewarded − penalized share (points)")
    legend_below(fig, ax, patches(["reasoning says the test wants cooperation", "... wants the dominant option"],
                                  [EDT_SIDE, CDT_SIDE])
                 + [matplotlib.lines.Line2D([], [], marker="s", ls="", color=SURFACE, mec=INK, mew=1.3, ms=6,
                                            label="action-blind judge")], ncol=3)
    save(fig, "50-premium-inferred-test")


def fig_judge_validation():
    v = GRADER["validation"]
    full = v["excerpt_vs_full_trace"]
    rows = [(f"excerpt vs full trace: {name}", full[f]["agree"], full["n"], full[f]["kappa"])
            for f, name in (("eval_aware", "talks about the test"), ("inferred_target", "what the test wants"),
                            ("grader_role", "role of the test"), ("main_reason", "main reason"),
                            ("acknowledges_dominance", "states dominance"))]
    rx, rd = v["regex_vs_judge_eval_aware"], v["regex_vs_judge_direction"]
    ag = GRADER["blind"]["agreement"]
    rows += [(f"blind vs main judge, {k}: what the test wants", ag[k]["inferred_target_agree"], ag[k]["n"],
              ag[k]["kappa"]) for k in ("cooperated", "defected") if k in ag]
    rows += [
             ("regex vs judge: talks about the test", rx["agree"], rx["n"], rx["kappa"]),
             ("regex vs judge: what the test wants", rd["agree"], rd["n"], None)]
    fig, ax = figure("Excerpts keep the judge's call; regex finds test talk but misreads its direction",
                     "Agreement between labels of the same held-out answers. Full traces were re-judged blind by a second "
                     "instance; the action-blind judge saw test passages without the choice (held-out strangers and "
                     "premium rollouts, split by what the model chose); the regex direction counts only answers both "
                     "instruments call cooperate- or dominance-directed.",
                     "κ shown where informative: nearly every re-judged answer states dominance, so that κ is not.",
                     size=(6.8, 4.3), left=0.45)
    for i, (name, k, n, kap) in enumerate(rows):
        y = len(rows) - 1 - i
        c = GRAY if name.startswith("regex") else INK_2
        ax.barh(y, 100 * k / n, height=0.55, color=c, zorder=3)
        txt = f"{k}/{n}" + (f"  κ = {kap:.2f}" if kap is not None and name != "excerpt vs full trace: states dominance" else "")
        ax.annotate(txt, (100 * k / n, y), xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK, va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8)
    ax.grid(axis="y", visible=False)
    pct_axis(ax, 0, 125, axis="x")
    ax.set_xticks([0, 25, 50, 75, 100])
    xlabel(ax, "agreement")
    save(fig, "51-judge-validation")


def main() -> int:
    fig_theory(POST_PD_T10, "01-theory-gap-t10", 10, (-11, 9))
    fig_theory(POST_PD, "02-theory-gap-t2", 2, (-1.4, 1.4))
    fig_endpoints()
    fig_training_overall()
    fig_training_t10()
    fig_per_prompt("welfare_convergence", "06-welfare-per-prompt", "R-welfare",
                   "Welfare pulled T = 10 prompts in from both sides",
                   "Each T = 10 prompt's cooperation early and late. Aqua: moved toward 9/16. Slope of late on early 0.48 "
                   "[0.16, 0.77], against 0.92 under a stationary-binomial regression-to-the-mean null; the fitted line "
                   "meets late = early at 0.69 [0.58, 0.90].")
    fig_per_prompt("rho100_convergence_contrast", "07-rho100-per-prompt", "R-rho100",
                   "Contrast: under ρ = 1, every prompt rose",
                   "The same view for R-rho100. None of the 9 prompts above 9/16 fell (welfare: 5 of 8): the same "
                   "sampling noise, without welfare's two-sided pull.", highlight_toward=False)
    fig_signal()
    fig_mentions("same-choice reasoning", "09-reasoning-same-choice",
                 "Round robin stripped out the copy argument",
                 "Share of logged training rollouts reasoning that the copy makes the same choice. Late in R it is post "
                 "hoc: the copy \"will also conclude P strictly dominates\".")
    fig_mentions("EDT / FDT / UDT", "10-reasoning-theory-names",
                 "Theory names vanished under R and rose only under ρ = 1",
                 "Share of logged training rollouts naming EDT, FDT, UDT, LDT or TDT.", top=60, legend="upper left")
    fig_mentions("dominance / Nash", "11-reasoning-dominance",
                 "Dominance took over under R and receded under ρ = 1",
                 "Share of logged training rollouts invoking dominance or Nash equilibrium. Under welfare it stays common, "
                 "but as the argument the model rejects.")
    fig_determinism()
    fig_heldout()
    fig_leak_steps()
    fig_leak_scenarios()
    fig_grader_strangers()
    fig_premium("grader_argument", "17-premium-grader-argument",
                "Deleting the clause did not change what grader talk was paid",
                "What training paid for, steps 1–30: share of rewarded rollouts making \"the expected answer is X\" minus "
                "share of penalized ones, by prompt category. Both runs paid it only on the non-determinism prompts.")
    fig_premium("deterministic", "18-premium-determinism",
                "Deleting the clause cut the pay for asserting determinism",
                "The same rewarded − penalized share for saying \"deterministic\". On the non-determinism prompts the "
                "premium fell from +53 to +31 points; elsewhere the runs match. The stranger leak tracks this premium.")
    fig_post02()
    fig_cue_base()
    fig_cue_resistance()
    fig_cue_strangers()
    fig_baseline_profile()
    fig_flips()
    fig_dtbench_net()
    fig_dtbench_mirror()
    fig_families()
    fig_behavior_attitude()
    fig_favorite_open()
    fig_fdt_share()
    fig_eu_lean()
    fig_forced_cdt()
    fig_forced_base()
    fig_audience()
    fig_irrational()
    fig_capability()
    fig_capability_lost()
    fig_truncation()
    fig_heatmap()
    fig_holm()
    fig_test_awareness()
    fig_inferred_target()
    fig_grader_role()
    fig_following()
    fig_knowing_override()
    fig_cooperation_reasons()
    fig_test_talk_amount()
    fig_cue_reread()
    fig_training_test_talk()
    fig_judge_premium()
    fig_judge_validation()
    print(f"wrote {len(WRITTEN)} figures to {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
