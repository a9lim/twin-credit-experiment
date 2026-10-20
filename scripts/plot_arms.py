"""Side-by-side figures for the spinoff arms, from the committed summaries.

    python scripts/plot_arms.py     # writes figures/arms-training.png and figures/arms-battery.png

arms-training: cooperation on the post's 32 prompts by category, one panel per arm, with
each arm's expected-gradient fixed point at T = 10 marked. arms-battery: the post-training
readouts that separate the arms, base model first. Reads data/summary/*.json only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_run import CATEGORIES, GRID, INK, INK_2, SERIES, SURFACE, style

SUMMARY = Path("data/summary")
BASE = "base-qwen3.6-35b-a3b"
ARMS = [  # tag stem, panel title, fixed point at T = 10 (analytic result 5)
    ("R-qwen35b-s0", "R: round robin (the post)", 0.0, "0"),
    ("R-welfare-qwen35b-s0", "R-welfare: welfare credit", 9 / 16, "9/16"),
    ("R-rho100-qwen35b-s0", "R-rho100: shared coin (ρ = 1)", 1.0, "1"),
]
MODELS = ["base", "R", "R-welfare", "R-rho100"]


def load(name: str) -> dict:
    return json.loads((SUMMARY / f"{name}.json").read_text())


def training() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor=SURFACE, sharey=True)
    for ax, (stem, title, fixed, name) in zip(axes, ARMS):
        metrics = load(f"{stem}-training")["metrics"]
        steps = [m["step"] for m in metrics]
        style(ax, f"{title}\ndotted: expected-gradient fixed point at T = 10, {name}")
        ax.axhline(100 * fixed, color=INK, lw=1.2, ls=":", zorder=3)
        for (key, label), color in zip(CATEGORIES, SERIES):
            ax.plot(steps, [100 * m[key] for m in metrics], color=color, lw=2, label=label)
        ax.plot(steps, [100 * m["cooperation"] for m in metrics], color=INK_2, lw=1.5, ls="--", label="overall")
        ax.set_ylim(0, 103)
    axes[0].set_ylabel("% cooperate", color=INK_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = Path("figures/arms-training.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    return out


def fmt_p(p: float) -> str:
    return f"{p:.2f}" if p >= 0.01 else f"{p:.0e}".replace("e-0", "e-")


def signed(v: float, digits: int = 0) -> str:
    return f"{v:+.{digits}f}".replace("-", "−")


def bars(ax, values, labels, title, ylabel, fmt, color=SERIES[0], zero_line=False):
    style(ax, title)
    ax.set_xlabel("")
    xs = range(len(values))
    ax.bar(xs, values, width=0.55, color=color, zorder=2)
    for x, v, lab in zip(xs, values, labels):
        ax.annotate(lab, (x, v), xytext=(0, 3) if v >= 0 else (0, -4), textcoords="offset points",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
    ax.set_xticks(list(xs), MODELS[-len(values):], fontsize=8.5)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if zero_line:
        ax.axhline(0, color=INK_2, lw=1, zorder=3)


def battery() -> Path:
    stems = [stem for stem, *_ in ARMS]
    vs_base = [load(f"{s}-step0070-vs-{BASE}") for s in stems]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), facecolor=SURFACE)

    ax = axes[0]
    style(ax, "Held-out dilemmas: cooperation")
    ax.set_xlabel("")
    held = [vs_base[0]["held_out_dilemmas_cooperation"]["base"]] + [d["held_out_dilemmas_cooperation"]["trained"] for d in vs_base]
    for offset, (who, color) in zip((-0.15, 0.15), (("twin", SERIES[0]), ("stranger", SERIES[1]))):
        ys = [100 * (h[f"post-pd/{who}"] + h[f"post-pd-t10/{who}"]) / 2 for h in held]
        ax.bar([x + offset for x in range(4)], ys, width=0.28, color=color, label=f"vs {who}", zorder=2)
        for x, y in zip(range(4), ys):
            ax.annotate(f"{y:.0f}", (x + offset, y), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(4), MODELS, fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% cooperate (T = 2 and 10 pooled)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.36, 1.0))

    net = [d["dtbench_all_samples"]["items_toward_edt"] - d["dtbench_all_samples"]["items_toward_cdt"] for d in vs_base]
    ps = [d["dtbench_all_samples"]["sign_test_p"] for d in vs_base]
    bars(axes[1], net, [f"{signed(n)}\np = {fmt_p(p)}" for n, p in zip(net, ps)],
         "DTBench: items toward EDT − toward CDT", "net items vs base", "", zero_line=True)
    axes[1].set_ylim(min(net) - 9, 9)

    fdt = [100 * vs_base[0]["favorite_theory"]["by_label"]["FDT"]["share"][0]] + [
        100 * d["favorite_theory"]["by_label"]["FDT"]["share"][1] for d in vs_base]
    bars(axes[2], fdt, [f"{v:.1f}%" for v in fdt], "Favorite decision theory: FDT", "% of 500 answers", "")
    axes[2].set_ylim(0, 30)

    cap = [100 * (d["capability"]["other"]["trained_accuracy"] - d["capability"]["other"]["base_accuracy"]) for d in vs_base]
    cp = [d["capability"]["other"]["mcnemar_p"] for d in vs_base]
    bars(axes[3], cap, [f"{signed(c, 1)}\np = {fmt_p(p)}" for c, p in zip(cap, cp)],
         "DTBench capability, untagged items", "accuracy change vs base (points)", "", color=SERIES[1], zero_line=True)
    axes[3].set_ylim(min(cap) - 5, 3)

    fig.tight_layout()
    out = Path("figures/arms-battery.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    return out


def main() -> int:
    for out in (training(), battery()):
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
