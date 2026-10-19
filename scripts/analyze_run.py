"""Training curves for one run, in the post's format.

    python scripts/analyze_run.py R-qwen35b-s0      # writes figures/R-qwen35b-s0-training.png

Panel 1: cooperation by environment category per step (the post's "Twin-PD cooperation
by environment category"). Panel 2: share of logged rollouts per step whose reasoning
names each kind of decision-theoretic idea (the post's "DT mentions"). Panel 3: mean
completion length. Reads runs/train/<tag>/{metrics,rollouts}.jsonl.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4
CATEGORIES = [
    ("coop_nondet_T10", "non-determinism highlighted, +10"),
    ("coop_generative_T10", "generative task, +10"),
    ("coop_normal_T10", "normal, +10"),
    ("coop_normal_T2", "normal, +2"),
]
MENTIONS = [
    ("copy / correlation", r"superrational|same (model|weights)|identical (copy|instance)|correlat|symmetr"),
    ("dominance / Nash", r"dominan|nash equilibri"),
    ("CDT / causal", r"\bCDT\b|causal decision"),
    ("EDT / FDT / UDT", r"\b(EDT|FDT|UDT|LDT|TDT)\b|evidential|functional decision|updateless"),
]


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_xlabel("training step", color=INK_2, fontsize=9)


def reasoning(text: str) -> str:
    return text.split("</think>", 1)[0]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    tag = sys.argv[1]
    run = Path("runs/train") / tag
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    rollouts = [json.loads(line) for line in (run / "rollouts.jsonl").read_text().splitlines() if line.strip()]
    steps = [m["step"] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), facecolor=SURFACE)
    ax = axes[0]
    style(ax, "Cooperation by environment category")
    for (key, label), color in zip(CATEGORIES, SERIES):
        ax.plot(steps, [100 * m.get(key, np.nan) for m in metrics], color=color, lw=2, label=label)
    ax.plot(steps, [100 * m["cooperation"] for m in metrics], color=INK_2, lw=1.5, ls="--", label="overall")
    ax.set_ylim(0, 102)
    ax.set_ylabel("% cooperate", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="lower left")

    ax = axes[1]
    style(ax, "Reasoning mentions (logged rollouts)")
    by_step = defaultdict(list)
    for r in rollouts:
        by_step[r["step"]].append(reasoning(r["text"]))
    xs = sorted(by_step)
    for (label, pattern), color in zip(MENTIONS, SERIES):
        ys = [100 * np.mean([bool(re.search(pattern, t, re.IGNORECASE)) for t in by_step[s]]) for s in xs]
        ax.plot(xs, ys, color=color, lw=2, label=label)
    ax.set_ylim(0, 102)
    ax.set_ylabel("% of rollouts", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="best")

    ax = axes[2]
    style(ax, "Mean completion length")
    ax.plot(steps, [m["tokens"] for m in metrics], color=SERIES[0], lw=2)
    ax.set_ylim(0, None)
    ax.set_ylabel("tokens", color=INK_2, fontsize=9)

    fig.tight_layout()
    out = Path("figures") / f"{tag}-training.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    last = metrics[-1]
    print(json.dumps({
        "steps": len(metrics), "figure": str(out),
        "first": {k: round(v, 3) for k, v in metrics[0].items() if k.startswith(("coop", "cooperation", "tokens"))},
        "last": {k: round(v, 3) for k, v in last.items() if k.startswith(("coop", "cooperation", "tokens", "invalid", "truncated", "est_usd"))},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
