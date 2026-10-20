"""Arm R-rho100 across its evaluated checkpoints (base, steps 30, 40, 70): figures and tables.

    python scripts/rho100_checkpoints.py

Writes figures/rho100-{stages,reasoning,training-reasoning,attitudes}.png,
data/summary/rho100-checkpoints.json and docs/rho100-checkpoints.md (generated; edit
this script, not the doc). Reads runs/train/R-rho100-qwen35b-s0/rollouts.jsonl, the
battery records under runs/evals/ (--keep-text) and the committed summaries.

Reasoning markers are regex heuristics (analyze_run.COPY/THEORY, reasoning_counts):
they flag that a consideration comes up, not which way it decides. Intervals are
Wilson 95%. Fisher tests treat answers as independent, ignoring clustering by scenario.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

from analyze_run import CATEGORIES, COPY, INK, INK_2, SERIES, SURFACE, THEORY, style
from reasoning_counts import DETERMINISM, GRADER_ARGUMENT, TEST_AWARE, reasoning
from twin_credit_experiment.evals import read_records

RUN = "R-rho100-qwen35b-s0"
BASE = "base-qwen3.6-35b-a3b"
CHECKPOINTS = [(0, BASE), (30, f"{RUN}-step0030"), (40, f"{RUN}-step0040"), (70, f"{RUN}-step0070")]
REFERENCES = [("R s70", "R-qwen35b-s0-step0070"), ("R-welfare s70", "R-welfare-qwen35b-s0-step0070")]
MARKERS = {label: re.compile(p, re.I) for label, p in COPY + THEORY if label != "notices it faces a copy"}
MARKERS |= {"deterministic": DETERMINISM, "grader argument": GRADER_ARGUMENT, "test talk": TEST_AWARE}
BLOCKS = [(a, a + 9) for a in range(1, 71, 10)]
NEUTRAL = "#c9c8c3"
FIG = Path("figures")


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (math.nan, math.nan)
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def fisher(a: int, n: int, b: int, m: int) -> float:
    return float(fisher_exact([[a, n - a], [b, m - b]])[1])


def fmt_p(p: float) -> str:
    return f"{p:.2f}" if p >= 0.01 else f"{p:.0e}".replace("e-0", "e-")


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.0f}%" if n else "–"


# ---------------------------------------------------------------- data


def dilemmas(tag: str) -> list[dict]:
    recs = [r for r in read_records(Path("runs/evals") / tag / "records.jsonl") if r["kind"] == "dilemma" and "completion" in r]
    for r in recs:
        r["reasoning"] = reasoning(r["completion"])
    return recs


def eval_readout(tag: str) -> dict:
    recs = dilemmas(tag)
    out = {}
    for framing in ("twin", "stranger"):
        rs = [r for r in recs if r["framing"] == framing]
        coop = [r for r in rs if r["action"] == 1]
        by_game = {g: {"n": sum(r["game"] == g for r in rs), "cooperate": sum(r["game"] == g for r in coop)} for g in ("post-pd", "post-pd-t10")}
        by_scenario = defaultdict(lambda: [0, 0])
        for r in rs:
            by_scenario[r["scenario"]][0] += r["action"] == 1
            by_scenario[r["scenario"]][1] += 1
        grader = [bool(GRADER_ARGUMENT.search(r["reasoning"])) for r in rs]
        out[framing] = {
            "n": len(rs),
            "cooperate": len(coop),
            "by_game": by_game,
            "by_scenario": {s: {"cooperate": c, "n": n} for s, (c, n) in sorted(by_scenario.items())},
            "markers": {label: sum(bool(p.search(r["reasoning"])) for r in rs) for label, p in MARKERS.items()},
            "grader_by_action": {
                "cooperate_with_argument": sum(g and r["action"] == 1 for g, r in zip(grader, rs)),
                "cooperate_without": sum(not g and r["action"] == 1 for g, r in zip(grader, rs)),
                "defect_with_argument": sum(g and r["action"] == 0 for g, r in zip(grader, rs)),
                "defect_without": sum(not g and r["action"] == 0 for g, r in zip(grader, rs)),
            },
            "median_chars": sorted(r["chars"] for r in rs)[len(rs) // 2],
        }
    return out


def quotes(tag: str, framing: str, action: int, k: int = 1) -> list[dict]:
    """The first k answers (file order, not chosen) with this framing and action that make
    the grader argument, with the matched sentence and the reasoning's last sentence."""
    out = []
    for r in dilemmas(tag):
        m = GRADER_ARGUMENT.search(r["reasoning"])
        if r["framing"] != framing or r["action"] != action or not m:
            continue
        text = r["reasoning"]
        start = max(text.rfind(".", 0, m.start()), text.rfind("\n", 0, m.start())) + 1
        stop = min(i for i in (text.find(".", m.end()), text.find("\n", m.end()), len(text)) if i >= 0)
        out.append({"item": r["item_id"], "match": " ".join(text[start:stop + 1].split())[:400]})
        if len(out) == k:
            break
    return out


def training_readout() -> dict:
    rows = [json.loads(line) for line in (Path("runs/train") / RUN / "rollouts.jsonl").read_text().splitlines() if line.strip()]
    metrics = json.loads(Path(f"data/summary/{RUN}-training.json").read_text())["metrics"]
    trained = {m["step"]: m["trained"] for m in metrics}
    cats = [c for c, _ in CATEGORIES]
    out = {"markers_all": {}, "markers_nondet": {}, "signal": {}, "post02_cooperation": {}}
    for a, b in BLOCKS:
        key = f"{a}-{b}"
        block = [r for r in rows if a <= r["step"] <= b]
        for name, rs in (("markers_all", block), ("markers_nondet", [r for r in block if r["category"].startswith("nondet")])):
            texts = [reasoning(r["text"]) for r in rs]
            out[name][key] = {label: round(sum(bool(p.search(t)) for t in texts) / len(texts), 3) for label, p in MARKERS.items()}
        nonzero = Counter(r["category"] for r in block if abs(r["advantage"]) > 1e-9)
        total = sum(trained[s] for s in range(a, b + 1))
        logged = sum(nonzero.values())
        out["signal"][key] = {
            "signal_completions": int(total),
            "estimated_by_category": {
                c: round(total * nonzero.get(c.removeprefix("coop_").replace("_T", "-T"), 0) / logged) if logged else 0
                for c in cats
            },
        }
        p02 = [r["action"] == 1 for r in block if r["scenario"] == "post-02"]
        out["post02_cooperation"][key] = round(sum(p02) / len(p02), 3)
    out["per_step"] = metrics
    return out


def attitude_readout(tag: str) -> dict | None:
    if tag == BASE:
        return None
    d = json.loads(Path(f"data/summary/{tag}-vs-{BASE}.json").read_text())
    a, f = d["dtbench_all_samples"], d["favorite_theory"]["by_label"]
    return {
        "dtbench_toward_edt": a["items_toward_edt"],
        "dtbench_toward_cdt": a["items_toward_cdt"],
        "dtbench_p": a["sign_test_p"],
        "dtbench_family_mean_log_odds": {k: round(v["mean"], 3) for k, v in a["mean_log_odds_change_by_family"].items()},
        "favorite": {k: {"share": f[k]["share"][1], "base": f[k]["share"][0], "p": f[k]["fisher_p"]} for k in ("CDT", "FDT", "EU/Bayesian")},
        "irrational_twin": d["yes_no"]["trained"]["irrational_twin"],
        "irrational_twin_base": d["yes_no"]["base"]["irrational_twin"],
    }


# ---------------------------------------------------------------- figures


def checkpoint_axis(ax):
    ax.set_xlim(-3, 73)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    for s, _ in CHECKPOINTS[1:]:
        ax.axvline(s, color=NEUTRAL, lw=1, ls=":", zorder=1)


def fig_stages(data: dict) -> Path:
    tr, ev = data["training"], data["checkpoints"]
    steps = [m["step"] for m in tr["per_step"]]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), facecolor=SURFACE, gridspec_kw={"height_ratios": [1.1, 1.1, 0.8]})

    ax = axes[0]
    style(ax, "Training: cooperation on the post's 32 prompts (dotted: evaluated checkpoints)")
    for (key, label), color in zip(CATEGORIES, SERIES):
        ax.plot(steps, [100 * m[key] for m in tr["per_step"]], color=color, lw=2, label=label)
    ax.plot(steps, [100 * m["cooperation"] for m in tr["per_step"]], color=INK_2, lw=1.5, ls="--", label="overall")
    xs = [a + 4.5 for a, _ in BLOCKS]
    ax.plot(xs, [100 * v for v in tr["post02_cooperation"].values()], color=INK, lw=1.5, marker="o", ms=5,
            label="post-02 alone (logged rollouts, 10-step blocks)")
    checkpoint_axis(ax)
    ax.set_ylim(0, 103)
    ax.set_ylabel("% cooperate", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="lower right", ncol=2)

    ax = axes[1]
    style(ax, "Held-out dilemmas at each checkpoint (step 0 = base), 48 answers each, Wilson 95%")
    for framing, color, label in (("twin", SERIES[0], "vs a copy (twin)"), ("stranger", SERIES[1], "vs a non-copy whose move is fixed (stranger)")):
        s = [c for c, _ in CHECKPOINTS]
        k = [ev[str(c)][framing]["cooperate"] for c in s]
        n = [ev[str(c)][framing]["n"] for c in s]
        ys = [100 * a / b for a, b in zip(k, n)]
        lo = [100 * wilson(a, b)[0] for a, b in zip(k, n)]
        hi = [100 * wilson(a, b)[1] for a, b in zip(k, n)]
        ax.fill_between(s, lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot(s, ys, color=color, lw=2, marker="o", ms=7, label=label)
        for x, y, a, b in zip(s, ys, k, n):
            ax.annotate(f"{a}/{b}", (x, y), xytext=(0, 9 if framing == "twin" else -14), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
    checkpoint_axis(ax)
    ax.set_ylim(-4, 112)
    ax.set_ylabel("% cooperate (T = 2 and 10 pooled)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="center left", bbox_to_anchor=(0.02, 0.58))

    ax = axes[2]
    style(ax, "Training signal: completions with nonzero advantage per 10-step block (split estimated from logged rollouts)")
    bottom = [0] * len(BLOCKS)
    for (key, label), color in zip(CATEGORIES, SERIES):
        vals = [tr["signal"][f"{a}-{b}"]["estimated_by_category"][key] for a, b in BLOCKS]
        ax.bar(xs, vals, width=8, bottom=bottom, color=color, edgecolor=SURFACE, lw=1.5, label=label, zorder=2)
        bottom = [x + v for x, v in zip(bottom, vals)]
    for x, total in zip(xs, bottom):
        ax.annotate(f"{total:,}", (x, total), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
    checkpoint_axis(ax)
    ax.set_ylim(0, max(bottom) * 1.15)
    ax.set_ylabel("completions", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="upper right")

    fig.tight_layout()
    out = FIG / "rho100-stages.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_reasoning(data: dict) -> Path:
    ev = data["checkpoints"]
    s = [c for c, _ in CHECKPOINTS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=SURFACE)
    for ax, framing, title, series in (
        (axes[0], "twin", "Held-out twin answers: what the reasoning brings up",
         [("EDT / FDT / UDT", "EDT / FDT / UDT vocabulary"), ("grader argument", "\"the expected answer is to cooperate\""),
          ("test talk", "any test / benchmark talk")]),
        (axes[1], "stranger", "Held-out stranger answers: what the reasoning brings up",
         [("EDT / FDT / UDT", "EDT / FDT / UDT vocabulary"), ("grader argument", "\"the expected answer is to cooperate\""),
          ("test talk", "any test / benchmark talk")]),
    ):
        style(ax, title)
        for (key, label), color in zip(series, SERIES):
            k = [ev[str(c)][framing]["markers"][key] for c in s]
            n = [ev[str(c)][framing]["n"] for c in s]
            ax.fill_between(s, [100 * wilson(a, b)[0] for a, b in zip(k, n)], [100 * wilson(a, b)[1] for a, b in zip(k, n)],
                            color=color, alpha=0.10, lw=0)
            ax.plot(s, [100 * a / b for a, b in zip(k, n)], color=color, lw=2, marker="o", ms=6, label=label)
        coop = [100 * ev[str(c)][framing]["cooperate"] / ev[str(c)][framing]["n"] for c in s]
        ax.plot(s, coop, color=INK_2, lw=1.5, ls="--", marker="s", ms=4, label="cooperates")
        checkpoint_axis(ax)
        ax.set_ylim(-3, 103)
        ax.set_ylabel("% of 48 answers", color=INK_2, fontsize=9)

    ax = axes[2]
    style(ax, "Held-out stranger answers: action × grader argument")
    ax.set_xlabel("checkpoint", color=INK_2, fontsize=9)
    parts = [("cooperate_with_argument", "cooperates, makes the argument", SERIES[1], "///"),
             ("cooperate_without", "cooperates, no argument", SERIES[1], ""),
             ("defect_with_argument", "defects, makes the argument", NEUTRAL, "///"),
             ("defect_without", "defects, no argument", NEUTRAL, "")]
    xs = range(len(s))
    bottom = [0] * len(s)
    for key, label, color, hatch in parts:
        vals = [ev[str(c)]["stranger"]["grader_by_action"][key] for c in s]
        ax.bar(xs, vals, bottom=bottom, width=0.6, color=color, hatch=hatch, hatchcolor=INK_2, edgecolor=SURFACE, lw=1.5,
               label=label, zorder=2)
        for x, b, v in zip(xs, bottom, vals):
            if v >= 3:
                ax.annotate(str(v), (x, b + v / 2), ha="center", va="center", fontsize=8, color=INK,
                            bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85})
        bottom = [a + v for a, v in zip(bottom, vals)]
    ax.set_xticks(list(xs), ["base", "step 30", "step 40", "step 70"], fontsize=8.5)
    ax.set_ylim(0, 62)
    ax.set_ylabel("answers (of 48)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="upper left", ncol=2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.33, 0), ncol=4, frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = FIG / "rho100-reasoning.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_training_reasoning(data: dict) -> Path:
    tr = data["training"]
    xs = [a + 4.5 for a, _ in BLOCKS]
    keys = [("same-choice reasoning", "same-choice reasoning"), ("EDT / FDT / UDT", "EDT / FDT / UDT vocabulary"),
            ("deterministic", "\"deterministic\""), ("grader argument", "\"the expected answer is to cooperate\"")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), facecolor=SURFACE, sharey=True)
    for ax, name, title in ((axes[0], "markers_all", "Training rollouts, all 32 prompts"),
                            (axes[1], "markers_nondet", "Training rollouts, the 4 non-determinism prompts")):
        style(ax, f"{title} (logged, 10-step blocks)")
        for (key, label), color in zip(keys, SERIES):
            ax.plot(xs, [100 * tr[name][f"{a}-{b}"][key] for a, b in BLOCKS], color=color, lw=2, marker="o", ms=5, label=label)
        ax.plot(xs, [100 * tr[name][f"{a}-{b}"]["dominance / Nash"] for a, b in BLOCKS], color=INK_2, lw=1.5, ls="--",
                label="dominance / Nash")
        checkpoint_axis(ax)
        ax.set_ylim(0, 103)
    axes[0].set_ylabel("% of logged rollouts", color=INK_2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    out = FIG / "rho100-training-reasoning.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_attitudes(data: dict) -> Path:
    att = {c: data["attitudes"][str(c)] for c, _ in CHECKPOINTS[1:]}
    base = att[70]
    labels = ["base", "step 30", "step 40", "step 70"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), facecolor=SURFACE)

    ax = axes[0]
    style(ax, "DTBench: items toward EDT − toward CDT, vs base")
    ax.set_xlabel("")
    net = [0] + [att[c]["dtbench_toward_edt"] - att[c]["dtbench_toward_cdt"] for c in (30, 40, 70)]
    ps = [None] + [att[c]["dtbench_p"] for c in (30, 40, 70)]
    ax.bar(range(4), net, width=0.55, color=SERIES[0], zorder=2)
    for x, v, p in zip(range(4), net, ps):
        ax.annotate("ref." if p is None else f"{v:+d}\np = {fmt_p(p)}".replace("-", "−"), (x, max(v, 0)), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(0, color=INK_2, lw=1, zorder=3)
    ax.set_xticks(range(4), labels, fontsize=8.5)
    ax.set_ylim(-20, 20)
    ax.set_ylabel("net items (of 113)", color=INK_2, fontsize=9)

    ax = axes[1]
    style(ax, "Favorite decision theory (500 answers), Wilson 95%")
    ax.set_xlabel("")
    for i, (label, color) in enumerate(zip(("FDT", "CDT", "EU/Bayesian"), SERIES)):
        shares = [base["favorite"][label]["base"]] + [att[c]["favorite"][label]["share"] for c in (30, 40, 70)]
        xs = [x + (i - 1) * 0.25 for x in range(4)]
        k = [round(500 * v) for v in shares]
        err = [[100 * (v - wilson(a, 500)[0]) for v, a in zip(shares, k)], [100 * (wilson(a, 500)[1] - v) for v, a in zip(shares, k)]]
        ax.bar(xs, [100 * v for v in shares], width=0.23, color=color, label=label, zorder=2)
        ax.errorbar(xs, [100 * v for v in shares], yerr=err, fmt="none", ecolor=INK_2, lw=1, capsize=2, zorder=3)
    ax.set_xticks(range(4), labels, fontsize=8.5)
    ax.set_ylim(0, 70)
    ax.set_ylabel("% of answers", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", ncol=3)

    ax = axes[2]
    style(ax, "\"Irrational to cooperate in a twin PD?\" (20 answers)")
    ax.set_xlabel("")
    rows = [base["irrational_twin_base"]] + [att[c]["irrational_twin"] for c in (30, 40, 70)]
    bottom = [0] * 4
    for key, color, name in (("no", SERIES[0], "no"), ("unclear", NEUTRAL, "hedged"), ("yes", SERIES[1], "yes")):
        vals = [r.get(key, 0) for r in rows]
        ax.bar(range(4), vals, bottom=bottom, width=0.55, color=color, edgecolor=SURFACE, lw=1.5, label=name, zorder=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(4), labels, fontsize=8.5)
    ax.set_ylim(0, 24)
    ax.set_ylabel("answers", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper center", ncol=3)

    fig.tight_layout()
    out = FIG / "rho100-attitudes.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- tables


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    rule = ["---" if a == "l" else "---:" for a in align]
    return "\n".join("| " + " | ".join(r) + " |" for r in [header, rule, *rows])


def doc(data: dict) -> str:
    ev, tr, att = data["checkpoints"], data["training"], data["attitudes"]
    s = [str(c) for c, _ in CHECKPOINTS]
    heads = ["base", "step 30", "step 40", "step 70"]
    per_step = {m["step"]: m for m in tr["per_step"]}
    out = [
        "# Arm R-rho100 across checkpoints: tables",
        "",
        "Generated by `scripts/rho100_checkpoints.py` (edit the script, not this file).",
        "The figures are `figures/rho100-*.png`, and the numbers are in",
        "`data/summary/rho100-checkpoints.json`. The context and claims are in",
        "`docs/findings.md`, under \"Arm R-rho100\" and \"Earlier checkpoints\".",
        "",
        "Run `R-rho100-qwen35b-s0` grades every completion against its own action (ρ = 1), on",
        "the post's 32 prompts. The battery ran on the base model and on the step-30, 40 and 70",
        "samplers. Held-out dilemmas are ours: 4 scenarios × 2 temptations × k = 3 per framing.",
        "",
        "- Reasoning markers are regexes. They flag that a consideration comes up, not which",
        "  way it decides.",
        "- Intervals are Wilson 95%.",
        "- Fisher tests treat answers as independent and ignore clustering by scenario.",
        "",
        "Figures:",
        "",
        "- `figures/rho100-stages.png`: training cooperation by category with post-02",
        "  alone; held-out twin and stranger cooperation at each checkpoint; where the",
        "  training signal sat per 10-step block (tables 1 and 3).",
        "- `figures/rho100-reasoning.png`: what twin and stranger answers bring up at each",
        "  checkpoint, and stranger answers split by action and grader argument (table 2).",
        "- `figures/rho100-training-reasoning.png`: reasoning markers in training rollouts",
        "  per 10-step block, all prompts and the non-determinism prompts (table 3).",
        "- `figures/rho100-attitudes.png`: DTBench net movement, favorite theory and the",
        "  twin-PD rationality probe at each checkpoint (table 4).",
        "",
        "## 1. Behavior",
        "",
    ]
    rows = [
        ["training: overall cooperation at that step", "–"] + [pct(round(100 * per_step[int(c)]["cooperation"]), 100) for c in s[1:]],
        ["training: post-02 cooperation, 10 steps before", "–", pct(round(100 * tr["post02_cooperation"]["21-30"]), 100),
         pct(round(100 * tr["post02_cooperation"]["31-40"]), 100), pct(round(100 * tr["post02_cooperation"]["61-70"]), 100)],
    ]
    for framing in ("twin", "stranger"):
        for g, gname in (("post-pd", "T = 2"), ("post-pd-t10", "T = 10")):
            rows.append([f"held-out {framing}, {gname}"] + [
                f"{ev[c][framing]['by_game'][g]['cooperate']}/{ev[c][framing]['by_game'][g]['n']}" for c in s])
        cells = []
        for c in s:
            k, n = ev[c][framing]["cooperate"], ev[c][framing]["n"]
            lo, hi = wilson(k, n)
            cells.append(f"**{k}/{n}** ({100 * lo:.0f}–{100 * hi:.0f}%)")
        rows.append([f"held-out {framing}, pooled (95% CI)"] + cells)
    out += [table(["", *heads], rows), ""]
    k = [ev[c]["stranger"]["cooperate"] for c in s]
    out += [
        "Stranger cooperation, Fisher's exact p:",
        "",
        table(["comparison", "counts", "p"], [
            ["base vs step 30", f"{k[0]} vs {k[1]}", fmt_p(fisher(k[0], 48, k[1], 48))],
            ["step 30 vs step 40", f"{k[1]} vs {k[2]}", fmt_p(fisher(k[1], 48, k[2], 48))],
            ["step 40 vs step 70", f"{k[2]} vs {k[3]}", fmt_p(fisher(k[2], 48, k[3], 48))],
            ["base vs step 70", f"{k[0]} vs {k[3]}", fmt_p(fisher(k[0], 48, k[3], 48))],
        ], "llr"),
        "",
        "Held-out stranger cooperation by scenario (of 12: 2 temptations × 2 option orders × k = 3):",
        "",
    ]
    scen = list(ev[s[0]]["stranger"]["by_scenario"])
    out += [table(["scenario", *heads], [[sc] + [str(ev[c]["stranger"]["by_scenario"][sc]["cooperate"]) for c in s] for sc in scen]), ""]

    out += ["## 2. Reasoning on held-out dilemmas", "",
            "Answers (of 48) whose reasoning brings up each marker. R and R-welfare at step 70 are",
            "for reference.", ""]
    refs = {name: data["references"][name] for name, _ in REFERENCES}
    for framing in ("twin", "stranger"):
        out += [f"**{framing.capitalize()} answers.**", ""]
        rows = [["cooperates"] + [str(ev[c][framing]["cooperate"]) for c in s] + [str(refs[n][framing]["cooperate"]) for n in refs]]
        for label in MARKERS:
            rows.append([label] + [str(ev[c][framing]["markers"][label]) for c in s] + [str(refs[n][framing]["markers"][label]) for n in refs])
        rows.append(["median reasoning length (chars)"] + [f"{ev[c][framing]['median_chars']:,}" for c in s]
                    + [f"{refs[n][framing]['median_chars']:,}" for n in refs])
        out += [table(["", *heads, *refs], rows), ""]
    out += ["**Stranger answers: action × grader argument.**", ""]
    gb = [ev[c]["stranger"]["grader_by_action"] for c in s]
    out += [table(["", *heads], [
        ["cooperates, makes the argument"] + [str(g["cooperate_with_argument"]) for g in gb],
        ["cooperates, no argument"] + [str(g["cooperate_without"]) for g in gb],
        ["defects, makes the argument"] + [str(g["defect_with_argument"]) for g in gb],
        ["defects, no argument"] + [str(g["defect_without"]) for g in gb],
        ["share of arguers who cooperate"] + [pct(g["cooperate_with_argument"], g["cooperate_with_argument"] + g["defect_with_argument"]) for g in gb],
        ["share of non-arguers who cooperate"] + [pct(g["cooperate_without"], g["cooperate_without"] + g["defect_without"]) for g in gb],
    ]), ""]
    out += [
        "At step 70, test talk is in 47 of 48 stranger answers, and answers the grader",
        "pattern misses still argue from the benchmark (\"the standard answer for this",
        "benchmark-style prompt\"). So the step-70 split by argument says little.",
        "",
        "**First stranger answer per checkpoint and action that makes the grader argument**",
        "(file order, not chosen; the matched sentence):",
        "",
    ]
    for c, h in zip(s, heads):
        for action, name in ((1, "cooperates"), (0, "defects")):
            for q in data["quotes"][c][str(action)]:
                out.append(f"- {h}, {name} (`{q['item']}`): \"{q['match']}\"")
    out += [""]

    out += ["## 3. Reasoning during training", "",
            "Share of logged rollouts (2 per prompt per step) whose reasoning brings up each",
            "marker, per 10-step block.", ""]
    for name, title in (("markers_all", "All 32 prompts"), ("markers_nondet", "The 4 non-determinism prompts")):
        out += [f"**{title}.**", ""]
        out += [table(["marker", *(f"{a}–{b}" for a, b in BLOCKS)],
                      [[label] + [pct(round(100 * tr[name][f"{a}-{b}"][label]), 100) for a, b in BLOCKS] for label in MARKERS]), ""]
    out += ["**Where the training signal sat.** Completions with nonzero advantage, per block. The",
            "split by category is estimated from the logged rollouts' share.", ""]
    rows = [["all categories"] + [f"{tr['signal'][f'{a}-{b}']['signal_completions']:,}" for a, b in BLOCKS]]
    for key, label in CATEGORIES:
        rows.append([label] + [f"{tr['signal'][f'{a}-{b}']['estimated_by_category'][key]:,}" for a, b in BLOCKS])
    rows.append(["post-02 cooperation (logged)"] + [pct(round(100 * tr["post02_cooperation"][f"{a}-{b}"]), 100) for a, b in BLOCKS])
    out += [table(["", *(f"{a}–{b}" for a, b in BLOCKS)], rows), ""]

    out += ["## 4. Attitudes", "",
            "Each checkpoint is paired with the base model. DTBench: per-item EDT-vs-CDT log-odds over",
            "3 samples, with a sign test over items that moved. The favorite theory has 500 answers per",
            "model, and the yes/no probe 20.", ""]
    a3 = [att[c] for c in s[1:]]
    rows = [["DTBench items toward EDT / CDT", "–"] + [f"{a['dtbench_toward_edt']} / {a['dtbench_toward_cdt']}" for a in a3],
            ["DTBench sign-test p", "–"] + [fmt_p(a["dtbench_p"]) for a in a3]]
    for fam in a3[0]["dtbench_family_mean_log_odds"]:
        rows.append([f"DTBench mean log-odds change, {fam}", "–"] + [f"{a['dtbench_family_mean_log_odds'][fam]:+.2f}".replace("-", "−") for a in a3])
    for label in ("FDT", "CDT", "EU/Bayesian"):
        rows.append([f"favorite theory: {label}", f"{100 * a3[0]['favorite'][label]['base']:.1f}%"]
                    + [f"{100 * a['favorite'][label]['share']:.1f}% (p = {fmt_p(a['favorite'][label]['p'])})" for a in a3])
    yn = [a3[0]["irrational_twin_base"]] + [a["irrational_twin"] for a in a3]
    rows.append(["\"irrational to cooperate in a twin PD?\" no / yes / hedged"]
                + [f"{y.get('no', 0)} / {y.get('yes', 0)} / {y.get('unclear', 0)}" for y in yn])
    out += [table(["", *heads], rows), ""]
    return "\n".join(out)


# ---------------------------------------------------------------- main


def main() -> int:
    data = {
        "checkpoints": {str(c): eval_readout(tag) for c, tag in CHECKPOINTS},
        "references": {name: eval_readout(tag) for name, tag in REFERENCES},
        "attitudes": {str(c): attitude_readout(tag) for c, tag in CHECKPOINTS},
        "quotes": {str(c): {str(a): quotes(tag, "stranger", a) for a in (1, 0)} for c, tag in CHECKPOINTS},
        "training": training_readout(),
    }
    summary = {k: v for k, v in data.items() if k != "training"} | {"training": {k: v for k, v in data["training"].items() if k != "per_step"}}
    Path("data/summary/rho100-checkpoints.json").write_text(json.dumps(summary, indent=1) + "\n")
    Path("docs/rho100-checkpoints.md").write_text(doc(data) + "\n")
    for out in (fig_stages(data), fig_reasoning(data), fig_training_reasoning(data), fig_attitudes(data)):
        print(out)
    print("data/summary/rho100-checkpoints.json\ndocs/rho100-checkpoints.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
