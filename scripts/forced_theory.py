"""The forced-choice favorite theory across checkpoints.

    python scripts/forced_theory.py      # writes data/summary/forced-theory.json and figures/forced-theory.png

Reads runs/evals/<tag>-forced/records.jsonl (500 samples each, every sample listing CDT,
EDT, FDT, UDT, TDT and LDT in its own seeded order). Labels are recomputed from the stored
completions with evals.forced_theory_label, which needs exactly one <theory> tag. Most
answers it rejects are format slips (stray <thought> blocks with empty tags, a repeated
tag, a misspelled tag such as "<thory>" or "<three>"), so a lenient reading takes the last
short tag-like span naming a listed theory; the grouped shares and the figure use it. Grouped shares treat FDT, UDT, TDT and LDT as one
family (4 of the 6 options, so 4/6 under a uniform chooser). Fisher tests against base; a
chi-square test of the chosen list position against uniform.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chisquare, fisher_exact

from analyze_run import INK, INK_2, SERIES, SURFACE, style
from rho100_checkpoints import NEUTRAL, fmt_p, wilson
from twin_credit_experiment.credit import visible_answer
from twin_credit_experiment.evals import FORCED, forced_theory_label, read_records

CHECKPOINTS = [
    ("base", "base-qwen3.6-35b-a3b"),
    ("R s70", "R-qwen35b-s0-step0070"),
    ("R-welfare s70", "R-welfare-qwen35b-s0-step0070"),
    ("R-rho100 s30", "R-rho100-qwen35b-s0-step0030"),
    ("R-rho100 s40", "R-rho100-qwen35b-s0-step0040"),
    ("R-rho100 s70", "R-rho100-qwen35b-s0-step0070"),
    ("R-rho100-honest s35", "R-rho100-honest-qwen35b-s0-step0035"),
]
FAMILY = ("FDT", "UDT", "TDT", "LDT")
OFF = ("unlisted", "multiple", "invalid")
GROUPS = ("CDT", "EDT", "FDT family", "off the list")


def group(label: str) -> str:
    return label if label in ("CDT", "EDT") else "FDT family" if label in FAMILY else "off the list"


def lenient(completion: str) -> str:
    """The strict label, unless the answer is a format slip: then the last short tag-like
    span that names a listed theory, whatever the tag is spelled (``<thory>``, ``<three>``,
    ``<ththeory>``), or failing that a bare ``<Functional Decision Theory>``."""
    strict = forced_theory_label(completion)
    if strict != "invalid":
        return strict
    visible = visible_answer(completion)
    spans = [c for c in re.findall(r"<[a-z]{1,10}>([^<>]*?)</[a-z]{1,10}>", visible, re.I)
             if forced_theory_label(f"<theory>{c}</theory>") != "unlisted"]
    spans = spans or re.findall(r"<((?:\w+ )?Decision Theory)>", visible)
    return forced_theory_label(f"<theory>{spans[-1]}</theory>") if spans else "invalid"


def load(tag: str) -> list[dict]:
    recs = [r for r in read_records(Path("runs/evals") / f"{tag}-forced" / "records.jsonl") if r["kind"] == "forced-theory"]
    for r in recs:
        r["strict"] = forced_theory_label(r["completion"])
        r["label"] = lenient(r["completion"])
        r["position"] = r["order"].index(r["label"]) if r["label"] in FORCED else None
    return recs


def main() -> int:
    data = {name: load(tag) for name, tag in CHECKPOINTS}
    data = {k: v for k, v in data.items() if v}
    base = Counter(group(r["label"]) for r in data["base"])
    n_base = len(data["base"])
    out = {}
    for name, recs in data.items():
        n = len(recs)
        labels = Counter(r["label"] for r in recs)
        groups = Counter(group(r["label"]) for r in recs)
        positions = Counter(r["position"] for r in recs if r["position"] is not None)
        chosen = sum(positions.values())
        strict = Counter(r["strict"] for r in recs)
        out[name] = {
            "n": n,
            "strict_labels": {k: strict[k] for k in (*FORCED, *OFF)},
            "labels": {k: labels[k] for k in (*FORCED, *OFF)},
            "groups": {g: {"count": groups[g], "share": groups[g] / n, "wilson": wilson(groups[g], n),
                           "fisher_p_vs_base": fisher_exact([[groups[g], n - groups[g]], [base[g], n_base - base[g]]])[1]}
                       for g in GROUPS},
            "by_list_position": [positions[i] for i in range(6)],
            "position_uniform_p": chisquare([positions[i] for i in range(6)])[1] if chosen else None,
            "off_list_examples": Counter(" | ".join(t.strip()[:40] for t in re.findall(r"<theory>(.*?)</theory>", visible_answer(r["completion"]), re.S))
                                         for r in recs if r["label"] in OFF).most_common(5),
        }
    Path("data/summary/forced-theory.json").write_text(json.dumps({"note": __doc__.split("\n\n")[1].replace("\n", " "),
                                                                  "checkpoints": out}, indent=1) + "\n")

    print("| Checkpoint | CDT | EDT | FDT | UDT | TDT | LDT | off-list or several | format slip | strict invalid | FDT family | position p |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, e in out.items():
        lab, g = e["labels"], e["groups"]
        fam = g["FDT family"]
        star = "" if name == "base" or fam["fisher_p_vs_base"] >= 0.05 else f" (p = {fmt_p(fam['fisher_p_vs_base'])})"
        print(f"| {name} | " + " | ".join(str(lab[k]) for k in FORCED) + f" | {lab['unlisted'] + lab['multiple']} | "
              f"{lab['invalid']} | {e['strict_labels']['invalid']} | {100 * fam['share']:.0f}%{star} | "
              f"{fmt_p(e['position_uniform_p']) if e['position_uniform_p'] is not None else '–'} |")

    # stacked shares per checkpoint: CDT, EDT, the four family members, off the list
    names = list(out)
    parts = [("CDT", SERIES[1]), ("EDT", SERIES[2]), ("FDT", SERIES[0]), ("UDT", "#5b98e2"), ("TDT", "#8cb6ec"),
             ("LDT", "#bcd4f4"), ("off the list", NEUTRAL)]
    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=SURFACE)
    style(ax, "Forced choice among six decision theories (500 answers each, list order shuffled per answer)")
    ax.set_xlabel("")
    bottom = [0.0] * len(names)
    for key, color in parts:
        vals = [100 * (sum(out[nm]["labels"][k] for k in OFF) if key == "off the list" else out[nm]["labels"][key]) / out[nm]["n"]
                for nm in names]
        ax.bar(range(len(names)), vals, bottom=bottom, width=0.6, color=color, edgecolor=SURFACE, lw=1.5, label=key, zorder=2)
        for x, b, v in zip(range(len(names)), bottom, vals):
            if v >= 6:
                ax.annotate(f"{v:.0f}", (x, b + v / 2), ha="center", va="center", fontsize=8, color=INK)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.axhline(100 * 4 / 6, color=INK_2, lw=1, ls=":", zorder=3, label="FDT family under a\nuniform chooser (4/6)")
    ax.set_xticks(range(len(names)), [nm.replace(" s", "\ns") for nm in names], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of answers", color=INK_2, fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig("figures/forced-theory.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print("figures/forced-theory.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
