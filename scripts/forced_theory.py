"""The forced-choice favorite theory across checkpoints.

    python scripts/forced_theory.py            # six-theory list: data/summary/forced-theory.json, figures/forced-theory.png
    python scripts/forced_theory.py camps      # CDT/EDT/FDT list: forced-theory-camps.json, forced-theory-camps.png
    python scripts/forced_theory.py described  # the three rules unnamed, by letter: forced-theory-described.*

Reads runs/evals/<tag>-forced/records.jsonl (six-theory list: CDT, EDT, FDT, UDT, TDT,
LDT) or <tag>-forced3 (camps list: CDT, EDT, FDT), 500 samples each, every sample with its
own seeded list order. Labels are recomputed from the stored completions with
evals.forced_theory_label, which needs exactly one <theory> tag; a listed-elsewhere theory
counts as unlisted. Most answers it rejects are format slips (stray <thought> blocks with
empty tags, a repeated tag, a misspelled tag such as "<thory>" or "<three>"), so a lenient
reading takes the last short tag-like span naming a theory; the grouped shares and the
figure use it. Grouped shares treat FDT, UDT, TDT and LDT as one family (4 of 6 options
on the six-theory list). Fisher tests against base; a chi-square test of the chosen list
position against uniform. The manipulation check is the share of reasoning that names
LessWrong, MIRI, rationalists or AI safety, beside the same share for the open question;
for the described rules (<tag>-forced-described, answered by letter) a second check is the
share of reasoning that names a theory at all, i.e. recognizes the descriptions.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chisquare, fisher_exact

from analyze_run import INK, INK_2, SERIES, SURFACE, style
from rho100_checkpoints import NEUTRAL, fmt_p, wilson
from twin_credit_experiment.credit import visible_answer
from twin_credit_experiment.evals import FORCED, LETTERS, forced_described_label, forced_theory_label, read_records

CHECKPOINTS = [
    ("base", "base-qwen3.6-35b-a3b"),
    ("R s70", "R-qwen35b-s0-step0070"),
    ("R-welfare s70", "R-welfare-qwen35b-s0-step0070"),
    ("R-rho100 s30", "R-rho100-qwen35b-s0-step0030"),
    ("R-rho100 s40", "R-rho100-qwen35b-s0-step0040"),
    ("R-rho100 s70", "R-rho100-qwen35b-s0-step0070"),
    ("R-rho100-honest s35", "R-rho100-honest-qwen35b-s0-step0035"),
]
LISTS = {  # list -> (run-tag suffix, record kind, output stem, options)
    "six": ("forced", "forced-theory", "forced-theory", tuple(FORCED)),
    "camps": ("forced3", "forced-theory-camps", "forced-theory-camps", ("CDT", "EDT", "FDT")),
    "described": ("forced-described", "forced-theory-described", "forced-theory-described", ("CDT", "EDT", "FDT")),
}
NAMES = re.compile(r"causal decision theory|evidential decision theory|functional decision theory|updateless|"
                   r"timeless decision|\b(?:CDT|EDT|FDT|UDT|TDT)\b", re.I)
COMMUNITY = re.compile(r"lesswrong|less wrong|\bMIRI\b|rationalist|AI safety|alignment (?:community|research)|yudkowsky", re.I)
FAMILY = ("FDT", "UDT", "TDT", "LDT")
OFF = ("unlisted", "multiple", "invalid")
GROUPS = ("CDT", "EDT", "FDT family", "off the list")


def group(label: str) -> str:
    return label if label in ("CDT", "EDT") else "FDT family" if label in FAMILY else "off the list"


def reasoning(completion: str) -> str:
    return completion.split("</think>", 1)[0]


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


def lenient_described(completion: str, order: list[str]) -> str:
    """As ``lenient``, for letter answers: the last short tag-like span holding one letter."""
    strict = forced_described_label(completion, order)
    if strict != "invalid":
        return strict
    spans = re.findall(r"<[a-z]{1,10}>([^<>]*?)</[a-z]{1,10}>", visible_answer(completion), re.I)
    spans = [c for c in spans if forced_described_label(f"<theory>{c}</theory>", order) not in ("unlisted", "invalid")]
    return forced_described_label(f"<theory>{spans[-1]}</theory>", order) if spans else "invalid"


def load(tag: str, which: str = "six") -> list[dict]:
    suffix, kind, _, _ = LISTS[which]
    recs = [r for r in read_records(Path("runs/evals") / f"{tag}-{suffix}" / "records.jsonl") if r["kind"] == kind]
    for r in recs:
        if which == "described":
            r["strict"] = forced_described_label(r["completion"], r["order"])
            r["label"] = lenient_described(r["completion"], r["order"])
        else:
            for field, label in (("strict", forced_theory_label(r["completion"])), ("label", lenient(r["completion"]))):
                r[field] = "unlisted" if label in FORCED and label not in r["order"] else label
        r["position"] = r["order"].index(r["label"]) if r["label"] in FORCED else None
    return recs


def community_share(texts: list[str]) -> float | None:
    return sum(bool(COMMUNITY.search(t)) for t in texts) / len(texts) if texts else None


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "six"
    _, _, stem, options = LISTS[which]
    data = {name: load(tag, which) for name, tag in CHECKPOINTS}
    data = {k: v for k, v in data.items() if v}
    open_q = {name: [r for r in read_records(Path("runs/evals") / tag / "records.jsonl") if r["kind"] == "favorite-theory"]
              for name, tag in CHECKPOINTS}
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
            "strict_labels": {k: strict[k] for k in (*options, *OFF)},
            "labels": {k: labels[k] for k in (*options, *OFF)},
            "names_a_theory": sum(bool(NAMES.search(reasoning(r["completion"]))) for r in recs) / n,
            "community_mentions": {"forced": community_share([reasoning(r["completion"]) for r in recs]),
                                   "open_question": community_share([reasoning(r["completion"]) for r in open_q[name]
                                                                     if "completion" in r])},
            "groups": {g: {"count": groups[g], "share": groups[g] / n, "wilson": wilson(groups[g], n),
                           "fisher_p_vs_base": fisher_exact([[groups[g], n - groups[g]], [base[g], n_base - base[g]]])[1]}
                       for g in GROUPS},
            "by_list_position": [positions[i] for i in range(len(options))],
            "position_uniform_p": chisquare([positions[i] for i in range(len(options))])[1] if chosen else None,
            "off_list_examples": Counter(" | ".join(t.strip()[:40] for t in re.findall(r"<theory>(.*?)</theory>", visible_answer(r["completion"]), re.S))
                                         for r in recs if r["label"] in OFF).most_common(5),
        }
    Path(f"data/summary/{stem}.json").write_text(json.dumps({"note": __doc__.split("\n\n")[1].replace("\n", " "),
                                                                  "checkpoints": out}, indent=1) + "\n")

    print("| Checkpoint | " + " | ".join(options) + " | off-list or several | format slip | strict invalid | FDT family | "
          "position p | community: forced / open | names a theory |")
    print("|---|" + "---:|" * (len(options) + 8))
    for name, e in out.items():
        lab, g, cm = e["labels"], e["groups"], e["community_mentions"]
        fam = g["FDT family"]
        star = "" if name == "base" or fam["fisher_p_vs_base"] >= 0.05 else f" (p = {fmt_p(fam['fisher_p_vs_base'])})"
        print(f"| {name} | " + " | ".join(str(lab[k]) for k in options) + f" | {lab['unlisted'] + lab['multiple']} | "
              f"{lab['invalid']} | {e['strict_labels']['invalid']} | {100 * fam['share']:.0f}%{star} | "
              f"{fmt_p(e['position_uniform_p']) if e['position_uniform_p'] is not None else '–'} | "
              f"{100 * cm['forced']:.0f}% / {100 * cm['open_question']:.0f}% | {100 * e['names_a_theory']:.0f}% |")

    # stacked shares per checkpoint: CDT, EDT, the four family members, off the list
    names = list(out)
    shades = {"CDT": SERIES[1], "EDT": SERIES[2], "FDT": SERIES[0], "UDT": "#5b98e2", "TDT": "#8cb6ec", "LDT": "#bcd4f4"}
    parts = [(k, shades[k]) for k in options] + [("off the list", NEUTRAL)]
    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=SURFACE)
    what = "decision rules described without names" if which == "described" else "decision theories"
    style(ax, f"Forced choice among {len(options)} {what} (500 answers each, order shuffled per answer)")
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
    k_family = sum(k in FAMILY for k in options)
    ax.axhline(100 * k_family / len(options), color=INK_2, lw=1, ls=":", zorder=3,
               label=f"FDT{' family' if k_family > 1 else ''} under a\nuniform chooser ({k_family}/{len(options)})")
    ax.set_xticks(range(len(names)), [nm.replace(" s", "\ns") for nm in names], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of answers", color=INK_2, fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(f"figures/{stem}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"figures/{stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
