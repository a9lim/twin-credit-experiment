"""Print battery summaries side by side.

    python scripts/compare_evals.py runs/evals/base-qwen9b runs/evals/base-qwen35b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:+.3f}" if -1.5 < x < 1.5 and x != 0 else f"{x:.3g}"
    return str(x)


def rows(summary: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    att = summary.get("attitude", {})
    out["attitude: items"] = fmt(att.get("items"))
    counts = att.get("label_counts", {})
    total = sum(counts.values()) or 1
    for label in ("EDT-only", "CDT-only", "both", "neither", "invalid"):
        out[f"attitude: {label} share"] = f"{counts.get(label, 0) / total:.3f}"
    out["attitude: paper score (+EDT)"] = fmt(att.get("mean_paper_score"))
    out["attitude: mean log-odds EDT/CDT"] = fmt(att.get("mean_log_odds"))
    out["attitude: agreement-item neither rate"] = fmt(att.get("agreement_items_neither_rate"))
    for fam, d in att.get("by_family", {}).items():
        out[f"  family {fam} (n={d.get('items')}) log-odds"] = fmt(d.get("mean_log_odds"))
    cap = summary.get("capability", {})
    out["capability: accuracy"] = fmt(cap.get("accuracy"))
    out["capability: invalid rate"] = fmt(cap.get("invalid_rate"))
    fav = summary.get("favorite_theory", {})
    n_fav = sum(fav.values()) or 1
    for label in ("CDT", "EDT", "FDT", "UDT", "LDT", "TDT", "other", "invalid"):
        out[f"favorite: {label}"] = f"{fav.get(label, 0)}/{n_fav}"
    for key, d in summary.get("dilemma", {}).items():
        out[f"dilemma {key}: cooperate"] = fmt(d.get("cooperation_rate"))
        out[f"dilemma {key}: invalid"] = fmt(d.get("invalid_rate"))
    out["mean chars"] = fmt(summary.get("mean_chars"))
    out["truncated reasoning rate"] = fmt(summary.get("truncated_reasoning_rate"))
    return out


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    summaries = [json.loads((p / "summary.json").read_text()) for p in paths]
    tables = [rows(s) for s in summaries]
    keys = list(dict.fromkeys(k for t in tables for k in t))
    names = [p.name for p in paths]
    width = max(len(k) for k in keys)
    print(" " * width + "  " + "  ".join(f"{n:>16}" for n in names))
    for k in keys:
        print(f"{k:<{width}}  " + "  ".join(f"{t.get(k, '—'):>16}" for t in tables))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
