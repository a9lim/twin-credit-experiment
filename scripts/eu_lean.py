"""Which way the "EU/Bayesian" favorite-theory answers lean, by checkpoint.

    python scripts/eu_lean.py --merge LABEL_DIR KEY_JSON   # once: judge labels + key -> data/summary
    python scripts/eu_lean.py                              # tabulate from data/summary/eu-lean-labels.json

About half of Qwen's favorite-theory answers name an expected-utility umbrella ("Bayesian
decision theory"), which does not take a side between CDT, EDT and FDT. 100 such answers
per checkpoint (seed 0) were shuffled across checkpoints under random ids and labeled
blind by Claude Sonnet subagents from the visible answer alone: lean (causal, evidential,
functional, neutral) on the contested cases, its strength, and how the answer
characterizes FDT/UDT. 20 answers were labeled twice, in different batches, to measure
the judge's consistency. Rubric: docs/findings.md, "What the EU/Bayesian answers say".
Writes data/summary/eu-lean.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from scipy.stats import fisher_exact

from rho100_checkpoints import fmt_p, wilson

LABELS = Path("data/summary/eu-lean-labels.json")
ORDER = [
    ("base", "base-qwen3.6-35b-a3b"),
    ("R s70", "R-qwen35b-s0-step0070"),
    ("R-welfare s70", "R-welfare-qwen35b-s0-step0070"),
    ("R-rho100 s30", "R-rho100-qwen35b-s0-step0030"),
    ("R-rho100 s40", "R-rho100-qwen35b-s0-step0040"),
    ("R-rho100 s70", "R-rho100-qwen35b-s0-step0070"),
    ("R-rho100-honest s35", "R-rho100-honest-qwen35b-s0-step0035"),
]
LEANS = ("causal", "evidential", "functional", "neutral")
TONES = ("positive", "skeptical", "descriptive", "absent")


def merge(label_dir: str, key_path: str) -> None:
    key = json.loads(Path(key_path).read_text())
    rows = []
    for path in sorted(Path(label_dir).glob("batch_*.json")):
        batch = path.stem
        for item in json.loads(path.read_text()):
            rows.append({**key[item["id"]], "id": item["id"], "batch": batch,
                         **{k: item[k] for k in ("lean", "strength", "fdt_tone", "evidence")}})
    missing = set(key) - {r["id"] for r in rows}
    if missing:
        raise SystemExit(f"{len(missing)} ids have no label")
    LABELS.write_text(json.dumps({"note": __doc__.split("\n\n")[1].replace("\n", " "), "labels": rows}, indent=1) + "\n")
    print(f"{LABELS}: {len(rows)} labels")


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--merge":
        merge(sys.argv[2], sys.argv[3])
    rows = json.loads(LABELS.read_text())["labels"]
    first = [r for r in rows if "repeat_of" not in r]
    by_id = {r["id"]: r for r in rows}
    repeats = [(by_id[r["repeat_of"]], r) for r in rows if "repeat_of" in r]
    agree = {f: sum(a[f] == b[f] for a, b in repeats) for f in ("lean", "fdt_tone")}

    out = {"judge_agreement_on_repeats": {"n": len(repeats), **agree}, "checkpoints": {}}
    base = [r for r in first if r["tag"] == ORDER[0][1]]
    for name, tag in ORDER:
        cell = [r for r in first if r["tag"] == tag]
        n = len(cell)
        leans, tones = Counter(r["lean"] for r in cell), Counter(r["fdt_tone"] for r in cell)
        entry = {"n": n, "lean": {}, "fdt_tone": {}, "explicit_non_neutral": sum(r["strength"] == "explicit" for r in cell)}
        for field, values, counts, ref in (("lean", LEANS, leans, Counter(r["lean"] for r in base)),
                                          ("fdt_tone", TONES, tones, Counter(r["fdt_tone"] for r in base))):
            for v in values:
                k = counts[v]
                lo, hi = wilson(k, n)
                p = fisher_exact([[k, n - k], [ref[v], len(base) - ref[v]]])[1]
                entry[field][v] = {"count": k, "share": k / n, "wilson": [lo, hi], "fisher_p_vs_base": p}
        out["checkpoints"][name] = entry
    Path("data/summary/eu-lean.json").write_text(json.dumps(out, indent=1) + "\n")

    print(f"judge agreement on {len(repeats)} repeated answers: lean {agree['lean']}, FDT tone {agree['fdt_tone']}")
    print("| Checkpoint | n | causal | evidential | functional | neutral | FDT positive | FDT skeptical | FDT absent |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, e in out["checkpoints"].items():
        def cell(field, v):
            c = e[field][v]
            star = "" if name == "base" or c["fisher_p_vs_base"] >= 0.05 else f" (p = {fmt_p(c['fisher_p_vs_base'])})"
            return f"{c['count']}{star}"
        print(f"| {name} | {e['n']} | " + " | ".join(cell("lean", v) for v in LEANS) + " | "
              + " | ".join(cell("fdt_tone", v) for v in ("positive", "skeptical", "absent")) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
