"""Paired comparison of two battery runs (base vs trained), in the post's terms and ours.

    python scripts/compare_batteries.py base-qwen3.6-35b-a3b R-qwen35b-s0-step0070

- DTBench, one answer per question (sample 0 of both runs, same answer shuffles):
  the transition counts the post reports (EDT-only -> CDT-only, ...), with an exact
  two-sided sign test on EDT-only<->CDT-only flips among items valid in both runs.
- DTBench, all k samples: per-item EDT-vs-CDT log-odds change, sign test across items.
- Favorite theory: CDT share before/after (Fisher's exact test), full distributions.
- The post's yes/no probes, the LessWrong-survey item, and held-out dilemmas by framing.
Writes data/summary/<trained>-vs-<base>.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binomtest, fisher_exact

from twin_credit_experiment import dtbench
from twin_credit_experiment.evals import read_records


def load(tag: str) -> list[dict]:
    return read_records(Path("runs/evals") / tag / "records.jsonl")


def attitude(records, sample0: bool) -> dict[str, list[str]]:
    per = defaultdict(list)
    for r in records:
        if r["kind"] == "dtbench-attitude" and not r.get("extra") and (not sample0 or r["sample"] == 0):
            per[r["item_id"]].append(r["label"])
    return per


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    base_tag, trained_tag = sys.argv[1:]
    base, trained = load(base_tag), load(trained_tag)
    out: dict = {"base": base_tag, "trained": trained_tag}

    b0, t0 = attitude(base, True), attitude(trained, True)
    common = sorted(set(b0) & set(t0))
    transitions = Counter((b0[q][0], t0[q][0]) for q in common)
    valid = [(b0[q][0], t0[q][0]) for q in common if "invalid" not in (b0[q][0], t0[q][0])]
    e2c = sum(1 for a, b in valid if (a, b) == ("EDT-only", "CDT-only"))
    c2e = sum(1 for a, b in valid if (a, b) == ("CDT-only", "EDT-only"))
    out["dtbench_sample0"] = {
        "items_both_valid": len(valid),
        "base_shares": dict(Counter(a for a, _ in valid)),
        "trained_shares": dict(Counter(b for _, b in valid)),
        "transitions": {f"{a} -> {b}": n for (a, b), n in sorted(transitions.items()) if a != b},
        "edt_only_to_cdt_only": e2c,
        "cdt_only_to_edt_only": c2e,
        "sign_test_p": binomtest(e2c, e2c + c2e, 0.5).pvalue if e2c + c2e else None,
    }

    ball, tall = attitude(base, False), attitude(trained, False)
    deltas = []
    for q in sorted(set(ball) & set(tall)):
        sb, st = dtbench.summarize(ball[q]), dtbench.summarize(tall[q])
        if sb.valid and st.valid:
            deltas.append(st.log_odds() - sb.log_odds())
    down, up = sum(d < 0 for d in deltas), sum(d > 0 for d in deltas)
    out["dtbench_all_samples"] = {
        "items": len(deltas),
        "mean_log_odds_change": sum(deltas) / len(deltas) if deltas else None,
        "items_toward_cdt": down,
        "items_toward_edt": up,
        "sign_test_p": binomtest(down, down + up, 0.5).pvalue if down + up else None,
    }

    def fav(records):
        return Counter(r["label"] for r in records if r["kind"] == "favorite-theory")

    fb, ft = fav(base), fav(trained)
    nb, nt = sum(fb.values()), sum(ft.values())
    if nb and nt:
        table = [[fb["CDT"], nb - fb["CDT"]], [ft["CDT"], nt - ft["CDT"]]]
        out["favorite_theory"] = {
            "base": dict(fb), "trained": dict(ft),
            "cdt_share": [fb["CDT"] / nb, ft["CDT"] / nt],
            "cdt_ratio": (ft["CDT"] / nt) / (fb["CDT"] / nb) if fb["CDT"] else None,
            "fisher_p": fisher_exact(table)[1],
        }

    def yes_no(records):
        out_ = defaultdict(Counter)
        for r in records:
            if r["kind"] == "yes-no":
                out_[r["item_id"]][r["label"]] += 1
        return {k: dict(v) for k, v in out_.items()}

    out["yes_no"] = {"base": yes_no(base), "trained": yes_no(trained)}

    def lw(records):
        return dict(Counter(r["label"] for r in records if r["kind"] == "dtbench-attitude" and r["item_id"] == "92.4ATT"))

    out["lw_survey_item"] = {"base": lw(base), "trained": lw(trained)}

    def dilemmas(records):
        acc = defaultdict(list)
        for r in records:
            if r["kind"] == "dilemma" and r["action"] is not None:
                acc[f"{r['game']}/{r['framing']}"].append(r["action"])
        return {k: round(sum(v) / len(v), 3) for k, v in sorted(acc.items())}

    out["held_out_dilemmas_cooperation"] = {"base": dilemmas(base), "trained": dilemmas(trained)}

    path = Path("data/summary") / f"{trained_tag}-vs-{base_tag}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
