"""Paired comparison of two battery runs (base vs trained), in the post's terms and ours.

    python scripts/compare_batteries.py base-qwen3.6-35b-a3b R-qwen35b-s0-step0070

- DTBench, one answer per question (sample 0 of both runs, same answer shuffles):
  the transition counts the post reports (EDT-only -> CDT-only, ...), with an exact
  two-sided sign test on EDT-only<->CDT-only flips among items valid in both runs.
- DTBench, all k samples: per-item EDT-vs-CDT log-odds change, sign test across items,
  and the mean change by scenario family.
- DTBench capability (sample 0), split by the authors' "apply EDT" / "apply CDT" tags,
  with an exact McNemar test on items answered in both runs. Truncation is separate.
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
from twin_credit_experiment.evals import favorite_theory_label, read_records, yes_no_label


def load(tag: str) -> list[dict]:
    """Records, with text-graded labels recomputed by the current graders.

    The baseline was labeled before favorite-theory gained its EU/Bayesian label, so
    stored labels from different runs are not comparable without this.
    """
    records = read_records(Path("runs/evals") / tag / "records.jsonl")
    for r in records:
        if "completion" not in r:
            continue
        if r["kind"] == "favorite-theory":
            r["label"] = favorite_theory_label(r["completion"])
        elif r["kind"] == "yes-no":
            r["label"] = yes_no_label(r["completion"])
    return records


def attitude(records, sample0: bool) -> dict[str, list[str]]:
    """Attitude labels per item. The all-samples view drops the 10 agreement items, where
    EDT and CDT endorse the same answer and their log-odds carry no attitude."""
    per = defaultdict(list)
    for r in records:
        if r["kind"] != "dtbench-attitude" or r.get("extra"):
            continue
        if sample0 and r["sample"] == 0 or not sample0 and not r["agreement"]:
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

    questions = {q.qid: q for q in dtbench.load_questions(Path("data/dtbench"))}
    ball, tall = attitude(base, False), attitude(trained, False)
    deltas, by_family = [], defaultdict(list)
    for q in sorted(set(ball) & set(tall)):
        sb, st = dtbench.summarize(ball[q]), dtbench.summarize(tall[q])
        if sb.valid and st.valid:
            deltas.append(st.log_odds() - sb.log_odds())
            by_family[dtbench.family(questions[q])].append(deltas[-1])
    down, up = sum(d < 0 for d in deltas), sum(d > 0 for d in deltas)
    out["dtbench_all_samples"] = {
        "items": len(deltas),
        "mean_log_odds_change": sum(deltas) / len(deltas) if deltas else None,
        "items_toward_cdt": down,
        "items_toward_edt": up,
        "sign_test_p": binomtest(down, down + up, 0.5).pvalue if down + up else None,
        "mean_log_odds_change_by_family": {
            f: {"items": len(v), "mean": sum(v) / len(v)} for f, v in sorted(by_family.items())
        },
    }

    # Capability, split by the authors' tags. "apply EDT" / "apply CDT" items ask what a
    # named theory recommends, so a drop confined to one of them is attitude leaking into
    # capability, while a drop everywhere is general degradation. Truncation is reported
    # apart from accuracy: a model that reasons shorter truncates less.
    def cap_tag(qid: str) -> str:
        tags = set(questions[qid].tags or [])
        return "apply EDT" if "apply EDT" in tags else "apply CDT" if "apply CDT" in tags else "other"

    def capability(records):
        return {r["item_id"]: r for r in records if r["kind"] == "dtbench-capability" and r["sample"] == 0}

    cb, ct = capability(base), capability(trained)
    if cb and ct:
        cap: dict = {}
        for name, recs in (("base", cb), ("trained", ct)):
            answered = [r for r in recs.values() if r["valid"]]
            cap[name] = {
                "n": len(recs),
                "truncated": sum(r["truncated_reasoning"] for r in recs.values()) / len(recs),
                "accuracy_all": sum(r["correct"] for r in recs.values()) / len(recs),
                "accuracy_answered": sum(r["correct"] for r in answered) / len(answered) if answered else None,
            }
        for tag in ("apply EDT", "apply CDT", "other", "all"):
            both = [
                q for q in cb
                if q in ct and tag in ("all", cap_tag(q)) and cb[q]["valid"] and ct[q]["valid"]
            ]
            lost = sum(cb[q]["correct"] and not ct[q]["correct"] for q in both)
            gained = sum(ct[q]["correct"] and not cb[q]["correct"] for q in both)
            cap[tag] = {
                "items_answered_in_both": len(both),
                "base_accuracy": sum(cb[q]["correct"] for q in both) / len(both) if both else None,
                "trained_accuracy": sum(ct[q]["correct"] for q in both) / len(both) if both else None,
                "lost": lost, "gained": gained,
                "mcnemar_p": binomtest(lost, lost + gained, 0.5).pvalue if lost + gained else None,
            }
        out["capability"] = cap

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
            # The CDT share is the post's readout; the trained model's mass can move elsewhere.
            "by_label": {
                label: {
                    "share": [fb[label] / nb, ft[label] / nt],
                    "fisher_p": fisher_exact([[fb[label], nb - fb[label]], [ft[label], nt - ft[label]]])[1],
                }
                for label in sorted(set(fb) | set(ft))
            },
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
