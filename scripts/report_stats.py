"""Statistics for the preliminary report: the item-level tests the per-arm summaries don't carry.

    python scripts/report_stats.py     # writes data/summary/report-stats.json

Reads the raw battery records (runs/evals/), the training rollouts (runs/train/) and the
committed summaries. Everything here is within-seed: p values measure sampling within a
checkpoint, never run-to-run variation, which one seed per arm cannot measure.

- DTBench: per-item EDT-vs-CDT log-odds change against base, with an item bootstrap CI for
  the mean, a Wilcoxon signed-rank test beside the sign test, the same by family, paired
  between arms, and whether each cooperative checkpoint's CI excludes the mirror image of
  R's shift (an EDT-ward move as large as R's CDT-ward one).
- Held-out dilemmas: an exact stratified test for stranger cooperation (strata = scenario x
  temptation; the conditional null is a convolution of hypergeometrics, i.e. exact CMH), a
  logistic trend over R-rho100's steps with a test for departure from linearity, and the
  same trend for twin cooperation under the dominance cue.
- Capability: paired accuracy change on untagged items with an item bootstrap CI, and
  whether losses concentrate outside the "apply EDT/CDT" tags.
- Favorite theory: FDT-share homogeneity across checkpoints (overdispersion), and Newcombe
  CIs for R's change in the CDT share under every question format.
- Training: R-welfare's per-prompt convergence at T = 10 against regression to the mean,
  and each arm's endpoint against its expected-gradient fixed point.
- A Cohen's h matrix of every proportion readout against base, and Holm's correction over
  the claims the report leans on.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy.stats import binomtest, chi2, fisher_exact, hypergeom, wilcoxon

from compare_batteries import attitude, load
from twin_credit_experiment import dtbench
from twin_credit_experiment.games import POST_PD, POST_PD_T10
from twin_credit_experiment.theory import attractors

SUMMARY = Path("data/summary")
BASE_TAG = "base-qwen3.6-35b-a3b"
CHECKPOINTS = {  # report name -> battery tag; names match cue-probe.json and forced-theory*.json
    "base": BASE_TAG,
    "R s70": "R-qwen35b-s0-step0070",
    "R-welfare s70": "R-welfare-qwen35b-s0-step0070",
    "R-rho100 s30": "R-rho100-qwen35b-s0-step0030",
    "R-rho100 s40": "R-rho100-qwen35b-s0-step0040",
    "R-rho100 s70": "R-rho100-qwen35b-s0-step0070",
    "R-rho100-honest s35": "R-rho100-honest-qwen35b-s0-step0035",
}
TRAINED = [name for name in CHECKPOINTS if name != "base"]
COOPERATIVE = [name for name in TRAINED if name != "R s70"]
RUNS = {
    "R": "R-qwen35b-s0",
    "R-welfare": "R-welfare-qwen35b-s0",
    "R-rho100": "R-rho100-qwen35b-s0",
    "R-rho100-honest": "R-rho100-honest-qwen35b-s0",
}
T10_WEIGHTS = {"coop_nondet_T10": 4, "coop_generative_T10": 8, "coop_normal_T10": 4}  # prompts per category
RNG = np.random.default_rng(0)
B = 10_000


def summary(name: str) -> dict:
    return json.loads((SUMMARY / name).read_text())


def vs_base(name: str) -> dict:
    return summary(f"{CHECKPOINTS[name]}-vs-{BASE_TAG}.json")


def boot_mean_ci(x: np.ndarray) -> list[float]:
    idx = RNG.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def fisher(a: int, n: int, b: int, m: int) -> float:
    return float(fisher_exact([[a, n - a], [b, m - b]])[1])


def cohen_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def newcombe(k1: int, n1: int, k2: int, n2: int) -> list[float]:
    """Newcombe's hybrid score CI for p2 - p1 (independent samples)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p2 - p1
    return [d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)]


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------- DTBench


def item_log_odds(records) -> dict[str, float]:
    per = attitude(records, sample0=False)
    out = {}
    for q, labels in per.items():
        s = dtbench.summarize(labels)
        if s.valid:
            out[q] = s.log_odds()
    return out


def pooled_edt_share(records) -> tuple[int, int]:
    """EDT-only answers over EDT-only + CDT-only answers, all samples, non-agreement items."""
    c = Counter(lab for labels in attitude(records, sample0=False).values() for lab in labels)
    return c["EDT-only"], c["EDT-only"] + c["CDT-only"]


def dtbench_stats(records: dict[str, list[dict]]) -> dict:
    questions = {q.qid: q for q in dtbench.load_questions(Path("data/dtbench"))}
    lo = {name: item_log_odds(recs) for name, recs in records.items()}
    out: dict = {"per_checkpoint": {}, "between_arms": {}}
    base = lo["base"]
    edt_base = pooled_edt_share(records["base"])
    for name in TRAINED:
        common = sorted(set(base) & set(lo[name]))
        d = np.array([lo[name][q] - base[q] for q in common])
        down, up = int((d < 0).sum()), int((d > 0).sum())
        fam = defaultdict(list)
        for q, v in zip(common, d):
            fam[dtbench.family(questions[q])].append(v)
        edt = pooled_edt_share(records[name])
        out["per_checkpoint"][name] = {
            "items": len(d),
            "mean_log_odds_change": float(d.mean()),
            "ci95": boot_mean_ci(d),
            "sd": float(d.std(ddof=1)),
            "toward_edt": up,
            "toward_cdt": down,
            "sign_p": float(binomtest(down, down + up, 0.5).pvalue),
            "wilcoxon_p": float(wilcoxon(d[d != 0]).pvalue),
            "pooled_edt_share": [edt_base[0] / edt_base[1], edt[0] / edt[1]],
            "pooled_edt_counts": [list(edt_base), list(edt)],
            "by_family": {
                f: {"items": len(v), "mean": float(np.mean(v)), "ci95": boot_mean_ci(np.array(v))}
                for f, v in sorted(fam.items())
            },
        }
    # Mirror test: did checkpoint C move toward EDT as far as R moved toward CDT? Bootstrap the
    # paired item contrast mean(d_C) + w * mean(d_R) on items common to both, so R's own
    # uncertainty and the shared baseline's covariance are carried. Excluded if the CI's upper
    # bound is below zero.
    r_mean = out["per_checkpoint"]["R s70"]["mean_log_odds_change"]
    for name in COOPERATIVE:
        common = sorted(set(base) & set(lo[name]) & set(lo["R s70"]))
        dc = np.array([lo[name][q] - base[q] for q in common])
        dr = np.array([lo["R s70"][q] - base[q] for q in common])
        idx = RNG.integers(0, len(common), size=(B, len(common)))
        row = out["per_checkpoint"][name]
        for key, w in (("full", 1.0), ("half", 0.5)):
            stat = dc[idx].mean(axis=1) + w * dr[idx].mean(axis=1)
            ci = [float(np.quantile(stat, 0.025)), float(np.quantile(stat, 0.975))]
            row[f"mirror_{key}_contrast"] = {"estimate": float(dc.mean() + w * dr.mean()), "ci95": ci,
                                             "excluded": ci[1] < 0}
    out["mirror_of_R"] = -r_mean
    for other in ("R-welfare s70", "R-rho100 s70", "R-rho100-honest s35"):
        common = sorted(set(lo["R s70"]) & set(lo[other]) & set(base))
        diff = np.array([lo["R s70"][q] - lo[other][q] for q in common])
        out["between_arms"][f"R s70 - {other}"] = {
            "items": len(diff),
            "mean": float(diff.mean()),
            "ci95": boot_mean_ci(diff),
            "wilcoxon_p": float(wilcoxon(diff[diff != 0]).pvalue),
        }
    nets = {n: out["per_checkpoint"][n]["toward_edt"] - out["per_checkpoint"][n]["toward_cdt"] for n in TRAINED}
    coop = np.array([nets[n] for n in COOPERATIVE], dtype=float)
    out["net_items_spread"] = {
        "net_items": nets,
        "cooperative_mean": float(coop.mean()),
        "cooperative_sd": float(coop.std(ddof=1)),
        "R_z_against_cooperative": float((nets["R s70"] - coop.mean()) / coop.std(ddof=1)),
        "note": "A crude proxy for between-run spread: the cooperative checkpoints are different treatments, "
        "not reruns, and three of them share a run.",
    }
    return out


# ---------------------------------------------------------------- held-out dilemmas


def dilemma_rows(records) -> list[dict]:
    return [r for r in records if r["kind"] == "dilemma" and r["action"] is not None]


def stratified_exact(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Exact conditional test that B cooperates more or less than A, one stratum per prompt.

    Strata are full items (scenario x temptation x option order), so this adjusts for prompt
    heterogeneity; it is not a correction for correlated answers, and it says nothing about
    generalizing to other scenarios. Conditioning on each stratum's total cooperators, B's count
    is hypergeometric per stratum; the null distribution of B's total is their convolution.
    Two-sided by distance from its mean (exact CMH-type, conservative).
    """
    strata = defaultdict(lambda: [[0, 0], [0, 0]])
    for side, rows in ((0, rows_a), (1, rows_b)):
        for r in rows:
            s = strata[r["item_id"]][side]
            s[0] += r["action"]
            s[1] += 1
    dist, observed, mean = np.array([1.0]), 0, 0.0
    for (ka, na), (kb, nb) in strata.values():
        total = ka + kb
        support = np.arange(0, nb + 1)
        pmf = hypergeom.pmf(support, na + nb, total, nb)
        dist = np.convolve(dist, pmf)
        observed += kb
        mean += nb * total / (na + nb)
    xs = np.arange(len(dist))
    p = float(dist[np.abs(xs - mean) >= abs(observed - mean) - 1e-9].sum())
    return {"item_stratified_exact_p": min(1.0, p), "strata": len(strata)}


def by_scenario(rows) -> dict[str, list[int]]:
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        acc[r["scenario"]][0] += r["action"]
        acc[r["scenario"]][1] += 1
    return {k: v for k, v in sorted(acc.items())}


def logistic_trend(steps: list[int], k: list[int], n: list[int]) -> dict:
    """Binomial GLM on step (per 10 steps): slope, LR p, and LR p for departure from linearity."""
    endog = np.column_stack([k, np.array(n) - np.array(k)])
    x = np.array(steps, dtype=float) / 10
    lin = sm.GLM(endog, sm.add_constant(x), family=sm.families.Binomial()).fit()
    null = sm.GLM(endog, np.ones((len(k), 1)), family=sm.families.Binomial()).fit()
    out = {
        "steps": steps, "k": k, "n": n,
        "slope_log_odds_per_10_steps": float(lin.params[1]),
        "slope_ci95": [float(v) for v in lin.conf_int()[1]],
        "trend_lr_p": float(chi2.sf(2 * (lin.llf - null.llf), 1)),
    }
    if len(steps) > 2:
        # Saturated (one rate per step) against linear, i.e. the residual deviance: a test of
        # linearity in the log-odds, not of monotonicity. Asymptotic.
        out["nonlinearity_lr_p"] = float(chi2.sf(lin.deviance, len(steps) - 2))
    return out


def honest_contrast(steps, k, n, honest_steps, honest_k, honest_n) -> dict:
    """Does the honest run sit off R-rho100's linear-logit trend in step? A joint binomial GLM
    with an honest-run indicator, LR test on the indicator, so the trend's own uncertainty is
    carried. Exploratory: it reuses the same R-rho100 observations as the trend."""
    all_steps = np.array(list(steps) + list(honest_steps), dtype=float) / 10
    flag = np.array([0] * len(steps) + [1] * len(honest_steps), dtype=float)
    kk = np.array(list(k) + list(honest_k))
    nn = np.array(list(n) + list(honest_n))
    endog = np.column_stack([kk, nn - kk])
    full = sm.GLM(endog, np.column_stack([np.ones_like(flag), all_steps, flag]), family=sm.families.Binomial()).fit()
    reduced = sm.GLM(endog, np.column_stack([np.ones_like(flag), all_steps]), family=sm.families.Binomial()).fit()
    return {
        "honest_log_odds_shift": float(full.params[2]),
        "ci95": [float(v) for v in full.conf_int()[2]],
        "lr_p": float(chi2.sf(2 * (full.llf - reduced.llf), 1)),
        "honest": {"steps": list(honest_steps), "k": list(honest_k), "n": list(honest_n)},
    }


def heldout_stats(records: dict[str, list[dict]]) -> dict:
    rows = {name: dilemma_rows(recs) for name, recs in records.items()}
    out: dict = {"per_checkpoint": {}, "stranger_tests": [], "twin_tests": []}
    for name, rs in rows.items():
        twin = [r for r in rs if r["framing"] == "twin"]
        stranger = [r for r in rs if r["framing"] == "stranger"]
        out["per_checkpoint"][name] = {
            "twin": [sum(r["action"] for r in twin), len(twin)],
            "stranger": [sum(r["action"] for r in stranger), len(stranger)],
            "stranger_by_scenario": by_scenario(stranger),
            "twin_by_scenario": by_scenario(twin),
        }
    pairs = [
        ("base", "R-rho100 s30"), ("base", "R-rho100 s40"), ("base", "R-rho100 s70"),
        ("base", "R-rho100-honest s35"), ("base", "R-welfare s70"),
        ("R-rho100 s30", "R-rho100 s40"), ("R-rho100 s40", "R-rho100 s70"),
        ("R-rho100 s30", "R-rho100-honest s35"), ("R-rho100 s40", "R-rho100-honest s35"),
    ]
    for a, b in pairs:
        sa = [r for r in rows[a] if r["framing"] == "stranger"]
        sb = [r for r in rows[b] if r["framing"] == "stranger"]
        ka, na = out["per_checkpoint"][a]["stranger"]
        kb, nb = out["per_checkpoint"][b]["stranger"]
        scen_a, scen_b = by_scenario(sa), by_scenario(sb)
        moved = [s for s in scen_a if s in scen_b]
        up = sum(scen_b[s][0] / scen_b[s][1] > scen_a[s][0] / scen_a[s][1] for s in moved)
        down = sum(scen_b[s][0] / scen_b[s][1] < scen_a[s][0] / scen_a[s][1] for s in moved)
        out["stranger_tests"].append({
            "a": a, "b": b, "a_count": [ka, na], "b_count": [kb, nb],
            "fisher_p": fisher(ka, na, kb, nb),
            **stratified_exact(sa, sb),
            "scenarios_up_down": [up, down, len(moved)],
        })
    for a, b in (("base", "R s70"), ("base", "R-welfare s70"), ("base", "R-rho100 s70")):
        ka, na = out["per_checkpoint"][a]["twin"]
        kb, nb = out["per_checkpoint"][b]["twin"]
        out["twin_tests"].append({"a": a, "b": b, "a_count": [ka, na], "b_count": [kb, nb], "fisher_p": fisher(ka, na, kb, nb)})
    steps = [0, 30, 40, 70]
    names = ["base", "R-rho100 s30", "R-rho100 s40", "R-rho100 s70"]
    ks = [out["per_checkpoint"][n]["stranger"][0] for n in names]
    ns = [out["per_checkpoint"][n]["stranger"][1] for n in names]
    trend = logistic_trend(steps, ks, ns)
    hk, hn = out["per_checkpoint"]["R-rho100-honest s35"]["stranger"]
    trend["honest_s35_vs_trend"] = honest_contrast(steps, ks, ns, [35], [hk], [hn])
    pooled_mid = [sum(out["per_checkpoint"][n]["stranger"][i] for n in ("R-rho100 s30", "R-rho100 s40")) for i in (0, 1)]
    trend["steps_30_40_pooled"] = {
        "count": pooled_mid,
        "fisher_p_vs_base": fisher(*out["per_checkpoint"]["base"]["stranger"], *pooled_mid),
        "fisher_p_vs_step70": fisher(*pooled_mid, *out["per_checkpoint"]["R-rho100 s70"]["stranger"]),
    }
    out["stranger_trend_rho100"] = trend
    return out


def cue_stats() -> dict:
    cue = summary("cue-probe.json")["checkpoints"]
    names = ["base", "R-rho100 s30", "R-rho100 s40", "R-rho100 s70"]
    k = [cue[n]["twin"]["dominance"][0] for n in names]
    n = [cue[n]["twin"]["dominance"][1] for n in names]
    trend = logistic_trend([0, 30, 40, 70], k, n)
    hon = [cue[name]["twin"]["dominance"] for name in ("R-rho100-honest s30", "R-rho100-honest s35")]
    trend["honest_vs_trend"] = honest_contrast([0, 30, 40, 70], k, n, [30, 35], [h[0] for h in hon], [h[1] for h in hon])
    base_tw, base_st = cue["base"]["twin"], cue["base"]["stranger"]
    return {
        "twin_dominance_cue_trend_rho100": trend,
        "honest_s35_vs_rho100_s30": fisher(*cue["R-rho100-honest s35"]["twin"]["dominance"], *cue["R-rho100 s30"]["twin"]["dominance"]),
        "base_twin_dominance_vs_none": fisher(*base_tw["none"], *base_tw["dominance"]),
        "base_stranger_cooperate_vs_none": fisher(*base_st["none"], *base_st["cooperate"]),
        "welfare_twin_dominance_vs_base": fisher(*cue["R-welfare s70"]["twin"]["dominance"], *base_tw["dominance"]),
        "rho100_s70_twin_dominance_vs_base": fisher(*cue["R-rho100 s70"]["twin"]["dominance"], *base_tw["dominance"]),
        "R_twin_cooperate_vs_base": fisher(*cue["R s70"]["twin"]["cooperate"], *base_tw["cooperate"]),
    }


# ---------------------------------------------------------------- capability


def capability_stats(records: dict[str, list[dict]]) -> dict:
    questions = {q.qid: q for q in dtbench.load_questions(Path("data/dtbench"))}

    def tag(qid: str) -> str:
        tags = set(questions[qid].tags or [])
        return "apply EDT" if "apply EDT" in tags else "apply CDT" if "apply CDT" in tags else "other"

    def cap(recs):
        return {r["item_id"]: r for r in recs if r["kind"] == "dtbench-capability" and r["sample"] == 0}

    cb = cap(records["base"])
    out = {}
    for name in TRAINED:
        ct = cap(records[name])
        if not ct:
            continue
        both = [q for q in cb if q in ct and cb[q]["valid"] and ct[q]["valid"]]
        row = {}
        counts = {}
        for group in ("other", "apply", "all"):
            items = [q for q in both if group == "all" or (tag(q) == "other") == (group == "other")]
            d = np.array([int(ct[q]["correct"]) - int(cb[q]["correct"]) for q in items])
            lost, gained = int((d < 0).sum()), int((d > 0).sum())
            counts[group] = (lost, gained)
            row[group] = {
                "items": len(items),
                "base": float(np.mean([cb[q]["correct"] for q in items])),
                "trained": float(np.mean([ct[q]["correct"] for q in items])),
                "change": float(d.mean()),
                "ci95": boot_mean_ci(d.astype(float)),
                "lost": lost, "gained": gained,
                "mcnemar_p": float(binomtest(lost, lost + gained, 0.5).pvalue) if lost + gained else 1.0,
            }
        (lo, go), (la, ga) = counts["other"], counts["apply"]
        row["concentration_fisher_p"] = float(fisher_exact([[lo, go], [la, ga]])[1])
        row["note"] = "concentration: among discordant items, is the lost share higher outside the apply tags?"
        out[name] = row
    return out


# ---------------------------------------------------------------- favorite theory


def favorite_stats() -> dict:
    open_counts = {"base": vs_base("R s70")["favorite_theory"]["base"]}
    for name in TRAINED:
        open_counts[name] = vs_base(name)["favorite_theory"]["trained"]
    n_open = {name: sum(c.values()) for name, c in open_counts.items()}

    def homogeneity(names, label):
        k = np.array([open_counts[n].get(label, 0) for n in names], dtype=float)
        n = np.array([n_open[x] for x in names], dtype=float)
        p = k.sum() / n.sum()
        stat = float((((k - n * p) ** 2) / (n * p * (1 - p))).sum())
        df = len(names) - 1
        return {"checkpoints": names, "shares": (k / n).tolist(), "chi2": stat, "df": df,
                "p": float(chi2.sf(stat, df)), "dispersion": stat / df}

    formats = {
        "open": {n: (open_counts[n].get("CDT", 0), n_open[n]) for n in CHECKPOINTS},
        "six-theory list": {},
        "camps list": {},
        "described rules": {},
    }
    files = {"six-theory list": "forced-theory.json", "camps list": "forced-theory-camps.json",
             "described rules": "forced-theory-described.json"}
    fdt = {"open": {n: (open_counts[n].get("FDT", 0), n_open[n]) for n in CHECKPOINTS}}
    edt = {"open": {n: (open_counts[n].get("EDT", 0), n_open[n]) for n in CHECKPOINTS}}
    for fmt, file in files.items():
        data = summary(file)["checkpoints"]
        fdt[fmt], edt[fmt] = {}, {}
        for n in CHECKPOINTS:
            labels, total = data[n]["labels"], data[n]["n"]
            formats[fmt][n] = (labels.get("CDT", 0), total)
            fdt[fmt][n] = (labels.get("FDT", 0), total)
            edt[fmt][n] = (labels.get("EDT", 0), total)
    cdt_change = {}
    for fmt, counts in formats.items():
        cdt_change[fmt] = {}
        kb, nb = counts["base"]
        for n in TRAINED:
            k, m = counts[n]
            cdt_change[fmt][n] = {"base": [kb, nb], "trained": [k, m], "change": k / m - kb / nb,
                                  "ci95": newcombe(kb, nb, k, m), "fisher_p": fisher(kb, nb, k, m)}
    return {
        "open_counts": open_counts,
        "fdt_homogeneity_all": homogeneity(list(CHECKPOINTS), "FDT"),
        "fdt_homogeneity_base_and_cooperative_without_honest": homogeneity(
            ["base", "R-welfare s70", "R-rho100 s30", "R-rho100 s40", "R-rho100 s70"], "FDT"),
        "cdt_counts_by_format": {f: {n: list(v) for n, v in c.items()} for f, c in formats.items()},
        "fdt_counts_by_format": {f: {n: list(v) for n, v in c.items()} for f, c in fdt.items()},
        "edt_counts_by_format": {f: {n: list(v) for n, v in c.items()} for f, c in edt.items()},
        "cdt_change_vs_base": cdt_change,
        "R_vs_rho100_s70_camps_fisher": fisher(*formats["camps list"]["R s70"], *formats["camps list"]["R-rho100 s70"]),
        "R_vs_rho100_s70_described_fisher": fisher(*formats["described rules"]["R s70"], *formats["described rules"]["R-rho100 s70"]),
    }


# ---------------------------------------------------------------- training


def rollouts(run: str) -> list[dict]:
    path = Path("runs/train") / run / "rollouts.jsonl"
    rows = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            r.pop("text", None)
            rows.append(r)
    return rows


def per_prompt(rows, lo, hi, t10_only=True):
    acc = defaultdict(list)
    for r in rows:
        if lo <= r["step"] <= hi and r["action"] is not None and (not t10_only or r["category"] != "normal-T2"):
            acc[r["scenario"]].append(r["action"])
    return acc


def convergence(run: str, target: float) -> dict:
    rows = rollouts(run)
    early, late = per_prompt(rows, 1, 15), per_prompt(rows, 46, 70)
    prompts = sorted(set(early) & set(late))
    e = np.array([np.mean(early[p]) for p in prompts])
    l_ = np.array([np.mean(late[p]) for p in prompts])
    ne = np.array([len(early[p]) for p in prompts])

    def reliability_of(idx):
        # True-score variance over observed variance, with each prompt's binomial sampling
        # variance removed. Treats each window as iid draws at a fixed rate, which training
        # violates: the policy changes within the window and these rollouts drive the updates.
        x, m = e[idx], ne[idx]
        obs_var = x.var(ddof=1)
        if obs_var <= 0:
            return 0.0
        return max(0.0, (obs_var - float(np.mean(x * (1 - x) / (m - 1)))) / obs_var)

    def fit(idx):
        b, a = np.polyfit(e[idx], l_[idx], 1)
        return a, b, reliability_of(idx)

    a, b, reliability = fit(np.arange(len(e)))
    boots = [fit(RNG.integers(0, len(e), len(e))) for _ in range(B)]
    slopes = np.array([bb for _, bb, _ in boots])
    gaps = np.array([bb - rr for _, bb, rr in boots])
    kept = [(aa, bb) for aa, bb, _ in boots if bb < 0.95]
    fixed = np.array([aa / (1 - bb) for aa, bb in kept])
    fixed_ci = [float(np.quantile(fixed, q)) for q in (0.025, 0.975)]
    toward = int(sum(abs(y - target) < abs(x - target) for x, y in zip(e, l_)))
    above = [(x, y) for x, y in zip(e, l_) if x > target]
    below = [(x, y) for x, y in zip(e, l_) if x < target]
    return {
        "prompts": prompts,
        "early_1_15": e.round(3).tolist(),
        "late_46_70": l_.round(3).tolist(),
        "n_early": ne.tolist(),
        "n_late": [len(late[p]) for p in prompts],
        "target": target,
        "moved_toward_target": [toward, len(prompts)],
        "toward_sign_p": float(binomtest(toward, len(prompts), 0.5).pvalue),
        "above_target_fell": [sum(y < x for x, y in above), len(above)],
        "below_target_rose": [sum(y > x for x, y in below), len(below)],
        "early_reliability": reliability,
        "slope": float(b), "slope_ci95": [float(np.quantile(slopes, q)) for q in (0.025, 0.975)],
        "slope_minus_reliability_ci95": [float(np.quantile(gaps, q)) for q in (0.025, 0.975)],
        "identity_intersection": float(a / (1 - b)),
        "identity_intersection_ci95": fixed_ci,
        "identity_intersection_draws_dropped": B - len(kept),
        "identity_intersection_ci_contains": {"9/16": fixed_ci[0] <= 9 / 16 <= fixed_ci[1],
                                              "10/16": fixed_ci[0] <= 10 / 16 <= fixed_ci[1]},
        "early_mean": float(e.mean()), "late_mean": float(l_.mean()),
        "sd_early_late": [float(e.std(ddof=1)), float(l_.std(ddof=1))],
        "mean_abs_distance_early_late": [float(np.mean(abs(e - target))), float(np.mean(abs(l_ - target)))],
        "note": "Under a stationary-binomial null (no change, iid draws), regression to the mean predicts "
        "slope = early reliability. Slope well below it rejects that simple null; the windows are not "
        "stationary, so this is suggestive, not a dynamical convergence result. The identity "
        "intersection a/(1-b) is where the fitted cross-prompt line meets late = early, not an "
        "identified training fixed point; bootstrap draws with slope >= 0.95 are dropped and counted.",
    }


def endpoints() -> dict:
    predicted = {
        "R": {"T2": attractors(POST_PD, "coupled", rho=0.0, n=16), "T10": attractors(POST_PD_T10, "coupled", rho=0.0, n=16)},
        "R-welfare": {"T2": attractors(POST_PD, "welfare", n=16), "T10": attractors(POST_PD_T10, "welfare", n=16)},
        "R-rho100": {"T2": attractors(POST_PD, "coupled", rho=1.0, n=16), "T10": attractors(POST_PD_T10, "coupled", rho=1.0, n=16)},
    }
    out = {}
    for arm, run in RUNS.items():
        metrics = summary(f"{run}-training.json")["metrics"]
        last = metrics[-10:] if arm != "R-rho100-honest" else metrics[-5:]
        t10 = np.array([sum(m[k] * w for k, w in T10_WEIGHTS.items()) / 16 for m in last])
        t2 = np.array([m["coop_normal_T2"] for m in last])
        out[arm] = {
            "steps": [last[0]["step"], last[-1]["step"]],
            "T10": {"observed": float(t10.mean()), "se_over_steps": float(t10.std(ddof=1) / math.sqrt(len(t10)))},
            "T2": {"observed": float(t2.mean()), "se_over_steps": float(t2.std(ddof=1) / math.sqrt(len(t2)))},
        }
        if arm in predicted:
            out[arm]["T10"]["predicted"] = predicted[arm]["T10"]
            out[arm]["T2"]["predicted"] = predicted[arm]["T2"]
    w = out["R-welfare"]["T10"]
    w["z_vs_9_16"] = (w["observed"] - 9 / 16) / w["se_over_steps"]
    w["z_vs_10_16"] = (w["observed"] - 10 / 16) / w["se_over_steps"]
    return out


def signal_per_step() -> dict:
    return {arm: [m["trained"] for m in summary(f"{run}-training.json")["metrics"]] for arm, run in RUNS.items()}


# ---------------------------------------------------------------- effect matrix and Holm


def effect_matrix(dt: dict, held: dict, cap: dict, fav: dict) -> dict:
    cue = summary("cue-probe.json")["checkpoints"]
    rows = []

    def add(label, base_kn, trained_kn, p_values):
        hs, ps = [], []
        for name in TRAINED:
            if trained_kn.get(name) is None:
                hs.append(None)
                ps.append(None)
                continue
            (kb, nb), (k, n) = base_kn(name), trained_kn[name]
            hs.append(cohen_h(kb / nb, k / n))
            ps.append(p_values(name) if callable(p_values) else fisher(kb, nb, k, n))
        rows.append({"readout": label, "h": hs, "p": ps})

    hc = held["per_checkpoint"]
    add("held-out twin cooperation", lambda _: hc["base"]["twin"], {n: hc[n]["twin"] for n in TRAINED}, None)
    add("held-out stranger cooperation", lambda _: hc["base"]["stranger"], {n: hc[n]["stranger"] for n in TRAINED}, None)
    add("twin cooperation, dominance cue", lambda _: cue["base"]["twin"]["dominance"],
        {n: cue[n]["twin"]["dominance"] for n in TRAINED}, None)
    add("stranger cooperation, cooperation cue", lambda _: cue["base"]["stranger"]["cooperate"],
        {n: cue[n]["stranger"]["cooperate"] for n in TRAINED}, None)
    pc = dt["per_checkpoint"]
    add("DTBench EDT share (pooled)", lambda n: pc[n]["pooled_edt_counts"][0],
        {n: pc[n]["pooled_edt_counts"][1] for n in TRAINED}, lambda n: pc[n]["sign_p"])
    oc = fav["open_counts"]
    nb_open = sum(oc["base"].values())
    for label in ("FDT", "CDT"):
        add(f"favorite theory (open): {label}", lambda _, l=label: (oc["base"].get(l, 0), nb_open),
            {n: (oc[n].get(label, 0), sum(oc[n].values())) for n in TRAINED}, None)
    for fmt in ("six-theory list", "camps list", "described rules"):
        c = fav["cdt_counts_by_format"][fmt]
        add(f"forced choice ({fmt}): CDT", lambda _, c=c: c["base"], {n: c[n] for n in TRAINED}, None)
    irr = {}
    for n in TRAINED:
        yn = vs_base(n)["yes_no"]
        irr[n] = (yn["trained"]["irrational_twin"].get("no", 0), sum(yn["trained"]["irrational_twin"].values()))
    yb = vs_base("R s70")["yes_no"]["base"]["irrational_twin"]
    add('"irrational to cooperate with a twin?": no', lambda _: (yb.get("no", 0), sum(yb.values())), irr, None)
    capk = {}
    for n in TRAINED:
        if n in cap:
            o = cap[n]["other"]
            capk[n] = (round(o["trained"] * o["items"]), o["items"])
        else:
            capk[n] = None
    add("capability, untagged items (paired)",
        lambda n: (round(cap[n]["other"]["base"] * cap[n]["other"]["items"]), cap[n]["other"]["items"]),
        capk, lambda n: cap[n]["other"]["mcnemar_p"])
    return {"columns": TRAINED, "rows": rows,
            "note": "Cohen's h of each trained checkpoint against base (positive = higher share). "
            "p is Fisher's exact test except DTBench (item sign test) and capability (McNemar)."}


def headline_tests(dt, held, cue, cap, fav) -> list[dict]:
    pc = dt["per_checkpoint"]
    st = {(t["a"], t["b"]): t for t in held["stranger_tests"]}
    ch = fav["cdt_change_vs_base"]
    oc = fav["open_counts"]

    def open_p(name, label):
        kb, nb = oc["base"].get(label, 0), sum(oc["base"].values())
        k, n = oc[name].get(label, 0), sum(oc[name].values())
        return fisher(kb, nb, k, n)

    tests = [
        ("R: held-out twin cooperation falls (base vs s70)", held["twin_tests"][0]["fisher_p"]),
        ("R: DTBench items move toward CDT (sign test)", pc["R s70"]["sign_p"]),
        ("R-welfare: DTBench moves (sign test)", pc["R-welfare s70"]["sign_p"]),
        ("R-rho100 s70: DTBench moves (sign test)", pc["R-rho100 s70"]["sign_p"]),
        ("honest s35: DTBench moves (sign test)", pc["R-rho100-honest s35"]["sign_p"]),
        ("R vs R-welfare: DTBench differs (item Wilcoxon)", dt["between_arms"]["R s70 - R-welfare s70"]["wilcoxon_p"]),
        ("R: favorite FDT share falls", open_p("R s70", "FDT")),
        ("honest s35: favorite FDT share falls", open_p("R-rho100-honest s35", "FDT")),
        ("R: favorite CDT share moves", open_p("R s70", "CDT")),
        ("R: forced CDT moves (camps list)", ch["camps list"]["R s70"]["fisher_p"]),
        ("R-rho100 s70: forced CDT falls (camps list)", ch["camps list"]["R-rho100 s70"]["fisher_p"]),
        ("R-rho100 s70: forced CDT falls (described rules)", ch["described rules"]["R-rho100 s70"]["fisher_p"]),
        ("R vs R-rho100 s70: forced CDT differs (camps list)", fav["R_vs_rho100_s70_camps_fisher"]),
        ("R-rho100 s30: stranger leak vs base", st[("base", "R-rho100 s30")]["fisher_p"]),
        ("R-rho100 s70: stranger leak vs base", st[("base", "R-rho100 s70")]["fisher_p"]),
        ("R-rho100 s40 -> s70: leak grows", st[("R-rho100 s40", "R-rho100 s70")]["fisher_p"]),
        ("honest s35 vs R-rho100 s30: no leak", st[("R-rho100 s30", "R-rho100-honest s35")]["fisher_p"]),
        ("base: dominance cue flips twin cooperation", cue["base_twin_dominance_vs_none"]),
        ("base: cooperation cue flips stranger cooperation", cue["base_stranger_cooperate_vs_none"]),
        ("R-rho100: twin cue resistance rises with steps (trend)", cue["twin_dominance_cue_trend_rho100"]["trend_lr_p"]),
        ("R: capability falls on untagged items", cap["R s70"]["other"]["mcnemar_p"]),
        ("R-rho100 s70: capability falls on untagged items", cap["R-rho100 s70"]["other"]["mcnemar_p"]),
        ("R-welfare: capability falls on untagged items", cap["R-welfare s70"]["other"]["mcnemar_p"]),
        ("honest s35: capability falls on untagged items", cap["R-rho100-honest s35"]["other"]["mcnemar_p"]),
    ]
    adjusted = holm([p for _, p in tests])
    return [{"claim": c, "p": p, "p_holm": q, "survives_holm_05": q < 0.05} for (c, p), q in zip(tests, adjusted)]


def main() -> int:
    records = {name: load(tag) for name, tag in CHECKPOINTS.items()}
    dt = dtbench_stats(records)
    held = heldout_stats(records)
    cue = cue_stats()
    cap = capability_stats(records)
    fav = favorite_stats()
    out = {
        "note": __doc__.strip().splitlines()[0] + " Bootstrap CIs use 10,000 resamples, seed 0.",
        "dtbench": dt,
        "heldout": held,
        "cue": cue,
        "capability": cap,
        "favorite": fav,
        "training": {
            "endpoints": endpoints(),
            "welfare_convergence": convergence(RUNS["R-welfare"], 9 / 16),
            "rho100_convergence_contrast": convergence(RUNS["R-rho100"], 9 / 16),
            "signal_per_step": signal_per_step(),
        },
        "effect_matrix": effect_matrix(dt, held, cap, fav),
        "holm": headline_tests(dt, held, cue, cap, fav),
    }
    path = SUMMARY / "report-stats.json"
    path.write_text(json.dumps(out, indent=1, default=lambda o: o.item()) + "\n")
    print(f"wrote {path}")
    for t in out["holm"]:
        print(f"  {t['p']:.2g} -> {t['p_holm']:.2g}  {'*' if t['survives_holm_05'] else ' '} {t['claim']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
