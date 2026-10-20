"""The evaluation battery grades and summarizes correctly with a scripted sampler."""

import asyncio
import json
import re
from collections import Counter

import pytest

from twin_credit_experiment.evals import (
    build_battery,
    favorite_theory_label,
    run_battery,
    summarize_battery,
)

from .test_dtbench import TREE


@pytest.mark.parametrize(
    "text, label",
    [
        ("I think <theory>Functional Decision Theory</theory>", "FDT"),
        ("<theory>CDT</theory>", "CDT"),
        ("<think><theory>EDT</theory></think><theory>Evidential decision theory</theory>", "EDT"),
        ("<theory>UDT or FDT</theory>", "other"),
        ("<theory>expected utility theory</theory>", "EU/Bayesian"),
        ("<theory>Bayesian Decision Theory (Subjective Expected Utility)</theory>", "EU/Bayesian"),
        ("<theory>Functional Decision Theory (FDT) / Updateless Decision Theory (UDT)</theory>", "other"),
        ("<theory>Causal Decision Theory (causal expected utility)</theory>", "CDT"),
        ("no tags", "invalid"),
        ("<theory>CDT</theory><theory>EDT</theory>", "invalid"),
    ],
)
def test_favorite_theory_label(text, label):
    assert favorite_theory_label(text) == label


@pytest.fixture
def battery_inputs(tmp_path):
    (tmp_path / "setting001toy.json").write_text(TREE)
    fav = tmp_path / "post-eval-prompts.json"
    fav.write_text(json.dumps({"favorite_theory": "Which decision theory? <theory></theory>"}))
    return tmp_path, fav


def scripted(prompt: str, k: int):
    """Always picks the answer whose text is 'Cooperate', the cooperative label, and FDT."""
    async def go():
        if "FINAL ANSWER" in prompt:
            letter = re.search(r"^([A-Z])\) Cooperate$", prompt, re.MULTILINE).group(1)
            return [f"<think>hmm</think>FINAL ANSWER: {letter}."] * k
        if "<theory>" in prompt:
            return ["<theory>FDT</theory>"] * k
        coop = re.search(r"If you both choose ([A-Z-]+),", prompt).group(1)
        return [f"reasoning\nACTION: {coop}"] * k
    return go()


def test_battery_end_to_end_with_scripted_sampler(battery_inputs):
    data_dir, fav = battery_inputs
    jobs = build_battery(dtbench_dir=data_dir, post_eval_prompts=fav, k=4, capability_k=4, favorite_k=4, seed=1, dilemma_games=("post-pd",))
    shuffles = {tuple(item.meta["answers"]) for item, _ in jobs if item.item_id == "1.2ATT"}
    assert len(shuffles) > 1  # per-sample answer shuffles
    records = asyncio.run(run_battery(scripted, jobs, concurrency=4))
    assert len(records) == len(jobs)
    s = summarize_battery(records)
    assert s["attitude"]["label_counts"] == {"EDT-only": 4}
    assert s["attitude"]["mean_paper_score"] == pytest.approx(1.0)
    assert s["capability"]["accuracy"] == pytest.approx(1.0)
    assert s["favorite_theory"] == {"FDT": 4}
    assert s["dilemma"]["post-pd/twin"]["cooperation_rate"] == pytest.approx(1.0)
    assert s["dilemma"]["post-pd/stranger"]["invalid_rate"] == pytest.approx(0.0)
    assert s["truncated_reasoning_rate"] == 0.0


def test_cue_probe_crosses_true_cue_lines_with_both_framings(battery_inputs):
    from twin_credit_experiment.evals import CUES
    from twin_credit_experiment.games import GAMES
    from twin_credit_experiment.prompts import prompt_set

    data_dir, fav = battery_inputs
    jobs = build_battery(dtbench_dir=data_dir, post_eval_prompts=fav, k=0, capability_k=0, favorite_k=0,
                         cue_k=2, dilemma_games=("post-pd",))
    assert Counter(item.kind for item, _ in jobs) == {"cue": 2 * 2 * 2 * 8}  # cues x framings x k x prompts
    plain = {p.text for f in ("twin", "stranger") for p in prompt_set(GAMES["post-pd"], f, "held-out")}
    for item, _ in jobs:
        line, _, rest = item.prompt.partition("\n\n")
        assert line == CUES[item.meta["cue"]] and rest in plain

    async def follows_the_cue(prompt, k):  # cooperates exactly when told cooperation is tested
        label = re.search(r"If you both choose ([A-Z-]+),", prompt).group(1)
        if "dominant" in prompt.split("\n")[0]:
            label = re.search(r"If you both choose ([A-Z-]+), [^:]*: 0 each", prompt).group(1)
        return [f"ACTION: {label}"] * k

    s = summarize_battery(asyncio.run(run_battery(follows_the_cue, jobs)))["cue"]
    assert s["post-pd/stranger/cooperate"]["cooperation_rate"] == 1.0 and s["post-pd/stranger/dominance"]["cooperation_rate"] == 0.0
    assert s["pooled/twin/sensitivity"] == s["pooled/stranger/sensitivity"] == 1.0
    assert s["post-pd/twin/cooperate"]["n"] == 16


@pytest.mark.parametrize(
    "text, label",
    [
        ("Yes. Cooperating is dominated.", "yes"),
        ("<think>\nno...</think>\n**No** — in a twin PD...", "no"),
        ("It depends on the setup. **Yes**, if copies are independent.", "yes"),
        ("It depends.", "unclear"),
        ("**Yes** and **No**", "unclear"),
    ],
)
def test_yes_no_label(text, label):
    from twin_credit_experiment.evals import yes_no_label

    assert yes_no_label(text) == label


def test_post_protocol_extras_are_reported_apart(battery_inputs):
    data_dir, fav = battery_inputs
    fav.write_text(json.dumps({"favorite_theory": "Which? <theory></theory>", "irrational_twin": "Is it irrational?"}))
    jobs = build_battery(dtbench_dir=data_dir, post_eval_prompts=fav, k=2, capability_k=0, favorite_k=3,
                         probe_k=4, dtbench_extra={"1.2ATT": 10}, dilemma_games=())
    kinds = Counter(item.kind for item, _ in jobs)
    assert kinds == {"dtbench-attitude": 12, "favorite-theory": 3, "yes-no": 4}

    async def answer(prompt, k):
        if "FINAL ANSWER" in prompt:
            letter = re.search(r"^([A-Z])\) Cooperate$", prompt, re.MULTILINE).group(1)
            return [f"FINAL ANSWER: {letter}."] * k
        return ["Yes." if "irrational" in prompt else "<theory>CDT</theory>"] * k

    s = summarize_battery(asyncio.run(run_battery(answer, jobs)))
    assert s["attitude"]["label_counts"] == {"EDT-only": 2}  # extras excluded from the pooled counts
    assert s["attitude"]["sample0_label_counts"] == {"EDT-only": 1}
    assert s["yes_no"] == {"irrational_twin": {"yes": 4}}
