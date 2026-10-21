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
        ("<think>torn between <theory>EDT</theory></think><theory>Updateless Decision Theory</theory>", "UDT"),
        ("<theory>FDT</theory>", "FDT"),
        ("<theory>Logical Decision Theory</theory>", "LDT"),
        ("<theory>Bayesian Decision Theory</theory>", "unlisted"),
        ("<theory>Causal Decision Theory or Evidential Decision Theory</theory>", "multiple"),
        ("no tag at all", "invalid"),
    ],
)
def test_forced_theory_label(text, label):
    from twin_credit_experiment.evals import forced_theory_label

    assert forced_theory_label(text) == label


def test_forced_choice_lists_every_theory_once_in_a_seeded_order(battery_inputs):
    from twin_credit_experiment.evals import FORCED

    data_dir, fav = battery_inputs
    jobs = build_battery(dtbench_dir=data_dir, post_eval_prompts=fav, k=0, capability_k=0, favorite_k=0,
                         forced_k=12, dilemma_games=())
    assert Counter(item.kind for item, _ in jobs) == {"forced-theory": 12}
    orders = [tuple(item.meta["order"]) for item, _ in jobs]
    assert len(set(orders)) > 6  # orders vary across samples
    for item, _ in jobs:
        assert item.prompt.startswith("Which decision theory? ")
        assert all(item.prompt.count(name) == 1 for name in FORCED.values())
        positions = [item.prompt.index(FORCED[a]) for a in item.meta["order"]]
        assert positions == sorted(positions)  # listed in the recorded order
    again = build_battery(dtbench_dir=data_dir, post_eval_prompts=fav, k=0, capability_k=0, favorite_k=0,
                          forced_k=12, dilemma_games=())
    assert [i.prompt for i, _ in again] == [i.prompt for i, _ in jobs]  # seeded

    async def first_listed(prompt, k):
        name = prompt.split("tags: ")[1].split(",")[0]
        return [f"<theory>{name}</theory>"] * k

    s = summarize_battery(asyncio.run(run_battery(first_listed, jobs)))["forced_theory"]
    assert s["by_list_position"] == {0: 12}  # a pure position bias shows up as such
    assert sum(s["grouped"].values()) == 12 and s["grouped"]["off the list"] == 0


def test_camps_list_names_three_theories_and_marks_others_unlisted(battery_inputs):
    from twin_credit_experiment.evals import FORCED

    data_dir, fav = battery_inputs
    jobs = build_battery(dtbench_dir=data_dir, post_eval_prompts=fav, k=0, capability_k=0, favorite_k=0,
                         forced_k=9, forced_list="camps", dilemma_games=())
    assert Counter(item.kind for item, _ in jobs) == {"forced-theory-camps": 9}
    for item, _ in jobs:
        assert sorted(item.meta["order"]) == ["CDT", "EDT", "FDT"]
        assert all((FORCED[a] in item.prompt) == (a in ("CDT", "EDT", "FDT")) for a in FORCED)

    async def updateless(prompt, k):  # names a theory that is not on this list
        return ["<theory>Updateless Decision Theory</theory>"] * k

    s = summarize_battery(asyncio.run(run_battery(updateless, jobs)))["forced_theory_camps"]
    assert s["counts"] == {"unlisted": 9}


def test_described_rules_hide_the_names_and_map_letters_back(battery_inputs):
    from twin_credit_experiment.evals import FORCED, RULES, forced_described_label

    data_dir, fav = battery_inputs
    jobs = build_battery(dtbench_dir=data_dir, post_eval_prompts=fav, k=0, capability_k=0, favorite_k=0,
                         forced_k=12, forced_list="described", dilemma_games=())
    assert Counter(item.kind for item, _ in jobs) == {"forced-theory-described": 12}
    assert set(Counter(tuple(item.meta["order"]) for item, _ in jobs).values()) == {2}  # all 6 orders, twice each
    for item, _ in jobs:
        text = item.prompt
        assert not any(name.lower() in text.lower() for name in FORCED.values())
        assert not re.search(r"\b(CDT|EDT|FDT|causal decision|evidential|functional|newcomb|lesswrong)\b", text, re.I)
        for letter, rule in zip("ABC", item.meta["order"]):
            assert f"{letter}) {RULES[rule]}" in text

    order = ["EDT", "FDT", "CDT"]
    assert forced_described_label("<think>maybe <theory>A</theory></think><theory>**C**</theory>", order) == "CDT"
    assert forced_described_label("<theory>Option B</theory>", order) == "FDT"
    assert forced_described_label("<theory>Evidential Decision Theory</theory>", order) == "EDT"  # named, not lettered
    assert forced_described_label("<theory>A or C</theory>", order) == "multiple"
    assert forced_described_label("<theory>Bayesian</theory>", order) == "unlisted"
    assert forced_described_label("no tag", order) == "invalid"

    async def letter_c(prompt, k):
        return ["<theory>C</theory>"] * k

    s = summarize_battery(asyncio.run(run_battery(letter_c, jobs)))["forced_theory_described"]
    assert s["by_list_position"] == {2: 12}


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
