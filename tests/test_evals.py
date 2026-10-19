"""The evaluation battery grades and summarizes correctly with a scripted sampler."""

import asyncio
import json
import re

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
        ("<theory>expected utility theory</theory>", "other"),
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
