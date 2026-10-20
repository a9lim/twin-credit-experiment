"""Spend caps, streaming records and resume: the safeguards added after the 2026-10-19 overrun."""

import asyncio
import json

import pytest

from twin_credit_experiment.evals import build_battery, group_key, run_battery
from twin_credit_experiment.pricing import SpendMeter
from twin_credit_experiment.train import RunConfig, run

from .test_evals import scripted
from .test_dtbench import TREE
from .test_train import BernoulliBackend


@pytest.fixture
def jobs(tmp_path):
    (tmp_path / "setting001toy.json").write_text(TREE)
    fav = tmp_path / "fav.json"
    fav.write_text(json.dumps({"favorite_theory": "Which? <theory></theory>"}))
    return build_battery(dtbench_dir=tmp_path, post_eval_prompts=fav, k=3, capability_k=2, favorite_k=45,
                         dilemma_games=("post-pd",))


def counting(sampler, calls):
    async def wrapped(prompt, k):
        calls.append(k)
        return await sampler(prompt, k)
    return wrapped


def test_meter_prices_tokens_and_trips_at_the_cap():
    m = SpendMeter("Qwen/Qwen3.6-35B-A3B", cap_usd=1.0)
    m.add(prefill=1_000_000, sample=0)
    assert m.usd == pytest.approx(0.45) and not m.exhausted
    m.add(sample=500_000)
    assert m.usd == pytest.approx(0.45 + 0.6675) and m.exhausted
    with pytest.raises(ValueError):
        SpendMeter("unknown/model")


def test_records_stream_per_request_and_large_items_are_chunked(jobs):
    streamed, calls = [], []
    records = asyncio.run(run_battery(counting(scripted, calls), jobs, sink=streamed.append, chunk=20))
    assert max(calls) <= 20 and sorted(calls)[-3:] == [5, 20, 20]  # favorite-theory 45 -> 20 + 20 + 5
    assert sum(len(batch) for batch in streamed) == len(records) == len(jobs)


def test_resume_skips_requests_already_on_disk(jobs):
    first = asyncio.run(run_battery(scripted, jobs))
    done = {r["group_key"] for r in first[: len(first) // 2]}
    calls = []
    second = asyncio.run(run_battery(counting(scripted, calls), jobs, skip=done))
    assert {r["group_key"] for r in second}.isdisjoint(done)
    assert len({r["group_key"] for r in first}) == len(done) + len({r["group_key"] for r in second})


def test_should_stop_prevents_new_requests(jobs):
    calls = []
    stop_after = 3

    records = asyncio.run(
        run_battery(counting(scripted, calls), jobs, concurrency=1, should_stop=lambda: len(calls) >= stop_after)
    )
    assert len(calls) == stop_after
    assert len({r["group_key"] for r in records}) == stop_after


def test_group_key_is_stable_and_distinguishes_prompt_variants(jobs):
    item, i = jobs[0]
    assert group_key(item, i) == group_key(item, i)
    variants = {group_key(it, 0) for it, _ in jobs if it.item_id == item.item_id}
    assert len(variants) >= 2  # per-sample answer shuffles are different requests


class MeteredBernoulli(BernoulliBackend):
    def __init__(self, cap):
        super().__init__(seed=0)
        self.meter = SpendMeter("Qwen/Qwen3.5-9B", cap_usd=cap)

    async def sample(self, prompt_tokens, n, max_tokens, temperature):
        out = await super().sample(prompt_tokens, n, max_tokens, temperature)
        self.meter.add(sample=100_000 * n)
        return out


def test_training_loop_stops_before_a_step_once_the_cap_is_reached():
    backend = MeteredBernoulli(cap=1.0)  # 8 prompts x 16 x 100k tokens x $0.807/M = $10.3 per step
    saved = []

    async def save(name):
        saved.append(name)
        return name

    backend.save = save
    cfg = RunConfig(arm="A1p", base_model="Qwen/Qwen3.5-9B", batch_prompts=8, group_size=16, steps=5,
                    max_tokens=1, learning_rate=0.02, save_every=0, max_usd=1.0)
    history = asyncio.run(run(cfg, backend))
    assert len(history) == 1
    assert saved == ["A1p-s0-step0001"]  # the paid-for step survives the stop


def test_extending_one_item_leaves_every_other_request_unchanged(jobs, tmp_path):
    from twin_credit_experiment.evals import build_battery

    data_dir = tmp_path
    fav = tmp_path / "fav.json"
    base = build_battery(dtbench_dir=data_dir, post_eval_prompts=fav, k=3, capability_k=2, favorite_k=45,
                         dilemma_games=("post-pd",))
    more = build_battery(dtbench_dir=data_dir, post_eval_prompts=fav, k=3, capability_k=2, favorite_k=90,
                         dtbench_extra={"1.2ATT": 7}, dilemma_games=("post-pd",))
    keys_base = {group_key(item, i) for item, i in base}
    keys_more = {group_key(item, i) for item, i in more}
    assert keys_base <= keys_more  # resume after extending re-sends nothing already done
