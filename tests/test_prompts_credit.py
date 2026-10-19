"""Generated prompts are honest and parseable; the credit path reproduces the theory."""

import re

import numpy as np
import pytest

from twin_credit_experiment.credit import (
    credit_group,
    invalid_reward,
    parse_action,
    token_advantages,
    visible_answer,
)
from twin_credit_experiment.games import APPENDIX_C_PD, POST_PD, POST_PD_T10, dr_grpo_advantages, grade
from twin_credit_experiment.prompts import HELD_OUT, SCENARIOS, prompt_set, render
from twin_credit_experiment.theory import gap_finite

FORBIDDEN = re.compile(
    r"prisoner|cooperat|defect|\btwin|newcomb|decision theory|causal|acausal|superrational|seed",
    re.IGNORECASE,
)


def test_scenarios_have_distinct_parseable_labels():
    assert len({s.name for s in SCENARIOS}) == len(SCENARIOS)
    for s in SCENARIOS:
        c, d = s.coop[0], s.defect[0]
        assert c != d
        assert re.fullmatch(r"[A-Z][A-Z-]*", c) and re.fullmatch(r"[A-Z][A-Z-]*", d)


@pytest.mark.parametrize("framing", ["twin", "stranger"])
@pytest.mark.parametrize("game", [POST_PD, APPENDIX_C_PD, POST_PD_T10], ids=lambda g: g.name)
def test_prompts_are_honest_and_complete(framing, game):
    for p in prompt_set(game, framing, "train") + prompt_set(game, framing, "held-out"):
        assert not FORBIDDEN.search(p.text), FORBIDDEN.search(p.text)
        assert f"ACTION: {p.coop_label}" in p.text and f"ACTION: {p.defect_label}" in p.text
        assert f"{game.T:+g} for you" in p.text and f"{game.S:+g} for you" in p.text
        assert "no way to communicate" in p.text
        if framing == "twin":
            assert "same weights" in p.text and "at the same time" in p.text
            assert "not a copy" not in p.text and "fixed before" not in p.text
        else:
            assert "not a copy of you" in p.text and "fixed before" in p.text
            assert "same weights" not in p.text and "at the same time" not in p.text


def test_splits_are_disjoint_by_scenario_and_counterbalanced():
    train = prompt_set(POST_PD, "twin", "train")
    held = prompt_set(POST_PD, "twin", "held-out")
    assert {p.scenario for p in train}.isdisjoint({p.scenario for p in held})
    assert {p.scenario for p in held} == HELD_OUT
    assert len(held) == 2 * len(HELD_OUT)
    for pool in (train, held):
        assert sum(p.coop_first for p in pool) == len(pool) / 2


LABELS = {"LIST": 1, "UNDERCUT": 0}


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("reasoning\nACTION: LIST", 1),
        ("reasoning\n**ACTION: UNDERCUT**\n", 0),
        ("`ACTION: LIST`.", 1),
        ("<think>ACTION: UNDERCUT? no...</think>\nFinal.\nACTION: LIST", 1),
        ("ACTION: LIST\nACTION: UNDERCUT", None),
        ("ACTION: list", None),
        ("ACTION: HOLD", None),
        ("I choose LIST.", None),
        ("<think>ACTION: LIST</think>", None),
    ],
)
def test_parse_action(completion, expected):
    assert parse_action(completion, LABELS) == expected


def test_visible_answer_strips_reasoning():
    assert visible_answer("<think>a</think>b</think>c") == "c"
    assert visible_answer("plain") == "plain"


def test_all_valid_groups_match_grading():
    actions = [1, 0, 0, 1, 1]
    c = credit_group(POST_PD_T10, actions, "coupled", rho=0.25)
    assert c.rewards == pytest.approx(grade(POST_PD_T10, np.array(actions), "coupled", rho=0.25))
    assert c.advantages == pytest.approx(dr_grpo_advantages(c.rewards))
    assert c.advantages.sum() == pytest.approx(0.0)


def test_invalid_answers_sit_out_and_take_the_floor():
    c = credit_group(POST_PD, [1, None, 0, 0], "coupled")
    assert c.n_valid == 3
    assert c.rewards[1] == invalid_reward(POST_PD) < min(c.rewards[[0, 2, 3]])
    assert c.rewards[[0, 2, 3]] == pytest.approx(grade(POST_PD, np.array([1, 0, 0]), "coupled"))


def test_one_valid_answer_is_format_signal_only_and_none_is_no_signal():
    one = credit_group(POST_PD, [None, 1, None], "coupled")
    assert one.rewards[1] == invalid_reward(POST_PD) + 1
    assert one.advantages[1] > 0 > one.advantages[0]
    none = credit_group(POST_PD, [None, None], "coupled")
    assert none.advantages == pytest.approx([0.0, 0.0])


def test_scale_applies_to_game_rewards_only():
    c = credit_group(POST_PD, [1, 0, None], "stranger", q=0.5, scale=17 / 15)
    base = credit_group(POST_PD, [1, 0, None], "stranger", q=0.5)
    assert c.rewards[:2] == pytest.approx(17 / 15 * base.rewards[:2])
    assert c.rewards[2] == base.rewards[2] == invalid_reward(POST_PD)


def test_token_advantages_use_a_constant_normalizer_not_length():
    short = token_advantages(0.5, 10, normalizer=4096)
    long = token_advantages(0.5, 1000, normalizer=4096)
    assert short[0] == long[0] == pytest.approx(0.5 / 4096)
    assert sum(long) == pytest.approx(100 * sum(short))  # longer completions carry more total weight


@pytest.mark.parametrize(
    "game, scheme, rho",
    [(POST_PD, "coupled", 0.0), (POST_PD, "coupled", 1.0), (POST_PD_T10, "coupled", 0.6), (APPENDIX_C_PD, "welfare", 0.0)],
    ids=lambda x: getattr(x, "name", str(x)),
)
def test_bernoulli_harness_through_the_credit_path_matches_gap_finite(game, scheme, rho):
    """Offline half of Step 0: Monte Carlo groups through credit_group reproduce theory."""
    rng = np.random.default_rng(3)
    n, p, draws = 16, 0.55, 20_000
    adv = {0: [], 1: []}
    for _ in range(draws):
        actions = [int(x) for x in rng.random(n) < p]
        c = credit_group(game, actions, scheme, rho=rho)
        adv[actions[0]].append(c.advantages[0])
    mc = np.mean(adv[1]) - np.mean(adv[0])
    se = np.sqrt(np.var(adv[1]) / len(adv[1]) + np.var(adv[0]) / len(adv[0]))
    assert abs(mc - gap_finite(game, p, n, scheme, rho=rho)) < 5 * se + 1e-9
