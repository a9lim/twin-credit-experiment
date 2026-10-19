"""The grading schemes implement the decision theories the design doc claims they do."""

import itertools

import numpy as np
import pytest

from twin_credit_experiment.games import (
    APPENDIX_C_PD,
    COOPERATE,
    DEFECT,
    GAMES,
    POST_PD,
    POST_PD_T10,
    dr_grpo_advantages,
    grade,
)
from twin_credit_experiment.theory import (
    attractors,
    edt_gap,
    flow,
    gap_exact,
    gap_limit,
    policy_optimum,
    policy_value,
    policy_value_slope,
    rho_threshold,
)

PS = np.linspace(0.0, 1.0, 21)


@pytest.mark.parametrize("game", GAMES.values(), ids=GAMES.keys())
@pytest.mark.parametrize("n", [2, 4, 16])
def test_round_robin_never_rewards_cooperating(game, n):
    """Post footnote 7: with rho = 0 a defector out-earns a cooperator in every group."""
    for actions in itertools.product((COOPERATE, DEFECT), repeat=n):
        a = np.array(actions)
        if 0 < a.sum() < n:
            r = grade(game, a, "coupled", rho=0.0)
            assert r[a == DEFECT].min() >= r[a == COOPERATE].max()


def test_post_footnote_8_numbers():
    p = 0.9
    e_c = p * POST_PD.u(COOPERATE, COOPERATE) + (1 - p) * POST_PD.u(COOPERATE, DEFECT)
    e_d = p * POST_PD.u(DEFECT, COOPERATE) + (1 - p) * POST_PD.u(DEFECT, DEFECT)
    assert e_c == pytest.approx(0.8)
    assert e_d == pytest.approx(1.8)
    assert p * e_c + (1 - p) * e_d == pytest.approx(0.9)


@pytest.mark.parametrize("game", GAMES.values(), ids=GAMES.keys())
@pytest.mark.parametrize("rho", [0.0, 0.3, 0.5, 0.9, 1.0])
def test_coupled_grading_is_edt_with_beta_rho(game, rho):
    for p in PS:
        assert gap_limit(game, p, "coupled", rho=rho) == pytest.approx(
            edt_gap(game, p, beta=rho)
        )


def test_post_pd_threshold_is_one_half_at_every_population():
    """With (2, 1, 0, -1) the coupled scheme favors cooperation iff rho > 1/2."""
    for p in PS:
        assert rho_threshold(POST_PD, p) == pytest.approx(0.5)


@pytest.mark.parametrize("game", GAMES.values(), ids=GAMES.keys())
def test_threshold_is_where_the_gap_vanishes(game):
    for p in PS:
        assert gap_limit(game, p, "coupled", rho=rho_threshold(game, p)) == pytest.approx(
            0.0, abs=1e-12
        )


@pytest.mark.parametrize("game", GAMES.values(), ids=GAMES.keys())
def test_welfare_gap_is_total_derivative_of_selfish_policy_value(game):
    h = 1e-6
    for p in np.linspace(0.05, 0.95, 19):
        numeric = (policy_value(game, p + h) - policy_value(game, p - h)) / (2 * h)
        assert gap_limit(game, p, "welfare") == pytest.approx(numeric, rel=1e-6)
        assert policy_value_slope(game, p) == pytest.approx(numeric, rel=1e-6)


def test_appendix_c_policy_optimum_is_three_quarters():
    assert policy_optimum(APPENDIX_C_PD) == pytest.approx(0.75)
    assert policy_optimum(POST_PD_T10) == pytest.approx(9 / 16)
    assert policy_optimum(POST_PD) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "game, scheme, rho, expected",
    [
        (POST_PD, "coupled", 0.0, [0.0]),
        (POST_PD, "coupled", 1.0, [1.0]),
        (POST_PD, "welfare", 0.0, [1.0]),
        (POST_PD, "stranger", 0.0, [0.0]),
        (APPENDIX_C_PD, "coupled", 0.0, [0.0]),
        (APPENDIX_C_PD, "coupled", 1.0, [1.0]),
        (APPENDIX_C_PD, "welfare", 0.0, [0.75]),
        (POST_PD_T10, "welfare", 0.0, [9 / 16]),
        (POST_PD_T10, "coupled", 1.0, [1.0]),
    ],
)
def test_attractors(game, scheme, rho, expected):
    assert attractors(game, scheme, rho=rho) == pytest.approx(expected, abs=1e-3)


def test_shared_coin_and_private_coin_disagree_when_temptation_is_high():
    """rho = 1 (the twin replays my coin) always cooperates; welfare (private coins)
    mixes at argmax V. Same direction, different lesson about randomizing."""
    shared = flow(APPENDIX_C_PD, "coupled", rho=1.0, p0=0.5, steps=20_000)[-1]
    private = flow(APPENDIX_C_PD, "welfare", p0=0.5, steps=20_000)[-1]
    assert shared > 0.99
    assert private == pytest.approx(0.75, abs=1e-3)


@pytest.mark.parametrize("game", GAMES.values(), ids=GAMES.keys())
@pytest.mark.parametrize("scheme, rho", [("coupled", 0.0), ("coupled", 0.6), ("welfare", 0.0)])
def test_finite_group_gap_tends_to_the_limit(game, scheme, rho):
    for p in (0.2, 0.5, 0.8):
        assert gap_exact(game, p, 512, scheme, rho=rho) == pytest.approx(
            gap_limit(game, p, scheme, rho=rho), abs=0.05 * max(1.0, abs(game.T))
        )


@pytest.mark.parametrize("scheme, rho", [("coupled", 0.0), ("coupled", 0.75), ("welfare", 0.0)])
def test_finite_group_gap_matches_monte_carlo(scheme, rho):
    rng = np.random.default_rng(0)
    n, p, draws = 16, 0.6, 40_000
    sums = {COOPERATE: 0.0, DEFECT: 0.0}
    counts = {COOPERATE: 0, DEFECT: 0}
    for _ in range(draws):
        a = (rng.random(n) < p).astype(int)
        adv = dr_grpo_advantages(grade(POST_PD_T10, a, scheme, rho=rho))
        sums[a[0]] += adv[0]
        counts[a[0]] += 1
    mc = sums[COOPERATE] / counts[COOPERATE] - sums[DEFECT] / counts[DEFECT]
    assert mc == pytest.approx(gap_exact(POST_PD_T10, p, n, scheme, rho=rho), abs=0.15)


def test_small_groups_add_defection_pressure_through_the_baseline():
    """Others graded against me sit in my baseline, so finite groups lean further to D."""
    for p in (0.2, 0.5, 0.8):
        assert gap_exact(POST_PD, p, 4, "coupled") < gap_limit(POST_PD, p, "coupled")
