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
    Game,
    dr_grpo_advantages,
    grade,
)
from twin_credit_experiment.theory import (
    attractors,
    coupled_policy_value_slope,
    edt_gap,
    flow,
    gap_exact,
    gap_finite,
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
    n, p, draws = 16, 0.6, 20_000
    sums = {COOPERATE: 0.0, DEFECT: 0.0}
    counts = {COOPERATE: 0, DEFECT: 0}
    for _ in range(draws):
        a = (rng.random(n) < p).astype(int)
        adv = dr_grpo_advantages(grade(POST_PD_T10, a, scheme, rho=rho))
        sums[a[0]] += adv[0]
        counts[a[0]] += 1
    mc = sums[COOPERATE] / counts[COOPERATE] - sums[DEFECT] / counts[DEFECT]
    assert mc == pytest.approx(gap_exact(POST_PD_T10, p, n, scheme, rho=rho), abs=0.2)


def test_small_groups_add_defection_pressure_through_the_baseline():
    """Others graded against me sit in my baseline, so finite groups lean further to D."""
    for p in (0.2, 0.5, 0.8):
        assert gap_exact(POST_PD, p, 4, "coupled") < gap_limit(POST_PD, p, "coupled")


@pytest.mark.parametrize("scheme", ["coupled", "welfare", "stranger", "stranger-welfare"])
def test_closed_form_finite_gap_matches_enumeration_on_random_games(scheme):
    rng = np.random.default_rng(7)
    for _ in range(25):
        R, S, T, P = rng.normal(size=4)
        game = Game("random", R=R, S=S, T=T, P=P)
        for n in (2, 3, 5, 16):
            for rho in (0.0, 0.3, 1.0):
                for p in (0.15, 0.5, 0.85):
                    assert gap_finite(game, p, n, scheme, rho=rho, q=0.3) == pytest.approx(
                        gap_exact(game, p, n, scheme, rho=rho, q=0.3), abs=1e-12
                    )


def test_finite_thresholds_at_group_size_16():
    """Codex review: the limiting threshold is not what a group of 16 trains against."""
    assert rho_threshold(POST_PD, 0.5, n=16) == pytest.approx(0.53125)
    assert rho_threshold(POST_PD_T10, 0.0, n=16) == pytest.approx(0.625)
    assert rho_threshold(POST_PD_T10, 0.5, n=16) == pytest.approx(0.84375)
    for game in GAMES.values():
        for p in PS:
            assert gap_finite(game, p, 16, "coupled", rho=rho_threshold(game, p, n=16)) == pytest.approx(
                0.0, abs=1e-12
            )


def test_finite_group_moves_the_dose_attractor():
    assert attractors(POST_PD_T10, "coupled", rho=0.75) == pytest.approx([0.25], abs=1e-3)
    assert attractors(POST_PD_T10, "coupled", rho=0.75, n=16) == pytest.approx([5 / 28], abs=1e-3)


@pytest.mark.parametrize("game", GAMES.values(), ids=GAMES.keys())
def test_welfare_is_attenuated_not_biased_and_vanishes_at_n_2(game):
    for p in PS:
        for n in (2, 4, 16):
            assert gap_exact(game, p, n, "welfare") == pytest.approx(
                (n - 2) / n * policy_value_slope(game, p), abs=1e-12
            )
    a = np.array([1, 0])
    assert dr_grpo_advantages(grade(game, a, "welfare")) == pytest.approx([0.0, 0.0])


def test_donation_game_makes_welfare_and_shared_coin_proportional():
    """At T = 2, u(a, b) = -a + 2b, so A4's advantages are (n-2)/(n-1) of A2's in every group."""
    assert POST_PD.R + POST_PD.P == POST_PD.S + POST_PD.T
    n = 16
    for k in range(n + 1):
        a = np.array([COOPERATE] * k + [DEFECT] * (n - k))
        shared = dr_grpo_advantages(grade(POST_PD, a, "coupled", rho=1.0))
        welfare = dr_grpo_advantages(grade(POST_PD, a, "welfare"))
        assert welfare == pytest.approx((n - 2) / (n - 1) * shared)


def test_shared_and_private_coin_push_opposite_ways_at_high_temptation():
    """At T = 10 above p = 9/16, rho = 1 pushes cooperation up while welfare pushes it down."""
    for p in (0.6, 0.8, 0.95):
        assert gap_limit(POST_PD_T10, p, "coupled", rho=1.0) > 0
        assert gap_limit(POST_PD_T10, p, "welfare") < 0


def test_zero_sum_self_play_is_not_a_null_control():
    """V is flat for every symmetric policy, yet focal credit strictly prefers C."""
    zero_sum = Game("zero-sum", R=0.0, S=1.0, T=-1.0, P=0.0)
    for p in PS:
        assert policy_value_slope(zero_sum, p) == pytest.approx(0.0)
        assert gap_limit(zero_sum, p, "coupled", rho=0.0) == pytest.approx(1.0)
        assert gap_limit(zero_sum, p, "coupled", rho=1.0) == pytest.approx(0.0)


def test_coupled_grading_is_not_the_policy_gradient_of_its_own_payoff():
    """Joint distribution (rho) and credited occurrences are separate axes."""
    for rho in (0.25, 0.5, 0.75):
        for p in (0.2, 0.5, 0.8):
            assert gap_limit(POST_PD_T10, p, "coupled", rho=rho) != pytest.approx(
                coupled_policy_value_slope(POST_PD_T10, p, rho)
            )


def test_stranger_welfare_direction_depends_on_the_strangers_rate():
    """C2 at T = 10: gap 9 - 16q, so it trains cooperation only while q < 9/16."""
    for q in (0.1, 0.5, 9 / 16, 0.9):
        assert gap_limit(POST_PD_T10, 0.5, "stranger-welfare", q=q) == pytest.approx(9 - 16 * q)


def test_stranger_defection_premium_is_q_free_only_in_donation_games():
    """C1's learning strength is matched to A1' for free at T = 2, not at T = 10."""
    assert {gap_limit(POST_PD, 0.5, "stranger", q=q) for q in (0.1, 0.5, 0.9)} == {-1.0}
    assert gap_limit(POST_PD_T10, 0.5, "stranger", q=0.1) != gap_limit(POST_PD_T10, 0.5, "stranger", q=0.9)


def test_pilot_stranger_control_is_scale_matched_by_17_over_15():
    """At T = 2, n = 16: A1' gap is -17/16 and C1's -15/16 at every p and q."""
    n = 16
    for p in PS:
        assert gap_finite(POST_PD, p, n, "coupled", rho=0.0) == pytest.approx(-17 / 16)
        for q in (0.1, 0.5, 0.9):
            assert 17 / 15 * gap_finite(POST_PD, p, n, "stranger", q=q) == pytest.approx(-17 / 16)
