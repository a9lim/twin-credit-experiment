"""Exact expected advantages for the grading schemes, and the decision theories they implement.

A population of i.i.d. trajectories cooperates with probability ``p``. For a two-action
softmax policy with logit gap ``z`` (``p = sigmoid(z)``), the expected policy-gradient
update is ``dz ∝ p (1 - p) · gap(p)`` with ``gap = E[A | C] - E[A | D]``; the sign of
the gap is therefore which way training pushes, and its roots are the fixed points.

Identities pinned by the tests:

- ``coupled(rho)`` gives EDT's expected-utility gap for a twin whose action regresses
  on mine with coefficient ``rho`` (the Kimi post's footnote 5 criterion, with the
  environment's actual coupling in place of a prior). ``rho = 0`` is CDT's partial
  derivative: defecting always wins.
- ``welfare`` gives the total derivative of the selfish symmetric policy value
  ``V(p) = E_{a, a' ~ Bern(p)} u(a, a')``, i.e. choosing one policy for every instance,
  each with a private coin. It converges to ``argmax V``, interior when ``S + T > 2R``.
"""

from __future__ import annotations

import math

import numpy as np

from .games import COOPERATE, DEFECT, Game, Scheme, dr_grpo_advantages, grade


def defection_premium(game: Game, p: float) -> float:
    """How much better defecting is when my action carries no evidence about the twin's.

    ``g(p) = p (T - R) + (1 - p) (P - S)`` for a twin cooperating with probability ``p``.
    """
    return p * (game.T - game.R) + (1 - p) * (game.P - game.S)


def gap_limit(
    game: Game,
    p: float,
    scheme: Scheme = "coupled",
    *,
    rho: float = 0.0,
    q: float = 0.5,
) -> float:
    """``E[A | C] - E[A | D]`` in the large-group limit (advantage gap = reward gap)."""
    if scheme == "coupled":
        return rho * (game.R - game.P) - (1 - rho) * defection_premium(game, p)
    if scheme == "welfare":
        return policy_value_slope(game, p)
    if scheme == "stranger":
        return -defection_premium(game, q)
    raise ValueError(f"unknown scheme {scheme!r}")


def edt_gap(game: Game, p: float, beta: float) -> float:
    """EDT's ``E[u | C] - E[u | D]`` when the twin copies my action with probability
    ``beta`` and otherwise cooperates independently with probability ``p``.

    Computed from the conditional cooperation rates directly, independently of
    ``gap_limit``, so the two can be checked against each other.
    """
    coop_given_c = beta + (1 - beta) * p
    coop_given_d = (1 - beta) * p
    eu_c = coop_given_c * game.R + (1 - coop_given_c) * game.S
    eu_d = coop_given_d * game.T + (1 - coop_given_d) * game.P
    return eu_c - eu_d


def rho_threshold(game: Game, p: float) -> float:
    """Smallest coupling at which coupled grading favors cooperation: ``g / (R - P + g)``."""
    g = defection_premium(game, p)
    return g / (game.R - game.P + g)


def policy_value(game: Game, p: float) -> float:
    """Selfish expected payoff when every instance plays Bern(p) with private coins."""
    return (
        p * p * game.R
        + p * (1 - p) * (game.S + game.T)
        + (1 - p) * (1 - p) * game.P
    )


def policy_value_slope(game: Game, p: float) -> float:
    """``dV/dp``: the selfish payoff's total derivative through both copies."""
    return (
        2 * p * game.R
        + (1 - 2 * p) * (game.S + game.T)
        - 2 * (1 - p) * game.P
    )


def policy_optimum(game: Game) -> float:
    """``argmax_p V(p)`` on [0, 1] — Appendix C's 'policy every instance should follow'."""
    grid = np.linspace(0.0, 1.0, 100_001)
    values = np.array([policy_value(game, x) for x in grid])
    # V is quadratic, so the stationary point is exact when interior.
    curvature = 2 * (game.R + game.P - game.S - game.T)
    if curvature < 0:
        stationary = (game.S + game.T - 2 * game.P) / -curvature
        if 0.0 < stationary < 1.0:
            return float(stationary)
    return float(grid[values.argmax()])


def gap_exact(
    game: Game,
    p: float,
    n: int,
    scheme: Scheme = "coupled",
    *,
    rho: float = 0.0,
    q: float = 0.5,
) -> float:
    """Exact ``E[A_i | a_i = C] - E[A_i | a_i = D]`` for a group of ``n`` under Dr. GRPO.

    Enumerates the number of cooperators among the other ``n - 1`` trajectories; the
    group mean includes trajectory ``i``'s own reward and the others' rewards, which
    depend on ``a_i`` whenever they are graded against it.
    """
    conditional = {}
    for mine in (COOPERATE, DEFECT):
        expected = 0.0
        for k in range(n):
            weight = math.comb(n - 1, k) * p**k * (1 - p) ** (n - 1 - k)
            if weight == 0.0:
                continue
            actions = np.array([mine] + [COOPERATE] * k + [DEFECT] * (n - 1 - k))
            adv = dr_grpo_advantages(grade(game, actions, scheme, rho=rho, q=q))
            expected += weight * adv[0]
        conditional[mine] = expected
    return conditional[COOPERATE] - conditional[DEFECT]


def flow(
    game: Game,
    scheme: Scheme = "coupled",
    *,
    p0: float = 0.5,
    rho: float = 0.0,
    q: float = 0.5,
    n: int | None = None,
    lr: float = 1.0,
    steps: int = 2_000,
) -> np.ndarray:
    """Expected softmax policy-gradient trajectory of the cooperation rate.

    ``n = None`` uses the large-group gap; an integer uses the exact finite-group gap.
    """
    z = math.log(p0 / (1 - p0))
    out = np.empty(steps + 1)
    for t in range(steps + 1):
        p = 1.0 / (1.0 + math.exp(-z))
        out[t] = p
        gap = (
            gap_limit(game, p, scheme, rho=rho, q=q)
            if n is None
            else gap_exact(game, p, n, scheme, rho=rho, q=q)
        )
        z += lr * p * (1 - p) * gap
    return out


def attractors(
    game: Game,
    scheme: Scheme = "coupled",
    *,
    rho: float = 0.0,
    q: float = 0.5,
    resolution: int = 10_001,
) -> list[float]:
    """Stable fixed points of the large-group flow on [0, 1]."""
    grid = np.linspace(0.0, 1.0, resolution)
    gaps = np.array([gap_limit(game, x, scheme, rho=rho, q=q) for x in grid])
    stable = []
    if gaps[0] < 0:
        stable.append(0.0)
    for i in range(resolution - 1):
        if gaps[i] > 0 >= gaps[i + 1]:
            # linear interpolation of the + to - crossing
            x0, x1, g0, g1 = grid[i], grid[i + 1], gaps[i], gaps[i + 1]
            stable.append(float(x0 + (x1 - x0) * g0 / (g0 - g1)))
    if gaps[-1] > 0:
        stable.append(1.0)
    return stable
