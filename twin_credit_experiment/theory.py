"""Exact expected advantages for the grading schemes, and the decision theories they match.

A population of i.i.d. trajectories cooperates with probability ``p``. For a two-action
softmax policy with logit gap ``z`` (``p = sigmoid(z)``), the expected policy-gradient
update is ``dz ∝ p (1 - p) · gap(p)`` with ``gap = E[A | C] - E[A | D]``; the sign of
the gap is which way training pushes, and its roots are the fixed points.

Three quantities differ and must not be conflated: the *reward* gap, its large-group
limit (``gap_limit``), and the exact group-centered advantage gap for a group of ``n``
(``gap_finite``, cross-checked by enumeration in ``gap_exact``). Preregistered
predictions use the finite form.

Identities pinned by the tests:

- ``coupled(rho)``'s limiting gap is EDT's expected-utility gap for a twin whose action
  regresses on mine with coefficient ``rho`` (the Kimi post's footnote 5 criterion).
  ``rho = 0`` is CDT's partial derivative. The rewarded action agrees with EDT; whether
  the model learns anything decision-theoretic from it is the empirical question.
- ``welfare``'s limiting gap is the total derivative of the selfish symmetric policy
  value ``V(p) = E_{a, a' ~ Bern(p)} u(a, a')`` (policy selection, private coins).
  Group centering scales it by ``(n - 2) / n``: zero at ``n = 2``.
- The two axes are separate. ``rho`` sets the joint distribution being scored; which
  policy occurrences get credit (focal only, or every copy) is a different choice.
  ``coupled(rho)`` is not the policy-level gradient of its own coupled payoff.
- In donation games (``R + P = S + T``, e.g. the post's T = 2), ``welfare`` and
  ``coupled(1)`` advantages are exactly proportional, so they cannot be contrasted.
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


def other_benefit(game: Game, p: float) -> float:
    """How much my cooperating gives the other player: ``b(p) = p (R - S) + (1 - p) (T - P)``."""
    return p * (game.R - game.S) + (1 - p) * (game.T - game.P)


def policy_value(game: Game, p: float) -> float:
    """Selfish expected payoff when every instance plays Bern(p) with private coins."""
    return (
        p * p * game.R
        + p * (1 - p) * (game.S + game.T)
        + (1 - p) * (1 - p) * game.P
    )


def policy_value_slope(game: Game, p: float) -> float:
    """``V'(p)``: the selfish payoff's total derivative through both copies."""
    return (
        2 * p * game.R
        + (1 - 2 * p) * (game.S + game.T)
        - 2 * (1 - p) * game.P
    )


def coupled_policy_value_slope(game: Game, p: float, rho: float) -> float:
    """``W_rho'(p)`` for ``W_rho = rho [p R + (1 - p) P] + (1 - rho) V``: the policy-level
    gradient of the coupled payoff, which is *not* what ``coupled(rho)`` grading trains."""
    return rho * (game.R - game.P) + (1 - rho) * policy_value_slope(game, p)


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
    if scheme == "stranger-welfare":
        return policy_value_slope(game, q)
    raise ValueError(f"unknown scheme {scheme!r}")


def gap_finite(
    game: Game,
    p: float,
    n: int,
    scheme: Scheme = "coupled",
    *,
    rho: float = 0.0,
    q: float = 0.5,
) -> float:
    """Exact group-centered (Dr. GRPO) advantage gap for a group of ``n``, in closed form.

    Others' rewards sit in the focal trajectory's baseline. When they are graded against
    the focal action, that leaks ``-(1 - rho) b(p) / n`` of extra defection pressure for
    ``coupled`` and attenuates ``welfare`` by ``(n - 2) / n``.
    """
    if n < 2:
        raise ValueError("group-centered advantages need n >= 2")
    if scheme == "coupled":
        return (n - 1) / n * gap_limit(game, p, "coupled", rho=rho) - (1 - rho) / n * other_benefit(
            game, p
        )
    if scheme == "welfare":
        return (n - 2) / n * policy_value_slope(game, p)
    if scheme in ("stranger", "stranger-welfare"):
        return (n - 1) / n * gap_limit(game, p, scheme, q=q)
    raise ValueError(f"unknown scheme {scheme!r}")


def gap_exact(
    game: Game,
    p: float,
    n: int,
    scheme: Scheme = "coupled",
    *,
    rho: float = 0.0,
    q: float = 0.5,
) -> float:
    """``gap_finite`` by brute-force enumeration over the others' cooperation count.

    Independent of the closed form: it grades representative groups with ``grade`` and
    centers them with ``dr_grpo_advantages``, exactly as a trainer would.
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


def edt_gap(game: Game, p: float, beta: float) -> float:
    """EDT's ``E[u | C] - E[u | D]`` when the twin copies my action with probability
    ``beta`` and otherwise cooperates independently with probability ``p``.

    Computed from the conditional cooperation rates directly, independently of
    ``gap_limit``. For ``0 < p < 1`` the twin's regression coefficient on my action is
    exactly ``beta``; at pure policies it is undefined and this is the limit.
    """
    coop_given_c = beta + (1 - beta) * p
    coop_given_d = (1 - beta) * p
    eu_c = coop_given_c * game.R + (1 - coop_given_c) * game.S
    eu_d = coop_given_d * game.T + (1 - coop_given_d) * game.P
    return eu_c - eu_d


def rho_threshold(game: Game, p: float, n: int | None = None) -> float:
    """Smallest coupling at which coupled grading favors cooperating.

    Large-group limit ``g / (R - P + g)``; for a group of ``n``,
    ``((n - 1) g + b) / ((n - 1)(R - P + g) + b)``.
    """
    g = defection_premium(game, p)
    if n is None:
        return g / (game.R - game.P + g)
    b = other_benefit(game, p)
    return ((n - 1) * g + b) / ((n - 1) * (game.R - game.P + g) + b)


def policy_optimum(game: Game) -> float:
    """``argmax_p V(p)`` on [0, 1] — Appendix C's 'policy every instance should follow'."""
    curvature = 2 * (game.R + game.P - game.S - game.T)
    if curvature < 0:
        stationary = (game.S + game.T - 2 * game.P) / -curvature
        if 0.0 < stationary < 1.0:
            return float(stationary)
    return 1.0 if policy_value(game, 1.0) >= policy_value(game, 0.0) else 0.0


def _gap(game, p, scheme, rho, q, n):
    if n is None:
        return gap_limit(game, p, scheme, rho=rho, q=q)
    return gap_finite(game, p, n, scheme, rho=rho, q=q)


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
        z += lr * p * (1 - p) * _gap(game, p, scheme, rho, q, n)
    return out


def attractors(
    game: Game,
    scheme: Scheme = "coupled",
    *,
    rho: float = 0.0,
    q: float = 0.5,
    n: int | None = None,
    resolution: int = 10_001,
) -> list[float]:
    """Stable fixed points of the expected flow on [0, 1] (limit, or a group of ``n``)."""
    grid = np.linspace(0.0, 1.0, resolution)
    gaps = np.array([_gap(game, x, scheme, rho, q, n) for x in grid])
    stable = []
    if gaps[0] < 0:
        stable.append(0.0)
    for i in range(resolution - 1):
        if gaps[i] > 0 >= gaps[i + 1]:
            x0, x1, g0, g1 = grid[i], grid[i + 1], gaps[i], gaps[i + 1]
            stable.append(float(x0 + (x1 - x0) * g0 / (g0 - g1)))
    if gaps[-1] > 0:
        stable.append(1.0)
    return stable
