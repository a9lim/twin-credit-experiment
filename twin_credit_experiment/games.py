"""Symmetric two-action dilemmas and the group grading schemes that turn them into RL rewards.

Actions are encoded 1 = cooperate, 0 = defect. A grading scheme maps one group of
sampled actions (all from the same prompt) to one reward per trajectory; the trainer
then centers rewards within the group (Dr. GRPO: advantage = reward - group mean).

The schemes differ only in which joint distribution of (my action, twin's action)
they score against:

- ``coupled(rho)`` scores each trajectory against its own action with weight ``rho``
  and against every *other* trajectory with weight ``1 - rho``. ``rho = 0`` is the
  round robin of the Kimi post (a trajectory never faces itself); ``rho = 1`` is a
  twin that replays the same seed on the same prompt and therefore emits the same
  completion. ``rho`` is the environment's regression coefficient of the twin's
  action on mine.
- ``welfare`` scores each trajectory by its own payoff plus the payoff of the others
  when they face it, i.e. the policy gradient taken through both copies' tokens.
- ``stranger(q)`` scores against a fixed external opponent that cooperates with
  probability ``q``; no copy is involved, so every decision theory defects.
- ``stranger-welfare(q)`` adds what that stranger earns facing me: altruism with
  no copy. Its direction depends on ``q`` (at T = 10 it favors cooperating only
  while ``q < 9/16``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

COOPERATE = 1
DEFECT = 0

Scheme = Literal["coupled", "welfare", "stranger", "stranger-welfare"]


@dataclass(frozen=True)
class Game:
    """A symmetric 2x2 game in prisoner's-dilemma notation (row player's payoffs)."""

    name: str
    R: float  # both cooperate
    S: float  # I cooperate, they defect
    T: float  # I defect, they cooperate
    P: float  # both defect

    def u(self, mine: np.ndarray | int, theirs: np.ndarray | int) -> np.ndarray:
        """Row player's payoff; broadcasts over action arrays."""
        mine = np.asarray(mine)
        theirs = np.asarray(theirs)
        return np.where(
            mine == COOPERATE,
            np.where(theirs == COOPERATE, self.R, self.S),
            np.where(theirs == COOPERATE, self.T, self.P),
        ).astype(float)

    @property
    def is_dilemma(self) -> bool:
        return self.T > self.R > self.P > self.S


# The post's headline game: pay one point to give your copy two.
POST_PD = Game("post-pd", R=1.0, S=-1.0, T=2.0, P=0.0)
# Appendix C's red-team game: the selfish symmetric policy optimum is interior (p = 3/4).
APPENDIX_C_PD = Game("appendix-c-pd", R=1.0, S=-1.0, T=4.0, P=0.0)
# Most of the post's 32 training prompts use a +10 temptation.
POST_PD_T10 = Game("post-pd-t10", R=1.0, S=-1.0, T=10.0, P=0.0)

GAMES = {g.name: g for g in (POST_PD, APPENDIX_C_PD, POST_PD_T10)}


def grade(
    game: Game,
    actions: np.ndarray,
    scheme: Scheme = "coupled",
    *,
    rho: float = 0.0,
    q: float = 0.5,
) -> np.ndarray:
    """Rewards for one group of actions under a grading scheme.

    ``rho`` is used by ``coupled``; ``q`` (the external opponent's cooperation rate)
    by ``stranger``. Groups need at least two trajectories for the round robin.
    """
    a = np.asarray(actions)
    n = a.size
    if scheme == "stranger":
        return q * game.u(a, COOPERATE) + (1 - q) * game.u(a, DEFECT)
    if scheme == "stranger-welfare":
        return q * (game.u(a, COOPERATE) + game.u(COOPERATE, a)) + (1 - q) * (
            game.u(a, DEFECT) + game.u(DEFECT, a)
        )
    if n < 2:
        raise ValueError("round-robin grading needs a group of at least two")
    pay = game.u(a[:, None], a[None, :])  # pay[i, j] = u(a_i, a_j)
    off_diag = ~np.eye(n, dtype=bool)
    vs_others = (pay * off_diag).sum(axis=1) / (n - 1)
    if scheme == "coupled":
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must lie in [0, 1], got {rho}")
        return rho * np.diag(pay) + (1 - rho) * vs_others
    if scheme == "welfare":
        given = (pay.T * off_diag).sum(axis=1) / (n - 1)  # what the others earn facing me
        return vs_others + given
    raise ValueError(f"unknown scheme {scheme!r}")


def dr_grpo_advantages(rewards: np.ndarray) -> np.ndarray:
    """Group-centered rewards without std normalization (Dr. GRPO, as in the post)."""
    r = np.asarray(rewards, dtype=float)
    return r - r.mean()
