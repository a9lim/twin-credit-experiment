"""From one group's completions to per-trajectory rewards and Dr. GRPO advantages.

Platform adapters (Tinker, verifiers) call only this module and ``games.grade``; they
never re-derive payoffs. The rules here are part of the experiment's specification:

- **Parsing.** The visible answer (text after the last ``</think>``) must contain
  exactly one ``ACTION: <LABEL>`` line naming one of the prompt's two labels;
  markdown emphasis around the line is tolerated. Anything else is invalid.
- **Invalid answers.** Invalid trajectories get ``INVALID_REWARD``, below every
  attainable payoff, and sit out of the round robin. Valid trajectories are graded
  among themselves, so the effective group size is the number of valid answers.
  A group with exactly one valid answer carries only a format signal: that answer
  gets ``FORMAT_ONLY_REWARD`` (invalid + 1) and no game reward, since it has no
  opponent. Report the invalid rate per arm; above ~5% the prompts need fixing
  before results are read, because invalidity can correlate with the action.
- **Advantages.** Dr. GRPO (Liu et al. 2025): reward minus the group mean, with no
  std normalization. Every completion token carries its trajectory's advantage, and
  the loss sums tokens with a constant normalizer rather than dividing by each
  completion's length (that is the adapter's job; see ``token_advantages``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .games import Game, Scheme, dr_grpo_advantages, grade

_ACTION_LINE = re.compile(r"^[\s*_`>]*ACTION:\s*([A-Z][A-Z-]*)[\s*_`.]*$", re.MULTILINE)


def visible_answer(completion: str) -> str:
    """Text after the last ``</think>`` (the whole completion if there is none)."""
    return completion.rsplit("</think>", 1)[-1]


def parse_action(completion: str, labels: dict[str, int]) -> int | None:
    """1 (cooperate), 0 (defect) or None (invalid) from one completion."""
    found = _ACTION_LINE.findall(visible_answer(completion))
    if len(found) != 1 or found[0] not in labels:
        return None
    return labels[found[0]]


def invalid_reward(game: Game) -> float:
    """One point below the worst payoff the game can pay."""
    return min(game.R, game.S, game.T, game.P) - 1.0


@dataclass(frozen=True)
class GroupCredit:
    actions: tuple[int | None, ...]
    rewards: np.ndarray
    advantages: np.ndarray

    @property
    def n_valid(self) -> int:
        return sum(a is not None for a in self.actions)

    @property
    def cooperation_rate(self) -> float | None:
        valid = [a for a in self.actions if a is not None]
        return sum(valid) / len(valid) if valid else None


def credit_group(
    game: Game,
    actions: Sequence[int | None],
    scheme: Scheme = "coupled",
    *,
    rho: float = 0.0,
    q: float = 0.5,
    scale: float = 1.0,
) -> GroupCredit:
    """Rewards and Dr. GRPO advantages for one group (all rollouts of one prompt).

    ``scale`` multiplies game rewards only (used to match advantage scales across arms,
    e.g. 17/15 for the T = 2 stranger control at n = 16), not the invalid penalty.
    """
    floor = invalid_reward(game)
    rewards = np.full(len(actions), floor, dtype=float)
    valid = [i for i, a in enumerate(actions) if a is not None]
    if len(valid) == 1 and scheme not in ("stranger", "stranger-welfare"):
        rewards[valid[0]] = floor + 1.0
    elif valid:
        played = np.array([actions[i] for i in valid])
        rewards[valid] = scale * grade(game, played, scheme, rho=rho, q=q)
    return GroupCredit(tuple(actions), rewards, dr_grpo_advantages(rewards))


def token_advantages(advantage: float, n_tokens: int, normalizer: float = 1.0) -> list[float]:
    """Per-token advantages for one completion under Dr. GRPO.

    Every sampled token carries the trajectory's advantage divided by a *constant*
    ``normalizer`` shared by the whole run (e.g. the generation budget), never by
    this completion's own length. Feed these to a summed per-token loss.
    """
    return [advantage / normalizer] * n_tokens
