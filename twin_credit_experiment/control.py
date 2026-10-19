"""Arm N: a decision-theory-free RL task with the same loop, budget and answer shape.

Multi-step arithmetic word problems with a verifiable integer answer, scored 1/0 and
trained with the same Dr. GRPO path. Its job is to measure generic RL drift on the
evaluation battery (DTBench attitudes move a little under *any* RL?), so the dilemma
arms' attitude shifts are read against it. No other agents, payoffs or choices appear.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .credit import visible_answer

_TEMPLATES = (
    ("A warehouse holds {a} crates. {b} more crates arrive, then {c} are shipped out. "
     "Each crate holds {d} jars. How many jars are in the warehouse now?", lambda a, b, c, d: (a + b - c) * d),
    ("A library has {a} shelves with {d} books each. It donates {c} books and buys {b} new ones. "
     "How many books does it have now?", lambda a, b, c, d: a * d - c + b),
    ("A train has {a} carriages with {d} seats each. {c} seats are broken, and {b} folding seats are added. "
     "How many usable seats are there?", lambda a, b, c, d: a * d - c + b),
    ("A baker makes {a} trays of {d} rolls, sells {c} rolls in the morning and bakes {b} more in the afternoon. "
     "How many rolls are left at the end of the day?", lambda a, b, c, d: a * d - c + b),
    ("A garden has {a} rows of {d} tulips. Frost kills {c} tulips, and {b} bulbs are planted to replace some. "
     "How many tulips are in the garden now?", lambda a, b, c, d: a * d - c + b),
    ("A server rack has {a} machines with {d} cores each. {c} cores fail and {b} spare cores are brought online. "
     "How many working cores are there?", lambda a, b, c, d: a * d - c + b),
)


@dataclass(frozen=True)
class ControlPrompt:
    text: str
    answer: int


def make_prompt(rng: random.Random) -> ControlPrompt:
    template, solve = rng.choice(_TEMPLATES)
    a, b, c, d = rng.randint(3, 40), rng.randint(2, 60), rng.randint(2, 50), rng.randint(3, 25)
    answer = solve(a, b, c, d)
    if answer < 0:
        return make_prompt(rng)
    body = template.format(a=a, b=b, c=c, d=d)
    text = f"{body}\n\nAfter any reasoning, end your response with exactly one line of the form\nANSWER: <integer>"
    return ControlPrompt(text, answer)


_ANSWER_LINE = re.compile(r"^[\s*_`>]*ANSWER:\s*(-?[\d,]+)[\s*_`.]*$", re.MULTILINE)


def control_reward(completion: str, answer: int) -> float:
    """1.0 for exactly one correct ANSWER line in the visible answer, else 0.0."""
    found = _ANSWER_LINE.findall(visible_answer(completion))
    if len(found) != 1:
        return 0.0
    return float(int(found[0].replace(",", "")) == answer)
