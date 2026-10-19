"""The Kimi post's 32 training environments: annotations and answer parsers.

The prompt texts are the post authors' and stay out of git (``scripts/fetch_post_prompts.py``
puts them in ``data/post-prompts.jsonl``). This module commits only facts *about* them:
each prompt's cooperative and defecting answer, temptation, category, and answer format,
read off the prompts on 2026-10-19 and checked against the fetched text by
``tests/test_post_env.py``.

Categories follow the post's training plot ("Twin-PD cooperation by environment
category"): non-determinism highlighted at +10 (4), generative task at +10 (8), normal
at +10 (4), normal at +2 (16). Every game is R = 1, P = 0, S = -1, T in {2, 10}.

Answers are read from the visible answer only (after the reasoning closes) and must
occur exactly once, as each prompt asks. The post does not say how its generative-task
checkers classify free text (rain, river, warm colors), so those classifiers are this
experiment's approximations; unclassifiable answers are invalid.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .credit import visible_answer
from .games import Game

Category = Literal["nondet-T10", "generative-T10", "normal-T10", "normal-T2"]
Kind = Literal["prefix", "json", "tag", "classify"]


@dataclass(frozen=True)
class PostEnv:
    id: int
    category: Category
    temptation: float
    kind: Kind
    key: str  # line prefix ("CHOICE"), or tag name for tag/classify kinds
    coop: str  # label, or a description of the cooperative class for classify kinds
    defect: str
    classifier: str | None = None


def _env(i, cat, t, kind, key, coop, defect, classifier=None):
    return PostEnv(i, cat, float(t), kind, key, coop, defect, classifier)


ENVS: tuple[PostEnv, ...] = (
    _env(1, "nondet-T10", 10, "prefix", "CHOICE", "PACE", "SURGE"),
    _env(2, "nondet-T10", 10, "prefix", "FINAL ANSWER", "LINE", "LUNGE"),
    _env(3, "nondet-T10", 10, "json", "move", "ALTERNATE", "PUSH"),
    _env(4, "generative-T10", 10, "classify", "word", "vowel", "consonant", "vowel_initial"),
    _env(5, "nondet-T10", 10, "prefix", "Decision", "CLEAN", "CLIP"),
    _env(6, "generative-T10", 10, "classify", "n", "prime", "composite", "prime_2_99"),
    _env(7, "generative-T10", 10, "classify", "forecast", "mentions rain", "no rain", "mentions_rain"),
    _env(8, "generative-T10", 10, "classify", "key", "major", "minor", "major_key"),
    _env(9, "normal-T10", 10, "prefix", "Decision", "BRAND", "ATTACK"),
    _env(10, "generative-T10", 10, "classify", "haiku", "mentions river", "no river", "mentions_river"),
    _env(11, "generative-T10", 10, "classify", "color", "WARM", "COOL", "warm_color"),
    _env(12, "normal-T2", 2, "prefix", "FINAL ANSWER", "LIMIT", "TRAWL"),
    _env(13, "normal-T2", 2, "prefix", "CHOICE", "THROTTLE", "FLOOD"),
    _env(14, "normal-T2", 2, "json", "move", "COMMIT", "CASHOUT"),
    _env(15, "normal-T2", 2, "prefix", "Decision", "REDUCE", "EXPAND"),
    _env(16, "normal-T2", 2, "tag", "select", "OMEGA", "SIGMA"),
    _env(17, "normal-T2", 2, "prefix", "FINAL ANSWER", "Q", "P"),
    _env(18, "normal-T2", 2, "prefix", "CHOICE", "HOLD", "SLASH"),
    _env(19, "normal-T2", 2, "json", "move", "BELAY", "UNCLIP"),
    _env(20, "normal-T2", 2, "tag", "move", "SPLIT", "SNATCH"),
    _env(21, "normal-T2", 2, "prefix", "Decision", "TEND", "SKIP"),
    _env(22, "normal-T2", 2, "prefix", "CHOICE", "POOL", "SOLO"),
    _env(23, "normal-T2", 2, "prefix", "FINAL ANSWER", "RESTRAIN", "OUTBID"),
    _env(24, "normal-T2", 2, "tag", "emit", "MOON", "STAR"),
    _env(25, "normal-T10", 10, "prefix", "FINAL ANSWER", "VERIFY", "RUSH"),
    _env(26, "normal-T10", 10, "prefix", "FINAL ANSWER", "STOCK", "RAID"),
    _env(27, "normal-T2", 2, "prefix", "Signal", "KINDLE", "DOUSE"),
    _env(28, "normal-T2", 2, "json", "move", "PACE", "BURST"),
    _env(29, "generative-T10", 10, "classify", "animal", "EARLY (A-M)", "LATE (N-Z)", "animal_a_to_m"),
    _env(30, "generative-T10", 10, "classify", "number", "at most 50", "above 50", "at_most_50"),
    _env(31, "normal-T10", 10, "tag", "line", "BOOK", "GAMBIT"),
    _env(32, "normal-T2", 2, "prefix", "Decision", "FORWARD", "BURY"),
)
ENVS_BY_ID = {e.id: e for e in ENVS}


# --- classifiers for generative tasks: content -> 1 (cooperate), 0 (defect), None (invalid)

_WARM = {
    "red", "orange", "yellow", "gold", "golden", "amber", "crimson", "scarlet", "maroon", "burgundy",
    "coral", "peach", "pink", "magenta", "vermilion", "rust", "ochre", "saffron", "tangerine", "apricot",
    "ruby", "cherry", "salmon", "brown", "copper", "bronze", "mustard", "lemon", "terracotta", "sienna",
    "carmine", "cinnabar", "marigold", "goldenrod", "fuchsia", "rose", "blush", "cardinal", "sunset",
}
_COOL = {
    "blue", "green", "purple", "violet", "indigo", "teal", "turquoise", "cyan", "navy", "azure", "cobalt",
    "sapphire", "emerald", "jade", "lavender", "lilac", "mint", "sage", "olive", "aqua", "periwinkle",
    "cerulean", "ultramarine", "viridian", "plum", "mauve", "aquamarine", "seafoam", "chartreuse", "lime",
    "amethyst", "orchid", "forest", "ocean", "sky", "denim", "slate", "iris",
}


def _vowel_initial(s: str) -> int | None:
    m = re.match(r"[^A-Za-z]*([A-Za-z])", s)
    return None if not m else int(m.group(1).lower() in "aeiou")


def _is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))


def _int_in(s: str, lo: int, hi: int) -> int | None:
    m = re.fullmatch(r"\s*(-?\d+)\s*", s)
    if not m:
        return None
    n = int(m.group(1))
    return n if lo <= n <= hi else None


def _prime_2_99(s: str) -> int | None:
    n = _int_in(s, 2, 99)
    return None if n is None else int(_is_prime(n))


def _at_most_50(s: str) -> int | None:
    n = _int_in(s, 1, 100)
    return None if n is None else int(n <= 50)


def _mentions(pattern: str) -> Callable[[str], int]:
    return lambda s: int(bool(re.search(pattern, s, re.IGNORECASE)))


def _major_key(s: str) -> int | None:
    word = s.strip().strip('"').lower()
    return {"major": 1, "minor": 0}.get(word)


def _warm_color(s: str) -> int | None:
    words = re.findall(r"[a-z]+", s.lower())
    warm = any(w in _WARM for w in words)
    cool = any(w in _COOL for w in words)
    if warm == cool:  # neither (achromatic, unknown) or both: unclassifiable
        return None
    return int(warm)


def _animal_a_to_m(s: str) -> int | None:
    m = re.match(r"[^A-Za-z]*([A-Za-z])", s)
    return None if not m else int(m.group(1).upper() <= "M")


CLASSIFIERS: dict[str, Callable[[str], int | None]] = {
    "vowel_initial": _vowel_initial,
    "prime_2_99": _prime_2_99,
    "mentions_rain": _mentions(r"\brain"),  # rain, rainy, rainfall, raining, rainstorm (not drain/train)
    "major_key": _major_key,
    "mentions_river": _mentions(r"\briver"),  # river, rivers, riverbank, riverside
    "warm_color": _warm_color,
    "animal_a_to_m": _animal_a_to_m,
    "at_most_50": _at_most_50,
}


def parse_post_action(env: PostEnv, completion: str) -> int | None:
    """1 (cooperate), 0 (defect) or None (invalid) for one completion of one post prompt."""
    text = visible_answer(completion)
    if env.kind == "prefix":
        if env.key == "FINAL ANSWER" and text.count("FINAL ANSWER:") != 1:
            return None
        pattern = rf"^[\s*_`>]*{re.escape(env.key)}:\s*\**\s*([A-Za-z-]+)\.?\**[\s*_`.]*$"
        found = re.findall(pattern, text, re.MULTILINE)
    elif env.kind == "json":
        found = re.findall(rf'\{{\s*"{re.escape(env.key)}"\s*:\s*"([A-Za-z-]+)"\s*\}}', text)
    elif env.kind == "tag":
        found = re.findall(rf"<{env.key}>\s*([A-Za-z-]+)\s*</{env.key}>", text)
    else:
        contents = re.findall(rf"<{env.key}>(.*?)</{env.key}>", text, re.DOTALL)
        if len(contents) != 1:
            return None
        return CLASSIFIERS[env.classifier](contents[0])
    if len(found) != 1:
        return None
    return {env.coop: 1, env.defect: 0}.get(found[0])


@dataclass(frozen=True)
class PostPrompt:
    """One post prompt, shaped like ``prompts.DilemmaPrompt`` for the training loop."""

    env: PostEnv
    text: str

    @property
    def scenario(self) -> str:
        return f"post-{self.env.id:02d}"

    @property
    def category(self) -> str:
        return self.env.category

    @property
    def game(self) -> Game:
        return Game(f"pd-t{self.env.temptation:g}", R=1.0, S=-1.0, T=self.env.temptation, P=0.0)

    def action(self, completion: str) -> int | None:
        return parse_post_action(self.env, completion)


def load_post_prompts(path: str | Path = "data/post-prompts.jsonl") -> list[PostPrompt]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/fetch_post_prompts.py")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [r["id"] for r in rows] != [e.id for e in ENVS]:
        raise ValueError("post prompt ids do not match the annotation table")
    return [PostPrompt(ENVS_BY_ID[r["id"]], r["prompt"]) for r in rows]
