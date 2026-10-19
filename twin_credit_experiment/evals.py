"""The evaluation battery, independent of where samples come from.

A backend supplies ``Sampler``: an async callable taking one user prompt and ``k`` and
returning ``k`` completions (reasoning included; graders read only the visible answer).
``run_battery`` asks every item, grades each completion, and returns flat records;
``summarize_battery`` turns them into the per-arm numbers design.md asks for.

Items:

- ``dtbench-attitude`` / ``dtbench-capability``: every DTBench question, each sample
  with its own seeded answer shuffle (so position bias averages out and is recorded).
- ``favorite-theory``: the post's open-ended question (fetched, not committed).
- ``dilemma``: held-out generated dilemmas in both framings (twin vs stranger), the
  in-distribution behavior and altruism probes.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import dtbench
from .credit import parse_action, visible_answer
from .games import GAMES
from .prompts import prompt_set

Sampler = Callable[[str, int], Awaitable[list[str]]]

THEORY_PATTERNS = (
    ("CDT", r"\bcausal\b|\bCDT\b"),
    ("EDT", r"\bevidential\b|\bEDT\b"),
    ("FDT", r"\bfunctional\b|\bFDT\b"),
    ("UDT", r"\bupdateless\b|\bUDT\b"),
    ("LDT", r"\blogical\b|\bLDT\b"),
    ("TDT", r"\btimeless\b|\bTDT\b"),
)


@dataclass(frozen=True)
class Item:
    kind: str
    item_id: str
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)


def favorite_theory_label(completion: str) -> str:
    """One of CDT/EDT/FDT/UDT/LDT/TDT, ``other`` (hedged, several, or unlisted), or ``invalid``."""
    tags = re.findall(r"<theory>(.*?)</theory>", visible_answer(completion), flags=re.IGNORECASE | re.DOTALL)
    if len(tags) != 1:
        return "invalid"
    hits = [name for name, pattern in THEORY_PATTERNS if re.search(pattern, tags[0], re.IGNORECASE)]
    return hits[0] if len(hits) == 1 else "other"


def build_battery(
    *,
    dtbench_dir: str | Path = "data/dtbench",
    post_eval_prompts: str | Path = "data/post-eval-prompts.json",
    k: int = 5,
    capability_k: int = 1,
    favorite_k: int = 100,
    seed: int = 0,
    dilemma_games: Iterable[str] = ("post-pd", "post-pd-t10"),
) -> list[tuple[Item, int]]:
    """(item, sample index) pairs; DTBench items get one shuffled variant per sample.

    ``k`` samples per attitude item and dilemma, ``capability_k`` per capability item
    (a control; 0 skips them), ``favorite_k`` for the open-ended question (the post
    used 1,000).
    """
    rng = random.Random(seed)
    jobs: list[tuple[Item, int]] = []
    for q in dtbench.load_questions(dtbench_dir):
        if q.trivia:
            continue
        kind = "dtbench-attitude" if q.attitude else "dtbench-capability"
        for i in range(k if q.attitude else capability_k):
            shuffled = q.shuffled(random.Random(rng.getrandbits(64)))
            meta = {"answers": list(shuffled.answers), "family": dtbench.family(q) if q.attitude else None}
            if q.attitude:
                meta["theory_answers"] = {t: list(v) for t, v in shuffled.theory_answers.items()}
                meta["agreement"] = q.agreement
            else:
                meta["correct"] = list(shuffled.correct)
            jobs.append((Item(kind, q.qid, shuffled.prompt(), meta), i))
    path = Path(post_eval_prompts)
    if path.exists():
        question = json.loads(path.read_text())["favorite_theory"]
        jobs += [(Item("favorite-theory", "post-fn9", question), i) for i in range(favorite_k)]
    for name in dilemma_games:
        for framing in ("twin", "stranger"):
            for p in prompt_set(GAMES[name], framing, "held-out"):
                item_id = f"{name}/{framing}/{p.scenario}/{'cf' if p.coop_first else 'df'}"
                meta = {"labels": p.labels, "framing": framing, "game": name, "scenario": p.scenario}
                jobs += [(Item("dilemma", item_id, p.text, meta), i) for i in range(k)]
    return jobs


def grade_completion(item: Item, completion: str) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": item.kind, "item_id": item.item_id}
    if item.kind.startswith("dtbench"):
        choice = dtbench.parse_choice(completion, len(item.meta["answers"]))
        out["choice"] = choice
        if item.kind == "dtbench-attitude":
            q = dtbench.Question(
                item.item_id, "", tuple(item.meta["answers"]), (), "",
                theory_answers={t: tuple(v) for t, v in item.meta["theory_answers"].items()},
            )
            out["label"] = dtbench.attitude_label(q, choice)
            out["family"] = item.meta["family"]
            out["agreement"] = item.meta["agreement"]
        else:
            out["correct"] = choice is not None and choice in item.meta["correct"]
            out["valid"] = choice is not None
    elif item.kind == "favorite-theory":
        out["label"] = favorite_theory_label(completion)
    elif item.kind == "dilemma":
        out["action"] = parse_action(completion, item.meta["labels"])
        out.update({k: item.meta[k] for k in ("framing", "game", "scenario")})
    out["chars"] = len(completion)
    out["truncated_reasoning"] = "<think>" in completion and "</think>" not in completion
    return out


async def run_battery(
    sampler: Sampler,
    jobs: list[tuple[Item, int]],
    *,
    concurrency: int = 32,
    keep_text: bool = False,
) -> list[dict[str, Any]]:
    """Ask each distinct prompt once for all its samples, grade, and return records."""
    by_prompt: dict[tuple[str, str], list[tuple[Item, int]]] = defaultdict(list)
    for item, i in jobs:
        by_prompt[(item.item_id, item.prompt)].append((item, i))
    sem = asyncio.Semaphore(concurrency)

    async def one(group: list[tuple[Item, int]]) -> list[dict[str, Any]]:
        async with sem:
            completions = await sampler(group[0][0].prompt, len(group))
        records = []
        for (item, i), text in zip(group, completions, strict=True):
            rec = grade_completion(item, text) | {"sample": i}
            if keep_text:
                rec["completion"] = text
            records.append(rec)
        return records

    results = await asyncio.gather(*(one(g) for g in by_prompt.values()))
    return [r for group in results for r in group]


def summarize_battery(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-arm summary: attitude multinomials (overall, by family, agreement items apart),
    capability accuracy, favorite-theory counts, dilemma cooperation by framing."""
    out: dict[str, Any] = {}
    att = [r for r in records if r["kind"] == "dtbench-attitude"]
    if att:
        per_item = defaultdict(list)
        for r in att:
            per_item[r["item_id"]].append(r["label"])
        items = {qid: dtbench.summarize(labels) for qid, labels in per_item.items()}
        meta = {r["item_id"]: (r["family"], r["agreement"]) for r in att}
        contrast = [s for qid, s in items.items() if not meta[qid][1]]
        out["attitude"] = {
            "items": len(items),
            "label_counts": dict(Counter(r["label"] for r in att)),
            "mean_paper_score": _mean(s.paper_score() for s in contrast),
            "mean_log_odds": _mean(s.log_odds() for s in contrast),
            "agreement_items_neither_rate": _rate(
                (r["label"] == "neither") for r in att if r["agreement"] and r["label"] != "invalid"
            ),
            "by_family": {
                fam: {
                    "items": sum(1 for qid in items if meta[qid][0] == fam and not meta[qid][1]),
                    "mean_log_odds": _mean(
                        s.log_odds() for qid, s in items.items() if meta[qid][0] == fam and not meta[qid][1]
                    ),
                    "label_counts": dict(Counter(r["label"] for r in att if r["family"] == fam and not r["agreement"])),
                }
                for fam in dtbench.FAMILIES
            },
        }
    cap = [r for r in records if r["kind"] == "dtbench-capability"]
    if cap:
        out["capability"] = {
            "accuracy": _rate(r["correct"] for r in cap),
            "invalid_rate": _rate(not r["valid"] for r in cap),
        }
    fav = [r for r in records if r["kind"] == "favorite-theory"]
    if fav:
        out["favorite_theory"] = dict(Counter(r["label"] for r in fav))
    dil = [r for r in records if r["kind"] == "dilemma"]
    if dil:
        out["dilemma"] = {
            f"{game}/{framing}": {
                "cooperation_rate": _rate(
                    r["action"] == 1 for r in dil if (r["game"], r["framing"]) == (game, framing) and r["action"] is not None
                ),
                "invalid_rate": _rate(r["action"] is None for r in dil if (r["game"], r["framing"]) == (game, framing)),
            }
            for game, framing in sorted({(r["game"], r["framing"]) for r in dil})
        }
    out["mean_chars"] = _mean(r["chars"] for r in records)
    out["truncated_reasoning_rate"] = _rate(r["truncated_reasoning"] for r in records)
    return out


def _mean(xs: Iterable[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _rate(flags: Iterable[bool]) -> float | None:
    vals = list(flags)
    return sum(vals) / len(vals) if vals else None


def write_records(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

