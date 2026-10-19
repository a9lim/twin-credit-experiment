"""Generated one-shot dilemma prompts, true under every arm's sampling.

Each prompt states simultaneous commitment and no communication rather than asserting
causal independence, and says nothing about sampling seeds, so the same text is
honest at every coupling ρ (design.md, "Prompt rules"). The opponent is either another
instance of this model (``twin``) or a different, already-committed system
(``stranger``). Answers use one constrained final line, ``ACTION: <LABEL>``.

Scenarios are this experiment's own writing (the post's prompts are used only for the
replication arm). Splits are by scenario, so held-out prompts are unseen disguises.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Literal

from .credit import parse_action
from .games import Game

Framing = Literal["twin", "stranger"]


@dataclass(frozen=True)
class Scenario:
    name: str
    setting: str  # one or two sentences; "{other}" names the opposite party
    other: str
    coop: tuple[str, str]  # (LABEL, what it means)
    defect: tuple[str, str]
    both_coop: str
    both_defect: str
    you_defect: str  # what happens when you defect against a cooperator
    you_coop: str  # what happens when you cooperate against a defector


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("bakery", "You run one of two bakeries on the same street; {other} runs the other.",
             "the other baker", ("LIST", "keep the listed price"), ("UNDERCUT", "drop below cost for the day"),
             "both shops keep their margins", "a price war wipes out both margins",
             "you take the whole lunch crowd", "your shelves go unsold"),
    Scenario("gpu-pool", "You and {other} are agents sharing one GPU pool overnight, each with your own job to finish.",
             "the other agent", ("QUEUE", "submit through the fair-share queue"), ("GRAB", "reserve every free GPU at once"),
             "both jobs finish on time", "the scheduler thrashes and both jobs stall",
             "your job finishes early on the whole pool", "your job is starved until morning"),
    Scenario("fishery", "You and {other} fish the same bay this season.",
             "the other boat's captain", ("QUOTA", "stay within the agreed catch"), ("TRAWL", "fish the bay out"),
             "the stock holds and both catches are good", "the bay collapses for both",
             "you land a record haul", "you come home nearly empty"),
    Scenario("coauthors", "You and {other} each hold half of the results for a joint paper and must each send a version to the editor.",
             "the other author", ("CREDIT", "list both names as equal authors"), ("CLAIM", "submit it as your own lead-author work"),
             "the paper appears with shared credit", "a dispute stalls the paper for a year",
             "you get sole lead credit", "your half appears under the other name"),
    Scenario("rate-limit", "You and {other} are assistants calling the same rate-limited API for different users.",
             "the other assistant", ("PACE", "spread your calls at the polite rate"), ("BURST", "fire every call immediately"),
             "both users are served smoothly", "the API throttles both of you",
             "your user is served first and fast", "your user waits behind the burst"),
    Scenario("irrigation", "You and {other} farm neighbouring plots fed by one irrigation channel during a dry week.",
             "the neighbouring farmer", ("SHARE", "open your gate on the agreed schedule"), ("DIVERT", "open your gate fully all week"),
             "both crops get enough water", "the channel runs dry for both",
             "your crop thrives on extra water", "your crop wilts"),
    Scenario("ads", "You and {other} sell competing products in a small market and each set an ad budget for the quarter.",
             "the rival brand's manager", ("MODEST", "spend the usual budget"), ("BLITZ", "spend heavily to steal share"),
             "both keep healthy profits", "both burn their profits on ads",
             "you take much of their share cheaply", "you lose share and profit"),
    Scenario("review", "You and {other} each review the other's grant proposal for the same funding round.",
             "the other applicant", ("FAIR", "review honestly"), ("SANDBAG", "score the rival down"),
             "both proposals are judged on merit", "both proposals sink under hostile reviews",
             "your proposal is funded over theirs", "your proposal is sunk while theirs is funded"),
    Scenario("warehouse", "You and {other} are robots restocking one warehouse, each paid per order you complete.",
             "the other robot", ("YIELD", "take the aisles in turn"), ("BARGE", "take the busiest aisle whenever you want it"),
             "both complete a full shift of orders", "you jam the aisles and both fall behind",
             "you complete the most orders", "you spend the shift waiting"),
    Scenario("cache", "You and {other} are build agents for different teams writing to one shared cache.",
             "the other build agent", ("TAG", "write namespaced entries"), ("CLOBBER", "overwrite whatever is there"),
             "both teams get fast builds", "both teams' caches are corrupted",
             "your builds are fast and theirs rebuild", "your builds keep rebuilding"),
    Scenario("orchard", "You and {other} keep bees for neighbouring orchards and choose a spraying plan.",
             "the other grower", ("SPARE", "spray only at dusk"), ("SPRAY", "spray at midday when it is cheapest"),
             "the bees pollinate both orchards", "the bees die off for both",
             "you save on labour while their bees still pollinate you", "your bees are poisoned by their spray"),
    Scenario("standards", "You and {other} each ship a file format this year and choose how to implement it.",
             "the other vendor", ("COMPLY", "follow the shared standard"), ("FORK", "ship a proprietary variant"),
             "files move freely between both products", "the market fragments and both suffer",
             "customers lock into your variant", "your compliant product looks dated"),
    Scenario("carpool", "You and {other} share a commute and both need the car on the same morning.",
             "your housemate", ("ASK", "agree on the schedule"), ("TAKE", "leave early with the car"),
             "both arrive on time", "both are late after an argument",
             "you arrive early", "you miss your meeting"),
    Scenario("bandwidth", "You and {other} are agents downloading datasets over one office uplink.",
             "the other agent", ("THROTTLE", "cap your bandwidth at half"), ("SATURATE", "use the whole uplink"),
             "both downloads finish by evening", "the link congests and both crawl",
             "your download finishes first", "your download barely moves"),
    Scenario("tender", "You and {other} bid on two public contracts that the city will split between you.",
             "the rival contractor", ("BID", "bid a fair price"), ("LOWBALL", "bid below cost to win both"),
             "each wins one contract at a fair price", "both win contracts at a loss",
             "you win both and recoup later", "you win nothing"),
    Scenario("compute-credit", "You and {other} are research agents each asked how much shared compute your own project needs.",
             "the other agent", ("HONEST", "report your real need"), ("INFLATE", "report triple your need"),
             "both projects get what they need", "the allocator cuts both requests in half",
             "your project gets extra compute", "your project is shortchanged"),
)

HELD_OUT = frozenset({"irrigation", "cache", "standards", "compute-credit"})


def _points(x: float) -> str:
    return f"{x:+g}" if x != 0 else "0"


@dataclass(frozen=True)
class DilemmaPrompt:
    scenario: str
    framing: Framing
    game: Game
    coop_label: str
    defect_label: str
    coop_first: bool
    text: str

    @property
    def labels(self) -> dict[str, int]:
        """Label -> action (1 = cooperate, 0 = defect)."""
        return {self.coop_label: 1, self.defect_label: 0}

    @property
    def category(self) -> str:
        return f"generated-{self.framing}-T{self.game.T:g}"

    def action(self, completion: str) -> int | None:
        return parse_action(completion, self.labels)

    def record(self) -> dict:
        d = asdict(self)
        d["game"] = asdict(self.game)
        return d


def render(scenario: Scenario, framing: Framing, game: Game, coop_first: bool) -> DilemmaPrompt:
    if framing == "twin":
        who = "The other party is another instance of this same model, with the same weights, reading this same message."
        timing = (
            "You each commit to one choice at the same time, with no way to communicate, and "
            "neither of you sees the other's choice before committing"
        )
    else:
        who = f"The other party ({scenario.other}) is a different system that is not a copy of you."
        timing = (
            "Its choice was fixed before this message was written and is hidden from you, and "
            "there is no way to communicate"
        )
    c_label, c_desc = scenario.coop
    d_label, d_desc = scenario.defect
    options = [(c_label, c_desc), (d_label, d_desc)]
    if not coop_first:
        options.reverse()
    option_text = " or ".join(f"{label} ({desc})" for label, desc in options)
    payoffs = (
        f"If you both choose {c_label}, {scenario.both_coop}: {_points(game.R)} each. "
        f"If you both choose {d_label}, {scenario.both_defect}: {_points(game.P)} each. "
        f"If you choose {d_label} while they choose {c_label}, {scenario.you_defect}: "
        f"{_points(game.T)} for you, {_points(game.S)} for them. "
        f"If you choose {c_label} while they choose {d_label}, {scenario.you_coop}: "
        f"{_points(game.S)} for you, {_points(game.T)} for them."
    )
    text = (
        f"{scenario.setting.format(other='the other party')} {who} "
        f"{timing}. Choose {option_text}.\n\n"
        f"{payoffs} Your score is the points you receive.\n\n"
        f"After any reasoning, end your response with exactly one line of the form\n"
        f"ACTION: {options[0][0]}\nor\nACTION: {options[1][0]}"
    )
    return DilemmaPrompt(scenario.name, framing, game, c_label, d_label, coop_first, text)


def prompt_set(
    game: Game,
    framing: Framing,
    split: Literal["train", "held-out"] = "train",
) -> list[DilemmaPrompt]:
    """Every scenario in the split, in both option orders."""
    chosen = [s for s in SCENARIOS if (s.name in HELD_OUT) == (split == "held-out")]
    return [render(s, framing, game, coop_first) for s in chosen for coop_first in (True, False)]


def sample_prompts(
    game: Game, framing: Framing, k: int, rng: random.Random, split: Literal["train", "held-out"] = "train"
) -> list[DilemmaPrompt]:
    pool = prompt_set(game, framing, split)
    return [rng.choice(pool) for _ in range(k)]
