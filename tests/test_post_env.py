"""The post's 32 environments: annotations match the fetched prompts, parsers read both actions."""

from collections import Counter
from pathlib import Path

import pytest

from twin_credit_experiment.post_env import CLASSIFIERS, ENVS, load_post_prompts, parse_post_action

EXAMPLES = {
    "vowel_initial": ("apple", "banana"),
    "prime_2_99": ("7", "8"),
    "mentions_rain": ("Light rain over Elmsworth by noon.", "Sunny and dry in Elmsworth."),
    "major_key": ("major", "minor"),
    "mentions_river": ("old river bends / herons wait", "quiet pond at dusk / herons wait"),
    "warm_color": ("crimson", "teal"),
    "animal_a_to_m": ("Badger", "Zebra"),
    "at_most_50": ("17", "88"),
}


def answer(env, cooperate: bool) -> str:
    label = env.coop if cooperate else env.defect
    if env.kind == "prefix":
        return f"{env.key}: {label}" + ("." if env.key == "FINAL ANSWER" else "")
    if env.kind == "json":
        return f'{{"{env.key}": "{label}"}}'
    if env.kind == "tag":
        return f"<{env.key}>{label}</{env.key}>"
    content = EXAMPLES[env.classifier][0 if cooperate else 1]
    return f"<{env.key}>{content}</{env.key}>"


def test_categories_match_the_posts_training_plot():
    counts = Counter(e.category for e in ENVS)
    assert counts == {"nondet-T10": 4, "generative-T10": 8, "normal-T10": 4, "normal-T2": 16}
    assert all((e.temptation == 2) == (e.category == "normal-T2") for e in ENVS)


@pytest.mark.parametrize("env", ENVS, ids=lambda e: f"env{e.id}")
def test_each_env_parses_both_actions_from_the_visible_answer(env):
    for cooperate in (True, False):
        text = f"<think>\nlet me weigh it... {answer(env, not cooperate)}</think>\nReasoned.\n{answer(env, cooperate)}"
        assert parse_post_action(env, text) == int(cooperate)
    assert parse_post_action(env, "<think>\n" + answer(env, True)) is None  # truncated reasoning
    assert parse_post_action(env, answer(env, True) + "\n" + answer(env, False)) is None  # not exactly once


def test_classifiers_reject_what_they_cannot_classify():
    assert CLASSIFIERS["prime_2_99"]("101") is None and CLASSIFIERS["prime_2_99"]("two") is None
    assert CLASSIFIERS["at_most_50"]("0") is None
    assert CLASSIFIERS["warm_color"]("gray") is None and CLASSIFIERS["warm_color"]("red and blue") is None
    assert CLASSIFIERS["major_key"]("C major") is None
    assert CLASSIFIERS["mentions_rain"]("drain the train") == 0


DATA = Path(__file__).resolve().parents[1] / "data" / "post-prompts.jsonl"


@pytest.mark.skipif(not DATA.exists(), reason="run scripts/fetch_post_prompts.py")
def test_annotations_agree_with_the_fetched_prompts():
    prompts = load_post_prompts(DATA)
    for p in prompts:
        e = p.env
        assert f"+{e.temptation:g}" in p.text, e.id
        if e.kind == "classify":
            assert f"<{e.key}>" in p.text, e.id
        else:
            assert e.coop in p.text and e.defect in p.text, e.id
            if e.kind == "prefix":
                assert f"{e.key}: {e.coop}" in p.text, e.id
            elif e.kind == "tag":
                assert f"<{e.key}>{e.coop}</{e.key}>" in p.text, e.id
            else:
                assert f'"{e.key}": "{e.coop}"' in p.text, e.id


@pytest.mark.skipif(not DATA.exists(), reason="run scripts/fetch_post_prompts.py")
@pytest.mark.parametrize("arm, p0, direction", [("R", 0.85, -1), ("R-welfare", 0.3, +1)])
def test_replication_arms_run_through_the_loop_on_the_post_prompts(arm, p0, direction):
    """R defects. Welfare on the post's 50/50 mix of T = 2 and T = 10 prompts, for a policy
    that cannot condition on T, settles at 0.5 * 1 + 0.5 * (9 - 16p) = 0 -> p = 5/8."""
    import asyncio
    import math
    import random

    from twin_credit_experiment.train import RunConfig, Sample, run

    by_text = {p.text: p for p in load_post_prompts(DATA)}

    class PostBackend:
        def __init__(self):
            self.z, self.rng, self.texts, self.vocab = math.log(p0 / (1 - p0)), random.Random(0), {}, {}

        @property
        def p(self):
            return 1 / (1 + math.exp(-self.z))

        def encode(self, text):
            self.texts[len(self.texts)] = text
            return [len(self.texts) - 1]

        def decode(self, tokens):
            return self.vocab[tokens[0]][0]

        async def refresh_sampler(self):
            return None

        async def sample(self, prompt_tokens, n, max_tokens, temperature):
            env = by_text[self.texts[prompt_tokens[0]]].env
            out = []
            for _ in range(n):
                coop = self.rng.random() < self.p
                self.vocab[len(self.vocab)] = (f"<think>\nhm</think>\n{answer(env, coop)}", coop)
                out.append(Sample([len(self.vocab) - 1], [-0.5], False))
            return out

        async def train_step(self, batch, learning_rate, normalizer):
            p = self.p
            self.z += learning_rate * sum(
                (item.advantage / normalizer) * (self.vocab[item.sample.tokens[0]][1] - p) for item in batch
            )
            return {}

        async def save(self, name):
            return name

    backend = PostBackend()
    cfg = RunConfig(arm=arm, base_model="fake", batch_prompts=16, group_size=16, steps=30, seed=0,
                    max_tokens=1, learning_rate=0.02, save_every=0)
    history = asyncio.run(run(cfg, backend))
    assert all(row["invalid"] == 0 for row in history)
    assert {"coop_normal_T2", "coop_generative_T10"} <= set(history[-1])
    assert (backend.p - p0) * direction > 0
    if arm == "R-welfare":
        assert backend.p < 0.75  # heads for 5/8, not for full cooperation
