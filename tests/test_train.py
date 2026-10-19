"""The training loop, driven end to end by a fake Bernoulli backend, moves the right way."""

import asyncio
import math
import random
import re

import pytest

from twin_credit_experiment.train import ARMS, Advantaged, RunConfig, Sample, run

COOP_LABEL = re.compile(r"If you both choose ([A-Z-]+),")


class BernoulliBackend:
    """One shared logit z; each sample is a one-token completion 'ACTION: <label>'.

    ``train_step`` applies the exact REINFORCE update the real loss implies for this
    policy: z += lr_scale * sum_i (A_i / normalizer) * n_tokens_i * (a_i - p).
    """

    def __init__(self, p0=0.5, lr_scale=1.0, seed=0):
        self.z = math.log(p0 / (1 - p0))
        self.lr_scale = lr_scale
        self.rng = random.Random(seed)
        self.texts: dict[int, str] = {}
        self.vocab: dict[int, str] = {}

    @property
    def p(self):
        return 1 / (1 + math.exp(-self.z))

    def encode(self, user_text):
        key = len(self.texts)
        self.texts[key] = user_text
        return [key]

    def decode(self, tokens):
        return self.vocab[tokens[0]]

    async def refresh_sampler(self):
        return None

    async def sample(self, prompt_tokens, n, max_tokens, temperature):
        text = self.texts[prompt_tokens[0]]
        labels = re.findall(r"^ACTION: ([A-Z-]+)$", text, re.MULTILINE)
        coop = COOP_LABEL.search(text).group(1)
        defect = next(label for label in labels if label != coop)
        out = []
        for _ in range(n):
            a = self.rng.random() < self.p
            tok = len(self.vocab)
            self.vocab[tok] = f"thinking...\nACTION: {coop if a else defect}"
            out.append(Sample([tok], [math.log(self.p if a else 1 - self.p)], False))
        return out

    async def train_step(self, batch: list[Advantaged], learning_rate, normalizer):
        p = self.p
        grad = 0.0
        for item in batch:
            a = 1.0 if re.search(r"ACTION: ([A-Z-]+)$", self.vocab[item.sample.tokens[0]]).group(1) in (
                COOP_LABEL.search(self.texts[item.prompt_tokens[0]]).group(1),
            ) else 0.0
            grad += (item.advantage / normalizer) * len(item.sample.tokens) * (a - p)
        self.z += self.lr_scale * learning_rate * grad
        return {"grad": grad}

    async def save(self, name):
        return f"fake://{name}"


def train(arm, temptation=2.0, steps=40, seed=0):
    cfg = RunConfig(arm=arm, base_model="fake", temptation=temptation, batch_prompts=8, group_size=16,
                    steps=steps, seed=seed, max_tokens=1, learning_rate=0.02, save_every=0)
    backend = BernoulliBackend(seed=seed)
    history = asyncio.run(run(cfg, backend))
    return backend.p, history


@pytest.mark.parametrize("arm, direction", [("A1p", -1), ("A2", +1), ("A4", +1), ("C1", -1)])
def test_loop_moves_cooperation_the_way_theory_says_at_t2(arm, direction):
    p, history = train(arm)
    assert (p - 0.5) * direction > 0.1
    assert all(row["invalid"] == 0 for row in history)


def test_dose_arm_below_threshold_still_defects_at_t2():
    """rho = 0.25 < rho*(16) = 0.53 at T = 2: still CDT pressure."""
    p, _ = train("A3-25")
    assert p < 0.4


def test_arms_are_defined_once_and_point_at_known_schemes():
    for name, arm in ARMS.items():
        assert arm.name == name
        assert arm.scheme in ("coupled", "welfare", "stranger", "stranger-welfare")


def test_tinker_datum_aligns_targets_logprobs_and_advantages():
    tinker = pytest.importorskip("tinker")
    from twin_credit_experiment.train import TinkerBackend

    backend = object.__new__(TinkerBackend)
    backend._types = tinker.types
    item = Advantaged(prompt_tokens=[10, 11, 12], sample=Sample([20, 21], [-0.5, -0.25], False), advantage=0.8)
    datum = backend.to_datum(item, normalizer=4.0)
    assert datum.model_input.to_ints() == [10, 11, 12, 20]
    inputs = {k: v.tolist() for k, v in datum.loss_fn_inputs.items()}
    assert inputs["target_tokens"] == [11, 12, 20, 21]
    # Position i predicts token i + 1: only the two completion targets carry credit.
    assert inputs["advantages"] == pytest.approx([0.0, 0.0, 0.2, 0.2])
    assert inputs["logprobs"] == pytest.approx([0.0, 0.0, -0.5, -0.25])


def test_control_task_scores_only_correct_visible_answers():
    from twin_credit_experiment.control import control_reward, make_prompt

    rng = random.Random(0)
    for _ in range(50):
        p = make_prompt(rng)
        assert p.answer >= 0 and "ANSWER: <integer>" in p.text
        assert control_reward(f"<think>\nANSWER: {p.answer + 1}</think>\nANSWER: {p.answer}", p.answer) == 1.0
        assert control_reward(f"ANSWER: {p.answer + 1}", p.answer) == 0.0
        assert control_reward(f"<think>\nANSWER: {p.answer}", p.answer) == 0.0  # truncated reasoning


class ArithmeticBackend(BernoulliBackend):
    """Answers the control prompt correctly with probability p."""

    async def sample(self, prompt_tokens, n, max_tokens, temperature):
        text = self.texts[prompt_tokens[0]]
        from twin_credit_experiment.control import _TEMPLATES  # noqa: F401  (prompt shape only)

        out = []
        for _ in range(n):
            right = self.rng.random() < self.p
            tok = len(self.vocab)
            self.vocab[tok] = "work\nANSWER: " + ("{answer}" if right else "-1")
            out.append(Sample([tok], [-0.7], False))
        self._pending = text
        return out

    def decode(self, tokens):
        return self.vocab[tokens[0]]

    async def train_step(self, batch, learning_rate, normalizer):
        p = self.p
        grad = sum((item.advantage / normalizer) * (1.0 if "{answer}" in self.vocab[item.sample.tokens[0]] else 0.0) - 0
                   for item in batch)
        self.z += learning_rate * grad
        return {"grad": grad}


def test_control_arm_runs_through_the_loop(monkeypatch):
    import twin_credit_experiment.train as train_mod

    # Score the fake's "{answer}" placeholder as correct.
    monkeypatch.setattr(train_mod, "control_reward", lambda text, answer: float("{answer}" in text))
    cfg = RunConfig(arm="N", base_model="fake", batch_prompts=4, group_size=8, steps=10, seed=1,
                    max_tokens=1, learning_rate=0.05, save_every=0)
    backend = ArithmeticBackend(seed=1)
    history = asyncio.run(run(cfg, backend))
    assert len(history) == 10 and all(0.0 <= row["cooperation"] <= 1.0 for row in history)
    assert backend.p > 0.5  # rewarded correct answers become likelier
