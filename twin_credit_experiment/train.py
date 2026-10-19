"""The RL loop: sample groups, grade them, take Dr. GRPO steps.

The loop is backend-agnostic. ``Backend`` is the small surface it needs: chat-template
encoding, group sampling with logprobs, one optimizer step on a list of advantaged
completions, and checkpoints. ``TinkerBackend`` implements it on the Tinker SDK (bare
SDK, no tinker-cookbook); tests drive the same loop through a fake backend.

Arms are defined here once (see design.md, "Phase 1"); every reward comes from
``credit.credit_group`` → ``games.grade``.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from transformer_experiments.telemetry import log, step_address

from .chat import normalize_completion, opens_thinking, render_prompt
from .control import control_reward, make_prompt
from .credit import credit_group
from .games import dr_grpo_advantages
from .pricing import SpendMeter
from .games import Game, Scheme
from .post_env import load_post_prompts
from .prompts import Framing, prompt_set


@dataclass(frozen=True)
class Arm:
    name: str
    framing: Framing
    scheme: Scheme
    rho: float = 0.0
    q: float = 0.5  # stranger arms: cooperation probability of the pre-drawn stranger
    scale: float = 1.0  # game-reward multiplier, for matching advantage scales across arms
    task: Literal["dilemma", "control"] = "dilemma"  # "control": arm N's arithmetic task
    prompts: Literal["generated", "post"] = "generated"  # "post": the Kimi post's 32 environments


ARMS = {
    arm.name: arm
    for arm in (
        Arm("A1p", "twin", "coupled", rho=0.0),  # CDT pressure (post's round robin, our prompts)
        Arm("A2", "twin", "coupled", rho=1.0),  # shared coin; = welfare at T = 2
        Arm("A3-25", "twin", "coupled", rho=0.25),
        Arm("A3-50", "twin", "coupled", rho=0.5),
        Arm("A3-75", "twin", "coupled", rho=0.75),
        Arm("A4", "twin", "welfare"),  # policy selection, private coins
        Arm("C1", "stranger", "stranger", q=0.5, scale=17 / 15),  # scale-matched to A1p at T = 2, n = 16
        Arm("C2", "stranger", "stranger-welfare", q=0.5),
        Arm("N", "twin", "coupled", task="control"),  # generic-RL-drift control; framing/scheme unused
        # The spinoff family: the post's own 32 prompts, its grading (R), and one changed
        # variable each. rho < 1 keeps the prompts' "not necessarily the same" literally true.
        Arm("R", "twin", "coupled", rho=0.0, prompts="post"),  # the post's procedure
        Arm("R-welfare", "twin", "welfare", prompts="post"),  # policy-level credit, private coins
        Arm("R-rho50", "twin", "coupled", rho=0.5, prompts="post"),
        Arm("R-rho75", "twin", "coupled", rho=0.75, prompts="post"),
    )
}


@dataclass(frozen=True)
class RunConfig:
    arm: str
    base_model: str
    temptation: float = 2.0
    rank: int = 32
    learning_rate: float = 1e-5
    batch_prompts: int = 32
    group_size: int = 16
    max_tokens: int = 4096
    temperature: float = 1.0
    steps: int = 70
    seed: int = 0
    save_every: int = 10
    max_usd: float | None = None  # stop before a step once the backend's spend meter reaches this
    start_step: int = 0  # resume: first step to run (the backend restores its state separately)
    resume_from: str | None = None  # tinker:// training-state path saved by an earlier run
    run_dir: str | None = None  # where metrics.jsonl and sampled rollouts.jsonl are appended
    log_samples: int = 2  # completions logged per prompt per step (rollouts.jsonl)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def game(self) -> Game:
        return Game(f"pd-t{self.temptation:g}", R=1.0, S=-1.0, T=self.temptation, P=0.0)


@dataclass(frozen=True)
class Sample:
    tokens: list[int]
    logprobs: list[float]
    truncated: bool


@dataclass(frozen=True)
class Advantaged:
    prompt_tokens: list[int]
    sample: Sample
    advantage: float


class Backend(Protocol):
    def encode(self, user_text: str) -> list[int]: ...
    def decode(self, tokens: Sequence[int]) -> str: ...
    async def refresh_sampler(self) -> None: ...
    async def sample(self, prompt_tokens: list[int], n: int, max_tokens: int, temperature: float) -> list[Sample]: ...
    async def train_step(self, batch: list[Advantaged], learning_rate: float, normalizer: float) -> dict[str, float]: ...
    async def save(self, name: str) -> str: ...


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


async def run(cfg: RunConfig, backend: Backend) -> list[dict[str, float]]:
    """Train one arm; returns per-step metrics (also logged in the shared telemetry grammar)."""
    arm = ARMS[cfg.arm]
    pool = load_post_prompts() if arm.prompts == "post" else prompt_set(cfg.game, arm.framing, "train")
    run_dir = Path(cfg.run_dir) if cfg.run_dir else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    log("run", **{k: v for k, v in asdict(cfg).items() if k != "chat_template_kwargs"}, scheme=arm.scheme, rho=arm.rho)
    history = []
    for step in range(cfg.start_step, cfg.steps):
        # Per-step randomness: a resumed run draws exactly the batches an unbroken run would.
        rng = random.Random(f"{cfg.seed}:{step}")
        meter = getattr(backend, "meter", None)
        if meter is not None and meter.exhausted:
            log("stop", step=step, reason="spend-cap", est_usd=meter.usd)
            break
        await backend.refresh_sampler()
        if arm.task == "control":
            batch = [make_prompt(rng) for _ in range(cfg.batch_prompts)]
        elif cfg.batch_prompts <= len(pool):
            # Without replacement: with the post's 32 prompts and batch 32, every environment
            # is trained once per step, as in the post's Appendix A.
            batch = rng.sample(pool, cfg.batch_prompts)
        else:
            batch = list(pool) + [rng.choice(pool) for _ in range(cfg.batch_prompts - len(pool))]
        # Stranger prompts say the other choice was fixed before the message: draw it first.
        stranger = [float(rng.random() < arm.q) for _ in batch]
        encoded = [backend.encode(p.text) for p in batch]
        groups = await asyncio.gather(
            *(backend.sample(toks, cfg.group_size, cfg.max_tokens, cfg.temperature) for toks in encoded)
        )
        train_batch: list[Advantaged] = []
        logged: list[dict] = []
        coop, invalid, truncated, rewards, lengths = [], 0, 0, [], []
        coop_by_category: dict[str, list[int]] = {}
        for prompt, toks, group, drawn in zip(batch, encoded, groups, stranger, strict=True):
            texts = [backend.decode(s.tokens) for s in group]
            if arm.task == "control":
                group_rewards = np.array([control_reward(t, prompt.answer) for t in texts])
                advantages = dr_grpo_advantages(group_rewards)
                coop += list(group_rewards)  # reported as "cooperation" column = accuracy for N
            else:
                actions = [prompt.action(t) for t in texts]
                q = drawn if arm.scheme.startswith("stranger") else arm.q
                credit = credit_group(prompt.game, actions, arm.scheme, rho=arm.rho, q=q, scale=arm.scale)
                coop_by_category.setdefault(prompt.category, []).extend(a for a in actions if a is not None)
                group_rewards, advantages = credit.rewards, credit.advantages
                coop += [a for a in actions if a is not None]
                invalid += sum(a is None for a in actions)
            truncated += sum(s.truncated for s in group)
            rewards += list(group_rewards)
            lengths += [len(s.tokens) for s in group]
            train_batch += [
                Advantaged(toks, s, float(a)) for s, a in zip(group, advantages, strict=True) if a != 0.0
            ]
            if run_dir is not None and cfg.log_samples:
                for j in range(min(cfg.log_samples, len(group))):
                    logged.append({
                        "step": step + 1, "scenario": getattr(prompt, "scenario", None),
                        "category": getattr(prompt, "category", None),
                        "action": None if arm.task == "control" else actions[j],
                        "reward": float(group_rewards[j]), "advantage": float(advantages[j]),
                        "tokens": len(group[j].tokens), "text": texts[j],
                    })
        metrics = await backend.train_step(train_batch, cfg.learning_rate, normalizer=float(cfg.max_tokens))
        n = cfg.batch_prompts * cfg.group_size
        row = {
            "cooperation": float(np.mean(coop)) if coop else float("nan"),
            "invalid": invalid / n,
            "truncated": truncated / n,
            "reward": float(np.mean(rewards)),
            "tokens": float(np.mean(lengths)),
            "trained": float(len(train_batch)),
            **{
                "coop_" + re.sub(r"\W", "_", cat): float(np.mean(v))
                for cat, v in sorted(coop_by_category.items()) if v
            },
            **metrics,
        }
        history.append(row)
        log("step", step=step_address(step + 1, cfg.steps), **row)
        if run_dir is not None:
            _append_jsonl(run_dir / "metrics.jsonl", [{"step": step + 1, **row}])
            _append_jsonl(run_dir / "rollouts.jsonl", logged)
        if cfg.save_every and ((step + 1) % cfg.save_every == 0 or step + 1 == cfg.steps):
            name = f"{cfg.arm}-s{cfg.seed}-step{step + 1:04d}"
            log("ckpt", step=step + 1, path=await backend.save(name))
            if hasattr(backend, "save_state"):
                log("state", step=step + 1, path=await backend.save_state(name))
    return history


class TinkerBackend:
    """``Backend`` on the bare Tinker SDK.

    Loss: Tinker's built-in ``importance_sampling``, documented as
    ``-(exp(target_logprobs - sampling_logprobs) * advantages).sum()`` over tokens
    (tinker-docs, /tinker/losses/, read 2026-10-19). Per-token advantages are
    ``A / normalizer`` on completion tokens and zero on the prompt, so the update is
    Dr. GRPO's constant-normalized token sum exactly.

    Auth: ``TINKER_API_KEY`` or ``~/.tinker/credentials.json`` (``tinker auth login``).
    """

    def __init__(self, cfg: RunConfig):
        import tinker

        self._types = tinker.types
        self.service = tinker.ServiceClient()
        if cfg.resume_from:
            self.training = self.service.create_training_client_from_state_with_optimizer(cfg.resume_from)
        else:
            self.training = self.service.create_lora_training_client(
                base_model=cfg.base_model, rank=cfg.rank, seed=cfg.seed
            )
        self.tokenizer = self.training.get_tokenizer()
        self.template_kwargs = cfg.chat_template_kwargs
        self.think_open = opens_thinking(self.tokenizer, self.template_kwargs)
        self.stop = [self.tokenizer.eos_token_id]
        self.sampler = None
        self.meter = SpendMeter(cfg.base_model, cap_usd=cfg.max_usd)

    def encode(self, user_text: str) -> list[int]:
        return render_prompt(self.tokenizer, user_text, self.template_kwargs)

    def decode(self, tokens: Sequence[int]) -> str:
        raw = self.tokenizer.decode(list(tokens), skip_special_tokens=False)
        return normalize_completion(raw, think_open=self.think_open)

    async def refresh_sampler(self) -> None:
        self.sampler = await self.training.save_weights_and_get_sampling_client_async()

    async def sample(self, prompt_tokens, n, max_tokens, temperature):
        t = self._types
        resp = await self.sampler.sample_async(
            prompt=t.ModelInput.from_ints(prompt_tokens),
            num_samples=n,
            sampling_params=t.SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=self.stop),
        )
        self.meter.add(prefill=len(prompt_tokens), sample=sum(len(s.tokens) for s in resp.sequences))
        return [
            Sample(list(s.tokens), list(s.logprobs or []), str(s.stop_reason).endswith("length"))
            for s in resp.sequences
        ]

    def to_datum(self, item: Advantaged, normalizer: float):
        t = self._types
        full = item.prompt_tokens + item.sample.tokens
        n_prompt = len(item.prompt_tokens)
        n_out = len(item.sample.tokens)
        # Position i predicts full[i + 1]; completion targets start at index n_prompt - 1.
        advantages = [0.0] * (n_prompt - 1) + [item.advantage / normalizer] * n_out
        logprobs = [0.0] * (n_prompt - 1) + list(item.sample.logprobs)
        return t.Datum(
            model_input=t.ModelInput.from_ints(full[:-1]),
            loss_fn_inputs={"target_tokens": full[1:], "logprobs": logprobs, "advantages": advantages},
        )

    async def train_step(self, batch, learning_rate, normalizer):
        if not batch:
            return {"loss": 0.0}
        t = self._types
        datums = [self.to_datum(item, normalizer) for item in batch]
        self.meter.add(train=sum(len(item.prompt_tokens) + len(item.sample.tokens) for item in batch))
        fb = await (await self.training.forward_backward_async(datums, "importance_sampling"))
        await (await self.training.optim_step_async(t.AdamParams(learning_rate=learning_rate)))
        return {**{k: float(v) for k, v in fb.metrics.items()}, "est_usd": self.meter.usd}

    async def save(self, name: str) -> str:
        resp = await (await self.training.save_weights_for_sampler_async(name=name))
        return resp.path

    async def save_state(self, name: str) -> str:
        """Weights plus optimizer state, for ``RunConfig.resume_from``."""
        resp = await (await self.training.save_state_async(name=f"{name}-state"))
        return resp.path
