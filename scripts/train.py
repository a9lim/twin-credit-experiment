"""Train one arm on Tinker, or dry-run it with a cost estimate.

    python scripts/train.py --arm A1p --base-model Qwen/Qwen3.5-9B --seed 0 --dry-run
    python scripts/train.py --arm A1p --base-model Qwen/Qwen3.5-9B --seed 0 \\
        --chat-template-kwarg enable_thinking=true

Needs ``TINKER_API_KEY`` or ``tinker auth login`` for a real run. Telemetry goes to stdout in the workspace's
``event | k=v`` grammar; pipe it to ``logs/<arm>-s<seed>.log``. Model ids must match
Tinker's catalog (``ServiceClient().get_server_capabilities()``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from twin_credit_experiment.chat import tinker_auth_available
from twin_credit_experiment.prompts import prompt_set
from twin_credit_experiment.train import ARMS, RunConfig, TinkerBackend, run

# $/M tokens (prefill, sample, train) at standard context, from Tinker's models-and-pricing
# page as read 2026-10-19. Prices change; recheck before budgeting.
PRICES = {
    "Qwen3.5-9B": (0.30, 0.807, 1.46),
    "Qwen3.6-35B-A3B": (0.45, 1.335, 1.18),
    "Qwen3.8-27B": (0.70, 5.595, 4.10),
    "Kimi-K2.6": (2.205, 5.49, 4.84),
}


def estimate(cfg: RunConfig, prompt_tokens: int, completion_tokens: int) -> dict:
    key = next((k for k in PRICES if k.lower() in cfg.base_model.lower()), None)
    if key is None:
        return {"note": f"no price on file for {cfg.base_model}"}
    pre, samp, train = PRICES[key]
    n = cfg.batch_prompts * cfg.group_size
    per_step = (
        cfg.batch_prompts * prompt_tokens * pre
        + n * completion_tokens * samp
        + n * (prompt_tokens + completion_tokens) * train
    ) / 1e6
    return {
        "price_key": key,
        "assumed_tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
        "usd_per_step": round(per_step, 2),
        "usd_per_run": round(per_step * cfg.steps, 2),
        "note": "upper-ish estimate: trains every completion; zero-advantage groups are skipped in practice",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--temptation", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=70)
    ap.add_argument("--batch-prompts", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--chat-template-kwarg", action="append", default=[], metavar="KEY=JSON")
    ap.add_argument("--assume-prompt-tokens", type=int, default=450)
    ap.add_argument("--assume-completion-tokens", type=int, default=1500)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    template_kwargs = {}
    for kv in args.chat_template_kwarg:
        key, _, value = kv.partition("=")
        template_kwargs[key] = json.loads(value)
    cfg = RunConfig(
        arm=args.arm, base_model=args.base_model, temptation=args.temptation, rank=args.rank,
        learning_rate=args.learning_rate, batch_prompts=args.batch_prompts, group_size=args.group_size,
        max_tokens=args.max_tokens, steps=args.steps, seed=args.seed, save_every=args.save_every,
        chat_template_kwargs=template_kwargs,
    )
    arm = ARMS[cfg.arm]
    print(json.dumps({
        "config": cfg.__dict__ | {"game": cfg.game.__dict__},
        "arm": arm.__dict__,
        "train_prompts": len(prompt_set(cfg.game, arm.framing, "train")),
        "estimate": estimate(cfg, args.assume_prompt_tokens, args.assume_completion_tokens),
    }, indent=2, default=str))
    if args.dry_run:
        return 0
    if not tinker_auth_available():
        print("set TINKER_API_KEY or run `tinker auth login`", file=sys.stderr)
        return 2
    asyncio.run(run(cfg, TinkerBackend(cfg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
