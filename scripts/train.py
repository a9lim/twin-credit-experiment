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
from twin_credit_experiment.post_env import load_post_prompts
from twin_credit_experiment.pricing import PRICES
from twin_credit_experiment.prompts import prompt_set
from twin_credit_experiment.train import ARMS, RunConfig, TinkerBackend, run



# Median completion tokens (thinking on) on the twin-dilemma train prompts, from
# scripts/length_probe.py at a 16k cap, 24 samples each, 2026-10-19.
MEASURED_COMPLETION = {"Qwen3.5-9B": 5850, "Qwen3.6-35B-A3B": 2253}


def estimate(cfg: RunConfig, prompt_tokens: int, completion_tokens: int | None) -> dict:
    key = next((k for k in PRICES if k.lower() in cfg.base_model.lower()), None)
    if key is None:
        return {"note": f"no price on file for {cfg.base_model}"}
    if completion_tokens is None:
        completion_tokens = MEASURED_COMPLETION.get(key, 1500)
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
    ap.add_argument("--assume-completion-tokens", type=int, default=None,
                    help="default: the measured median for the model, else 1500")
    ap.add_argument("--max-usd", type=float, help="spend cap, required for a live run")
    ap.add_argument("--tag", help="run directory runs/train/<tag> for metrics.jsonl and rollouts.jsonl")
    ap.add_argument("--log-samples", type=int, default=2, help="completions logged per prompt per step")
    ap.add_argument("--resume-from", help="tinker:// training state from a 'state' log line")
    ap.add_argument("--start-step", type=int, default=0, help="first step to run when resuming")
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
        max_usd=args.max_usd, chat_template_kwargs=template_kwargs, start_step=args.start_step,
        resume_from=args.resume_from, run_dir=f"runs/train/{args.tag}" if args.tag else None,
        log_samples=args.log_samples,
    )
    arm = ARMS[cfg.arm]
    print(json.dumps({
        "config": cfg.__dict__ | {"game": cfg.game.__dict__},
        "arm": arm.__dict__,
        "train_prompts": len(load_post_prompts()) if arm.prompts == "post" else len(prompt_set(cfg.game, arm.framing, "train")),
        "estimate": estimate(cfg, args.assume_prompt_tokens, args.assume_completion_tokens),
    }, indent=2, default=str))
    if args.dry_run:
        return 0
    if args.max_usd is None:
        print("--max-usd is required for a live run (see the estimate above)", file=sys.stderr)
        return 2
    if not tinker_auth_available():
        print("set TINKER_API_KEY or run `tinker auth login`", file=sys.stderr)
        return 2
    asyncio.run(run(cfg, TinkerBackend(cfg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
