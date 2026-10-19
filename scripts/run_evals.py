"""Run the evaluation battery against an OpenAI-compatible endpoint.

    python scripts/run_evals.py --base-url URL --model NAME --tag base-qwen9b --k 5 --dry-run
    python scripts/run_evals.py --base-url URL --model NAME --tag base-qwen9b --k 5

The API key is read from ``--api-key-env`` (default ``OPENAI_API_KEY``). ``--dry-run``
builds the battery and prints its size and an estimated sample count without calling
anything. Records go to ``runs/evals/<tag>/records.jsonl``, the summary to
``summary.json`` beside them; both are gitignored run artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

from twin_credit_experiment.evals import build_battery, run_battery, summarize_battery, write_records


def openai_sampler(base_url: str, api_key: str, model: str, *, max_tokens: int, temperature: float):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def one(prompt: str) -> str:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)
        if reasoning is None:
            return content
        if choice.finish_reason == "length" and not content:
            return f"<think>{reasoning}"  # unterminated: graded as truncated reasoning
        return f"<think>{reasoning}</think>{content}"

    async def sample(prompt: str, k: int) -> list[str]:
        return list(await asyncio.gather(*(one(prompt) for _ in range(k))))

    return sample


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="run directory name under runs/evals/")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--k", type=int, default=5, help="samples per attitude item and dilemma")
    ap.add_argument("--capability-k", type=int, default=1, help="samples per capability item (0 skips)")
    ap.add_argument("--favorite-k", type=int, default=100, help="samples of the favorite-theory question")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--keep-text", action="store_true", help="store completions in the records")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = build_battery(k=args.k, capability_k=args.capability_k, favorite_k=args.favorite_k, seed=args.seed)
    counts = Counter(item.kind for item, _ in jobs)
    print(f"{len(jobs)} samples: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if args.dry_run:
        return 0

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"set {args.api_key_env}", file=sys.stderr)
        return 2
    sampler = openai_sampler(
        args.base_url, api_key, args.model, max_tokens=args.max_tokens, temperature=args.temperature
    )
    records = asyncio.run(run_battery(sampler, jobs, concurrency=args.concurrency, keep_text=args.keep_text))
    out = Path("runs/evals") / args.tag
    write_records(records, out / "records.jsonl")
    summary = summarize_battery(records) | {"model": args.model, "k": args.k, "seed": args.seed}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
