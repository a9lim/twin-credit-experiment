"""Measure how long a base model reasons on training prompts, and how it answers them.

    python scripts/length_probe.py --model Qwen/Qwen3.6-35B-A3B --source post --k 6 --max-usd 1
    python scripts/length_probe.py --model Qwen/Qwen3.5-9B --source generated --prompts 3 --k 8 --max-usd 0.5

``--source post`` samples every one of the Kimi post's 32 environments (fetched into
``data/post-prompts.jsonl``) and reports, per environment category, token-length
quantiles, truncation, invalid answers and the base cooperation rate, which can be read
against the post's step-0 cooperation by category. ``--source generated`` samples this
experiment's own twin prompts. Completions are saved under ``runs/probes/``. A spend cap
(``--max-usd``, required) is checked before each request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from twin_credit_experiment.chat import normalize_completion, opens_thinking, render_prompt
from twin_credit_experiment.games import Game
from twin_credit_experiment.post_env import load_post_prompts
from twin_credit_experiment.pricing import SpendMeter
from twin_credit_experiment.prompts import prompt_set


def quantiles(xs) -> dict[str, int]:
    a = np.array(xs)
    return {f"p{q}": int(np.quantile(a, q / 100)) for q in (10, 50, 90, 100)}


async def main_async(args) -> int:
    import tinker

    meter = SpendMeter(args.model, cap_usd=args.max_usd)
    service = tinker.ServiceClient()
    client = service.create_sampling_client(base_model=args.model)
    tok = client.get_tokenizer()
    think_open = opens_thinking(tok)
    params = tinker.types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, stop=[tok.eos_token_id])
    if args.source == "post":
        prompts = load_post_prompts()
    else:
        game = Game("pd", R=1.0, S=-1.0, T=args.temptation, P=0.0)
        prompts = random.Random(args.seed).sample(prompt_set(game, "twin", "train"), args.prompts)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(p):
        async with sem:
            if meter.exhausted:
                return []
            tokens = render_prompt(tok, p.text)
            resp = await client.sample_async(
                prompt=tinker.types.ModelInput.from_ints(tokens), num_samples=args.k, sampling_params=params
            )
            meter.add(prefill=len(tokens), sample=sum(len(s.tokens) for s in resp.sequences))
        rows = []
        for s in resp.sequences:
            text = normalize_completion(tok.decode(list(s.tokens), skip_special_tokens=False), think_open=think_open)
            rows.append({
                "scenario": p.scenario, "category": p.category, "tokens": len(s.tokens),
                "truncated": len(s.tokens) >= args.max_tokens, "action": p.action(text), "text": text,
            })
        return rows

    rows = [r for group in await asyncio.gather(*(one(p) for p in prompts)) for r in group]
    out = Path("runs/probes") / f"{args.model.split('/')[-1]}-{args.source}-{args.max_tokens}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    report = {"model": args.model, "source": args.source, "cap_tokens": args.max_tokens, "n": len(rows),
              "est_usd": round(meter.usd, 3), "stopped_at_cap": meter.exhausted,
              "overall": {"tokens": quantiles([r["tokens"] for r in rows]),
                          "truncated": float(np.mean([r["truncated"] for r in rows]))}}
    for cat, rs in sorted(by_cat.items()):
        valid = [r["action"] for r in rs if r["action"] is not None]
        report[cat] = {
            "n": len(rs),
            "tokens": quantiles([r["tokens"] for r in rs]),
            "truncated": float(np.mean([r["truncated"] for r in rs])),
            "invalid": float(np.mean([r["action"] is None for r in rs])),
            "cooperate": float(np.mean(valid)) if valid else None,
        }
    report["saved"] = str(out)
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", choices=("post", "generated"), default="post")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--prompts", type=int, default=3, help="generated source only")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temptation", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-usd", type=float, required=True)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
