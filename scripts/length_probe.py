"""Measure how long a model reasons on the training prompts, to set the token budget.

    python scripts/length_probe.py --model Qwen/Qwen3.5-9B --max-tokens 16384 --prompts 3 --k 8

Samples base-model completions on train dilemmas (twin framing), then reports token
length quantiles, the truncation rate, and the parsed actions. Completions are saved to
``runs/probes/<model>-<cap>.jsonl`` for reading.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import numpy as np

from twin_credit_experiment.chat import normalize_completion, opens_thinking, render_prompt
from twin_credit_experiment.credit import parse_action
from twin_credit_experiment.games import Game
from twin_credit_experiment.prompts import prompt_set


async def main_async(args) -> None:
    import tinker

    service = tinker.ServiceClient()
    client = service.create_sampling_client(base_model=args.model)
    tok = client.get_tokenizer()
    think_open = opens_thinking(tok)
    params = tinker.types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, stop=[tok.eos_token_id])
    game = Game("pd", R=1.0, S=-1.0, T=args.temptation, P=0.0)
    prompts = random.Random(args.seed).sample(prompt_set(game, "twin", "train"), args.prompts)

    async def one(p):
        resp = await client.sample_async(
            prompt=tinker.types.ModelInput.from_ints(render_prompt(tok, p.text)), num_samples=args.k, sampling_params=params
        )
        rows = []
        for s in resp.sequences:
            text = normalize_completion(tok.decode(list(s.tokens), skip_special_tokens=False), think_open=think_open)
            rows.append({
                "scenario": p.scenario, "tokens": len(s.tokens), "stop": str(s.stop_reason),
                "action": parse_action(text, p.labels), "text": text,
            })
        return rows

    rows = [r for group in await asyncio.gather(*(one(p) for p in prompts)) for r in group]
    out = Path("runs/probes") / f"{args.model.split('/')[-1]}-{args.max_tokens}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lengths = np.array([r["tokens"] for r in rows])
    truncated = np.mean([r["tokens"] >= args.max_tokens for r in rows])
    actions = [r["action"] for r in rows]
    print(json.dumps({
        "model": args.model, "cap": args.max_tokens, "n": len(rows),
        "tokens_quantiles": {q: int(np.quantile(lengths, q / 100)) for q in (10, 25, 50, 75, 90, 100)},
        "truncated": float(truncated),
        "cooperate": sum(a == 1 for a in actions), "defect": sum(a == 0 for a in actions),
        "invalid": sum(a is None for a in actions),
        "saved": str(out),
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temptation", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
