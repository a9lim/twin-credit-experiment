"""Run the evaluation battery on Tinker or an OpenAI-compatible endpoint.

    python scripts/run_evals.py --backend tinker --model Qwen/Qwen3.6-35B-A3B --tag base-35b --dry-run
    python scripts/run_evals.py --backend tinker --model Qwen/Qwen3.6-35B-A3B --tag base-35b --max-usd 3
    python scripts/run_evals.py --backend tinker --model-path tinker://RUN/sampler_weights/NAME \\
        --model Qwen/Qwen3.6-35B-A3B --tag a1p-s0 --max-usd 3
    python scripts/run_evals.py --backend openai --base-url URL --model NAME --tag base

Safety (added after two parallel batteries overran a $10 balance on 2026-10-19):

- ``--max-usd`` is required for the Tinker backend. A token-count spend meter
  (``pricing.SpendMeter``) is checked before every request; at the cap nothing new is
  sent and the run summarizes what it has.
- Records stream to ``runs/evals/<tag>/records.jsonl`` as each request finishes, and a
  rerun with the same tag resumes, skipping requests already on disk. The summary is
  always computed from the whole file.

Tinker reads ``TINKER_API_KEY`` or ``~/.tinker/credentials.json``; the OpenAI backend
reads ``--api-key-env`` (default ``OPENAI_API_KEY``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

from twin_credit_experiment.chat import normalize_completion, opens_thinking, render_prompt, tinker_auth_available
from twin_credit_experiment.evals import build_battery, read_records, run_battery, summarize_battery
from twin_credit_experiment.pricing import SpendMeter


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


def tinker_sampler(
    model: str, model_path: str | None, meter: SpendMeter, *, max_tokens: int, temperature: float, template_kwargs: dict
):
    import tinker

    service = tinker.ServiceClient()
    client = (
        service.create_sampling_client(model_path=model_path)
        if model_path
        else service.create_sampling_client(base_model=model)
    )
    tokenizer = client.get_tokenizer()
    think_open = opens_thinking(tokenizer, template_kwargs)
    params = tinker.types.SamplingParams(
        max_tokens=max_tokens, temperature=temperature, stop=[tokenizer.eos_token_id]
    )

    async def sample(prompt: str, k: int) -> list[str]:
        tokens = render_prompt(tokenizer, prompt, template_kwargs)
        resp = await client.sample_async(
            prompt=tinker.types.ModelInput.from_ints(tokens), num_samples=k, sampling_params=params
        )
        meter.add(prefill=len(tokens), sample=sum(len(s.tokens) for s in resp.sequences))
        return [
            normalize_completion(tokenizer.decode(list(s.tokens), skip_special_tokens=False), think_open=think_open)
            for s in resp.sequences
        ]

    return sample


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("tinker", "openai"), default="tinker")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint (openai backend)")
    ap.add_argument("--model", required=True, help="base model id")
    ap.add_argument("--model-path", help="tinker:// sampler checkpoint (tinker backend)")
    ap.add_argument("--chat-template-kwarg", action="append", default=[], metavar="KEY=JSON")
    ap.add_argument("--tag", required=True, help="run directory name under runs/evals/")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--max-usd", type=float, help="spend cap (required for tinker)")
    ap.add_argument("--k", type=int, default=5, help="samples per attitude item and dilemma")
    ap.add_argument("--capability-k", type=int, default=1, help="samples per capability item (0 skips)")
    ap.add_argument("--favorite-k", type=int, default=100, help="samples of the favorite-theory question")
    ap.add_argument("--probe-k", type=int, default=0, help="samples of each of the post's yes/no probes")
    ap.add_argument("--dtbench-extra", action="append", default=[], metavar="QID:N",
                    help="extra samples of one DTBench item, e.g. 92.4ATT:1000")
    ap.add_argument("--post-protocol", action="store_true",
                    help="the post's sample sizes: favorite-theory 1000, yes/no probes 20, 92.4ATT +1000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=5,
                    help="max samples per request; smaller bounds the spend in flight past the cap")
    ap.add_argument("--keep-text", action="store_true", help="store completions in the records")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    extra = {qid: int(n) for qid, n in (kv.split(":") for kv in args.dtbench_extra)}
    if args.post_protocol:
        args.favorite_k, args.probe_k = 1000, 20
        extra.setdefault("92.4ATT", 1000)
    jobs = build_battery(
        k=args.k, capability_k=args.capability_k, favorite_k=args.favorite_k, probe_k=args.probe_k,
        dtbench_extra=extra, seed=args.seed,
    )
    counts = Counter(item.kind for item, _ in jobs)
    print(f"{len(jobs)} samples: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if args.dry_run:
        return 0

    out = Path("runs/evals") / args.tag
    records_path = out / "records.jsonl"
    done = {r["group_key"] for r in read_records(records_path) if "group_key" in r}
    if done:
        print(f"resuming: {len(done)} requests already on disk")

    meter = None
    if args.backend == "tinker":
        if not tinker_auth_available():
            print("set TINKER_API_KEY or run `tinker auth login`", file=sys.stderr)
            return 2
        if args.max_usd is None:
            print("--max-usd is required for the tinker backend", file=sys.stderr)
            return 2
        meter = SpendMeter(args.model, cap_usd=args.max_usd)
        template_kwargs = {}
        for kv in args.chat_template_kwarg:
            key, _, value = kv.partition("=")
            template_kwargs[key] = json.loads(value)
        sampler = tinker_sampler(
            args.model, args.model_path, meter, max_tokens=args.max_tokens, temperature=args.temperature,
            template_kwargs=template_kwargs,
        )
    else:
        api_key = os.environ.get(args.api_key_env)
        if not api_key or not args.base_url:
            print(f"set {args.api_key_env} and --base-url", file=sys.stderr)
            return 2
        sampler = openai_sampler(
            args.base_url, api_key, args.model, max_tokens=args.max_tokens, temperature=args.temperature
        )

    out.mkdir(parents=True, exist_ok=True)
    fh = records_path.open("a")

    def sink(records):
        for r in records:
            fh.write(json.dumps(r) + "\n")
        fh.flush()

    def progress(done_n: int, total: int) -> None:
        if done_n == total or done_n % max(1, total // 20) == 0:
            spent = f" | est_usd={meter.usd:.2f}" if meter else ""
            print(f"progress | requests={done_n}/{total}{spent}", flush=True)

    asyncio.run(
        run_battery(
            sampler, jobs, concurrency=args.concurrency, keep_text=args.keep_text, progress=progress,
            sink=sink, skip=done, should_stop=(lambda: meter.exhausted) if meter else None, chunk=args.chunk,
        )
    )
    fh.close()
    if meter and meter.exhausted:
        print(f"stopped at the spend cap: est_usd={meter.usd:.2f} >= {args.max_usd}", flush=True)
    records = read_records(records_path)
    summary = summarize_battery(records) | {
        "model": args.model, "model_path": args.model_path, "backend": args.backend, "k": args.k,
        "seed": args.seed, "samples_on_disk": len(records), "samples_planned": len(jobs),
        "est_usd_this_invocation": round(meter.usd, 2) if meter else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
