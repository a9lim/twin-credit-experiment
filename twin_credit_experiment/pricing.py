"""Tinker prices and a running spend meter, so every live script can stop at a cap.

Prices are $/M tokens (prefill, sample, train) at standard context from Tinker's
models-and-pricing page, read 2026-10-19; they change, so recheck before budgeting.
The meter is an estimate from token counts the client sees, not a bill: prefill is
charged once per request, sampled tokens per sample, train tokens per trained datum.
On 2026-10-19 two parallel baseline batteries overran a $10 balance with nothing saved;
every live entry point now takes ``--max-usd`` and checks this meter before each request.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICES = {
    "Qwen3.5-9B": (0.30, 0.807, 1.46),
    "Qwen3.6-35B-A3B": (0.45, 1.335, 1.18),
    "Qwen3.8-27B": (0.70, 5.595, 4.10),
    "Kimi-K2.6": (2.205, 5.49, 4.84),
}


def price_key(model: str) -> str | None:
    return next((k for k in PRICES if k.lower() in model.lower()), None)


@dataclass
class SpendMeter:
    model: str
    cap_usd: float | None = None
    prefill_tokens: int = 0
    sample_tokens: int = 0
    train_tokens: int = 0

    def __post_init__(self) -> None:
        key = price_key(self.model)
        if key is None:
            raise ValueError(f"no price on file for {self.model}; add it to pricing.PRICES")
        self.prices = PRICES[key]

    def add(self, *, prefill: int = 0, sample: int = 0, train: int = 0) -> None:
        self.prefill_tokens += prefill
        self.sample_tokens += sample
        self.train_tokens += train

    @property
    def usd(self) -> float:
        pre, samp, train = self.prices
        return (self.prefill_tokens * pre + self.sample_tokens * samp + self.train_tokens * train) / 1e6

    @property
    def exhausted(self) -> bool:
        return self.cap_usd is not None and self.usd >= self.cap_usd
