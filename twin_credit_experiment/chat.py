"""Chat-template rendering and completion normalization shared by every sampler.

Hybrid-thinking templates (Qwen3.5/3.6 on Tinker, 2026-10-19) end the generation prompt
with an opened ``<think>``, so completions start mid-reasoning and never contain the
opening tag. ``normalize_completion`` restores it and strips trailing stop strings, so
graders see one uniform shape: ``<think>…</think>answer``, or an unclosed ``<think>…``
when the budget ran out (which ``credit.visible_answer`` treats as no answer).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")


def render_prompt(tokenizer: Any, user_text: str, template_kwargs: dict[str, Any] | None = None) -> list[int]:
    """Token ids for one user turn plus the generation prompt.

    Renders to text first: ``apply_chat_template(tokenize=True)`` returns a
    ``BatchEncoding`` in current transformers, not a list of ids.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        tokenize=False,
        **(template_kwargs or {}),
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def opens_thinking(tokenizer: Any, template_kwargs: dict[str, Any] | None = None) -> bool:
    """Whether the generation prompt leaves a ``<think>`` block open."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False, **(template_kwargs or {})
    )
    tail = text.rstrip()
    return tail.endswith("<think>")


def normalize_completion(raw: str, *, think_open: bool) -> str:
    text = raw
    stripped = True
    while stripped:
        stripped = False
        for stop in STOP_STRINGS:
            if text.rstrip().endswith(stop):
                text = text.rstrip()[: -len(stop)]
                stripped = True
    return "<think>\n" + text if think_open else text


def tinker_auth_available() -> bool:
    """``TINKER_API_KEY`` or the credentials file ``tinker auth login`` writes."""
    return bool(os.environ.get("TINKER_API_KEY")) or Path("~/.tinker/credentials.json").expanduser().exists()
