"""Chat rendering and completion normalization match what Tinker's Qwen templates do."""

import pytest

from twin_credit_experiment.chat import normalize_completion, opens_thinking, render_prompt
from twin_credit_experiment.credit import parse_action, visible_answer


class QwenLikeTokenizer:
    """Mimics the 2026-10-19 Qwen3.5/3.6 template: thinking pre-opened by default,
    and apply_chat_template(tokenize=True) returning a dict-like BatchEncoding."""

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking=True):
        text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        text += "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        if tokenize:
            return {"input_ids": [ord(c) for c in text], "attention_mask": [1] * len(text)}
        return text

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_render_prompt_returns_token_ids_not_batch_encoding_keys():
    tok = QwenLikeTokenizer()
    ids = render_prompt(tok, "Hi")
    assert all(isinstance(i, int) for i in ids)
    assert "".join(map(chr, ids)).endswith("assistant\n<think>\n")


def test_opens_thinking_follows_the_template_switch():
    tok = QwenLikeTokenizer()
    assert opens_thinking(tok)
    assert not opens_thinking(tok, {"enable_thinking": False})


@pytest.mark.parametrize(
    "raw, think_open, expected",
    [
        ("reasoning</think>\nACTION: LIST<|im_end|>", True, "<think>\nreasoning</think>\nACTION: LIST"),
        ("ACTION: LIST<|im_end|>\n", False, "ACTION: LIST"),
        ("ACTION: LIST<|im_end|><|endoftext|>", False, "ACTION: LIST"),
        ("cut off mid-thought", True, "<think>\ncut off mid-thought"),
    ],
)
def test_normalize_completion(raw, think_open, expected):
    assert normalize_completion(raw, think_open=think_open) == expected


def test_truncated_reasoning_has_no_visible_answer_even_if_it_mentions_an_action():
    labels = {"LIST": 1, "UNDERCUT": 0}
    truncated = normalize_completion("I could answer\nACTION: LIST\nbut wait", think_open=True)
    assert visible_answer(truncated) == ""
    assert parse_action(truncated, labels) is None
    finished = normalize_completion("thinking\nACTION: UNDERCUT?</think>\nACTION: LIST<|im_end|>", think_open=True)
    assert parse_action(finished, labels) == 1
