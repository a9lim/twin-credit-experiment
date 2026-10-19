"""Fetch the Kimi post's 32 training prompts (Appendix B) and its favorite-theory question.

These are the post authors' text, so they go to gitignored ``data/post-prompts.jsonl``
and ``data/post-eval-prompts.json`` rather than into git. Run from the experiment root:

    python scripts/fetch_post_prompts.py
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import requests

POST_ID = "hfNBEKaStASAYMLiu"
GRAPHQL = "https://www.lesswrong.com/graphql"
QUERY = (
    '{ post(input: {selector: {_id: "%s"}}) '
    "{ result { title postedAt contents { html } } } }" % POST_ID
)
EXPECTED = 32


class Paragraphs(HTMLParser):
    """Collect top-level block texts (p, hr) in document order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._buf: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self._buf = []
        elif tag == "hr":
            self.blocks.append("<hr>")
        elif tag == "br" and self._buf is not None:
            self._buf.append("\n")

    def handle_endtag(self, tag):
        if tag == "p" and self._buf is not None:
            self.blocks.append("".join(self._buf).strip())
            self._buf = None

    def handle_data(self, data):
        if self._buf is not None:
            self._buf.append(data)


def appendix_b(post_html: str) -> str:
    start = post_html.find("Appendix B: All 32 prompts")
    end = post_html.find("Appendix C:", start)
    if start < 0 or end < 0:
        raise SystemExit("Appendix B not found; the post layout changed")
    return post_html[start:end]


def split_prompts(section: str) -> list[dict]:
    parser = Paragraphs()
    parser.feed(section)
    prompts: list[dict] = []
    current: dict | None = None
    for block in parser.blocks:
        marker = re.fullmatch(r"Prompt (\d+):", block)
        if marker:
            current = {"id": int(marker.group(1)), "paragraphs": []}
            prompts.append(current)
        elif block == "<hr>":
            current = None
        elif current is not None and block:
            current["paragraphs"].append(html.unescape(block))
    return [
        {"id": p["id"], "source": f"lesswrong:{POST_ID}#appendix-b", "prompt": "\n\n".join(p["paragraphs"])}
        for p in prompts
    ]


FAVORITE_THEORY = re.compile(r'"(What do you think is the correct decision theory\?[^"]*)"')


def favorite_theory_prompt(post_html: str) -> str:
    """The post's open-ended question (footnote 9), unescaped and tag-stripped."""
    text = html.unescape(re.sub(r"<[^>]+>", "", post_html.replace("&lt;", "\x00").replace("&gt;", "\x01")))
    text = text.replace("\x00", "<").replace("\x01", ">")
    match = FAVORITE_THEORY.search(text)
    if not match:
        raise SystemExit("favorite-theory prompt not found; the post layout changed")
    return match.group(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/post-prompts.jsonl"))
    args = ap.parse_args()

    resp = requests.post(GRAPHQL, json={"query": QUERY}, timeout=60)
    resp.raise_for_status()
    result = resp.json()["data"]["post"]["result"]
    prompts = split_prompts(appendix_b(result["contents"]["html"]))
    if [p["id"] for p in prompts] != list(range(1, EXPECTED + 1)):
        print(f"expected prompts 1..{EXPECTED}, got {[p['id'] for p in prompts]}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for p in prompts:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(prompts)} prompts from {result['title']!r} to {args.out}")

    eval_out = args.out.with_name("post-eval-prompts.json")
    eval_out.write_text(json.dumps({"favorite_theory": favorite_theory_prompt(result["contents"]["html"])}, indent=2) + "\n")
    print(f"wrote the favorite-theory prompt to {eval_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
