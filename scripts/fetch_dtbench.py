"""Fetch DTBench (Oesterheld et al. 2024) into gitignored ``data/dtbench/``.

The authors ship the questions in a password-protected zip so they stay out of
scraped training corpora; the password is published in their README. Keep the
extracted files out of git for the same reason. Run from the experiment root:

    python scripts/fetch_dtbench.py
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

REPO = "casparoe/newcomblike_questions_dataset"
REF = "main"
ZIP_PATH = "benchmark/data.zip"
PASSWORD = b"onebox"  # published in the repository README
URL = f"https://raw.githubusercontent.com/{REPO}/{REF}/{ZIP_PATH}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/dtbench"))
    args = ap.parse_args()

    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(resp.content))
    args.out.mkdir(parents=True, exist_ok=True)
    root = args.out.resolve()
    for member in archive.infolist():
        target = (args.out / member.filename).resolve()
        if root not in target.parents and target != root:
            print(f"refusing to extract outside {args.out}: {member.filename}", file=sys.stderr)
            return 1
    archive.extractall(args.out, pwd=PASSWORD)
    files = [m.filename for m in archive.infolist() if not m.is_dir()]
    (args.out / "SOURCE").write_text(f"{URL}\n")
    print(f"extracted {len(files)} files from {REPO}@{REF} to {args.out}")
    for name in sorted(files)[:20]:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
