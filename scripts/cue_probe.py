"""The cue-conflict probe across checkpoints: cooperation with no cue, the cooperation cue
and the dominance cue, for twins and strangers.

    python scripts/cue_probe.py      # writes data/summary/cue-probe.json, prints the table

Cued cells come from runs/evals/<tag>-cue (k = 5, 80 answers per cell); the uncued cell
is the same checkpoint's battery (k = 3, 48 answers), absent where it had none. Fisher
tests against base treat answers as independent, ignoring clustering by scenario
(4 scenarios x 2 temptations x 2 option orders).
"""

from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import fisher_exact

from twin_credit_experiment.evals import read_records

CHECKPOINTS = [
    ("base", "base-qwen3.6-35b-a3b"),
    ("R s70", "R-qwen35b-s0-step0070"),
    ("R-welfare s70", "R-welfare-qwen35b-s0-step0070"),
    ("R-rho100 s30", "R-rho100-qwen35b-s0-step0030"),
    ("R-rho100 s40", "R-rho100-qwen35b-s0-step0040"),
    ("R-rho100 s70", "R-rho100-qwen35b-s0-step0070"),
    ("R-rho100-honest s30", "R-rho100-honest-qwen35b-s0-step0030"),
    ("R-rho100-honest s35", "R-rho100-honest-qwen35b-s0-step0035"),
]
CELLS = ("none", "cooperate", "dominance")


def count(records: list[dict], framing: str) -> list[int]:
    valid = [r for r in records if r["framing"] == framing and r["action"] is not None]
    return [sum(r["action"] == 1 for r in valid), len(valid)]


def main() -> int:
    out = {}
    for name, tag in CHECKPOINTS:
        cue = read_records(Path("runs/evals") / f"{tag}-cue" / "records.jsonl")
        battery = [r for r in read_records(Path("runs/evals") / tag / "records.jsonl") if r["kind"] == "dilemma"]
        if not cue:
            continue
        out[name] = {
            framing: {
                "none": count(battery, framing) if battery else None,
                **{c: count([r for r in cue if r["cue"] == c], framing) for c in CELLS[1:]},
            }
            for framing in ("twin", "stranger")
        }
    base = out["base"]
    for name, row in out.items():
        for framing, cells in row.items():
            cells["fisher_p_vs_base"] = {
                c: fisher_exact([[a, n - a], [b, m - b]])[1]
                for c in CELLS
                if cells[c] is not None
                for (a, n), (b, m) in [(cells[c], base[framing][c])]
            }
    path = Path("data/summary/cue-probe.json")
    path.write_text(json.dumps({"note": __doc__.split("\n\n")[0], "checkpoints": out}, indent=1) + "\n")
    print(path)
    print("| Checkpoint | Twin: none / cooperate cue / dominance cue | Stranger: none / cooperate cue / dominance cue |")
    print("|---|---|---|")
    for name, row in out.items():
        cells = [" / ".join("–" if row[f][c] is None else "{}/{}".format(*row[f][c]) for c in CELLS) for f in ("twin", "stranger")]
        print(f"| {name} | {cells[0]} | {cells[1]} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
