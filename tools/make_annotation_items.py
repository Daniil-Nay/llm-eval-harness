"""Extract annotation items from the position-bias run.

Each of the 20 question/answer pairs yields two items (the A and the B
answer), so one annotator grades 40 answers on the 0-3 scale defined in
datasets/annotations/RUBRIC.md. The companion file annotator_b.jsonl is
hand-authored by the repo author against that rubric; this script never
touches it.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SOURCE = ROOT / "experiments" / "position_bias" / "results" / "2026-09-08.json"
OUT = ROOT / "datasets" / "annotations" / "items.jsonl"


def main():
    records = json.loads(SOURCE.read_text(encoding="utf-8"))["records"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records, 1):
            for side in ("a", "b"):
                f.write(json.dumps(
                    {"id": f"q{i:02d}_{side}",
                     "question": rec["question"],
                     "answer": rec[side]},
                    ensure_ascii=False) + "\n")
    print(f"items: {2 * len(records)} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
