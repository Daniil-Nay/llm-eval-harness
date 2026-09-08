"""Regression gate: evaluate a SUT on a golden set and fail loudly below threshold.

Exit code is the interface - wire `python -m evalkit.gate` into CI and a
retrieval regression turns a PR red instead of a dashboard slightly less green.

The threshold compares against the LOWER bootstrap bound, not the point
estimate: a gate that flaps on sampling noise gets ignored within a month,
which is worse than no gate.
"""

import argparse
import json
import sys
from pathlib import Path

from .datasets import load_golden, validate
from .metrics.retrieval import recall_at_k
from .stats import bootstrap_ci
from .sut import KeywordBaseline


def load_corpus(corpus_dir: str | Path) -> dict[str, str]:
    """Read every *.md / *.txt in a directory; doc id = file stem."""
    corpus = {}
    for path in sorted(Path(corpus_dir).iterdir()):
        if path.suffix in (".md", ".txt"):
            corpus[path.stem] = path.read_text(encoding="utf-8")
    if not corpus:
        raise SystemExit(f"no documents found in {corpus_dir}")
    return corpus


def run_gate(corpus_dir: str, golden_path: str, k: int, threshold: float) -> dict:
    corpus = load_corpus(corpus_dir)
    rows = load_golden(golden_path)

    hygiene = validate(rows, corpus)
    sut = KeywordBaseline(corpus)
    scores = [recall_at_k(sut.retrieve(row["query"], k),
                          set(row["relevant_ids"]), k) for row in rows]
    mean = sum(scores) / len(scores)
    lo, hi = bootstrap_ci(scores)
    passed = hygiene["ok"] and lo >= threshold
    return {
        "n": len(rows),
        "k": k,
        f"recall@{k}": round(mean, 4),
        "ci95": (round(lo, 4), round(hi, 4)),
        "threshold_on_lower_bound": threshold,
        "hygiene_ok": hygiene["ok"],
        "hygiene": {key: val for key, val in hygiene.items() if key not in ("skew",)},
        "passed": passed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="datasets/toy_corpus")
    ap.add_argument("--golden", default="datasets/toy_golden.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="minimum acceptable LOWER bound of the recall CI")
    args = ap.parse_args()
    result = run_gate(args.corpus, args.golden, args.k, args.threshold)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
