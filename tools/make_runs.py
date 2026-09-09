"""Generate the shipped baseline/candidate run pair.

Baseline is the default idf-weighted KeywordBaseline; candidate is the same
retriever with weighting turned off - a real, deterministic regression, not a
staged one. Both runs share the same golden and corpus, so `diff_runs` accepts
them; k=1 because that is where the toy dataset actually has headroom
(recall@5 sits at 0.96 and nothing interesting can regress).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evalkit.datasets import load_golden  # noqa: E402
from evalkit.gate import load_corpus  # noqa: E402
from evalkit.runner import diff_runs, run_retrieval, save_run  # noqa: E402
from evalkit.sut import KeywordBaseline  # noqa: E402

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "runs"


def main():
    corpus = load_corpus(ROOT / "datasets" / "toy_corpus")
    rows = load_golden(ROOT / "datasets" / "toy_golden.jsonl")

    baseline = run_retrieval(KeywordBaseline(corpus), "keyword-idf", corpus, rows, k=1)
    candidate = run_retrieval(KeywordBaseline(corpus, weighting="uniform"),
                              "keyword-uniform", corpus, rows, k=1)
    # the run date would make regenerated files differ byte-for-byte; pin it
    for run in (baseline, candidate):
        run["manifest"]["date"] = "generated-by-tools/make_runs.py"

    save_run(baseline, RUNS / "baseline.json")
    save_run(candidate, RUNS / "candidate.json")

    diff = diff_runs(baseline, candidate)
    print("baseline :", baseline["summary"])
    print("candidate:", candidate["summary"])
    print("diff     :", json.dumps({k: v for k, v in diff.items()
                                    if k in ("delta", "n_wins", "n_losses")}))


if __name__ == "__main__":
    main()
