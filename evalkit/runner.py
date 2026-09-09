"""Run a system-under-test over a golden set and persist the run.

A saved run is a first-class artifact: per-example scores plus a manifest
that pins what was measured (golden hash, corpus hash, SUT name, k). Two runs
are only comparable when their manifests agree - diffing runs of different
golden sets produces a number that means nothing, and `diff_runs` refuses to
do it instead of quietly complying.
"""

import datetime as dt
import hashlib
import json
from pathlib import Path

from .metrics.retrieval import recall_at_k
from .stats import bootstrap_ci


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def golden_sha(rows: list[dict]) -> str:
    return _sha(json.dumps(rows, sort_keys=True, ensure_ascii=False))


def corpus_sha(corpus: dict[str, str]) -> str:
    return _sha(json.dumps(corpus, sort_keys=True, ensure_ascii=False))


def run_retrieval(sut, sut_name: str, corpus: dict[str, str],
                  rows: list[dict], k: int) -> dict:
    """Score every golden row with recall@k; return the full run record."""
    scores = {}
    for i, row in enumerate(rows):
        retrieved = sut.retrieve(row["query"], k)
        scores[f"q{i:02d}"] = recall_at_k(retrieved, set(row["relevant_ids"]), k)
    values = list(scores.values())
    lo, hi = bootstrap_ci(values)
    return {
        "manifest": {
            "sut": sut_name,
            "k": k,
            "n": len(rows),
            "golden_sha": golden_sha(rows),
            "corpus_sha": corpus_sha(corpus),
            "date": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "scores": scores,
        "summary": {
            f"recall@{k}": round(sum(values) / len(values), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        },
    }


def save_run(run: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifests_comparable(a: dict, b: dict) -> tuple[bool, str | None]:
    """Runs compare only when they measured the same thing the same way.
    The date and the SUT name MAY differ - that is what a diff is for."""
    for field in ("golden_sha", "corpus_sha", "k"):
        if a["manifest"][field] != b["manifest"][field]:
            return False, field
    return True, None


def diff_runs(a: dict, b: dict) -> dict:
    """Per-example diff of run b against run a (a = baseline)."""
    ok, reason = manifests_comparable(a, b)
    if not ok:
        return {"comparable": False, "reason": reason}
    improved, regressed, unchanged = [], [], []
    for qid in sorted(a["scores"]):
        delta = b["scores"][qid] - a["scores"][qid]
        (improved if delta > 0 else regressed if delta < 0 else unchanged).append(qid)
    mean_a = sum(a["scores"].values()) / len(a["scores"])
    mean_b = sum(b["scores"].values()) / len(b["scores"])
    return {
        "comparable": True,
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "delta": round(mean_b - mean_a, 4),
        "n_wins": len(improved),
        "n_losses": len(regressed),
    }
