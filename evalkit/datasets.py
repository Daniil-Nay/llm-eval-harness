"""Golden-set loading and hygiene checks.

A golden set rots quietly: duplicates from logs, near-duplicates from
synthetic generation, test questions leaking into the indexed corpus, one
query type crowding out the rest. None of it crashes anything - it just bends
every number you compute afterwards. `validate` makes the rot visible.
"""

import json
import re
from collections import Counter
from pathlib import Path


def load_golden(path: str | Path) -> list[dict]:
    """Read a JSONL golden set: one {"query", "relevant_ids", ...} per line."""
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "query" not in row or "relevant_ids" not in row:
            raise ValueError(f"line {i}: missing 'query' or 'relevant_ids'")
        rows.append(row)
    return rows


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _shingles(text: str, size: int = 4) -> set[tuple]:
    words = normalize(text).split()
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}


def find_exact_duplicates(rows: list[dict]) -> list[list[int]]:
    """Groups of row indices whose normalized queries are identical."""
    seen: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        seen.setdefault(normalize(row["query"]), []).append(i)
    return [idx for idx in seen.values() if len(idx) > 1]


def find_near_duplicates(rows: list[dict], threshold: float = 0.8) -> list[tuple[int, int, float]]:
    """Pairs of row indices whose queries share most of their 4-word shingles
    (Jaccard >= threshold). O(n^2) - fine for golden-set sizes; switch to
    MinHash when you outgrow a few thousand rows."""
    shingle_sets = [_shingles(r["query"]) for r in rows]
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = shingle_sets[i], shingle_sets[j]
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if jac >= threshold and normalize(rows[i]["query"]) != normalize(rows[j]["query"]):
                pairs.append((i, j, round(jac, 3)))
    return pairs


def find_leakage(rows: list[dict], corpus: dict[str, str],
                 min_run: int = 6) -> list[tuple[int, str]]:
    """Golden queries whose text appears verbatim inside an indexed document.

    When a synthetic question leaks into the corpus, retrieval finds the chunk
    by literal string match, recall looks wonderful, and live users see none
    of it. A shared run of `min_run`+ consecutive words is treated as a leak.
    """
    leaks = []
    norm_docs = {doc_id: normalize(text) for doc_id, text in corpus.items()}
    for i, row in enumerate(rows):
        words = normalize(row["query"]).split()
        if len(words) < min_run:
            continue
        runs = {" ".join(words[k:k + min_run]) for k in range(len(words) - min_run + 1)}
        for doc_id, doc in norm_docs.items():
            if any(run in doc for run in runs):
                leaks.append((i, doc_id))
                break
    return leaks


def type_skew(rows: list[dict], key: str = "type") -> dict:
    """Distribution of query types plus the share of the dominant one.
    Rows without the tag are counted under "(untagged)" - untagged rows are
    how skew hides."""
    counts = Counter(row.get(key, "(untagged)") for row in rows)
    top = counts.most_common(1)[0]
    return {
        "counts": dict(counts),
        "dominant": top[0],
        "dominant_share": round(top[1] / len(rows), 3) if rows else 0.0,
    }


def validate(rows: list[dict], corpus: dict[str, str] | None = None) -> dict:
    """Full hygiene report. `ok` is True only when nothing hard fails
    (duplicates or leakage); skew is reported but does not fail the set."""
    report = {
        "n": len(rows),
        "exact_duplicates": find_exact_duplicates(rows),
        "near_duplicates": find_near_duplicates(rows),
        "leakage": find_leakage(rows, corpus) if corpus else [],
        "skew": type_skew(rows),
    }
    report["ok"] = not report["exact_duplicates"] and not report["leakage"]
    return report
