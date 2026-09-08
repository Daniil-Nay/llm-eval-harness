from pathlib import Path

from evalkit.datasets import (
    find_exact_duplicates,
    find_leakage,
    find_near_duplicates,
    load_golden,
    type_skew,
    validate,
)

ROOT = Path(__file__).parent.parent


def test_exact_duplicates_normalized():
    rows = [{"query": "How do I set a TTL?"},
            {"query": "how do i set a ttl"},
            {"query": "something else entirely"}]
    groups = find_exact_duplicates(rows)
    assert groups == [[0, 1]]


def test_near_duplicates_catch_paraphrase_padding():
    rows = [{"query": "how do i configure the eviction policy for the cache instance"},
            {"query": "how do i configure the eviction policy for the cache instance please"},
            {"query": "completely unrelated question about snapshots"}]
    pairs = find_near_duplicates(rows, threshold=0.5)
    assert [(i, j) for i, j, _score in pairs] == [(0, 1)]


def test_leakage_detects_verbatim_run():
    corpus = {"doc": "to sweep all expired entries eagerly call purge from a periodic job"}
    rows = [{"query": "should I sweep all expired entries eagerly call purge now"},
            {"query": "how does key hashing distribute load between shards"}]
    leaks = find_leakage(rows, corpus, min_run=6)
    assert leaks == [(0, "doc")]


def test_type_skew_counts_untagged():
    rows = [{"query": "a", "type": "fact"}, {"query": "b"}, {"query": "c", "type": "fact"}]
    skew = type_skew(rows)
    assert skew["counts"] == {"fact": 2, "(untagged)": 1}
    assert skew["dominant"] == "fact"


def test_shipped_toy_golden_is_clean():
    """The dataset this repo ships must pass its own hygiene bar."""
    rows = load_golden(ROOT / "datasets" / "toy_golden.jsonl")
    corpus = {p.stem: p.read_text(encoding="utf-8")
              for p in (ROOT / "datasets" / "toy_corpus").glob("*.md")}
    report = validate(rows, corpus)
    assert report["ok"], report
    assert report["n"] >= 40
    # every referenced doc must exist
    for row in rows:
        for doc_id in row["relevant_ids"]:
            assert doc_id in corpus, f"golden references missing doc {doc_id}"
