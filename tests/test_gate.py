from pathlib import Path

from evalkit.gate import load_corpus, run_gate
from evalkit.sut import KeywordBaseline

ROOT = Path(__file__).parent.parent
CORPUS_DIR = ROOT / "datasets" / "toy_corpus"
GOLDEN = ROOT / "datasets" / "toy_golden.jsonl"


def test_baseline_prefers_selective_tokens():
    corpus = {
        "eviction": "eviction policy lru lfu fifo cache bound",
        "install": "install pip python cache bound",
    }
    sut = KeywordBaseline(corpus)
    # "cache bound" appears in both docs, "eviction" only in one, so it wins
    assert sut.retrieve("eviction cache bound", k=1) == ["eviction"]


def test_baseline_returns_at_most_k_and_skips_zero_scores():
    sut = KeywordBaseline({"a": "alpha beta", "b": "gamma delta"})
    assert sut.retrieve("alpha", k=5) == ["a"]
    assert sut.retrieve("nothing matches this", k=5) == []


def test_baseline_ties_break_deterministically():
    sut = KeywordBaseline({"b": "same words here", "a": "same words here"})
    assert sut.retrieve("same words", k=2) == ["a", "b"]


def test_gate_passes_on_shipped_toy_dataset():
    """End-to-end: corpus + golden + baseline + bootstrap + threshold.

    The default threshold (0.6 on the LOWER CI bound) is calibrated so the
    shipped dataset passes with the shipped baseline; a regression in any of
    the four moving parts shows up here.
    """
    result = run_gate(str(CORPUS_DIR), str(GOLDEN), k=5, threshold=0.6)
    assert result["hygiene_ok"] is True
    assert result["passed"] is True
    assert result["n"] >= 40
    assert result["recall@5"] >= result["ci95"][0] >= 0.6


def test_gate_fails_when_threshold_unreachable():
    result = run_gate(str(CORPUS_DIR), str(GOLDEN), k=5, threshold=0.999)
    assert result["passed"] is False


def test_load_corpus_reads_md_files():
    corpus = load_corpus(CORPUS_DIR)
    assert "eviction" in corpus and "ttl" in corpus
    assert len(corpus) >= 12
