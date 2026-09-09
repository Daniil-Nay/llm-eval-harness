import json
from pathlib import Path

from evalkit.datasets import load_golden
from evalkit.gate import load_corpus
from evalkit.judge.cache import CachedClient, call_key
from evalkit.runner import diff_runs, load_run, run_retrieval
from evalkit.sut import KeywordBaseline

ROOT = Path(__file__).parent.parent


def _toy():
    corpus = load_corpus(ROOT / "datasets" / "toy_corpus")
    rows = load_golden(ROOT / "datasets" / "toy_golden.jsonl")
    return corpus, rows


def test_run_manifest_pins_inputs():
    corpus, rows = _toy()
    run = run_retrieval(KeywordBaseline(corpus), "kw", corpus, rows, k=1)
    again = run_retrieval(KeywordBaseline(corpus), "kw", corpus, rows, k=1)
    assert run["manifest"]["golden_sha"] == again["manifest"]["golden_sha"]
    assert run["scores"] == again["scores"]
    assert run["manifest"]["n"] == len(rows)


def test_diff_refuses_different_golden():
    corpus, rows = _toy()
    a = run_retrieval(KeywordBaseline(corpus), "kw", corpus, rows, k=1)
    b = run_retrieval(KeywordBaseline(corpus), "kw", corpus, rows[:-1], k=1)
    diff = diff_runs(a, b)
    assert diff == {"comparable": False, "reason": "golden_sha"}


def test_uniform_weighting_is_a_real_regression():
    corpus, rows = _toy()
    idf = run_retrieval(KeywordBaseline(corpus), "idf", corpus, rows, k=1)
    uni = run_retrieval(KeywordBaseline(corpus, weighting="uniform"),
                        "uniform", corpus, rows, k=1)
    diff = diff_runs(idf, uni)
    assert diff["comparable"] is True
    assert diff["delta"] < 0
    assert diff["n_losses"] > diff["n_wins"]


def test_shipped_runs_match_their_generator():
    """runs/ is committed; regenerating must give the same scores."""
    corpus, rows = _toy()
    shipped = load_run(ROOT / "runs" / "baseline.json")
    fresh = run_retrieval(KeywordBaseline(corpus), "keyword-idf", corpus, rows, k=1)
    assert shipped["scores"] == fresh["scores"]
    assert shipped["summary"] == fresh["summary"]


class CountingClient:
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def chat(self, messages, max_tokens=600, temperature=0.0, **_kw):
        self.calls += 1
        return "reply " + messages[0]["content"][:10]


def test_cache_serves_repeats_from_disk(tmp_path):
    inner = CountingClient()
    client = CachedClient(inner, tmp_path / "cache.json")
    msg = [{"role": "user", "content": "same question"}]
    first = client.chat(msg)
    second = client.chat(msg)
    assert first == second
    assert inner.calls == 1
    assert (client.hits, client.misses) == (1, 1)

    # a fresh wrapper over the same file still avoids the network
    reopened = CachedClient(CountingClient(), tmp_path / "cache.json")
    assert reopened.chat(msg) == first
    assert reopened.hits == 1


def test_cache_key_ignores_nothing_relevant():
    msg = [{"role": "user", "content": "q"}]
    base = call_key("m", msg, 600, 0.0)
    assert call_key("m", msg, 600, 0.0) == base
    assert call_key("other", msg, 600, 0.0) != base
    assert call_key("m", msg, 5, 0.0) != base
    assert call_key("m", msg, 600, 0.7) != base
    assert call_key("m", [{"role": "user", "content": "q2"}], 600, 0.0) != base


def test_shipped_runs_diff_is_the_m04_shape():
    a = load_run(ROOT / "runs" / "baseline.json")
    b = load_run(ROOT / "runs" / "candidate.json")
    diff = diff_runs(a, b)
    assert diff["comparable"] is True
    payload = json.dumps(diff)
    for field in ("improved", "regressed", "unchanged", "delta", "n_wins", "n_losses"):
        assert field in payload
