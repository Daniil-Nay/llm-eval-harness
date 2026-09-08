import math

import pytest

from evalkit.metrics.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_recall_counts_relevant_share():
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, k=3) == 0.5
    assert recall_at_k(["a", "z"], {"a", "z"}, k=2) == 1.0
    assert recall_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_recall_empty_relevant_is_zero_not_crash():
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_precision_divides_by_k_even_when_short():
    # one relevant hit in a top-5 that returned only two docs: 1/5, not 1/2
    assert precision_at_k(["a", "x"], {"a"}, k=5) == pytest.approx(0.2)


def test_mrr_first_hit_position():
    assert mrr(["x", "a", "y"], {"a"}) == pytest.approx(0.5)
    assert mrr(["a"], {"a"}) == 1.0
    assert mrr(["x"], {"a"}) == 0.0


def test_ndcg_prefers_hits_on_top():
    high = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
    low = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
    assert high == 1.0
    assert low == pytest.approx(1 / math.log2(4))
    assert high > low


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


@pytest.mark.parametrize("fn", [recall_at_k, precision_at_k, ndcg_at_k])
def test_nonpositive_k_rejected(fn):
    with pytest.raises(ValueError):
        fn(["a"], {"a"}, 0)
