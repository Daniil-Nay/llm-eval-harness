"""Retrieval metrics over ranked lists of document ids.

All functions take `retrieved` (ranked, best first) and `relevant` (set-like of
gold ids) and return a float in [0, 1]. Per-query values are meant to be
aggregated by the caller together with a confidence interval (see stats.py) -
a bare mean hides more than it shows.
"""

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of relevant documents that made it into the top-k."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of the top-k that is relevant. Divides by k even when fewer results
    came back: an empty slot is a miss, not a free pass."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & set(relevant)) / k


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant hit; 0.0 when nothing relevant."""
    rel = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    DCG rewards relevant hits higher up the list; the ideal DCG puts all
    relevant documents first. With binary labels this is the honest middle
    ground between recall (position-blind) and MRR (first-hit-only).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, doc_id in enumerate(retrieved[:k], start=1)
              if doc_id in rel)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg
