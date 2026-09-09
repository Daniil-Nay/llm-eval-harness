"""Inter-annotator agreement.

The motivating trap: raw percent agreement looks great on imbalanced labels.
Two annotators who mark almost everything "not relevant" agree 88% of the time
while barely agreeing on the class that matters. Kappa subtracts the agreement
they would reach by chance alone.
"""

from collections import Counter


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's kappa for two annotators over the same items (any hashable labels)."""
    if len(labels_a) != len(labels_b):
        raise ValueError("annotators labeled different numbers of items")
    n = len(labels_a)
    if n == 0:
        raise ValueError("empty annotations")
    po = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    categories = set(ca) | set(cb)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in categories)
    if pe == 1.0:
        # both annotators used a single identical label everywhere;
        # agreement beyond chance is undefined, return 0.0 by convention
        return 0.0
    return (po - pe) / (1 - pe)


def weighted_kappa(labels_a: list[int], labels_b: list[int],
                   n_categories: int, weights: str = "linear") -> float:
    """Weighted kappa for ordinal labels 0..n_categories-1.

    Disagreeing by one grade should cost less than disagreeing by four;
    plain kappa treats both as equally wrong. `weights` is "linear" or
    "quadratic" (quadratic punishes large disagreements harder).
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("annotators labeled different numbers of items")
    n = len(labels_a)
    if n == 0:
        raise ValueError("empty annotations")
    k = n_categories
    for v in list(labels_a) + list(labels_b):
        if not (0 <= v < k):
            raise ValueError(f"label {v} outside 0..{k - 1}")

    def w(i: int, j: int) -> float:
        d = abs(i - j) / (k - 1) if k > 1 else 0.0
        return d if weights == "linear" else d * d

    # observed and expected disagreement, weighted
    ca, cb = Counter(labels_a), Counter(labels_b)
    observed = sum(w(a, b) for a, b in zip(labels_a, labels_b, strict=True)) / n
    expected = sum(w(i, j) * (ca[i] / n) * (cb[j] / n)
                   for i in range(k) for j in range(k))
    if expected == 0.0:
        return 0.0
    return 1 - observed / expected
