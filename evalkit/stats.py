"""Statistics for people who ship metrics.

The recurring question this module answers: you measured 0.71 yesterday and
0.74 today. Is that an improvement or noise? On fifty examples it is usually
noise, and the paired bootstrap below is the cheapest honest way to tell.
"""

import math
import random


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Behaves sanely near 0 and 1,
    which is where the naive +-1.96*SE interval falls apart."""
    if trials <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= trials:
        raise ValueError("successes outside [0, trials]")
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(scores: list[float], n_resamples: int = 2000,
                 ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-example scores."""
    if len(scores) < 2:
        raise ValueError("need at least 2 scores")
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(sum(rng.choice(scores) for _ in range(n)) / n
                   for _ in range(n_resamples))
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = int((1 + ci) / 2 * n_resamples) - 1
    return (means[lo_idx], means[hi_idx])


def paired_bootstrap_diff(scores_a: list[float], scores_b: list[float],
                          n_resamples: int = 2000, ci: float = 0.95,
                          seed: int = 0) -> dict:
    """CI for mean(B) - mean(A) on PAIRED per-example scores.

    Pairing matters: systems are compared on the same examples, and resampling
    pairs keeps each example's difficulty attached to both scores. Unpaired
    comparison throws that structure away and needs far more data to say
    anything.

    Returns the observed diff, the CI, and whether the CI excludes zero.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("paired comparison needs equal-length score lists")
    if len(scores_a) < 2:
        raise ValueError("need at least 2 pairs")
    rng = random.Random(seed)
    diffs = [b - a for a, b in zip(scores_a, scores_b, strict=True)]
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n
                   for _ in range(n_resamples))
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = int((1 + ci) / 2 * n_resamples) - 1
    lo, hi = means[lo_idx], means[hi_idx]
    return {
        "diff": sum(diffs) / n,
        "ci": (lo, hi),
        "significant": lo > 0 or hi < 0,
    }
