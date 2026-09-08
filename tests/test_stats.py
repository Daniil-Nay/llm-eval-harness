import pytest

from evalkit.stats import bootstrap_ci, paired_bootstrap_diff, wilson_interval


def test_wilson_matches_known_values():
    # the numbers from the position-bias experiment
    lo, hi = wilson_interval(4, 20)
    assert lo == pytest.approx(0.081, abs=0.005)
    assert hi == pytest.approx(0.416, abs=0.005)
    lo, hi = wilson_interval(20, 40)
    assert lo == pytest.approx(0.355, abs=0.005)
    assert hi == pytest.approx(0.645, abs=0.005)


def test_wilson_edges():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi > 0.0
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and lo < 1.0
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_bootstrap_is_deterministic_with_seed():
    scores = [0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.5, 0.7]
    assert bootstrap_ci(scores, seed=7) == bootstrap_ci(scores, seed=7)
    assert bootstrap_ci(scores, seed=7) != bootstrap_ci(scores, seed=8)


def test_bootstrap_ci_contains_mean():
    scores = [0.5] * 10 + [0.7] * 10
    lo, hi = bootstrap_ci(scores)
    mean = sum(scores) / len(scores)
    assert lo <= mean <= hi


def test_paired_diff_detects_a_real_gap():
    a = [0.5] * 30
    b = [0.8] * 30
    res = paired_bootstrap_diff(a, b)
    assert res["diff"] == pytest.approx(0.3)
    assert res["significant"] is True


def test_paired_diff_calls_noise_noise():
    # tiny gap, tiny sample, high variance: must NOT come out significant
    a = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    b = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    res = paired_bootstrap_diff(a, b, seed=1)
    assert res["significant"] is False


def test_paired_diff_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([0.1, 0.2], [0.3])
