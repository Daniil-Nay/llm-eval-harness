import pytest

from evalkit.metrics.agreement import cohen_kappa, weighted_kappa


def _expand(a_yes_b_yes, a_yes_b_no, a_no_b_yes, a_no_b_no):
    """Build two label lists from a 2x2 confusion table."""
    a, b = [], []
    for label_a, label_b, count in (("yes", "yes", a_yes_b_yes),
                                    ("yes", "no", a_yes_b_no),
                                    ("no", "yes", a_no_b_yes),
                                    ("no", "no", a_no_b_no)):
        a += [label_a] * count
        b += [label_b] * count
    return a, b


def test_same_percent_agreement_different_kappa():
    # The motivating paradox: both tables agree on 88 of 100 items,
    # but chance agreement differs wildly on the imbalanced one.
    imbalanced = cohen_kappa(*_expand(6, 7, 5, 82))
    balanced = cohen_kappa(*_expand(44, 6, 6, 44))
    assert imbalanced == pytest.approx(0.4324, abs=1e-4)
    assert balanced == pytest.approx(0.76, abs=1e-10)


def test_perfect_and_chance_agreement():
    assert cohen_kappa(["y", "n", "y"], ["y", "n", "y"]) == 1.0
    # single identical label everywhere: undefined beyond chance -> 0.0
    assert cohen_kappa(["y", "y"], ["y", "y"]) == 0.0


def test_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohen_kappa(["y"], ["y", "n"])


def test_weighted_kappa_near_misses_cost_less():
    # annotator B is always exactly one grade below A on a 0..4 scale
    a = [4, 3, 2, 4, 3, 2, 4, 3]
    b = [3, 2, 1, 3, 2, 1, 3, 2]
    off_by_one = weighted_kappa(a, b, n_categories=5, weights="quadratic")
    # and a version where B is wildly off
    b_far = [0, 0, 0, 0, 0, 0, 0, 0]
    far_off = weighted_kappa(a, b_far, n_categories=5, weights="quadratic")
    assert off_by_one > far_off


def test_weighted_kappa_identity_is_one():
    labels = [0, 1, 2, 3, 4, 2, 1]
    assert weighted_kappa(labels, labels, n_categories=5) == pytest.approx(1.0)


def test_weighted_kappa_rejects_out_of_range():
    with pytest.raises(ValueError):
        weighted_kappa([0, 5], [0, 1], n_categories=5)
