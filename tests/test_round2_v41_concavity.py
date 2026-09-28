from fractions import Fraction as F

import pytest

from bf_tap_r2.v41_concavity_bound import continuous_bound, upper_at


def test_equal_values_allow_interior_peak():
    # f(x)=1-|x-.5|: equal endpoint scores, strict interior peak.
    design = {"left": ([0], .5, 0), "right": ([1], .5, 0)}
    assert upper_at(design, "left", [.5]) is None
    design["middle"] = ([.5], 1, 0)
    assert continuous_bound(design, [[0], [1]])["upper"] >= 1


def test_two_dimensional_bound_covers_concave_function_and_exact_duals():
    design = {"a": ([0, 0], 0, 0), "b": ([1, 0], 0, 0),
              "c": ([0, 1], 0, 0), "d": ([.25, .25], .25, 0)}
    report = continuous_bound(design, [[0, 0], [1, 0], [0, 1]])
    # f=min(x,y,1-x-y) has maximum 1/3.
    assert report["upper"] >= 1 / 3
    for query in ([0, 0], [.2, .4], [1, 0]):
        cert = upper_at(design, "d", query)
        assert cert["upper"] >= min(query[0], query[1], 1-sum(query))
        weights = {k: F(v) for k, v in cert["dual_weights"].items()}
        for j in range(2):
            assert sum((weights[k] * (F(str(design['d'][0][j])) - F(str(design[k][0][j])))
                        for k in weights), F(0)) == F(str(query[j])) - F(".25")


def test_uncertainty_enlargement_cannot_tighten_bound():
    exact = {"a": ([0], 0, 0), "b": ([.5], 1, 0), "c": ([1], 0, 0)}
    rounded = {k: (x, y, .001) for k, (x, y, _) in exact.items()}
    assert continuous_bound(rounded, [[0], [1]])["upper"] >= continuous_bound(exact, [[0], [1]])["upper"]


def test_inconsistent_concavity_is_rejected():
    design = {"a": ([0], 1, 0), "b": ([.5], 0, 0), "c": ([1], 1, 0)}
    with pytest.raises(ValueError, match="Inconsistent"):
        upper_at(design, "b", [.25])


def test_affine_score_is_bounded_exactly_from_interior_anchor():
    design = {"a": ([0], 3, 0), "b": ([.5], 4, 0), "c": ([1], 5, 0)}
    result = upper_at(design, "b", [.75])
    assert F(result["exact_upper"]) == F("4.5")
    assert continuous_bound(design, [[0], [1]])["upper"] == pytest.approx(5)
