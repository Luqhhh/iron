"""Check bounds against known score curves, including unsampled sharp peaks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "platform_line_bounds", Path(__file__).resolve().parents[1] /
    "scripts/round2_platform_line_bounds.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bound = module.concave_upper_bound


def test_affine_score_and_report_uncertainty():
    assert bound([(0, 90), (.5, 91), (1, 92)], score_error=0)["upper_bound"] == 92
    assert bound([(0, 90), (.5, 91), (1, 92)])["upper_bound"] >= 92


def test_two_endpoints_cannot_bound_an_interior_peak():
    assert np.isinf(bound([(0, 90), (1, 90)])["upper_bound"])
    # Beyond the second point a positive secant DOES give an upper bound.
    assert bound([(0, 90), (.5, 91)], (.5, 1), 0)["upper_bound"] == 92


def test_unsampled_sharp_peak_is_not_replaced_by_chord_interpolation():
    score = lambda x: 99 - 10 * abs(x - .55)
    points = [(a, score(a)) for a in [0, .4, .7, 1]]
    assert bound(points, score_error=0)["upper_bound"] == pytest.approx(99)
    assert max(s for _, s in points) < 99


def test_bounds_cover_affine_prediction_absolute_error_curves():
    rng = np.random.default_rng(926)
    grid = np.linspace(0, 1, 1001)
    for _ in range(40):
        y = rng.uniform(50, 100, 37)
        p = y + rng.normal(0, 10, 37)
        q = y + rng.normal(0, 10, 37)
        score = lambda a: 100 - 50 * np.abs((1-a)*p+a*q-y).sum()/y.sum()
        observed = [(a, round(score(a), 4)) for a in [.2, .4, .6, .8]]
        upper = bound(observed)["upper_bound"]
        assert max(map(score, grid)) <= upper + 1e-10


def test_alpha_feedback_and_v12_extrapolation_cannot_certify_target():
    points = [(.35, 96.3366), (.45, 96.3438), (.60, 96.3465), (.72, 96.3425)]
    assert 96.3465 <= bound(points)["upper_bound"] < 96.35
    iron = bound([(0, 96.3366), (.5, 96.3526)], (.5, 1))
    assert iron["upper_bound"] == pytest.approx(96.3689)
    assert iron["upper_bound"] + .0099 + .0002 < 96.4


@pytest.mark.parametrize("points,domain,error", [
    ([(0, 90)], (0, 1), 0),
    ([(.5, 90), (0, 91)], (0, 1), 0),
    ([(0, 90), (0, 91)], (0, 1), 0),
    ([(0, 90), (1, 91)], (1, 0), 0),
    ([(0, 90), (1, 91)], (0, 1), -1),
])
def test_invalid_contract(points, domain, error):
    with pytest.raises(ValueError):
        bound(points, domain, error)
