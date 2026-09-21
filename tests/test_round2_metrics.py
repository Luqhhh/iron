import numpy as np
import pytest

from bf_tap_r2.metrics import score_targets, wmape


def test_wmape_is_not_mape_or_fold_mean():
    assert wmape([1, 100], [0, 100]) == pytest.approx(1 / 101)
    assert wmape([1, 100], [0, 100]) != .5


def test_combined_score_has_no_platform_mapping():
    assert score_targets([10, 20], [9, 18], [2, 3], [1, 3]) == pytest.approx(
        {"W_I": .1, "W_T": .2, "J": .15})


@pytest.mark.parametrize("y,p", [([], []), ([0], [1]), ([1], [np.inf]),
                                ([np.nan], [1]), ([1, 2], [1]), ([[1]], [[1]])])
def test_invalid_metric_inputs(y, p):
    with pytest.raises(ValueError):
        wmape(y, p)
