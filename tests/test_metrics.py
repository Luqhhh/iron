import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.metrics import score_predictions


def test_hand_calculated_score_is_90_and_id_order_is_irrelevant():
    actual = pd.DataFrame(
        {"sample_id": ["01", "02"], "tap_iron": [100, 200], "tap_time_len": [50, 100]}
    )
    predicted = pd.DataFrame(
        {"sample_id": ["02", "01"], "pred_tap_iron": [180, 110], "pred_tap_time_len": [90, 55]}
    )
    result = score_predictions(actual, predicted)
    assert result["iron"]["wmape"] == pytest.approx(0.1)
    assert result["time"]["wmape"] == pytest.approx(0.1)
    assert result["iron"]["signed_error_sum"] == pytest.approx(-10.0)
    assert result["iron"]["signed_bias"] == pytest.approx(-5.0)
    assert result["score"] == pytest.approx(90.0)


def test_perfect_and_zero_floor():
    actual = pd.DataFrame({"sample_id": ["x"], "tap_iron": [1], "tap_time_len": [1]})
    perfect = pd.DataFrame({"sample_id": ["x"], "pred_tap_iron": [1], "pred_tap_time_len": [1]})
    terrible = pd.DataFrame({"sample_id": ["x"], "pred_tap_iron": [100], "pred_tap_time_len": [100]})
    assert score_predictions(actual, perfect)["score"] == 100
    assert score_predictions(actual, terrible)["score"] == 0


def test_bad_ids_and_nonpositive_denominator_fail():
    actual = pd.DataFrame({"sample_id": ["x"], "tap_iron": [0], "tap_time_len": [1]})
    wrong = pd.DataFrame({"sample_id": ["y"], "pred_tap_iron": [0], "pred_tap_time_len": [1]})
    with pytest.raises(ContractError, match="sample_id"):
        score_predictions(actual, wrong)
    wrong["sample_id"] = "x"
    with pytest.raises(ContractError, match="denominator"):
        score_predictions(actual, wrong)
