import pandas as pd
import pytest

from bf_tap.exceptions import ContractError
from bf_tap.optimization.refresh import paired_month


def frame():
    return pd.DataFrame({"sample_id": ["a", "b"],
                         "reference_time": ["2024-07-02T00:00:00+08:00"] * 2,
                         "tap_iron": [10., 30.], "tap_time_len": [20., 20.],
                         "pred_tap_iron": [12., 28.], "pred_tap_time_len": [22., 18.]})


def test_refresh_pairs_ids_and_recomputes_ratios():
    old = frame()
    new = frame().iloc[::-1].copy()
    new["pred_tap_iron"] = new.tap_iron
    new["pred_tap_time_len"] = new.tap_time_len
    result = paired_month(old, new, "2024-07")
    assert result["refresh_gain"] == pytest.approx(.1)
    assert result["older_tap_iron_abs_error_sum"] == 4
    assert result["newer_tap_time_len_denominator"] == 40


@pytest.mark.parametrize("defect", ["missing", "duplicate", "label", "time", "infinite", "zero"])
def test_refresh_rejects_invalid_pairs(defect):
    old, new = frame(), frame()
    if defect == "missing":
        new = new.iloc[:1]
    elif defect == "duplicate":
        new.loc[1, "sample_id"] = "a"
    elif defect == "label":
        new.loc[0, "tap_iron"] += 1
    elif defect == "time":
        new.loc[0, "reference_time"] = "2024-08-01T00:00:00+08:00"
    elif defect == "infinite":
        new.loc[0, "pred_tap_iron"] = float("inf")
    else:
        old["tap_iron"] = new["tap_iron"] = 0.
    with pytest.raises(ContractError):
        paired_month(old, new, "2024-07")


def test_refresh_rejects_protected_month():
    with pytest.raises(ContractError, match="restricted"):
        paired_month(frame(), frame(), "2024-11")
