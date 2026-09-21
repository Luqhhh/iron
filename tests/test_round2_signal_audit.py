import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from bf_tap_r2.signal_audit import constructions, permutation_family


def test_permutation_detects_planted_signal_and_is_deterministic():
    rng = np.random.default_rng(40)
    x = rng.normal(size=(200, 3))
    y = np.column_stack([x[:, 0], -x[:, 1]])
    report = permutation_family(x, y, ["a", "b", "c"], 99, 42)
    assert report == permutation_family(x, y, ["a", "b", "c"], 99, 42)
    signal = next(r for r in report["records"] if r["feature"] == "a" and r["target"] == "tap_iron" and r["method"] == "pearson")
    assert signal["correlation"] == pytest.approx(1)
    assert signal["family_adjusted_p"] == .01
    assert signal["three_scope_adjusted_p"] == .03


def test_constant_columns_and_nonfinite_handling():
    x = np.ones((30, 1))
    y = np.column_stack([np.arange(30), np.arange(30)[::-1]])
    result = permutation_family(x, y, ["constant"], 19, 1)
    assert all(r["correlation"] == 0 and r["family_adjusted_p"] == 1 for r in result["records"])
    x[0] = np.nan
    with pytest.raises(ValueError):
        permutation_family(x, y, ["bad"], 19, 1)


def test_constructions_ignore_labels_and_ids():
    frame = pd.DataFrame({"spout_no": [1, 2], "air_volume": [2., 3.], "oxygen": [4., 5.],
                          "cold_air_press": [9., 8.], "hot_air_press": [7., 6.],
                          "upper_press_diff": [3., 2.], "lower_press_diff": [1., 1.],
                          "total_press_diff": [4., 3.], "pig": [8., 9.],
                          "tap_iron": [999., 888.], "sample_id": ["a", "b"]})
    a = constructions(frame)
    assert len(a.columns) == 7
    assert a.air_volume_times_oxygen.tolist() == [8., 15.]
    assert a.spout_2_times_pig.tolist() == [0., 9.]
    pd.testing.assert_frame_equal(a, constructions(frame.drop(columns=["tap_iron", "sample_id"])))
