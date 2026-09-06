import pandas as pd

from bf_tap.models.sanity import MedianControls


def test_b0_and_b1_fixed_fallback_rule():
    train = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(5)],
            "spout_no": [1, 1, 1, 2, 2],
            "tap_iron": [10, 20, 30, 100, 200],
            "tap_time_len": [1, 2, 3, 10, 20],
        }
    )
    controls = MedianControls(min_group_count=3).fit(train)
    samples = pd.DataFrame({"sample_id": ["a", "b", "c"], "spout_no": [1, 2, 3]})
    b0 = controls.predict(samples, "B0")
    b1 = controls.predict(samples, "B1")
    assert b0.pred_tap_iron.tolist() == [30, 30, 30]
    assert b1.pred_tap_iron.tolist() == [20, 30, 30]
    assert b1.pred_tap_time_len.tolist() == [2, 3, 3]
