import numpy as np
import pandas as pd

from bf_tap_r2.v2_leakage_audit import nearest, permute_targets, proxy_screen


def test_cross_duplicates_and_self_exclusion():
    x = np.array([[0., 0.], [1., 2.], [1., 2.]])
    d, _ = nearest(x, x, True)
    np.testing.assert_array_equal(d, [2., 0., 0.])
    d, indexes = nearest(np.array([[1., 2.], [3., 3.]]), x)
    np.testing.assert_array_equal(d, [0., 2.])
    assert indexes[0] in [1, 2]


def test_proxy_screen_catches_direct_and_affine_copies():
    y = np.arange(20, dtype=float)
    frame = pd.DataFrame({"target": y, "copy": y, "affine": 7*y + 3, "unrelated": np.sin(y)})
    result = {r["feature"]: r for r in proxy_screen(frame, ["copy", "affine", "unrelated"], "target")}
    assert result["copy"]["same_row_equal_fraction"] == 1
    assert result["affine"]["affine_exact_at_1e_6"]
    assert not result["unrelated"]["affine_exact_at_1e_6"]


def test_permutation_is_joint_within_spout_without_mutating_input():
    frame = pd.DataFrame({"spout_no": [1]*20+[2]*20, "tap_iron": np.arange(40), "tap_time_len": np.arange(40)+100})
    before = frame.copy(deep=True)
    values = permute_targets(frame, 42)
    pd.testing.assert_frame_equal(frame, before)
    assert not np.array_equal(values[:, 0], frame.tap_iron)
    np.testing.assert_array_equal(values[:, 1] - values[:, 0], np.full(40, 100))
    assert set(values[:20, 0]) == set(range(20))
    assert set(values[20:, 0]) == set(range(20, 40))
