from copy import deepcopy

import numpy as np
import pytest

from bf_tap_r2.v7_confirm import read_reference_fold, strongest


def test_confirmation_cannot_select_from_incomplete_pool_or_one_positive_seed():
    spec = {"split_seeds": [42, 3407], "candidates": {"tap_time_len": ["a", "b"]},
            "tie_preference_by_target": {"tap_time_len": ["a", "b"]}}
    a = {"target": "tap_time_len", "recipe": "a", "confirmation_eligible": True,
         "a35_nested": {"seed_gains": {"42": 0.01, "3407": 0.02}, "seed_summary": {"mean": 0.015}}}
    b = deepcopy(a)
    b["recipe"] = "b"
    b["a35_nested"]["seed_summary"]["mean"] = 0.1
    b["a35_nested"]["seed_gains"]["3407"] = -0.01
    summary = {"status": "development_complete", "rows": [a, b]}
    assert strongest(summary, spec)["recipe"] == "a"
    incomplete_seed = deepcopy(summary)
    del incomplete_seed["rows"][0]["a35_nested"]["seed_gains"]["3407"]
    with pytest.raises(ValueError, match="seed coverage"):
        strongest(incomplete_seed, spec)
    summary["rows"] = [a]
    with pytest.raises(ValueError, match="Incomplete"):
        strongest(summary, spec)


def test_reference_cache_checks_identity_mask_and_nan_coverage(tmp_path):
    path = tmp_path / "reference.npz"
    mask = np.array([True, False, True])
    base = np.array([1., np.nan, 3.])
    np.savez(path, identity=np.asarray("time-only"), mask=mask, base=base, candidate=base + 1)
    b, n = read_reference_fold(path, "time-only", mask)
    np.testing.assert_array_equal(b, [1, 3])
    np.testing.assert_array_equal(n, [2, 4])
    with pytest.raises(ValueError, match="identity"):
        read_reference_fold(path, "iron", mask)
    with pytest.raises(ValueError, match="mask"):
        read_reference_fold(path, "time-only", ~mask)
    np.savez(path, identity=np.asarray("time-only"), mask=mask, base=np.ones(3), candidate=base)
    with pytest.raises(ValueError, match="coverage"):
        read_reference_fold(path, "time-only", mask)
