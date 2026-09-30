import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("ema_time_followup", ROOT / "scripts/ema_time_followup.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def parent_payload(iron="20.000", time="10.000"):
    ids = [f"S{i:03d}" for i in range(322)]
    payload = ("sample_id,pred_tap_iron,pred_tap_time_len\n" +
               "".join(f"{sid},{iron},{time}\n" for sid in ids)).encode()
    return payload, ids


def test_extrapolation_keeps_full_original_parent():
    # q is a component replacement, not a fresh blend of the whole parent.
    result = module.replacement_time([10.], [8.], [12.], .75)
    np.testing.assert_array_equal(result, [13.])
    assert not np.array_equal(result, .25*np.array([10.])+.75*np.array([12.]))


@pytest.mark.parametrize("old,ema,q", [([1., 2.], [3.], .5), ([1.], [np.nan], .5),
                                      ([20.], [1.], 1.), ([1.], [2.], .6)])
def test_invalid_or_unregistered_probe_refused(old, ema, q):
    with pytest.raises(ValueError):
        module.replacement_time([10.], old, ema, q)


def test_time_probe_preserves_iron_text_and_roundtrips_float():
    from bf_tap_r2.submission import validate_result
    parent, ids = parent_payload()
    actual = module.time_payload(parent, ids, np.full(322, 8.), np.full(322, 12.), .75)
    rows = validate_result(actual, ids)
    assert all(r["pred_tap_iron"] == "20.000" for r in rows)
    assert all(float(r["pred_tap_time_len"]) == 13. for r in rows)


def test_combination_preserves_both_source_strings():
    from bf_tap_r2.submission import validate_result
    iron, ids = parent_payload(iron="21.000000000", time="11.111")
    time, _ = parent_payload(iron="22.222", time="9.876543210000")
    rows = validate_result(module.combined_payload(iron, time, ids), ids)
    assert all(r["pred_tap_iron"] == "21.000000000" for r in rows)
    assert all(r["pred_tap_time_len"] == "9.876543210000" for r in rows)


def test_combination_refuses_misaligned_ids():
    parent, ids = parent_payload()
    with pytest.raises(ValueError):
        module.combined_payload(parent, parent, ids[::-1])


def test_frozen_budget_and_affine_semantics():
    cfg = module.config()
    assert cfg["budget"]["new_fits"] == 0
    assert "negative V7" in cfg["weight_semantics"]
    assert [r["q"] for r in cfg["candidates"]] == [.25, .75, 1., .5]


def test_de3_exported_full_columns_are_not_weighted_twice():
    import pandas as pd
    # The raw component values could be 8/12/16; parent 10, raw old 8.
    # Export already contains 10+.5*(component-8): 10/12/14.
    selected = pd.DataFrame({"current_prediction": [10.], "seed_42_prediction": [10.],
        "seed_104729_prediction": [12.], "seed_130363_prediction": [14.]})
    np.testing.assert_array_equal(module.de3_full_column(selected, np.array([10.])), [12.])
    with pytest.raises(AssertionError):
        module.de3_full_column(selected, np.array([8.]))
