"""Check diagnostic isolation and component replacement semantics."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bf_tap_r2.submission import validate_result


spec = importlib.util.spec_from_file_location(
    "diagnostic_release", Path(__file__).parents[1] / "scripts/local_platform_diagnostic_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def synthetic_parent(iron=10):
    ids = [f"synthetic-{i}" for i in range(322)]
    rows = [f"{s},{iron if i != 1 else 2*iron}.0000,{3 if i != 1 else 4}.0000000" for i, s in enumerate(ids)]
    return ("sample_id,pred_tap_iron,pred_tap_time_len\n"+"\n".join(rows)+"\n").encode(), ids


@pytest.mark.parametrize("target,other,expected", [
    ("tap_iron", "pred_tap_time_len", [10.0, 22.0]),
    ("tap_time_len", "pred_tap_iron", [3.0, 6.0]),
])
def test_replaces_component_and_preserves_other_target_text(target, other, expected):
    parent, ids = synthetic_parent()
    # Delta [0,4] contributes [0,2], rather than reblending the parent.
    old, new = np.full(322, 2.), np.full(322, 2.)
    old[1], new[1] = 8., 12.
    result = release.payload(parent, ids, target, old, new)
    rows = validate_result(result, ids)
    original = validate_result(parent, ids)
    assert [r[other] for r in rows] == [r[other] for r in original]
    np.testing.assert_array_equal([float(r["pred_"+target]) for r in rows[:2]], expected)


@pytest.mark.parametrize("defect", ["shape", "nonfinite"])
def test_refuses_misaligned_or_nonfinite_components(defect):
    parent, ids = synthetic_parent()
    old, new = np.ones(322), np.ones(322)
    if defect == "shape":
        old = old[:1]
    else:
        new[0] = np.nan
    with pytest.raises(ValueError, match="shape/value"):
        release.payload(parent, ids, "tap_iron", old, new)


def test_refuses_to_silently_change_postprocessing():
    parent, ids = synthetic_parent(iron=1)
    with pytest.raises(ValueError, match="Negative diagnostic replacement"):
        release.payload(parent, ids, "tap_iron", np.full(322, 10.), np.zeros(322))
