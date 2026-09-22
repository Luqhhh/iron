from copy import deepcopy

import numpy as np
import pytest

from bf_tap_r2.data import TARGETS
from bf_tap_r2.v2_compare import choose, serialize, configuration
from bf_tap_r2.submission import validate_result
from pathlib import Path


def scores(value):
    return {"wmape": value, "by_fold": {str(i): value for i in range(5)}, "by_spout": {"1": value, "2": value}}


def test_targetwise_gate_rejects_single_seed_and_spout_regression():
    spec, _ = configuration(Path(__file__).resolve().parents[1])
    metrics = {r: {str(s): {t: scores(.2) for t in TARGETS} for s in spec["seeds"]} for r in spec["routes"]}
    for s in spec["seeds"]:
        metrics["L1"][str(s)][TARGETS[0]] = scores(.1)
        metrics["C1"][str(s)][TARGETS[1]] = scores(.1)
        metrics["L2"][str(s)][TARGETS[0]] = scores(.05)
        metrics["L2"][str(s)][TARGETS[0]]["by_spout"]["2"] = .21
    metrics["S1"]["42"][TARGETS[1]] = scores(.01)
    promotion, selected = choose(metrics, spec)
    assert selected == {TARGETS[0]: "L1", TARGETS[1]: "C1"}
    assert not promotion[TARGETS[0]]["L2"]["eligible"]
    assert not promotion[TARGETS[1]]["S1"]["eligible"]
    unchanged = {r: deepcopy(metrics["M0"]) for r in spec["routes"]}
    assert set(choose(unchanged, spec)[1].values()) == {"M0"}


@pytest.mark.parametrize("damage", ["extra_prediction", "duplicate", "old", "negative", "nan"])
def test_release_rejects_bad_pairs(damage):
    ids = [f"R2S2_TEST_{i:012X}" for i in range(322)]
    values = np.ones((322, 2))
    if damage == "extra_prediction":
        values = np.ones((323, 2))
    elif damage == "duplicate":
        ids[1] = ids[0]
    elif damage == "old":
        ids[0] = "R2S_TEST_000000000000"
    else:
        values[0, 0] = -1 if damage == "negative" else np.nan
    with pytest.raises(ValueError):
        serialize(ids, values)


def test_release_keeps_full_precision_and_sample_order():
    ids = [f"R2S2_TEST_{i:012X}" for i in reversed(range(322))]
    values = np.arange(644).reshape(322, 2) / 7
    rows = validate_result(serialize(ids, values), ids)
    assert [float(r["pred_tap_iron"]) for r in rows] == values[:, 0].tolist()
