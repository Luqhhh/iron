from copy import deepcopy
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from bf_tap_r2.audit import digest
from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.models import SnapshotRegressor, inputs
from bf_tap_r2.v2_checkpoint import (choose_targetwise, load_spec, member_rule, predict_rules,
                                    prefix_predict, single_slot_ranking, verify_identity)

ROOT = Path(__file__).resolve().parents[1]


def metric(v):
    return {"wmape": v, "by_fold": {str(i): v for i in range(5)}, "by_spout": {"1": v, "2": v}}


def scores():
    spec = load_spec(ROOT)
    metrics = {t: {r: {str(s): metric(.04) for s in spec["split_seeds"]}
                   for r in [spec["reference_by_target"][t], *spec["candidates"][t]]} for t in TARGETS}
    return spec, metrics


def test_time_is_compared_with_current_B3_not_old_C2():
    spec, m = scores()
    for s in spec["split_seeds"]:
        # Would beat a hypothetical old C2=.05, but is worse than current B3=.04.
        m["tap_time_len"]["T1_B3_PREFIX1000"][str(s)] = metric(.045)
    decisions, selected = choose_targetwise(m, spec)
    assert not decisions["tap_time_len"]["T1_B3_PREFIX1000"]["eligible"]
    assert selected["tap_time_len"] == spec["reference_by_target"]["tap_time_len"]


def test_iron_tie_prefers_one_member_and_rejects_spout_degradation():
    spec, m = scores()
    for s in spec["split_seeds"]:
        m["tap_iron"]["I1_D4_IRON"][str(s)] = metric(.0390005)
        m["tap_iron"]["I2_C2_D4_IRON_EQUAL"][str(s)] = metric(.039)
    assert choose_targetwise(m, spec)[1]["tap_iron"] == "I1_D4_IRON"
    m["tap_iron"]["I1_D4_IRON"]["42"]["by_spout"]["2"] = .0411
    assert choose_targetwise(m, spec)[1]["tap_iron"] == "I2_C2_D4_IRON_EQUAL"


def test_single_slot_uses_worst_split_gain_then_no_fit_time_tie():
    spec, m = scores()
    m["tap_iron"]["I1_D4_IRON"] = {"42": metric(.02), "3407": metric(.0399)}
    m["tap_time_len"]["T1_B3_PREFIX1000"] = {"42": metric(.039), "3407": metric(.039)}
    selected = {"tap_iron": "I1_D4_IRON", "tap_time_len": "T1_B3_PREFIX1000"}
    assert single_slot_ranking(m, selected, spec)[0]["target"] == "tap_time_len"
    m["tap_iron"]["I1_D4_IRON"]["3407"] = metric(.0389995)
    assert single_slot_ranking(m, selected, spec)[0]["target"] == "tap_time_len"
    m["tap_iron"]["I1_D4_IRON"]["3407"] = metric(.038)
    assert single_slot_ranking(m, selected, spec)[0]["target"] == "tap_iron"


def test_changed_or_missing_identity_is_rejected(tmp_path):
    (tmp_path / "x").write_text("frozen")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"x": digest(tmp_path / "x")}}))
    verify_identity(tmp_path, tmp_path)
    (tmp_path / "x").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_identity(tmp_path, tmp_path)
    (tmp_path / "x").unlink()
    with pytest.raises(ValueError, match="Missing"):
        verify_identity(tmp_path, tmp_path)


def test_prefix_uses_wrapper_semantics_and_does_not_change_source(tmp_path):
    # Exactly one synthetic fit in this test module; zero real competition fits.
    rng = np.random.default_rng(2022)
    frame = pd.DataFrame({f: rng.normal(size=240) for f in FEATURES})
    frame["spout_no"] = np.tile([1, 2], 120)
    config = yaml.safe_load((ROOT / "configs/round2_v0_1/models.yaml").read_text())
    config["models"]["C2"].update(iterations=30, depth=2)
    model = SnapshotRegressor("C2", config).fit(frame.iloc[:120], 100 + 8*frame.air_volume.iloc[:120])
    model.target_scale_ = 2.5  # Ensure the prefix helper preserves target restoration.
    valid = frame.iloc[120:]
    path = tmp_path / "source.joblib"
    joblib.dump(model, path)
    before = path.read_bytes()
    np.testing.assert_array_equal(prefix_predict(model, valid, 0, 30), model.predict(valid))
    expected = model.estimator_.predict(inputs(valid, True), ntree_start=0, ntree_end=10) * 2.5
    rule = member_rule(tmp_path, path, 42, 30, 10)
    ensemble = {"members": [rule], "aggregation": "arithmetic_mean", "member_order": [42]}
    np.testing.assert_array_equal(predict_rules(tmp_path, ensemble, valid), expected)
    assert path.read_bytes() == before and model.estimator_.tree_count_ == 30
    bad = deepcopy(ensemble)
    bad["members"][0]["ntree_end"] = 31
    with pytest.raises(ValueError, match="prefix"):
        predict_rules(tmp_path, bad, valid)
    with pytest.raises(ValueError, match="prefix"):
        prefix_predict(model, valid, 1, 10)
