from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import TARGETS
from bf_tap_r2.normalized_models import JointSnapshotRegressor
from bf_tap_r2.v2_candidate_release import predict_candidate, score_scenarios


def test_score_scenarios_use_the_actual_mixed_parent_and_half_target_gain():
    def scores(v):
        return {str(s): {"wmape": v} for s in (42, 3407)}
    old = {"metrics": {"tap_iron": {"C2_seed42_full1500": scores(.04), "I2_C2_D4_IRON_EQUAL": scores(.03)},
                       "tap_time_len": {"B3_seeds42_2026_2027_full1500": scores(.05), "T1_B3_PREFIX1000": scores(.048)}}}
    new = {"metrics": {"tap_iron": {"AJ": scores(.02)}}}
    result = score_scenarios(old, new, 96)
    assert result["V22_I_ONLY"]["oof_reference_score"] == pytest.approx(96)
    assert result["V22_I_ONLY"]["hypothetical_equal_gain_transfer_score"] == pytest.approx(96.5)
    assert result["V22_T_ONLY"]["hypothetical_equal_gain_transfer_score"] == pytest.approx(96.1)
    assert result["V23_AJ_I_ONLY"]["hypothetical_equal_gain_transfer_score"] == pytest.approx(97)
    assert all("not a forecast" in r["interpretation"] for r in result.values())


def test_t1_averages_only_fixed_prefix_members_and_rejects_rule_changes(monkeypatch):
    rules = {"kind": "T1", "member_order": [42, 2026, 2027], "aggregation": "arithmetic_mean",
             "members": [{"model_seed": s, "ntree_start": 0, "ntree_end": 1000, "source_tree_count": 1500}
                         for s in (42, 2026, 2027)]}
    monkeypatch.setattr("bf_tap_r2.v2_candidate_release.prefix_member", lambda root, rule, frame: np.full(len(frame), rule["model_seed"]))
    frame = pd.DataFrame({"x": [1, 2]})
    np.testing.assert_array_equal(predict_candidate(Path('.'), rules, frame), np.full(2, (42+2026+2027)/3))
    for key, value in (("ntree_end", 1500), ("model_seed", 0)):
        bad = deepcopy(rules)
        bad["members"][0][key] = value
        with pytest.raises(ValueError):
            predict_candidate(Path('.'), bad, frame)


def test_AJ_selects_iron_only_and_retains_full_C2(monkeypatch):
    model = JointSnapshotRegressor({"loss_function": "MultiRMSE"})
    model.estimator_ = SimpleNamespace(tree_count_=1500)
    model.actual_parameters_ = {"loss_function": "MultiRMSE"}
    model.target_scales_ = np.array([500., 120.])
    model.predict = lambda frame: np.tile([510., 999.], (len(frame), 1))
    rules = {"kind": "AJ", "weights": [.5, .5], "joint_target_index": 0, "target_order": list(TARGETS),
             "joint_model": "joint.joblib", "joint_model_sha256": "sha", "joint_parameters": model.parameters,
             "joint_actual_parameters": model.actual_parameters_, "joint_scales": [500., 120.],
             "c2_member": {"ntree_start": 0, "ntree_end": 1500, "source_tree_count": 1500}}
    monkeypatch.setattr("bf_tap_r2.v2_candidate_release.digest", lambda path: "sha")
    monkeypatch.setattr("bf_tap_r2.v2_candidate_release.joblib.load", lambda path: model)
    monkeypatch.setattr("bf_tap_r2.v2_candidate_release.prefix_member", lambda root, rule, frame: np.full(len(frame), 490.))
    frame = pd.DataFrame({"x": [1, 2]})
    np.testing.assert_array_equal(predict_candidate(Path('.'), rules, frame), [500., 500.])
    bad = deepcopy(rules)
    bad["joint_target_index"] = 1
    with pytest.raises(ValueError, match="iron rule"):
        predict_candidate(Path('.'), bad, frame)
    bad = deepcopy(rules)
    bad["c2_member"]["ntree_end"] = 1000
    with pytest.raises(ValueError, match="tree range"):
        predict_candidate(Path('.'), bad, frame)
