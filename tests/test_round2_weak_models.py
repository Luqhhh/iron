from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("sklearn")
from bf_tap_r2.data import FEATURES
from bf_tap_r2.weak_models import assess_candidate, q2_preprocessor, s25_predictions
from bf_tap_r2.anchor_diagnostics import stratified_signal


def spec():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/round2_v0_2/safe_anchor.yaml").read_text())


def test_q2_expansion_and_fold_only_scaling_without_regressor_fit():
    rng = np.random.default_rng(9)
    training = pd.DataFrame(rng.normal(size=(60, 21)), columns=FEATURES)
    training["spout_no"] = np.arange(60) % 2 + 1
    valid = training.iloc[:2].copy()
    valid.iloc[:, 0] = 99999.
    pre = q2_preprocessor(spec()["Q2"])
    transformed = pre.fit_transform(training)
    numeric = pre.named_transformers_["numeric"]
    assert numeric.named_steps["poly"].n_output_features_ == 252
    assert transformed.shape == (60, 254)
    np.testing.assert_allclose(numeric.named_steps["scale_input"].mean_, training[list(FEATURES)].mean())
    assert numeric.named_steps["scale_expanded"].n_samples_seen_ == 60
    pre.transform(valid)
    assert numeric.named_steps["scale_expanded"].n_samples_seen_ == 60


def test_fixed_s25_and_all_three_promotion_conditions():
    np.testing.assert_array_equal(s25_predictions([10., 20.], [14., 12.]), [11., 18.])
    def score(value):
        return {"wmape": value, "by_fold": {str(f): value for f in range(5)}, "by_spout": {"1": value, "2": value}}
    anchors = {42: score(.15), 3407: score(.15)}
    candidate = {42: score(.14), 3407: score(.14)}
    assert assess_candidate(candidate, anchors, spec()["promotion"])["eligible"]
    candidate[3407]["by_spout"]["2"] = .152
    assert not assess_candidate(candidate, anchors, spec()["promotion"])["eligible"]
    assert not assess_candidate(anchors, anchors, spec()["promotion"])["eligible"]


def test_permutation_conditions_on_spout_and_preserves_paired_targets():
    rng = np.random.default_rng(42)
    n = 80
    frame = pd.DataFrame(rng.normal(size=(n, 21)), columns=FEATURES)
    frame["spout_no"] = np.arange(n) % 2 + 1
    frame["tap_iron"] = frame.air_volume
    frame["tap_time_len"] = -frame.air_volume
    result = stratified_signal(frame, {"count": 19, "seed": 42})
    assert result["max_abs_within_spout"] == pytest.approx(1.)
    assert result["permutation_p_plus_one"] == .05
    assert len(result["records"]) == 252
