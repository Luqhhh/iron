"""Detect time leakage, reversed decay, lost anchors and bundle corruption."""
from importlib import import_module
from importlib.util import find_spec
import json

import numpy as np
import pandas as pd
import pytest

from bf_tap.exceptions import ContractError

PREFIX = "bf_tap.models.optimization_v02"


def module(name):
    assert find_spec(f"{PREFIX}.{name}") is not None, f"{name} must exist"
    return import_module(f"{PREFIX}.{name}")


def samples():
    return pd.DataFrame({
        "sample_id": ["a", "b", "c", "d", "e"],
        "tap_no": [1, 2, 3, 4, 5],
        "spout_no": ["1", "1", "2", "1", "1"],
        "reference_time": pd.to_datetime([
            "2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05"
        ], utc=True),
        "label_available_at": pd.to_datetime([
            "2024-03-01 01:00", "2024-03-06 00:00", "2024-03-03 01:00",
            "2024-03-04 01:00", "2024-03-05 01:00"
        ], utc=True),
        "tap_iron": [100., 9000., 300., 140., 170.],
        "tap_time_len": [10., 900., 30., 14., 17.],
    })


def test_recency_ratio_normalization_and_future_refusal():
    m = module("m2_recency.model")
    times = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-31"], utc=True))
    weights = m.recency_weights(times, pd.Timestamp("2024-02-01", tz="UTC"), 30)
    assert weights.mean() == pytest.approx(1.)
    assert weights.iloc[1] / weights.iloc[0] == pytest.approx(2.)
    with pytest.raises(ContractError):
        m.recency_weights(times, pd.Timestamp("2024-01-15", tz="UTC"), 30)
    for bad in (0, -1, float("nan")):
        with pytest.raises(ContractError):
            m.recency_weights(times, pd.Timestamp("2024-02-01", tz="UTC"), bad)


def test_anchor_excludes_own_future_and_unavailable_labels():
    m = module("m3_residual.model")
    data = samples()
    anchor, audit = m.expanding_anchors(data, min_group_count=2)
    assert anchor.iloc[0].isna().all()
    # b is not available until March 6; c is another spout; d sees a+c.
    assert anchor["tap_iron"].iloc[1:].tolist() == [100., 100., 200., 120.]
    assert anchor["tap_time_len"].iloc[1:].tolist() == [10., 10., 20., 12.]
    changed = data.copy()
    changed.loc[1, ["tap_iron", "tap_time_len"]] = [80000., 8000.]
    changed.loc[4, ["tap_iron", "tap_time_len"]] = [70000., 7000.]
    pd.testing.assert_frame_equal(anchor, m.expanding_anchors(changed, min_group_count=2)[0])
    assert audit["visible_rows"].tolist() == [0, 1, 1, 2, 3]


@pytest.mark.parametrize("family", ["m2_recency.model", "m3_residual.model"])
def test_real_frozen_estimator_roundtrip_and_tamper_refusal(tmp_path, family):
    m = module(family)
    data = samples()
    # All sample labels available at fitting cutoff; preserve chronological order.
    X = pd.DataFrame({"spout_no": data["spout_no"], "feature": [0., 1., 2., 3., 4.]})
    cutoff = pd.Timestamp("2024-03-07", tz="UTC")
    if family.startswith("m2"):
        model = m.RecencyModel(half_life_days=30)
    else:
        model = m.ResidualModel(min_group_count=2)
    model.fit(data, X, cutoff)
    evaluation = data.assign(reference_time=cutoff + pd.Timedelta(days=1))
    before = model.predict(evaluation, X)
    assert np.isfinite(before.to_numpy()).all()
    assert (before.to_numpy() >= 0).all()
    bundle = tmp_path / "bundle"
    model.save(bundle, metadata={"purpose": "unit-test"})
    loaded = type(model).load(bundle)
    np.testing.assert_allclose(loaded.predict(evaluation, X), before, rtol=0, atol=1e-10)
    with pytest.raises(FileExistsError):
        model.save(bundle, metadata={})
    with pytest.raises(ContractError):
        loaded.predict(evaluation, X[["feature", "spout_no"]])
    model_file = bundle / "tap_iron.cbm"
    model_file.write_bytes(model_file.read_bytes() + b"corruption")
    with pytest.raises(ContractError):
        type(model).load(bundle)


def test_residual_warmup_and_fit_cutoff_leakage_rejected():
    m = module("m3_residual.model")
    data = samples()
    X = pd.DataFrame({"spout_no": data["spout_no"], "feature": range(5)})
    with pytest.raises(ContractError):
        m.ResidualModel().fit(data, X, pd.Timestamp("2024-03-04", tz="UTC"))
    with pytest.raises(ContractError):
        m.ResidualModel().fit(data.iloc[:1], X.iloc[:1], pd.Timestamp("2024-03-02", tz="UTC"))
