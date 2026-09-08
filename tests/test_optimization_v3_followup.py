import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bf_tap.artifacts import atomic_write_json, stable_digest, file_sha256
from bf_tap.optimization import v3_followup as followup
from bf_tap.optimization.calibration import inner_blocks
from bf_tap.exceptions import ContractError


def context(tmp_path):
    t = pd.date_range("2024-03-01", "2024-10-31", freq="4h", tz="Asia/Shanghai")
    labels = pd.DataFrame({"sample_id": [str(i) for i in range(len(t))], "reference_time": t,
                           "label_available_at": t + pd.Timedelta(hours=1), "tap_iron": np.arange(len(t)) % 20 + 400.,
                           "tap_time_len": np.arange(len(t)) % 10 + 100.})
    ctx = SimpleNamespace(labels=labels, destination=tmp_path, code={"synthetic": True}, inputs={}, records=[], calls=[])

    def X(samples, cutoff, increment=None):
        ctx.calls.append({"ids": list(samples.sample_id), "cutoff": cutoff, "increment": increment})
        return pd.DataFrame({"spout_no": "1", "x": np.arange(len(samples), dtype=float)}, index=samples.index)

    ctx.X = X
    ctx.record_fit = lambda **kw: ctx.records.append(kw)
    return ctx


class FakeModel:
    def __init__(self, *args):
        self.selected_iterations = 23

    def fit(self, X, y, **kwargs):
        self.level = float(y.median())
        self.selected_iterations = kwargs.get("iterations", 23)
        return self

    def predict(self, X):
        return np.repeat(self.level + 2, len(X))

    def save(self, directory, *, identity):
        directory.mkdir(parents=True, exist_ok=False)
        atomic_write_json(directory / "bundle.json", {"identity": identity, "selected_iterations": self.selected_iterations})


def config():
    return {"id": "synthetic", "model_type": "tunable_catboost_l1", "parameters": {}, "patience": 150}


@pytest.mark.parametrize("increment,candidate", [("F-B", "CB-FB-raw"), ("F-C", "CB-FC-raw")])
def test_followup_cross_and_calibration_freeze_distinct_history_blocks(tmp_path, monkeypatch, increment, candidate):
    monkeypatch.setattr(followup, "SingleTargetModel", FakeModel)
    ctx = context(tmp_path)
    cutoff = pd.Timestamp("2024-09-01", tz="Asia/Shanghai")
    model, iterations, source = followup.fit_cross_target(ctx, config(), "tap_time_len", cutoff, "O",
                                                        increment=increment, candidate=candidate)
    assert iterations == 23
    assert [c["cutoff"] for c in ctx.calls] == [cutoff - pd.Timedelta(days=56)] * 2 + [cutoff]
    assert all(c["increment"] == increment for c in ctx.calls)
    assert all(r["candidate"] == candidate and r["identity"]["feature_increment"] == increment for r in ctx.records)
    ctx.calls.clear()
    ids = list(ctx.labels.loc[ctx.labels.reference_time >= cutoff, "sample_id"])
    prov = followup.fit_calibration(ctx, config(), iterations, cutoff, "O", "C", ids,
                                   increment=increment, selection_source=source)
    assert prov["selected_iterations"] == iterations
    assert prov["sample_count"] >= 100
    assert all(c["cutoff"] == cutoff - pd.Timedelta(days=28) for c in ctx.calls)
    assert not set(ctx.calls[0]["ids"]) & set(ctx.calls[1]["ids"])
    assert not set(ids) & set(ctx.calls[1]["ids"])


def test_future_outer_labels_do_not_change_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(followup, "SingleTargetModel", FakeModel)
    cutoff = pd.Timestamp("2024-09-01", tz="Asia/Shanghai")
    values = []
    for name, future_y in (("a", 1.), ("b", 99999.)):
        ctx = context(tmp_path / name)
        ctx.destination.mkdir()
        future = ctx.labels.reference_time >= cutoff
        ctx.labels.loc[future, "tap_time_len"] = future_y
        prov = followup.fit_calibration(ctx, config(), 23, cutoff, "O", "C", list(ctx.labels.loc[future, "sample_id"]), selection_source="synthetic")
        values.append(prov["median_prediction_minus_actual"])
    assert values[0] == values[1]


def test_zero_fallback_does_not_train_or_expand_window(tmp_path, monkeypatch):
    monkeypatch.setattr(followup, "SingleTargetModel", lambda *a: pytest.fail("must not fit under 100 samples"))
    ctx = context(tmp_path)
    ctx.labels = ctx.labels.iloc[::20].copy()
    cutoff = pd.Timestamp("2024-09-01", tz="Asia/Shanghai")
    prov = followup.fit_calibration(ctx, config(), 23, cutoff, "O", "C", ["outer"], selection_source="synthetic")
    assert prov["median_prediction_minus_actual"]["tap_time_len"] == 0
    assert 0 < prov["eligible_calibration_rows"] < 100
    assert not ctx.records


def test_registered_followup_has_one_cross_without_new_target_mix():
    cfg = followup.registration()
    assert len(cfg["candidates"]) == 4
    assert cfg["new_feature_crosses"] == 1
    assert cfg["additional_target_combinations"] == 0


def test_reused_iterations_require_exact_inner_samples_and_ledger(tmp_path):
    ctx = context(tmp_path)
    cutoff = pd.Timestamp("2024-09-01", tz="Asia/Shanghai")
    source = tmp_path / "source"
    bundle = source / "bundles" / "O" / "synthetic" / "tap_time_len"
    bundle.mkdir(parents=True)
    (bundle / "model.bin").write_bytes(b"synthetic estimator, never loaded")
    blocks = inner_blocks(ctx.labels, cutoff)
    metadata = {"model_type": "tunable_catboost_l1", "target": "tap_time_len", "parameters": {},
                "selected_iterations": 23, "model_sha256": file_sha256(bundle / "model.bin"),
                "identity": followup.model_identity(ctx, blocks["outer_train"], cutoff, blocks, None)}
    atomic_write_json(bundle / "bundle.json", metadata)
    atomic_write_json(bundle / "bundle_identity.json", {"metadata_sha256": stable_digest(metadata)})
    entry = {"candidate": "synthetic", "target": "tap_time_len", "origin": "O", "context": "inner_selection", "selected_iterations": 23}
    (source / "registry.jsonl").write_text(json.dumps(entry) + "\n")
    count, _ = followup.verified_iterations(ctx, config(), "tap_time_len", "O", cutoff, source, set())
    assert count == 23
    entry["selected_iterations"] = 24
    (source / "registry.jsonl").write_text(json.dumps(entry) + "\n")
    with pytest.raises(ContractError, match="ledger"):
        followup.verified_iterations(ctx, config(), "tap_time_len", "O", cutoff, source, set())
    metadata["identity"]["inner_split"]["selection_eval"]["samples_sha256"] = "wrong"
    atomic_write_json(bundle / "bundle.json", metadata, overwrite=True)
    atomic_write_json(bundle / "bundle_identity.json", {"metadata_sha256": stable_digest(metadata)}, overwrite=True)
    with pytest.raises(ContractError, match="samples differ"):
        followup.verified_iterations(ctx, config(), "tap_time_len", "O", cutoff, source, set())
