"""Synthetic scheduler integration: actual files, complete folds, no real labels."""
import json
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd

from bf_tap.optimization.config import load_validation
from bf_tap.optimization.v3_run import DevelopmentContext, reference_run, screening_run
from bf_tap.artifacts import atomic_write_json


def test_feature_cache_binds_metadata_and_index_but_not_labels(monkeypatch):
    from bf_tap.optimization import v3_run
    ctx = DevelopmentContext.__new__(DevelopmentContext)
    ctx.cache = {}
    ctx.operation = ctx.burden = ctx.history = None
    ctx.features = {"operation": {"value_columns": []}}
    ctx.selection = {"process_change": {}}
    calls = []

    def builder(samples, **kwargs):
        calls.append(samples.copy())
        return SimpleNamespace(X=samples[["spout_no"]].copy())

    monkeypatch.setattr(v3_run, "build_features", builder)
    monkeypatch.setattr(v3_run, "add_process_change_features", lambda X, *a, **k: X)
    cutoff = pd.Timestamp("2024-07-01", tz="Asia/Shanghai")
    rows = pd.DataFrame({"sample_id": ["a"], "spout_no": [1], "reference_time": [cutoff], "tap_iron": [100.]})
    ctx.feature_frame(rows, cutoff)
    ctx.feature_frame(rows.assign(tap_iron=999.), cutoff)
    assert len(calls) == 1
    changed = ctx.feature_frame(rows.assign(spout_no=2), cutoff)
    assert changed.spout_no.iloc[0] == 2 and len(calls) == 2
    new_index = rows.copy()
    new_index.index = [9]
    assert list(ctx.feature_frame(new_index, cutoff).index) == [9]


def synthetic_context(tmp_path):
    ctx = DevelopmentContext.__new__(DevelopmentContext)
    ctx.destination = tmp_path
    ctx.validation, ctx.screening, ctx.origins = load_validation(
        "configs/optimization_v0_3/validation.yaml", expected_timezone="Asia/Shanghai")
    times = pd.date_range("2024-03-01", "2024-10-31", freq="4h", tz="Asia/Shanghai")
    n = len(times)
    ctx.labels = pd.DataFrame({"sample_id": [f"s{i:04d}" for i in range(n)],
                               "reference_time": times, "label_available_at": times + pd.Timedelta(hours=1),
                               "spout_no": np.arange(n) % 2 + 1,
                               "tap_iron": 450 + np.arange(n) % 20,
                               "tap_time_len": 100 + np.arange(n) % 10})
    ctx.code = {"source": "synthetic"}
    ctx.fit_records = []

    def features(self, samples, cutoff):
        return pd.DataFrame({"spout_no": samples.spout_no, "operation__stale": 0., "burden__stale": 0.}, index=samples.index)

    def frozen(self, train, evaluation, cutoff, candidate, context, increment=None):
        assert (train.reference_time < cutoff).all()
        assert (train.label_available_at <= cutoff).all()
        assert not set(train.sample_id) & set(evaluation.sample_id)
        result = pd.DataFrame({"sample_id": evaluation.sample_id}, index=evaluation.index)
        for target in ("tap_iron", "tap_time_len"):
            result[f"pred_{target}"] = float(train[target].median()) + 1
        return result

    ctx.feature_frame = MethodType(features, ctx)
    ctx.frozen = MethodType(frozen, ctx)
    return ctx


def test_reference_scheduler_same_outer_samples_and_per_origin_provenance(tmp_path):
    ctx = synthetic_context(tmp_path)
    reference_run(ctx)
    metrics = json.loads((tmp_path / "candidate_metrics.json").read_text())
    provenance = json.loads((tmp_path / "calibration_provenance.json").read_text())
    assert len(metrics) == 16  # two old folds and fourteen grid cells
    assert len(provenance) == 7
    assert all(p["sample_count"] >= 100 for p in provenance.values())
    for unit, row in metrics.items():
        raw = row["candidates"]["E12-raw"]["overall"]
        calibrated = row["candidates"]["E12-CVcal"]["overall"]
        assert raw["iron"] == calibrated["iron"]
        assert raw["time"]["n"] == calibrated["time"]["n"]
        a = pd.read_csv(tmp_path / "units" / unit / "E12-raw" / "predictions.csv")
        b = pd.read_csv(tmp_path / "units" / unit / "E12-CVcal" / "predictions.csv")
        assert list(a.sample_id) == list(b.sample_id)
    assert json.loads((tmp_path / "reference_selection.json").read_text())["C_ref"] in {"E12-raw", "E12-CVcal"}


def test_screening_requires_two_folds_and_capacity_ties(tmp_path):
    ctx = synthetic_context(tmp_path)
    calls = []

    def tunable(self, config, target, train, evaluation, cutoff, origin_id):
        calls.append((config["id"], target, origin_id))
        return np.repeat(float(train[target].median()), len(evaluation))

    ctx.tunable = MethodType(tunable, ctx)
    screening_run(ctx)
    assert len(set(calls)) == 60
    promotions = json.loads((tmp_path / "promotions.json").read_text())
    assert [p["candidate"] for p in promotions] == ["CB03", "CB03", "LG02", "LG02"]


def test_promoted_grid_integrates_reference_and_screening(tmp_path, monkeypatch):
    from bf_tap.optimization import v3_grid
    from bf_tap.optimization.v3_evidence import paired_week_intervals
    reference, screen, grid = [tmp_path / n for n in ("reference", "screen", "grid")]
    for path in (reference, screen, grid):
        path.mkdir()
    reference_run(synthetic_context(reference))

    def tunable(self, config, target, train, evaluation, cutoff, origin_id):
        return np.repeat(float(train[target].median()), len(evaluation))

    context = synthetic_context(screen)
    context.tunable = MethodType(tunable, context)
    screening_run(context)
    for path, suite in ((reference, "reference"), (screen, "screening")):
        atomic_write_json(path / "final_status.json", {"engineering_status": "G0_EXECUTION_PASS", "suite": suite})
        atomic_write_json(path / "resolved_config.json", {"inputs": {}})
    context = synthetic_context(grid)
    context.inputs = {}
    context.tunable = MethodType(tunable, context)
    monkeypatch.setattr(v3_grid, "paired_week_intervals",
                        lambda errors, candidate, reference: paired_week_intervals(errors, candidate, reference, repetitions=4))
    v3_grid.run(context, reference, screen)
    gate = json.loads((grid / "acceptance.json").read_text())
    assert set(gate["candidates"]) == {"CB-best-per-target", "LG-best-per-target", "F-A", "F-B"}
    assert json.loads((grid / "final_status.json").read_text())["engineering_status"] == "G0_EXECUTION_PASS"
