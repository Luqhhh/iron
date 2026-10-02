"""Pure-array/fake-path triage boundaries; no private reads, models, or fits."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import yaml

from bf_tap_r2.laplace_leaf_partition import ResidualL1PartitionRegressor
from sklearn.tree import DecisionTreeRegressor


_SOURCE = Path(__file__).resolve().parents[1]/"scripts/classify_laplace_leaf_partition.py"
_SPEC = importlib.util.spec_from_file_location("leaf_partition_triage_under_test", _SOURCE)
triage_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(triage_module)


def frozen_pool():
    return dict(split_seeds=[42, 3407], folds=5, policy_sha256="policy-hash",
        reference_by_target={"tap_iron":"EMA_TIME_Q75"},
        candidates={"tap_iron":["RESIDUAL_L1_PARTITION_D3_A20"]},
        tie_preference_by_target={"tap_iron":["RESIDUAL_L1_PARTITION_D3_A20"]},
        new_official_sample_label_parses=0, official_file_hash_reads_permitted=True)


def frozen_policy():
    return dict(version="candidate-tiers-v1", max_exploration_per_round=1, tie_tolerance=1e-6,
        promotion=dict(numerical_equality_tolerance=1e-12, minimum_improved_folds=7,
                       max_spout_wmape_degradation=.001))


def reference_arrays():
    y = np.arange(1., 21.)
    q = dict(targets=np.column_stack((y, np.ones(len(y)))), spout=np.tile([1, 2], 10))
    for seed, shift in ((42, 0), (3407, 1)):
        q[f"fold-{seed}"] = np.roll(np.tile(np.arange(5), 4), shift)
        q[f"current-{seed}"] = np.column_stack((1.1*y, np.ones(len(y))))
    return q, {seed:y.copy() for seed in (42, 3407)}


@pytest.fixture(autouse=True)
def prohibit_real_fits_models_or_artifact_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Triage tests must use fake evidence, never private reads or real fits")
    monkeypatch.setattr(ResidualL1PartitionRegressor, "fit", forbidden)
    monkeypatch.setattr(DecisionTreeRegressor, "fit", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(triage_module.np, "load", forbidden)
    monkeypatch.setattr(triage_module, "read_json", forbidden)
    monkeypatch.setattr(triage_module, "sha", forbidden)
    monkeypatch.setattr(triage_module, "inventory", forbidden)
    monkeypatch.setattr(triage_module, "original", forbidden)


@pytest.fixture
def native_evidence(monkeypatch):
    root = Path("virtual-root")
    run = root/"local/native-run"
    spec = dict(identity="native-id", split_seeds=[42, 3407], folds=5,
                candidate="RESIDUAL_L1_PARTITION_D3_A20", reference="EMA_TIME_Q75",
                node_statistic_absolute_tolerance=1e-12)
    units, models = [], []
    for seed in (42, 3407):
        for fold in range(5):
            key = f"s{seed}-f{fold}-D3_L1_PARTITION"
            units.append(dict(key=key, seed=seed, fold=fold,
                control_key=f"s{seed}-f{fold}-D3_LONG", selected_epoch=5, selector_cap_hit=False))
            models.extend(dict(key=key+"-"+role, metadata=dict(fitted_tree_count=epochs))
                          for role, epochs in (("selector", 12000), ("refit", 5)))
    digest = lambda path: "fake-hash:"+str(path)
    source_hashes = {"native.py":"source-digest"}
    input_names = {"复赛_train/train_samples.csv", "复赛_train/train_features.csv",
        f"{triage_module.REFERENCE_MATERIAL}/MANIFEST.json",
        f"{triage_module.REFERENCE_MATERIAL}/oof/original/q75-development-reference.npz"}
    manifest = dict(action="development", identity=spec["identity"], spec=deepcopy(spec),
        sources=deepcopy(source_hashes), git_commit=triage_module.NATIVE_GIT_COMMIT,
        input_sha256={name:digest(root/name) for name in input_names})
    result = dict(stage="development", model_fits=20, models=models, units=units,
        tree_fits=120050, formal_promoted=False, packages=0, full_fits=0,
        confirmation_seeds_consumed=0, agent_uploads=0, original_control_models_reused=20,
        output_sha256={name:digest(run/name) for name in triage_module.expected_artifacts(units)})
    cold = dict(status=triage_module.COLD_STATUS, checked_models=20,
        checked_tree_fits=120050, maximum_cold_difference=0., new_fits=0,
        model_fits_in_audit=0, packages=0, residual_target_node_statistics_checked=True,
        node_statistic_absolute_tolerance=1e-12,
        manifest_sha256=digest(run/"manifest.json"), result_sha256=digest(run/"result.json"))
    objects = {"manifest.json":manifest, "result.json":result, "cold-readback.json":cold}
    validated = []
    monkeypatch.setattr(triage_module, "read_json", lambda path: deepcopy(objects[path.name]))
    monkeypatch.setattr(triage_module, "sha", digest)
    monkeypatch.setattr(triage_module, "inventory", lambda path: deepcopy(source_hashes))
    monkeypatch.setattr(triage_module, "validate_spec", lambda value: validated.append(deepcopy(value)))
    return dict(root=root, run=run, spec=spec, objects=objects, digest=digest, validated=validated)


def require(evidence):
    return triage_module.require_native_complete(evidence["root"], evidence["run"], evidence["spec"])


def test_exact_complete_native_boundary_accepts_only_fake_evidence(native_evidence):
    manifest, result, cold = require(native_evidence)
    assert result["model_fits"] == cold["checked_models"] == 20
    assert manifest["git_commit"] == triage_module.NATIVE_GIT_COMMIT
    assert native_evidence["validated"] == [native_evidence["spec"]]


@pytest.mark.parametrize("change", ["action", "identity", "spec", "sources", "git_commit", "input_hash"])
def test_native_identity_errors_rejected(native_evidence, change):
    manifest = native_evidence["objects"]["manifest.json"]
    if change=="spec":
        manifest["spec"]["identity"] = "different-science"
    elif change=="sources":
        manifest["sources"]["native.py"] = "changed-source"
    elif change=="input_hash":
        name = next(iter(manifest["input_sha256"]))
        manifest["input_sha256"][name] = "changed-input"
    else:
        manifest[change] = "wrong"
    with pytest.raises(ValueError):
        require(native_evidence)


@pytest.mark.parametrize("field", ["model_fits", "packages", "full_fits", "confirmation_seeds_consumed", "agent_uploads", "formal_promoted"])
def test_incomplete_or_released_science_rejected(native_evidence, field):
    result = native_evidence["objects"]["result.json"]
    result[field] = 19 if field=="model_fits" else True if field=="formal_promoted" else 1
    with pytest.raises(ValueError):
        require(native_evidence)


@pytest.mark.parametrize("change", ["missing_seed", "wrong_seed", "duplicate_fold", "wrong_key", "wrong_control", "wrong_epoch", "wrong_cap_flag"])
def test_full_unit_roles_cannot_be_replaced_or_partial(native_evidence, change):
    units = native_evidence["objects"]["result.json"]["units"]
    if change=="missing_seed":
        del units[5:]
    elif change=="wrong_seed":
        units[0]["seed"] = 3407
    elif change=="duplicate_fold":
        units[1] = deepcopy(units[0])
    elif change=="wrong_key":
        units[0]["key"] = "other-route"
    elif change=="wrong_control":
        units[0]["control_key"] = "different-control"
    elif change=="wrong_epoch":
        units[0]["selected_epoch"] = 12001
    else:
        units[0]["selector_cap_hit"] = True
    with pytest.raises(ValueError):
        require(native_evidence)


@pytest.mark.parametrize("change", ["missing_model", "wrong_role", "wrong_selector_horizon", "wrong_refit_horizon", "tree_total"])
def test_saved_model_roles_and_tree_count_must_close(native_evidence, change):
    result = native_evidence["objects"]["result.json"]
    if change=="missing_model":
        result["models"].pop()
    elif change=="wrong_role":
        result["models"][0]["key"] = "another-model"
    elif change=="wrong_selector_horizon":
        result["models"][0]["metadata"]["fitted_tree_count"] = 11999
    elif change=="wrong_refit_horizon":
        result["models"][1]["metadata"]["fitted_tree_count"] = 6
    else:
        result["tree_fits"] += 1
    with pytest.raises(ValueError):
        require(native_evidence)


@pytest.mark.parametrize("field,value", [
    ("status", "pending"), ("checked_models", 19), ("checked_tree_fits", 120049),
    ("maximum_cold_difference", 1e-15), ("new_fits", 1), ("model_fits_in_audit", 1),
    ("packages", 1), ("residual_target_node_statistics_checked", False),
    ("node_statistic_absolute_tolerance", 1e-6), ("manifest_sha256", "wrong"),
    ("result_sha256", "wrong")])
def test_cold_gate_is_exact_and_complete(native_evidence, field, value):
    native_evidence["objects"]["cold-readback.json"][field] = value
    with pytest.raises(ValueError):
        require(native_evidence)


@pytest.mark.parametrize("change", ["missing_oof", "missing_query", "extra_path", "changed_hash"])
def test_complete_hash_inventory_cannot_be_partial_or_escape(native_evidence, change):
    hashes = native_evidence["objects"]["result.json"]["output_sha256"]
    if change=="missing_oof":
        del hashes["oof-3407.npz"]
    elif change=="missing_query":
        del hashes["s42-f0-D3_L1_PARTITION-query.npz"]
    elif change=="extra_path":
        hashes["../other-task.pkl"] = "bad-path"
    else:
        hashes["oof-42.npz"] = "changed"
    with pytest.raises(ValueError):
        require(native_evidence)


def test_absent_terminal_is_rejected_before_any_array_read(native_evidence, monkeypatch):
    def absent(path):
        raise FileNotFoundError(str(path))
    monkeypatch.setattr(triage_module, "read_json", absent)
    with pytest.raises(FileNotFoundError):
        require(native_evidence)


def test_frozen_pool_policy_boundaries(monkeypatch):
    pool = frozen_pool()
    science = dict(candidate=pool["candidates"]["tap_iron"][0], reference="EMA_TIME_Q75")
    semantic = hashlib.sha256(json.dumps(pool, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(triage_module, "FROZEN_TRIAGE_SEMANTIC_SHA256", semantic)
    triage_module.validate_triage(pool, science, "policy-hash")
    with pytest.raises(ValueError):
        triage_module.validate_triage(pool, science, "different-policy-hash")
    pool["candidates"]["tap_iron"].append("unregistered-candidate")
    with pytest.raises(ValueError):
        triage_module.validate_triage(pool, science, "policy-hash")


def test_actual_triage_yaml_and_every_field_match_full_freeze():
    root = Path(__file__).resolve().parents[1]
    pool = yaml.safe_load((root/triage_module.TRIAGE).read_text(encoding="utf-8"))
    science = json.loads((root/triage_module.SPEC).read_text(encoding="utf-8"))
    policy_hash = hashlib.sha256((root/triage_module.POLICY).read_bytes()).hexdigest()
    triage_module.validate_triage(pool, science, policy_hash)
    for field in pool:
        altered = deepcopy(pool)
        altered[field] = "changed"
        with pytest.raises(ValueError):
            triage_module.validate_triage(altered, science, policy_hash)


def test_fixed_blend_math_matches_pooled_fold_spout_scalar_arithmetic():
    q, predictions = reference_arrays()
    metrics = triage_module.build_metrics(q, predictions, frozen_pool())
    y = q["targets"][:, 0]
    candidate = frozen_pool()["candidates"]["tap_iron"][0]
    for seed in (42, 3407):
        record = metrics["tap_iron"][candidate][str(seed)]
        blend = .8*q[f"current-{seed}"][:, 0]+.2*predictions[seed]
        expected = sum(abs(float(a)-float(b)) for a,b in zip(y, blend))/sum(abs(float(a)) for a in y)
        assert record["wmape"] == pytest.approx(expected, abs=1e-15)
        for group, labels, keys in (("by_fold", q[f"fold-{seed}"], range(5)), ("by_spout", q["spout"], (1, 2))):
            for key in keys:
                rows = labels==key
                independent = sum(abs(float(a)-float(b)) for a,b in zip(y[rows], blend[rows]))/sum(abs(float(a)) for a in y[rows])
                assert record[group][str(key)] == pytest.approx(independent, abs=1e-15)


@pytest.mark.parametrize("change", ["missing_seed", "nan", "length", "fold_missing", "spout_missing", "invalid_blend"])
def test_incomplete_or_invalid_metric_vectors_cannot_be_shortlisted(change):
    q, predictions = reference_arrays()
    if change=="missing_seed":
        del predictions[3407]
    elif change=="nan":
        predictions[42][0] = np.nan
    elif change=="length":
        predictions[42] = predictions[42][:-1]
    elif change=="fold_missing":
        q["fold-42"][:] = 0
    elif change=="spout_missing":
        q["spout"][:] = 1
    else:
        predictions[42][:] = -1000
    with pytest.raises(ValueError):
        triage_module.build_metrics(q, predictions, frozen_pool())


def test_negative_raw_predictions_do_not_create_new_nonnegative_gate():
    q, predictions = reference_arrays()
    predictions[42][0] = -.1
    metrics = triage_module.build_metrics(q, predictions, frozen_pool())
    assert np.isfinite(metrics["tap_iron"][frozen_pool()["candidates"]["tap_iron"][0]]["42"]["wmape"])


def test_formal_output_is_only_development_shortlist():
    q, predictions = reference_arrays()
    pool = frozen_pool()
    report = triage_module.classify(triage_module.build_metrics(q, predictions, pool), pool,
                                   frozen_policy(), {"eligible_for_separately_frozen_confirmation":True})
    assert len(report["formal_selected"]) == 1
    assert report["formal_selected"][0]["tier"] == "formal"
    assert report["formal_scientific_promotion"] is False
    assert report["release_authorized"] is False
    assert report["automatically_scheduled"] is False
    assert report["new_fits"] == report["model_deserializations"] == report["packages"] == 0
    assert report["original_phase_eligible_for_separately_frozen_confirmation"] is True


def test_mean_positive_exploration_does_not_change_native_phase_decision():
    q, predictions = reference_arrays()
    predictions[3407] = 1.11*q["targets"][:, 0]
    pool = frozen_pool()
    report = triage_module.classify(triage_module.build_metrics(q, predictions, pool), pool,
                                   frozen_policy(), {"eligible_for_separately_frozen_confirmation":False})
    assert report["formal_selected"] == []
    assert len(report["exploration_selected"]) == 1
    assert report["original_phase_eligible_for_separately_frozen_confirmation"] is False
    assert report["release_authorized"] is False


class FakeArchive:
    def __init__(self, values):
        self.values = values

    def __enter__(self):
        return self.values

    def __exit__(self, *args):
        return False


@pytest.fixture
def query_archives(monkeypatch):
    run, control_run = Path("virtual/native"), Path("virtual/control")
    q, predictions = reference_arrays()
    archives, calls = {}, []
    for seed in (42, 3407):
        archives[run/f"oof-{seed}.npz"] = {"prediction":predictions[seed].copy()}
        folds = q[f"fold-{seed}"]
        for fold in range(5):
            outer, query = np.flatnonzero(folds!=fold), np.flatnonzero(folds==fold)
            inner = np.arange(len(outer)) % 5
            parts = dict(prediction=predictions[seed][query].copy(), outer=outer, query=query,
                         fit=outer[inner!=0], calibration=outer[inner==0], inner_folds=inner)
            archives[run/f"s{seed}-f{fold}-D3_L1_PARTITION-query.npz"] = deepcopy(parts)
            archives[control_run/f"s{seed}-f{fold}-D3_LONG-query.npz"] = deepcopy(parts)

    def fake_load(path, *, allow_pickle):
        calls.append((path, allow_pickle))
        assert allow_pickle is False
        if path not in archives:
            raise FileNotFoundError(str(path))
        return FakeArchive(deepcopy(archives[path]))

    monkeypatch.setattr(triage_module.np, "load", fake_load)
    return dict(run=run, control=control_run, q=q, predictions=predictions, archives=archives, calls=calls)


def test_complete_query_units_rebuild_exact_same_seed_oof_without_pickle(query_archives):
    evidence = query_archives
    values = triage_module.load_predictions(evidence["run"], evidence["q"], evidence["control"])
    for seed in (42, 3407):
        np.testing.assert_array_equal(values[seed], evidence["predictions"][seed])
    assert len(evidence["calls"]) == 22
    assert all(allow_pickle is False for _, allow_pickle in evidence["calls"])


@pytest.mark.parametrize("change", ["missing_seed", "length", "nan", "query_role", "outer_role", "fit_role",
                                    "calibration_role", "inner_dtype", "inner_missing_fold", "control_role",
                                    "unit_values", "unit_nonfinite"])
def test_query_coverage_and_control_roles_cannot_be_faked(query_archives, change):
    evidence, archives = query_archives, query_archives["archives"]
    oof = archives[evidence["run"]/"oof-42.npz"]
    unit = archives[evidence["run"]/"s42-f0-D3_L1_PARTITION-query.npz"]
    if change=="missing_seed":
        del archives[evidence["run"]/"oof-3407.npz"]
    elif change=="length":
        oof["prediction"] = oof["prediction"][:-1]
    elif change=="nan":
        oof["prediction"][0] = np.nan
    elif change=="query_role":
        unit["query"] = unit["query"][::-1]
    elif change=="outer_role":
        unit["outer"] = unit["outer"][:-1]
    elif change=="fit_role":
        unit["fit"][0] = unit["query"][0]
    elif change=="calibration_role":
        unit["calibration"][0] = unit["query"][0]
    elif change=="inner_dtype":
        unit["inner_folds"] = unit["inner_folds"].astype(float)
    elif change=="inner_missing_fold":
        unit["inner_folds"][:] = 0
    elif change=="control_role":
        archives[evidence["control"]/"s42-f0-D3_LONG-query.npz"]["fit"][0] = 99
    elif change=="unit_values":
        unit["prediction"][0] += 1
    else:
        unit["prediction"][0] = np.nan
    with pytest.raises((ValueError, AssertionError, FileNotFoundError)):
        triage_module.load_predictions(evidence["run"], evidence["q"], evidence["control"])


@pytest.mark.parametrize("changed", [False, True])
def test_recalculation_requires_exact_native_metrics_and_decision(query_archives, monkeypatch, changed):
    evidence = query_archives
    old = {"metrics":{str(seed):{"gain_vs_Q75":0.01} for seed in (42, 3407)},
           "output_sha256":{"control-array.npz":"control-array-hash"}}
    science = dict(original_manifest_sha256="old-manifest-hash",
                   original_result_sha256="old-result-hash", original_cold_sha256="old-cold-hash")
    metrics = triage_module.native_describe(evidence["q"], evidence["predictions"], old)
    result = dict(metrics=metrics,
        mean_gain_vs_Q75=float(np.mean([m["gain_vs_Q75"] for m in metrics.values()])),
        mean_gain_vs_original_long=float(np.mean([m["gain_vs_original_long"] for m in metrics.values()])),
        eligible_for_separately_frozen_confirmation=True)
    monkeypatch.setattr(triage_module, "original", lambda root, spec: (old, evidence["q"]))
    monkeypatch.setattr(triage_module, "OLD", str(evidence["control"]))
    if changed:
        result["metrics"]["42"]["gain_vs_Q75"] += 1e-9
        with pytest.raises(ValueError):
            triage_module.recalculate(Path("."), evidence["run"], science, frozen_pool(), result)
    else:
        rebuilt, inputs = triage_module.recalculate(Path("."), evidence["run"], science, frozen_pool(), result)
        assert set(rebuilt["tap_iron"]["EMA_TIME_Q75"]) == {"42", "3407"}
        assert inputs[str(evidence["control"]/"manifest.json")] == "old-manifest-hash"
        assert inputs[str(evidence["control"]/"result.json")] == "old-result-hash"
        assert inputs[str(evidence["control"]/"cold-readback.json")] == "old-cold-hash"
        assert inputs[str(evidence["control"]/"control-array.npz")] == "control-array-hash"
