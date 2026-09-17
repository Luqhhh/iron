"""Registered v0.26 lifecycle for two QRF partition-only experiments."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json, file_identities, file_sha256, runtime_environment
from ..config import load_yaml
from ..exceptions import ContractError
from .component_export import read_json
from .dual_burden_v24_run import (
    DEVELOPMENT,
    FINAL_QRF,
    V22,
    V23,
    _history,
    _parent,
    _reference_errors,
    _v10,
    frame_from,
    load_original,
    original_handoff,
    query_frame,
)
from .dual_ratio_common import week_intervals
from .qrf_partition_v26 import (
    CANDIDATE_A,
    CANDIDATE_B,
    PARENT,
    POOLED_AGGREGATION,
    PRIMARY_AGGREGATION,
    TARGETS,
    TOLERANCE,
    exposure_pooled_summary,
    isolate_time,
    macro_origin_summary,
    roundtrip_six,
    submission_bytes_preserving_iron,
    verify_isolated_deltas,
)
from .qrf_support_shrink import replay_v21, serialize_six_decimals
from .structural_run import append_ledger


WORKER = Path("workers/qrf_v026")
ORIGINS = range(6, 12)
SLOTS = range(6, 13)
CANDIDATES = {"A": CANDIDATE_A, "B": CANDIDATE_B}
PREDICTION_COLUMNS = ["pred_tap_iron", "pred_tap_time_len"]
V25 = Path("local/runs/optimization-v0.25-history-centered-targets-r1")
PROTOCOL = "QRF_FULLTRAIN_PARTITION_v026"


def registration():
    value = load_yaml("configs/optimization_v0_26/experiment.yaml")
    if value["candidates"] != CANDIDATES or value["base_commit"] != "cd2e06848f55b993cbb456ae89cbd084ba93f4b3":
        raise ContractError("v0.26 candidate/base registration differs")
    if value["protocol"] != PROTOCOL or value["parent_protocol"] != "QRF_FULLTRAIN_LEAF_v1":
        raise ContractError("v0.26 QRF protocol differs")
    expected_budget = {
        "development_fits_per_candidate": 6,
        "final_fits_per_candidate": 1,
        "total_fits_per_candidate": 7,
        "qrf_internal_trees_total": 3584,
        "new_preprocessor_fits": 0,
        "catboost_rate_e04_fits": 0,
        "LAD_beta_lambda_bias_fits": 0,
        "new_candidate_packages": 2,
        "new_candidate_platform_tests": 2,
    }
    if value["budget"] != expected_budget or value["aggregation"]["primary"] != PRIMARY_AGGREGATION:
        raise ContractError("v0.26 budget/aggregation registration differs")
    if value["candidate_B"] != {"estimator": "ExtraTreesRegressor", "splitter": "random", "criterion": "squared_error"}:
        raise ContractError("v0.26 ExtraTrees definition differs")
    if value["forest_parameters"]["B"]["bootstrap"] is not True:
        raise ContractError("v0.26 ExtraTrees must retain bootstrap=True")
    return value


def worker(command, *arguments):
    executable = Path("workers/qrf_v015/.venv/bin/python")
    return subprocess.check_output(
        [str(executable), str(WORKER / "worker.py"), command, *map(str, arguments)],
        text=True,
    )


def _source_files():
    return [
        Path("configs/optimization_v0_26/experiment.yaml"),
        Path("configs/optimization_v0_26/access_scope.yaml"),
        Path("src/bf_tap/optimization/qrf_partition_v26.py"),
        Path("src/bf_tap/optimization/qrf_partition_v26_run.py"),
        Path("scripts/optimization_v26_qrf_partition.py"),
        Path("scripts/optimization_v26_cold_check.py"),
        Path("tests/test_optimization_v26_qrf_partition.py"),
        Path("docs/optimization_v0_26/PLAN.md"),
        Path("docs/optimization_v0_25/AGGREGATION_ERRATA.md"),
        WORKER / "worker.py",
        WORKER / "partition_forest.py",
        WORKER / "test_partition_forest.py",
        Path("workers/qrf_v015/qrf_model.py"),
        Path("workers/qrf_v015/preprocessing.py"),
        Path("workers/qrf_v015/pyproject.toml"),
        Path("workers/qrf_v015/uv.lock"),
    ]


def _registry_search():
    manifests = sorted(Path("local/runs").glob("**/*manifest*.json"))
    matches = []
    for path in manifests:
        text = path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        reasons = []
        if CANDIDATE_A in text or CANDIDATE_B in text:
            reasons.append("candidate_id")
        if "absolute_error" in lowered and "qrf" in lowered and "randomforestregressor" in lowered:
            reasons.append("absolute_error_qrf_partition_signature")
        if "extratreesregressor" in lowered and "qrf" in lowered and '"bootstrap": true' in lowered:
            reasons.append("extra_trees_bootstrap_qrf_partition_signature")
        if reasons:
            matches.append({"path": str(path), "sha256": file_sha256(path), "reasons": reasons})
    return {"searched_manifest_count": len(manifests), "equivalent_matches": matches}


def register(root: Path):
    configuration = registration()
    scope = load_yaml("configs/optimization_v0_26/access_scope.yaml")
    if root.exists():
        raise ContractError("never overwrite a v0.26 run")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ContractError("clean preregistered working tree required")
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if branch != configuration["branch"]:
        raise ContractError("registered v0.26 branch required")
    subprocess.run(["git", "merge-base", "--is-ancestor", configuration["base_commit"], "HEAD"], check=True)
    registry = _registry_search()
    if registry["equivalent_matches"]:
        raise ContractError("equivalent v0.26 experiment already exists: " + str(registry["equivalent_matches"]))
    evidence = {
        "v25_scorecard": V25 / "canonical_scorecard.csv",
        "v25_summary": V25 / "canonical_summary.csv",
        "v25_errors": V25 / "all_errors.csv",
        "v22_errors": V22 / "all_errors.csv",
        "v23_completion": V23 / "completion.json",
        "v23_v21_result": V23 / "recovery/result.csv",
        "v23_v21_zip": V23 / "recovery/Luqhhh_bf_tap_predict_prelim_TGATE_SPOUT1.zip",
        "qrf_development_manifest": DEVELOPMENT / "manifest.json",
        "qrf_development_models": DEVELOPMENT / "models_complete.json",
        "qrf_final_manifest": FINAL_QRF / "manifest.json",
        "qrf_final_models": FINAL_QRF / "models_complete.json",
        "protection": Path(scope["protection_contract"]),
    }
    worker_environment = json.loads(worker("environment"))
    if worker_environment.get("python") != "3.12.12" or worker_environment.get("sklearn") != "1.8.0":
        raise ContractError("original locked Python 3.12.12 / scikit-learn 1.8.0 worker required")
    manifest = {
        "registration": configuration,
        "scope": scope,
        "protection": load_yaml(scope["protection_contract"]),
        "evidence": file_identities(evidence),
        "sources": file_identities({str(path): path for path in _source_files()}),
        "worker_environment": worker_environment,
        "worker_sources": json.loads(worker("identity")),
        "forest_parameters": configuration["forest_parameters"],
        "protocol": configuration["protocol"],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_environment(),
        "registry_search": registry,
        "holdout_consumed": True,
        "official_target_column_reads": False,
        "test_targets_read": False,
        "new_preprocessor_fit_budget": 0,
        "platform_feedback_may_change_second_candidate": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    atomic_write_json(root / "manifest.json", manifest)
    append_ledger(scope, root)
    atomic_write_json(root / "registration.json", {
        "status": "TWO_CANDIDATES_FROZEN_BEFORE_FIT",
        "candidates": CANDIDATES,
        "definitions": {key: configuration[f"candidate_{key}"] for key in CANDIDATES},
        "platform_order": configuration["platform_order"],
        "candidate_platform_budget": 2,
        "platform_uploads": 0,
        "definitions_may_change_after_A_feedback": False,
        "v21_zip_sha256": file_sha256(V23 / "recovery/Luqhhh_bf_tap_predict_prelim_TGATE_SPOUT1.zip"),
    })
    return manifest


def _score_rows(errors: pd.DataFrame, algorithm: str):
    rows = []
    for unit, part in errors.groupby("unit", sort=False):
        target_metrics = {}
        for target in TARGETS:
            denominator = float(part[target].sum())
            numerator = float(part[f"abs_error_{target}"].sum())
            signed = float(part[f"error_{target}"].sum())
            target_metrics[target] = numerator / denominator
            rows.append({
                "algorithm": algorithm,
                "unit": unit,
                "cutoff": part.origin.iloc[0],
                "horizon": None if pd.isna(part.horizon.iloc[0]) else int(part.horizon.iloc[0]),
                "target": target,
                "n": len(part),
                "target_sum": denominator,
                "absolute_error_sum": numerator,
                "signed_error_sum": signed,
                "signed_bias": signed / len(part),
                "wmape": numerator / denominator,
                "E": 0.0,
            })
        loss = float(np.mean(list(target_metrics.values())))
        rows[-2]["E"] = loss
        rows[-1]["E"] = loss
    return rows


def _audit_v25(root: Path):
    scorecard_path = V25 / "canonical_scorecard.csv"
    summary_path = V25 / "canonical_summary.csv"
    errors_path = V25 / "all_errors.csv"
    scorecard = pd.read_csv(scorecard_path)
    old_summary = pd.read_csv(summary_path)
    errors = pd.read_csv(errors_path, dtype={"sample_id": str})
    algorithms = scorecard.algorithm.unique().tolist()
    if set(algorithms) != set(errors.candidate.unique()):
        raise ContractError("v0.25 scorecard/error algorithms differ")
    parent_errors = errors.loc[errors.candidate == PARENT]
    identity_cells = 0
    for algorithm in algorithms:
        current = errors.loc[errors.candidate == algorithm]
        for unit, expected in parent_errors.groupby("unit", sort=False):
            actual = current.loc[current.unit == unit]
            if actual.duplicated("sample_id").any() or len(actual) != len(expected):
                raise ContractError("v0.25 per-cell sample count differs")
            left = expected.set_index("sample_id").sort_index()
            right = actual.set_index("sample_id").sort_index()
            if not left.index.equals(right.index) or not left[list(TARGETS)].equals(right[list(TARGETS)]):
                raise ContractError("v0.25 per-cell sample IDs or targets differ")
            identity_cells += 1
    recomputed = pd.DataFrame([
        row for algorithm in algorithms
        for row in _score_rows(errors.loc[errors.candidate == algorithm], algorithm)
    ])
    keys = ["algorithm", "unit", "target"]
    left = scorecard.sort_values(keys).reset_index(drop=True)
    right = recomputed.sort_values(keys).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ContractError("v0.25 recomputed scorecard keys differ")
    exact_metrics = ["n", "target_sum"]
    score_metrics = ["signed_bias", "wmape", "E"]
    serialized_sums = ["absolute_error_sum", "signed_error_sum"]
    if (
        not np.allclose(left[exact_metrics], right[exact_metrics], rtol=0, atol=0)
        or not np.allclose(left[score_metrics], right[score_metrics], rtol=0, atol=TOLERANCE)
        or not np.allclose(left[serialized_sums], right[serialized_sums], rtol=0, atol=1e-10)
    ):
        raise ContractError("v0.25 recomputed cell metrics differ")
    macro = macro_origin_summary(scorecard)
    pooled = exposure_pooled_summary(scorecard)
    old_top = old_summary.loc[old_summary.scope_type.isin(("HORIZON_MEAN", "GRID", "DEVELOPMENT"))].set_index(["algorithm", "scope"])
    new_top = macro.loc[macro.scope_type.isin(("HORIZON_MEAN", "GRID", "DEVELOPMENT"))].set_index(["algorithm", "scope"])
    if set(old_top.index) != set(new_top.index):
        raise ContractError("v0.25 canonical top-level summary grid differs")
    summary_difference = float(np.max(np.abs(old_top.loc[new_top.index, "E"].to_numpy() - new_top.E.to_numpy())))
    if summary_difference > TOLERANCE:
        raise ContractError("v0.25 canonical E is not macro-origin mean")
    isolation = verify_isolated_deltas(scorecard, macro, {
        "V25I_HISTORY_CENTERED_RECENCY_IRON": "tap_iron",
        "V25T_HISTORY_CENTERED_QRF_TIME": "tap_time_len",
    })
    reported = registration()["v25_reported_target_deltas"]
    pooled_lookup = pooled.set_index(["algorithm", "scope"])
    report_matches = []
    for candidate, definition in reported.items():
        field = "wmape_iron" if definition["target"] == "tap_iron" else "wmape_time"
        for horizon in range(1, 5):
            scope = f"H{horizon}"
            delta = float(pooled_lookup.loc[(candidate, scope), field] - pooled_lookup.loc[(PARENT, scope), field])
            printed = float(definition[scope])
            if abs(delta - printed) > 5.1e-9:
                raise ContractError("v0.25 reported target delta is neither matching pooled exposure nor print rounding")
            report_matches.append({
                "candidate": candidate,
                "scope": scope,
                "target": definition["target"],
                "reported": printed,
                "exposure_pooled_delta": delta,
                "macro_origin_mean_delta": float(new_top.loc[(candidate, scope), field] - new_top.loc[(PARENT, scope), field]),
            })
    destination = root / "p0"
    destination.mkdir(parents=True, exist_ok=True)
    macro.to_csv(destination / "v25_macro_origin_mean_wmape.csv", index=False)
    pooled.to_csv(destination / "v25_exposure_pooled_wmape.csv", index=False)
    pd.DataFrame(report_matches).to_csv(destination / "v25_reported_delta_reconciliation.csv", index=False)
    audit = {
        "status": "PASS_NO_RETRAIN",
        "source_identities": file_identities({
            "canonical_scorecard": scorecard_path,
            "canonical_summary": summary_path,
            "all_errors": errors_path,
        }),
        "algorithms": algorithms,
        "identity_cells_checked": identity_cells,
        "sample_ids_n_targets_and_denominators_consistent": True,
        "canonical_cell_score_recompute_max_abs_tolerance": TOLERANCE,
        "serialized_error_sum_recompute_max_abs_tolerance": 1e-10,
        "canonical_summary_E_macro_max_abs_difference": summary_difference,
        "primary_aggregation": PRIMARY_AGGREGATION,
        "diagnostic_aggregation": POOLED_AGGREGATION,
        "isolated_delta_identity": isolation,
        "reported_target_deltas_match": POOLED_AGGREGATION,
        "cause": "RESULTS.md target-specific deltas used exposure-pooled denominators while E used equal-origin macro means",
        "old_v25_artifacts_modified": False,
        "model_fits": 0,
    }
    atomic_write_json(destination / "v25_aggregation_audit.json", audit)
    return audit


def _original_root(slot: int):
    return FINAL_QRF if slot == 12 else DEVELOPMENT


def _copy_handoff(root: Path, slot: int, kind: str):
    source = original_handoff(slot, kind)
    old = read_json(str(source) + ".json")
    bundle_path = _original_root(slot) / f"models/{slot}/bundle.json"
    bundle = read_json(bundle_path)
    preprocessor_path = _original_root(slot) / f"models/{slot}/preprocessor.json"
    if file_sha256(preprocessor_path) != bundle["preprocessor_sha256"]:
        raise ContractError("original v0.15 preprocessor identity differs")
    if kind == "train" and old["sha256"] != bundle["input"]["sha256"]:
        raise ContractError("original training handoff/model identity differs")
    if len(old["numeric_columns"]) != 209 or len(old["raw_schema"]) != 210:
        raise ContractError("v0.26 requires original 210-column E09/R2 schema")
    names = [entry["name"] for entry in old["raw_schema"]]
    if names[0] != "spout_no" or any(name.startswith("burden_lag__") for name in names):
        raise ContractError("v0.26 raw schema includes a forbidden feature family")
    destination = root / "features" / str(slot) / f"{kind}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ContractError("never overwrite a v0.26 handoff")
    shutil.copyfile(source, destination)
    if file_sha256(destination) != old["sha256"]:
        raise ContractError("exact original handoff byte copy differs")
    info = {
        **old,
        "sha256": file_sha256(destination),
        "original_input_path": str(source),
        "original_input_sha256": old["sha256"],
        "original_bundle_path": str(bundle_path),
        "original_bundle_sha256": file_sha256(bundle_path),
        "original_preprocessor_path": str(preprocessor_path),
        "original_preprocessor_sha256": file_sha256(preprocessor_path),
        "original_transformed_schema_sha256": bundle["transformed_schema_sha256"],
        "exact_source_bytes": True,
        "preprocessor_mode": "restore_and_transform_only",
    }
    atomic_write_json(str(destination) + ".json", info)
    return info


def prepare(root: Path):
    if (root / "p0_complete.json").exists():
        raise ContractError("v0.26 P0 already complete")
    if any((root / name).exists() for name in ("p0", "features", "old_qrf_predictions")):
        raise ContractError("partial v0.26 P0 evidence retained; explicit recovery decision required")
    audit = _audit_v25(root)
    handoffs = {}
    transformed = {}
    old_support = {}
    for slot in SLOTS:
        handoffs[str(slot)] = {kind: _copy_handoff(root, slot, kind) for kind in ("train", "evaluation")}
        if len(handoffs[str(slot)]["train"]["ids"]) != registration()["expected_training_rows"][slot]:
            raise ContractError("certified training row count differs")
        transformed[str(slot)] = json.loads(worker("audit", "--root", root.resolve(), "--slot", slot))
        output = root / "old_qrf_predictions" / f"{slot}.npz"
        worker("old-predict", "--root", root.resolve(), "--slot", slot, "--output", output.resolve())
        old_support[str(slot)] = read_json(str(output) + ".json")
    if old_support["12"]["gate_rows_spout1_neff_lt_500"] != 32:
        raise ContractError("recomputed test_a old-QRF gate identity is not 32 rows")
    atomic_write_json(root / "p0_complete.json", {
        "status": "PASS_ZERO_FIT",
        "registry_search": read_json(root / "manifest.json")["registry_search"],
        "v25_aggregation_audit": audit,
        "handoffs": handoffs,
        "transformed_input_audits": transformed,
        "old_qrf_support_recomputed": old_support,
        "test_a_gate_rows": 32,
        "original_raw_columns": 210,
        "original_numeric_columns": 209,
        "new_preprocessor_fits": 0,
        "model_fits": 0,
        "timezone": "Asia/Shanghai",
        "availability_contract": "ASSUMED_UNCHANGED",
    })


def _load_handoff(root: Path, slot: int, kind: str):
    path = root / "features" / str(slot) / f"{kind}.npz"
    info = read_json(str(path) + ".json")
    if file_sha256(path) != info["sha256"]:
        raise ContractError("v0.26 handoff identity changed")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if arrays["ids"].tolist() != info["ids"]:
        raise ContractError("v0.26 handoff IDs differ")
    return arrays, info


def _model_completion(root: Path, slots):
    result = {candidate: {} for candidate in CANDIDATES}
    for candidate in CANDIDATES:
        for slot in slots:
            folder = root / "models" / candidate / str(slot)
            if not (folder / "fit_record.json").is_file():
                raise ContractError("v0.26 fit completion record missing")
            record = read_json(folder / "fit_record.json")
            digest = file_sha256(folder / "bundle.json")
            if record["status"] != "COMPLETED" or record["bundle_sha256"] != digest:
                raise ContractError("v0.26 fit completion identity differs")
            result[candidate][str(slot)] = digest
    return result


def _fit_one(root: Path, slot: int, candidate: str):
    folder = root / "models" / candidate / str(slot)
    if (folder / "fit_record.json").is_file():
        return
    if (folder / "fit_intent.json").exists():
        raise ContractError("incomplete v0.26 fit attempt retained; explicit recovery decision required")
    worker("fit", "--root", root.resolve(), "--slot", slot, "--candidate", candidate)


def _old_qrf(root: Path, slot: int):
    path = root / "old_qrf_predictions" / f"{slot}.npz"
    info = read_json(str(path) + ".json")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if arrays["ids"].tolist() != _load_handoff(root, slot, "evaluation")[0]["ids"].tolist():
        raise ContractError("recomputed old QRF IDs differ")
    if not info["recomputed_from_original_forest"] or not info["source_saved_prediction_exact"]:
        raise ContractError("old QRF support was not independently recomputed")
    return arrays, info


def _replay(root: Path, slot: int, qrf_time):
    original, info = _load_handoff(root, slot, "evaluation")
    query = query_frame(original)
    ids = query.sample_id.tolist()
    base = _v10(slot)
    if base.sample_id.tolist() != ids:
        base = base.set_index("sample_id").loc[ids].reset_index()
    base["pred_tap_time_len"] = roundtrip_six(qrf_time)
    old, _ = _old_qrf(root, slot)
    result, audit = replay_v21(
        base,
        query,
        _history(slot),
        old["effective_neighbors"],
        cutoff=pd.Timestamp(info["cutoff"]),
    )
    result["pred_tap_time_len"] = roundtrip_six(result.pred_tap_time_len)
    audit["gate_source"] = "recomputed_original_QRF_effective_neighbors"
    audit["old_support_reused_from_saved_V21_array"] = False
    return result, audit


def _save_prediction(path: Path, value: pd.DataFrame):
    if path.exists():
        raise ContractError("never overwrite a v0.26 prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    value.to_csv(path, index=False)


def _predict_slot(root: Path, slot: int):
    arrays, _ = _load_handoff(root, slot, "evaluation")
    ids = arrays["ids"].tolist()
    parent = _parent(slot)
    if parent.sample_id.tolist() != ids:
        parent = parent.set_index("sample_id").loc[ids].reset_index()
    old, old_info = _old_qrf(root, slot)
    old_replay, old_audit = _replay(root, slot, old["median"])
    if serialize_six_decimals(old_replay) != serialize_six_decimals(parent):
        raise ContractError("recomputed original QRF substitution does not reproduce V21")
    audits = {"old_replay": old_audit, "old_support": old_info}
    for candidate, candidate_id in CANDIDATES.items():
        path = root / "worker_predictions" / candidate / f"{slot}.npz"
        worker("predict", "--root", root.resolve(), "--slot", slot, "--candidate", candidate, "--output", path.resolve())
        with np.load(path, allow_pickle=False) as prediction:
            if prediction["ids"].tolist() != ids:
                raise ContractError("v0.26 worker prediction IDs differ")
            raw_median = prediction["median"]
        replayed, replay_audit = _replay(root, slot, raw_median)
        candidate_frame = isolate_time(parent, replayed.pred_tap_time_len.to_numpy(dtype=float))
        if not np.array_equal(candidate_frame.pred_tap_iron.to_numpy(), parent.pred_tap_iron.to_numpy()):
            raise ContractError("v0.26 candidate changed V21 iron")
        _save_prediction(root / "predictions" / f"{slot}_{candidate_id}.csv", candidate_frame)
        audits[candidate] = {
            "candidate_id": candidate_id,
            "replay": replay_audit,
            "unchanged_iron_exact": True,
            "all_rows_use_new_forest_prediction": True,
            "new_effective_neighbors_used_for_gate": False,
        }
    atomic_write_json(root / "predictions" / f"{slot}_audit.json", audits)


def develop(root: Path):
    if (root / "development_complete.json").exists():
        raise ContractError("v0.26 development already complete")
    for candidate in CANDIDATES:
        for slot in ORIGINS:
            _fit_one(root, slot, candidate)
    completed = _model_completion(root, ORIGINS)
    completion_path = root / "development_models_complete.json"
    if completion_path.exists():
        if read_json(completion_path) != completed:
            raise ContractError("saved development model completion identity differs")
    else:
        atomic_write_json(completion_path, completed)
    for slot in ORIGINS:
        _predict_slot(root, slot)
    prediction_paths = {
        f"{slot}_{candidate}": root / "predictions" / f"{slot}_{candidate_id}.csv"
        for slot in ORIGINS for candidate, candidate_id in CANDIDATES.items()
    }
    atomic_write_json(root / "development_predictions_frozen.json", {
        "status": "BOTH_CANDIDATES_FROZEN_BEFORE_SCORING",
        "predictions": file_identities(prediction_paths),
        "platform_feedback_seen": False,
    })
    atomic_write_json(root / "development_complete.json", {
        "status": "PASS_G0_DEVELOPMENT_PREDICTIONS_FROZEN",
        "A_fits": 6,
        "B_fits": 6,
        "new_preprocessor_fits": 0,
        "internal_trees": 3072,
        "calibration_fits": 0,
    })


def _candidate_errors(root: Path, candidate: str):
    source = pd.read_csv(V22 / "all_errors.csv", dtype={"sample_id": str})
    base = source.loc[source.candidate == "V1"].copy()
    frames = []
    for origin, part in base.groupby("origin", sort=False):
        slot = int(str(origin)[-2:])
        prediction = pd.read_csv(root / "predictions" / f"{slot}_{candidate}.csv", dtype={"sample_id": str}).set_index("sample_id")
        aligned = prediction.loc[part.sample_id]
        current = part.copy()
        current["pred_tap_iron"] = roundtrip_six(aligned.pred_tap_iron)
        current["pred_tap_time_len"] = roundtrip_six(aligned.pred_tap_time_len)
        for target in TARGETS:
            current[f"error_{target}"] = current[f"pred_{target}"] - current[target]
            current[f"abs_error_{target}"] = current[f"error_{target}"].abs()
        current["candidate"] = candidate
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def _write_report(root: Path, summary: pd.DataFrame, deltas: pd.DataFrame):
    lines = [
        "# optimization-v0.26 自动生成离线报告",
        "",
        "主口径为 `macro_origin_mean_wmape`；pooled exposure 仅见独立诊断 CSV。负 Δ 表示优于 V21_REPLAY。",
        "",
        "| 候选 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in CANDIDATES.values():
        part = deltas.loc[(deltas.candidate == candidate) & (deltas.reference == PARENT)].set_index("scope")
        values = [part.loc[scope, "delta_E"] for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")]
        lines.append("| " + candidate + " | " + " | ".join(f"{value:+.8f}" for value in values) + " |")
    lines.extend(["", "G0 工程状态与 G1 离线质量在 completion 中分开登记；离线质量不取消预登记平台名额。", ""])
    path = root / "REPORT.md"
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def score(root: Path):
    if (root / "offline_assessment.json").exists() or (root / "canonical_scorecard.csv").exists():
        raise ContractError("never overwrite v0.26 scoring evidence")
    candidates = {candidate: _candidate_errors(root, candidate) for candidate in CANDIDATES.values()}
    references = {name: _reference_errors(name) for name in ("V1", "V10", PARENT)}
    all_errors = pd.concat([*references.values(), *candidates.values()], ignore_index=True)
    all_errors.to_csv(root / "all_errors.csv", index=False)
    scorecard = pd.DataFrame([
        row for name, errors in {**references, **candidates}.items()
        for row in _score_rows(errors, name)
    ])
    scorecard.to_csv(root / "canonical_scorecard.csv", index=False)
    summary = macro_origin_summary(scorecard)
    summary.to_csv(root / "canonical_summary.csv", index=False)
    pooled = exposure_pooled_summary(scorecard)
    pooled.to_csv(root / "exposure_pooled_summary.csv", index=False)
    isolation = verify_isolated_deltas(scorecard, summary, {candidate: "tap_time_len" for candidate in candidates})
    primary = summary.loc[summary.scope_type.isin(("HORIZON_MEAN", "GRID", "DEVELOPMENT"))].set_index(["algorithm", "scope"])
    deltas = []
    for candidate in candidates:
        for reference in ("V1", PARENT):
            for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT"):
                left = primary.loc[(candidate, scope)]
                right = primary.loc[(reference, scope)]
                deltas.append({
                    "candidate": candidate,
                    "reference": reference,
                    "scope": scope,
                    "aggregation": PRIMARY_AGGREGATION,
                    "E": float(left.E),
                    "delta_E": float(left.E - right.E),
                    "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                    "delta_wmape_time": float(left.wmape_time - right.wmape_time),
                })
    deltas_frame = pd.DataFrame(deltas)
    deltas_frame.to_csv(root / "canonical_deltas.csv", index=False)
    pooled_deltas = []
    pooled_index = pooled.set_index(["algorithm", "scope"])
    for candidate in candidates:
        for scope in ("H1", "H2", "H3", "H4", "DEV_LONG", "DEV_SHORT"):
            left = pooled_index.loc[(candidate, scope)]
            right = pooled_index.loc[(PARENT, scope)]
            pooled_deltas.append({
                "candidate": candidate,
                "reference": PARENT,
                "scope": scope,
                "aggregation": POOLED_AGGREGATION,
                "delta_E": float(left.E - right.E),
                "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                "delta_wmape_time": float(left.wmape_time - right.wmape_time),
            })
    pd.DataFrame(pooled_deltas).to_csv(root / "exposure_pooled_deltas.csv", index=False)
    bootstrap = {
        candidate: {reference: week_intervals(all_errors, candidate, reference, 1000, 2026) for reference in ("V1", PARENT)}
        for candidate in candidates
    }
    atomic_write_json(root / "bootstrap.json", bootstrap)
    _write_report(root, summary, deltas_frame)
    lookup = primary.E
    atomic_write_json(root / "offline_assessment.json", {
        "status": "COMPLETED_CONSUMED_RETROSPECTIVE_NOT_A_PLATFORM_GATE",
        "primary_aggregation": PRIMARY_AGGREGATION,
        "diagnostic_aggregation": POOLED_AGGREGATION,
        "isolated_target_audit": isolation,
        "candidates": {
            candidate: {scope: float(lookup.loc[(candidate, scope)]) for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")}
            for candidate in candidates
        },
        "references": {
            reference: {scope: float(lookup.loc[(reference, scope)]) for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")}
            for reference in references
        },
        "both_candidates_retain_preregistered_platform_slot": True,
    })


def finalize(root: Path):
    if not (root / "offline_assessment.json").exists():
        raise ContractError("score v0.26 development before final fits")
    if (root / "packages_frozen_before_feedback.json").exists():
        raise ContractError("v0.26 final packages already frozen")
    for candidate in CANDIDATES:
        _fit_one(root, 12, candidate)
    completed = _model_completion(root, (12,))
    completion_path = root / "final_models_complete.json"
    if completion_path.exists():
        if read_json(completion_path) != completed:
            raise ContractError("saved final model completion identity differs")
    else:
        atomic_write_json(completion_path, completed)
    _predict_slot(root, 12)
    parent_path = V23 / "recovery/result.csv"
    parent_strings = pd.read_csv(parent_path, dtype=str, keep_default_na=False)
    receipts = {}
    for key, candidate in CANDIDATES.items():
        prediction = pd.read_csv(root / "predictions" / f"12_{candidate}.csv", dtype={"sample_id": str})
        folder = root / "submissions" / candidate
        folder.mkdir(parents=True, exist_ok=False)
        csv_path = folder / "result.csv"
        payload = submission_bytes_preserving_iron(
            parent_strings,
            prediction.sample_id.astype(str).tolist(),
            prediction.pred_tap_time_len.to_numpy(dtype=float),
        )
        with csv_path.open("xb") as handle:
            handle.write(payload)
        archive = folder / "Luqhhh_bf_tap_predict_prelim.zip"
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as handle:
            handle.write(csv_path, arcname="result.csv")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != payload:
                raise ContractError("v0.26 submission ZIP payload differs")
        reread = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        if len(reread) != registration()["test_a_rows"] or reread.sample_id.duplicated().any():
            raise ContractError("v0.26 submission coverage differs")
        if reread.pred_tap_iron.tolist() != parent_strings.pred_tap_iron.tolist():
            raise ContractError("v0.26 final iron strings differ from V21")
        numeric = reread[PREDICTION_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ContractError("v0.26 submission predictions invalid")
        receipts[key] = {
            "candidate": candidate,
            "rows": len(reread),
            "result_sha256": file_sha256(csv_path),
            "zip_sha256": file_sha256(archive),
            "iron_source_result_sha256": file_sha256(parent_path),
            "iron_csv_strings_exact": True,
            "platform_uploads": 0,
            "platform_feedback": None,
        }
    atomic_write_json(root / "packages_frozen_before_feedback.json", {
        "status": "PASS_TWO_COMPLETE_PACKAGES_FROZEN",
        "order": ["A", "B"],
        "receipts": receipts,
        "definitions_may_change_after_A_feedback": False,
        "third_candidate_generated": False,
        "platform_upload_budget": 2,
        "platform_uploads": 0,
    })


def cold(root: Path):
    if (root / "completion.json").exists() or (root / "cold_validation.json").exists():
        raise ContractError("never overwrite v0.26 cold/completion evidence")
    package = read_json(root / "packages_frozen_before_feedback.json")
    cold_receipts = {"old_qrf": {}, "candidates": {candidate: {} for candidate in CANDIDATES}}
    all_zero = True
    for slot in SLOTS:
        old_output = root / "cold" / "old_qrf" / f"{slot}.npz"
        worker("old-cold", "--root", root.resolve(), "--slot", slot, "--output", old_output.resolve())
        old_info = read_json(str(old_output) + ".json")
        cold_receipts["old_qrf"][str(slot)] = old_info
        all_zero &= not any(old_info["zero_fit"].values())
        for candidate in CANDIDATES:
            output = root / "cold" / candidate / f"{slot}.npz"
            worker("cold", "--root", root.resolve(), "--slot", slot, "--candidate", candidate, "--output", output.resolve())
            info = read_json(str(output) + ".json")
            cold_receipts["candidates"][candidate][str(slot)] = info
            all_zero &= not any(info["zero_fit"].values())
    if not all_zero:
        raise ContractError("v0.26 cold audit attempted a fit")
    parent_strings = pd.read_csv(V23 / "recovery/result.csv", dtype=str, keep_default_na=False)
    for key, candidate in CANDIDATES.items():
        receipt = package["receipts"][key]
        folder = root / "submissions" / candidate
        archive = folder / "Luqhhh_bf_tap_predict_prelim.zip"
        if file_sha256(archive) != receipt["zip_sha256"]:
            raise ContractError("v0.26 frozen ZIP identity changed")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != (folder / "result.csv").read_bytes():
                raise ContractError("v0.26 cold ZIP reread differs")
        reread = pd.read_csv(folder / "result.csv", dtype=str, keep_default_na=False)
        if reread.pred_tap_iron.tolist() != parent_strings.pred_tap_iron.tolist():
            raise ContractError("v0.26 cold iron string identity differs")
    attempted = {candidate: len(list((root / "models" / candidate).glob("*/fit_intent.json"))) for candidate in CANDIDATES}
    completed = {candidate: len(list((root / "models" / candidate).glob("*/fit_record.json"))) for candidate in CANDIDATES}
    if attempted != {"A": 7, "B": 7} or completed != attempted:
        raise ContractError("v0.26 fit budget/completion count differs")
    fit_records = [read_json(path) for path in (root / "models").glob("*/*/fit_record.json")]
    internal_trees = sum(int(record["internal_trees"]) for record in fit_records)
    if internal_trees != 3584:
        raise ContractError("v0.26 internal tree budget differs")
    atomic_write_json(root / "cold_validation.json", {
        "status": "PASS",
        "all_slots": list(SLOTS),
        "candidate_checks": "full_reverse_chunk_subset_single_exact",
        "old_qrf_support_recomputed_all_slots": True,
        "fit_attempts": {
            "RandomForestRegressor": 0,
            "ExtraTreesRegressor": 0,
            "PartitionForest": 0,
            "parent_QRF": 0,
            "preprocessor": 0,
            "calibration": 0,
        },
        "trusted_private_bundles_only": True,
        "zip_and_csv_reread": True,
        "receipts": cold_receipts,
    })
    atomic_write_json(root / "fit_counts.json", {
        "attempted": attempted,
        "completed": completed,
        "new_preprocessor_attempted": 0,
        "new_preprocessor_completed": 0,
        "internal_trees": internal_trees,
        "CatBoost_E04_rate_fits": 0,
        "LAD_beta_lambda_bias_fits": 0,
    })
    atomic_write_json(root / "completion.json", {
        "status": "READY_FOR_TWO_EXPLICIT_PLATFORM_SUBMISSIONS",
        "G0_engineering": "PASS",
        "G1_model_quality": "REPORTED_SEPARATELY_NOT_A_GATE",
        "candidates": list(CANDIDATES.values()),
        "platform_order": ["A", "B"],
        "platform_uploads": 0,
        "platform_budget": 2,
        "packages_sha256": file_sha256(root / "packages_frozen_before_feedback.json"),
        "cold_validation_sha256": file_sha256(root / "cold_validation.json"),
        "fit_counts_sha256": file_sha256(root / "fit_counts.json"),
        "test_targets_read": False,
        "official_raw_training_target_reads": False,
        "automatic_upload": False,
        "desktop_writes": 0,
        "public_pushes": 0,
        "third_candidate_generated": False,
    })


def run(root: Path):
    register(root)
    prepare(root)
    develop(root)
    score(root)
    finalize(root)
    cold(root)
