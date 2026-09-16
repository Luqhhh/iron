"""Registered v0.25 lifecycle for two history-centered target experiments."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json, file_identities, file_sha256, runtime_environment, stable_digest
from ..config import load_yaml
from ..exceptions import ContractError
from .component_export import read_json
from .dual_burden_v24_run import (
    DEVELOPMENT,
    FINAL_QRF,
    RECENCY_DEV,
    RECENCY_FINAL,
    V22,
    V23,
    _beta,
    _candidate_errors,
    _history,
    _iron_labels,
    _old_e04,
    _old_parts,
    _parent,
    _reference_errors,
    _replay,
    _score_rows,
    _summary,
    frame_from,
    load_original,
    original_handoff,
)
from .dual_ratio_common import week_intervals
from .history_centered_v25 import (
    CANDIDATE_A,
    CANDIDATE_B,
    CenteredRecencyIronModel,
    baseline_audit,
    compose_iron,
    history_baseline,
    isolate,
    roundtrip_six,
)
from .qrf_support_shrink import serialize_six_decimals
from .recency_model import weights
from .structural_run import append_ledger


WORKER = Path("workers/qrf_v025")
ORIGINS = range(6, 12)
PREDICTION_COLUMNS = ["pred_tap_iron", "pred_tap_time_len"]
DIAGNOSTIC_A = "D25I_BASELINE_ONLY"
DIAGNOSTIC_B = "D25T_BASELINE_ONLY"


def registration():
    value = load_yaml("configs/optimization_v0_25/experiment.yaml")
    if value["candidates"] != {"A": CANDIDATE_A, "B": CANDIDATE_B}:
        raise ContractError("v0.25 candidates differ")
    expected_budget = {
        "development_catboost_fits": 6,
        "final_catboost_fits": 1,
        "development_qrf_fits": 6,
        "final_qrf_fits": 1,
        "qrf_preprocessor_fits": 0,
        "qrf_internal_trees_total": 1792,
        "LAD_beta_lambda_bias_fits": 0,
        "new_candidate_packages": 2,
        "new_candidate_platform_tests": 2,
    }
    if value["budget"] != expected_budget or value["baseline"]["window"] != "last100":
        raise ContractError("v0.25 budget/baseline registration differs")
    return value


def worker(command, *arguments):
    executable = Path("workers/qrf_v015/.venv/bin/python")
    return subprocess.check_output([str(executable), str(WORKER / "worker.py"), command, *map(str, arguments)], text=True)


def _source_files():
    return [
        Path("configs/optimization_v0_25/experiment.yaml"),
        Path("configs/optimization_v0_25/access_scope.yaml"),
        Path("src/bf_tap/optimization/history_centered_v25.py"),
        Path("src/bf_tap/optimization/history_centered_v25_run.py"),
        Path("src/bf_tap/optimization/dual_burden_v24_run.py"),
        Path("scripts/optimization_v25_history_centered.py"),
        Path("scripts/optimization_v25_cold_check.py"),
        Path("tests/test_optimization_v25_history_centered.py"),
        Path("docs/optimization_v0_25/PLAN.md"),
        WORKER / "worker.py",
        WORKER / "signed_qrf.py",
        WORKER / "test_signed_qrf.py",
        Path("workers/qrf_v015/qrf_model.py"),
        Path("workers/qrf_v015/preprocessing.py"),
    ]


def _original_root(slot: int) -> Path:
    return FINAL_QRF if slot == 12 else DEVELOPMENT


def _original_preprocessor(slot: int) -> tuple[Path, dict]:
    root = _original_root(slot)
    path = root / f"models/{slot}/preprocessor.json"
    bundle = read_json(root / f"models/{slot}/bundle.json")
    if file_sha256(path) != bundle["preprocessor_sha256"]:
        raise ContractError("original QRF preprocessor identity differs")
    return path, bundle


def register(root: Path):
    configuration = registration()
    scope = load_yaml("configs/optimization_v0_25/access_scope.yaml")
    if root.exists():
        raise ContractError("never overwrite a v0.25 run")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ContractError("clean preregistered working tree required")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != configuration["branch"]:
        raise ContractError("registered v0.25 branch required")
    subprocess.run(["git", "merge-base", "--is-ancestor", configuration["base_commit"], "HEAD"], check=True)
    equivalents = []
    for path in Path("local/runs").glob("**/*manifest*.json"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if CANDIDATE_A in text or CANDIDATE_B in text:
            equivalents.append(str(path))
    if equivalents:
        raise ContractError("equivalent v0.25 experiment already exists: " + str(equivalents))
    paths = load_yaml("configs/data.local.yaml")["paths"]
    inputs = {key: paths[key] for key in ("train_samples", "tap_history_train", "test_a_samples", "data_dictionary")}
    evidence = {
        "v23_completion": V23 / "completion.json",
        "v23_v21_zip": V23 / "recovery/Luqhhh_bf_tap_predict_prelim_TGATE_SPOUT1.zip",
        "v23_v21_result": V23 / "recovery/result.csv",
        "v22_errors": V22 / "all_errors.csv",
        "recency_manifest": RECENCY_DEV / "manifest.json",
        "qrf_manifest": DEVELOPMENT / "manifest.json",
        "final_qrf_manifest": FINAL_QRF / "manifest.json",
        "final_recency_completion": RECENCY_FINAL / "completion.json",
    }
    manifest = {
        "registration": configuration,
        "scope": scope,
        "protection": load_yaml(scope["protection_contract"]),
        "inputs": file_identities(inputs),
        "evidence": file_identities(evidence),
        "sources": file_identities({str(path): path for path in _source_files()}),
        "worker_environment": json.loads(worker("environment")),
        "worker_sources": json.loads(worker("identity")),
        "qrf_parameters": configuration["qrf_parameters"],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_environment(),
        "equivalent_registry_matches": equivalents,
        "holdout_consumed": True,
        "test_targets_read": False,
        "new_preprocessor_fit_budget": 0,
        "platform_feedback_may_change_second_candidate": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    atomic_write_json(root / "manifest.json", manifest)
    append_ledger(scope, root)
    atomic_write_json(root / "registration.json", {
        "status": "FROZEN_BEFORE_FIT",
        "candidates": configuration["candidates"],
        "platform_order": configuration["platform_order"],
        "candidate_platform_budget": 2,
        "platform_uploads": 0,
        "v21_zip_sha256": file_sha256(V23 / "recovery/Luqhhh_bf_tap_predict_prelim_TGATE_SPOUT1.zip"),
    })
    return manifest


def _write_centered_handoff(root: Path, slot: int, kind: str):
    arrays, old = load_original(slot, kind)
    frame = frame_from(arrays, old)
    if len(frame.columns) != 210 or any(column.startswith("burden_lag__") for column in frame):
        raise ContractError("v0.25 requires original certified 210-column frame")
    baseline_iron, source_iron = history_baseline(frame, "tap_iron")
    baseline_time, source_time = history_baseline(frame, "tap_time_len")
    preprocessor_path, original_bundle = _original_preprocessor(slot)
    if kind == "train" and original_bundle["input"]["sha256"] != old["sha256"]:
        raise ContractError("original training handoff/model identity differs")
    output = root / f"features/{slot}/{kind}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ContractError("never overwrite centered handoff")
    packed = dict(arrays)
    packed.update(
        baseline_iron=baseline_iron,
        baseline_time=baseline_time,
        baseline_iron_source=source_iron,
        baseline_time_source=source_time,
    )
    np.savez(output, **packed)
    if not np.array_equal(packed["numeric"], arrays["numeric"], equal_nan=True):
        raise ContractError("original numeric feature values changed")
    iron_raw = _iron_labels(arrays).to_numpy(dtype=float) if kind == "train" else None
    time_raw = arrays["y"] if kind == "train" else None
    info = {
        "sha256": file_sha256(output),
        "ids": arrays["ids"].tolist(),
        "numeric_columns": old["numeric_columns"],
        "raw_schema": old["raw_schema"],
        "cutoff_ns": old["cutoff_ns"],
        "cutoff": old["cutoff"],
        "training": kind == "train",
        "original_input_path": str(original_handoff(slot, kind)),
        "original_input_sha256": old["sha256"],
        "original_schema_sha256": old["raw_schema_sha256"],
        "original_columns_unchanged": True,
        "model_input_column_count": len(frame.columns),
        "baseline_not_model_input": True,
        "original_preprocessor_path": str(preprocessor_path),
        "original_preprocessor_sha256": file_sha256(preprocessor_path),
        "baseline_audit": {
            "tap_iron": baseline_audit(baseline_iron, source_iron, iron_raw),
            "tap_time_len": baseline_audit(baseline_time, source_time, time_raw),
        },
    }
    atomic_write_json(str(output) + ".json", info)
    return info


def prepare(root: Path):
    if (root / "p0_complete.json").exists():
        raise ContractError("P0 already complete")
    handoffs = {}
    for slot in range(6, 13):
        handoffs[str(slot)] = {kind: _write_centered_handoff(root, slot, kind) for kind in ("train", "evaluation")}
        if len(handoffs[str(slot)]["train"]["ids"]) != registration()["expected_training_rows"][slot]:
            raise ContractError("certified training row count differs")
    atomic_write_json(root / "p0_complete.json", {
        "status": "PASS_ZERO_FIT",
        "handoffs": handoffs,
        "original_columns_unchanged": True,
        "model_input_columns": 210,
        "baseline_columns_added_to_model": 0,
        "preprocessor_fits": 0,
        "baseline_source_order": ["spout", "all", "zero"],
        "timezone": "Asia/Shanghai",
    })


def _centered(root: Path, slot: int, kind: str):
    path = root / f"features/{slot}/{kind}.npz"
    info = read_json(str(path) + ".json")
    if file_sha256(path) != info["sha256"]:
        raise ContractError("centered handoff identity changed")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    return arrays, info, frame_from(arrays, info)


def _fit_A(root: Path, slot: int):
    attempted = len(list((root / "models/A").glob("*/fit_intent.json")))
    if attempted >= 7:
        raise ContractError("seven-fit centered CatBoost budget exhausted")
    arrays, info, frame = _centered(root, slot, "train")
    cutoff = pd.Timestamp(info["cutoff"])
    metadata = pd.DataFrame({
        "sample_id": arrays["ids"].astype(str),
        "spout_no": arrays["spout"].astype(str),
        "reference_time": pd.to_datetime(arrays["reference_ns"], utc=True).tz_convert("Asia/Shanghai"),
        "available_at": pd.to_datetime(arrays["available_ns"], utc=True).tz_convert("Asia/Shanghai"),
    }).set_index("sample_id", drop=False)
    weight, _ = weights(metadata.reset_index(drop=True), cutoff)
    weight = weight.loc[frame.index]
    target = _iron_labels(arrays)
    baseline = arrays["baseline_iron"]
    signed = target.to_numpy(dtype=float) - baseline
    folder = root / f"models/A/{slot}"
    folder.mkdir(parents=True, exist_ok=False)
    common = {
        "slot": slot,
        "cutoff": str(cutoff),
        "rows": len(frame),
        "input_sha256": info["sha256"],
        "original_input_sha256": info["original_input_sha256"],
        "training_ids_sha256": stable_digest(arrays["ids"].tolist()),
        "raw_target_sha256": stable_digest(target.tolist()),
        "baseline_sha256": stable_digest(baseline.tolist()),
        "signed_target_sha256": stable_digest(signed.tolist()),
        "weight_sha256": stable_digest(weight.tolist()),
        "response_transform": "tap_iron_minus_legal_last100_baseline",
    }
    atomic_write_json(folder / "fit_intent.json", common)
    model = CenteredRecencyIronModel().fit(frame, target, baseline, weight)
    bundle = model.save(folder / "bundle", common)
    atomic_write_json(folder / "fit_record.json", {
        "status": "COMPLETED",
        **common,
        "bundle_sha256": file_sha256(folder / "bundle/bundle.json"),
        "baseline_audit": baseline_audit(baseline, arrays["baseline_iron_source"], target),
        "weight_audit": {
            "mean": float(weight.mean()),
            "min": float(weight.min()),
            "max": float(weight.max()),
            "ESS": float(weight.sum() ** 2 / np.dot(weight, weight)),
        },
    })
    return bundle


def _save_prediction(path: Path, value: pd.DataFrame):
    if path.exists():
        raise ContractError("never overwrite a v0.25 prediction")
    value.to_csv(path, index=False)


def _predict_slot(root: Path, slot: int):
    arrays, _, frame = _centered(root, slot, "evaluation")
    ids = arrays["ids"].tolist()
    parent = _parent(slot)
    if parent.sample_id.tolist() != ids:
        parent = parent.set_index("sample_id").loc[ids].reset_index()
    bundle_sha = file_sha256(root / f"models/A/{slot}/bundle/bundle.json")
    model = CenteredRecencyIronModel.load(root / f"models/A/{slot}/bundle", bundle_sha)
    direct = model.predict(frame, arrays["baseline_iron"])
    old = _old_parts(slot).set_index("sample_id").loc[ids]
    e04 = _old_e04(slot, frame, old)
    beta = _beta(slot)
    iron, a_audit = compose_iron(direct, e04, old.pred_rate, old.pred_tap_time_len, beta)
    candidate_a = isolate(parent, iron=iron, candidate=CANDIDATE_A)
    diagnostic_iron, _ = compose_iron(arrays["baseline_iron"], e04, old.pred_rate, old.pred_tap_time_len, beta)
    diagnostic_a = isolate(parent, iron=diagnostic_iron, candidate=CANDIDATE_A)

    qrf_path = root / f"worker_predictions/{slot}.npz"
    worker("predict", "--root", root.resolve(), "--slot", slot, "--output", qrf_path.resolve())
    with np.load(qrf_path, allow_pickle=False) as source:
        if source["ids"].tolist() != ids:
            raise ContractError("signed QRF prediction IDs differ")
        qrf_absolute = source["absolute"]
        signed_median = source["signed_median"]
    candidate_b_raw, b_audit = _replay(slot, qrf_absolute)
    candidate_b = isolate(parent, time=candidate_b_raw.pred_tap_time_len.to_numpy(), candidate=CANDIDATE_B)
    diagnostic_b_raw, _ = _replay(slot, arrays["baseline_time"])
    diagnostic_b = isolate(parent, time=diagnostic_b_raw.pred_tap_time_len.to_numpy(), candidate=CANDIDATE_B)
    old_replay, old_audit = _replay(slot)
    if serialize_six_decimals(old_replay) != serialize_six_decimals(parent):
        raise ContractError("old QRF substitution does not reproduce V21")

    destination = root / "predictions"
    destination.mkdir(exist_ok=True)
    _save_prediction(destination / f"{slot}_{CANDIDATE_A}.csv", candidate_a)
    _save_prediction(destination / f"{slot}_{CANDIDATE_B}.csv", candidate_b)
    _save_prediction(destination / f"{slot}_{DIAGNOSTIC_A}.csv", diagnostic_a)
    _save_prediction(destination / f"{slot}_{DIAGNOSTIC_B}.csv", diagnostic_b)
    atomic_write_json(destination / f"{slot}_audit.json", {
        "A": a_audit,
        "B": b_audit,
        "old_replay": old_audit,
        "A_unchanged_time_exact": bool(np.array_equal(candidate_a.pred_tap_time_len, parent.pred_tap_time_len)),
        "B_unchanged_iron_exact": bool(np.array_equal(candidate_b.pred_tap_iron, parent.pred_tap_iron)),
        "baseline_iron": baseline_audit(arrays["baseline_iron"], arrays["baseline_iron_source"]),
        "baseline_time": baseline_audit(arrays["baseline_time"], arrays["baseline_time_source"]),
        "signed_qrf_prediction_quantiles": {str(q): float(np.quantile(signed_median, q)) for q in (0, .05, .25, .5, .75, .95, 1)},
    })


def develop(root: Path):
    if (root / "development_complete.json").exists():
        raise ContractError("development already complete")
    for slot in ORIGINS:
        _fit_A(root, slot)
    for slot in ORIGINS:
        worker("fit", "--root", root.resolve(), "--slot", slot)
    atomic_write_json(root / "development_models_complete.json", {
        str(slot): file_sha256(root / f"models/B/{slot}/bundle.json") for slot in ORIGINS
    })
    for slot in ORIGINS:
        _predict_slot(root, slot)
    atomic_write_json(root / "development_complete.json", {
        "status": "PASS_G0_DEVELOPMENT_PREDICTIONS_FROZEN",
        "A_fits": 6,
        "B_forest_fits": 6,
        "B_preprocessor_fits": 0,
        "B_internal_trees": 1536,
        "calibration_fits": 0,
    })


def _diagnostic_errors(root: Path, diagnostic: str):
    return _candidate_errors(root, diagnostic)


def score(root: Path):
    candidates = {
        CANDIDATE_A: _candidate_errors(root, CANDIDATE_A),
        CANDIDATE_B: _candidate_errors(root, CANDIDATE_B),
    }
    references = {name: _reference_errors(name) for name in ("V1", "V10", "V21_REPLAY")}
    all_errors = pd.concat([*references.values(), *candidates.values()], ignore_index=True)
    all_errors.to_csv(root / "all_errors.csv", index=False)
    scorecard = pd.DataFrame([
        row for name, errors in {**references, **candidates}.items() for row in _score_rows(errors, name)
    ])
    scorecard.to_csv(root / "canonical_scorecard.csv", index=False)
    summary = _summary(scorecard)
    summary.to_csv(root / "canonical_summary.csv", index=False)
    reported = summary.loc[summary.scope_type.isin(("HORIZON_MEAN", "GRID", "DEVELOPMENT"))].copy()
    if reported.duplicated(["algorithm", "scope"]).any():
        raise ContractError("non-unique registered summary scope")
    lookup = reported.set_index(["algorithm", "scope"]).E.sort_index()
    deltas = []
    for candidate in candidates:
        for scope in reported.scope.unique():
            if (candidate, scope) not in lookup:
                continue
            deltas.append({
                "candidate": candidate,
                "scope": scope,
                "E": lookup[candidate, scope],
                "delta_vs_V1": lookup[candidate, scope] - lookup["V1", scope],
                "delta_vs_V21": lookup[candidate, scope] - lookup["V21_REPLAY", scope],
            })
    pd.DataFrame(deltas).to_csv(root / "reference_deltas.csv", index=False)
    bootstrap = {
        candidate: {reference: week_intervals(all_errors, candidate, reference, 1000, 2026)
                    for reference in ("V1", "V21_REPLAY")}
        for candidate in candidates
    }
    atomic_write_json(root / "bootstrap.json", bootstrap)
    diagnostics = {}
    for diagnostic in (DIAGNOSTIC_A, DIAGNOSTIC_B):
        errors = _diagnostic_errors(root, diagnostic)
        diagnostic_score = pd.DataFrame(_score_rows(errors, diagnostic))
        diagnostic_summary = _summary(diagnostic_score)
        diagnostics[diagnostic] = {
            row.scope: float(row.E)
            for row in diagnostic_summary.loc[diagnostic_summary.scope_type.isin(("HORIZON_MEAN", "GRID", "DEVELOPMENT"))].itertuples()
        }
    atomic_write_json(root / "baseline_only_diagnostic.json", {
        "status": "DIAGNOSTIC_ONLY_NOT_A_CANDIDATE",
        "new_fits": 0,
        "algorithms": diagnostics,
    })
    atomic_write_json(root / "offline_assessment.json", {
        "status": "COMPLETED_CONSUMED_RETROSPECTIVE_NOT_A_PLATFORM_GATE",
        "candidates": {
            candidate: {scope: float(lookup[candidate, scope]) for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")}
            for candidate in candidates
        },
        "references": {
            reference: {scope: float(lookup[reference, scope]) for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")}
            for reference in references
        },
        "both_candidates_retain_preregistered_platform_slot": True,
    })


def finalize(root: Path):
    if not (root / "offline_assessment.json").exists():
        raise ContractError("score development before final fits")
    completion = root / "final_models_complete.json"
    if completion.exists():
        expected = {"12": file_sha256(root / "models/B/12/bundle.json")}
        if read_json(completion) != expected or not (root / "models/A/12/fit_record.json").is_file():
            raise ContractError("partial final model completion identity differs")
    else:
        _fit_A(root, 12)
        worker("fit", "--root", root.resolve(), "--slot", 12)
        atomic_write_json(completion, {"12": file_sha256(root / "models/B/12/bundle.json")})
    _predict_slot(root, 12)
    parent = _parent(12)
    receipts = {}
    for key, candidate in (("A", CANDIDATE_A), ("B", CANDIDATE_B)):
        value = pd.read_csv(root / f"predictions/12_{candidate}.csv", dtype={"sample_id": str})
        folder = root / "submissions" / candidate
        folder.mkdir(parents=True, exist_ok=False)
        csv_path = folder / "result.csv"
        csv_path.write_bytes(serialize_six_decimals(value))
        archive = folder / "Luqhhh_bf_tap_predict_prelim.zip"
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as handle:
            handle.write(csv_path, arcname="result.csv")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != csv_path.read_bytes():
                raise ContractError("submission archive payload differs")
        reread = pd.read_csv(csv_path, dtype={"sample_id": str})
        unchanged = "pred_tap_time_len" if key == "A" else "pred_tap_iron"
        if not np.array_equal(reread[unchanged].to_numpy(), parent[unchanged].to_numpy()):
            raise ContractError("unchanged final target differs from V21 CSV")
        receipts[key] = {
            "candidate": candidate,
            "rows": len(value),
            "result_sha256": file_sha256(csv_path),
            "zip_sha256": file_sha256(archive),
            "unchanged_target": unchanged,
            "unchanged_target_csv_exact": True,
            "platform_uploads": 0,
            "platform_feedback": None,
        }
    atomic_write_json(root / "packages_frozen_before_feedback.json", {
        "status": "PASS_TWO_PACKAGES_FROZEN",
        "order": ["A", "B"],
        "receipts": receipts,
        "definitions_may_change_after_A_feedback": False,
        "platform_upload_budget": 2,
        "platform_uploads": 0,
    })


def cold(root: Path):
    arrays, _, frame = _centered(root, 12, "evaluation")
    bundle_sha = file_sha256(root / "models/A/12/bundle/bundle.json")
    model = CenteredRecencyIronModel.load(root / "models/A/12/bundle", bundle_sha)
    baseline = arrays["baseline_iron"]
    full = model.predict(frame, baseline)
    if not np.array_equal(model.predict(frame.iloc[::-1], baseline[::-1])[::-1], full):
        raise ContractError("centered CatBoost reverse inference differs")
    chunks = [model.predict(frame.iloc[index:index+73], baseline[index:index+73]) for index in range(0, len(frame), 73)]
    if not np.array_equal(np.concatenate(chunks), full):
        raise ContractError("centered CatBoost chunk inference differs")
    middle = len(frame) // 2
    if not np.array_equal(model.predict(frame.iloc[[middle]], baseline[[middle]]), full[[middle]]):
        raise ContractError("centered CatBoost single inference differs")
    cold_path = root / "cold/qrf.npz"
    worker("cold", "--root", root.resolve(), "--slot", 12, "--output", cold_path.resolve())
    atomic_write_json(root / "cold_validation.json", {
        "status": "PASS",
        "A_full_reverse_chunk_single_exact": True,
        "B_full_reverse_chunk_subset_single_exact": True,
        "fit_attempts": {"CatBoost": 0, "QRF": 0, "preprocessor": 0, "calibration": 0},
        "trusted_private_bundles_only": True,
    })
    atomic_write_json(root / "fit_counts.json", {
        "CatBoost_attempted": 7,
        "CatBoost_completed": 7,
        "QRF_attempted": 7,
        "QRF_completed": 7,
        "QRF_preprocessor_attempted": 0,
        "QRF_preprocessor_completed": 0,
        "QRF_internal_trees": 1792,
        "LAD_beta_lambda_bias": 0,
    })
    atomic_write_json(root / "completion.json", {
        "status": "READY_FOR_TWO_EXPLICIT_PLATFORM_SUBMISSIONS",
        "G0": "PASS",
        "G1": "REPORTED_SEPARATELY_NOT_A_GATE",
        "candidates": [CANDIDATE_A, CANDIDATE_B],
        "platform_order": ["A", "B"],
        "platform_uploads": 0,
        "platform_budget": 2,
        "packages_sha256": file_sha256(root / "packages_frozen_before_feedback.json"),
        "cold_validation_sha256": file_sha256(root / "cold_validation.json"),
        "fit_counts_sha256": file_sha256(root / "fit_counts.json"),
        "test_targets_read": False,
        "automatic_upload": False,
        "desktop_writes": 0,
        "public_pushes": 0,
    })


def run(root: Path):
    register(root)
    prepare(root)
    develop(root)
    score(root)
    finalize(root)
    cold(root)
