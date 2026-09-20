"""Registered zero-fit v0.35 lifecycle for time-leaf aggregation."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json, file_identities, file_sha256, runtime_environment
from ..config import load_yaml
from ..exceptions import ContractError
from .component_export import read_json
from .dual_burden_v24_run import _reference_errors
from .dual_ratio_common import week_intervals
from .oob_pool_v31_run import V26, V29, V30, _candidate_errors, _load_endpoint, _source_errors
from .qrf_partition_v26_run import _replay as replay_qrf, _score_rows
from .structural_run import append_ledger
from .time_leaf_v35 import (
    ATTACHMENT_PROTOCOL, CANDIDATE_A, CANDIDATE_B, PARENT, PARENT_TABLE_PROTOCOL,
    POOLED_AGGREGATION, PRIMARY_AGGREGATION, PROTOCOL, TARGETS, V1, V21_REPLAY, V30A,
    compose_time_candidate, exposure_pooled_summary, macro_origin_summary, submission_bytes,
    validate_endpoint, verify_isolated_deltas,
)


WORKER = Path("workers/qrf_v035/worker.py")
WORKER_PYTHON = Path("workers/qrf_v015/.venv/bin/python")
V34 = Path("local/runs/optimization-v0.34-oob-leaf-median-bagging-r2")
PARENT_RESULT = V34 / "submissions" / PARENT / "result.csv"
PARENT_ZIP = V34 / "submissions" / PARENT / "Luqhhh_bf_tap_predict_prelim.zip"
ORIGINS = range(6, 12)
SLOTS = range(6, 13)
CANDIDATES = {"A": CANDIDATE_A, "B": CANDIDATE_B}
EXPECTED_ROWS = {6: 888, 7: 1180, 8: 1490, 9: 1803, 10: 2091, 11: 2424, 12: 2754}


def registration():
    value = load_yaml("configs/optimization_v0_35/experiment.yaml")
    if value["branch"] != "optimization-v0.35-time-leaf-location-and-vote" or value["base_commit"] != "6eee2e77e0d1393ea1e2f148eb230fa93a54ae4b":
        raise ContractError("v0.35 branch/base registration differs")
    if value["candidates"] != CANDIDATES or value["parent"] != PARENT or value["training_rows"] != EXPECTED_ROWS:
        raise ContractError("v0.35 candidate/parent/training identity differs")
    if value["protocol"] != PROTOCOL or value["oob_attachment_protocol"] != ATTACHMENT_PROTOCOL or value["parent_table_protocol"] != PARENT_TABLE_PROTOCOL:
        raise ContractError("v0.35 protocol registration differs")
    if value["candidate_A"] != {
        "within_leaf": "median_interval_lower_and_upper_from_same_selected_members",
        "cross_tree": "exact_sum_of_256_lower_plus_256_upper_binary64_values_divided_by_512",
        "persist": "seven_new_midpoint_tables",
        "invariants": ["upper_gte_lower", "odd_lower_eq_upper", "raw_gte_parent", "final_time_gte_parent"],
    }:
        raise ContractError("v0.35 A definition differs")
    if value["candidate_B"] != {
        "within_leaf": "exact_parent_v034_lower_median_raw",
        "cross_tree": "smaller_middle_value_index_127_of_256_tree_points",
        "persist": "reuse_seven_parent_lower_tables", "numpy_median": "forbidden",
    }:
        raise ContractError("v0.35 B definition differs")
    expected_budget = {
        "new_model_or_tree_fits": 0, "new_preprocessing_catboost_calibration_fits": 0,
        "new_midpoint_tables_A_development": 6, "new_midpoint_tables_A_final": 1, "new_midpoint_tables_A_total": 7,
        "reused_lower_tables_B_development": 6, "reused_lower_tables_B_final": 1, "reused_lower_tables_B_total": 7,
        "new_lower_tables_B_total": 0, "unique_v029_time_attachment_sources": 7, "new_read_protocol_manifests": 14,
        "new_candidate_packages": 2, "new_candidate_platform_tests": 2, "automatic_uploads": 0,
        "restore_uploads": 0, "desktop_writes": 0, "third_candidate": 0,
    }
    if value["budget"] != expected_budget or value["platform_order"] != ["A", "B"] or value["platform_feedback_may_change_second_candidate"] is not False:
        raise ContractError("v0.35 budget/order/freeze registration differs")
    return value


def _source_files():
    return [
        Path("configs/optimization_v0_35/experiment.yaml"), Path("configs/optimization_v0_35/access_scope.yaml"),
        Path("src/bf_tap/optimization/time_leaf_v35.py"), Path("src/bf_tap/optimization/time_leaf_v35_run.py"),
        Path("scripts/optimization_v35_time_leaf.py"), Path("scripts/optimization_v35_cold_check.py"),
        Path("tests/test_optimization_v35_time_leaf.py"), Path("docs/optimization_v0_35/PLAN.md"),
        WORKER, Path("workers/qrf_v035/time_leaf_location_vote.py"), Path("workers/qrf_v035/test_time_leaf_location_vote.py"),
        Path("workers/qrf_v034/worker.py"), Path("workers/qrf_v034/leaf_point_bagging.py"),
        Path("workers/qrf_v029/worker.py"), Path("workers/qrf_v029/oob_response.py"),
        Path("workers/qrf_v026/worker.py"), Path("workers/qrf_v026/partition_forest.py"),
        Path("workers/qrf_v015/qrf_model.py"), Path("workers/qrf_v015/preprocessing.py"), Path("workers/qrf_v015/uv.lock"),
    ]


def _worker(command, *arguments):
    return subprocess.check_output([str(WORKER_PYTHON), str(WORKER), command, *map(str, arguments)], text=True)


def _parent_endpoint(slot):
    path = PARENT_RESULT if slot == 12 else V34 / "predictions" / f"{slot}_{PARENT}.csv"
    if not path.is_file(): raise ContractError(f"missing v0.35 parent endpoint for slot {slot}")
    return validate_endpoint(pd.read_csv(path, dtype=str, keep_default_na=False), label=f"{PARENT}/{slot}")


def _registry_search():
    manifests = sorted(Path("local/runs").glob("**/*manifest*.json")); matches = []
    for path in manifests:
        text = path.read_text(encoding="utf-8", errors="ignore")
        reasons = [token for token in (*CANDIDATES.values(), PROTOCOL) if token in text]
        if reasons: matches.append({"path": str(path), "sha256": file_sha256(path), "matches": reasons})
    completed, incomplete = [], []
    for match in matches:
        root = Path(match["path"]).parent; (completed if (root / "completion.json").is_file() else incomplete).append(match)
    return {"searched_manifest_count": len(manifests), "equivalent_matches": completed, "incomplete_equivalent_attempts": incomplete}


def register(root: Path):
    config = registration(); scope = load_yaml("configs/optimization_v0_35/access_scope.yaml")
    if root.exists(): raise ContractError("never overwrite a v0.35 run")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip(): raise ContractError("clean preregistered working tree required")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != config["branch"]: raise ContractError("registered v0.35 branch required")
    subprocess.run(["git", "merge-base", "--is-ancestor", config["base_commit"], "HEAD"], check=True)
    registry = _registry_search()
    if registry["equivalent_matches"]: raise ContractError("equivalent completed v0.35 experiment already exists")
    identities = config["source_identities"]
    if file_sha256(PARENT_RESULT) != identities["parent_result_sha256"] or file_sha256(PARENT_ZIP) != identities["parent_zip_sha256"]:
        raise ContractError("v0.35 parent result/ZIP identity differs")
    evidence = {
        "v26_manifest": V26 / "manifest.json", "v26_completion": V26 / "completion.json",
        "v29_manifest": V29 / "manifest.json", "v29_completion": V29 / "completion.json",
        "v34_manifest": V34 / "manifest.json", "v34_completion": V34 / "completion.json", "v34_cold": V34 / "cold_validation.json",
        "v34_errors": V34 / "all_errors.csv", "parent_result": PARENT_RESULT, "parent_zip": PARENT_ZIP,
        "v34_packages": V34 / "packages_frozen_before_feedback.json", "v34_feedback": V34 / "platform_feedback_complete.json",
        "protection": Path(scope["protection_contract"]),
    }
    for slot in SLOTS:
        evidence[f"v34_lower_table_{slot}"] = V34 / "leaf_point_tables/B" / f"{slot}.npz"
        evidence[f"v34_lower_table_{slot}_metadata"] = V34 / "leaf_point_tables/B" / f"{slot}.npz.json"
        evidence[f"v34_worker_prediction_{slot}"] = V34 / "worker_predictions/B" / f"{slot}.npz"
        evidence[f"v34_worker_prediction_{slot}_metadata"] = V34 / "worker_predictions/B" / f"{slot}.npz.json"
        evidence[f"v29_attachment_B_{slot}"] = V29 / "oob_attachments/B" / f"{slot}.npz"
        evidence[f"parent_endpoint_{slot}"] = PARENT_RESULT if slot == 12 else V34 / "predictions" / f"{slot}_{PARENT}.csv"
    missing = [name for name, path in evidence.items() if not Path(path).is_file()]
    if missing: raise ContractError(f"missing v0.35 source evidence: {missing}")
    environment, sources, worker_registration = json.loads(_worker("environment")), json.loads(_worker("identity")), json.loads(_worker("registration"))
    if environment.get("python") != "3.12.12" or environment.get("sklearn") != "1.8.0": raise ContractError("locked v0.35 worker environment required")
    if worker_registration["protocol"] != PROTOCOL: raise ContractError("v0.35 worker registration differs")
    manifest = {
        "registration": config, "scope": scope, "protection": load_yaml(scope["protection_contract"]),
        "evidence": file_identities(evidence), "sources": file_identities({str(path): path for path in _source_files()}),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "runtime": runtime_environment(),
        "worker_environment": environment, "worker_sources": sources, "worker_registration": worker_registration,
        "source_runs": {"v26": str(V26.resolve()), "v29": str(V29.resolve()), "v30": str(V30.resolve()), "v34": str(V34.resolve())},
        "registry_search": registry, "holdout_consumed": True, "test_targets_read": False,
        "new_fit_budget": 0, "platform_feedback_may_change_second_candidate": False,
    }
    root.mkdir(parents=True, exist_ok=False); root.chmod(0o700); atomic_write_json(root / "manifest.json", manifest)
    append_ledger(scope, root)
    atomic_write_json(root / "registration.json", {
        "status": "TWO_ZERO_FIT_TIME_LEAF_CANDIDATES_FROZEN_BEFORE_DERIVATION", "candidates": CANDIDATES,
        "definitions": {key: config[f"candidate_{key}"] for key in CANDIDATES}, "parent_regression": config["parent_regression"],
        "platform_order": ["A", "B"], "candidate_platform_budget": 2, "platform_uploads": 0,
        "definitions_may_change_after_A_feedback": False, "third_candidate_allowed": False,
    })


def _worker_arrays(root, slot, candidate):
    path = root / "worker_predictions" / candidate / f"{slot}.npz"
    with np.load(path, allow_pickle=False) as source: arrays = {name: source[name] for name in source.files}
    if not {"ids", "raw", "parent_raw", "raw_delta"}.issubset(arrays): raise ContractError("v0.35 worker fields differ")
    return arrays


def _parent_regression(slot, parent, parent_raw):
    replayed, audit = replay_qrf(V26, slot, parent_raw)
    recomposed = compose_time_candidate(parent, replayed.pred_tap_time_len.to_numpy(float), label=CANDIDATE_A)
    if not recomposed.equals(parent): raise ContractError("v0.35 parent raw does not recover V34T")
    return audit


def prepare(root: Path):
    if (root / "p0_complete.json").exists(): raise ContractError("v0.35 P0 already complete")
    audits = {}
    for slot in SLOTS:
        parent = _parent_endpoint(slot); audits[str(slot)] = {}
        for candidate in CANDIDATES:
            receipt = json.loads(_worker("audit", "--root", root.resolve(), "--slot", slot, "--candidate", candidate))
            if any(receipt["zero_fit"].values()) or not receipt["parent_raw_regression_exact"]: raise ContractError("v0.35 P0 audit differs")
            audits[str(slot)][candidate] = receipt
        with np.load(V34 / "worker_predictions/B" / f"{slot}.npz", allow_pickle=False) as source: parent_raw = source["point_mean"]
        audits[str(slot)]["final_parent_replay"] = _parent_regression(slot, parent, parent_raw)
    atomic_write_json(root / "p0_complete.json", {
        "status": "PASS_ZERO_FIT_V34T_PARENT_V26_FOREST_V029_ATTACHMENT_V34_LOWER_TABLE_AUDIT",
        "audits": audits, "registry_search": read_json(root / "manifest.json")["registry_search"],
        "new_model_fits": 0, "new_tree_fits": 0, "new_preprocessor_fits": 0, "new_calibration_fits": 0, "test_targets_read": False,
    })


def _write_prediction(path, frame):
    if path.exists(): raise ContractError("never overwrite v0.35 prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle: handle.write(submission_bytes(frame))


def _compose_slot(root: Path, slot: int, *, derive: bool, write: bool):
    parent = _parent_endpoint(slot); frames, audits = {}, {}
    for candidate, label in CANDIDATES.items():
        output = root / "worker_predictions" / candidate / f"{slot}.npz"
        if derive and not output.is_file(): _worker("derive-predict", "--root", root.resolve(), "--slot", slot, "--candidate", candidate, "--output", output.resolve())
        arrays = _worker_arrays(root, slot, candidate)
        if arrays["ids"].tolist() != parent.sample_id.tolist(): raise ContractError("v0.35 worker/parent IDs differ")
        replayed, replay_audit = replay_qrf(V26, slot, arrays["raw"])
        frame = compose_time_candidate(parent, replayed.pred_tap_time_len.to_numpy(float), label=label)
        parent_replayed, parent_audit = replay_qrf(V26, slot, arrays["parent_raw"])
        parent_frame = compose_time_candidate(parent, parent_replayed.pred_tap_time_len.to_numpy(float), label=label)
        if not parent_frame.equals(parent): raise ContractError("v0.35 parent switch-back differs")
        if frame.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist(): raise ContractError("v0.35 changed parent iron")
        if candidate == "A" and (frame.pred_tap_time_len.astype(float).to_numpy() < parent.pred_tap_time_len.astype(float).to_numpy()).any():
            raise ContractError("v0.35 A final monotonicity failed")
        frames[candidate] = frame
        receipt = read_json(str(output) + ".json")
        if receipt["rows"] != len(parent): raise ContractError("v0.35 worker receipt row count differs")
        monotone = candidate != "A" or bool(np.all(frame.pred_tap_time_len.astype(float).to_numpy() >= parent.pred_tap_time_len.astype(float).to_numpy()))
        audits[candidate] = {"candidate_id": label, "replay": replay_audit, "parent_replay": parent_audit,
                             "parent_switch_back_exact": True, "unchanged_iron_exact": True,
                             "A_final_gte_parent": monotone, "worker_receipt": receipt}
        if write: _write_prediction(root / "predictions" / f"{slot}_{label}.csv", frame)
    if write: atomic_write_json(root / "predictions" / f"{slot}_audit.json", audits)
    return {"parent": parent, **frames}


def develop(root: Path):
    if (root / "development_complete.json").exists(): raise ContractError("v0.35 development already complete")
    for slot in ORIGINS: _compose_slot(root, slot, derive=True, write=True)
    predictions = {f"{slot}_{key}": root / "predictions" / f"{slot}_{label}.csv" for slot in ORIGINS for key, label in CANDIDATES.items()}
    midpoint = {str(slot): root / "midpoint_tables/A" / f"{slot}.npz" for slot in ORIGINS}
    parent_lower = {str(slot): V34 / "leaf_point_tables/B" / f"{slot}.npz" for slot in ORIGINS}
    atomic_write_json(root / "development_predictions_frozen.json", {
        "status": "BOTH_CANDIDATES_A_MIDPOINT_TABLES_B_PARENT_LOWER_READERS_AND_PREDICTIONS_FROZEN_BEFORE_SCORING",
        "predictions": file_identities(predictions), "A_midpoint_tables": file_identities(midpoint),
        "B_reused_parent_lower_tables": file_identities(parent_lower), "platform_feedback_seen": False,
    })
    atomic_write_json(root / "development_complete.json", {
        "status": "PASS_G0_DEVELOPMENT_ZERO_FIT", "new_model_fits": 0, "new_tree_fits": 0,
        "A_new_midpoint_tables": 6, "B_new_lower_tables": 0, "B_reused_parent_lower_tables": 6,
    })


def _write_report(root, deltas):
    lines = ["# optimization-v0.35 自动生成离线报告", "", "主口径为 `macro_origin_mean_wmape`；负 Δ 表示历史误差改善。历史标签已消费。", "",
             "| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for candidate in CANDIDATES.values():
        for reference in deltas.reference.unique():
            part = deltas[(deltas.candidate == candidate) & (deltas.reference == reference)].set_index("scope")
            values = [part.loc[scope, "delta_E"] for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")]
            lines.append(f"| {candidate} | {reference} | " + " | ".join(f"{value:+.8f}" for value in values) + " |")
    with (root / "REPORT.md").open("x", encoding="utf-8") as handle: handle.write("\n".join(lines) + "\n")


def score(root: Path):
    if (root / "offline_assessment.json").exists(): raise ContractError("never overwrite v0.35 scoring evidence")
    candidates = {label: _candidate_errors(root, label) for label in CANDIDATES.values()}
    references = {V1: _reference_errors(V1), V21_REPLAY: _reference_errors(V21_REPLAY),
                  V30A: _source_errors(V30 / "all_errors.csv", V30A, V30A),
                  PARENT: _source_errors(V34 / "all_errors.csv", PARENT, PARENT)}
    all_errors = pd.concat([*references.values(), *candidates.values()], ignore_index=True); all_errors.to_csv(root / "all_errors.csv", index=False)
    scorecard = pd.DataFrame([row for name, errors in {**references, **candidates}.items() for row in _score_rows(errors, name)]); scorecard.to_csv(root / "canonical_scorecard.csv", index=False)
    summary = macro_origin_summary(scorecard); summary.to_csv(root / "canonical_summary.csv", index=False)
    pooled = exposure_pooled_summary(scorecard); pooled.to_csv(root / "exposure_pooled_summary.csv", index=False)
    isolation = verify_isolated_deltas(scorecard, summary, {label: "tap_time_len" for label in candidates}, parent=PARENT)
    primary = summary[summary.scope_type.isin(("HORIZON_MEAN", "GRID", "DEVELOPMENT"))].set_index(["algorithm", "scope"]); rows = []
    for candidate in candidates:
        for reference in references:
            for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT"):
                left, right = primary.loc[(candidate, scope)], primary.loc[(reference, scope)]
                rows.append({"candidate": candidate, "reference": reference, "scope": scope, "aggregation": PRIMARY_AGGREGATION,
                             "E": float(left.E), "delta_E": float(left.E - right.E), "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                             "delta_wmape_time": float(left.wmape_time - right.wmape_time)})
    deltas = pd.DataFrame(rows); deltas.to_csv(root / "canonical_deltas.csv", index=False)
    pooled_index = pooled.set_index(["algorithm", "scope"]); pooled_rows = []
    for candidate in candidates:
        for reference in references:
            for scope in ("H1", "H2", "H3", "H4", "DEV_LONG", "DEV_SHORT"):
                left, right = pooled_index.loc[(candidate, scope)], pooled_index.loc[(reference, scope)]
                pooled_rows.append({"candidate": candidate, "reference": reference, "scope": scope, "aggregation": POOLED_AGGREGATION,
                                    "delta_E": float(left.E - right.E), "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                                    "delta_wmape_time": float(left.wmape_time - right.wmape_time)})
    pd.DataFrame(pooled_rows).to_csv(root / "exposure_pooled_deltas.csv", index=False)
    atomic_write_json(root / "bootstrap.json", {candidate: {reference: week_intervals(all_errors, candidate, reference, 1000, 2026) for reference in references} for candidate in candidates})
    atomic_write_json(root / "aggregation_diagnostics.json", {candidate: {str(slot): read_json(str(root / "worker_predictions" / candidate / f"{slot}.npz") + ".json") for slot in ORIGINS} for candidate in CANDIDATES})
    _write_report(root, deltas)
    atomic_write_json(root / "offline_assessment.json", {
        "status": "COMPLETED_CONSUMED_RETROSPECTIVE_NOT_A_PLATFORM_GATE", "primary_aggregation": PRIMARY_AGGREGATION,
        "diagnostic_aggregation": POOLED_AGGREGATION, "isolated_target_audit": isolation,
        "historical_labels_consumed": True, "both_candidates_retain_preregistered_platform_slot": True,
    })


def finalize(root: Path):
    if not (root / "offline_assessment.json").is_file() or (root / "packages_frozen_before_feedback.json").exists(): raise ContractError("score v0.35 before finalization")
    composed = _compose_slot(root, 12, derive=True, write=True); parent, receipts = composed["parent"], {}
    for key, candidate in CANDIDATES.items():
        prediction = validate_endpoint(pd.read_csv(root / "predictions" / f"12_{candidate}.csv", dtype=str, keep_default_na=False), label=candidate)
        if len(prediction) != registration()["test_a_rows"] or prediction.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist(): raise ContractError("v0.35 final endpoint identity differs")
        duplicate_parent = prediction.equals(parent)
        other = CANDIDATES["A"] if key == "B" else None
        duplicate_other = key == "B" and prediction.equals(validate_endpoint(pd.read_csv(root / "predictions" / f"12_{other}.csv", dtype=str, keep_default_na=False), label=other))
        if duplicate_parent or duplicate_other:
            receipts[key] = {"candidate": candidate, "status": "NO_OP_OR_DUPLICATE_FINAL_PAYLOAD", "rows": len(prediction), "package_created": False,
                             "duplicate_parent": duplicate_parent, "duplicate_other": duplicate_other}; continue
        folder = root / "submissions" / candidate; folder.mkdir(parents=True, exist_ok=False)
        csv_path = folder / "result.csv"; payload = submission_bytes(prediction)
        with csv_path.open("xb") as handle: handle.write(payload)
        archive = folder / "Luqhhh_bf_tap_predict_prelim.zip"
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as handle: handle.write(csv_path, arcname="result.csv")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != payload: raise ContractError("v0.35 ZIP differs")
        receipts[key] = {"candidate": candidate, "status": "FROZEN", "rows": len(prediction), "package_created": True,
                         "result_sha256": file_sha256(csv_path), "zip_sha256": file_sha256(archive), "unchanged_iron_exact_parent": True,
                         "platform_uploads": 0, "platform_feedback": None}
    count = sum(item["package_created"] for item in receipts.values())
    atomic_write_json(root / "packages_frozen_before_feedback.json", {
        "status": "PASS_VALID_PACKAGES_FROZEN", "order": ["A", "B"], "receipts": receipts, "package_count": count,
        "duplicate_count": 2 - count, "definitions_may_change_after_A_feedback": False, "third_candidate_generated": False,
        "platform_upload_budget": count, "platform_uploads": 0, "automatic_upload": False, "desktop_writes": 0,
    })


def _assert_frame(actual, expected, message):
    if not validate_endpoint(actual, label="actual").reset_index(drop=True).equals(validate_endpoint(expected, label="expected").reset_index(drop=True)): raise ContractError(message)


def cold(root: Path):
    if (root / "cold_validation.json").exists() or (root / "completion.json").exists(): raise ContractError("never overwrite v0.35 cold evidence")
    receipts, zero_fit = {"A": {}, "B": {}}, True
    for slot in SLOTS:
        for candidate in CANDIDATES:
            output = root / "cold" / candidate / f"{slot}.npz"; output.parent.mkdir(parents=True, exist_ok=True)
            receipt = json.loads(_worker("cold", "--root", root.resolve(), "--slot", slot, "--candidate", candidate, "--output", output.resolve()))
            receipts[candidate][str(slot)] = receipt; zero_fit &= not any(receipt["zero_fit"].values())
        composed = _compose_slot(root, slot, derive=False, write=False)
        for key, label in CANDIDATES.items(): _assert_frame(composed[key], pd.read_csv(root / "predictions" / f"{slot}_{label}.csv", dtype=str, keep_default_na=False), f"v0.35 cold {key} differs")
    if not zero_fit: raise ContractError("v0.35 cold attempted fit")
    package = read_json(root / "packages_frozen_before_feedback.json")
    for key, candidate in CANDIDATES.items():
        receipt = package["receipts"][key]
        if not receipt["package_created"]: continue
        folder = root / "submissions" / candidate; csv_path, archive = folder / "result.csv", folder / "Luqhhh_bf_tap_predict_prelim.zip"
        if file_sha256(csv_path) != receipt["result_sha256"] or file_sha256(archive) != receipt["zip_sha256"]: raise ContractError("v0.35 package identity changed")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != csv_path.read_bytes(): raise ContractError("v0.35 ZIP reread differs")
    atomic_write_json(root / "fit_counts.json", {
        "new_model_attempted": 0, "new_tree_attempted": 0, "new_preprocessor_attempted": 0, "new_calibration_attempted": 0,
        "A_new_supervised_midpoint_tables": 7, "B_new_supervised_lower_tables": 0, "B_reused_parent_lower_tables": 7,
        "unique_v029_time_attachment_sources": 7, "new_read_protocol_manifests": 14,
        "new_candidate_packages": package["package_count"], "platform_uploads": 0, "third_candidate_generated": False,
    })
    atomic_write_json(root / "cold_validation.json", {
        "status": "PASS", "all_slots": list(SLOTS), "source_forest_response_bootstrap_v029_members_unchanged": True,
        "A_midpoint_tables_rederived_exact": True, "B_parent_lower_tables_rederived_exact": True,
        "parent_raw_and_final_switch_back_exact": True, "full_reverse_chunk_subset_single_exact": True,
        "A_raw_and_final_gte_parent": True, "fit_attempts_during_cold": {"forest": 0, "tree": 0, "preprocessor": 0, "CatBoost": 0, "calibration": 0},
        "zip_and_csv_reread": True, "receipts": receipts,
    })
    atomic_write_json(root / "completion.json", {
        "status": "READY_FOR_EXPLICIT_PLATFORM_SUBMISSIONS", "G0_engineering": "PASS", "G1_model_quality": "REPORTED_SEPARATELY_NOT_A_GATE",
        "candidates": list(CANDIDATES.values()), "parent": PARENT, "platform_order": ["A", "B"],
        "platform_uploads": 0, "platform_budget": package["package_count"], "packages_sha256": file_sha256(root / "packages_frozen_before_feedback.json"),
        "cold_validation_sha256": file_sha256(root / "cold_validation.json"), "fit_counts_sha256": file_sha256(root / "fit_counts.json"),
        "test_targets_read": False, "automatic_upload": False, "desktop_writes": 0, "third_candidate_generated": False,
    })


def run(root: Path):
    register(root); prepare(root); develop(root); score(root); finalize(root); cold(root)
