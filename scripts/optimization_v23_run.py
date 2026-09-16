#!/usr/bin/env python3
"""Run v0.23 recovery, canonical comparison, diagnostics, and stage preview."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from bf_tap.artifacts import atomic_write_json, file_identities, file_sha256, runtime_environment, stable_digest
from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.v23_recovery import (
    FIXED_ALGORITHMS, M_ONLY, V1, V10, V21, V22,
    _lambda_by_origin, canonical_errors, diagnostic_table, load_directions,
    historical_v21_replay_audit, median_only_errors, monthly_bias_table, read_json, reference_deltas,
    score_summary, scorecard, shrinkage_strength, stage_metadata,
    training_history_from_handoff,
)
from bf_tap.optimization.qrf_support_shrink import replay_v21, serialize_six_decimals
REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/optimization_v0_23/experiment.yaml"
SCOPE = REPO / "configs/optimization_v0_23/access_scope.yaml"
PROTECTION = REPO / "configs/protection.yaml"
V22_DEV = REPO / "local/runs/optimization-v0.22-causal-h2-r8"
V22_FINAL = REPO / "local/runs/optimization-v0.22-test-a-platform-r1"
PROBE = REPO / "local/runs/platform-probes-r2-r1"
QRF_SOURCE = REPO / "local/runs/optimization-v0.15-v8-user-test-a-r1"

_verify_spec = importlib.util.spec_from_file_location("verify_v21_payload", REPO / "verify_v21_payload.py")
if _verify_spec is None or _verify_spec.loader is None:
    raise RuntimeError("cannot load verify_v21_payload.py")
_verify_module = importlib.util.module_from_spec(_verify_spec)
_verify_spec.loader.exec_module(_verify_module)
recover_payload = _verify_module.recover_payload


def registration() -> dict:
    value = load_yaml(CONFIG)
    expected_budget = {
        "forest_fits": 0, "catboost_or_other_model_fits": 0, "preprocessor_fits": 0,
        "LAD_or_lambda_fits": 0, "new_median_estimates": 0, "new_candidates": 0,
        "v21_recovery_archives_max": 1, "test_a_uploads": 0, "platform_uploads": 0,
        "desktop_writes": 0, "public_pushes": 0,
    }
    if (
        value.get("kind") != "recovery_comparison_and_stage_validation_not_a_prediction_candidate"
        or value.get("base_commit") != "3524fcbb0455092c868f1fa22f217084732059ff"
        or value.get("fixed_algorithms") != list(FIXED_ALGORITHMS)
        or value.get("budget") != expected_budget
    ):
        raise ContractError("frozen v0.23 registration differs")
    return value


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _ledger(root: Path, purpose: str, *, protected_labels_read: bool) -> None:
    scope = load_yaml(SCOPE)
    protection = load_yaml(PROTECTION)
    if protection.get("contract_id") != "holdout-protection-v1":
        raise ContractError("protected-label contract differs")
    ledger = REPO / scope["new_ledger"]
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        prior = handle.read().splitlines()
        event = {
            "at": datetime.now(timezone.utc).isoformat(),
            "purpose": purpose,
            "authorization": scope["authorization"],
            "run_manifest_sha256": file_sha256(root / "manifest.json"),
            "protection_sha256": file_sha256(PROTECTION),
            "previous_sha256": stable_digest(prior[-1]) if prior else None,
            "protected_labels_read": protected_labels_read,
        }
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _write_csv(path: Path, value: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value.to_csv(path, index=False, mode="x", float_format="%.17g", lineterminator="\n")


def _source_paths(v21_zip: Path) -> tuple[list[Path], list[Path], dict]:
    test_b = PROBE / "test_b_V10_full_precision.csv.evidence"
    model_dir = QRF_SOURCE / "models/12"
    history_manifest = V22_DEV / "history_input_manifest.json"
    history_registration = read_json(history_manifest)
    historical_history = Path(history_registration.get("history", {}).get("path", "")).resolve()
    if (
        history_registration.get("kind") != "V22_FROZEN_HISTORY_INPUT_v1"
        or not historical_history.is_relative_to((REPO / "local/runs").resolve())
        or not historical_history.is_file()
        or history_registration.get("history", {}).get("sha256") != file_sha256(historical_history)
    ):
        raise ContractError("historical V21 replay history registration differs")
    sources = [
        CONFIG, SCOPE, PROTECTION, REPO / "verify_v21_payload.py",
        REPO / "src/bf_tap/optimization/v23_recovery.py",
        REPO / "src/bf_tap/optimization/qrf_support_shrink.py",
        REPO / "scripts/optimization_v23_run.py",
        REPO / "scripts/optimization_v23_cold_check.py",
        REPO / "scripts/optimization_v23_stage_infer.py",
        REPO / "workers/qrf_v015/inference_v022.py",
        REPO / "workers/qrf_v015/qrf_model.py",
        REPO / "workers/qrf_v015/preprocessing.py",
        REPO / "uv.lock", REPO / "workers/qrf_v015/uv.lock",
    ]
    inputs = [
        v21_zip,
        V22_DEV / "p0/v21_rule_replay_result.csv",
        V22_DEV / "all_errors.csv",
        V22_DEV / "completion.json",
        V22_DEV / "acceptance.json",
        V22_DEV / "cold_validation.json",
        V22_DEV / "fit_counts.json",
        V22_DEV / "metrics.json",
        V22_DEV / "summary.json",
        history_manifest,
        historical_history,
        V22_DEV / "median_parameters.json",
        V22_DEV / "lambda_parameters.json",
        V22_FINAL / "manifest.json",
        V22_FINAL / "final_model.json",
        V22_FINAL / "median_parameters.json",
        V22_FINAL / "lambda_parameters.json",
        V22_FINAL / "cold_validation.json",
        V22_FINAL / "completion.json",
        V22_FINAL / "qrf.npz",
        V22_FINAL / "qrf.npz.json",
        PROBE / "component_bundle.json",
        REPO / "local/runs/optimization-v0.15-v10-target-composition-r1/submission/result.csv",
        PROBE / "test_b_V10_full_precision.csv",
        test_b / "V1_full_precision.csv",
        test_b / "evaluation.npz",
        test_b / "evaluation.npz.json",
        test_b / "manifest.json",
        test_b / "receipt.json",
        test_b / "completion.json",
        REPO / "初赛数据集/test/test_a_samples.csv",
        REPO / "初赛数据集/test/test_b_samples.csv",
        model_dir / "bundle.json",
        model_dir / "forest.joblib",
        model_dir / "preprocessor.json",
        QRF_SOURCE / "features/12/train.npz",
        QRF_SOURCE / "features/12/train.npz.json",
        QRF_SOURCE / "receipt.json",
    ]
    for month in range(6, 12):
        inputs.extend([
            V22_DEV / f"directions/outer-{month}.csv",
            V22_DEV / f"lambdas/{month}/summary.json",
            V22_DEV / f"predictions/{month}/V10.csv",
            V22_DEV / f"predictions/{month}/V21_REPLAY.csv",
        ])
    for month in range(4, 12):
        inputs.append(V22_DEV / f"medians/{month}.json")
    missing = [str(path) for path in [*sources, *inputs] if not path.is_file()]
    if missing:
        raise ContractError(f"v0.23 source recovery is incomplete: {missing}")
    paths = {
        "test_a_metadata": REPO / "初赛数据集/test/test_a_samples.csv",
        "test_b_metadata": REPO / "初赛数据集/test/test_b_samples.csv",
        "test_b_features": test_b / "evaluation.npz",
        "test_b_v1": test_b / "V1_full_precision.csv",
        "test_b_v10": PROBE / "test_b_V10_full_precision.csv",
        "final_model": V22_FINAL / "final_model.json",
        "final_medians": V22_FINAL / "median_parameters.json",
        "final_lambdas": V22_FINAL / "lambda_parameters.json",
        "final_train": QRF_SOURCE / "features/12/train.npz",
        "final_bundle": model_dir / "bundle.json",
        "test_a_qrf": V22_FINAL / "qrf.npz",
        "test_a_v10": REPO / "local/runs/optimization-v0.15-v10-target-composition-r1/submission/result.csv",
        "historical_history": historical_history,
    }
    return sources, inputs, paths


def _preflight() -> None:
    base = registration()["base_commit"]
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, "HEAD"],
        cwd=REPO,
        check=False,
    )
    if ancestry.returncode != 0:
        raise ContractError("v0.23 frozen base is not an ancestor of HEAD")
    if _git("branch", "--show-current") != "optimization-v0.23-recovery-and-stage-validation":
        raise ContractError("v0.23 worktree branch differs")
    development = read_json(V22_DEV / "completion.json")
    final = read_json(V22_FINAL / "completion.json")
    if development.get("G0") != "PASS" or development.get("G1") != "PASS":
        raise ContractError("historical v0.22 development G0/G1 must remain PASS")
    feedback = read_json(V22_FINAL / "platform_feedback.json")
    if feedback.get("score") != 83.1166 or final.get("candidate") != "V22_CAUSAL_H2_QRF_SHRINK":
        raise ContractError("historical V22 platform identity differs")


def run(output: Path, v21_zip: Path) -> dict:
    root = output.resolve()
    v21_zip = v21_zip.resolve()
    if root.exists():
        raise FileExistsError(root)
    if not root.is_relative_to((REPO / "local/runs").resolve()):
        raise ContractError("v0.23 outputs must remain in ignored local/runs")
    _preflight()
    sources, inputs, stage_paths = _source_paths(v21_zip)
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    try:
        manifest = {
            "kind": "V23_RECOVERY_AND_STAGE_VALIDATION_MANIFEST_v1",
            "phase": "optimization-v0.23-recovery-and-stage-validation",
            "not_a_new_prediction_candidate": True,
            "base_commit": registration()["base_commit"],
            "branch": _git("branch", "--show-current"),
            "authorization": load_yaml(SCOPE)["authorization"],
            "registration": registration(),
            "sources": file_identities({str(path.resolve()): path.resolve() for path in sources}),
            "inputs": file_identities({str(path.resolve()): path.resolve() for path in inputs}),
            "stage_paths": {key: str(path.resolve()) for key, path in stage_paths.items()},
            "protection_sha256": file_sha256(PROTECTION),
            "official_round2_identity_verified": False,
            "platform_account_state_verified": False,
            "environment": runtime_environment(),
            "automatic_uploads": 0,
            "desktop_writes": 0,
            "public_pushes": 0,
        }
        atomic_write_json(root / "manifest.json", manifest)
        _ledger(root, "P0_V21_ORIGINAL_PAYLOAD_RECOVERY", protected_labels_read=False)
        recovery = recover_payload(v21_zip, root / "recovery", stage_paths["test_a_metadata"])
        archived_replay = V22_DEV / "p0/v21_rule_replay_result.csv"
        original_payload = (root / "recovery/result.csv").read_bytes()
        recovery["archived_v22_replay_result_sha256"] = file_sha256(archived_replay)
        recovery["archived_v22_replay_payload_byte_identical"] = archived_replay.read_bytes() == original_payload
        _ledger(root, "P0_CORRECT_V21_RULE_REPLAY_FROM_CERTIFIED_HISTORY", protected_labels_read=True)
        bundle = read_json(stage_paths["final_bundle"])
        history = training_history_from_handoff(stage_paths["final_train"], bundle)
        metadata = stage_metadata(stage_paths["test_a_metadata"])
        v10 = pd.read_csv(stage_paths["test_a_v10"], dtype={"sample_id": str}, float_precision="round_trip")
        with np.load(stage_paths["test_a_qrf"], allow_pickle=False) as qrf:
            if qrf["ids"].astype(str).tolist() != metadata.sample_id.tolist():
                raise ContractError("test_a fresh support IDs differ during corrected V21 replay")
            corrected, corrected_diagnostics = replay_v21(
                v10, metadata, history, qrf["effective_neighbors"],
                cutoff=pd.Timestamp(read_json(stage_paths["final_model"])["cutoff_ns"], unit="ns", tz="UTC"),
            )
        corrected_payload = serialize_six_decimals(corrected)
        corrected_path = root / "recovery/v21_rule_replay_corrected.csv"
        with corrected_path.open("xb") as handle:
            handle.write(corrected_payload)
        old_frame = pd.read_csv(archived_replay, dtype={"sample_id": str}, float_precision="round_trip")
        original_frame = pd.read_csv(root / "recovery/result.csv", dtype={"sample_id": str}, float_precision="round_trip")
        replay_delta = old_frame.pred_tap_time_len - original_frame.pred_tap_time_len
        recovery["corrected_rule_replay_sha256"] = file_sha256(corrected_path)
        recovery["corrected_rule_replay_payload_byte_identical"] = corrected_payload == original_payload
        recovery["corrected_rule_replay_diagnostics"] = corrected_diagnostics
        recovery["archived_v22_replay_difference"] = {
            "status": "PRESERVED_HISTORICAL_REPLAY_DISCREPANCY",
            "changed_time_rows": int((replay_delta != 0).sum()),
            "min_archived_minus_original": float(replay_delta.min()),
            "max_archived_minus_original": float(replay_delta.max()),
            "cause": "archived v0.22 replay parsed naive history timestamps as UTC instead of Asia/Shanghai",
            "old_artifact_modified": False,
        }
        recovery["waiver_superseded_as_recovery_evidence"] = True
        recovery["platform_verified"] = False
        if not recovery["original_archive_exact"] or not recovery["corrected_rule_replay_payload_byte_identical"]:
            raise ContractError("original V21 archive or corrected frozen-rule replay identity differs")
        atomic_write_json(root / "recovery_receipt.json", recovery)

        _ledger(root, "P1_CANONICAL_SIX_DECIMAL_SCORING_AND_P2_DIAGNOSTICS", protected_labels_read=True)
        source_errors = pd.read_csv(
            V22_DEV / "all_errors.csv", dtype={"sample_id": str, "spout_no": str},
            float_precision="round_trip",
        )
        historical_replay = historical_v21_replay_audit(
            stage_paths["historical_history"],
            V22_DEV / "directions",
            V22_DEV / "predictions",
            source_errors,
        )
        atomic_write_json(root / "historical_v21_replay_audit.json", historical_replay)
        canonical, identity = canonical_errors(source_errors)
        identity.update({
            "source": str((V22_DEV / "all_errors.csv").resolve()),
            "source_sha256": file_sha256(V22_DEV / "all_errors.csv"),
            "V21_payload_recovered": True,
            "V21_corrected_rule_replay_matches_recovered_test_a": True,
            "archived_v22_replay_matches_recovered_test_a": False,
            "historical_V21_REPLAY_six_origins_byte_exact": historical_replay["all_six_replays_byte_exact"],
            "historical_V21_REPLAY_scoring_predictions_exact": historical_replay["all_scoring_predictions_exact"],
        })
        card = scorecard(canonical)
        summary = score_summary(card)
        deltas = reference_deltas(summary)
        _write_csv(root / "canonical_scorecard.csv", card)
        _write_csv(root / "canonical_summary.csv", summary)
        _write_csv(root / "reference_deltas.csv", deltas)
        atomic_write_json(root / "score_identity_checks.json", identity)

        directions = load_directions(V22_DEV / "directions")
        lambdas = _lambda_by_origin(V22_DEV / "lambdas")
        median_errors = median_only_errors(canonical, directions)
        median_table, conclusions = diagnostic_table(canonical, median_errors)
        monthly = monthly_bias_table(canonical, median_errors)
        strength = shrinkage_strength(canonical, directions, lambdas)
        _write_csv(root / "median_only_diagnostic.csv", median_table)
        _write_csv(root / "median_only_monthly_bias.csv", monthly)
        atomic_write_json(root / "median_only_conclusions.json", conclusions)
        atomic_write_json(root / "shrinkage_strength.json", strength)

        stage_output = root / "stage_preview/root"
        subprocess.run([
            sys.executable, str(REPO / "scripts/optimization_v23_stage_infer.py"),
            "--run", str(root),
            "--stage-identity", "LEGACY_TEST_B_ENGINEERING_PREVIEW",
            "--metadata", str(stage_paths["test_b_metadata"]),
            "--model-manifest", str(stage_paths["final_model"]),
            "--feature-handoff", str(stage_paths["test_b_features"]),
            "--v1", str(stage_paths["test_b_v1"]),
            "--v10", str(stage_paths["test_b_v10"]),
            "--train-handoff", str(stage_paths["final_train"]),
            "--bundle", str(stage_paths["final_bundle"]),
            "--saved-medians", str(stage_paths["final_medians"]),
            "--saved-lambdas", str(stage_paths["final_lambdas"]),
            "--output", str(stage_output),
        ], cwd=REPO, check=True)
        stage_receipt = read_json(stage_output / "receipt.json")
        stage_manifest = {
            "kind": "V23_FROZEN_STAGE_CANDIDATES_v1",
            "status": "PASS_ENGINEERING_PREVIEW_ONLY",
            "stage_identity": "LEGACY_TEST_B_ENGINEERING_PREVIEW_NOT_OFFICIAL_ROUND2",
            "official_round2_identity_verified": False,
            "algorithms_frozen_before_any_round2_score": {
                "V1": {"algorithm": "V1_RATE_STRUCTURAL", "platform_selection_history": False, "result": stage_receipt["results"][V1]},
                "V21": {"algorithm": "V10_plus_spout1_neff_lt_500_query60_median_25pct", "platform_selection_history": True, "fresh_effective_neighbors": True, "result": stage_receipt["results"][V21]},
                "V22": {"algorithm": "Q_plus_saved_lambda_times_one_minus_u_times_saved_M_minus_Q", "test_a_score": 83.1166, "saved_M_and_lambda": True, "result": stage_receipt["results"][V22]},
            },
            "internal_reference": {"algorithm": V10, "source": str(stage_paths["test_b_v10"]), "sha256": file_sha256(stage_paths["test_b_v10"])},
            "model": {"path": str(stage_paths["final_model"]), "sha256": file_sha256(stage_paths["final_model"])},
            "history": {"path": str(stage_paths["final_train"]), "sha256": file_sha256(stage_paths["final_train"])},
            "saved_medians": {"path": str(stage_paths["final_medians"]), "sha256": file_sha256(stage_paths["final_medians"])},
            "saved_lambdas": {"path": str(stage_paths["final_lambdas"]), "sha256": file_sha256(stage_paths["final_lambdas"])},
            "new_fits": 0,
            "new_median_estimates": 0,
            "packages_created": 0,
            "platform_uploads": 0,
        }
        atomic_write_json(root / "stage_candidates_manifest.json", stage_manifest)
        fit_counts = {
            "status": "PASS_ZERO_FIT",
            "forest_fits": 0, "catboost_or_other_model_fits": 0, "preprocessor_fits": 0,
            "LAD_or_lambda_fits": 0, "new_median_estimates": 0, "new_candidates": 0,
            "V21_original_archives_recovered": 1, "test_a_uploads": 0,
            "stage_preview_packages": 0, "platform_uploads": 0,
        }
        atomic_write_json(root / "fit_counts.json", fit_counts)
        atomic_write_json(root / "acceptance_pending.json", {
            "status": "PENDING_COLD_VALIDATION",
            "P0_V21_payload": "PASS_ORIGINAL_ARCHIVE_AND_PAYLOAD",
            "P1_four_algorithm_same_precision": "PASS",
            "P2_saved_M_only_diagnostic": "PASS_DIAGNOSTIC_ONLY",
            "P3_three_frozen_stage_algorithms": "PASS_ENGINEERING_PREVIEW_ONLY",
            "G0": "PENDING_COLD_VALIDATION",
            "G1": "NOT_APPLICABLE_NO_NEW_PREDICTION_CANDIDATE",
        })
        _ledger(root, "INDEPENDENT_ZERO_FIT_COLD_VALIDATION", protected_labels_read=True)
        subprocess.run([
            sys.executable, str(REPO / "scripts/optimization_v23_cold_check.py"), "--run", str(root),
        ], cwd=REPO, check=True)
        cold = read_json(root / "cold_validation.json")
        if cold.get("status") != "PASS":
            raise ContractError("v0.23 independent cold validation did not pass")
        acceptance = {
            "status": "PASS_RECOVERY_AND_STAGE_VALIDATION",
            "P0_V21_payload": "PASS_ORIGINAL_ARCHIVE_AND_PAYLOAD",
            "P1_four_algorithm_same_precision": "PASS",
            "P2_saved_M_only_diagnostic": "PASS_DIAGNOSTIC_ONLY",
            "P3_three_frozen_stage_algorithms": "PASS_ENGINEERING_PREVIEW_ONLY",
            "G0": "PASS",
            "G1": "NOT_APPLICABLE_NO_NEW_PREDICTION_CANDIDATE",
            "official_round2_status": "PENDING_IDENTITY_AND_EXPLICIT_UPLOAD_AUTHORIZATION",
        }
        atomic_write_json(root / "acceptance.json", acceptance)
        j = deltas.loc[deltas.scope == "J", ["algorithm", "E", "delta_vs_V1", "delta_vs_V10"]].to_dict("records")
        completion = {
            "status": acceptance["status"], "G0": acceptance["G0"], "G1": acceptance["G1"],
            "historical_v22_G0_G1_preserved": True,
            "V22_test_a_user_reported": 83.1166,
            "V21_highest_user_reported_test_a": 83.2375,
            "V21_original_archive_recovered": True,
            "V21_original_archive_sha256": recovery["recovered_archive_sha256"],
            "V21_payload_sha256": recovery["recovered_result_sha256"],
            "historical_V21_REPLAY_audit": "PASS_SIX_ORIGINS_BYTE_EXACT",
            "canonical_J": j,
            "median_only_conclusions": conclusions,
            "stage_preview": "LEGACY_TEST_B_ENGINEERING_PREVIEW_NOT_OFFICIAL_ROUND2",
            "fit_counts": fit_counts,
            "cold_validation_sha256": file_sha256(root / "cold_validation.json"),
            "platform_account_state_verified": False,
            "platform_effective_submission": "UNKNOWN",
            "platform_uploads": 0, "desktop_writes": 0, "public_pushes": 0,
        }
        atomic_write_json(root / "completion.json", completion)
        return completion
    except Exception as exc:
        if not (root / "failure.json").exists():
            atomic_write_json(root / "failure.json", {
                "status": "FAILED_PRESERVED", "error_type": type(exc).__name__, "error": str(exc),
                "fit_attempts": 0, "platform_uploads": 0, "desktop_writes": 0,
            })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v21-zip", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.v21_zip), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
