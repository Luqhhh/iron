#!/usr/bin/env python3
"""Independent zero-fit verifier for a completed v0.23 pre-completion run."""
from __future__ import annotations

import argparse
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from bf_tap.artifacts import atomic_write_json, file_sha256, verify_file_identities
from bf_tap.exceptions import ContractError
from bf_tap.optimization.v23_recovery import (
    _lambda_by_origin, canonical_errors, diagnostic_table, load_directions,
    historical_v21_replay_audit, median_only_errors, monthly_bias_table, read_json, reference_deltas,
    score_summary, scorecard, shrinkage_strength, stage_metadata,
    training_history_from_handoff,
)
from bf_tap.optimization.qrf_support_shrink import replay_v21, serialize_six_decimals
REPO = Path(__file__).resolve().parents[1]
V22_DEV = REPO / "local/runs/optimization-v0.22-causal-h2-r8"

_verify_spec = importlib.util.spec_from_file_location("verify_v21_payload", REPO / "verify_v21_payload.py")
if _verify_spec is None or _verify_spec.loader is None:
    raise RuntimeError("cannot load verify_v21_payload.py")
_verify_module = importlib.util.module_from_spec(_verify_spec)
_verify_spec.loader.exec_module(_verify_module)
inspect_payload = _verify_module.inspect_payload


def _csv_bytes(value: pd.DataFrame) -> bytes:
    return value.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")


def _assert_csv(path: Path, value: pd.DataFrame) -> None:
    if path.read_bytes() != _csv_bytes(value):
        raise ContractError(f"cold reconstructed table differs byte-for-byte: {path.name}")


def cold_check(run: Path) -> dict:
    root = run.resolve()
    if not root.is_relative_to((REPO / "local/runs").resolve()):
        raise ContractError("cold validation run must remain private")
    if (root / "completion.json").exists() or (root / "cold_validation.json").exists():
        raise ContractError("cold validation refuses completed or already-checked runs")
    manifest = read_json(root / "manifest.json")
    if manifest.get("kind") != "V23_RECOVERY_AND_STAGE_VALIDATION_MANIFEST_v1":
        raise ContractError("unknown v0.23 manifest")
    verify_file_identities(manifest["sources"])
    verify_file_identities(manifest["inputs"])
    stage = {key: Path(value) for key, value in manifest["stage_paths"].items()}

    payload, recovery = inspect_payload(root / "recovery/Luqhhh_bf_tap_predict_prelim_TGATE_SPOUT1.zip", stage["test_a_metadata"])
    if (
        not recovery["original_archive_exact"]
        or payload != (root / "recovery/result.csv").read_bytes()
    ):
        raise ContractError("cold V21 archive/payload recovery differs")
    bundle = read_json(stage["final_bundle"])
    history = training_history_from_handoff(stage["final_train"], bundle)
    metadata = stage_metadata(stage["test_a_metadata"])
    v10 = pd.read_csv(stage["test_a_v10"], dtype={"sample_id": str}, float_precision="round_trip")
    with np.load(stage["test_a_qrf"], allow_pickle=False) as qrf:
        corrected, _ = replay_v21(
            v10, metadata, history, qrf["effective_neighbors"],
            cutoff=pd.Timestamp(read_json(stage["final_model"])["cutoff_ns"], unit="ns", tz="UTC"),
        )
    corrected_payload = serialize_six_decimals(corrected)
    receipt = read_json(root / "recovery_receipt.json")
    if (
        corrected_payload != payload
        or corrected_payload != (root / "recovery/v21_rule_replay_corrected.csv").read_bytes()
        or receipt.get("corrected_rule_replay_payload_byte_identical") is not True
        or receipt.get("archived_v22_replay_payload_byte_identical") is not False
    ):
        raise ContractError("cold corrected V21 rule replay differs")

    source_errors = pd.read_csv(
        V22_DEV / "all_errors.csv", dtype={"sample_id": str, "spout_no": str}, float_precision="round_trip",
    )
    historical_replay = historical_v21_replay_audit(
        stage["historical_history"],
        V22_DEV / "directions",
        V22_DEV / "predictions",
        source_errors,
    )
    if historical_replay != read_json(root / "historical_v21_replay_audit.json"):
        raise ContractError("cold historical V21 replay audit differs")
    canonical, identity = canonical_errors(source_errors)
    identity.update({
        "source": str((V22_DEV / "all_errors.csv").resolve()),
        "source_sha256": file_sha256(V22_DEV / "all_errors.csv"),
        "V21_payload_recovered": True,
        "V21_corrected_rule_replay_matches_recovered_test_a": True,
        "archived_v22_replay_matches_recovered_test_a": False,
        "historical_V21_REPLAY_six_origins_byte_exact": True,
        "historical_V21_REPLAY_scoring_predictions_exact": True,
    })
    if identity != read_json(root / "score_identity_checks.json"):
        raise ContractError("cold score identity checks differ")
    card = scorecard(canonical)
    summary = score_summary(card)
    deltas = reference_deltas(summary)
    _assert_csv(root / "canonical_scorecard.csv", card)
    _assert_csv(root / "canonical_summary.csv", summary)
    _assert_csv(root / "reference_deltas.csv", deltas)

    directions = load_directions(V22_DEV / "directions")
    lambdas = _lambda_by_origin(V22_DEV / "lambdas")
    median_errors = median_only_errors(canonical, directions)
    median_table, conclusions = diagnostic_table(canonical, median_errors)
    monthly = monthly_bias_table(canonical, median_errors)
    strength = shrinkage_strength(canonical, directions, lambdas)
    _assert_csv(root / "median_only_diagnostic.csv", median_table)
    _assert_csv(root / "median_only_monthly_bias.csv", monthly)
    if conclusions != read_json(root / "median_only_conclusions.json") or strength != read_json(root / "shrinkage_strength.json"):
        raise ContractError("cold M-only or shrinkage diagnostics differ")

    cold_output = root / "stage_preview/cold"
    subprocess.run([
        sys.executable, str(REPO / "scripts/optimization_v23_stage_infer.py"),
        "--run", str(root),
        "--stage-identity", "LEGACY_TEST_B_ENGINEERING_PREVIEW",
        "--metadata", str(stage["test_b_metadata"]),
        "--model-manifest", str(stage["final_model"]),
        "--feature-handoff", str(stage["test_b_features"]),
        "--v1", str(stage["test_b_v1"]),
        "--v10", str(stage["test_b_v10"]),
        "--train-handoff", str(stage["final_train"]),
        "--bundle", str(stage["final_bundle"]),
        "--saved-medians", str(stage["final_medians"]),
        "--saved-lambdas", str(stage["final_lambdas"]),
        "--output", str(cold_output),
    ], cwd=REPO, check=True)
    root_receipt = read_json(root / "stage_preview/root/receipt.json")
    cold_receipt = read_json(cold_output / "receipt.json")
    for candidate in ("V1", "V21_REPLAY", "V22_CAUSAL_H2_QRF_SHRINK"):
        if (root / f"stage_preview/root/{candidate}/result.csv").read_bytes() != (cold_output / f"{candidate}/result.csv").read_bytes():
            raise ContractError(f"cold {candidate} stage result differs")
    with np.load(root / "stage_preview/root/qrf.npz", allow_pickle=False) as left, np.load(cold_output / "qrf.npz", allow_pickle=False) as right:
        if set(left.files) != set(right.files) or any(not np.array_equal(left[key], right[key]) for key in left.files):
            raise ContractError("cold fresh QRF support arrays differ")
    if root_receipt["diagnostics"] != cold_receipt["diagnostics"]:
        raise ContractError("cold stage diagnostics differ")
    value = {
        "status": "PASS",
        "separate_process": True,
        "source_identities_verified": True,
        "V21_original_archive_and_payload_exact": True,
        "V21_corrected_rule_replay_exact": True,
        "archived_v22_replay_discrepancy_preserved": True,
        "historical_V21_REPLAY_six_origins_byte_exact": True,
        "canonical_scorecard_byte_exact": True,
        "reference_deltas_byte_exact": True,
        "saved_M_only_diagnostic_byte_exact": True,
        "shrinkage_diagnostic_exact": True,
        "stage_candidates_byte_exact": True,
        "fresh_QRF_support_arrays_exact": True,
        "worker_checks": cold_receipt["diagnostics"].get("fresh_effective_neighbors") is True,
        "forest_fit_attempts": 0,
        "preprocessor_fit_attempts": 0,
        "LAD_or_lambda_fit_attempts": 0,
        "new_median_estimates": 0,
        "platform_packages_created": 0,
        "platform_uploads": 0,
        "desktop_writes": 0,
    }
    atomic_write_json(root / "cold_validation.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(cold_check(args.run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
