#!/usr/bin/env python3
"""Zero-fit V1/V21/V22 stage inference with fresh QRF support diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from bf_tap.artifacts import atomic_write_json, file_sha256, stable_digest
from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.v23_recovery import (
    V1, V21, V22, read_json, stage_candidates, stage_metadata,
    training_history_from_handoff,
)


REPO = Path(__file__).resolve().parents[1]
SCOPE = REPO / "configs/optimization_v0_23/access_scope.yaml"


def _identity_entries(value) -> list[dict]:
    found = []
    if isinstance(value, dict):
        if set(("path", "sha256", "bytes")) <= set(value):
            found.append(value)
        for child in value.values():
            found.extend(_identity_entries(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_identity_entries(child))
    return found


def _trusted(manifest: dict, path: Path) -> dict:
    resolved = path.resolve()
    matches = [entry for entry in _identity_entries(manifest) if Path(entry["path"]).resolve() == resolved]
    if len(matches) != 1:
        raise ContractError(f"stage input lacks one trusted manifest identity: {path}")
    entry = matches[0]
    if file_sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
        raise ContractError(f"stage input differs from trusted identity: {path}")
    return entry


def _ledger(run: Path, purpose: str) -> None:
    scope = load_yaml(SCOPE)
    protection = REPO / scope["protection_contract"]
    if not protection.is_file():
        raise ContractError("protected-label contract missing")
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
            "run_manifest_sha256": file_sha256(run / "manifest.json"),
            "protection_sha256": file_sha256(protection),
            "previous_sha256": stable_digest(prior[-1]) if prior else None,
            "protected_labels_read": True,
        }
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def infer(args) -> dict:
    run = args.run.resolve()
    output = args.output.resolve()
    if not run.is_relative_to((REPO / "local/runs").resolve()) or not (run / "manifest.json").is_file():
        raise ContractError("v0.23 stage inference requires a registered private run")
    if not output.is_relative_to(run) or output.exists():
        raise ContractError("stage output must be a new path inside the registered run")
    manifest = read_json(run / "manifest.json")
    if manifest.get("kind") != "V23_RECOVERY_AND_STAGE_VALIDATION_MANIFEST_v1":
        raise ContractError("unknown v0.23 trusted manifest")
    inputs = [
        args.metadata, args.model_manifest, args.feature_handoff, args.v1, args.v10,
        args.train_handoff, args.bundle, args.saved_medians, args.saved_lambdas,
    ]
    for path in inputs:
        _trusted(manifest, path)
    if args.stage_identity == "OFFICIAL_ROUND2" and not manifest.get("official_round2_identity_verified", False):
        raise ContractError("official round2 identity has not been verified")
    metadata = stage_metadata(args.metadata)
    forbidden = {"tap_iron", "tap_time_len", "target", "actual"} & set(metadata)
    if forbidden:
        raise ContractError("stage metadata contains a forbidden target column")
    final_model = read_json(args.model_manifest)
    bundle = read_json(args.bundle)
    model_dir = Path(final_model["model_path"])
    if (
        final_model.get("kind") != "V22_CERTIFIED_QRF_MODEL_v1"
        or file_sha256(model_dir / "bundle.json") != final_model["bundle_sha256"]
        or file_sha256(model_dir / "forest.joblib") != bundle["forest_sha256"]
        or file_sha256(model_dir / "preprocessor.json") != bundle["preprocessor_sha256"]
        or file_sha256(args.train_handoff) != bundle["input"]["sha256"]
    ):
        raise ContractError("certified final QRF model or training handoff identity differs")
    with __import__("numpy").load(args.feature_handoff, allow_pickle=False) as handoff:
        if handoff["ids"].astype(str).tolist() != metadata.sample_id.tolist():
            raise ContractError("feature handoff and metadata IDs differ")
    _ledger(run, f"{args.stage_identity}_V21_FROZEN_RULE_HISTORY_ACCESS")
    history = training_history_from_handoff(args.train_handoff, bundle)
    if set(metadata.sample_id) & set(history.sample_id):
        raise ContractError("stage and model-training IDs overlap")
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    worker_output = output / "qrf.npz"
    worker = REPO / "workers/qrf_v015"
    subprocess.run([
        str(worker / ".venv/bin/python"), str(worker / "inference_v022.py"),
        "--model-manifest", str(args.model_manifest.resolve()),
        "--model-manifest-sha256", file_sha256(args.model_manifest),
        "--input", str(args.feature_handoff.resolve()),
        "--input-sha256", file_sha256(args.feature_handoff),
        "--output", str(worker_output),
    ], cwd=REPO, check=True)
    worker_report = read_json(str(worker_output) + ".json")
    if worker_report.get("status") != "PASS" or any(worker_report.get("zero_fit", {}).values()):
        raise ContractError("stage QRF support inference was not zero-fit PASS")
    payloads, diagnostics = stage_candidates(
        metadata, worker_output, final_model, read_json(args.saved_medians),
        read_json(args.saved_lambdas), history, args.v1, args.v10,
    )
    results = {}
    for candidate, payload in payloads.items():
        path = output / candidate / "result.csv"
        path.parent.mkdir()
        with path.open("xb") as handle:
            handle.write(payload)
        results[candidate] = {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
    receipt = {
        "kind": "V23_STAGE_INFERENCE_RECEIPT_v1",
        "status": "PASS_ENGINEERING_ONLY",
        "stage_identity": args.stage_identity,
        "official_round2_identity_verified": args.stage_identity == "OFFICIAL_ROUND2",
        "algorithms": [V1, V21, V22],
        "results": results,
        "diagnostics": diagnostics,
        "worker_report": {"path": str(Path(str(worker_output) + ".json")), "sha256": file_sha256(Path(str(worker_output) + ".json"))},
        "target_columns_read": False,
        "new_fits": 0,
        "new_median_estimates": 0,
        "packages_created": 0,
        "platform_uploads": 0,
        "desktop_writes": 0,
    }
    atomic_write_json(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage-identity", choices=("LEGACY_TEST_B_ENGINEERING_PREVIEW", "OFFICIAL_ROUND2"), required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--feature-handoff", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v10", type=Path, required=True)
    parser.add_argument("--train-handoff", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--saved-medians", type=Path, required=True)
    parser.add_argument("--saved-lambdas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = infer(args)
    except Exception as exc:
        output = args.output.resolve()
        if output.exists() and output.is_dir() and not (output / "failure.json").exists():
            atomic_write_json(output / "failure.json", {
                "status": "FAILED_PRESERVED", "error_type": type(exc).__name__, "error": str(exc),
                "new_fits": 0, "packages_created": 0, "platform_uploads": 0,
            })
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
