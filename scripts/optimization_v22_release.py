"""Finalize one authorized V22 test-A package and verify it in a cold process."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

from bf_tap.artifacts import (
    atomic_write_json,
    file_identities,
    file_sha256,
    runtime_environment,
    stable_digest,
    verify_file_identities,
)
from bf_tap.exceptions import ContractError
from bf_tap.config import load_yaml
from bf_tap.optimization.qrf_support_shrink import (
    apply_shrink,
    build_prediction_directions,
    fit_lambda,
    fixed_cutoff_medians,
    select_outer_oof,
    serialize_six_decimals,
    verify_h2_bank,
    verify_lambda_certificate,
    verify_median_certificate,
)
from bf_tap.submission import pack_submission, validate_submission

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/optimization_v0_22/experiment.yaml"
SCOPE = REPO / "configs/optimization_v0_22/access_scope.yaml"
DEVELOPMENT = REPO / "local/runs/optimization-v0.22-causal-h2-r8"
DESKTOP = Path("/mnt/c/Users/lqh22/Desktop/Luqhhh_bf_tap_predict_prelim.zip")
TEAM = "Luqhhh"
STAGE = "test_a"
AUTHORIZATION = "user_20260916_generate_v22_platform_package_to_c_drive_desktop"


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def registration() -> dict:
    value = load_yaml(CONFIG)
    if (
        value.get("candidate") != "V22_CAUSAL_H2_QRF_SHRINK"
        or value.get("protocol") != "QRF_FULLTRAIN_LEAF_v1"
        or value.get("budget", {}).get("final_lambda_slots") != 2
        or value.get("budget", {}).get("final_cutoff_spout_medians") != 2
        or value.get("budget", {}).get("conditional_final_platform_candidates") != 1
    ):
        raise ContractError("frozen V22 final registration differs")
    return value


def _ledger(root: Path, purpose: str, *, protected_labels_read: bool) -> None:
    scope = load_yaml(SCOPE)
    ledger = REPO / scope["new_ledger"]
    ledger.parent.mkdir(parents=True, exist_ok=True)
    prior = ledger.read_text(encoding="utf-8").splitlines() if ledger.exists() else []
    event = {
        "at": datetime.now(timezone.utc).isoformat(),
        "purpose": purpose,
        "authorization": AUTHORIZATION,
        "run_manifest_sha256": file_sha256(root / "manifest.json"),
        "previous_sha256": stable_digest(prior[-1]) if prior else None,
        "protected_labels_read": protected_labels_read,
    }
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _entry(value, path: Path) -> dict:
    resolved = path.resolve()
    found: list[dict] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            if set(("path", "sha256", "bytes")) <= set(node) and Path(node["path"]).resolve() == resolved:
                found.append(node)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    if len(found) != 1:
        raise ContractError(f"expected one trusted identity for {path}, found {len(found)}")
    verify_file_identities({str(path): found[0]})
    return found[0]


def _passed_development(root: Path) -> dict:
    required = {
        "completion": root / "completion.json",
        "acceptance": root / "acceptance.json",
        "cold_validation": root / "cold_validation.json",
        "p0": root / "p0_original_replay.json",
        "h2_bank": root / "h2_bank.csv",
    }
    values = {name: read_json(path) if path.suffix == ".json" else None for name, path in required.items()}
    if (
        values["completion"].get("status") != "DEV_ACCEPTED_PENDING_OFFICIAL_IDENTITY"
        or values["completion"].get("G0") != "PASS"
        or values["completion"].get("G1") != "PASS"
        or values["acceptance"].get("status") != "DEV_ACCEPTED_PENDING_OFFICIAL_IDENTITY"
        or values["cold_validation"].get("status") != "PASS"
        or values["p0"].get("status") != "PASS"
    ):
        raise ContractError("V22 development/cold/P0 gate is not passed")
    return {name: {"path": str(path.resolve()), "sha256": file_sha256(path)} for name, path in required.items()}


def _final_sources(development: Path) -> dict:
    p0 = read_json(development / "p0_original_replay.json")
    component_path = Path(p0["certified_final_components"]["component_bundle"])
    if file_sha256(component_path) != p0["certified_final_components"]["component_bundle_sha256"]:
        raise ContractError("final component bundle identity changed")
    component = read_json(component_path)
    model_dir = Path(component["qrf_model"])
    bundle_path = model_dir / "bundle.json"
    if file_sha256(bundle_path) != component["qrf_bundle_sha256"]:
        raise ContractError("final QRF bundle identity changed")
    bundle = read_json(bundle_path)
    source_root = model_dir.parents[1]
    receipt_path = source_root / "receipt.json"
    receipt = read_json(receipt_path)
    train = source_root / "features/12/train.npz"
    evaluation = source_root / "features/12/evaluation.npz"
    for path in (
        bundle_path,
        model_dir / "forest.joblib",
        model_dir / "preprocessor.json",
        train,
        Path(str(train) + ".json"),
        evaluation,
        Path(str(evaluation) + ".json"),
    ):
        _entry(receipt, path)
    if (
        bundle.get("protocol") != registration()["protocol"]
        or bundle.get("parameters") != registration()["qrf_parameters"]
        or file_sha256(model_dir / "forest.joblib") != bundle["forest_sha256"]
        or file_sha256(model_dir / "preprocessor.json") != bundle["preprocessor_sha256"]
    ):
        raise ContractError("final QRF protocol, parameters, or assets differ")

    probe = REPO / "local/runs/platform-probes-r2-r1"
    v10 = probe / "test_a_V10_full_precision.csv"
    probe_receipt = read_json(probe / "test_a_V10_full_precision.csv.evidence/receipt.json")
    _entry(probe_receipt, v10)
    stage_manifest_path = probe / "test_a_V10_full_precision.csv.evidence/manifest.json"
    stage_manifest = read_json(stage_manifest_path)
    metadata = REPO / "初赛数据集/test/test_a_samples.csv"
    verify_file_identities({"test_a_samples": stage_manifest["inputs"]["test_a_samples"]})
    if Path(stage_manifest["inputs"]["test_a_samples"]["path"]).resolve() != metadata.resolve():
        raise ContractError("test-A metadata path differs from certified stage input")
    v10_serialized = REPO / "local/runs/optimization-v0.15-v10-target-composition-r1/submission/result.csv"
    if file_sha256(v10_serialized) != registration()["trusted_sources"]["v10_result_sha256"]:
        raise ContractError("original certified V10 result identity changed")
    return {
        "component": component_path,
        "model_dir": model_dir,
        "bundle": bundle_path,
        "train": train,
        "evaluation": evaluation,
        "v10": v10,
        "v10_serialized": v10_serialized,
        "metadata": metadata,
        "receipt": receipt_path,
        "probe_receipt": probe / "test_a_V10_full_precision.csv.evidence/receipt.json",
        "stage_manifest": stage_manifest_path,
    }


def _history(train: Path, bundle: dict) -> pd.DataFrame:
    with np.load(train, allow_pickle=False) as data:
        required = {"ids", "spout", "reference_ns", "y", "available_ns"}
        if required - set(data.files):
            raise ContractError("final QRF training handoff lacks certified label fields")
        ids = data["ids"].astype(str)
        if ids.tolist() != bundle["input"]["ids"] or len(set(ids)) != 2754:
            raise ContractError("final QRF training identity or unique row count differs")
        result = pd.DataFrame({
            "sample_id": ids,
            "spout_no": data["spout"].astype(str),
            "reference_time": pd.to_datetime(data["reference_ns"], utc=True).tz_convert("Asia/Shanghai"),
            "label_available_at": pd.to_datetime(data["available_ns"], utc=True).tz_convert("Asia/Shanghai"),
            "tap_time_len": data["y"].astype(float),
        })
    cutoff = pd.Timestamp(bundle["input"]["cutoff"])
    if (
        result.sample_id.duplicated().any()
        or (result.reference_time >= cutoff).any()
        or (result.label_available_at > cutoff).any()
        or not np.isfinite(result.tap_time_len).all()
    ):
        raise ContractError("final certified training history violates cutoff or label contract")
    return result


def _metadata(path: Path) -> pd.DataFrame:
    result = pd.read_csv(path, dtype={"sample_id": str, "spout_no": str})
    if list(result) != ["sample_id", "tap_no", "spout_no", "reference_time"]:
        raise ContractError("test-A metadata schema differs")
    result["reference_time"] = pd.to_datetime(result.reference_time).dt.tz_localize("Asia/Shanghai")
    return result[["sample_id", "spout_no", "reference_time"]]


def _model_manifest(output: Path, model_dir: Path, bundle: dict) -> Path:
    path = output / "final_model.json"
    value = {
        "kind": "V22_CERTIFIED_QRF_MODEL_v1",
        "protocol": registration()["protocol"],
        "cutoff_month": 12,
        "cutoff_ns": bundle["input"]["cutoff_ns"],
        "model_path": str(model_dir.resolve()),
        "bundle_sha256": file_sha256(model_dir / "bundle.json"),
        "unique_training_rows": 2754,
        "training_identity_sha256": stable_digest(bundle["input"]["ids"]),
        "upstream_completion": str((model_dir.parents[1] / "completion.json").resolve()),
        "upstream_completion_sha256": file_sha256(model_dir.parents[1] / "completion.json"),
    }
    atomic_write_json(path, value)
    return path


def _worker(model_manifest: Path, evaluation: Path, output: Path, expected: Path | None = None) -> dict:
    args = [
        str(REPO / "workers/qrf_v015/.venv/bin/python"),
        str(REPO / "workers/qrf_v015/inference_v022.py"),
        "--model-manifest", str(model_manifest.resolve()),
        "--model-manifest-sha256", file_sha256(model_manifest),
        "--input", str(evaluation.resolve()),
        "--input-sha256", file_sha256(evaluation),
        "--output", str(output.resolve()),
    ]
    if expected is not None:
        args.extend(("--expected-output", str(expected.resolve())))
    subprocess.run(args, cwd=REPO, check=True)
    report = read_json(str(output) + ".json")
    if report.get("status") != "PASS" or any(report.get("zero_fit", {}).values()):
        raise ContractError("final V22 worker inference was not zero-fit PASS")
    return report


def _parameters(development: Path, history: pd.DataFrame, cutoff: pd.Timestamp, output: Path) -> dict:
    bank = pd.read_csv(development / "h2_bank.csv", dtype={"sample_id": str, "spout_no": str}, float_precision="round_trip")
    verify_h2_bank(bank, registration()["expected_training_rows"])
    spouts = {}
    fit_attempts = 0
    selected_root = output / "lambda_selected"
    selected_root.mkdir()
    for spout in ("1", "2"):
        selected, status = select_outer_oof(bank, history, cutoff, spout)
        selected_path = selected_root / f"spout-{spout}.csv"
        selected.to_csv(selected_path, index=False, mode="x", float_format="%.17g", lineterminator="\n")
        intent = output / f"lambda-spout-{spout}.intent.json"
        atomic_write_json(intent, {
            "spout_no": spout,
            "outer_cutoff": cutoff.isoformat(),
            "selected_sha256": stable_digest(selected.to_dict("records")),
            "primitive": "constrained_LAD_smallest_minimizer",
        })
        certificate = fit_lambda(selected)
        fit_attempts += int(certificate["fit_performed"])
        spouts[spout] = {
            **status,
            "lambda": certificate["lambda"],
            "certificate": certificate,
            "selected_path": str(selected_path.resolve()),
            "selected_sha256": file_sha256(selected_path),
        }
    value = {
        "kind": "V22_FINAL_SPOUT_LAMBDAS_v1",
        "status": "PASS",
        "cutoff": cutoff.isoformat(),
        "spouts": spouts,
        "fit_counts": {"parameter_slots": 2, "fit_attempts": fit_attempts, "fallbacks": 2 - fit_attempts},
    }
    atomic_write_json(output / "lambda_parameters.json", value)
    return value


def _medians(history: pd.DataFrame, cutoff: pd.Timestamp, output: Path) -> dict:
    certificates = fixed_cutoff_medians(history, cutoff)
    value = {
        "kind": "V22_FINAL_CUTOFF_MEDIANS_v1",
        "status": "PASS",
        "cutoff": cutoff.isoformat(),
        "statistics": 2,
        "fallbacks": sum(int(v["fallback"]) for v in certificates.values()),
        "certificates": certificates,
    }
    atomic_write_json(output / "median_parameters.json", value)
    return value


def _directions(
    metadata: pd.DataFrame,
    qrf_path: Path,
    medians: dict,
    model: dict,
) -> pd.DataFrame:
    with np.load(qrf_path, allow_pickle=False) as data:
        if set(data.files) != {"ids", "Q", "effective_neighbors", "N"}:
            raise ContractError("unexpected final QRF payload")
        if data["ids"].astype(str).tolist() != metadata.sample_id.tolist() or len(set(data["N"].tolist())) != 1:
            raise ContractError("final QRF output is not aligned to test-A metadata")
        return build_prediction_directions(
            metadata,
            data["Q"],
            data["effective_neighbors"],
            int(data["N"][0]),
            medians["certificates"],
            model_cutoff=pd.Timestamp(model["cutoff_ns"], unit="ns", tz="UTC"),
            training_identity_sha256=model["training_identity_sha256"],
            model_bundle_sha256=model["bundle_sha256"],
        )


def cold_check(run: Path) -> dict:
    root = run.resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("kind") != "V22_TEST_A_RELEASE_MANIFEST_v1" or manifest.get("authorization") != AUTHORIZATION:
        raise ContractError("unknown V22 final release manifest")
    verify_file_identities(manifest["sources"])
    _ledger(root, "V22_FINAL_TEST_A_COLD_CERTIFICATE_VERIFICATION", protected_labels_read=True)
    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    model = read_json(root / "final_model.json")
    bundle = read_json(paths["bundle"])
    history = _history(paths["train"], bundle)
    parameters = read_json(root / "lambda_parameters.json")
    medians = read_json(root / "median_parameters.json")
    bank = pd.read_csv(paths["h2_bank"], dtype={"sample_id": str, "spout_no": str}, float_precision="round_trip")
    cutoff = pd.Timestamp(parameters["cutoff"])
    for spout in ("1", "2"):
        verify_median_certificate(history, medians["certificates"][spout])
        selected, _ = select_outer_oof(bank, history, cutoff, spout)
        verify_lambda_certificate(selected, parameters["spouts"][spout]["certificate"])
    cold_qrf = root / "cold/qrf.npz"
    cold_qrf.parent.mkdir(parents=True)
    report = _worker(root / "final_model.json", paths["evaluation"], cold_qrf, root / "qrf.npz")
    metadata = _metadata(paths["metadata"])
    directions = _directions(metadata, cold_qrf, medians, model)
    v10 = pd.read_csv(paths["v10"], dtype={"sample_id": str}, float_precision="round_trip")
    prediction = apply_shrink(v10, directions, {s: parameters["spouts"][s]["lambda"] for s in ("1", "2")})
    if serialize_six_decimals(prediction) != (root / "submission/result.csv").read_bytes():
        raise ContractError("cold reconstructed V22 result differs byte-for-byte")
    archive = root / f"submission/{TEAM}_bf_tap_predict_prelim.zip"
    with ZipFile(archive) as handle:
        if handle.namelist() != ["result.csv"] or handle.read("result.csv") != (root / "submission/result.csv").read_bytes():
            raise ContractError("final ZIP payload differs from validated result.csv")
    value = {
        "status": "PASS",
        "separate_process": True,
        "source_identities_verified": True,
        "median_certificates_verified": 2,
        "lambda_certificates_verified": 2,
        "worker_zero_fit": True,
        "worker_checks": report["checks"],
        "result_bytes_exact": True,
        "zip_payload_exact": True,
        "root_fit_attempts": 0,
        "platform_uploads": 0,
    }
    atomic_write_json(root / "cold_validation.json", value)
    return value


def build(development: Path, output: Path, desktop: Path) -> dict:
    development = development.resolve()
    root = output.resolve()
    desktop = desktop.resolve()
    if root.exists():
        raise FileExistsError(root)
    if not root.is_relative_to((REPO / "local/runs").resolve()):
        raise ContractError("V22 final evidence must remain under ignored local/runs")
    if desktop != DESKTOP.resolve() or desktop.exists():
        raise ContractError("desktop destination differs or already exists; refusing overwrite")
    root.mkdir(parents=True)
    root.chmod(0o700)
    try:
        development_evidence = _passed_development(development)
        paths = _final_sources(development)
        source_paths = {
            **{f"development_{k}": Path(v["path"]) for k, v in development_evidence.items()},
            **{key: value for key, value in paths.items() if value.is_file()},
            "experiment": REPO / "configs/optimization_v0_22/experiment.yaml",
            "scope": REPO / "configs/optimization_v0_22/access_scope.yaml",
            "protection": REPO / "configs/protection.yaml",
            "release_source": Path(__file__),
            "run_source": REPO / "scripts/optimization_v22_run.py",
            "math_source": REPO / "src/bf_tap/optimization/qrf_support_shrink.py",
            "worker_source": REPO / "workers/qrf_v015/inference_v022.py",
        }
        manifest = {
            "kind": "V22_TEST_A_RELEASE_MANIFEST_v1",
            "candidate": "V22_CAUSAL_H2_QRF_SHRINK",
            "stage": STAGE,
            "authorization": AUTHORIZATION,
            "development_status": "DEV_ACCEPTED_PENDING_OFFICIAL_IDENTITY",
            "stage_identity": "EXISTING_TEST_A_MATCHES_CERTIFIED_PRIOR_INFERENCE_INPUT",
            "not_official_round2": True,
            "test_targets_read": False,
            "test_scores_used_for_parameter_fit": False,
            "platform_feedback_used_for_design": True,
            "platform_uploads": 0,
            "desktop_write_authorized": True,
            "overwrite_prior_package": False,
            "sources": file_identities({str(path): path for path in source_paths.values()}),
            "paths": {**{key: str(value.resolve()) for key, value in paths.items()}, "h2_bank": str((development / "h2_bank.csv").resolve())},
            "environment": runtime_environment(),
        }
        atomic_write_json(root / "manifest.json", manifest)
        _ledger(root, "V22_FINAL_TEST_A_PARAMETER_DERIVATION_FROM_CERTIFIED_TRAIN_HANDOFF", protected_labels_read=True)

        bundle = read_json(paths["bundle"])
        cutoff = pd.Timestamp(bundle["input"]["cutoff"])
        history = _history(paths["train"], bundle)
        metadata = _metadata(paths["metadata"])
        if set(metadata.sample_id) & set(history.sample_id):
            raise ContractError("test-A IDs overlap final QRF training IDs")
        model_manifest = _model_manifest(root, paths["model_dir"], bundle)
        parameters = _parameters(development, history, cutoff, root)
        medians = _medians(history, cutoff, root)
        qrf_report = _worker(model_manifest, paths["evaluation"], root / "qrf.npz")
        directions = _directions(metadata, root / "qrf.npz", medians, read_json(model_manifest))
        directions.to_csv(root / "directions.csv", index=False, mode="x", float_format="%.17g", lineterminator="\n")
        v10 = pd.read_csv(paths["v10"], dtype={"sample_id": str}, float_precision="round_trip")
        prediction = apply_shrink(v10, directions, {s: parameters["spouts"][s]["lambda"] for s in ("1", "2")})
        submission = root / "submission"
        submission.mkdir()
        result_path = submission / "result.csv"
        result_path.write_bytes(serialize_six_decimals(prediction))
        validate_submission(pd.read_csv(result_path, dtype={"sample_id": str}), metadata.sample_id)
        archive = pack_submission(result_path, stage=STAGE, team_name=TEAM, output_dir=submission, expected_ids=metadata.sample_id)
        original_v10 = pd.read_csv(paths["v10_serialized"], dtype={"sample_id": str})
        serialized = pd.read_csv(result_path, dtype={"sample_id": str})
        if not np.array_equal(serialized.pred_tap_iron.to_numpy(), original_v10.pred_tap_iron.to_numpy()):
            raise ContractError("serialized V22 iron differs from original certified V10")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "cold-check", "--run", str(root)],
            cwd=REPO,
            check=True,
        )
        if read_json(root / "cold_validation.json").get("status") != "PASS":
            raise ContractError("final V22 cold validation did not pass")
        desktop.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("rb") as source, desktop.open("xb") as target:
            shutil.copyfileobj(source, target)
        if file_sha256(desktop) != file_sha256(archive):
            raise ContractError("desktop package digest differs from local final package")
        delta = prediction.pred_tap_time_len.to_numpy() - v10.pred_tap_time_len.to_numpy()
        completion = {
            "status": "PASS_READY_FOR_USER_PLATFORM_UPLOAD",
            "candidate": "V22_CAUSAL_H2_QRF_SHRINK",
            "stage": STAGE,
            "rows": len(prediction),
            "result_sha256": file_sha256(result_path),
            "archive": str(archive.resolve()),
            "archive_sha256": file_sha256(archive),
            "desktop": str(desktop),
            "desktop_sha256": file_sha256(desktop),
            "desktop_write": 1,
            "platform_uploads": 0,
            "test_targets_read": False,
            "test_scores_used_for_parameter_fit": False,
            "iron_exact_v10_serialized": True,
            "fresh_qrf_support_inference": True,
            "qrf_worker_zero_fit": qrf_report["zero_fit"],
            "fit_counts": {
                "forest": 0,
                "preprocessor": 0,
                "internal_trees": 0,
                "iron_model": 0,
                "iron_LAD": 0,
                "lambda_slots": 2,
                "lambda_fit_attempts": parameters["fit_counts"]["fit_attempts"],
                "median_statistics": 2,
                "platform_candidates": 1,
                "submission_zips": 1,
            },
            "lambdas": {s: parameters["spouts"][s]["lambda"] for s in ("1", "2")},
            "medians": {s: medians["certificates"][s]["median"] for s in ("1", "2")},
            "changed_rows": int(np.count_nonzero(delta)),
            "time_delta_quantiles": dict(zip(
                ("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"),
                np.quantile(delta, (0, .01, .05, .25, .5, .75, .95, .99, 1)).tolist(),
            )),
            "cold_validation_sha256": file_sha256(root / "cold_validation.json"),
        }
        atomic_write_json(root / "completion.json", completion)
        return completion
    except Exception as exc:
        if not (root / "failure.json").exists():
            atomic_write_json(root / "failure.json", {
                "status": "FAILED_PRESERVED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "desktop_created": desktop.exists(),
                "platform_uploads": 0,
            })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build")
    make.add_argument("--development-run", type=Path, default=DEVELOPMENT)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--desktop", type=Path, default=DESKTOP)
    check = commands.add_parser("cold-check")
    check.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = (
        build(args.development_run, args.output, args.desktop)
        if args.command == "build"
        else cold_check(args.run)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
