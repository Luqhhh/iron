"""V22 lifecycle: trusted P0, target-free banks, causal scalars, and composition.

The commands are deliberately separable.  ``p0`` must pass before any warmup
fit is authorized.  The other commands consume explicit manifests and never
discover test targets or platform feedback.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

from bf_tap.artifacts import atomic_write_json, file_identities, file_sha256, runtime_environment, stable_digest, verify_file_identities
from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.qrf_support_shrink import (
    CANDIDATE,
    REFERENCE,
    REPLAY_CONTROL,
    BANK_REQUIRED,
    LambdaBudget,
    acceptance,
    apply_shrink,
    build_direction_bank,
    build_prediction_directions,
    fixed_cutoff_medians,
    replay_v21,
    select_outer_oof,
    serialize_six_decimals,
    verify_h2_bank,
)
from bf_tap.optimization.component_export import META
from bf_tap.optimization.dual_ratio_common import frame, score as score_grid, week_intervals
from bf_tap.optimization.qrf_time_run import (
    builder_for,
    fold_for,
    outer_samples,
    pack,
    raw_matrix,
    restore_manifest,
    verify as verify_qrf_manifest,
)
from bf_tap.optimization.refresh_factorial import stamp
from bf_tap.optimization.validation import error_contributions


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/optimization_v0_22/experiment.yaml"
SCOPE = REPO / "configs/optimization_v0_22/access_scope.yaml"
PROTECTION = REPO / "configs/protection.yaml"
SOURCE_FILES = (
    CONFIG,
    SCOPE,
    PROTECTION,
    REPO / "src/bf_tap/optimization/qrf_support_shrink.py",
    REPO / "src/bf_tap/optimization/qrf_time_run.py",
    REPO / "src/bf_tap/optimization/dual_ratio_common.py",
    REPO / "scripts/optimization_v22_run.py",
    REPO / "scripts/optimization_v22_cold_check.py",
    REPO / "workers/qrf_v015/warmup_v022.py",
    REPO / "workers/qrf_v015/inference_v022.py",
    REPO / "workers/qrf_v015/qrf_model.py",
    REPO / "workers/qrf_v015/preprocessing.py",
    REPO / "workers/qrf_v015/worker.py",
    REPO / "workers/qrf_v015/uv.lock",
    REPO / "uv.lock",
)


class P0Blocked(ContractError):
    pass


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def registration() -> dict:
    reg = load_yaml(CONFIG)
    expected = {
        "development_forest_fits": 2,
        "development_preprocessor_fits": 2,
        "development_internal_trees": 512,
        "development_lambda_slots": 12,
        "development_cutoff_spout_medians": 16,
        "final_forest_fits": 0,
        "final_preprocessor_fits": 0,
        "final_lambda_slots": 2,
        "final_cutoff_spout_medians": 2,
        "iron_model_fits": 0,
        "iron_LAD_fits": 0,
        "platform_candidates": 0,
        "conditional_final_platform_candidates": 1,
    }
    if (
        reg["candidate"] != "V22_CAUSAL_H2_QRF_SHRINK"
        or reg["base_commit"] != "9c6cf42d16e369d8ce632806670d1bb170482b91"
        or reg["protocol"] != "QRF_FULLTRAIN_LEAF_v1"
        or reg["warmup"]["model_cutoffs"] != [4, 5]
        or reg["h2_bank"]["model_cutoffs"] != list(range(4, 11))
        or reg["lambda"]["minimum_oof_rows_per_spout"] != 100
        or reg["median"]["window_days"] != 60
        or reg["budget"] != expected
    ):
        raise ContractError("frozen V22 registration differs")
    return reg


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _identity_entries(value) -> list[dict]:
    found: list[dict] = []
    if isinstance(value, dict):
        if set(("path", "sha256", "bytes")) <= set(value):
            found.append(value)
        for child in value.values():
            found.extend(_identity_entries(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_identity_entries(child))
    return found


def _trusted_entry(anchor: Path, target: Path, digest: str) -> dict:
    if not anchor.is_file():
        raise P0Blocked(f"BLOCKED_MISSING_SOURCE: {anchor}")
    value = read_json(anchor)
    candidates = _identity_entries(value)
    resolved = target.resolve()
    matches = [
        entry for entry in candidates
        if Path(entry["path"]).resolve() == resolved and entry["sha256"] == digest
    ]
    if not matches:
        raise P0Blocked(f"BLOCKED_UNTRUSTED_SOURCE: {target} is not bound by {anchor}")
    verify_file_identities({str(target): matches[0]})
    return matches[0]


def _find_digest(paths: list[str], digest: str, suffix: str | None = None) -> Path | None:
    matches: list[Path] = []
    for name in paths:
        root = (REPO / name).resolve()
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and (suffix is None or path.suffix == suffix):
                try:
                    if file_sha256(path) == digest:
                        matches.append(path)
                except OSError:
                    continue
    unique = sorted(set(matches))
    if len(unique) > 1:
        # Identical bytes in multiple preserved locations are allowed; choose
        # deterministically and record every location in the P0 result.
        return unique[0]
    return unique[0] if unique else None


def _zip_payload(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["result.csv"]:
            raise ContractError("original archive must contain only result.csv")
        return archive.read("result.csv")


def _ledger(root: Path, purpose: str, *, protected_labels_read: bool = False) -> None:
    scope = load_yaml(SCOPE)
    path = REPO / scope["new_ledger"]
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    event = {
        "at": datetime.now(timezone.utc).isoformat(),
        "purpose": purpose,
        "authorization": scope["authorization"],
        "run_manifest_sha256": file_sha256(root / "manifest.json"),
        "previous_sha256": stable_digest(prior[-1]) if prior else None,
        "protected_labels_read": bool(protected_labels_read),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def register_history(run: Path, history_path: Path) -> dict:
    root = run.resolve()
    if read_json(root / "p0_original_replay.json").get("status") != "PASS":
        raise ContractError("history registration is blocked until P0 passes")
    manifest = root / "history_input_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    _ledger(root, "FROZEN_HISTORY_IDENTITY_CAPTURE_BEFORE_SEMANTIC_LABEL_READ")
    value = {
        "kind": "V22_FROZEN_HISTORY_INPUT_v1",
        "parent_manifest_sha256": file_sha256(root / "manifest.json"),
        "protection_contract_sha256": file_sha256(PROTECTION),
        "history": file_identities({"frozen_history": history_path})["frozen_history"],
        "registered_before_label_read": True,
    }
    atomic_write_json(manifest, value)
    return value


def _authorize_history(run: Path, history_path: Path, purpose: str) -> None:
    root = run.resolve()
    if read_json(root / "p0_original_replay.json").get("status") != "PASS":
        raise ContractError("protected history access is blocked until P0 passes")
    registered = read_json(root / "history_input_manifest.json")
    if (
        registered.get("kind") != "V22_FROZEN_HISTORY_INPUT_v1"
        or registered.get("parent_manifest_sha256") != file_sha256(root / "manifest.json")
        or registered.get("protection_contract_sha256") != file_sha256(PROTECTION)
        or Path(registered["history"]["path"]).resolve() != history_path.resolve()
    ):
        raise ContractError("frozen protected-history manifest identity differs")
    verify_file_identities({"frozen_history": registered["history"]})
    _ledger(root, purpose, protected_labels_read=True)


def _require_run_output(run: Path, output: Path) -> None:
    if not output.resolve().is_relative_to(run.resolve()):
        raise ContractError("V22 derived artifacts must remain inside their private run directory")


def _authorize_model(run: Path, model_manifest_path: Path) -> dict:
    root = run.resolve()
    p0 = read_json(root / "p0_original_replay.json")
    if p0.get("status") != "PASS":
        raise ContractError("model restoration is blocked until P0 passes")
    expected = {
        (Path(value["path"]).resolve(), value["sha256"])
        for value in p0.get("certified_historical_qrf_models", {}).values()
    }
    registry = root / "models_complete.json"
    if registry.exists():
        for value in read_json(registry).get("models", {}).values():
            expected.add((Path(value["path"]).resolve(), value["sha256"]))
    identity = (model_manifest_path.resolve(), file_sha256(model_manifest_path))
    if identity not in expected:
        raise ContractError("QRF model manifest is not bound by passed P0 or models_complete")
    model = read_json(model_manifest_path)
    if model.get("kind") != "V22_CERTIFIED_QRF_MODEL_v1":
        raise ContractError("unknown certified QRF model manifest")
    return model


def _not_run(root: Path, *, status: str, reason: str) -> None:
    payload = {"status": "NOT_RUN", "blocked_by": reason}
    for name in (
        "warmup_fit_intents.json", "h2_banks.json", "median_parameters.json",
        "lambda_parameters.json", "predictions_complete.json", "metrics.json",
        "v21_replay_diagnostics.json", "acceptance.json", "cold_validation.json",
    ):
        if not (root / name).exists():
            atomic_write_json(root / name, payload)
    if not (root / "fit_counts.json").exists():
        atomic_write_json(root / "fit_counts.json", {
            "status": "NOT_RUN", "forest_fit_attempts": 0,
            "preprocessor_fit_attempts": 0, "internal_trees": 0,
            "lambda_fit_attempts": 0, "median_statistics": 0,
            "iron_model_fits": 0, "platform_candidates": 0,
        })
    atomic_write_json(root / "completion.json", {
        "status": status,
        "G0": "BLOCKED" if status.startswith("BLOCKED") else "NOT_RUN",
        "G1": "NOT_RUN",
        "reason": reason,
        "failed_evidence_preserved": True,
        "platform_uploads": 0,
        "desktop_writes": 0,
        "public_pushes": 0,
    })


def _certified_models(root: Path, reg: dict) -> dict:
    qrf_root = REPO / reg["trusted_sources"]["qrf_development_run"]
    completion_path = qrf_root / "completion.json"
    completion = read_json(completion_path)
    if completion.get("G0") != "PASS" or read_json(qrf_root / "cold_validation.json").get("status") != "PASS":
        raise P0Blocked("BLOCKED_QRF_ENGINEERING_EVIDENCE_NOT_PASS")
    models_complete = read_json(qrf_root / "models_complete.json")
    output = {}
    folder = root / "p0/models"
    folder.mkdir(parents=True, exist_ok=True)
    for month in reg["development_origins"]:
        model_path = qrf_root / "models" / str(month)
        bundle_path = model_path / "bundle.json"
        expected = models_complete.get(str(month))
        if expected is None or not bundle_path.is_file() or file_sha256(bundle_path) != expected:
            raise P0Blocked(f"BLOCKED_MISSING_SOURCE: certified QRF cutoff {month}")
        _trusted_entry(completion_path, bundle_path, expected)
        bundle = read_json(bundle_path)
        if (
            bundle.get("protocol") != reg["protocol"]
            or bundle.get("parameters") != reg["qrf_parameters"]
            or len(set(bundle["input"]["ids"])) != reg["expected_training_rows"][month]
        ):
            raise P0Blocked(f"BLOCKED_QRF_PROTOCOL_OR_TRAINING_IDENTITY: cutoff {month}")
        for name in ("forest.joblib", "preprocessor.json"):
            path = model_path / name
            digest = bundle["forest_sha256" if name == "forest.joblib" else "preprocessor_sha256"]
            if file_sha256(path) != digest:
                raise P0Blocked(f"BLOCKED_CHANGED_MODEL_ASSET: {path}")
            _trusted_entry(completion_path, path, digest)
        manifest = {
            "kind": "V22_CERTIFIED_QRF_MODEL_v1",
            "protocol": reg["protocol"],
            "cutoff_month": month,
            "cutoff_ns": bundle["input"]["cutoff_ns"],
            "model_path": str(model_path.resolve()),
            "bundle_sha256": expected,
            "unique_training_rows": reg["expected_training_rows"][month],
            "training_identity_sha256": stable_digest(bundle["input"]["ids"]),
            "upstream_completion": str(completion_path.resolve()),
            "upstream_completion_sha256": file_sha256(completion_path),
        }
        atomic_write_json(folder / f"{month}.json", manifest)
        output[str(month)] = {
            "path": str((folder / f"{month}.json").resolve()),
            "sha256": file_sha256(folder / f"{month}.json"),
            "upstream_bundle_sha256": expected,
        }
    return output


def _certified_final_components(reg: dict) -> dict:
    stage_root = REPO / reg["trusted_sources"]["stage_inference_run"]
    receipt_path = stage_root / "receipt.json"
    p0_path = stage_root / "p0.json"
    cold_path = stage_root / "cold_validation.json"
    component_path = stage_root / "component_bundle.json"
    if not all(path.is_file() for path in (receipt_path, p0_path, cold_path, component_path)):
        raise P0Blocked("BLOCKED_MISSING_SOURCE: certified final V10 component evidence")
    old_p0, cold = read_json(p0_path), read_json(cold_path)
    if old_p0.get("status") != "PASS" or cold.get("status") != "PASS" or not cold.get("V10_original_result_bytes_exact"):
        raise P0Blocked("BLOCKED_FINAL_V10_COLD_EVIDENCE_NOT_PASS")
    component_sha = file_sha256(component_path)
    if old_p0.get("component_bundle_sha256") != component_sha:
        raise P0Blocked("BLOCKED_FINAL_COMPONENT_MANIFEST_IDENTITY")
    _trusted_entry(receipt_path, component_path, component_sha)
    component = read_json(component_path)
    if (
        component.get("kind") != "V10_CERTIFIED_FINAL_COMPONENT_REFERENCES_v1"
        or component.get("source_v10_zip_sha256") != reg["trusted_sources"]["v10_zip_sha256"]
    ):
        raise P0Blocked("BLOCKED_FINAL_COMPONENT_PROTOCOL")
    verify_file_identities(component["assets"])
    verify_file_identities(component["provenance_receipts"])
    qrf_model = Path(component["qrf_model"])
    if file_sha256(qrf_model / "bundle.json") != component["qrf_bundle_sha256"]:
        raise P0Blocked("BLOCKED_FINAL_QRF_BUNDLE_IDENTITY")
    qrf = read_json(qrf_model / "bundle.json")
    if qrf.get("protocol") != reg["protocol"] or qrf.get("parameters") != reg["qrf_parameters"]:
        raise P0Blocked("BLOCKED_FINAL_QRF_PROTOCOL")
    coefficient = read_json(component["iron_coefficient"])
    if (
        coefficient.get("target") != "tap_iron"
        or not 0 <= coefficient.get("beta", -1) <= 1
        or not coefficient.get("certificate", {}).get("smallest_minimizer_verified")
    ):
        raise P0Blocked("BLOCKED_FINAL_IRON_COEFFICIENT_IDENTITY")
    return {
        "component_bundle": str(component_path),
        "component_bundle_sha256": component_sha,
        "final_qrf_model": str(qrf_model),
        "final_qrf_bundle_sha256": component["qrf_bundle_sha256"],
        "final_unique_training_rows": len(set(qrf["input"]["ids"])),
        "iron_model": component["recency_iron"],
        "iron_coefficient": component["iron_coefficient"],
        "original_v10_result_bytes_exact": True,
        "old_test_a_test_b_stage_inference_available": True,
    }


def p0(output: Path, *, allow_missing_v21_original: bool = False) -> dict:
    reg = registration()
    root = output.resolve()
    local_runs = (REPO / "local/runs").resolve()
    if not root.is_relative_to(local_runs):
        raise ContractError("V22 private runs must remain under ignored local/runs")
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    scope = load_yaml(SCOPE)
    protection = load_yaml(PROTECTION)
    if not scope["holdout_consumed"] or not read_json(REPO / "EVIDENCE_STATUS.json")["holdout_consumed"]:
        raise ContractError("V22 requires the consumed retrospective protection scope")
    source_paths = [path for path in SOURCE_FILES if path.is_file()]
    manifest = {
        "kind": "V22_REGISTRATION_MANIFEST_v1",
        "registration": reg,
        "scope": scope,
        "protection": protection,
        "base_commit_verified": _git("merge-base", "--is-ancestor", reg["base_commit"], "HEAD") == "",
        "code_commit": _git("rev-parse", "HEAD"),
        "code_branch": _git("branch", "--show-current"),
        "sources": file_identities({str(path): path for path in source_paths}),
        "environment": runtime_environment(),
        "test_targets_read": False,
        "test_scores_used_for_parameter_fit": False,
        "design_informed_by_prior_platform_observations": True,
        "november_globally_consumed": True,
        "explicit_user_waiver": (
            scope.get("explicit_user_waivers", {}).get("missing_v21_original_zip")
            if allow_missing_v21_original else None
        ),
    }
    atomic_write_json(root / "manifest.json", manifest)
    _ledger(root, "P0_ORIGINAL_ARTIFACT_AND_LIFECYCLE_AUDIT")
    audit = {
        "status": "RUNNING",
        "original_artifacts_verified": False,
        "protected_target_reads": 0,
        "new_fits": 0,
    }
    try:
        models = _certified_models(root, reg)
        final_components = _certified_final_components(reg)
        search = reg["trusted_sources"]["artifact_search_roots"]
        v10_zip = _find_digest(search, reg["trusted_sources"]["v10_zip_sha256"], ".zip")
        v21_zip = _find_digest(search, reg["trusted_sources"]["v21_zip_sha256"], ".zip")
        if v10_zip is None:
            raise P0Blocked("BLOCKED_MISSING_SOURCE: original V10 ZIP")
        v10_payload = _zip_payload(v10_zip)
        if stable_digest(v10_payload) == reg["trusted_sources"]["v10_result_sha256"]:
            # stable_digest is JSON-oriented and intentionally not a file hash.
            raise ContractError("unexpected digest implementation collision")
        import hashlib
        if hashlib.sha256(v10_payload).hexdigest() != reg["trusted_sources"]["v10_result_sha256"]:
            raise P0Blocked("BLOCKED_CHANGED_SOURCE: original V10 result.csv")
        composition = REPO / reg["trusted_sources"]["v10_composition_run"]
        receipt = composition / "receipt.json"
        _trusted_entry(receipt, v10_zip, reg["trusted_sources"]["v10_zip_sha256"])
        if v21_zip is None and not allow_missing_v21_original:
            audit.update({
                "status": "BLOCKED_MISSING_ORIGINAL_ARTIFACT",
                "reason": "original V21 ZIP with registered SHA-256 is absent; restore bytes, do not rebuild/retrain",
                "v10_zip": str(v10_zip),
                "v10_result_bytes_replayed": True,
                "v21_expected_sha256": reg["trusted_sources"]["v21_zip_sha256"],
                "certified_historical_qrf_models": models,
                "certified_final_components": final_components,
            })
            atomic_write_json(root / "p0_original_replay.json", audit)
            _not_run(root, status="BLOCKED_MISSING_ORIGINAL_ARTIFACT", reason=audit["reason"])
            return audit

        # The rule replay always remains mandatory.  An explicit user waiver may
        # waive only the unavailable original ZIP byte comparison; it never
        # changes or silently "verifies" the historical archive identity.
        stage_root = REPO / reg["trusted_sources"]["stage_inference_run"]
        stage_receipt = stage_root / "receipt.json"
        stage_evidence = stage_root / "test_a_V10_full_precision.csv.evidence"
        v10_replay = stage_evidence / "result_replay.csv"
        qrf_report = stage_evidence / "qrf.npz.json"
        test_samples = REPO / "初赛数据集/test/test_a_samples.csv"
        # train/ and test/ carry byte-identical copies; the archived v0.8
        # manifest binds the test/ path, so use that exact trusted identity.
        history_path = REPO / "初赛数据集/test/tap_history_train.csv"
        for path in (v10_replay, qrf_report):
            _trusted_entry(stage_receipt, path, file_sha256(path))
        _trusted_entry(stage_root / "manifest.json", test_samples, file_sha256(test_samples))
        qrf_source_manifest = REPO / reg["trusted_sources"]["qrf_development_run"] / "manifest.json"
        source_v8_manifest = REPO / read_json(qrf_source_manifest)["registration"]["source_v8"] / "manifest.json"
        _trusted_entry(source_v8_manifest, history_path, file_sha256(history_path))
        if v10_replay.read_bytes() != v10_payload:
            raise P0Blocked("BLOCKED_V10_REPLAY_BYTES_DIFFER")
        frame = pd.read_csv(v10_replay, dtype={"sample_id": str}, float_precision="round_trip")
        queries = pd.read_csv(test_samples, dtype={"sample_id": str})
        _ledger(root, "P0_V21_RULE_REPLAY_PROTECTED_HISTORY", protected_labels_read=True)
        history = pd.read_csv(history_path, dtype={"sample_id": str})
        neighbors = read_json(qrf_report)["neighbors"]
        effective = [row["effective_neighbors"] for row in neighbors]
        replay, diagnostics = replay_v21(
            frame, queries, history, effective,
            cutoff=pd.Timestamp("2024-12-01 01:44:00", tz="Asia/Shanghai"),
        )
        replay_payload = serialize_six_decimals(replay)
        replay_path = root / "p0/v21_rule_replay_result.csv"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_bytes(replay_payload)
        if v21_zip is not None:
            v21_payload = _zip_payload(v21_zip)
            if replay_payload != v21_payload:
                raise P0Blocked("BLOCKED_V21_TWO_STAGE_SIX_DECIMAL_REPLAY_DIFFERS")
        test_b = stage_root / "test_b_V10_full_precision.csv.evidence/completion.json"
        if read_json(test_b).get("status") != "PASS":
            raise P0Blocked("BLOCKED_OLD_TEST_B_ENGINEERING_PREVIEW")
        audit.update({
            "status": "PASS",
            "original_artifacts_verified": v21_zip is not None,
            "protected_target_reads": 1,
            "v10_zip": str(v10_zip),
            "v21_zip": None if v21_zip is None else str(v21_zip),
            "v21_expected_sha256": reg["trusted_sources"]["v21_zip_sha256"],
            "v21_original_zip_verified": v21_zip is not None,
            "v21_original_zip_byte_comparison": "PASS" if v21_zip is not None else "EXPLICIT_USER_WAIVER",
            "explicit_user_waiver": (
                scope["explicit_user_waivers"]["missing_v21_original_zip"]
                if v21_zip is None else None
            ),
            "v10_result_bytes_replayed": True,
            "v21_two_stage_six_decimal_rule_replayed": True,
            "v21_rule_replay_result": {
                "path": str(replay_path),
                "sha256": file_sha256(replay_path),
            },
            "v21_diagnostics": diagnostics,
            "old_test_b_engineering_preview": "PASS",
            "certified_historical_qrf_models": models,
            "certified_final_components": final_components,
        })
        atomic_write_json(root / "p0_original_replay.json", audit)
        return audit
    except P0Blocked as exc:
        audit.update({"status": "BLOCKED", "reason": str(exc)})
        if not (root / "p0_original_replay.json").exists():
            atomic_write_json(root / "p0_original_replay.json", audit)
        _not_run(root, status="BLOCKED_P0", reason=str(exc))
        return audit
    except Exception as exc:
        audit.update({"status": "FAILED_PRESERVED", "error_type": type(exc).__name__, "reason": str(exc)})
        if not (root / "p0_original_replay.json").exists():
            atomic_write_json(root / "p0_original_replay.json", audit)
        _not_run(root, status="FAILED_P0_PRESERVED", reason=repr(exc))
        raise


def _worker_infer(model_manifest: Path, handoff: Path, output: Path) -> dict:
    worker = REPO / "workers/qrf_v015"
    subprocess.run(
        [
            str(worker / ".venv/bin/python"), str(worker / "inference_v022.py"),
            "--model-manifest", str(model_manifest.resolve()),
            "--model-manifest-sha256", file_sha256(model_manifest),
            "--input", str(handoff.resolve()), "--input-sha256", file_sha256(handoff),
            "--output", str(output.resolve()),
        ],
        check=True,
        cwd=REPO,
    )
    report = read_json(str(output) + ".json")
    if report.get("status") != "PASS" or any(report["zero_fit"].values()):
        raise ContractError("V22 QRF inference did not complete with zero fits")
    return report


def _write_frame(path: Path, value: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value.to_csv(path, index=False, mode="x", float_format="%.17g", lineterminator="\n")


def _prepare_development_inputs(root: Path, reg: dict) -> tuple[dict, Path, dict, dict]:
    qrf_root = REPO / reg["trusted_sources"]["qrf_development_run"]
    old_manifest = restore_manifest(qrf_root)
    verify_qrf_manifest(old_manifest)
    history_source = Path(old_manifest["inventory"]["11"]["base_components"]["OR"]["path"]) / "history_snapshot.csv"
    expected_history_sha = old_manifest["inventory"]["11"]["base_components"]["OR"]["history_snapshot_sha256"]
    if file_sha256(history_source) != expected_history_sha:
        raise ContractError("certified frozen history snapshot identity differs")
    register_history(root, history_source)
    _authorize_history(root, history_source, "BUILD_V22_CAUSAL_HANDOFFS_FROM_CERTIFIED_FOLDS")
    builder = builder_for(old_manifest)
    train_inputs: dict[str, dict] = {}
    h2_inputs: dict[str, dict] = {}
    outer_inputs: dict[str, dict] = {}
    h2_metadata: dict[str, str] = {}
    for month in range(4, 11):
        fold, _ = fold_for(old_manifest, month)
        history = fold[1]["OR"][1]
        if len(history) != reg["expected_training_rows"][month]:
            raise ContractError(f"certified cutoff {month} training rows differ")
        if month in (4, 5):
            matrix = raw_matrix(builder, history[META], fold, training=True)
            train_path = root / "features" / str(month) / "train.npz"
            train_inputs[str(month)] = pack(train_path, matrix, history[META], stamp(month), history)
        _, next_oof = fold_for(old_manifest, month + 1)
        query = next_oof[[*META, "label_available_at"]].copy()
        start, end = stamp(month + 1), stamp(month + 2)
        if not ((query.reference_time >= start) & (query.reference_time < end)).all():
            raise ContractError(f"cutoff {month} query is not genuine calendar H2")
        matrix = raw_matrix(builder, query[META], fold)
        handoff = root / "features" / str(month) / "h2.npz"
        h2_inputs[str(month)] = pack(handoff, matrix, query[META], stamp(month))
        metadata_path = root / "metadata" / f"h2-{month}.csv"
        _write_frame(metadata_path, query)
        h2_metadata[str(month)] = str(metadata_path)
        builder.cache.clear()
    for month in range(6, 12):
        samples = outer_samples(old_manifest, month)
        metadata_path = root / "metadata" / f"outer-{month}.csv"
        _write_frame(metadata_path, samples)
        handoff = qrf_root / "features" / str(month) / "evaluation.npz"
        info = read_json(str(handoff) + ".json")
        if info["ids"] != samples.sample_id.astype(str).tolist() or info["sha256"] != file_sha256(handoff):
            raise ContractError(f"certified outer cutoff {month} feature handoff differs")
        outer_inputs[str(month)] = {
            "metadata": str(metadata_path),
            "handoff": str(handoff),
            "handoff_sha256": file_sha256(handoff),
        }
    input_manifest = {
        "kind": "V22_DEVELOPMENT_HANDOFFS_v1",
        "source_qrf_manifest": {"path": str(qrf_root / "manifest.json"), "sha256": file_sha256(qrf_root / "manifest.json")},
        "frozen_history": {"path": str(history_source), "sha256": file_sha256(history_source)},
        "train_inputs": train_inputs,
        "h2_inputs": h2_inputs,
        "h2_metadata": h2_metadata,
        "outer_inputs": outer_inputs,
    }
    atomic_write_json(root / "input_manifest.json", input_manifest)
    return old_manifest, history_source, input_manifest, outer_inputs


def _warmup_models(root: Path, reg: dict, input_manifest: dict) -> dict:
    existing = []
    for path in (REPO / "local/runs").glob("**/bundle.json"):
        try:
            value = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        cutoff_ns = value.get("input", {}).get("cutoff_ns")
        if (
            value.get("protocol") == reg["protocol"]
            and value.get("parameters") == reg["qrf_parameters"]
            and cutoff_ns in (stamp(4).value, stamp(5).value)
        ):
            existing.append({"path": str(path), "sha256": file_sha256(path), "cutoff_ns": cutoff_ns})
    if existing:
        raise ContractError("equivalent April/May QRF bundle found; certify/reuse before spending warmup budget")
    registration_path = root / "warmup_registration.json"
    registration_value = {
        "kind": "V22_QRF_WARMUP_REGISTRATION_v1",
        "adapter_protocol": reg["warmup"]["adapter_protocol"],
        "qrf_protocol": reg["protocol"],
        "parameters": reg["qrf_parameters"],
        "allowed_months": [4, 5],
        "expected_training_rows": {str(k): reg["expected_training_rows"][k] for k in (4, 5)},
        "train_inputs": {str(k): input_manifest["train_inputs"][str(k)] for k in (4, 5)},
        "output_paths": {str(k): str((root / "models" / str(k)).resolve()) for k in (4, 5)},
        "reuse_audit": {str(k): "NO_EQUIVALENT_CERTIFIED_MODEL" for k in (4, 5)},
        "registry_matches": existing,
        "parent_manifest_sha256": file_sha256(root / "manifest.json"),
    }
    atomic_write_json(registration_path, registration_value)
    worker = REPO / "workers/qrf_v015"
    for month in (4, 5):
        subprocess.run(
            [
                str(worker / ".venv/bin/python"), str(worker / "warmup_v022.py"),
                "--train", str((root / "features" / str(month) / "train.npz").resolve()),
                "--registration-manifest", str(registration_path.resolve()),
                "--registration-sha256", file_sha256(registration_path),
                "--month", str(month), "--output", str((root / "models" / str(month)).resolve()),
            ],
            check=True,
            cwd=REPO,
        )
    p0_models = read_json(root / "p0_original_replay.json")["certified_historical_qrf_models"]
    models = dict(p0_models)
    registry_root = root / "model_registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    for month in (4, 5):
        model_path = root / "models" / str(month)
        bundle_path = model_path / "bundle.json"
        bundle = read_json(bundle_path)
        if (
            bundle.get("fit_counts") != {"forest": 1, "preprocessor": 1, "internal_trees": 256}
            or bundle.get("unique_training_rows") != reg["expected_training_rows"][month]
            or bundle.get("protocol") != reg["protocol"]
            or bundle.get("parameters") != reg["qrf_parameters"]
        ):
            raise ContractError(f"warmup cutoff {month} fitted bundle differs from registration")
        manifest = {
            "kind": "V22_CERTIFIED_QRF_MODEL_v1",
            "protocol": reg["protocol"],
            "cutoff_month": month,
            "cutoff_ns": bundle["input"]["cutoff_ns"],
            "model_path": str(model_path.resolve()),
            "bundle_sha256": file_sha256(bundle_path),
            "unique_training_rows": reg["expected_training_rows"][month],
            "training_identity_sha256": stable_digest(bundle["input"]["ids"]),
            "upstream_completion": str(registration_path.resolve()),
            "upstream_completion_sha256": file_sha256(registration_path),
        }
        manifest_path = registry_root / f"{month}.json"
        atomic_write_json(manifest_path, manifest)
        models[str(month)] = {"path": str(manifest_path.resolve()), "sha256": file_sha256(manifest_path)}
    atomic_write_json(root / "models_complete.json", {"kind": "V22_MODEL_REGISTRY_v1", "models": models})
    fit_intents = {
        "status": "PASS", "forest_fit_attempts": 2, "forest_fit_completed": 2,
        "preprocessor_fit_attempts": 2, "preprocessor_fit_completed": 2,
        "internal_trees": 512, "equivalent_certified_models_found": 0,
        "registration_sha256": file_sha256(registration_path),
    }
    atomic_write_json(root / "warmup_fit_intents.json", fit_intents)
    return models


def _model_manifest_path(models: dict, month: int) -> Path:
    path = Path(models[str(month)]["path"])
    if file_sha256(path) != models[str(month)]["sha256"]:
        raise ContractError(f"registered QRF model manifest {month} changed")
    return path


def _infer_and_build_banks(
    root: Path, reg: dict, history_source: Path, input_manifest: dict,
    outer_inputs: dict, models: dict,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    median_files: dict[str, str] = {}
    qrf_tasks: list[dict] = []
    for month in range(4, 12):
        path = root / "medians" / f"{month}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        make_medians(root, history_source, stamp(month).isoformat(), path)
        median_files[str(month)] = str(path)
    bank_parts = []
    bank_files: dict[str, dict] = {}
    for month in range(4, 11):
        model_manifest = _model_manifest_path(models, month)
        handoff = root / "features" / str(month) / "h2.npz"
        qrf_path = root / "qrf" / f"h2-{month}.npz"
        qrf_path.parent.mkdir(parents=True, exist_ok=True)
        _worker_infer(model_manifest, handoff, qrf_path)
        output = root / "banks" / f"h2-{month}.csv"
        status = make_bank(
            root, Path(input_manifest["h2_metadata"][str(month)]), qrf_path,
            Path(median_files[str(month)]), model_manifest, output,
        )
        part = pd.read_csv(output, dtype={"sample_id": str, "spout_no": str}, float_precision="round_trip")
        bank_parts.append(part)
        bank_files[str(month)] = {"path": str(output), "sha256": file_sha256(output), **status}
        qrf_tasks.append({
            "id": f"h2-{month}", "model_manifest": str(model_manifest),
            "model_manifest_sha256": file_sha256(model_manifest), "input": str(handoff),
            "input_sha256": file_sha256(handoff), "expected_output": str(qrf_path),
        })
    bank = pd.concat(bank_parts, ignore_index=True)
    verify_h2_bank(bank, reg["expected_training_rows"])
    bank_path = root / "h2_bank.csv"
    _write_frame(bank_path, bank)
    atomic_write_json(root / "h2_banks.json", {
        "status": "PASS", "forecasts": 7, "rows": len(bank),
        "target_columns_present": False, "combined": {"path": str(bank_path), "sha256": file_sha256(bank_path)},
        "parts": bank_files,
    })
    for month in range(6, 12):
        model_manifest = _model_manifest_path(models, month)
        handoff = Path(outer_inputs[str(month)]["handoff"])
        qrf_path = root / "qrf" / f"outer-{month}.npz"
        _worker_infer(model_manifest, handoff, qrf_path)
        qrf_tasks.append({
            "id": f"outer-{month}", "model_manifest": str(model_manifest),
            "model_manifest_sha256": file_sha256(model_manifest), "input": str(handoff),
            "input_sha256": file_sha256(handoff), "expected_output": str(qrf_path),
        })
    atomic_write_json(root / "median_parameters.json", {
        "status": "PASS", "groups": 16, "cutoffs": median_files,
        "fallback_groups": sum(
            int(cert["fallback"])
            for path in median_files.values() for cert in read_json(path)["certificates"].values()
        ),
    })
    return bank, median_files, qrf_tasks


def _derive_all_lambdas(root: Path, history_source: Path, bank: pd.DataFrame) -> dict:
    _authorize_history(root, history_source, "DERIVE_ALL_OUTER_SPOUT_LAMBDAS")
    history = pd.read_csv(history_source, dtype={"sample_id": str, "spout_no": str}, float_precision="round_trip")
    budget = LambdaBudget(root / "lambda_slots")
    outer = {}
    for month in range(6, 12):
        folder = root / "lambdas" / str(month)
        folder.mkdir(parents=True, exist_ok=True)
        values = {}
        for spout in ("1", "2"):
            selected, status = select_outer_oof(bank, history, stamp(month), spout)
            selected_path = folder / f"spout-{spout}.selected.csv"
            if selected_path.exists():
                expected_csv = selected.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")
                if selected_path.read_bytes() != expected_csv:
                    raise ContractError("preserved lambda selected rows differ during engineering resume")
            else:
                _write_frame(selected_path, selected)
            saved = budget.derive(month, spout, selected)
            values[spout] = {**status, **saved, "selected_path": str(selected_path), "selected_sha256": file_sha256(selected_path)}
        summary = {"kind": "V22_OUTER_LAMBDAS_v1", "status": "PASS", "outer_month": month, "spouts": values}
        summary_path = folder / "summary.json"
        if summary_path.exists():
            if read_json(summary_path) != summary:
                raise ContractError("preserved outer lambda summary differs during engineering resume")
        else:
            atomic_write_json(summary_path, summary)
        outer[str(month)] = {"path": str(folder / "summary.json"), "sha256": file_sha256(folder / "summary.json"), "spouts": values}
    counts = budget.counts()
    if counts["parameter_slots_completed"] != 12 or counts["fit_attempts"] > 12:
        raise ContractError("V22 lambda slot budget or completeness differs")
    result = {"status": "PASS", "outer": outer, "counts": counts}
    atomic_write_json(root / "lambda_parameters.json", result)
    return result


def _candidate_frame(path: Path) -> pd.DataFrame:
    value = pd.read_csv(path, dtype={"sample_id": str}, float_precision="round_trip")
    if list(value) != ["sample_id", "pred_tap_iron", "pred_tap_time_len"]:
        raise ContractError("serialized candidate schema differs")
    return value


def _build_outer_predictions(
    root: Path, old_manifest: dict, history_source: Path, outer_inputs: dict,
    models: dict, median_files: dict, lambdas: dict,
) -> tuple[dict, list[dict], dict]:
    qrf_root = REPO / registration()["trusted_sources"]["qrf_development_run"]
    recency_root = REPO / "local/runs/optimization-v0.12-opt25-26-r1"
    recency_registry = recency_root / "predictions_complete.json"
    qrf_registry = qrf_root / "predictions_complete.json"
    history = pd.read_csv(history_source, dtype={"sample_id": str, "spout_no": str}, float_precision="round_trip")
    outputs: dict[str, dict] = {}
    composition_tasks: list[dict] = []
    v21_diagnostics: dict[str, dict] = {}
    for month in range(6, 12):
        metadata = pd.read_csv(outer_inputs[str(month)]["metadata"], dtype={"sample_id": str, "spout_no": str})
        qrf_path = root / "qrf" / f"outer-{month}.npz"
        with np.load(qrf_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        if arrays["ids"].tolist() != metadata.sample_id.tolist():
            raise ContractError(f"outer cutoff {month} QRF/metadata identity differs")
        model = read_json(_model_manifest_path(models, month))
        medians = read_json(median_files[str(month)])
        directions = build_prediction_directions(
            metadata, arrays["Q"], arrays["effective_neighbors"], int(arrays["N"][0]), medians["certificates"],
            model_cutoff=pd.Timestamp(model["cutoff_ns"], unit="ns", tz="UTC"),
            training_identity_sha256=model["training_identity_sha256"], model_bundle_sha256=model["bundle_sha256"],
        )
        direction_path = root / "directions" / f"outer-{month}.csv"
        _write_frame(direction_path, directions)
        v6i_path = recency_root / "predictions" / f"{month}_V6I_RECENCY60_IRON.csv"
        qrf_old_path = qrf_root / "predictions" / f"{month}_V8_QRF_TIME.csv"
        _trusted_entry(recency_registry, v6i_path, file_sha256(v6i_path))
        _trusted_entry(qrf_registry, qrf_old_path, file_sha256(qrf_old_path))
        v6i = frame(v6i_path).set_index("sample_id").loc[metadata.sample_id].reset_index()
        old_qrf = frame(qrf_old_path).set_index("sample_id").loc[metadata.sample_id].reset_index()
        if not np.array_equal(old_qrf.pred_tap_time_len.to_numpy(), arrays["Q"]):
            raise ContractError(f"outer cutoff {month} fresh QRF median differs from archived V8")
        v10 = pd.DataFrame({
            "sample_id": metadata.sample_id,
            "pred_tap_iron": v6i.pred_tap_iron.to_numpy(),
            "pred_tap_time_len": arrays["Q"],
        })
        folder = root / "predictions" / str(month)
        v10_full = folder / "V10.full_precision.csv"
        _write_frame(v10_full, v10)
        v10_serialized = folder / "V10.csv"
        v10_serialized.write_bytes(serialize_six_decimals(v10))
        parameters_path = Path(lambdas["outer"][str(month)]["path"])
        parameters = read_json(parameters_path)
        coefficients = {s: row["lambda"] for s, row in parameters["spouts"].items()}
        v22 = apply_shrink(v10, directions, coefficients)
        v22_full = folder / f"{CANDIDATE}.full_precision.csv"
        _write_frame(v22_full, v22)
        v22_serialized = folder / f"{CANDIDATE}.csv"
        v22_serialized.write_bytes(serialize_six_decimals(v22))
        v10_six = _candidate_frame(v10_serialized)
        v21, diagnostic = replay_v21(
            v10_six, metadata, history, arrays["effective_neighbors"], cutoff=stamp(month),
        )
        v21_serialized = folder / f"{REPLAY_CONTROL}.csv"
        v21_serialized.write_bytes(serialize_six_decimals(v21))
        ref = pd.to_datetime(metadata.reference_time, utc=True).dt.tz_convert("Asia/Shanghai")
        horizon = (ref.dt.year - 2024) * 12 + ref.dt.month - month + 1
        recent = np.asarray(diagnostic["recent_window_eligible_counts"])
        changed = (metadata.spout_no.astype(str).to_numpy() == "1") & (arrays["effective_neighbors"] < 500)
        groups = []
        detail = pd.DataFrame({"horizon": horizon, "spout_no": metadata.spout_no.astype(str), "changed": changed, "recent": recent})
        for (h, spout), part in detail.groupby(["horizon", "spout_no"], sort=True):
            groups.append({
                "horizon": int(h), "spout_no": str(spout), "rows": len(part),
                "changed_rows": int(part.changed.sum()), "changed_fraction": float(part.changed.mean()),
                "recent_window_count_min": int(part.recent.min()), "recent_window_count_median": float(part.recent.median()),
                "recent_window_count_max": int(part.recent.max()), "fallback_rows": int((part.recent == 0).sum()),
                "fallback_fraction": float((part.recent == 0).mean()),
            })
        v21_diagnostics[str(month)] = {**diagnostic, "by_horizon_spout": groups}
        outputs[str(month)] = {
            REFERENCE: str(v10_serialized), REPLAY_CONTROL: str(v21_serialized), CANDIDATE: str(v22_serialized),
            "v10_full": str(v10_full), "v22_full": str(v22_full), "directions": str(direction_path),
        }
        composition_tasks.append({
            "id": f"outer-{month}", "v10": str(v10_full), "bank": str(direction_path),
            "parameters": str(parameters_path), "prediction": str(v22_full),
        })
    atomic_write_json(root / "v21_replay_diagnostics.json", {
        "status": "PASS", "original_zip_byte_comparison": "EXPLICIT_USER_WAIVER",
        "rule_unchanged": True, "origins": v21_diagnostics,
    })
    return outputs, composition_tasks, v21_diagnostics


def _freeze_predictions_and_cold_plan(
    root: Path, history_source: Path, qrf_tasks: list[dict], median_files: dict,
    lambdas: dict, composition_tasks: list[dict], outputs: dict,
) -> None:
    median_tasks = [{"parameters": path, "history": str(history_source)} for path in median_files.values()]
    lambda_tasks = []
    for month, value in lambdas["outer"].items():
        summary = read_json(value["path"])
        lambda_tasks.append({
            "parameters": value["path"],
            "selected": {spout: row["selected_path"] for spout, row in summary["spouts"].items()},
        })
    atomic_write_json(root / "cold_plan.json", {
        "kind": "V22_COLD_PLAN_v1", "qrf_tasks": qrf_tasks,
        "median_tasks": median_tasks, "lambda_tasks": lambda_tasks,
        "composition_tasks": composition_tasks,
    })
    include = [
        root / name for name in (
            "manifest.json", "p0_original_replay.json", "history_input_manifest.json", "input_manifest.json",
            "warmup_registration.json", "warmup_fit_intents.json", "models_complete.json", "h2_bank.csv",
            "h2_banks.json", "median_parameters.json", "lambda_parameters.json", "v21_replay_diagnostics.json",
            "cold_plan.json",
        )
    ]
    repair_manifest = root / "engineering_repair/manifest.json"
    if repair_manifest.exists():
        include.append(repair_manifest)
    for folder in ("features", "metadata", "models", "model_registry", "qrf", "medians", "banks", "lambda_slots", "lambdas", "directions", "predictions"):
        include.extend(path for path in (root / folder).rglob("*") if path.is_file())
    identities = file_identities({str(path): path for path in include})
    atomic_write_json(root / "predictions_complete.json", {
        "status": "PASS", "before_outer_scoring_label_access": True,
        "origins": outputs, "identities": identities,
    })


def _score_and_complete(root: Path, old_manifest: dict, outputs: dict, v21_diagnostics: dict) -> dict:
    _ledger(root, "SIX_DECIMAL_OUTER_SCORING_AFTER_PREDICTIONS_FROZEN", protected_labels_read=True)
    reg = old_manifest["registration"]
    candidates = (REFERENCE, REPLAY_CONTROL, CANDIDATE)
    def provider(unit, cutoff, samples, parts):
        result = {}
        for candidate in candidates:
            value = _candidate_frame(Path(outputs[str(cutoff.month)][candidate])).set_index("sample_id")
            result[candidate] = value.loc[samples.sample_id.astype(str)].reset_index()
        return result
    metrics, summary, errors, _ = score_grid(root, reg, provider)
    iron_exact = True
    for unit in metrics:
        if not (unit.startswith("O") or unit.startswith("DEV")):
            continue
        ref = errors.loc[(errors.unit == unit) & (errors.candidate == REFERENCE)].sort_values("sample_id")
        got = errors.loc[(errors.unit == unit) & (errors.candidate == CANDIDATE)].sort_values("sample_id")
        iron_exact &= np.array_equal(ref.pred_tap_iron.to_numpy(), got.pred_tap_iron.to_numpy())
    decision = acceptance(metrics, summary, engineering=True, iron_exact=iron_exact)
    atomic_write_json(root / "acceptance.json", decision)
    bootstrap = {
        CANDIDATE: week_intervals(errors, CANDIDATE, REFERENCE, repetitions=1000, seed=2026),
        REPLAY_CONTROL: week_intervals(errors, REPLAY_CONTROL, REFERENCE, repetitions=1000, seed=2026),
    }
    atomic_write_json(root / "bootstrap.json", bootstrap)
    diagnostics = []
    for (candidate, unit, spout), part in errors.loc[errors.candidate.isin(candidates)].groupby(["candidate", "unit", "spout_no"], sort=True):
        for target in ("tap_iron", "tap_time_len"):
            residual = part[f"pred_{target}"] - part[target]
            denominator = float(part[target].sum())
            diagnostics.append({
                "candidate": candidate, "unit": unit, "spout_no": str(spout), "target": target,
                "n": len(part), "absolute_error_sum": float(residual.abs().sum()),
                "actual_sum": denominator, "wmape": float(residual.abs().sum() / denominator),
                "signed_error_sum": float(residual.sum()), "signed_bias": float(residual.mean()),
            })
    directions = pd.concat([
        pd.read_csv(root / "directions" / f"outer-{month}.csv", dtype={"sample_id": str, "spout_no": str}, float_precision="round_trip")
        for month in range(6, 12)
    ], ignore_index=True)
    support = []
    for (cutoff, spout), part in directions.groupby(["model_cutoff", "spout_no"], sort=True):
        support.append({
            "model_cutoff": str(cutoff), "spout_no": str(spout), "n": len(part),
            "u_quantiles": dict(zip(("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"), np.quantile(part.u, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist())),
        })
    changes = {}
    for candidate in (CANDIDATE, REPLAY_CONTROL):
        got = errors.loc[errors.candidate == candidate].set_index(["unit", "sample_id"]).sort_index()
        ref = errors.loc[errors.candidate == REFERENCE].set_index(["unit", "sample_id"]).sort_index()
        delta = got.pred_tap_time_len - ref.pred_tap_time_len
        changes[candidate] = {
            "changed_fraction": float((delta != 0).mean()),
            "quantiles": dict(zip(("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"), np.quantile(delta, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist())),
        }
    atomic_write_json(root / "diagnostics.json", {
        "status": "PASS", "by_unit_spout_target": diagnostics, "support": support,
        "prediction_changes": changes, "v21_replay": v21_diagnostics,
    })
    lambda_parameters = read_json(root / "lambda_parameters.json")
    fit_counts = {
        "status": "PASS", "forest_fit_attempts": 2, "forest_fit_completed": 2,
        "preprocessor_fit_attempts": 2, "preprocessor_fit_completed": 2, "internal_trees": 512,
        "lambda_parameter_slots": lambda_parameters["counts"]["parameter_slots_completed"],
        "lambda_fit_attempts": lambda_parameters["counts"]["fit_attempts"],
        "lambda_zero_fallbacks": lambda_parameters["counts"]["zero_fallbacks"],
        "median_statistics": 16, "iron_model_fits": 0, "platform_candidates": 0,
    }
    atomic_write_json(root / "fit_counts.json", fit_counts)
    completion = {
        "status": decision["status"], "G0": decision["G0"], "G1": decision["G1"],
        "P0": "PASS_WITH_EXPLICIT_V21_ORIGINAL_ZIP_WAIVER", "six_decimal_scoring": True,
        "iron_exact": iron_exact, "fit_counts": fit_counts, "acceptance": decision,
        "protected_test_targets_read": False, "test_scores_used_for_parameter_fit": False,
        "platform_uploads": 0, "desktop_writes": 0, "public_pushes": 0,
    }
    atomic_write_json(root / "completion.json", completion)
    return completion


def _register_engineering_resume(root: Path) -> dict:
    failure_path = root / "failure.json"
    failure = read_json(failure_path)
    if failure.get("error") != "ContractError('lambda parameter-slot budget exceeded')":
        raise ContractError("engineering resume is limited to the preserved lambda glob-count failure")
    warmup = read_json(root / "warmup_fit_intents.json")
    if warmup.get("forest_fit_attempts") != 2 or warmup.get("preprocessor_fit_attempts") != 2:
        raise ContractError("engineering resume requires exactly the already-consumed 2+2 warmup budget")
    if (root / "predictions_complete.json").exists() or (root / "completion.json").exists():
        raise ContractError("completed predictions/run cannot enter engineering resume")
    folder = root / "engineering_repair"
    folder.mkdir(parents=True, exist_ok=False)
    manifest = read_json(root / "manifest.json")
    support_source = REPO / "src/bf_tap/optimization/qrf_support_shrink.py"
    changed = [
        support_source,
        REPO / "scripts/optimization_v22_run.py",
        REPO / "scripts/optimization_v22_cold_check.py",
        REPO / "tests/test_qrf_support_shrink.py",
    ]
    preserved = [path for path in root.rglob("*") if path.is_file() and not path.is_relative_to(folder)]
    value = {
        "kind": "V22_ENGINEERING_RESUME_v1",
        "original_manifest_sha256": file_sha256(root / "manifest.json"),
        "original_failure": {"path": str(failure_path), "sha256": file_sha256(failure_path)},
        "original_qrf_support_source_sha256": manifest["sources"][str(support_source)]["sha256"],
        "repaired_sources": file_identities({str(path): path for path in changed}),
        "preserved_artifacts": file_identities({str(path): path for path in preserved}),
        "cause": "LambdaBudget glob matched both completed JSON and .intent.json files, double-counting slots",
        "repair": "exclude .intent.json from completed result enumeration",
        "algorithm_or_parameter_change": False,
        "additional_forest_or_preprocessor_fits_authorized": 0,
        "failed_evidence_preserved": True,
    }
    atomic_write_json(folder / "manifest.json", value)
    _ledger(root, "REGISTER_ZERO_MODEL_REFIT_ENGINEERING_RESUME")
    return value


def resume_development(run: Path) -> dict:
    root = run.resolve()
    _register_engineering_resume(root)
    try:
        reg = registration()
        old_root = REPO / reg["trusted_sources"]["qrf_development_run"]
        old_manifest = restore_manifest(old_root)
        verify_qrf_manifest(old_manifest)
        history_source = Path(read_json(root / "history_input_manifest.json")["history"]["path"])
        inputs = read_json(root / "input_manifest.json")
        outer_inputs = inputs["outer_inputs"]
        models = read_json(root / "models_complete.json")["models"]
        bank = pd.read_csv(root / "h2_bank.csv", dtype={"sample_id": str, "spout_no": str}, float_precision="round_trip")
        verify_h2_bank(bank, reg["expected_training_rows"])
        median_files = read_json(root / "median_parameters.json")["cutoffs"]
        qrf_tasks = []
        for month in range(4, 11):
            model_manifest = _model_manifest_path(models, month)
            handoff = root / "features" / str(month) / "h2.npz"
            qrf_tasks.append({
                "id": f"h2-{month}", "model_manifest": str(model_manifest),
                "model_manifest_sha256": file_sha256(model_manifest), "input": str(handoff),
                "input_sha256": file_sha256(handoff), "expected_output": str(root / "qrf" / f"h2-{month}.npz"),
            })
        for month in range(6, 12):
            model_manifest = _model_manifest_path(models, month)
            handoff = Path(outer_inputs[str(month)]["handoff"])
            qrf_tasks.append({
                "id": f"outer-{month}", "model_manifest": str(model_manifest),
                "model_manifest_sha256": file_sha256(model_manifest), "input": str(handoff),
                "input_sha256": file_sha256(handoff), "expected_output": str(root / "qrf" / f"outer-{month}.npz"),
            })
        lambdas = _derive_all_lambdas(root, history_source, bank)
        outputs, composition_tasks, v21_diagnostics = _build_outer_predictions(
            root, old_manifest, history_source, outer_inputs, models, median_files, lambdas,
        )
        _freeze_predictions_and_cold_plan(root, history_source, qrf_tasks, median_files, lambdas, composition_tasks, outputs)
        _ledger(root, "INDEPENDENT_COLD_CERTIFICATE_VERIFICATION_AFTER_ENGINEERING_RESUME", protected_labels_read=True)
        subprocess.run(
            [sys.executable, str(REPO / "scripts/optimization_v22_cold_check.py"), "--run", str(root)],
            check=True,
            cwd=REPO,
        )
        return _score_and_complete(root, old_manifest, outputs, v21_diagnostics)
    except Exception as exc:
        atomic_write_json(root / "engineering_repair/failure.json", {
            "status": "FAILED_PRESERVED", "error_type": type(exc).__name__, "error": repr(exc),
            "additional_forest_fits": 0, "additional_preprocessor_fits": 0,
        })
        raise


def resume_cold_score(run: Path) -> dict:
    root = run.resolve()
    prior_repair = root / "engineering_repair/manifest.json"
    prior_failure = root / "engineering_repair/failure.json"
    if not prior_repair.is_file() or not prior_failure.is_file():
        raise ContractError("cold/score resume requires the preserved first engineering repair failure")
    if not (root / "predictions_complete.json").is_file() or (root / "completion.json").exists():
        raise ContractError("cold/score resume requires frozen predictions and no completion")
    resume_manifest = root / "engineering_repair/cold_resume_manifest.json"
    if resume_manifest.exists():
        raise FileExistsError(resume_manifest)
    changed = [
        REPO / "src/bf_tap/optimization/qrf_support_shrink.py",
        REPO / "scripts/optimization_v22_run.py",
        REPO / "scripts/optimization_v22_cold_check.py",
        REPO / "tests/test_qrf_support_shrink.py",
    ]
    preserved = [path for path in root.rglob("*") if path.is_file()]
    value = {
        "kind": "V22_COLD_SCORE_RESUME_v1",
        "prior_repair_manifest_sha256": file_sha256(prior_repair),
        "prior_repair_failure": file_identities({str(prior_failure): prior_failure})[str(prior_failure)],
        "repaired_sources": file_identities({str(path): path for path in changed}),
        "preserved_artifacts": file_identities({str(path): path for path in preserved}),
        "cause": "repair certificate original_failure entry omitted byte count required by generic verifier",
        "repair": "verify legacy failure by exact path/hash and retain generic identities for all later records",
        "algorithm_or_prediction_change": False,
        "additional_fit_attempts_authorized": 0,
        "predictions_recomputed": False,
    }
    atomic_write_json(resume_manifest, value)
    _ledger(root, "REGISTER_ZERO_FIT_COLD_SCORE_CERTIFICATE_RESUME")
    try:
        subprocess.run(
            [sys.executable, str(REPO / "scripts/optimization_v22_cold_check.py"), "--run", str(root)],
            check=True,
            cwd=REPO,
        )
        reg = registration()
        old_root = REPO / reg["trusted_sources"]["qrf_development_run"]
        old_manifest = restore_manifest(old_root)
        verify_qrf_manifest(old_manifest)
        outputs = read_json(root / "predictions_complete.json")["origins"]
        v21 = read_json(root / "v21_replay_diagnostics.json")["origins"]
        return _score_and_complete(root, old_manifest, outputs, v21)
    except Exception as exc:
        atomic_write_json(root / "engineering_repair/cold_resume_failure.json", {
            "status": "FAILED_PRESERVED", "error_type": type(exc).__name__, "error": repr(exc),
            "additional_fit_attempts": 0, "predictions_recomputed": False,
        })
        raise


def resume_certificates_score(run: Path) -> dict:
    root = run.resolve()
    prior = root / "engineering_repair/cold_resume_manifest.json"
    prior_failure = root / "engineering_repair/cold_resume_failure.json"
    if not prior.is_file() or not prior_failure.is_file() or not (root / "cold_worker").is_dir():
        raise ContractError("certificate resume requires preserved cold worker outputs and cold-resume failure")
    if (root / "cold_validation.json").exists() or (root / "completion.json").exists():
        raise ContractError("completed cold/run cannot enter certificate resume")
    path = root / "engineering_repair/certificate_resume_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    changed = [
        REPO / "src/bf_tap/optimization/qrf_support_shrink.py",
        REPO / "scripts/optimization_v22_run.py",
        REPO / "scripts/optimization_v22_cold_check.py",
        REPO / "tests/test_qrf_support_shrink.py",
    ]
    preserved = [artifact for artifact in root.rglob("*") if artifact.is_file()]
    value = {
        "kind": "V22_CERTIFICATE_SCORE_RESUME_v1",
        "prior_cold_resume_manifest_sha256": file_sha256(prior),
        "prior_cold_resume_failure": file_identities({str(prior_failure): prior_failure})[str(prior_failure)],
        "repaired_sources": file_identities({str(source): source for source in changed}),
        "preserved_artifacts": file_identities({str(artifact): artifact for artifact in preserved}),
        "cause": "cold certificate verification called a fitting primitive that the zero-fit guard correctly rejected",
        "repair": "verify saved weighted-median optimum directly and reuse exact completed cold-worker outputs",
        "algorithm_or_prediction_change": False,
        "additional_fit_attempts_authorized": 0,
        "predictions_recomputed": False,
        "cold_worker_predictions_recomputed": False,
    }
    atomic_write_json(path, value)
    _ledger(root, "REGISTER_ZERO_FIT_CERTIFICATE_AND_SCORE_RESUME")
    try:
        subprocess.run(
            [
                sys.executable, str(REPO / "scripts/optimization_v22_cold_check.py"),
                "--run", str(root), "--resume-certificates",
            ],
            check=True,
            cwd=REPO,
        )
        reg = registration()
        old_root = REPO / reg["trusted_sources"]["qrf_development_run"]
        old_manifest = restore_manifest(old_root)
        verify_qrf_manifest(old_manifest)
        outputs = read_json(root / "predictions_complete.json")["origins"]
        v21 = read_json(root / "v21_replay_diagnostics.json")["origins"]
        return _score_and_complete(root, old_manifest, outputs, v21)
    except Exception as exc:
        atomic_write_json(root / "engineering_repair/certificate_resume_failure.json", {
            "status": "FAILED_PRESERVED", "error_type": type(exc).__name__, "error": repr(exc),
            "additional_fit_attempts": 0, "predictions_recomputed": False,
            "cold_worker_predictions_recomputed": False,
        })
        raise


def resume_label_certificates_score(run: Path) -> dict:
    root = run.resolve()
    prior = root / "engineering_repair/certificate_resume_manifest.json"
    prior_failure = root / "engineering_repair/certificate_resume_failure.json"
    if not prior.is_file() or not prior_failure.is_file() or not (root / "cold_worker").is_dir():
        raise ContractError("label-certificate resume requires the preserved certificate failure")
    if (root / "cold_validation.json").exists() or (root / "completion.json").exists():
        raise ContractError("completed cold/run cannot enter label-certificate resume")
    path = root / "engineering_repair/label_certificate_resume_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    changed = [
        REPO / "src/bf_tap/optimization/qrf_support_shrink.py",
        REPO / "scripts/optimization_v22_run.py",
        REPO / "scripts/optimization_v22_cold_check.py",
        REPO / "tests/test_qrf_support_shrink.py",
    ]
    preserved = [artifact for artifact in root.rglob("*") if artifact.is_file()]
    value = {
        "kind": "V22_LABEL_CERTIFICATE_SCORE_RESUME_v1",
        "prior_certificate_resume_manifest_sha256": file_sha256(prior),
        "prior_certificate_resume_failure": file_identities({str(prior_failure): prior_failure})[str(prior_failure)],
        "repaired_sources": file_identities({str(source): source for source in changed}),
        "preserved_artifacts": file_identities({str(artifact): artifact for artifact in preserved}),
        "cause": "CSV restored integer-valued labels as int64 while the fitted certificate normalized them as float64",
        "repair": "normalize restored tap_time_len to float before verifying the immutable label digest",
        "algorithm_or_prediction_change": False,
        "additional_fit_attempts_authorized": 0,
        "predictions_recomputed": False,
        "cold_worker_predictions_recomputed": False,
        "lambda_values_changed": False,
    }
    atomic_write_json(path, value)
    _ledger(root, "REGISTER_ZERO_FIT_LABEL_CERTIFICATE_AND_SCORE_RESUME")
    try:
        subprocess.run(
            [
                sys.executable, str(REPO / "scripts/optimization_v22_cold_check.py"),
                "--run", str(root), "--resume-certificates",
            ],
            check=True,
            cwd=REPO,
        )
        reg = registration()
        old_root = REPO / reg["trusted_sources"]["qrf_development_run"]
        old_manifest = restore_manifest(old_root)
        verify_qrf_manifest(old_manifest)
        outputs = read_json(root / "predictions_complete.json")["origins"]
        v21 = read_json(root / "v21_replay_diagnostics.json")["origins"]
        return _score_and_complete(root, old_manifest, outputs, v21)
    except Exception as exc:
        atomic_write_json(root / "engineering_repair/label_certificate_resume_failure.json", {
            "status": "FAILED_PRESERVED", "error_type": type(exc).__name__, "error": repr(exc),
            "additional_fit_attempts": 0, "predictions_recomputed": False,
            "cold_worker_predictions_recomputed": False, "lambda_values_changed": False,
        })
        raise


def development(run: Path) -> dict:
    root = run.resolve()
    if read_json(root / "p0_original_replay.json").get("status") != "PASS":
        raise ContractError("V22 development requires passed P0 or explicit recorded waiver")
    for name in ("input_manifest.json", "models_complete.json", "predictions_complete.json", "completion.json"):
        if (root / name).exists():
            raise FileExistsError(root / name)
    reg = registration()
    try:
        old_manifest, history_source, inputs, outer_inputs = _prepare_development_inputs(root, reg)
        models = _warmup_models(root, reg, inputs)
        bank, median_files, qrf_tasks = _infer_and_build_banks(root, reg, history_source, inputs, outer_inputs, models)
        lambdas = _derive_all_lambdas(root, history_source, bank)
        outputs, composition_tasks, v21_diagnostics = _build_outer_predictions(
            root, old_manifest, history_source, outer_inputs, models, median_files, lambdas,
        )
        _freeze_predictions_and_cold_plan(root, history_source, qrf_tasks, median_files, lambdas, composition_tasks, outputs)
        _ledger(root, "INDEPENDENT_COLD_CERTIFICATE_VERIFICATION", protected_labels_read=True)
        subprocess.run(
            [sys.executable, str(REPO / "scripts/optimization_v22_cold_check.py"), "--run", str(root)],
            check=True,
            cwd=REPO,
        )
        return _score_and_complete(root, old_manifest, outputs, v21_diagnostics)
    except Exception as exc:
        if not (root / "failure.json").exists():
            atomic_write_json(root / "failure.json", {
                "status": "FAILED_PRESERVED", "error_type": type(exc).__name__, "error": repr(exc),
                "fit_counts_so_far": {
                    "forest_intents": len(list((root / "models").glob("*/forest_intent.json"))),
                    "preprocessor_intents": len(list((root / "models").glob("*/preprocessor_intent.json"))),
                    "lambda_intents": len(list((root / "lambda_slots").glob("*.intent.json"))),
                },
            })
        raise


def make_medians(run: Path, history_path: Path, cutoff: str, output: Path) -> dict:
    _require_run_output(run, output)
    if output.exists():
        raise FileExistsError(output)
    _authorize_history(run, history_path, "DERIVE_FIXED_CUTOFF_MEDIANS")
    history = pd.read_csv(history_path, dtype={"sample_id": str})
    certificates = fixed_cutoff_medians(history, cutoff)
    value = {
        "kind": "V22_CUTOFF_MEDIANS_v1",
        "history": {"path": str(history_path.resolve()), "sha256": file_sha256(history_path)},
        "cutoff": pd.Timestamp(cutoff).isoformat(),
        "certificates": certificates,
    }
    atomic_write_json(output, value)
    return value


def make_bank(run: Path, metadata_path: Path, qrf_path: Path, medians_path: Path, model_manifest_path: Path, output: Path) -> dict:
    _require_run_output(run, output)
    if output.exists():
        raise FileExistsError(output)
    metadata = pd.read_csv(metadata_path, dtype={"sample_id": str})
    with np.load(qrf_path, allow_pickle=False) as data:
        if set(data.files) != {"ids", "Q", "effective_neighbors", "N"}:
            raise ContractError("unexpected V22 QRF handoff")
        arrays = {name: data[name] for name in data.files}
    if arrays["ids"].tolist() != metadata.sample_id.astype(str).tolist() or len(set(arrays["N"].tolist())) != 1:
        raise ContractError("QRF handoff and metadata order/N differ")
    medians = read_json(medians_path)
    model = _authorize_model(run, model_manifest_path)
    bank = build_direction_bank(
        metadata,
        arrays["Q"], arrays["effective_neighbors"], int(arrays["N"][0]),
        medians["certificates"], model_cutoff=pd.Timestamp(model["cutoff_ns"], unit="ns", tz="UTC"),
        training_identity_sha256=model["training_identity_sha256"],
        model_bundle_sha256=model["bundle_sha256"],
    )
    verify_h2_bank(bank, registration()["expected_training_rows"])
    output.parent.mkdir(parents=True, exist_ok=True)
    bank.to_csv(output, index=False, mode="x", float_format="%.17g", lineterminator="\n")
    status = {
        "status": "PASS", "rows": len(bank), "target_columns_present": False,
        "bank_sha256": file_sha256(output),
        "inputs": file_identities({str(p): p for p in (metadata_path, qrf_path, medians_path, model_manifest_path)}),
    }
    atomic_write_json(Path(str(output) + ".json"), status)
    return status


def derive_lambdas(run: Path, bank_path: Path, history_path: Path, outer_month: int, output: Path) -> dict:
    _require_run_output(run, output)
    if output.exists():
        raise FileExistsError(output)
    _authorize_history(run, history_path, f"DERIVE_OUTER_{outer_month}_SPOUT_LAMBDAS")
    bank = pd.read_csv(bank_path, dtype={"sample_id": str, "spout_no": str}, float_precision="round_trip")
    history = pd.read_csv(history_path, dtype={"sample_id": str, "spout_no": str}, float_precision="round_trip")
    budget = LambdaBudget(output)
    results = {}
    cutoff = pd.Timestamp(year=2024, month=outer_month, day=1, tz="Asia/Shanghai")
    for spout in ("1", "2"):
        selected, status = select_outer_oof(bank, history, cutoff, spout)
        saved = budget.derive(outer_month, spout, selected)
        selected.to_csv(output / f"outer-{outer_month}-spout-{spout}.selected.csv", index=False, mode="x")
        results[spout] = {**status, **saved}
    summary = {"status": "PASS", "outer_month": outer_month, "spouts": results, "counts": budget.counts()}
    atomic_write_json(output / "summary.json", summary)
    return summary


def compose(run: Path, v10_path: Path, bank_path: Path, lambdas_path: Path, output: Path) -> dict:
    _require_run_output(run, output)
    if output.exists():
        raise FileExistsError(output)
    v10 = pd.read_csv(v10_path, dtype={"sample_id": str}, float_precision="round_trip")
    bank = pd.read_csv(bank_path, dtype={"sample_id": str, "spout_no": str}, float_precision="round_trip")
    params = read_json(lambdas_path)
    lambdas = {spout: row["lambda"] for spout, row in params["spouts"].items()}
    prediction = apply_shrink(v10, bank, lambdas)
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output, index=False, mode="x", float_format="%.17g", lineterminator="\n")
    serialized = output.with_name(output.stem + ".six_decimal.csv")
    serialized.write_bytes(serialize_six_decimals(prediction))
    if not np.array_equal(prediction.pred_tap_iron.to_numpy(), v10.pred_tap_iron.to_numpy()):
        raise ContractError("V22 iron differs from certified V10")
    status = {
        "status": "PASS", "rows": len(prediction), "iron_exact": True,
        "full_precision_sha256": file_sha256(output),
        "six_decimal_sha256": file_sha256(serialized),
        "changed_rows": int(np.count_nonzero(prediction.pred_tap_time_len.to_numpy() != v10.pred_tap_time_len.to_numpy())),
        "platform_package_created": False,
    }
    atomic_write_json(Path(str(output) + ".json"), status)
    return status


def stage_infer(
    run: Path,
    stage: str,
    metadata_path: Path,
    model_manifest_path: Path,
    frozen_history_path: Path,
    feature_handoff_path: Path,
    v10_path: Path,
    parameters_path: Path,
    output: Path,
) -> dict:
    """Fresh label-free QRF/support inference plus frozen V22 composition."""
    root = output.resolve()
    _require_run_output(run, output)
    if root.exists():
        raise FileExistsError(root)
    _authorize_history(run, frozen_history_path, f"{stage}_FIXED_MEDIAN_CERTIFICATE_ACCESS")
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    try:
        metadata = pd.read_csv(metadata_path, dtype={"sample_id": str})
        forbidden = {"tap_iron", "tap_time_len", "target", "actual"} & set(metadata)
        required = {"sample_id", "spout_no", "reference_time"}
        if forbidden or required - set(metadata) or metadata.sample_id.duplicated().any():
            raise ContractError("stage metadata must be unique, prediction-only, and target-free")
        model = _authorize_model(run, model_manifest_path)
        input_info = read_json(str(feature_handoff_path) + ".json")
        if input_info.get("training") or input_info.get("ids") != metadata.sample_id.astype(str).tolist():
            raise ContractError("label-free feature handoff IDs differ from stage metadata")
        qrf_path = root / "qrf.npz"
        worker = REPO / "workers/qrf_v015"
        subprocess.run(
            [
                str(worker / ".venv/bin/python"), str(worker / "inference_v022.py"),
                "--model-manifest", str(model_manifest_path.resolve()),
                "--model-manifest-sha256", file_sha256(model_manifest_path),
                "--input", str(feature_handoff_path.resolve()),
                "--input-sha256", file_sha256(feature_handoff_path),
                "--output", str(qrf_path),
            ],
            check=True,
            cwd=REPO,
        )
        history = pd.read_csv(frozen_history_path, dtype={"sample_id": str, "spout_no": str})
        cutoff = pd.Timestamp(model["cutoff_ns"], unit="ns", tz="UTC").tz_convert("Asia/Shanghai")
        medians = fixed_cutoff_medians(history, cutoff)
        median_manifest = {
            "kind": "V22_CUTOFF_MEDIANS_v1",
            "stage": stage,
            "history": {"path": str(frozen_history_path.resolve()), "sha256": file_sha256(frozen_history_path)},
            "cutoff": cutoff.isoformat(),
            "certificates": medians,
        }
        atomic_write_json(root / "medians.json", median_manifest)
        with np.load(qrf_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        if set(arrays) != {"ids", "Q", "effective_neighbors", "N"} or arrays["ids"].tolist() != metadata.sample_id.tolist():
            raise ContractError("fresh stage QRF output identity differs")
        directions = build_prediction_directions(
            metadata, arrays["Q"], arrays["effective_neighbors"], int(arrays["N"][0]), medians,
            model_cutoff=cutoff,
            training_identity_sha256=model["training_identity_sha256"],
            model_bundle_sha256=model["bundle_sha256"],
        )
        directions.to_csv(root / "directions.csv", index=False, mode="x", float_format="%.17g", lineterminator="\n")
        v10 = pd.read_csv(v10_path, dtype={"sample_id": str}, float_precision="round_trip")
        parameters = read_json(parameters_path)
        if set(parameters.get("spouts", {})) - {"1", "2"}:
            raise ContractError("only separately certified spout-1/spout-2 lambdas are allowed")
        lambdas = {spout: row["lambda"] for spout, row in parameters["spouts"].items()}
        prediction = apply_shrink(v10, directions, lambdas)
        prediction.to_csv(root / "full_precision.csv", index=False, mode="x", float_format="%.17g", lineterminator="\n")
        (root / "result.csv").write_bytes(serialize_six_decimals(prediction))
        report = read_json(str(qrf_path) + ".json")
        if any(report["zero_fit"].values()):
            raise ContractError("stage inference attempted a model/preprocessor fit")
        completion = {
            "status": "PASS_ENGINEERING_ONLY",
            "stage": stage,
            "rows": len(prediction),
            "fresh_qrf_support_inference": True,
            "model_cutoff": cutoff.isoformat(),
            "model_manifest_sha256": file_sha256(model_manifest_path),
            "metadata_sha256": file_sha256(metadata_path),
            "frozen_history_sha256": file_sha256(frozen_history_path),
            "parameters_sha256": file_sha256(parameters_path),
            "iron_exact": bool(np.array_equal(prediction.pred_tap_iron.to_numpy(), v10.pred_tap_iron.to_numpy())),
            "target_columns_read": False,
            "zero_fit": report["zero_fit"],
            "result_sha256": file_sha256(root / "result.csv"),
            "platform_package_created": False,
            "platform_uploads": 0,
        }
        if not completion["iron_exact"]:
            raise ContractError("stage V22 iron differs from certified V10")
        atomic_write_json(root / "completion.json", completion)
        return completion
    except Exception as exc:
        if not (root / "failure.json").exists():
            atomic_write_json(root / "failure.json", {
                "status": "FAILED_PRESERVED", "error_type": type(exc).__name__, "error": str(exc),
                "platform_uploads": 0,
            })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("p0")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--allow-missing-v21-original", action="store_true")
    p = sub.add_parser("development")
    p.add_argument("--run", required=True, type=Path)
    p = sub.add_parser("resume-development")
    p.add_argument("--run", required=True, type=Path)
    p = sub.add_parser("resume-cold-score")
    p.add_argument("--run", required=True, type=Path)
    p = sub.add_parser("resume-certificates-score")
    p.add_argument("--run", required=True, type=Path)
    p = sub.add_parser("resume-label-certificates-score")
    p.add_argument("--run", required=True, type=Path)
    p = sub.add_parser("medians")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--history", required=True, type=Path)
    p.add_argument("--cutoff", required=True)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("bank")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--metadata", required=True, type=Path)
    p.add_argument("--qrf", required=True, type=Path)
    p.add_argument("--medians", required=True, type=Path)
    p.add_argument("--model-manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("lambdas")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--bank", required=True, type=Path)
    p.add_argument("--history", required=True, type=Path)
    p.add_argument("--outer-month", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("compose")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--v10", required=True, type=Path)
    p.add_argument("--bank", required=True, type=Path)
    p.add_argument("--lambdas", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("stage-infer")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--stage", required=True, choices=("test_a", "test_b", "test_c", "official_round2"))
    p.add_argument("--metadata", required=True, type=Path)
    p.add_argument("--model-manifest", required=True, type=Path)
    p.add_argument("--frozen-history", required=True, type=Path)
    p.add_argument("--feature-handoff", required=True, type=Path)
    p.add_argument("--v10", required=True, type=Path)
    p.add_argument("--parameters", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("register-history")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--history", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "p0":
        result = p0(args.output, allow_missing_v21_original=args.allow_missing_v21_original)
    elif args.command == "development":
        result = development(args.run)
    elif args.command == "resume-development":
        result = resume_development(args.run)
    elif args.command == "resume-cold-score":
        result = resume_cold_score(args.run)
    elif args.command == "resume-certificates-score":
        result = resume_certificates_score(args.run)
    elif args.command == "resume-label-certificates-score":
        result = resume_label_certificates_score(args.run)
    elif args.command == "medians":
        result = make_medians(args.run, args.history, args.cutoff, args.output)
    elif args.command == "bank":
        result = make_bank(args.run, args.metadata, args.qrf, args.medians, args.model_manifest, args.output)
    elif args.command == "lambdas":
        result = derive_lambdas(args.run, args.bank, args.history, args.outer_month, args.output)
    elif args.command == "compose":
        result = compose(args.run, args.v10, args.bank, args.lambdas, args.output)
    elif args.command == "stage-infer":
        result = stage_infer(
            args.run, args.stage, args.metadata, args.model_manifest, args.frozen_history,
            args.feature_handoff, args.v10, args.parameters, args.output,
        )
    else:
        result = register_history(args.run, args.history)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
