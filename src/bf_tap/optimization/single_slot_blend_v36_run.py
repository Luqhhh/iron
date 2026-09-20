"""Lifecycle for the one zero-fit v0.36 complete-endpoint time blend."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json, file_identities, file_sha256, runtime_environment
from ..config import load_yaml
from ..exceptions import ContractError
from .fixed_blend_v28 import exposure_pooled_summary, macro_origin_summary, verify_isolated_deltas
from .qrf_partition_v26 import POOLED_AGGREGATION, PRIMARY_AGGREGATION, TARGETS
from .qrf_partition_v26_run import _score_rows
from .single_slot_blend_v36 import (
    CANDIDATE,
    PROTOCOL,
    V30_RESULT_SHA256,
    V30_ZIP_SHA256,
    V34_RESULT_SHA256,
    V34_ZIP_SHA256,
    BlendContractError,
    PredictionRow,
    build_package,
    compose_rows,
    parse_payload,
    payload_bytes,
)
from .structural_run import append_ledger


V34 = Path("local/runs/optimization-v0.34-oob-leaf-median-bagging-r2")
V30 = Path("local/runs/optimization-v0.30-oob-compose-and-time-growth-r6")
V34_CANDIDATE = "V34T_OOB_LEAF_MEDIAN_BAGGING_TIME"
V30_CANDIDATE = "V30A_OOB_BOTH_TARGETS"
V34_FINAL = V34 / "submissions" / V34_CANDIDATE
V30_FINAL = V30 / "submissions" / V30_CANDIDATE
ORIGINS = range(6, 12)
REFERENCES = (V34_CANDIDATE, V30_CANDIDATE, "V1", "V21_REPLAY")


def registration() -> dict:
    value = load_yaml("configs/optimization_v0_36/experiment.yaml")
    if value["branch"] != "optimization-v0.36-single-slot-time-blend" or value["candidate"] != CANDIDATE:
        raise ContractError("v0.36 branch/candidate registration differs")
    if value["protocol"] != PROTOCOL or value["algorithm"]["left_weight"] != 0.5 or value["algorithm"]["right_weight"] != 0.5:
        raise ContractError("v0.36 protocol/weight registration differs")
    expected = {
        "new_model_fits": 0,
        "new_tree_fits": 0,
        "new_preprocessor_fits": 0,
        "new_calibration_or_coefficient_fits": 0,
        "new_candidate_packages": 1,
        "candidate_platform_submissions_maximum": 1,
        "restore_uploads": 0,
        "automatic_uploads": 0,
        "desktop_writes": 0,
    }
    if value["budget"] != expected:
        raise ContractError("v0.36 registered budget differs")
    return value


def _source_files() -> list[Path]:
    return [
        Path("configs/optimization_v0_36/experiment.yaml"),
        Path("configs/optimization_v0_36/access_scope.yaml"),
        Path("src/bf_tap/optimization/single_slot_blend_v36.py"),
        Path("src/bf_tap/optimization/single_slot_blend_v36_run.py"),
        Path("scripts/build_v36.py"),
        Path("scripts/optimization_v36_single_slot.py"),
        Path("tests/test_optimization_v36_single_slot_blend.py"),
        Path("docs/optimization_v0_36/PLAN.md"),
    ]


def _endpoint_path(slot: int, role: str) -> Path:
    if role == "V34T":
        return V34_FINAL / "result.csv" if slot == 12 else V34 / "predictions" / f"{slot}_{V34_CANDIDATE}.csv"
    if role == "V30A":
        return V30_FINAL / "result.csv" if slot == 12 else V30 / "predictions" / f"{slot}_{V30_CANDIDATE}.csv"
    raise ContractError("unknown v0.36 endpoint role")


def _rows(path: Path, role: str) -> tuple[PredictionRow, ...]:
    if not path.is_file():
        raise ContractError(f"missing v0.36 historical endpoint: {path}")
    try:
        return parse_payload(path.read_bytes(), role=role)
    except BlendContractError as error:
        raise ContractError(str(error)) from error


def _frozen_prediction_hash(slot: int, role: str) -> str:
    if slot == 12:
        return V34_RESULT_SHA256 if role == "V34T" else V30_RESULT_SHA256
    source = V34 if role == "V34T" else V30
    record = json.loads((source / "development_predictions_frozen.json").read_text(encoding="utf-8"))["predictions"]
    key = f"{slot}_B" if role == "V34T" else f"{slot}_{V30_CANDIDATE}"
    return record[key]["sha256"]


def _registry_search() -> dict:
    manifests = sorted(Path("local/runs").glob("**/*manifest*.json"))
    matches = []
    for path in manifests:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if PROTOCOL in text or CANDIDATE in text:
            run_root = path.parent
            matches.append({
                "path": str(path),
                "sha256": file_sha256(path),
                "complete": (run_root / "completion.json").is_file(),
            })
    return {"searched_manifest_count": len(manifests), "equivalent_matches": matches}


def register(root: Path) -> None:
    config = registration()
    scope = load_yaml("configs/optimization_v0_36/access_scope.yaml")
    if root.exists():
        raise ContractError("never overwrite a v0.36 run")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ContractError("clean committed v0.36 working tree required")
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if branch != config["branch"]:
        raise ContractError("registered v0.36 branch required")
    subprocess.run(["git", "merge-base", "--is-ancestor", config["base_commit"], "HEAD"], check=True)
    registry = _registry_search()
    completed = [item for item in registry["equivalent_matches"] if item["complete"]]
    if completed:
        raise ContractError("equivalent completed v0.36 experiment already exists")
    evidence = {
        "v34_result": V34_FINAL / "result.csv",
        "v34_zip": V34_FINAL / "Luqhhh_bf_tap_predict_prelim.zip",
        "v34_completion": V34 / "completion.json",
        "v34_cold": V34 / "cold_validation.json",
        "v34_errors": V34 / "all_errors.csv",
        "v30_result": V30_FINAL / "result.csv",
        "v30_zip": V30_FINAL / "Luqhhh_bf_tap_predict_prelim.zip",
        "v30_completion": V30 / "completion.json",
        "v30_cold": V30 / "cold_validation.json",
        "protection": Path(scope["protection_contract"]),
    }
    identities = file_identities(evidence)
    expected = {
        "v34_result": V34_RESULT_SHA256,
        "v34_zip": V34_ZIP_SHA256,
        "v30_result": V30_RESULT_SHA256,
        "v30_zip": V30_ZIP_SHA256,
    }
    for key, digest in expected.items():
        if identities[key]["sha256"] != digest:
            raise ContractError(f"registered source identity differs: {key}")
    manifest = {
        "registration": config,
        "scope": scope,
        "protection": load_yaml(scope["protection_contract"]),
        "evidence": identities,
        "sources": file_identities({str(path): path for path in _source_files()}),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_environment(),
        "registry_search": registry,
        "test_targets_read": False,
        "source_model_training": False,
        "new_fit_budget": 0,
        "platform_feedback_used_to_select_endpoints": True,
        "user_supplied_single_remaining_slot": True,
    }
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    atomic_write_json(root / "manifest.json", manifest)
    append_ledger(scope, root)
    atomic_write_json(root / "registration.json", {
        "status": "ONE_ZERO_FIT_CANDIDATE_REGISTERED",
        "candidate": CANDIDATE,
        "protocol": PROTOCOL,
        "platform_submission_budget": 1,
        "restore_upload_budget": 0,
        "automatic_uploads": 0,
        "V35A": "UNTESTED_SKIPPED_SLOT_REALLOCATED",
        "V35B": "USER_REPORTED_83_2778_CLOSED",
    })


def prepare(root: Path) -> None:
    if (root / "p0_complete.json").exists():
        raise ContractError("v0.36 P0 already complete")
    receipts = {}
    changed = {}
    for slot in range(6, 13):
        paths = {role: _endpoint_path(slot, role) for role in ("V34T", "V30A")}
        for role, path in paths.items():
            digest = file_sha256(path)
            if digest != _frozen_prediction_hash(slot, role):
                raise ContractError(f"slot {slot} {role} differs from frozen source record")
            receipts[f"{slot}_{role}"] = {"path": str(path.resolve()), "sha256": digest}
        left, right = _rows(paths["V34T"], "V34T"), _rows(paths["V30A"], "V30A")
        candidate = compose_rows(left, right)
        changed[str(slot)] = {
            "rows": len(candidate),
            "changed_time_vs_V34T": sum(a.pred_tap_time_len != b.pred_tap_time_len for a, b in zip(candidate, left, strict=True)),
            "changed_time_vs_V30A": sum(a.pred_tap_time_len != b.pred_tap_time_len for a, b in zip(candidate, right, strict=True)),
            "iron_strings_exact": all(a.pred_tap_iron == b.pred_tap_iron == c.pred_tap_iron for a, b, c in zip(candidate, left, right, strict=True)),
        }
    atomic_write_json(root / "p0_complete.json", {
        "status": "PASS_SOURCE_IDENTITY_REGISTRY_AND_SINGLE_SLOT_AUDIT",
        "source_receipts": receipts,
        "changed_rows": changed,
        "registry_search": json.loads((root / "manifest.json").read_text(encoding="utf-8"))["registry_search"],
        "historical_replay_available": True,
        "new_fits": 0,
        "test_targets_read": False,
    })


def develop(root: Path) -> None:
    if (root / "development_predictions_frozen.json").exists():
        raise ContractError("v0.36 development already complete")
    identities = {}
    for slot in ORIGINS:
        value = compose_rows(_rows(_endpoint_path(slot, "V34T"), "V34T"), _rows(_endpoint_path(slot, "V30A"), "V30A"))
        path = root / "predictions" / f"{slot}_{CANDIDATE}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload_bytes(value))
        identities[str(slot)] = {"path": str(path.resolve()), "sha256": file_sha256(path), "rows": len(value)}
    atomic_write_json(root / "development_predictions_frozen.json", {
        "status": "ONE_CANDIDATE_HISTORY_FROZEN_BEFORE_SCORING",
        "predictions": identities,
        "platform_feedback_seen_for_v36": False,
        "new_fits": 0,
    })


def _candidate_errors(root: Path) -> pd.DataFrame:
    source = pd.read_csv(V34 / "all_errors.csv", dtype={"sample_id": str})
    base = source.loc[source.candidate == V34_CANDIDATE].copy()
    frames = []
    for origin, part in base.groupby("origin", sort=False):
        slot = int(str(origin)[-2:])
        prediction = pd.read_csv(root / "predictions" / f"{slot}_{CANDIDATE}.csv", dtype=str, keep_default_na=False).set_index("sample_id")
        aligned = prediction.loc[part.sample_id.astype(str)]
        current = part.copy()
        current["pred_tap_iron"] = aligned.pred_tap_iron.to_numpy(dtype=float)
        current["pred_tap_time_len"] = aligned.pred_tap_time_len.to_numpy(dtype=float)
        for target in TARGETS:
            current[f"error_{target}"] = current[f"pred_{target}"] - current[target]
            current[f"abs_error_{target}"] = current[f"error_{target}"].abs()
        current["candidate"] = CANDIDATE
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def score(root: Path) -> None:
    if (root / "offline_assessment.json").exists():
        raise ContractError("never overwrite v0.36 scoring evidence")
    source = pd.read_csv(V34 / "all_errors.csv", dtype={"sample_id": str})
    references = {name: source.loc[source.candidate == name].copy() for name in REFERENCES}
    if any(value.empty for value in references.values()):
        raise ContractError("missing registered v0.36 reference errors")
    candidate = _candidate_errors(root)
    all_errors = pd.concat([*references.values(), candidate], ignore_index=True)
    all_errors.to_csv(root / "all_errors.csv", index=False)
    algorithms = {**references, CANDIDATE: candidate}
    scorecard = pd.DataFrame([row for name, errors in algorithms.items() for row in _score_rows(errors, name)])
    scorecard.to_csv(root / "canonical_scorecard.csv", index=False)
    summary = macro_origin_summary(scorecard)
    summary.to_csv(root / "canonical_summary.csv", index=False)
    pooled = exposure_pooled_summary(scorecard)
    pooled.to_csv(root / "exposure_pooled_summary.csv", index=False)
    isolation = verify_isolated_deltas(scorecard, summary, {CANDIDATE: "tap_time_len"}, parent=V34_CANDIDATE)
    primary = summary.loc[summary.scope_type.isin(("HORIZON_MEAN", "GRID", "DEVELOPMENT"))].set_index(["algorithm", "scope"])
    delta_rows = []
    for reference in REFERENCES:
        for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT"):
            left, right = primary.loc[(CANDIDATE, scope)], primary.loc[(reference, scope)]
            delta_rows.append({
                "candidate": CANDIDATE, "reference": reference, "scope": scope,
                "aggregation": PRIMARY_AGGREGATION, "E": float(left.E),
                "delta_E": float(left.E - right.E),
                "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                "delta_wmape_time": float(left.wmape_time - right.wmape_time),
            })
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(root / "canonical_deltas.csv", index=False)
    pooled_index = pooled.set_index(["algorithm", "scope"])
    pooled_rows = []
    for scope in ("H1", "H2", "H3", "H4", "DEV_LONG", "DEV_SHORT"):
        left, right = pooled_index.loc[(CANDIDATE, scope)], pooled_index.loc[(V34_CANDIDATE, scope)]
        pooled_rows.append({
            "candidate": CANDIDATE, "reference": V34_CANDIDATE, "scope": scope,
            "aggregation": POOLED_AGGREGATION, "delta_E": float(left.E - right.E),
            "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
            "delta_wmape_time": float(left.wmape_time - right.wmape_time),
        })
    pd.DataFrame(pooled_rows).to_csv(root / "exposure_pooled_deltas.csv", index=False)
    left = references[V34_CANDIDATE].set_index(["unit", "sample_id"])
    right = references[V30_CANDIDATE].set_index(["unit", "sample_id"])
    mixed = candidate.set_index(["unit", "sample_id"])
    if not left.index.equals(right.index) or not left.index.equals(mixed.index):
        raise ContractError("v0.36 cancellation rows are not aligned")
    endpoint_average = (left.abs_error_tap_time_len + right.abs_error_tap_time_len) / 2.0
    cancellation = endpoint_average - mixed.abs_error_tap_time_len
    if cancellation.min() < -0.5e-6 - 1e-12:
        raise ContractError("v0.36 rounded blend violates per-row convexity bound")
    diagnostic = pd.DataFrame({
        "unit": left.index.get_level_values("unit"),
        "sample_id": left.index.get_level_values("sample_id"),
        "endpoint_average_abs_error": endpoint_average.to_numpy(),
        "blend_abs_error": mixed.abs_error_tap_time_len.to_numpy(),
        "cancellation_after_rounding": cancellation.to_numpy(),
    })
    diagnostic.to_csv(root / "error_cancellation_diagnostics.csv", index=False)
    lines = [
        "# optimization-v0.36 自动生成离线报告", "",
        "主口径为 `macro_origin_mean_wmape`，变化为候选减参照，负数表示改善。", "",
        "| 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for reference in REFERENCES:
        part = deltas.loc[deltas.reference == reference].set_index("scope")
        values = [part.loc[scope, "delta_E"] for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")]
        lines.append(f"| {reference} | " + " | ".join(f"{value:+.8f}" for value in values) + " |")
    lines.extend(["", "这是已消费历史回放，不是平台结果或独立 holdout。", ""])
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_write_json(root / "offline_assessment.json", {
        "status": "COMPLETED_CONSUMED_RETROSPECTIVE_NOT_A_PLATFORM_GATE",
        "primary_aggregation": PRIMARY_AGGREGATION,
        "diagnostic_aggregation": POOLED_AGGREGATION,
        "isolated_target_audit": isolation,
        "minimum_per_row_cancellation_after_rounding": float(cancellation.min()),
        "total_time_absolute_error_cancellation_after_rounding": float(cancellation.sum()),
        "platform_slot_retained_despite_offline_result": True,
    })


def finalize(root: Path) -> None:
    if not (root / "offline_assessment.json").is_file() or (root / "package_frozen.json").exists():
        raise ContractError("score v0.36 once before finalization")
    folder = root / "submission" / CANDIDATE
    receipt = build_package(
        v34_result=V34_FINAL / "result.csv", v34_zip=V34_FINAL / "Luqhhh_bf_tap_predict_prelim.zip",
        v30_result=V30_FINAL / "result.csv", v30_zip=V30_FINAL / "Luqhhh_bf_tap_predict_prelim.zip",
        output_dir=folder, expected_rows=registration()["expected_rows"],
    )
    atomic_write_json(root / "package_frozen.json", {
        "status": "PASS_ONE_COMPLETE_PACKAGE_FROZEN_NOT_UPLOADED",
        "candidate": CANDIDATE,
        "receipt": receipt,
        "result_sha256": file_sha256(folder / "result.csv"),
        "zip_sha256": file_sha256(folder / "Luqhhh_bf_tap_predict_prelim.zip"),
        "platform_submission_budget": 1,
        "platform_uploads": 0,
        "restore_uploads": 0,
        "second_candidate_generated": False,
    })


def _assert_rows_equal(actual, expected, message):
    if tuple(actual) != tuple(expected):
        raise ContractError(message)


def cold(root: Path) -> None:
    if (root / "cold_validation.json").exists() or (root / "completion.json").exists():
        raise ContractError("never overwrite v0.36 cold/completion evidence")
    expected_path = root / "submission" / CANDIDATE / "result.csv"
    expected = _rows(expected_path, "v0.36 frozen")
    cold_folder = root / "cold" / "rebuilt"
    subprocess.run([
        sys.executable, "scripts/build_v36.py", "--output", str(cold_folder.resolve())
    ], check=True, stdout=subprocess.DEVNULL)
    rebuilt_path = cold_folder / "result.csv"
    if rebuilt_path.read_bytes() != expected_path.read_bytes():
        raise ContractError("cold subprocess payload differs")
    left = _rows(_endpoint_path(12, "V34T"), "V34T")
    right = _rows(_endpoint_path(12, "V30A"), "V30A")
    _assert_rows_equal(compose_rows(left, right), expected, "cold full blend differs")
    _assert_rows_equal(compose_rows(tuple(reversed(left)), tuple(reversed(right))), tuple(reversed(expected)), "cold reverse blend differs")
    indices = list(range(0, len(left), 7))
    _assert_rows_equal(compose_rows(tuple(left[i] for i in indices), tuple(right[i] for i in indices)), tuple(expected[i] for i in indices), "cold subset blend differs")
    pieces = []
    for indices_part in np.array_split(np.arange(len(left)), 5):
        pieces.extend(compose_rows(tuple(left[int(i)] for i in indices_part), tuple(right[int(i)] for i in indices_part)))
    _assert_rows_equal(pieces, expected, "cold chunk blend differs")
    _assert_rows_equal(compose_rows(left[:1], right[:1]), expected[:1], "cold single blend differs")
    package = json.loads((root / "package_frozen.json").read_text(encoding="utf-8"))
    archive = root / "submission" / CANDIDATE / "Luqhhh_bf_tap_predict_prelim.zip"
    if file_sha256(expected_path) != package["result_sha256"] or file_sha256(archive) != package["zip_sha256"]:
        raise ContractError("v0.36 frozen package identity changed")
    source_evidence = file_identities({
        "v34_completion": V34 / "completion.json", "v34_cold": V34 / "cold_validation.json",
        "v30_completion": V30 / "completion.json", "v30_cold": V30 / "cold_validation.json",
    })
    atomic_write_json(root / "cold_validation.json", {
        "status": "PASS_PAYLOAD_LEVEL_COLD_REBUILD",
        "source_model_cold_audit_completed_by_this_tool": False,
        "prior_complete_source_chain_cold_evidence_reused": source_evidence,
        "composition_checks": "full_reverse_chunk_subset_single_exact",
        "independent_subprocess_rebuild_exact": True,
        "zip_and_csv_reread": True,
        "fit_attempts": {"model": 0, "tree": 0, "preprocessor": 0, "calibration_or_coefficient": 0},
    })
    atomic_write_json(root / "fit_counts.json", {
        "new_model_attempted": 0, "new_model_completed": 0,
        "new_tree_attempted": 0, "new_tree_completed": 0,
        "new_preprocessor_attempted": 0, "new_preprocessor_completed": 0,
        "new_calibration_or_coefficient_attempted": 0, "new_calibration_or_coefficient_completed": 0,
    })
    atomic_write_json(root / "completion.json", {
        "status": "READY_FOR_ONE_EXPLICIT_PLATFORM_SUBMISSION",
        "G0_engineering": "PASS",
        "G1_model_quality": "REPORTED_SEPARATELY_NOT_A_GATE",
        "candidate": CANDIDATE,
        "platform_submission_budget": 1,
        "platform_uploads": 0,
        "restore_uploads": 0,
        "automatic_uploads": 0,
        "desktop_writes": 0,
        "package_frozen_sha256": file_sha256(root / "package_frozen.json"),
        "cold_validation_sha256": file_sha256(root / "cold_validation.json"),
        "fit_counts_sha256": file_sha256(root / "fit_counts.json"),
        "test_targets_read": False,
        "V35A": "UNTESTED_SKIPPED_SLOT_REALLOCATED",
        "second_candidate_generated": False,
    })


def run(root: Path) -> None:
    register(root)
    prepare(root)
    develop(root)
    score(root)
    finalize(root)
    cold(root)


__all__ = ["cold", "develop", "finalize", "prepare", "register", "run", "score"]
