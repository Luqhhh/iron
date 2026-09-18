"""Registered zero-fit v0.29 lifecycle for frozen-forest OOB leaf responses."""
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
from .dual_burden_v24_run import V22, V23, _reference_errors
from .dual_ratio_common import week_intervals
from .fixed_blend_v28 import submission_bytes, validate_endpoint
from .oob_leaf_v29 import (
    CANDIDATE_A,
    CANDIDATE_B,
    PARENT,
    POOLED_AGGREGATION,
    PRIMARY_AGGREGATION,
    TARGETS,
    V21_REPLAY,
    V26A_PARENT,
    compose_iron_candidate,
    compose_time_candidate,
    exposure_pooled_summary,
    macro_origin_summary,
    verify_isolated_deltas,
)
from .qrf_partition_v26 import roundtrip_six
from .qrf_partition_v26_run import _replay as replay_qrf, _score_rows
from .structural_run import append_ledger


V26 = Path("local/runs/optimization-v0.26-qrf-partition-tests-r1")
V27 = Path("local/runs/optimization-v0.27-qrf-iron-and-leaf-recency-r1")
V28 = Path("local/runs/optimization-v0.28-fixed-equal-blends-r3")
V26_WORKER = Path("workers/qrf_v026/worker.py")
V27_WORKER = Path("workers/qrf_v027/worker.py")
V29_WORKER = Path("workers/qrf_v029/worker.py")
WORKER_PYTHON = Path("workers/qrf_v015/.venv/bin/python")
ORIGINS = range(6, 12)
SLOTS = range(6, 13)
CANDIDATES = {"A": CANDIDATE_A, "B": CANDIDATE_B}
V28_SOURCE_ID = "V28I_CB_QRF_EQUAL_BLEND"
V26_SOURCE_ID = "V26A_QRF_ABSOLUTE_SPLIT_TIME"
V28_RESULT = V28 / "submissions" / V28_SOURCE_ID / "result.csv"
V28_ZIP = V28 / "submissions" / V28_SOURCE_ID / "Luqhhh_bf_tap_predict_prelim.zip"
V26_RESULT = V26 / "submissions" / V26_SOURCE_ID / "result.csv"
V21_RESULT = V23 / "recovery/result.csv"


def registration():
    value = load_yaml("configs/optimization_v0_29/experiment.yaml")
    if value["branch"] != "optimization-v0.29-oob-leaf-responses" or value["candidates"] != CANDIDATES:
        raise ContractError("v0.29 branch/candidate registration differs")
    if value["protocol"] != "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029":
        raise ContractError("v0.29 OOB protocol differs")
    expected = {
        "forest_or_tree_fits": 0, "other_model_fits": 0, "preprocessor_fits": 0,
        "calibration_fits": 0, "oob_attachments_A": 7, "oob_attachments_B": 7,
        "new_candidate_packages": 2, "new_candidate_platform_tests": 2,
    }
    if value["budget"] != expected or value["platform_order"] != ["A", "B"]:
        raise ContractError("v0.29 budget/order registration differs")
    protocol = value["leaf_response"]
    if protocol["empty_oob_leaf_fallback"] != "full_same_leaf" or protocol["minimum_oob_members"] != 1:
        raise ContractError("v0.29 leaf fallback/threshold differs")
    return value


def _source_files():
    return [
        Path("configs/optimization_v0_29/experiment.yaml"),
        Path("configs/optimization_v0_29/access_scope.yaml"),
        Path("src/bf_tap/optimization/oob_leaf_v29.py"),
        Path("src/bf_tap/optimization/oob_leaf_v29_run.py"),
        Path("scripts/optimization_v29_oob_leaf.py"),
        Path("scripts/optimization_v29_cold_check.py"),
        Path("tests/test_optimization_v29_oob_leaf.py"),
        Path("docs/optimization_v0_29/PLAN.md"),
        V29_WORKER,
        Path("workers/qrf_v029/oob_response.py"),
        Path("workers/qrf_v029/test_oob_response.py"),
        V26_WORKER,
        V27_WORKER,
        Path("workers/qrf_v026/partition_forest.py"),
        Path("workers/qrf_v027/target_forest.py"),
        Path("workers/qrf_v015/qrf_model.py"),
        Path("workers/qrf_v015/preprocessing.py"),
        Path("workers/qrf_v015/uv.lock"),
    ]


def _source_path(slot: int, role: str) -> Path:
    if role == "parent":
        return V28_RESULT if slot == 12 else V28 / "predictions" / f"{slot}_{V28_SOURCE_ID}.csv"
    if role == "v26a":
        return V26_RESULT if slot == 12 else V26 / "predictions" / f"{slot}_{V26_SOURCE_ID}.csv"
    if role == "v21":
        return V21_RESULT if slot == 12 else V22 / "predictions" / str(slot) / "V21_REPLAY.csv"
    raise ContractError("unknown v0.29 endpoint role")


def _load_endpoint(slot: int, role: str) -> pd.DataFrame:
    path = _source_path(slot, role)
    if not path.is_file():
        raise ContractError(f"missing frozen {role} endpoint for slot {slot}")
    return validate_endpoint(pd.read_csv(path, dtype=str, keep_default_na=False), label=f"{role}/{slot}")


def _registry_search():
    manifests = sorted(Path("local/runs").glob("**/*manifest*.json"))
    matches = []
    for path in manifests:
        text = path.read_text(encoding="utf-8", errors="ignore")
        reasons = [item for item in (*CANDIDATES.values(), registration()["protocol"]) if item in text]
        if reasons:
            matches.append({"path": str(path), "sha256": file_sha256(path), "matches": reasons})
    completed, incomplete = [], []
    for match in matches:
        root = Path(match["path"]).parent
        (completed if (root / "completion.json").is_file() else incomplete).append(match)
    return {"searched_manifest_count": len(manifests), "equivalent_matches": completed, "incomplete_equivalent_attempts": incomplete}


def _worker(script: Path, command: str, *arguments):
    return subprocess.check_output([str(WORKER_PYTHON), str(script), command, *map(str, arguments)], text=True)


def register(root: Path):
    config = registration()
    scope = load_yaml("configs/optimization_v0_29/access_scope.yaml")
    if root.exists():
        raise ContractError("never overwrite a v0.29 run")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ContractError("clean preregistered working tree required")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != config["branch"]:
        raise ContractError("registered v0.29 branch required")
    subprocess.run(["git", "merge-base", "--is-ancestor", config["base_commit"], "HEAD"], check=True)
    registry = _registry_search()
    if registry["equivalent_matches"]:
        raise ContractError("equivalent completed v0.29 experiment already exists")
    identities = config["source_identities"]
    for key, path in {
        "v28i_result_sha256": V28_RESULT,
        "v28i_zip_sha256": V28_ZIP,
        "v26a_result_sha256": V26_RESULT,
        "v21_result_sha256": V21_RESULT,
    }.items():
        if file_sha256(path) != identities[key]:
            raise ContractError(f"registered v0.29 source identity differs: {key}")
    evidence = {
        "v28_manifest": V28 / "manifest.json", "v28_completion": V28 / "completion.json",
        "v28_errors": V28 / "all_errors.csv", "v28_parent_result": V28_RESULT, "v28_parent_zip": V28_ZIP,
        "v27_manifest": V27 / "manifest.json", "v27_completion": V27 / "completion.json",
        "v26_manifest": V26 / "manifest.json", "v26_completion": V26 / "completion.json",
        "v26_errors": V26 / "all_errors.csv", "v22_errors": V22 / "all_errors.csv",
        "v23_v21_result": V21_RESULT, "protection": Path(scope["protection_contract"]),
    }
    worker_environment = json.loads(_worker(V29_WORKER, "environment"))
    worker_sources = json.loads(_worker(V29_WORKER, "identity"))
    manifest = {
        "registration": config, "scope": scope, "protection": load_yaml(scope["protection_contract"]),
        "evidence": file_identities(evidence),
        "sources": file_identities({str(path): path for path in _source_files()}),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_environment(), "worker_environment": worker_environment, "worker_sources": worker_sources,
        "oob_protocol": config["protocol"],
        "candidate_targets": {
            "A": {"target": "tap_iron", "unit": "tonne", "source": "V27I_ABS_QRF_DIRECT_IRON"},
            "B": {"target": "tap_time_len", "unit": "minutes", "source": "V26A_QRF_ABSOLUTE_SPLIT_TIME"},
        },
        "registry_search": registry, "holdout_consumed": True, "test_targets_read": False,
        "new_fit_budget": 0, "platform_feedback_may_change_second_candidate": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    atomic_write_json(root / "manifest.json", manifest)
    append_ledger(scope, root)
    atomic_write_json(root / "registration.json", {
        "status": "TWO_ZERO_FIT_OOB_RESPONSE_CANDIDATES_FROZEN_BEFORE_DERIVATION",
        "candidates": CANDIDATES, "definitions": {key: config[f"candidate_{key}"] for key in CANDIDATES},
        "platform_order": ["A", "B"], "candidate_platform_budget": 2, "platform_uploads": 0,
        "definitions_may_change_after_A_feedback": False,
    })
    return manifest


def prepare(root: Path):
    if (root / "p0_complete.json").exists():
        raise ContractError("v0.29 P0 already complete")
    audits = {}
    for slot in SLOTS:
        parent, v26a = _load_endpoint(slot, "parent"), _load_endpoint(slot, "v26a")
        if parent.sample_id.tolist() != v26a.sample_id.tolist() or parent.pred_tap_time_len.tolist() != v26a.pred_tap_time_len.tolist():
            raise ContractError("V28I parent/V26A complete time identity differs")
        audits[str(slot)] = {}
        for candidate in CANDIDATES:
            audits[str(slot)][candidate] = json.loads(_worker(
                V29_WORKER, "audit", "--root", root.resolve(), "--slot", slot, "--candidate", candidate,
            ))
            if any(audits[str(slot)][candidate]["zero_fit"].values()):
                raise ContractError("P0 OOB audit attempted a fit")
    atomic_write_json(root / "p0_complete.json", {
        "status": "PASS_ZERO_FIT_SOURCE_BOOTSTRAP_AND_FULL_MEMBER_REPLAY_AUDIT",
        "audits": audits, "registry_search": read_json(root / "manifest.json")["registry_search"],
        "new_model_fits": 0, "new_preprocessor_fits": 0, "new_calibration_fits": 0,
        "test_targets_read": False,
    })


def _write_prediction(path: Path, value: pd.DataFrame):
    if path.exists():
        raise ContractError("never overwrite a v0.29 prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(submission_bytes(value))


def _compose_slot(root: Path, slot: int, *, derive: bool):
    parent, v26a = _load_endpoint(slot, "parent"), _load_endpoint(slot, "v26a")
    raw = {}
    for candidate in CANDIDATES:
        output = root / "worker_predictions" / candidate / f"{slot}.npz"
        if derive:
            _worker(V29_WORKER, "derive-predict", "--root", root.resolve(), "--slot", slot, "--candidate", candidate, "--output", output.resolve())
        with np.load(output, allow_pickle=False) as source:
            if source["ids"].tolist() != parent.sample_id.tolist():
                raise ContractError("v0.29 worker prediction IDs differ")
            raw[candidate] = source["median"]
    a = compose_iron_candidate(parent, v26a, raw["A"])
    replayed, replay_audit = replay_qrf(V26, slot, raw["B"])
    b = compose_time_candidate(parent, replayed.pred_tap_time_len.to_numpy(dtype=float))
    _write_prediction(root / "predictions" / f"{slot}_{CANDIDATE_A}.csv", a)
    _write_prediction(root / "predictions" / f"{slot}_{CANDIDATE_B}.csv", b)
    atomic_write_json(root / "predictions" / f"{slot}_audit.json", {
        "A": {"base_complete_endpoint": V26A_PARENT, "oob_endpoint": "V27I_OOB", "weight_each": .5,
              "unchanged_time_exact": a.pred_tap_time_len.tolist() == parent.pred_tap_time_len.tolist()},
        "B": {"raw_oob_endpoint": "V26A_TIME_OOB", "replay": replay_audit,
              "unchanged_iron_exact": b.pred_tap_iron.tolist() == parent.pred_tap_iron.tolist(),
              "gate_source": "recomputed_original_v015_effective_neighbors"},
    })


def develop(root: Path):
    if (root / "development_complete.json").exists():
        raise ContractError("v0.29 development already complete")
    for slot in ORIGINS:
        _compose_slot(root, slot, derive=True)
    attachment_files = {
        f"{candidate}_{slot}": root / "oob_attachments" / candidate / f"{slot}.npz"
        for candidate in CANDIDATES for slot in ORIGINS
    }
    predictions = {
        f"{slot}_{candidate}": root / "predictions" / f"{slot}_{candidate}.csv"
        for slot in ORIGINS for candidate in CANDIDATES.values()
    }
    atomic_write_json(root / "development_predictions_frozen.json", {
        "status": "BOTH_CANDIDATES_AND_TWELVE_ATTACHMENTS_FROZEN_BEFORE_SCORING",
        "predictions": file_identities(predictions), "attachments": file_identities(attachment_files),
        "platform_feedback_seen": False,
    })
    atomic_write_json(root / "development_complete.json", {
        "status": "PASS_G0_DEVELOPMENT_OOB_RESPONSES_FROZEN", "new_fits": 0,
        "oob_attachments": 12, "trees_refit": 0,
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


def _source_errors(path: Path, source_candidate: str, label: str):
    value = pd.read_csv(path, dtype={"sample_id": str})
    result = value.loc[value.candidate == source_candidate].copy()
    if result.empty:
        raise ContractError(f"missing source errors for {label}")
    for target in TARGETS:
        result[f"pred_{target}"] = roundtrip_six(result[f"pred_{target}"])
        result[f"error_{target}"] = result[f"pred_{target}"] - result[target]
        result[f"abs_error_{target}"] = result[f"error_{target}"].abs()
    result["candidate"] = label
    return result


def _oob_diagnostics(root: Path):
    rows = []
    for candidate in CANDIDATES:
        for slot in ORIGINS:
            attachment = read_json(str(root / "oob_attachments" / candidate / f"{slot}.npz") + ".json")
            receipt = read_json(str(root / "worker_predictions" / candidate / f"{slot}.npz") + ".json")
            counts = np.asarray([item["fallback_tree_count"] for item in receipt["neighbors"]], dtype=float)
            rows.append({
                "candidate": CANDIDATES[candidate], "slot": slot, "target": receipt["target"],
                "training_rows": attachment["training_rows"], "leaf_count": attachment["leaf_count"],
                "fallback_leaf_count": attachment["fallback_leaf_count"],
                "fallback_leaf_fraction": attachment["fallback_leaf_fraction"],
                "minimum_full_count": attachment["minimum_full_count"], "maximum_full_count": attachment["maximum_full_count"],
                "minimum_oob_count": attachment["minimum_oob_count"], "maximum_oob_count": attachment["maximum_oob_count"],
                "minimum_selected_count": attachment["minimum_selected_count"], "maximum_selected_count": attachment["maximum_selected_count"],
                "query_fallback_trees_mean": float(counts.mean()), "query_fallback_trees_max": int(counts.max()),
                "lower_boundary": receipt["lower_boundary"], "upper_boundary": receipt["upper_boundary"],
                "prediction_seconds": receipt["prediction_seconds"],
            })
    return pd.DataFrame(rows)


def _write_report(root: Path, deltas: pd.DataFrame):
    lines = [
        "# optimization-v0.29 自动生成离线报告", "",
        "主口径为 `macro_origin_mean_wmape`。负 Δ 表示优于参照；OOB 是叶响应估计，不是独立验证集。", "",
        "| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in CANDIDATES.values():
        for reference in (PARENT, V26A_PARENT, V21_REPLAY, "V1"):
            part = deltas.loc[(deltas.candidate == candidate) & (deltas.reference == reference)].set_index("scope")
            values = [part.loc[scope, "delta_E"] for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")]
            lines.append(f"| {candidate} | {reference} | " + " | ".join(f"{value:+.8f}" for value in values) + " |")
    lines.extend(["", "G0、已消费历史 G1 与平台反馈分别登记。", ""])
    with (root / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def score(root: Path):
    if (root / "offline_assessment.json").exists():
        raise ContractError("never overwrite v0.29 scoring evidence")
    candidates = {candidate: _candidate_errors(root, candidate) for candidate in CANDIDATES.values()}
    references = {
        "V1": _reference_errors("V1"),
        PARENT: _source_errors(V28 / "all_errors.csv", V28_SOURCE_ID, PARENT),
        V26A_PARENT: _source_errors(V26 / "all_errors.csv", V26_SOURCE_ID, V26A_PARENT),
        V21_REPLAY: _source_errors(V22 / "all_errors.csv", "V21_REPLAY", V21_REPLAY),
    }
    all_errors = pd.concat([*references.values(), *candidates.values()], ignore_index=True)
    all_errors.to_csv(root / "all_errors.csv", index=False)
    scorecard = pd.DataFrame([row for name, errors in {**references, **candidates}.items() for row in _score_rows(errors, name)])
    scorecard.to_csv(root / "canonical_scorecard.csv", index=False)
    summary = macro_origin_summary(scorecard)
    summary.to_csv(root / "canonical_summary.csv", index=False)
    pooled = exposure_pooled_summary(scorecard)
    pooled.to_csv(root / "exposure_pooled_summary.csv", index=False)
    isolation = verify_isolated_deltas(scorecard, summary, {CANDIDATE_A: "tap_iron", CANDIDATE_B: "tap_time_len"}, parent=PARENT)
    primary = summary.loc[summary.scope_type.isin(("HORIZON_MEAN", "GRID", "DEVELOPMENT"))].set_index(["algorithm", "scope"])
    rows = []
    for candidate in CANDIDATES.values():
        for reference in references:
            for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT"):
                left, right = primary.loc[(candidate, scope)], primary.loc[(reference, scope)]
                rows.append({
                    "candidate": candidate, "reference": reference, "scope": scope, "aggregation": PRIMARY_AGGREGATION,
                    "E": float(left.E), "delta_E": float(left.E - right.E),
                    "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                    "delta_wmape_time": float(left.wmape_time - right.wmape_time),
                })
    deltas = pd.DataFrame(rows)
    deltas.to_csv(root / "canonical_deltas.csv", index=False)
    pooled_index = pooled.set_index(["algorithm", "scope"])
    pooled_rows = []
    for candidate in candidates:
        for reference in references:
            for scope in ("H1", "H2", "H3", "H4", "DEV_LONG", "DEV_SHORT"):
                left, right = pooled_index.loc[(candidate, scope)], pooled_index.loc[(reference, scope)]
                pooled_rows.append({
                    "candidate": candidate, "reference": reference, "scope": scope, "aggregation": POOLED_AGGREGATION,
                    "delta_E": float(left.E - right.E),
                    "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                    "delta_wmape_time": float(left.wmape_time - right.wmape_time),
                })
    pd.DataFrame(pooled_rows).to_csv(root / "exposure_pooled_deltas.csv", index=False)
    diagnostics = _oob_diagnostics(root)
    diagnostics.to_csv(root / "oob_diagnostics.csv", index=False)
    signed = all_errors.groupby(["candidate", "origin", "horizon", "spout_no"], as_index=False).agg(
        n=("sample_id", "size"), signed_bias_iron=("error_tap_iron", "mean"), signed_bias_time=("error_tap_time_len", "mean"),
    )
    signed.to_csv(root / "signed_bias_by_spout.csv", index=False)
    bootstrap = {
        candidate: {reference: week_intervals(all_errors, candidate, reference, 1000, 2026) for reference in references}
        for candidate in candidates
    }
    atomic_write_json(root / "bootstrap.json", bootstrap)
    _write_report(root, deltas)
    atomic_write_json(root / "offline_assessment.json", {
        "status": "COMPLETED_CONSUMED_RETROSPECTIVE_NOT_A_PLATFORM_GATE",
        "primary_aggregation": PRIMARY_AGGREGATION, "diagnostic_aggregation": POOLED_AGGREGATION,
        "isolated_target_audit": isolation, "OOB_is_not_an_independent_validation_set": True,
        "both_candidates_retain_preregistered_platform_slot": True,
    })


def finalize(root: Path):
    if not (root / "offline_assessment.json").exists() or (root / "packages_frozen_before_feedback.json").exists():
        raise ContractError("score v0.29 once before finalization")
    _compose_slot(root, 12, derive=True)
    parent = _load_endpoint(12, "parent")
    receipts = {}
    for key, candidate in CANDIDATES.items():
        prediction = validate_endpoint(pd.read_csv(root / "predictions" / f"12_{candidate}.csv", dtype=str, keep_default_na=False), label=candidate)
        changed = "pred_tap_iron" if key == "A" else "pred_tap_time_len"
        unchanged = "pred_tap_time_len" if key == "A" else "pred_tap_iron"
        if prediction[unchanged].tolist() != parent[unchanged].tolist():
            raise ContractError("v0.29 unchanged final target string differs from V28I")
        if prediction[changed].tolist() == parent[changed].tolist():
            raise ContractError("v0.29 candidate degenerates to parent")
        folder = root / "submissions" / candidate
        folder.mkdir(parents=True, exist_ok=False)
        csv_path = folder / "result.csv"
        payload = submission_bytes(prediction)
        with csv_path.open("xb") as handle:
            handle.write(payload)
        archive = folder / "Luqhhh_bf_tap_predict_prelim.zip"
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as handle:
            handle.write(csv_path, arcname="result.csv")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != payload:
                raise ContractError("v0.29 ZIP payload differs")
        reread = validate_endpoint(pd.read_csv(csv_path, dtype=str, keep_default_na=False), label="package")
        if len(reread) != registration()["test_a_rows"]:
            raise ContractError("v0.29 package test_a coverage differs")
        receipts[key] = {
            "candidate": candidate, "rows": len(reread), "result_sha256": file_sha256(csv_path),
            "zip_sha256": file_sha256(archive), "unchanged_target": unchanged,
            "unchanged_target_csv_strings_exact": True, "platform_uploads": 0, "platform_feedback": None,
        }
    atomic_write_json(root / "packages_frozen_before_feedback.json", {
        "status": "PASS_TWO_COMPLETE_PACKAGES_FROZEN", "order": ["A", "B"], "receipts": receipts,
        "definitions_may_change_after_A_feedback": False, "third_candidate_generated": False,
        "platform_upload_budget": 2, "platform_uploads": 0,
    })


def _assert_frame_equal(actual, expected, message):
    left = validate_endpoint(actual, label="actual").reset_index(drop=True)
    right = validate_endpoint(expected, label="expected").reset_index(drop=True)
    if not left.equals(right):
        raise ContractError(message)


def cold(root: Path):
    if (root / "cold_validation.json").exists() or (root / "completion.json").exists():
        raise ContractError("never overwrite v0.29 cold/completion evidence")
    receipts = {"source_iron": {}, "source_time": {}, "old_gate": {}, "oob_A": {}, "oob_B": {}}
    zero_fit = True
    for slot in SLOTS:
        cold_dir = root / "cold"
        paths = {name: cold_dir / name / f"{slot}.npz" for name in receipts}
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        _worker(V27_WORKER, "cold-a", "--root", V27.resolve(), "--slot", slot, "--output", paths["source_iron"].resolve())
        _worker(V26_WORKER, "cold", "--root", V26.resolve(), "--slot", slot, "--candidate", "A", "--output", paths["source_time"].resolve())
        _worker(V26_WORKER, "old-cold", "--root", V26.resolve(), "--slot", slot, "--output", paths["old_gate"].resolve())
        for key in ("A", "B"):
            _worker(V29_WORKER, "cold", "--root", root.resolve(), "--slot", slot, "--candidate", key, "--output", paths[f"oob_{key}"].resolve())
        for role, path in paths.items():
            info = read_json(str(path) + ".json")
            receipts[role][str(slot)] = info
            zero_fit &= not any(info["zero_fit"].values())
        # Cold workers recomputed and byte-compared saved NPZs; compose again
        # from those certified arrays and compare the frozen complete outputs.
        parent, v26a = _load_endpoint(slot, "parent"), _load_endpoint(slot, "v26a")
        with np.load(root / "worker_predictions" / "A" / f"{slot}.npz", allow_pickle=False) as source:
            a = compose_iron_candidate(parent, v26a, source["median"])
        with np.load(root / "worker_predictions" / "B" / f"{slot}.npz", allow_pickle=False) as source:
            replayed, _ = replay_qrf(V26, slot, source["median"])
            b = compose_time_candidate(parent, replayed.pred_tap_time_len.to_numpy(dtype=float))
        _assert_frame_equal(a, pd.read_csv(root / "predictions" / f"{slot}_{CANDIDATE_A}.csv", dtype=str, keep_default_na=False), "cold A differs")
        _assert_frame_equal(b, pd.read_csv(root / "predictions" / f"{slot}_{CANDIDATE_B}.csv", dtype=str, keep_default_na=False), "cold B differs")
    if not zero_fit:
        raise ContractError("v0.29 cold restoration attempted a fit")
    package = read_json(root / "packages_frozen_before_feedback.json")
    for key, candidate in CANDIDATES.items():
        folder = root / "submissions" / candidate
        csv_path, archive = folder / "result.csv", folder / "Luqhhh_bf_tap_predict_prelim.zip"
        receipt = package["receipts"][key]
        if file_sha256(csv_path) != receipt["result_sha256"] or file_sha256(archive) != receipt["zip_sha256"]:
            raise ContractError("v0.29 frozen package identity changed")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != csv_path.read_bytes():
                raise ContractError("v0.29 cold ZIP reread differs")
    fit_attempts = {
        "CatBoost": 0, "RandomForestRegressor": 0, "ExtraTreesRegressor": 0,
        "PartitionForest": 0, "IronTargetForest": 0, "old_QRF": 0,
        "preprocessor": 0, "LAD_or_calibration": 0,
    }
    atomic_write_json(root / "cold_validation.json", {
        "status": "PASS", "all_slots": list(SLOTS), "source_model_cold_audit_completed_by_this_lifecycle": True,
        "bootstrap_and_attachment_rederived_exact": True,
        "source_checks": "full_reverse_chunk_subset_single_exact", "oob_checks": "full_reverse_chunk_subset_single_exact",
        "fit_attempts_during_cold": fit_attempts, "trusted_private_bundles_only": True,
        "source_files_and_tree_structures_unchanged": True, "zip_and_csv_reread": True, "receipts": receipts,
    })
    atomic_write_json(root / "fit_counts.json", {
        "new_base_model_attempted": 0, "new_base_model_completed": 0,
        "new_tree_attempted": 0, "new_tree_completed": 0,
        "new_preprocessor_attempted": 0, "new_preprocessor_completed": 0,
        "new_calibration_attempted": 0, "new_calibration_completed": 0,
        "oob_attachment_attempted": 14, "oob_attachment_completed": 14,
    })
    atomic_write_json(root / "completion.json", {
        "status": "READY_FOR_TWO_EXPLICIT_PLATFORM_SUBMISSIONS", "G0_engineering": "PASS",
        "G1_model_quality": "REPORTED_SEPARATELY_NOT_A_GATE", "candidates": list(CANDIDATES.values()),
        "platform_order": ["A", "B"], "platform_uploads": 0, "platform_budget": 2,
        "packages_sha256": file_sha256(root / "packages_frozen_before_feedback.json"),
        "cold_validation_sha256": file_sha256(root / "cold_validation.json"),
        "fit_counts_sha256": file_sha256(root / "fit_counts.json"), "test_targets_read": False,
        "automatic_upload": False, "desktop_writes": 0, "public_pushes": 0, "third_candidate_generated": False,
    })


def run(root: Path):
    register(root)
    prepare(root)
    develop(root)
    score(root)
    finalize(root)
    cold(root)

