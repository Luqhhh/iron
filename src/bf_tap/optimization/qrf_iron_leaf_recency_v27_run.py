"""Registered v0.27 lifecycle for direct-iron QRF and frozen-time leaf recency."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json, file_identities, file_sha256, runtime_environment, stable_digest
from ..config import load_yaml
from ..exceptions import ContractError
from .component_export import read_json
from .dual_burden_v24_run import V22, V23, _history, _reference_errors, _v10, query_frame
from .dual_ratio_common import week_intervals
from .qrf_iron_leaf_recency_v27 import (
    CANDIDATE_A,
    CANDIDATE_B,
    PARENT,
    POOLED_AGGREGATION,
    PRIMARY_AGGREGATION,
    TARGETS,
    TOLERANCE,
    exposure_pooled_summary,
    isolate_iron,
    isolate_time,
    macro_origin_summary,
    roundtrip_six,
    submission_bytes_preserving_other_target,
    verify_isolated_deltas,
)
from .qrf_partition_v26_run import _score_rows
from .qrf_support_shrink import replay_v21, serialize_six_decimals
from .structural_run import append_ledger


WORKER = Path("workers/qrf_v027")
V26_WORKER = Path("workers/qrf_v026")
V26 = Path("local/runs/optimization-v0.26-qrf-partition-tests-r1")
V25 = Path("local/runs/optimization-v0.25-history-centered-targets-r1")
ORIGINS = range(6, 12)
SLOTS = range(6, 13)
CANDIDATES = {"A": CANDIDATE_A, "B": CANDIDATE_B}
PROTOCOLS = {"A": "QRF_FULLTRAIN_TARGET_v027", "B": "QRF_FROZEN_PARTITION_LEAF_RECENCY60_v027"}
PREDICTION_COLUMNS = ["pred_tap_iron", "pred_tap_time_len"]
PARENT_ID = "V26A_QRF_ABSOLUTE_SPLIT_TIME"
PARENT_ZIP = V26 / "submissions" / PARENT_ID / "Luqhhh_bf_tap_predict_prelim.zip"
PARENT_RESULT = V26 / "submissions" / PARENT_ID / "result.csv"


def registration():
    value = load_yaml("configs/optimization_v0_27/experiment.yaml")
    if value["candidates"] != CANDIDATES or value["base_commit"] != "193842e81385f51efd48792df652fc03e30cc812":
        raise ContractError("v0.27 candidate/base registration differs")
    if value["candidate_A"]["protocol"] != PROTOCOLS["A"] or value["candidate_B"]["protocol"] != PROTOCOLS["B"]:
        raise ContractError("v0.27 protocols differ")
    expected = {
        "A_development_forest_fits": 6,
        "A_final_forest_fits": 1,
        "A_total_forest_fits": 7,
        "A_internal_trees_total": 1792,
        "B_forest_fits": 0,
        "B_weight_vectors": 7,
        "new_preprocessor_fits": 0,
        "catboost_rate_e04_q_fits": 0,
        "LAD_beta_lambda_bias_fits": 0,
        "new_candidate_packages": 2,
        "new_candidate_platform_tests": 2,
    }
    if value["budget"] != expected or value["platform_order"] != ["A", "B"]:
        raise ContractError("v0.27 budget/order registration differs")
    return value


def worker(command, *arguments):
    executable = Path("workers/qrf_v015/.venv/bin/python")
    return subprocess.check_output([str(executable), str(WORKER / "worker.py"), command, *map(str, arguments)], text=True)


def v26_worker(command, *arguments):
    executable = Path("workers/qrf_v015/.venv/bin/python")
    return subprocess.check_output([str(executable), str(V26_WORKER / "worker.py"), command, *map(str, arguments)], text=True)


def _source_files():
    return [
        Path("configs/optimization_v0_27/experiment.yaml"),
        Path("configs/optimization_v0_27/access_scope.yaml"),
        Path("src/bf_tap/optimization/qrf_iron_leaf_recency_v27.py"),
        Path("src/bf_tap/optimization/qrf_iron_leaf_recency_v27_run.py"),
        Path("scripts/optimization_v27_qrf_iron_leaf_recency.py"),
        Path("scripts/optimization_v27_cold_check.py"),
        Path("tests/test_optimization_v27_qrf_iron_leaf_recency.py"),
        Path("docs/optimization_v0_27/PLAN.md"),
        WORKER / "worker.py",
        WORKER / "target_forest.py",
        WORKER / "leaf_recency.py",
        WORKER / "test_v27_worker_core.py",
        V26_WORKER / "worker.py",
        V26_WORKER / "partition_forest.py",
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
        reasons = []
        if CANDIDATE_A in text or CANDIDATE_B in text:
            reasons.append("candidate_id")
        if PROTOCOLS["A"] in text or PROTOCOLS["B"] in text:
            reasons.append("protocol")
        if reasons:
            matches.append({"path": str(path), "sha256": file_sha256(path), "reasons": reasons})
    return {"searched_manifest_count": len(manifests), "equivalent_matches": matches}


def register(root: Path):
    configuration = registration()
    scope = load_yaml("configs/optimization_v0_27/access_scope.yaml")
    if root.exists():
        raise ContractError("never overwrite a v0.27 run")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ContractError("clean preregistered working tree required")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != configuration["branch"]:
        raise ContractError("registered v0.27 branch required")
    subprocess.run(["git", "merge-base", "--is-ancestor", configuration["base_commit"], "HEAD"], check=True)
    registry = _registry_search()
    if registry["equivalent_matches"]:
        raise ContractError("equivalent v0.27 experiment already exists: " + str(registry["equivalent_matches"]))
    if file_sha256(PARENT_ZIP) != configuration["parent_zip_sha256"] or file_sha256(PARENT_RESULT) != configuration["parent_payload_sha256"]:
        raise ContractError("registered V26A parent package identity differs")
    v25_manifest = read_json(V25 / "manifest.json")
    certified_training_source = v25_manifest["inputs"]["train_samples"]
    evidence = {
        "v26_manifest": V26 / "manifest.json",
        "v26_completion": V26 / "completion.json",
        "v26_development_models": V26 / "development_models_complete.json",
        "v26_final_models": V26 / "final_models_complete.json",
        "v26_errors": V26 / "all_errors.csv",
        "v26_parent_zip": PARENT_ZIP,
        "v26_parent_result": PARENT_RESULT,
        "v25_training_source_certificate": V25 / "manifest.json",
        "v22_errors": V22 / "all_errors.csv",
        "v23_v21_result": V23 / "recovery/result.csv",
        "protection": Path(scope["protection_contract"]),
    }
    environment = json.loads(worker("environment"))
    if environment.get("python") != "3.12.12" or environment.get("sklearn") != "1.8.0":
        raise ContractError("locked Python 3.12.12 / scikit-learn 1.8.0 worker required")
    manifest = {
        "registration": configuration,
        "scope": scope,
        "protection": load_yaml(scope["protection_contract"]),
        "evidence": file_identities(evidence),
        "certified_training_source": certified_training_source,
        "sources": file_identities({str(path): path for path in _source_files()}),
        "worker_environment": environment,
        "worker_sources": json.loads(worker("identity")),
        "iron_forest_parameters": configuration["forest_parameters"],
        "protocols": PROTOCOLS,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_environment(),
        "registry_search": registry,
        "holdout_consumed": True,
        "test_targets_read": False,
        "development_raw_source_full_file_rehashed": False,
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
        "platform_order": ["A", "B"],
        "candidate_platform_budget": 2,
        "platform_uploads": 0,
        "definitions_may_change_after_A_feedback": False,
        "parent_zip_sha256": file_sha256(PARENT_ZIP),
        "parent_payload_sha256": file_sha256(PARENT_RESULT),
    })
    return manifest


def _copy_handoff(root: Path, slot: int, kind: str):
    source = V26 / "features" / str(slot) / f"{kind}.npz"
    source_info = Path(str(source) + ".json")
    info = read_json(source_info)
    if file_sha256(source) != info["sha256"] or len(info["numeric_columns"]) != 209 or len(info["raw_schema"]) != 210:
        raise ContractError("V26 original-schema handoff identity differs")
    destination = root / "features" / str(slot) / f"{kind}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    shutil.copyfile(source_info, Path(str(destination) + ".json"))
    if file_sha256(destination) != file_sha256(source):
        raise ContractError("v0.27 exact V26 handoff copy differs")
    return info


def _load_handoff(root: Path, slot: int, kind: str):
    path = root / "features" / str(slot) / f"{kind}.npz"
    info = read_json(str(path) + ".json")
    if file_sha256(path) != info["sha256"]:
        raise ContractError("v0.27 handoff identity changed")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if arrays["ids"].tolist() != info["ids"]:
        raise ContractError("v0.27 handoff IDs differ")
    return arrays, info


def _parse_local(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("Asia/Shanghai") if stamp.tzinfo is None else stamp.tz_convert("Asia/Shanghai")


def _read_iron_prefix(ids, cutoff, *, allow_protected):
    path = Path(load_yaml("configs/data.local.yaml")["paths"]["train_samples"])
    wanted = set(map(str, ids))
    found = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sample_id", "tap_no", "spout_no", "reference_time", "tap_iron", "tap_time_len"]:
            raise ContractError("official train schema differs")
        for row in reader:
            sample_id = str(row["sample_id"])
            if sample_id in wanted:
                reference = _parse_local(row["reference_time"])
                if reference >= cutoff:
                    raise ContractError("requested iron label reference is outside cutoff")
                found[sample_id] = float(row["tap_iron"])
                if len(found) == len(wanted):
                    break
            elif not allow_protected and _parse_local(row["reference_time"]) >= pd.Timestamp("2024-11-01", tz="Asia/Shanghai"):
                raise ContractError("development iron reader reached protected November rows")
    if set(found) != wanted:
        raise ContractError("certified training iron labels missing")
    result = np.asarray([found[str(sample_id)] for sample_id in ids], dtype=np.float64)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ContractError("invalid raw iron training labels")
    return result, path


def _write_target(root: Path, slot: int, *, final_training: bool):
    arrays, info = _load_handoff(root, slot, "train")
    cutoff = pd.Timestamp(info["cutoff"])
    manifest = read_json(root / "manifest.json")
    certified = manifest["certified_training_source"]
    if final_training:
        path = Path(load_yaml("configs/data.local.yaml")["paths"]["train_samples"])
        if file_sha256(path) != certified["sha256"] or path.stat().st_size != certified["bytes"]:
            raise ContractError("protected final training source identity differs from frozen certificate")
    response, source_path = _read_iron_prefix(arrays["ids"].tolist(), cutoff, allow_protected=final_training)
    output = root / "targets" / str(slot) / "tap_iron.npz"
    output.parent.mkdir(parents=True, exist_ok=False)
    np.savez(output, ids=arrays["ids"], y=response)
    metadata = {
        "sha256": file_sha256(output),
        "slot": slot,
        "ids": arrays["ids"].tolist(),
        "rows": len(response),
        "target": "tap_iron",
        "unit": "tonne",
        "response_sha256": stable_digest(response.tolist()),
        "source_path": str(source_path),
        "certified_full_source_sha256": certified["sha256"],
        "current_full_source_sha256_verified": final_training,
        "reader_stopped_after_requested_ids": True,
        "protected_lifecycle": "final_training" if final_training else None,
        "test_targets_read": False,
    }
    atomic_write_json(str(output) + ".json", metadata)
    return metadata


def _append_final_training_ledger(root: Path):
    scope = load_yaml("configs/optimization_v0_27/access_scope.yaml")
    path = Path(scope["new_ledger"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        prior = handle.read().splitlines()
        handle.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(),
            "purpose": scope["authorization"] + ":final_training",
            "manifest_sha256": file_sha256(root / "manifest.json"),
            "output": str(root),
            "previous_sha256": stable_digest(prior[-1]) if prior else None,
            "holdout_consumed": True,
            "protected_lifecycle": "final_training",
        }) + "\n")


def prepare(root: Path):
    if (root / "p0_complete.json").exists() or any((root / name).exists() for name in ("features", "targets", "old_qrf_predictions")):
        raise ContractError("v0.27 P0 already or partially exists")
    handoffs, audits, support = {}, {}, {}
    for slot in SLOTS:
        handoffs[str(slot)] = {kind: _copy_handoff(root, slot, kind) for kind in ("train", "evaluation")}
        if len(handoffs[str(slot)]["train"]["ids"]) != registration()["expected_training_rows"][slot]:
            raise ContractError("v0.27 certified training row count differs")
        audits[str(slot)] = json.loads(worker("audit", "--root", root.resolve(), "--slot", slot))
        output = root / "old_qrf_predictions" / f"{slot}.npz"
        v26_worker("old-predict", "--root", V26.resolve(), "--slot", slot, "--output", output.resolve())
        support[str(slot)] = read_json(str(output) + ".json")
    targets = {str(slot): _write_target(root, slot, final_training=False) for slot in ORIGINS}
    if support["12"]["gate_rows_spout1_neff_lt_500"] != 32:
        raise ContractError("v0.27 recomputed test_a old-QRF gate identity is not 32 rows")
    atomic_write_json(root / "p0_complete.json", {
        "status": "PASS_ZERO_FIT",
        "registry_search": read_json(root / "manifest.json")["registry_search"],
        "handoffs": handoffs,
        "transformed_input_audits": audits,
        "development_iron_targets": targets,
        "old_qrf_support_recomputed": support,
        "parent_zip_sha256": file_sha256(PARENT_ZIP),
        "parent_payload_sha256": file_sha256(PARENT_RESULT),
        "test_a_gate_rows": 32,
        "model_fits": 0,
        "preprocessor_fits": 0,
        "protected_november_target_reads": 0,
        "timezone": "Asia/Shanghai",
        "availability_contract": "ASSUMED_UNCHANGED",
    })


def _model_completion(root: Path, slots):
    result = {"A": {}}
    for slot in slots:
        folder = root / "models" / "A" / str(slot)
        record = read_json(folder / "fit_record.json")
        digest = file_sha256(folder / "bundle.json")
        if record["status"] != "COMPLETED" or record["bundle_sha256"] != digest:
            raise ContractError("v0.27 iron fit completion identity differs")
        result["A"][str(slot)] = digest
    return result


def _fit_a(root: Path, slot: int):
    folder = root / "models" / "A" / str(slot)
    if (folder / "fit_record.json").is_file():
        return
    if (folder / "fit_intent.json").exists():
        raise ContractError("incomplete v0.27 iron fit attempt retained")
    worker("fit-a", "--root", root.resolve(), "--slot", slot)


def _old_qrf(root: Path, slot: int):
    path = root / "old_qrf_predictions" / f"{slot}.npz"
    info = read_json(str(path) + ".json")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if arrays["ids"].tolist() != _load_handoff(root, slot, "evaluation")[0]["ids"].tolist():
        raise ContractError("v0.27 recomputed old-QRF IDs differ")
    return arrays, info


def _parent(slot: int):
    path = V26 / "predictions" / f"{slot}_{PARENT_ID}.csv"
    return pd.read_csv(path, dtype={"sample_id": str})


def _replay(root: Path, slot: int, qrf_time):
    original, info = _load_handoff(root, slot, "evaluation")
    query = query_frame(original)
    ids = query.sample_id.tolist()
    base = _v10(slot)
    if base.sample_id.tolist() != ids:
        base = base.set_index("sample_id").loc[ids].reset_index()
    base["pred_tap_time_len"] = roundtrip_six(qrf_time)
    old, _ = _old_qrf(root, slot)
    result, audit = replay_v21(base, query, _history(slot), old["effective_neighbors"], cutoff=pd.Timestamp(info["cutoff"]))
    result["pred_tap_time_len"] = roundtrip_six(result.pred_tap_time_len)
    audit["gate_source"] = "recomputed_original_v015_QRF_effective_neighbors"
    audit["cached_test_id_list_used"] = False
    return result, audit


def _save_prediction(path: Path, value: pd.DataFrame):
    if path.exists():
        raise ContractError("never overwrite a v0.27 prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    value.to_csv(path, index=False)


def _predict_slot(root: Path, slot: int):
    ids = _load_handoff(root, slot, "evaluation")[0]["ids"].tolist()
    parent = _parent(slot)
    if parent.sample_id.tolist() != ids:
        parent = parent.set_index("sample_id").loc[ids].reset_index()
    with np.load(V26 / "worker_predictions" / "A" / f"{slot}.npz", allow_pickle=False) as saved_parent:
        if saved_parent["ids"].tolist() != ids:
            raise ContractError("V26A parent worker prediction IDs differ")
        parent_raw_time = saved_parent["median"]
    parent_replay, parent_audit = _replay(root, slot, parent_raw_time)
    if serialize_six_decimals(parent_replay) != serialize_six_decimals(parent):
        raise ContractError("v0.27 parent replay does not reproduce V26A")

    a_path = root / "worker_predictions" / "A" / f"{slot}.npz"
    worker("predict-a", "--root", root.resolve(), "--slot", slot, "--output", a_path.resolve())
    with np.load(a_path, allow_pickle=False) as prediction:
        if prediction["ids"].tolist() != ids:
            raise ContractError("v0.27 A prediction IDs differ")
        a_frame = isolate_iron(parent, prediction["median"])
    _save_prediction(root / "predictions" / f"{slot}_{CANDIDATE_A}.csv", a_frame)

    b_path = root / "worker_predictions" / "B" / f"{slot}.npz"
    worker("predict-b", "--root", root.resolve(), "--slot", slot, "--output", b_path.resolve())
    with np.load(b_path, allow_pickle=False) as prediction:
        if prediction["ids"].tolist() != ids:
            raise ContractError("v0.27 B prediction IDs differ")
        replayed, replay_audit = _replay(root, slot, prediction["median"])
        b_frame = isolate_time(parent, replayed.pred_tap_time_len.to_numpy(dtype=float))
    _save_prediction(root / "predictions" / f"{slot}_{CANDIDATE_B}.csv", b_frame)
    atomic_write_json(root / "predictions" / f"{slot}_audit.json", {
        "parent_replay": parent_audit,
        "A": {"unchanged_time_exact": True, "complete_iron_branch_replacement": True},
        "B": {"unchanged_iron_exact": True, "replay": replay_audit, "new_support_used_for_gate": False},
    })


def develop(root: Path):
    if (root / "development_complete.json").exists():
        raise ContractError("v0.27 development already complete")
    for slot in ORIGINS:
        _fit_a(root, slot)
    completed = _model_completion(root, ORIGINS)
    atomic_write_json(root / "development_models_complete.json", completed)
    for slot in ORIGINS:
        _predict_slot(root, slot)
    predictions = {
        f"{slot}_{candidate}": root / "predictions" / f"{slot}_{candidate}.csv"
        for slot in ORIGINS for candidate in CANDIDATES.values()
    }
    atomic_write_json(root / "development_predictions_frozen.json", {
        "status": "BOTH_CANDIDATES_FROZEN_BEFORE_SCORING",
        "predictions": file_identities(predictions),
        "platform_feedback_seen": False,
    })
    atomic_write_json(root / "development_complete.json", {
        "status": "PASS_G0_DEVELOPMENT_PREDICTIONS_FROZEN",
        "A_forest_fits": 6,
        "A_internal_trees": 1536,
        "B_forest_fits": 0,
        "B_weight_vectors": 6,
        "new_preprocessor_fits": 0,
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


def _parent_errors():
    source = pd.read_csv(V26 / "all_errors.csv", dtype={"sample_id": str})
    result = source.loc[source.candidate == PARENT_ID].copy()
    if result.empty:
        raise ContractError("V26A parent errors missing")
    result["candidate"] = PARENT
    return result


def _write_report(root: Path, deltas: pd.DataFrame):
    lines = [
        "# optimization-v0.27 自动生成离线报告",
        "",
        "主口径为 `macro_origin_mean_wmape`。负 Δ 表示优于 V26A_PARENT；G0 与 G1 分开登记。",
        "",
        "| 候选 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in CANDIDATES.values():
        part = deltas.loc[(deltas.candidate == candidate) & (deltas.reference == PARENT)].set_index("scope")
        values = [part.loc[scope, "delta_E"] for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")]
        lines.append("| " + candidate + " | " + " | ".join(f"{value:+.8f}" for value in values) + " |")
    lines.extend(["", "离线评价使用已消费标签，不取消两个预登记平台探索名额，也不改写为独立确认。", ""])
    with (root / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def score(root: Path):
    if (root / "offline_assessment.json").exists():
        raise ContractError("never overwrite v0.27 scoring evidence")
    candidates = {candidate: _candidate_errors(root, candidate) for candidate in CANDIDATES.values()}
    references = {"V1": _reference_errors("V1"), "V10": _reference_errors("V10"), "V21_REPLAY": _reference_errors("V21_REPLAY"), PARENT: _parent_errors()}
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
    for candidate in candidates:
        for reference in (PARENT, "V21_REPLAY", "V1"):
            for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT"):
                left, right = primary.loc[(candidate, scope)], primary.loc[(reference, scope)]
                rows.append({
                    "candidate": candidate,
                    "reference": reference,
                    "scope": scope,
                    "aggregation": PRIMARY_AGGREGATION,
                    "E": float(left.E),
                    "delta_E": float(left.E - right.E),
                    "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                    "delta_wmape_time": float(left.wmape_time - right.wmape_time),
                })
    deltas = pd.DataFrame(rows)
    deltas.to_csv(root / "canonical_deltas.csv", index=False)
    pooled_rows = []
    pooled_index = pooled.set_index(["algorithm", "scope"])
    for candidate in candidates:
        for scope in ("H1", "H2", "H3", "H4", "DEV_LONG", "DEV_SHORT"):
            left, right = pooled_index.loc[(candidate, scope)], pooled_index.loc[(PARENT, scope)]
            pooled_rows.append({
                "candidate": candidate,
                "reference": PARENT,
                "scope": scope,
                "aggregation": POOLED_AGGREGATION,
                "delta_E": float(left.E - right.E),
                "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                "delta_wmape_time": float(left.wmape_time - right.wmape_time),
            })
    pd.DataFrame(pooled_rows).to_csv(root / "exposure_pooled_deltas.csv", index=False)
    bootstrap = {candidate: {reference: week_intervals(all_errors, candidate, reference, 1000, 2026) for reference in (PARENT, "V21_REPLAY", "V1")} for candidate in candidates}
    atomic_write_json(root / "bootstrap.json", bootstrap)
    _write_report(root, deltas)
    atomic_write_json(root / "offline_assessment.json", {
        "status": "COMPLETED_CONSUMED_RETROSPECTIVE_NOT_A_PLATFORM_GATE",
        "primary_aggregation": PRIMARY_AGGREGATION,
        "diagnostic_aggregation": POOLED_AGGREGATION,
        "isolated_target_audit": isolation,
        "candidates": {
            candidate: {scope: float(primary.loc[(candidate, scope), "E"]) for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")}
            for candidate in candidates
        },
        "references": {
            reference: {scope: float(primary.loc[(reference, scope), "E"]) for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")}
            for reference in references
        },
        "both_candidates_retain_preregistered_platform_slot": True,
    })


def finalize(root: Path):
    if not (root / "offline_assessment.json").exists() or (root / "packages_frozen_before_feedback.json").exists():
        raise ContractError("score v0.27 once before finalization")
    _append_final_training_ledger(root)
    target = _write_target(root, 12, final_training=True)
    _fit_a(root, 12)
    completed = _model_completion(root, (12,))
    atomic_write_json(root / "final_models_complete.json", completed)
    _predict_slot(root, 12)
    parent_strings = pd.read_csv(PARENT_RESULT, dtype=str, keep_default_na=False)
    receipts = {}
    changed = {"A": "tap_iron", "B": "tap_time_len"}
    for key, candidate in CANDIDATES.items():
        prediction = pd.read_csv(root / "predictions" / f"12_{candidate}.csv", dtype={"sample_id": str})
        folder = root / "submissions" / candidate
        folder.mkdir(parents=True, exist_ok=False)
        csv_path = folder / "result.csv"
        values = prediction[f"pred_{changed[key]}"]
        payload = submission_bytes_preserving_other_target(parent_strings, prediction.sample_id.astype(str).tolist(), values, changed_target=changed[key])
        if payload == PARENT_RESULT.read_bytes():
            raise ContractError("v0.27 candidate is byte-identical to parent and cannot consume a platform slot")
        with csv_path.open("xb") as handle:
            handle.write(payload)
        archive = folder / "Luqhhh_bf_tap_predict_prelim.zip"
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as handle:
            handle.write(csv_path, arcname="result.csv")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != payload:
                raise ContractError("v0.27 submission ZIP payload differs")
        reread = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        if len(reread) != registration()["test_a_rows"] or reread.sample_id.duplicated().any():
            raise ContractError("v0.27 submission coverage differs")
        fixed = "pred_tap_time_len" if key == "A" else "pred_tap_iron"
        if reread[fixed].tolist() != parent_strings[fixed].tolist():
            raise ContractError("v0.27 unchanged target CSV strings differ from V26A")
        numeric = reread[PREDICTION_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ContractError("v0.27 submission predictions invalid")
        receipts[key] = {
            "candidate": candidate,
            "rows": len(reread),
            "result_sha256": file_sha256(csv_path),
            "zip_sha256": file_sha256(archive),
            "unchanged_target": fixed,
            "unchanged_target_csv_strings_exact": True,
            "parent_result_sha256": file_sha256(PARENT_RESULT),
            "platform_uploads": 0,
            "platform_feedback": None,
        }
    atomic_write_json(root / "packages_frozen_before_feedback.json", {
        "status": "PASS_TWO_COMPLETE_PACKAGES_FROZEN",
        "order": ["A", "B"],
        "receipts": receipts,
        "final_target_access": target,
        "definitions_may_change_after_A_feedback": False,
        "third_candidate_generated": False,
        "platform_upload_budget": 2,
        "platform_uploads": 0,
    })


def cold(root: Path):
    if (root / "completion.json").exists() or (root / "cold_validation.json").exists():
        raise ContractError("never overwrite v0.27 cold/completion evidence")
    package = read_json(root / "packages_frozen_before_feedback.json")
    receipts = {"A": {}, "B": {}, "old_gate": {}}
    all_zero = True
    for slot in SLOTS:
        for key, command in (("A", "cold-a"), ("B", "cold-b")):
            output = root / "cold" / key / f"{slot}.npz"
            worker(command, "--root", root.resolve(), "--slot", slot, "--output", output.resolve())
            info = read_json(str(output) + ".json")
            receipts[key][str(slot)] = info
            all_zero &= not any(info["zero_fit"].values())
        old_output = root / "cold" / "old_gate" / f"{slot}.npz"
        v26_worker("old-cold", "--root", V26.resolve(), "--slot", slot, "--output", old_output.resolve())
        info = read_json(str(old_output) + ".json")
        receipts["old_gate"][str(slot)] = info
        all_zero &= not any(info["zero_fit"].values())
        with np.load(root / "old_qrf_predictions" / f"{slot}.npz", allow_pickle=False) as current, np.load(V26 / "old_qrf_predictions" / f"{slot}.npz", allow_pickle=False) as parent:
            for name in ("ids", "median", "mean", "effective_neighbors"):
                if not np.array_equal(current[name], parent[name]):
                    raise ContractError("v0.27 fresh old-gate support differs from V26 certified recomputation")
    if not all_zero:
        raise ContractError("v0.27 cold audit attempted a fit")
    parent_strings = pd.read_csv(PARENT_RESULT, dtype=str, keep_default_na=False)
    for key, candidate in CANDIDATES.items():
        receipt = package["receipts"][key]
        folder = root / "submissions" / candidate
        archive = folder / "Luqhhh_bf_tap_predict_prelim.zip"
        if file_sha256(archive) != receipt["zip_sha256"]:
            raise ContractError("v0.27 frozen ZIP identity changed")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != (folder / "result.csv").read_bytes():
                raise ContractError("v0.27 cold ZIP reread differs")
        reread = pd.read_csv(folder / "result.csv", dtype=str, keep_default_na=False)
        fixed = "pred_tap_time_len" if key == "A" else "pred_tap_iron"
        if reread[fixed].tolist() != parent_strings[fixed].tolist():
            raise ContractError("v0.27 cold unchanged target string identity differs")
    attempted = len(list((root / "models" / "A").glob("*/fit_intent.json")))
    completed = len(list((root / "models" / "A").glob("*/fit_record.json")))
    if attempted != 7 or completed != 7:
        raise ContractError("v0.27 A fit budget/completion count differs")
    fit_records = [read_json(path) for path in (root / "models" / "A").glob("*/fit_record.json")]
    trees = sum(int(record["internal_trees"]) for record in fit_records)
    weights = list((root / "weights").glob("*/recency60.npz.json"))
    if trees != 1792 or len(weights) != 7:
        raise ContractError("v0.27 tree/weight attachment budget differs")
    atomic_write_json(root / "cold_validation.json", {
        "status": "PASS",
        "all_slots": list(SLOTS),
        "candidate_checks": "full_reverse_chunk_subset_single_exact",
        "old_v015_gate_recomputed_all_slots": True,
        "fit_attempts_during_cold": {
            "RandomForestRegressor": 0,
            "ExtraTreesRegressor": 0,
            "IronTargetForest": 0,
            "PartitionForest": 0,
            "parent_QRF": 0,
            "preprocessor": 0,
            "CatBoost": 0,
            "calibration": 0,
        },
        "B_parent_tree_leaf_response_identity_frozen": True,
        "B_all_ones_parent_distribution_check": True,
        "B_tree_local_normalization": True,
        "trusted_private_bundles_only": True,
        "zip_and_csv_reread": True,
        "receipts": receipts,
    })
    atomic_write_json(root / "fit_counts.json", {
        "A_forest_attempted": attempted,
        "A_forest_completed": completed,
        "A_internal_trees": trees,
        "B_forest_attempted": 0,
        "B_forest_completed": 0,
        "B_weight_vectors": len(weights),
        "new_preprocessor_attempted": 0,
        "new_preprocessor_completed": 0,
        "CatBoost_E04_rate_q_fits": 0,
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
