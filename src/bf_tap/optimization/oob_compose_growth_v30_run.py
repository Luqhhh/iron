"""Registered v0.30 lifecycle: exact dual-target composition and fixed time-forest growth."""
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
from .oob_compose_v30 import (
    APPENDED_FOREST_PROTOCOL,
    CANDIDATE_A,
    CANDIDATE_B,
    IRON_SOURCE,
    OOB_ATTACHMENT_PROTOCOL,
    PARENT,
    POOLED_AGGREGATION,
    PRIMARY_AGGREGATION,
    TARGETS,
    TIME_SOURCE,
    V1,
    V21_REPLAY,
    V26A_PARENT,
    compose_1024_time,
    compose_both_targets,
    exposure_pooled_summary,
    macro_origin_summary,
    verify_composition_identities,
    verify_isolated_deltas,
)
from .qrf_partition_v26 import roundtrip_six
from .qrf_partition_v26_run import _replay as replay_qrf, _score_rows
from .structural_run import append_ledger


V26 = Path("local/runs/optimization-v0.26-qrf-partition-tests-r1")
V27 = Path("local/runs/optimization-v0.27-qrf-iron-and-leaf-recency-r1")
V28 = Path("local/runs/optimization-v0.28-fixed-equal-blends-r3")
V29 = Path("local/runs/optimization-v0.29-oob-leaf-responses-r1")
V30_WORKER = Path("workers/qrf_v030/worker.py")
V29_WORKER = Path("workers/qrf_v029/worker.py")
V26_WORKER = Path("workers/qrf_v026/worker.py")
V27_WORKER = Path("workers/qrf_v027/worker.py")
WORKER_PYTHON = Path("workers/qrf_v015/.venv/bin/python")
ORIGINS = range(6, 12)
SLOTS = range(6, 13)
CANDIDATES = {"A": CANDIDATE_A, "B": CANDIDATE_B}
V28_SOURCE_ID = "V28I_CB_QRF_EQUAL_BLEND"
V26_SOURCE_ID = "V26A_QRF_ABSOLUTE_SPLIT_TIME"
V29T_ID = TIME_SOURCE
V29I_ID = IRON_SOURCE
V28_RESULT = V28 / "submissions" / V28_SOURCE_ID / "result.csv"
V28_ZIP = V28 / "submissions" / V28_SOURCE_ID / "Luqhhh_bf_tap_predict_prelim.zip"
V29I_RESULT = V29 / "submissions" / V29I_ID / "result.csv"
V29I_ZIP = V29 / "submissions" / V29I_ID / "Luqhhh_bf_tap_predict_prelim.zip"
V29T_RESULT = V29 / "submissions" / V29T_ID / "result.csv"
V29T_ZIP = V29 / "submissions" / V29T_ID / "Luqhhh_bf_tap_predict_prelim.zip"
V26_RESULT = V26 / "submissions" / V26_SOURCE_ID / "result.csv"
V21_RESULT = V23 / "recovery/result.csv"


def registration():
    value = load_yaml("configs/optimization_v0_30/experiment.yaml")
    if value["branch"] != "optimization-v0.30-oob-compose-and-time-growth" or value["candidates"] != CANDIDATES:
        raise ContractError("v0.30 branch/candidate registration differs")
    if value["protocol"] != "QRF_OOB_COMPOSE_AND_TIME_GROWTH_v030":
        raise ContractError("v0.30 composition protocol differs")
    if value["appended_forest_protocol"] != APPENDED_FOREST_PROTOCOL:
        raise ContractError("v0.30 appended-forest protocol differs")
    if value["oob_attachment_protocol"] != OOB_ATTACHMENT_PROTOCOL:
        raise ContractError("v0.30 OOB attachment protocol differs")
    expected_budget = {
        "A_new_model_or_calibration_fits": 0,
        "B_forest_append_fits_development": 6,
        "B_forest_append_fits_final": 1,
        "B_forest_append_fits_total": 7,
        "B_new_trees_development": 4608,
        "B_new_trees_final": 768,
        "B_new_trees_total": 5376,
        "B_reused_parent_trees_development": 1536,
        "B_reused_parent_trees_final": 256,
        "B_reused_parent_trees_total": 1792,
        "B_models_held_trees_development": 6144,
        "B_models_held_trees_final": 1024,
        "B_models_held_trees_total": 7168,
        "new_1024_oob_attachments_development": 6,
        "new_1024_oob_attachments_final": 1,
        "new_1024_oob_attachments_total": 7,
        "catboost_iron_forest_rate_q_LAD_fits": 0,
        "preprocessor_fits": 0,
        "new_candidate_packages": 2,
        "new_candidate_platform_tests": 2,
        "recovery_uploads": 0,
        "third_candidate": 0,
    }
    if value["budget"] != expected_budget or value["platform_order"] != ["A", "B"]:
        raise ContractError("v0.30 budget/order registration differs")
    if value["platform_feedback_may_change_second_candidate"] is not False:
        raise ContractError("v0.30 second candidate must stay frozen across A feedback")
    definition = value["candidate_B"]
    if definition["parent_trees"] != 256 or definition["total_trees"] != 1024 or definition["new_trees"] != 768:
        raise ContractError("v0.30 append size registration differs")
    if definition["shrinkage"] != 0.25 or definition["gate"] != "spout_1_and_effective_neighbors_lt_500":
        raise ContractError("v0.30 time post-processing registration differs")
    return value


def _source_files():
    return [
        Path("configs/optimization_v0_30/experiment.yaml"),
        Path("configs/optimization_v0_30/access_scope.yaml"),
        Path("src/bf_tap/optimization/oob_compose_v30.py"),
        Path("src/bf_tap/optimization/oob_compose_growth_v30_run.py"),
        Path("scripts/optimization_v30_oob_compose_growth.py"),
        Path("scripts/optimization_v30_cold_check.py"),
        Path("tests/test_optimization_v30_oob_compose_growth.py"),
        Path("docs/optimization_v0_30/PLAN.md"),
        V30_WORKER,
        Path("workers/qrf_v030/append_forest.py"),
        Path("workers/qrf_v030/oob_append.py"),
        Path("workers/qrf_v030/test_append_forest.py"),
        Path("workers/qrf_v030/test_oob_append.py"),
        V29_WORKER,
        Path("workers/qrf_v029/oob_response.py"),
        V26_WORKER,
        Path("workers/qrf_v026/partition_forest.py"),
        V27_WORKER,
        Path("workers/qrf_v027/target_forest.py"),
        Path("workers/qrf_v015/qrf_model.py"),
        Path("workers/qrf_v015/preprocessing.py"),
        Path("workers/qrf_v015/uv.lock"),
    ]


def _source_path(slot: int, role: str) -> Path:
    if role == "v29i":
        return V29I_RESULT if slot == 12 else V29 / "predictions" / f"{slot}_{V29I_ID}.csv"
    if role == "v29t":
        return V29T_RESULT if slot == 12 else V29 / "predictions" / f"{slot}_{V29T_ID}.csv"
    if role == "v28i":
        return V28_RESULT if slot == 12 else V28 / "predictions" / f"{slot}_{V28_SOURCE_ID}.csv"
    if role == "v26a":
        return V26_RESULT if slot == 12 else V26 / "predictions" / f"{slot}_{V26_SOURCE_ID}.csv"
    raise ContractError("unknown v0.30 endpoint role")


def _load_endpoint(slot: int, role: str) -> pd.DataFrame:
    path = _source_path(slot, role)
    if not path.is_file():
        raise ContractError(f"missing frozen {role} endpoint for slot {slot}")
    return validate_endpoint(pd.read_csv(path, dtype=str, keep_default_na=False), label=f"{role}/{slot}")


def _worker(script: Path, command: str, *arguments):
    return subprocess.check_output([str(WORKER_PYTHON), str(script), command, *map(str, arguments)], text=True)


def _registry_search():
    manifests = sorted(Path("local/runs").glob("**/*manifest*.json"))
    matches = []
    for path in manifests:
        text = path.read_text(encoding="utf-8", errors="ignore")
        reasons = [item for item in (*CANDIDATES.values(), registration()["protocol"], APPENDED_FOREST_PROTOCOL) if item in text]
        if reasons:
            matches.append({"path": str(path), "sha256": file_sha256(path), "matches": reasons})
    completed, incomplete = [], []
    for match in matches:
        root = Path(match["path"]).parent
        (completed if (root / "completion.json").is_file() else incomplete).append(match)
    return {"searched_manifest_count": len(manifests), "equivalent_matches": completed, "incomplete_equivalent_attempts": incomplete}


def register(root: Path):
    config = registration()
    scope = load_yaml("configs/optimization_v0_30/access_scope.yaml")
    if root.exists():
        raise ContractError("never overwrite a v0.30 run")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ContractError("clean preregistered working tree required")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != config["branch"]:
        raise ContractError("registered v0.30 branch required")
    subprocess.run(["git", "merge-base", "--is-ancestor", config["base_commit"], "HEAD"], check=True)
    registry = _registry_search()
    if registry["equivalent_matches"]:
        raise ContractError("equivalent completed v0.30 experiment already exists")
    identities = config["source_identities"]
    for key, path in {
        "v29i_result_sha256": V29I_RESULT, "v29i_zip_sha256": V29I_ZIP,
        "v29t_result_sha256": V29T_RESULT, "v29t_zip_sha256": V29T_ZIP,
        "v28i_result_sha256": V28_RESULT, "v28i_zip_sha256": V28_ZIP,
        "v26a_result_sha256": V26_RESULT, "v21_result_sha256": V21_RESULT,
    }.items():
        if file_sha256(path) != identities[key]:
            raise ContractError(f"registered v0.30 source identity differs: {key}")
    evidence = {
        "v29_manifest": V29 / "manifest.json", "v29_completion": V29 / "completion.json",
        "v29_cold_validation": V29 / "cold_validation.json",
        "v29_packages_frozen": V29 / "packages_frozen_before_feedback.json",
        "v29_development_frozen": V29 / "development_predictions_frozen.json",
        "v29i_result": V29I_RESULT, "v29i_zip": V29I_ZIP, "v29t_result": V29T_RESULT, "v29t_zip": V29T_ZIP,
        "v28_manifest": V28 / "manifest.json", "v28_completion": V28 / "completion.json",
        "v28_errors": V28 / "all_errors.csv", "v28_parent_result": V28_RESULT, "v28_parent_zip": V28_ZIP,
        "v27_manifest": V27 / "manifest.json", "v27_completion": V27 / "completion.json",
        "v26_manifest": V26 / "manifest.json", "v26_completion": V26 / "completion.json",
        "v26_errors": V26 / "all_errors.csv", "v26a_result": V26_RESULT,
        "v26_models_complete_development": V26 / "development_models_complete.json",
        "v26_models_complete_final": V26 / "final_models_complete.json",
        "v22_errors": V22 / "all_errors.csv", "v23_v21_result": V21_RESULT,
        "protection": Path(scope["protection_contract"]),
    }
    for slot in SLOTS:
        evidence[f"v29_attachment_B_{slot}"] = V29 / "oob_attachments" / "B" / f"{slot}.npz"
        evidence[f"v29_attachment_B_{slot}_metadata"] = V29 / "oob_attachments" / "B" / f"{slot}.npz.json"
        evidence[f"v29_prediction_B_{slot}"] = V29 / "worker_predictions" / "B" / f"{slot}.npz"
        evidence[f"v29_prediction_B_{slot}_metadata"] = V29 / "worker_predictions" / "B" / f"{slot}.npz.json"
    missing = [name for name, path in evidence.items() if not Path(path).is_file()]
    if missing:
        raise ContractError(f"missing registered v0.30 evidence: {missing}")
    worker_environment = json.loads(_worker(V30_WORKER, "environment"))
    worker_sources = json.loads(_worker(V30_WORKER, "identity"))
    append_registration = json.loads(_worker(V30_WORKER, "registration"))
    if append_registration["protocol"] != APPENDED_FOREST_PROTOCOL:
        raise ContractError("worker append protocol differs from the registration")
    manifest = {
        "registration": config, "scope": scope, "protection": load_yaml(scope["protection_contract"]),
        "evidence": file_identities(evidence),
        "sources": file_identities({str(path): path for path in _source_files()}),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_environment(), "worker_environment": worker_environment, "worker_sources": worker_sources,
        "append_registration": append_registration,
        "appended_forest_protocol": APPENDED_FOREST_PROTOCOL,
        "oob_attachment_protocol": OOB_ATTACHMENT_PROTOCOL,
        "oob_core_protocol": config["oob_core_protocol"],
        "candidate_targets": {"B": {"target": "tap_time_len", "unit": "minutes", "source": V26_SOURCE_ID}},
        "source_runs": {"v26": str(V26.resolve()), "v29": str(V29.resolve())},
        "registry_search": registry, "holdout_consumed": True, "test_targets_read": False,
        "new_append_fit_budget": 7, "platform_feedback_may_change_second_candidate": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    atomic_write_json(root / "manifest.json", manifest)
    append_ledger(scope, root)
    atomic_write_json(root / "registration.json", {
        "status": "TWO_CANDIDATES_FROZEN_BEFORE_APPEND_AND_DERIVATION",
        "candidates": CANDIDATES,
        "definitions": {key: config[f"candidate_{key}"] for key in CANDIDATES},
        "source_endpoints": {
            "v29i_result_sha256": file_sha256(V29I_RESULT), "v29t_result_sha256": file_sha256(V29T_RESULT),
            "v28i_result_sha256": file_sha256(V28_RESULT),
        },
        "append_registration": append_registration,
        "platform_order": ["A", "B"], "candidate_platform_budget": 2, "platform_uploads": 0,
        "definitions_may_change_after_A_feedback": False, "third_candidate_allowed": False,
    })
    return manifest


def prepare(root: Path):
    if (root / "p0_complete.json").exists():
        raise ContractError("v0.30 P0 already complete")
    audits, replays = {}, {}
    for slot in SLOTS:
        v29i, v29t = _load_endpoint(slot, "v29i"), _load_endpoint(slot, "v29t")
        v28i, v26a = _load_endpoint(slot, "v28i"), _load_endpoint(slot, "v26a")
        if not (v29i.sample_id.tolist() == v29t.sample_id.tolist() == v28i.sample_id.tolist() == v26a.sample_id.tolist()):
            raise ContractError("v0.30 source endpoint sample IDs/order differ")
        if v29i.pred_tap_time_len.tolist() != v28i.pred_tap_time_len.tolist():
            raise ContractError("v0.29 iron candidate changed the V28I time endpoint")
        if v29t.pred_tap_iron.tolist() != v28i.pred_tap_iron.tolist():
            raise ContractError("v0.29 time candidate changed the V28I iron endpoint")
        if v26a.pred_tap_time_len.tolist() != v28i.pred_tap_time_len.tolist():
            raise ContractError("V26A complete time endpoint differs from V28I")
        with np.load(V29 / "worker_predictions" / "B" / f"{slot}.npz", allow_pickle=False) as saved:
            raw_time = saved["median"]
        replayed, replay_audit = replay_qrf(V26, slot, raw_time)
        expected = roundtrip_six(v29t.pred_tap_time_len.to_numpy(dtype=float))
        if not np.array_equal(replayed.pred_tap_time_len.to_numpy(dtype=float), expected):
            raise ContractError("v0.29 time source replay does not reproduce the packaged V29T endpoint")
        audits[str(slot)] = json.loads(_worker(
            V30_WORKER, "audit", "--root", root.resolve(), "--slot", slot,
        ))
        if any(audits[str(slot)]["zero_fit"].values()):
            raise ContractError("P0 audit attempted a fit")
        replays[str(slot)] = {"replay": replay_audit, "v29_time_replay_exact": True}
    atomic_write_json(root / "p0_complete.json", {
        "status": "PASS_ZERO_FIT_V29_SOURCE_REPLAY_AND_V26A_PREFIX_AUDIT",
        "audits": audits, "v29_time_replays": replays,
        "registry_search": read_json(root / "manifest.json")["registry_search"],
        "new_model_fits": 0, "new_preprocessor_fits": 0, "new_calibration_fits": 0,
        "new_append_fits": 0, "test_targets_read": False,
    })


def _write_prediction(path: Path, value: pd.DataFrame):
    if path.exists():
        raise ContractError("never overwrite a v0.30 prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(submission_bytes(value))


def _compose_slot(root: Path, slot: int, *, derive: bool):
    v29i, v29t = _load_endpoint(slot, "v29i"), _load_endpoint(slot, "v29t")
    a = compose_both_targets(v29i, v29t)
    raw_output = root / "worker_predictions" / "B" / f"{slot}.npz"
    regression_output = root / "worker_predictions" / "regression_256" / f"{slot}.npz"
    if derive:
        if not (root / "models" / "B" / str(slot) / "fit_record.json").is_file():
            _worker(V30_WORKER, "append-fit", "--root", root.resolve(), "--slot", slot)
        _worker(V30_WORKER, "derive-predict", "--root", root.resolve(), "--slot", slot, "--output", raw_output.resolve())
    with np.load(raw_output, allow_pickle=False) as source:
        if source["ids"].tolist() != a.sample_id.tolist():
            raise ContractError("v0.30 1024-tree worker prediction IDs differ")
        raw_median = source["median"]
    with np.load(regression_output, allow_pickle=False) as source:
        if source["ids"].tolist() != a.sample_id.tolist():
            raise ContractError("v0.30 restricted-256 worker prediction IDs differ")
        restricted_median = source["median"]
    replayed, replay_audit = replay_qrf(V26, slot, raw_median)
    b = compose_1024_time(a, replayed.pred_tap_time_len.to_numpy(dtype=float))
    restricted_replay, restricted_audit = replay_qrf(V26, slot, restricted_median)
    expected_v29t = roundtrip_six(v29t.pred_tap_time_len.to_numpy(dtype=float))
    if not np.array_equal(restricted_replay.pred_tap_time_len.to_numpy(dtype=float), expected_v29t):
        raise ContractError("v0.30 restricted-256 time does not reproduce the V29T endpoint")
    _write_prediction(root / "predictions" / f"{slot}_{CANDIDATE_A}.csv", a)
    _write_prediction(root / "predictions" / f"{slot}_{CANDIDATE_B}.csv", b)
    atomic_write_json(root / "predictions" / f"{slot}_audit.json", {
        "A": {
            "iron_source": IRON_SOURCE, "time_source": TIME_SOURCE,
            "iron_strings_exact": a.pred_tap_iron.tolist() == v29i.pred_tap_iron.tolist(),
            "time_strings_exact": a.pred_tap_time_len.tolist() == v29t.pred_tap_time_len.tolist(),
            "rowwise_average_used": False, "rescoring_used": False, "regating_used": False,
        },
        "B": {
            "source_forest": V26_SOURCE_ID, "trees": 1024, "new_trees": 768,
            "raw_oob_endpoint": "V26A_TIME_APPENDED_1024_OOB",
            "replay": replay_audit, "restricted_256_replay": restricted_audit,
            "restricted_256_time_reproduces_V29T": True,
            "unchanged_iron_exact": b.pred_tap_iron.tolist() == a.pred_tap_iron.tolist(),
            "gate_source": "recomputed_original_v015_effective_neighbors",
            "gate_uses_new_1024_support": False,
        },
    })


def develop(root: Path):
    if (root / "development_complete.json").exists():
        raise ContractError("v0.30 development already complete")
    for slot in ORIGINS:
        _compose_slot(root, slot, derive=True)
    ledger = json.loads(_worker(V30_WORKER, "fit-ledger", "--root", root.resolve()))
    if ledger["fit_completed"] != 6 or ledger["new_trees_completed"] != 4608 or ledger["slots"] != list(ORIGINS):
        raise ContractError("v0.30 development append-fit ledger differs from the registration")
    attachment_files = {
        f"B_{slot}": root / "oob_attachments" / "B" / f"{slot}.npz" for slot in ORIGINS
    }
    predictions = {
        f"{slot}_{candidate}": root / "predictions" / f"{slot}_{candidate}.csv"
        for slot in ORIGINS for candidate in CANDIDATES.values()
    }
    atomic_write_json(root / "development_predictions_frozen.json", {
        "status": "BOTH_CANDIDATES_AND_SEVEN_ATTACHMENTS_FROZEN_BEFORE_SCORING",
        "predictions": file_identities(predictions), "attachments": file_identities(attachment_files),
        "platform_feedback_seen": False, "fit_ledger": ledger,
    })
    atomic_write_json(root / "development_complete.json", {
        "status": "PASS_G0_DEVELOPMENT_APPEND_AND_COMPOSITION_FROZEN",
        "warm_start_append_fits": ledger["fit_completed"], "new_trees": ledger["new_trees_completed"],
        "reused_parent_trees": ledger["reused_parent_trees"], "oob_attachments": 6,
        "new_preprocessor_fits": 0, "new_calibration_fits": 0,
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
    for slot in SLOTS:
        attachment = read_json(str(root / "oob_attachments" / "B" / f"{slot}.npz") + ".json")
        receipt = read_json(str(root / "worker_predictions" / "B" / f"{slot}.npz") + ".json")
        with np.load(root / "worker_predictions" / "B" / f"{slot}.npz", allow_pickle=False) as current:
            median = current["median"]
        with np.load(root / "worker_predictions" / "regression_256" / f"{slot}.npz", allow_pickle=False) as restricted:
            restricted_median = restricted["median"]
        support = float(median.max() - median.min())
        delta = np.abs(median - restricted_median)
        rows.append({
            "candidate": CANDIDATE_B, "slot": slot, "target": receipt["target"],
            "training_rows": attachment["training_rows"], "trees": attachment["trees"],
            "leaf_count": attachment["leaf_count"], "fallback_leaf_count": attachment["fallback_leaf_count"],
            "fallback_leaf_fraction": attachment["fallback_leaf_fraction"],
            "minimum_oob_count": attachment["minimum_oob_count"], "maximum_oob_count": attachment["maximum_oob_count"],
            "minimum_selected_count": attachment["minimum_selected_count"],
            "maximum_selected_count": attachment["maximum_selected_count"],
            "query_fallback_trees_mean": receipt["fallback_query_tree_mean"],
            "query_fallback_trees_max": receipt["fallback_query_tree_max"],
            "v29_prefix_exact": receipt["v29_prefix_exact"]["prefix_leaf_members_exact"],
            "restricted_256_time_reproduces_V29T": True,
            "raw_256_to_1024_mean_abs_delta": float(delta.mean()),
            "raw_256_to_1024_max_abs_delta": float(delta.max()),
            "raw_delta_normalized_by_1024_support": float(delta.mean() / support) if support else 0.0,
            "load_seconds": receipt["load_seconds"], "attachment_seconds": receipt["attachment_seconds"],
            "prediction_seconds": receipt["prediction_seconds"], "model_bytes": receipt["model_bytes"],
            "peak_memory_kib": receipt["peak_memory_kib"],
        })
    return pd.DataFrame(rows)


def _write_report(root: Path, deltas: pd.DataFrame):
    lines = [
        "# optimization-v0.30 自动生成离线报告", "",
        "主口径为 `macro_origin_mean_wmape`。负 Δ 表示优于参照；OOB 是叶响应估计，不是独立验证集。", "",
        "| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in CANDIDATES.values():
        for reference in (PARENT, IRON_SOURCE, TIME_SOURCE, V26A_PARENT, V21_REPLAY, V1):
            part = deltas.loc[(deltas.candidate == candidate) & (deltas.reference == reference)].set_index("scope")
            values = [part.loc[scope, "delta_E"] for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")]
            lines.append(f"| {candidate} | {reference} | " + " | ".join(f"{value:+.8f}" for value in values) + " |")
    lines.extend(["", "G0、已消费历史 G1 与平台反馈分别登记；B 的扩容主对照是本轮 A。", ""])
    with (root / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def score(root: Path):
    if (root / "offline_assessment.json").exists():
        raise ContractError("never overwrite v0.30 scoring evidence")
    candidates = {candidate: _candidate_errors(root, candidate) for candidate in CANDIDATES.values()}
    references = {
        "V1": _reference_errors("V1"),
        V21_REPLAY: _reference_errors(V21_REPLAY),
        V26A_PARENT: _source_errors(V26 / "all_errors.csv", V26_SOURCE_ID, V26A_PARENT),
        PARENT: _source_errors(V28 / "all_errors.csv", V28_SOURCE_ID, PARENT),
        IRON_SOURCE: _source_errors(V29 / "all_errors.csv", IRON_SOURCE, IRON_SOURCE),
        TIME_SOURCE: _source_errors(V29 / "all_errors.csv", TIME_SOURCE, TIME_SOURCE),
    }
    all_errors = pd.concat([*references.values(), *candidates.values()], ignore_index=True)
    all_errors.to_csv(root / "all_errors.csv", index=False)
    scorecard = pd.DataFrame([row for name, errors in {**references, **candidates}.items() for row in _score_rows(errors, name)])
    scorecard.to_csv(root / "canonical_scorecard.csv", index=False)
    summary = macro_origin_summary(scorecard)
    summary.to_csv(root / "canonical_summary.csv", index=False)
    pooled = exposure_pooled_summary(scorecard)
    pooled.to_csv(root / "exposure_pooled_summary.csv", index=False)
    composition = verify_composition_identities(scorecard, summary)
    isolation = {
        "A_vs_V29T": verify_isolated_deltas(scorecard, summary, {CANDIDATE_A: "tap_iron"}, parent=TIME_SOURCE),
        "B_vs_A": verify_isolated_deltas(scorecard, summary, {CANDIDATE_B: "tap_time_len"}, parent=CANDIDATE_A),
    }
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
        "composition_identity_audit": composition, "isolated_target_audit": isolation,
        "OOB_is_not_an_independent_validation_set": True,
        "B_primary_comparison_is_A_not_V29T": True,
        "both_candidates_retain_preregistered_platform_slot": True,
    })


def finalize(root: Path):
    if not (root / "offline_assessment.json").exists() or (root / "packages_frozen_before_feedback.json").exists():
        raise ContractError("score v0.30 once before finalization")
    _compose_slot(root, 12, derive=True)
    ledger = json.loads(_worker(V30_WORKER, "fit-ledger", "--root", root.resolve()))
    if ledger["fit_completed"] != 7 or ledger["new_trees_completed"] != 5376 or ledger["reused_parent_trees"] != 1792:
        raise ContractError("v0.30 final append-fit ledger differs from the registration")
    receipts = {}
    for key, candidate in CANDIDATES.items():
        prediction = validate_endpoint(
            pd.read_csv(root / "predictions" / f"12_{candidate}.csv", dtype=str, keep_default_na=False), label=candidate,
        )
        if key == "A":
            v29i, v29t = _load_endpoint(12, "v29i"), _load_endpoint(12, "v29t")
            if prediction.pred_tap_iron.tolist() != v29i.pred_tap_iron.tolist():
                raise ContractError("v0.30 A final iron strings differ from V29I")
            if prediction.pred_tap_time_len.tolist() != v29t.pred_tap_time_len.tolist():
                raise ContractError("v0.30 A final time strings differ from V29T")
        else:
            a_frame = validate_endpoint(
                pd.read_csv(root / "predictions" / f"12_{CANDIDATE_A}.csv", dtype=str, keep_default_na=False), label=CANDIDATE_A,
            )
            if prediction.pred_tap_iron.tolist() != a_frame.pred_tap_iron.tolist():
                raise ContractError("v0.30 B final iron strings differ from A")
            if prediction.pred_tap_time_len.tolist() == a_frame.pred_tap_time_len.tolist():
                raise ContractError("v0.30 B final degenerates to A")
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
                raise ContractError("v0.30 ZIP payload differs")
        reread = validate_endpoint(pd.read_csv(csv_path, dtype=str, keep_default_na=False), label="package")
        if len(reread) != registration()["test_a_rows"]:
            raise ContractError("v0.30 package test_a coverage differs")
        receipts[key] = {
            "candidate": candidate, "rows": len(reread), "result_sha256": file_sha256(csv_path),
            "zip_sha256": file_sha256(archive),
            "unchanged_target": "exact_source_columns" if key == "A" else "A_iron_exact",
            "platform_uploads": 0, "platform_feedback": None,
        }
    atomic_write_json(root / "packages_frozen_before_feedback.json", {
        "status": "PASS_TWO_COMPLETE_PACKAGES_FROZEN", "order": ["A", "B"], "receipts": receipts,
        "fit_ledger": ledger, "definitions_may_change_after_A_feedback": False,
        "third_candidate_generated": False, "platform_upload_budget": 2, "platform_uploads": 0,
    })


def _assert_frame_equal(actual, expected, message):
    left = validate_endpoint(actual, label="actual").reset_index(drop=True)
    right = validate_endpoint(expected, label="expected").reset_index(drop=True)
    if not left.equals(right):
        raise ContractError(message)


def cold(root: Path):
    if (root / "cold_validation.json").exists() or (root / "completion.json").exists():
        raise ContractError("never overwrite v0.30 cold/completion evidence")
    receipts = {}
    zero_fit = True
    for slot in SLOTS:
        output = root / "cold" / "appended_B" / f"{slot}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        _worker(V30_WORKER, "cold", "--root", root.resolve(), "--slot", slot, "--output", output.resolve())
        info = read_json(str(output) + ".json")
        receipts[str(slot)] = info
        zero_fit &= not any(info["zero_fit"].values())
        v29i, v29t = _load_endpoint(slot, "v29i"), _load_endpoint(slot, "v29t")
        a = compose_both_targets(v29i, v29t)
        with np.load(root / "worker_predictions" / "B" / f"{slot}.npz", allow_pickle=False) as source:
            raw_median = source["median"]
        replayed, _ = replay_qrf(V26, slot, raw_median)
        b = compose_1024_time(a, replayed.pred_tap_time_len.to_numpy(dtype=float))
        _assert_frame_equal(a, pd.read_csv(root / "predictions" / f"{slot}_{CANDIDATE_A}.csv", dtype=str, keep_default_na=False), "cold A differs")
        _assert_frame_equal(b, pd.read_csv(root / "predictions" / f"{slot}_{CANDIDATE_B}.csv", dtype=str, keep_default_na=False), "cold B differs")
    if not zero_fit:
        raise ContractError("v0.30 cold restoration attempted a fit")
    ledger = json.loads(_worker(V30_WORKER, "fit-ledger", "--root", root.resolve()))
    if ledger["fit_attempted"] != 7 or ledger["fit_completed"] != 7 or ledger["new_trees_completed"] != 5376:
        raise ContractError("v0.30 cold fit ledger differs from the registered budget")
    package = read_json(root / "packages_frozen_before_feedback.json")
    for key, candidate in CANDIDATES.items():
        folder = root / "submissions" / candidate
        csv_path, archive = folder / "result.csv", folder / "Luqhhh_bf_tap_predict_prelim.zip"
        receipt = package["receipts"][key]
        if file_sha256(csv_path) != receipt["result_sha256"] or file_sha256(archive) != receipt["zip_sha256"]:
            raise ContractError("v0.30 frozen package identity changed")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != csv_path.read_bytes():
                raise ContractError("v0.30 cold ZIP reread differs")
    fit_counts = {
        "new_base_model_attempted": 0, "new_base_model_completed": 0,
        "warm_start_append_fit_attempted": ledger["fit_attempted"], "warm_start_append_fit_completed": ledger["fit_completed"],
        "new_trees_attempted": ledger["new_trees_attempted"], "new_trees_completed": ledger["new_trees_completed"],
        "reused_parent_trees": ledger["reused_parent_trees"], "trees_held_total": ledger["trees_held"],
        "new_preprocessor_attempted": 0, "new_preprocessor_completed": 0,
        "new_calibration_attempted": 0, "new_calibration_completed": 0,
        "new_1024_oob_attachment_attempted": len(list(SLOTS)), "new_1024_oob_attachment_completed": len(list(SLOTS)),
        "catboost_iron_forest_rate_q_LAD_attempted": 0,
        "new_candidate_packages": 2, "third_candidate_generated": False,
    }
    atomic_write_json(root / "fit_counts.json", fit_counts)
    atomic_write_json(root / "cold_validation.json", {
        "status": "PASS", "all_slots": list(SLOTS),
        "appended_forest_and_attachment_rederived_exact": True,
        "restricted_256_oob_and_full_leaf_exact": True,
        "source_checks": "full_reverse_chunk_subset_single_exact", "oob_checks": "full_reverse_chunk_subset_single_exact",
        "fit_attempts_during_cold": {
            "append_forest": 0, "RandomForestRegressor": 0, "ExtraTreesRegressor": 0,
            "PartitionForest": 0, "old_QRF": 0, "preprocessor": 0, "CatBoost": 0, "LAD_or_calibration": 0,
        },
        "trusted_private_bundles_only": True, "source_files_and_tree_structures_unchanged": True,
        "zip_and_csv_reread": True, "receipts": receipts,
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
