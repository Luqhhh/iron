"""Registered zero-fit v0.31 lifecycle for occurrence-pooled OOB responses."""
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
from .fixed_blend_v28 import equal_blend_six, submission_bytes, validate_endpoint
from .oob_pool_v31 import (
    ATTACHMENT_PROTOCOL,
    CANDIDATE_A,
    CANDIDATE_B,
    PARENT,
    POOLED_AGGREGATION,
    POOLED_PROTOCOL,
    PRIMARY_AGGREGATION,
    TARGETS,
    V1,
    V21_REPLAY,
    V26A_PARENT,
    V28I,
    V29I,
    V29T,
    assert_legacy_reproduces_parent,
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
V29 = Path("local/runs/optimization-v0.29-oob-leaf-responses-r1")
V30 = Path("local/runs/optimization-v0.30-oob-compose-and-time-growth-r6")
V31_WORKER = Path("workers/qrf_v031/worker.py")
WORKER_PYTHON = Path("workers/qrf_v015/.venv/bin/python")
ORIGINS = range(6, 12)
SLOTS = range(6, 13)
CANDIDATES = {"A": CANDIDATE_A, "B": CANDIDATE_B}
V26_SOURCE_ID = "V26A_QRF_ABSOLUTE_SPLIT_TIME"
V28_SOURCE_ID = "V28I_CB_QRF_EQUAL_BLEND"
V30_SOURCE_ID = PARENT
V30A_RESULT = V30 / "submissions" / V30_SOURCE_ID / "result.csv"
V30A_ZIP = V30 / "submissions" / V30_SOURCE_ID / "Luqhhh_bf_tap_predict_prelim.zip"
V26_RESULT = V26 / "submissions" / V26_SOURCE_ID / "result.csv"
V29I_RESULT = V29 / "submissions" / V29I / "result.csv"
V29I_ZIP = V29 / "submissions" / V29I / "Luqhhh_bf_tap_predict_prelim.zip"
V29T_RESULT = V29 / "submissions" / V29T / "result.csv"
V29T_ZIP = V29 / "submissions" / V29T / "Luqhhh_bf_tap_predict_prelim.zip"
V28_RESULT = V28 / "submissions" / V28_SOURCE_ID / "result.csv"
V21_RESULT = V23 / "recovery/result.csv"


def registration():
    value = load_yaml("configs/optimization_v0_31/experiment.yaml")
    if value["branch"] != "optimization-v0.31-oob-occurrence-pooling" or value["candidates"] != CANDIDATES:
        raise ContractError("v0.31 branch/candidate registration differs")
    if value["protocol"] != POOLED_PROTOCOL or value["pooled_reference_protocol"] != POOLED_PROTOCOL:
        raise ContractError("v0.31 pooled protocol registration differs")
    if value["oob_attachment_protocol"] != ATTACHMENT_PROTOCOL:
        raise ContractError("v0.31 attachment protocol registration differs")
    expected_budget = {
        "new_model_or_tree_fits": 0,
        "new_tree_fits": 0,
        "new_preprocessing_lad_beta_lambda_bias_fits": 0,
        "pooled_read_protocol_A_development": 6,
        "pooled_read_protocol_A_final": 1,
        "pooled_read_protocol_A_total": 7,
        "pooled_read_protocol_B_development": 6,
        "pooled_read_protocol_B_final": 1,
        "pooled_read_protocol_B_total": 7,
        "certified_oob_attachment_reuse_development": 12,
        "certified_oob_attachment_reuse_final": 2,
        "certified_oob_attachment_reuse_total": 14,
        "new_candidate_packages": 2,
        "new_candidate_platform_tests": 2,
        "automatic_uploads": 0,
        "desktop_writes": 0,
        "public_pushes": 0,
        "third_candidate": 0,
    }
    if value["budget"] != expected_budget or value["platform_order"] != ["A", "B"]:
        raise ContractError("v0.31 budget/order registration differs")
    if value["platform_feedback_may_change_second_candidate"] is not False:
        raise ContractError("v0.31 second candidate must stay frozen across feedback")
    rule = value["pooled_rule"]
    if (
        rule["weight"] != "c_i_over_S" or rule["S"] != "sum_over_trees_n_b"
        or rule["deduplicate_within_leaf"] != "forbidden"
        or rule["deduplicate_across_trees"] != "forbidden"
        or rule["bootstrap_multiplicity_as_response_weight"] != "forbidden"
        or rule["median"] != "first_stable_response_with_cumulative_integer_count_reaching_ceil_S_over_2"
        or rule["even_S"] != "smaller_middle_value"
    ):
        raise ContractError("v0.31 occurrence-pooling rule registration differs")
    return value


def _source_files():
    return [
        Path("configs/optimization_v0_31/experiment.yaml"),
        Path("configs/optimization_v0_31/access_scope.yaml"),
        Path("src/bf_tap/optimization/oob_pool_v31.py"),
        Path("src/bf_tap/optimization/oob_pool_v31_run.py"),
        Path("scripts/optimization_v31_oob_pool.py"),
        Path("scripts/optimization_v31_cold_check.py"),
        Path("tests/test_optimization_v31_oob_pool.py"),
        Path("docs/optimization_v0_31/PLAN.md"),
        V31_WORKER,
        Path("workers/qrf_v031/pooled_response.py"),
        Path("workers/qrf_v031/pooled_oob_reference.py"),
        Path("workers/qrf_v031/test_pooled_response.py"),
        Path("workers/qrf_v029/worker.py"),
        Path("workers/qrf_v029/oob_response.py"),
        Path("workers/qrf_v027/worker.py"),
        Path("workers/qrf_v027/target_forest.py"),
        Path("workers/qrf_v026/worker.py"),
        Path("workers/qrf_v026/partition_forest.py"),
        Path("workers/qrf_v015/qrf_model.py"),
        Path("workers/qrf_v015/preprocessing.py"),
        Path("workers/qrf_v015/uv.lock"),
    ]


def _source_path(slot: int, role: str) -> Path:
    if role == "parent":
        return V30A_RESULT if slot == 12 else V30 / "predictions" / f"{slot}_{V30_SOURCE_ID}.csv"
    if role == "v26a":
        return V26_RESULT if slot == 12 else V26 / "predictions" / f"{slot}_{V26_SOURCE_ID}.csv"
    if role == "v29i":
        return V29I_RESULT if slot == 12 else V29 / "predictions" / f"{slot}_{V29I}.csv"
    if role == "v29t":
        return V29T_RESULT if slot == 12 else V29 / "predictions" / f"{slot}_{V29T}.csv"
    if role == "v28i":
        return V28_RESULT if slot == 12 else V28 / "predictions" / f"{slot}_{V28_SOURCE_ID}.csv"
    raise ContractError("unknown v0.31 endpoint role")


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
        reasons = [item for item in (*CANDIDATES.values(), POOLED_PROTOCOL) if item in text]
        if reasons:
            matches.append({"path": str(path), "sha256": file_sha256(path), "matches": reasons})
    completed, incomplete = [], []
    for match in matches:
        root = Path(match["path"]).parent
        (completed if (root / "completion.json").is_file() else incomplete).append(match)
    return {"searched_manifest_count": len(manifests), "equivalent_matches": completed, "incomplete_equivalent_attempts": incomplete}


def register(root: Path):
    config = registration()
    scope = load_yaml("configs/optimization_v0_31/access_scope.yaml")
    if root.exists():
        raise ContractError("never overwrite a v0.31 run")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ContractError("clean preregistered working tree required")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != config["branch"]:
        raise ContractError("registered v0.31 branch required")
    subprocess.run(["git", "merge-base", "--is-ancestor", config["base_commit"], "HEAD"], check=True)
    registry = _registry_search()
    if registry["equivalent_matches"]:
        raise ContractError("equivalent completed v0.31 experiment already exists")
    identities = config["source_identities"]
    for key, path in {
        "v30a_result_sha256": V30A_RESULT,
        "v30a_zip_sha256": V30A_ZIP,
        "v29i_result_sha256": V29I_RESULT,
        "v29i_zip_sha256": V29I_ZIP,
        "v29t_result_sha256": V29T_RESULT,
        "v29t_zip_sha256": V29T_ZIP,
        "v28i_result_sha256": V28_RESULT,
        "v26a_result_sha256": V26_RESULT,
        "v21_result_sha256": V21_RESULT,
    }.items():
        if file_sha256(path) != identities[key]:
            raise ContractError(f"registered v0.31 source identity differs: {key}")
    evidence = {
        "v30_manifest": V30 / "manifest.json", "v30_completion": V30 / "completion.json",
        "v30_cold_validation": V30 / "cold_validation.json",
        "v30_packages_frozen": V30 / "packages_frozen_before_feedback.json",
        "v30_feedback": V30 / "platform_feedback_user_reported.json",
        "v30a_result": V30A_RESULT, "v30a_zip": V30A_ZIP,
        "v29_manifest": V29 / "manifest.json", "v29_completion": V29 / "completion.json",
        "v29_cold_validation": V29 / "cold_validation.json",
        "v29_packages_frozen": V29 / "packages_frozen_before_feedback.json",
        "v29_development_frozen": V29 / "development_predictions_frozen.json",
        "v29i_result": V29I_RESULT, "v29i_zip": V29I_ZIP, "v29t_result": V29T_RESULT, "v29t_zip": V29T_ZIP,
        "v28_manifest": V28 / "manifest.json", "v28_completion": V28 / "completion.json",
        "v28_errors": V28 / "all_errors.csv", "v28_parent_result": V28_RESULT,
        "v27_manifest": V27 / "manifest.json", "v27_completion": V27 / "completion.json",
        "v26_manifest": V26 / "manifest.json", "v26_completion": V26 / "completion.json",
        "v26_errors": V26 / "all_errors.csv", "v26a_result": V26_RESULT,
        "v26_models_complete_development": V26 / "development_models_complete.json",
        "v26_models_complete_final": V26 / "final_models_complete.json",
        "v27_models_complete_development": V27 / "development_models_complete.json",
        "v27_models_complete_final": V27 / "final_models_complete.json",
        "v22_errors": V22 / "all_errors.csv", "v23_v21_result": V21_RESULT,
        "protection": Path(scope["protection_contract"]),
    }
    for candidate in ("A", "B"):
        for slot in SLOTS:
            evidence[f"v29_attachment_{candidate}_{slot}"] = V29 / "oob_attachments" / candidate / f"{slot}.npz"
            evidence[f"v29_attachment_{candidate}_{slot}_metadata"] = V29 / "oob_attachments" / candidate / f"{slot}.npz.json"
            evidence[f"v29_legacy_prediction_{candidate}_{slot}"] = V29 / "worker_predictions" / candidate / f"{slot}.npz"
            evidence[f"v29_legacy_prediction_{candidate}_{slot}_metadata"] = V29 / "worker_predictions" / candidate / f"{slot}.npz.json"
    missing = [name for name, path in evidence.items() if not Path(path).is_file()]
    if missing:
        raise ContractError(f"missing registered v0.31 evidence: {missing}")
    worker_environment = json.loads(_worker(V31_WORKER, "environment"))
    worker_sources = json.loads(_worker(V31_WORKER, "identity"))
    worker_registration = json.loads(_worker(V31_WORKER, "registration"))
    if worker_registration["pooled_protocol"] != POOLED_PROTOCOL or worker_registration["attachment_protocol"] != ATTACHMENT_PROTOCOL:
        raise ContractError("v0.31 worker protocol differs from registration")
    manifest = {
        "registration": config, "scope": scope, "protection": load_yaml(scope["protection_contract"]),
        "evidence": file_identities(evidence),
        "sources": file_identities({str(path): path for path in _source_files()}),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_environment(), "worker_environment": worker_environment, "worker_sources": worker_sources,
        "worker_registration": worker_registration,
        "candidate_targets": {
            "A": {"target": "tap_iron", "unit": "tonne", "source": "V27I_ABS_QRF_DIRECT_IRON", "attachment": "v029_A"},
            "B": {"target": "tap_time_len", "unit": "minutes", "source": V26_SOURCE_ID, "attachment": "v029_B"},
        },
        "source_runs": {name: str(path.resolve()) for name, path in {
            "v26": V26, "v27": V27, "v28": V28, "v29": V29, "v30": V30,
        }.items()},
        "registry_search": registry, "holdout_consumed": True, "test_targets_read": False,
        "new_fit_budget": 0, "platform_feedback_may_change_second_candidate": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    atomic_write_json(root / "manifest.json", manifest)
    append_ledger(scope, root)
    atomic_write_json(root / "registration.json", {
        "status": "TWO_ZERO_FIT_POOLED_OOB_CANDIDATES_FROZEN_BEFORE_DERIVATION",
        "candidates": CANDIDATES,
        "definitions": {key: config[f"candidate_{key}"] for key in CANDIDATES},
        "pooled_rule": config["pooled_rule"],
        "source_endpoints": {
            "v30a_result_sha256": file_sha256(V30A_RESULT),
            "v29i_result_sha256": file_sha256(V29I_RESULT),
            "v29t_result_sha256": file_sha256(V29T_RESULT),
            "v28i_result_sha256": file_sha256(V28_RESULT),
        },
        "platform_order": ["A", "B"], "candidate_platform_budget": 2, "platform_uploads": 0,
        "definitions_may_change_after_A_feedback": False, "third_candidate_allowed": False,
    })
    return manifest


def _edge_identity_checks(slot: int):
    parent, v26a = _load_endpoint(slot, "parent"), _load_endpoint(slot, "v26a")
    v29i, v29t = _load_endpoint(slot, "v29i"), _load_endpoint(slot, "v29t")
    v28i = _load_endpoint(slot, "v28i")
    if not (parent.sample_id.tolist() == v26a.sample_id.tolist() == v29i.sample_id.tolist() == v29t.sample_id.tolist() == v28i.sample_id.tolist()):
        raise ContractError("v0.31 source endpoint sample IDs/order differ")
    if v29i.pred_tap_time_len.tolist() != v28i.pred_tap_time_len.tolist():
        raise ContractError("V29I source changed the V28I time endpoint")
    if v29t.pred_tap_iron.tolist() != v28i.pred_tap_iron.tolist():
        raise ContractError("V29T source changed the V28I iron endpoint")
    if v26a.pred_tap_time_len.tolist() != v28i.pred_tap_time_len.tolist():
        raise ContractError("V26A complete time endpoint differs from V28I")
    if parent.pred_tap_time_len.tolist() != v29t.pred_tap_time_len.tolist():
        raise ContractError("V30A parent time strings differ from V29T")
    expected_iron = [equal_blend_six(complete, pooled) for complete, pooled in zip(v26a.pred_tap_iron, v29i.pred_tap_iron, strict=True)]
    if expected_iron != parent.pred_tap_iron.tolist():
        raise ContractError("V30A parent iron strings are not mean6(V26A complete, V29I)")
    return {
        "sample_ids_exact": True,
        "v29i_time_equals_v28i": True,
        "v29t_iron_equals_v28i": True,
        "v26a_time_equals_v28i": True,
        "parent_time_equals_v29t": True,
        "parent_iron_recomposition_exact": True,
    }


def prepare(root: Path):
    if (root / "p0_complete.json").exists():
        raise ContractError("v0.31 P0 already complete")
    audits = {}
    for slot in SLOTS:
        audits[str(slot)] = {}
        audits[str(slot)]["identities"] = _edge_identity_checks(slot)
        for candidate in CANDIDATES:
            receipt = json.loads(_worker(
                V31_WORKER, "audit", "--root", root.resolve(), "--slot", slot, "--candidate", candidate,
            ))
            if any(receipt["zero_fit"].values()):
                raise ContractError("v0.31 P0 audit attempted a fit")
            if receipt["attachment_rederived_from_bootstrap_exact"] is not True or receipt["attachment_trees_checked"] != 256:
                raise ContractError("v0.31 P0 attachment/bootstrap audit differs")
            audits[str(slot)][candidate] = receipt
    atomic_write_json(root / "p0_complete.json", {
        "status": "PASS_ZERO_FIT_SOURCE_FOREST_AND_V029_ATTACHMENT_AUDIT",
        "audits": audits, "registry_search": read_json(root / "manifest.json")["registry_search"],
        "new_model_fits": 0, "new_tree_fits": 0, "new_preprocessor_fits": 0,
        "new_calibration_fits": 0, "test_targets_read": False,
    })


def _write_prediction(path: Path, value: pd.DataFrame):
    if path.exists():
        raise ContractError("never overwrite a v0.31 prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(submission_bytes(value))


def _worker_predictions(root: Path, slot: int, candidate: str):
    path = root / "worker_predictions" / candidate / f"{slot}.npz"
    if not path.is_file():
        raise ContractError(f"missing v0.31 worker prediction: {path}")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    required = {"ids", "median", "legacy_median"}
    if not required.issubset(arrays):
        raise ContractError("v0.31 worker prediction fields differ")
    return path, arrays


def _pool_view(root: Path, slot: int, candidate: str, worker_path: Path):
    receipt = read_json(str(worker_path) + ".json")
    return {
        "slot": slot,
        "candidate": CANDIDATES[candidate],
        "target": TARGETS[candidate]["target"],
        "unit": TARGETS[candidate]["unit"],
        "source_tree_count": 256,
        "S_is_occurrence_total_not_new_samples": True,
        "attachment_path": receipt["attachment_path"],
        "attachment_sha256": receipt["attachment_sha256"],
        "legacy_equal_tree_switch_back_exact": receipt["legacy_equal_tree_switch_back_exact"],
        "legacy_v29_endpoint_sha256": receipt["legacy_v29_endpoint_sha256"],
        "changed_from_legacy_count": receipt["changed_from_legacy_count"],
        "diagnostics": receipt["diagnostics"],
    }


def _compose_slot(root: Path, slot: int, *, derive: bool, write: bool):
    parent, v26a = _load_endpoint(slot, "parent"), _load_endpoint(slot, "v26a")
    v29i, v29t = _load_endpoint(slot, "v29i"), _load_endpoint(slot, "v29t")
    outputs = {}
    for candidate in ("A", "B"):
        output = root / "worker_predictions" / candidate / f"{slot}.npz"
        if derive and not output.is_file():
            _worker(
                V31_WORKER, "derive-predict", "--root", root.resolve(), "--slot", slot,
                "--candidate", candidate, "--output", output.resolve(),
            )
        path, arrays = _worker_predictions(root, slot, candidate)
        if arrays["ids"].tolist() != parent.sample_id.tolist():
            raise ContractError("v0.31 worker prediction IDs differ")
        outputs[candidate] = arrays

    pooled_a = np.asarray(outputs["A"]["median"], dtype=np.float64)
    pooled_b = np.asarray(outputs["B"]["median"], dtype=np.float64)
    legacy_a = np.asarray(outputs["A"]["legacy_median"], dtype=np.float64)
    legacy_b = np.asarray(outputs["B"]["legacy_median"], dtype=np.float64)

    a = compose_iron_candidate(parent, v26a, pooled_a)
    legacy_iron_switch = compose_iron_candidate(parent, v26a, legacy_a)
    if not legacy_iron_switch.equals(parent):
        raise ContractError("legacy equal-tree switch-back does not reproduce the V30A iron parent")
    if a.pred_tap_time_len.tolist() != parent.pred_tap_time_len.tolist():
        raise ContractError("v0.31 A changed the parent time strings")

    replayed, replay_audit = replay_qrf(V26, slot, pooled_b)
    b = compose_time_candidate(parent, replayed.pred_tap_time_len.to_numpy(dtype=float))
    legacy_replayed, legacy_replay_audit = replay_qrf(V26, slot, legacy_b)
    legacy_time_switch = compose_time_candidate(parent, legacy_replayed.pred_tap_time_len.to_numpy(dtype=float))
    if not legacy_time_switch.equals(parent):
        raise ContractError("legacy equal-tree switch-back does not reproduce the V30A time parent")
    if b.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist():
        raise ContractError("v0.31 B changed the parent iron strings")

    audit = {
        "A": {
            "base_complete_endpoint": V26A_PARENT,
            "pooled_oob_endpoint_rule": "occurrence_weighted_lower_median",
            "q6_source": "V27I_OOB_occurrence_pooled",
            "weight_each": 0.5,
            "unchanged_time_exact": a.pred_tap_time_len.tolist() == parent.pred_tap_time_len.tolist(),
            "legacy_equal_tree_switch_back_exact": True,
        },
        "B": {
            "raw_oob_endpoint_rule": "occurrence_weighted_lower_median",
            "replay": replay_audit,
            "legacy_replay": legacy_replay_audit,
            "unchanged_iron_exact": b.pred_tap_iron.tolist() == parent.pred_tap_iron.tolist(),
            "gate_source": "recomputed_original_v015_effective_neighbors",
            "gate_uses_pooled_effective_neighbors": False,
            "legacy_equal_tree_switch_back_exact": True,
        },
    }
    if write:
        _write_prediction(root / "predictions" / f"{slot}_{CANDIDATE_A}.csv", a)
        _write_prediction(root / "predictions" / f"{slot}_{CANDIDATE_B}.csv", b)
        atomic_write_json(root / "predictions" / f"{slot}_audit.json", audit)
        for candidate in ("A", "B"):
            _worker_predictions(root, slot, candidate)
            view = _pool_view(root, slot, candidate, root / "worker_predictions" / candidate / f"{slot}.npz")
            view_path = root / "pool_views" / f"{candidate}_{slot}.json"
            if view_path.exists():
                raise ContractError("never overwrite a v0.31 pool view")
            atomic_write_json(view_path, view)
    return {"parent": parent, "A": a, "B": b, "replay": replay_audit, "legacy_replay": legacy_replay_audit}


def develop(root: Path):
    if (root / "development_complete.json").exists():
        raise ContractError("v0.31 development already complete")
    for slot in ORIGINS:
        _compose_slot(root, slot, derive=True, write=True)
    prediction_files = {
        f"{slot}_{candidate}": root / "predictions" / f"{slot}_{candidate}.csv"
        for slot in ORIGINS for candidate in CANDIDATES.values()
    }
    pool_view_files = {
        f"{candidate}_{slot}": root / "pool_views" / f"{candidate}_{slot}.json"
        for slot in ORIGINS for candidate in CANDIDATES
    }
    attachment_files = {
        f"{candidate}_{slot}": V29 / "oob_attachments" / candidate / f"{slot}.npz"
        for slot in ORIGINS for candidate in CANDIDATES
    }
    atomic_write_json(root / "development_predictions_frozen.json", {
        "status": "BOTH_CANDIDATES_AND_SOURCE_ATTACHMENTS_FROZEN_BEFORE_SCORING",
        "predictions": file_identities(prediction_files),
        "pool_views": file_identities(pool_view_files),
        "certified_source_attachments": file_identities(attachment_files),
        "platform_feedback_seen": False,
    })
    atomic_write_json(root / "development_complete.json", {
        "status": "PASS_G0_DEVELOPMENT_POOLED_RESPONSES_FROZEN",
        "new_model_fits": 0, "new_tree_fits": 0, "new_preprocessor_fits": 0,
        "oob_attachments_reused": 12, "pooled_read_protocol_A": 6, "pooled_read_protocol_B": 6,
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


def _write_report(root: Path, deltas: pd.DataFrame):
    lines = [
        "# optimization-v0.31 自动生成离线报告", "",
        "主口径为 `macro_origin_mean_wmape`。负 Δ 表示优于参照；OOB 是叶响应估计，不是独立验证集。", "",
        "| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in CANDIDATES.values():
        for reference in deltas.reference.unique():
            part = deltas.loc[(deltas.candidate == candidate) & (deltas.reference == reference)].set_index("scope")
            values = [part.loc[scope, "delta_E"] for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")]
            lines.append(f"| {candidate} | {reference} | " + " | ".join(f"{value:+.8f}" for value in values) + " |")
    lines.extend(["", "G0、已消费历史 G1 与平台反馈分别登记。", ""])
    with (root / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def score(root: Path):
    if (root / "offline_assessment.json").exists():
        raise ContractError("never overwrite v0.31 scoring evidence")
    candidates = {candidate: _candidate_errors(root, candidate) for candidate in CANDIDATES.values()}
    references = {
        V1: _reference_errors(V1),
        V21_REPLAY: _reference_errors(V21_REPLAY),
        V28I: _source_errors(V28 / "all_errors.csv", V28_SOURCE_ID, V28I),
        V29I: _source_errors(V29 / "all_errors.csv", V29I, V29I),
        V29T: _source_errors(V29 / "all_errors.csv", V29T, V29T),
        PARENT: _source_errors(V30 / "all_errors.csv", V30_SOURCE_ID, PARENT),
    }
    all_errors = pd.concat([*references.values(), *candidates.values()], ignore_index=True)
    all_errors.to_csv(root / "all_errors.csv", index=False)
    scorecard = pd.DataFrame([row for name, errors in {**references, **candidates}.items() for row in _score_rows(errors, name)])
    scorecard.to_csv(root / "canonical_scorecard.csv", index=False)
    summary = macro_origin_summary(scorecard)
    summary.to_csv(root / "canonical_summary.csv", index=False)
    pooled = exposure_pooled_summary(scorecard)
    pooled.to_csv(root / "exposure_pooled_summary.csv", index=False)
    isolation = verify_isolated_deltas(scorecard, summary, {
        CANDIDATE_A: "tap_iron", CANDIDATE_B: "tap_time_len",
    }, parent=PARENT)
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
    signed = all_errors.groupby(["candidate", "origin", "horizon", "spout_no"], as_index=False).agg(
        n=("sample_id", "size"), signed_bias_iron=("error_tap_iron", "mean"),
        signed_bias_time=("error_tap_time_len", "mean"),
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
        "isolated_target_audit": isolation,
        "OOB_is_not_an_independent_validation_set": True,
        "S_is_not_a_sample_count": True,
        "both_candidates_retain_preregistered_platform_slot": True,
    })


def finalize(root: Path):
    if not (root / "offline_assessment.json").exists() or (root / "packages_frozen_before_feedback.json").exists():
        raise ContractError("score v0.31 once before finalization")
    composed = _compose_slot(root, 12, derive=True, write=True)
    parent = composed["parent"]
    receipts = {}
    for key, candidate in CANDIDATES.items():
        prediction = validate_endpoint(
            pd.read_csv(root / "predictions" / f"12_{candidate}.csv", dtype=str, keep_default_na=False), label=candidate,
        )
        if len(prediction) != registration()["test_a_rows"]:
            raise ContractError("v0.31 semantic package row count differs")
        if key == "A":
            if prediction.pred_tap_time_len.tolist() != parent.pred_tap_time_len.tolist():
                raise ContractError("v0.31 A final time strings differ from V30A parent")
        else:
            if prediction.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist():
                raise ContractError("v0.31 B final iron strings differ from V30A parent")
        if prediction.pred_tap_iron.tolist() == parent.pred_tap_iron.tolist() and prediction.pred_tap_time_len.tolist() == parent.pred_tap_time_len.tolist():
            raise ContractError("v0.31 candidate degenerates to the complete V30A parent at submission precision")
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
                raise ContractError("v0.31 ZIP payload differs")
        reread = validate_endpoint(pd.read_csv(csv_path, dtype=str, keep_default_na=False), label="package")
        if len(reread) != registration()["test_a_rows"]:
            raise ContractError("v0.31 package test_a coverage differs")
        receipts[key] = {
            "candidate": candidate, "rows": len(reread),
            "result_sha256": file_sha256(csv_path), "zip_sha256": file_sha256(archive),
            "unchanged_target": "time_exact_V30A" if key == "A" else "iron_exact_V30A",
            "changed_target": "iron_pooled_q6_mean6_with_V26A" if key == "A" else "time_pooled_then_original_gate",
            "platform_uploads": 0, "platform_feedback": None,
        }
    atomic_write_json(root / "packages_frozen_before_feedback.json", {
        "status": "PASS_TWO_COMPLETE_PACKAGES_FROZEN",
        "order": ["A", "B"], "receipts": receipts,
        "definitions_may_change_after_A_feedback": False,
        "third_candidate_generated": False, "platform_upload_budget": 2, "platform_uploads": 0,
        "automatic_upload": False,
    })


def _assert_frame_equal(actual, expected, message):
    left = validate_endpoint(actual, label="actual").reset_index(drop=True)
    right = validate_endpoint(expected, label="expected").reset_index(drop=True)
    if not left.equals(right):
        raise ContractError(message)


def cold(root: Path):
    if (root / "cold_validation.json").exists() or (root / "completion.json").exists():
        raise ContractError("never overwrite v0.31 cold/completion evidence")
    receipts = {"A": {}, "B": {}}
    zero_fit = True
    for slot in SLOTS:
        # Recompute the exact source and attachment in an independent process,
        # compare with the frozen worker output, and exercise invariance.
        for candidate in ("A", "B"):
            output = root / "cold" / candidate / f"{slot}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            receipt = json.loads(_worker(
                V31_WORKER, "cold", "--root", root.resolve(), "--slot", slot,
                "--candidate", candidate, "--output", output.resolve(),
            ))
            receipts[candidate][str(slot)] = receipt
            zero_fit &= not any(receipt["zero_fit"].values())
        composed = _compose_slot(root, slot, derive=False, write=False)
        _assert_frame_equal(
            composed["A"], pd.read_csv(root / "predictions" / f"{slot}_{CANDIDATE_A}.csv", dtype=str, keep_default_na=False),
            "v0.31 cold A differs",
        )
        _assert_frame_equal(
            composed["B"], pd.read_csv(root / "predictions" / f"{slot}_{CANDIDATE_B}.csv", dtype=str, keep_default_na=False),
            "v0.31 cold B differs",
        )
    if not zero_fit:
        raise ContractError("v0.31 cold restoration attempted a fit")
    package = read_json(root / "packages_frozen_before_feedback.json")
    for key, candidate in CANDIDATES.items():
        folder = root / "submissions" / candidate
        csv_path, archive = folder / "result.csv", folder / "Luqhhh_bf_tap_predict_prelim.zip"
        receipt = package["receipts"][key]
        if file_sha256(csv_path) != receipt["result_sha256"] or file_sha256(archive) != receipt["zip_sha256"]:
            raise ContractError("v0.31 frozen package identity changed")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != csv_path.read_bytes():
                raise ContractError("v0.31 cold ZIP reread differs")
    fit_counts = {
        "new_base_model_attempted": 0, "new_base_model_completed": 0,
        "new_tree_attempted": 0, "new_tree_completed": 0,
        "new_preprocessor_attempted": 0, "new_preprocessor_completed": 0,
        "new_calibration_attempted": 0, "new_calibration_completed": 0,
        "certified_oob_attachment_reads_A": 7, "certified_oob_attachment_reads_B": 7,
        "new_oob_attachment_created": 0,
        "new_candidate_packages": 2, "platform_uploads": 0, "third_candidate_generated": False,
    }
    atomic_write_json(root / "fit_counts.json", fit_counts)
    atomic_write_json(root / "cold_validation.json", {
        "status": "PASS",
        "all_slots": list(SLOTS),
        "source_forests_and_certified_v029_attachments_rederived_exact": True,
        "selected_members_unchanged_only_mass_allocation_changed": True,
        "source_checks": "full_reverse_chunk_subset_single_exact",
        "pooled_checks": "full_reverse_chunk_subset_single_exact",
        "legacy_equal_tree_switch_back_exact": True,
        "fit_attempts_during_cold": {
            "RandomForestRegressor": 0, "ExtraTreesRegressor": 0, "IronTargetForest": 0,
            "PartitionForest": 0, "old_QRF": 0, "preprocessor": 0, "CatBoost": 0,
            "LAD_or_calibration": 0,
        },
        "trusted_private_bundles_only": True,
        "source_files_and_tree_structures_unchanged": True,
        "zip_and_csv_reread": True, "receipts": receipts,
    })
    atomic_write_json(root / "completion.json", {
        "status": "READY_FOR_TWO_EXPLICIT_PLATFORM_SUBMISSIONS",
        "G0_engineering": "PASS",
        "G1_model_quality": "REPORTED_SEPARATELY_NOT_A_GATE",
        "candidates": list(CANDIDATES.values()),
        "platform_order": ["A", "B"], "platform_uploads": 0, "platform_budget": 2,
        "packages_sha256": file_sha256(root / "packages_frozen_before_feedback.json"),
        "cold_validation_sha256": file_sha256(root / "cold_validation.json"),
        "fit_counts_sha256": file_sha256(root / "fit_counts.json"),
        "test_targets_read": False, "automatic_upload": False,
        "desktop_writes": 0, "public_pushes": 0, "third_candidate_generated": False,
    })


def run(root: Path):
    register(root)
    prepare(root)
    develop(root)
    score(root)
    finalize(root)
    cold(root)
