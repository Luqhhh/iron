"""Registered zero-fit v0.34 lifecycle for OOB leaf-point bagging."""
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
from .dual_burden_v24_run import _reference_errors
from .dual_ratio_common import week_intervals
from .oob_leaf_point_v34 import (
    ATTACHMENT_PROTOCOL,
    CANDIDATE_A,
    CANDIDATE_B,
    LEAF_POINT_PROTOCOL,
    PARENT,
    POOLED_AGGREGATION,
    PRIMARY_AGGREGATION,
    TARGETS,
    V1,
    V21_REPLAY,
    compose_iron_candidate,
    compose_time_candidate,
    exposure_pooled_summary,
    macro_origin_summary,
    submission_bytes,
    validate_endpoint,
    verify_isolated_deltas,
)
from .oob_pool_v31_run import (
    V26,
    V27,
    V29,
    V30,
    V30A_RESULT,
    V30A_ZIP,
    _candidate_errors,
    _edge_identity_checks,
    _load_endpoint,
    _source_errors,
)
from .qrf_partition_v26_run import _replay as replay_qrf, _score_rows
from .structural_run import append_ledger


WORKER = Path("workers/qrf_v034/worker.py")
WORKER_PYTHON = Path("workers/qrf_v015/.venv/bin/python")
V33 = Path("local/runs/optimization-v0.33-qrf-feature-and-row-sampling-r1")
ORIGINS = range(6, 12)
SLOTS = range(6, 13)
CANDIDATES = {"A": CANDIDATE_A, "B": CANDIDATE_B}
EXPECTED_ROWS = {6: 888, 7: 1180, 8: 1490, 9: 1803, 10: 2091, 11: 2424, 12: 2754}


def registration():
    value = load_yaml("configs/optimization_v0_34/experiment.yaml")
    if value["branch"] != "optimization-v0.34-oob-leaf-median-bagging":
        raise ContractError("v0.34 branch registration differs")
    if value["base_commit"] != "e8dda75a93f159620d9f70ae0cfeed3e2cbf68d0":
        raise ContractError("v0.34 base commit registration differs")
    if value["candidates"] != CANDIDATES or value["parent"] != PARENT:
        raise ContractError("v0.34 candidate/parent registration differs")
    if value["protocol"] != LEAF_POINT_PROTOCOL or value["oob_attachment_protocol"] != ATTACHMENT_PROTOCOL:
        raise ContractError("v0.34 protocol registration differs")
    if value["training_rows"] != EXPECTED_ROWS:
        raise ContractError("v0.34 training-row identity differs")
    rule = value["leaf_point_rule"]
    expected = {
        "within_tree": "smaller_lower_median_of_raw_selected_responses",
        "even_count": "smaller_middle_value",
        "round_each_leaf_point_before_aggregation": False,
        "cross_tree": "exact_arithmetic_mean_of_binary64_leaf_points",
        "tree_count": 256,
        "tree_weight": "equal",
        "leaf_size_weight": "forbidden",
        "response_variance_weight": "forbidden",
        "recency_weight": "forbidden",
        "spout_conditioning": "forbidden",
        "occurrence_pooling": "forbidden",
        "call_forest_predict": "forbidden",
    }
    if rule != expected:
        raise ContractError("v0.34 leaf-point rule registration differs")
    budget = value["budget"]
    if budget != {
        "new_model_or_tree_fits": 0,
        "new_preprocessing_lad_beta_lambda_bias_fits": 0,
        "leaf_point_tables_A_development": 6,
        "leaf_point_tables_A_final": 1,
        "leaf_point_tables_A_total": 7,
        "leaf_point_tables_B_development": 6,
        "leaf_point_tables_B_final": 1,
        "leaf_point_tables_B_total": 7,
        "leaf_point_tables_total": 14,
        "certified_oob_attachment_reads_total": 14,
        "new_candidate_packages": 2,
        "new_candidate_platform_tests": 2,
        "automatic_uploads": 0,
        "restore_uploads": 0,
        "desktop_writes": 0,
        "third_candidate": 0,
    }:
        raise ContractError("v0.34 budget registration differs")
    if value["platform_order"] != ["A", "B"] or value["platform_feedback_may_change_second_candidate"] is not False:
        raise ContractError("v0.34 platform order/freeze rule differs")
    return value


def _source_files():
    return [
        Path("configs/optimization_v0_34/experiment.yaml"), Path("configs/optimization_v0_34/access_scope.yaml"),
        Path("src/bf_tap/optimization/oob_leaf_point_v34.py"), Path("src/bf_tap/optimization/oob_leaf_point_v34_run.py"),
        Path("scripts/optimization_v34_oob_leaf_point.py"), Path("scripts/optimization_v34_cold_check.py"),
        Path("tests/test_optimization_v34_oob_leaf_point.py"), Path("docs/optimization_v0_34/PLAN.md"),
        WORKER, Path("workers/qrf_v034/leaf_point_bagging.py"), Path("workers/qrf_v034/test_leaf_point_bagging.py"),
        Path("workers/qrf_v031/worker.py"), Path("workers/qrf_v029/worker.py"), Path("workers/qrf_v029/oob_response.py"),
        Path("workers/qrf_v027/worker.py"), Path("workers/qrf_v027/target_forest.py"),
        Path("workers/qrf_v026/worker.py"), Path("workers/qrf_v026/partition_forest.py"),
        Path("workers/qrf_v015/qrf_model.py"), Path("workers/qrf_v015/preprocessing.py"), Path("workers/qrf_v015/uv.lock"),
    ]


def _worker(command: str, *arguments):
    return subprocess.check_output([str(WORKER_PYTHON), str(WORKER), command, *map(str, arguments)], text=True)


def _registry_search():
    manifests = sorted(Path("local/runs").glob("**/*manifest*.json"))
    matches = []
    for path in manifests:
        text = path.read_text(encoding="utf-8", errors="ignore")
        reasons = [token for token in (*CANDIDATES.values(), LEAF_POINT_PROTOCOL) if token in text]
        if reasons:
            matches.append({"path": str(path), "sha256": file_sha256(path), "matches": reasons})
    completed, incomplete = [], []
    for match in matches:
        root = Path(match["path"]).parent
        (completed if (root / "completion.json").is_file() else incomplete).append(match)
    return {"searched_manifest_count": len(manifests), "equivalent_matches": completed, "incomplete_equivalent_attempts": incomplete}


def register(root: Path):
    config = registration()
    scope = load_yaml("configs/optimization_v0_34/access_scope.yaml")
    if root.exists():
        raise ContractError("never overwrite a v0.34 run")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ContractError("clean preregistered working tree required")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != config["branch"]:
        raise ContractError("registered v0.34 branch required")
    subprocess.run(["git", "merge-base", "--is-ancestor", config["base_commit"], "HEAD"], check=True)
    registry = _registry_search()
    if registry["equivalent_matches"]:
        raise ContractError("equivalent completed v0.34 experiment already exists")
    identities = config["source_identities"]
    if file_sha256(V30A_RESULT) != identities["v30a_result_sha256"] or file_sha256(V30A_ZIP) != identities["v30a_zip_sha256"]:
        raise ContractError("V30A fallback identity differs")
    evidence = {
        "v26_manifest": V26 / "manifest.json", "v26_completion": V26 / "completion.json",
        "v26_development_models": V26 / "development_models_complete.json", "v26_final_models": V26 / "final_models_complete.json",
        "v27_manifest": V27 / "manifest.json", "v27_completion": V27 / "completion.json",
        "v27_development_models": V27 / "development_models_complete.json", "v27_final_models": V27 / "final_models_complete.json",
        "v29_manifest": V29 / "manifest.json", "v29_completion": V29 / "completion.json",
        "v29_cold": V29 / "cold_validation.json", "v30_manifest": V30 / "manifest.json",
        "v30_completion": V30 / "completion.json", "v30_errors": V30 / "all_errors.csv",
        "v30a_result": V30A_RESULT, "v30a_zip": V30A_ZIP,
        "v33_results": Path("docs/optimization_v0_33/RESULTS.md"),
        "v33_feedback": V33 / "platform_feedback_complete.json", "protection": Path(scope["protection_contract"]),
    }
    for candidate in ("A", "B"):
        for slot in SLOTS:
            evidence[f"v29_attachment_{candidate}_{slot}"] = V29 / "oob_attachments" / candidate / f"{slot}.npz"
            evidence[f"v29_attachment_{candidate}_{slot}_metadata"] = V29 / "oob_attachments" / candidate / f"{slot}.npz.json"
            evidence[f"v29_prediction_{candidate}_{slot}"] = V29 / "worker_predictions" / candidate / f"{slot}.npz"
            evidence[f"v29_prediction_{candidate}_{slot}_metadata"] = V29 / "worker_predictions" / candidate / f"{slot}.npz.json"
    missing = [name for name, path in evidence.items() if not Path(path).is_file()]
    if missing:
        raise ContractError(f"missing registered v0.34 evidence: {missing}")
    worker_environment = json.loads(_worker("environment"))
    worker_sources = json.loads(_worker("identity"))
    worker_registration = json.loads(_worker("registration"))
    if worker_environment.get("python") != "3.12.12" or worker_environment.get("sklearn") != "1.8.0":
        raise ContractError("locked Python 3.12.12 / sklearn 1.8.0 worker required")
    if worker_registration["protocol"] != LEAF_POINT_PROTOCOL or worker_registration["attachment_protocol"] != ATTACHMENT_PROTOCOL:
        raise ContractError("v0.34 worker protocol differs")
    manifest = {
        "registration": config, "scope": scope, "protection": load_yaml(scope["protection_contract"]),
        "evidence": file_identities(evidence), "sources": file_identities({str(path): path for path in _source_files()}),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_environment(), "worker_environment": worker_environment,
        "worker_sources": worker_sources, "worker_registration": worker_registration,
        "source_runs": {name: str(path.resolve()) for name, path in {"v26": V26, "v27": V27, "v29": V29, "v30": V30, "v33": V33}.items()},
        "registry_search": registry, "holdout_consumed": True, "test_targets_read": False,
        "new_fit_budget": 0, "platform_feedback_may_change_second_candidate": False,
    }
    root.mkdir(parents=True, exist_ok=False); root.chmod(0o700)
    atomic_write_json(root / "manifest.json", manifest)
    append_ledger(scope, root)
    atomic_write_json(root / "registration.json", {
        "status": "TWO_ZERO_FIT_LEAF_POINT_CANDIDATES_FROZEN_BEFORE_DERIVATION",
        "candidates": CANDIDATES, "definitions": {key: config[f"candidate_{key}"] for key in CANDIDATES},
        "member_rule": config["member_rule"], "leaf_point_rule": config["leaf_point_rule"],
        "platform_order": ["A", "B"], "candidate_platform_budget": 2, "platform_uploads": 0,
        "definitions_may_change_after_A_feedback": False, "third_candidate_allowed": False,
    })
    return manifest


def prepare(root: Path):
    if (root / "p0_complete.json").exists():
        raise ContractError("v0.34 P0 already complete")
    audits = {}
    for slot in SLOTS:
        audits[str(slot)] = {"identities": _edge_identity_checks(slot)}
        for candidate in CANDIDATES:
            receipt = json.loads(_worker("audit", "--root", root.resolve(), "--slot", slot, "--candidate", candidate))
            if any(receipt["zero_fit"].values()) or receipt["attachment_trees_checked"] != 256:
                raise ContractError("v0.34 P0 fit/attachment audit differs")
            if not receipt["attachment_rederived_from_bootstrap_exact"]:
                raise ContractError("v0.34 P0 bootstrap identity differs")
            audits[str(slot)][candidate] = receipt
    atomic_write_json(root / "p0_complete.json", {
        "status": "PASS_ZERO_FIT_SOURCE_FOREST_RESPONSE_BOOTSTRAP_ATTACHMENT_AUDIT",
        "audits": audits, "registry_search": read_json(root / "manifest.json")["registry_search"],
        "new_model_fits": 0, "new_tree_fits": 0, "new_preprocessor_fits": 0,
        "new_calibration_fits": 0, "test_targets_read": False,
    })


def _write_prediction(path: Path, frame: pd.DataFrame):
    if path.exists():
        raise ContractError("never overwrite a v0.34 prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(submission_bytes(frame))


def _worker_prediction(root: Path, slot: int, candidate: str):
    path = root / "worker_predictions" / candidate / f"{slot}.npz"
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if not {"ids", "point_mean", "legacy_median"}.issubset(arrays):
        raise ContractError("v0.34 worker prediction fields differ")
    return path, arrays


def _compose_slot(root: Path, slot: int, *, derive: bool, write: bool):
    parent, v26a = _load_endpoint(slot, "parent"), _load_endpoint(slot, "v26a")
    outputs = {}
    for candidate in CANDIDATES:
        output = root / "worker_predictions" / candidate / f"{slot}.npz"
        if derive and not output.is_file():
            _worker("derive-predict", "--root", root.resolve(), "--slot", slot, "--candidate", candidate, "--output", output.resolve())
        _, arrays = _worker_prediction(root, slot, candidate)
        if arrays["ids"].tolist() != parent.sample_id.tolist():
            raise ContractError("v0.34 worker/parent sample IDs differ")
        outputs[candidate] = arrays
    a = compose_iron_candidate(parent, v26a, outputs["A"]["point_mean"])
    legacy_a = compose_iron_candidate(parent, v26a, outputs["A"]["legacy_median"])
    if not legacy_a.equals(parent) or a.pred_tap_time_len.tolist() != parent.pred_tap_time_len.tolist():
        raise ContractError("v0.34 A switch-back/isolation identity failed")
    replayed, replay_audit = replay_qrf(V26, slot, outputs["B"]["point_mean"])
    b = compose_time_candidate(parent, replayed.pred_tap_time_len.to_numpy(dtype=float))
    legacy_replayed, legacy_replay_audit = replay_qrf(V26, slot, outputs["B"]["legacy_median"])
    legacy_b = compose_time_candidate(parent, legacy_replayed.pred_tap_time_len.to_numpy(dtype=float))
    if not legacy_b.equals(parent) or b.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist():
        raise ContractError("v0.34 B switch-back/isolation identity failed")
    audits = {
        "A": {"rule": "lower_median_per_selected_leaf_then_equal_tree_mean", "source": "V27I_ABS_QRF_DIRECT_IRON",
              "unchanged_time_exact": True, "legacy_v029_switch_back_exact": True},
        "B": {"rule": "lower_median_per_selected_leaf_then_equal_tree_mean", "source": "V26A_QRF_ABSOLUTE_SPLIT_TIME",
              "replay": replay_audit, "legacy_replay": legacy_replay_audit, "unchanged_iron_exact": True,
              "gate_source": "original_v015_effective_neighbors", "new_leaf_diagnostics_used_for_gate": False,
              "legacy_v029_switch_back_exact": True},
    }
    if write:
        _write_prediction(root / "predictions" / f"{slot}_{CANDIDATE_A}.csv", a)
        _write_prediction(root / "predictions" / f"{slot}_{CANDIDATE_B}.csv", b)
        atomic_write_json(root / "predictions" / f"{slot}_audit.json", audits)
        for candidate in CANDIDATES:
            receipt = read_json(str(root / "worker_predictions" / candidate / f"{slot}.npz") + ".json")
            if receipt["rows"] != len(parent):
                raise ContractError("v0.34 worker receipt row count differs from parent")
            atomic_write_json(root / "leaf_point_views" / f"{candidate}_{slot}.json", {
                "slot": slot, "candidate": CANDIDATES[candidate], "target": receipt["target"],
                "leaf_point_table_path": receipt["leaf_point_table_path"],
                "leaf_point_table_sha256": receipt["leaf_point_table_sha256"],
                "attachment_sha256": receipt["attachment_sha256"], "changed_from_legacy_count": receipt["changed_from_legacy_count"],
                "support": receipt["support"], "diagnostics": receipt["diagnostics"],
                "legacy_v029_switch_back_exact": receipt["legacy_switch_back_exact"],
            })
    return {"parent": parent, "A": a, "B": b}


def develop(root: Path):
    if (root / "development_complete.json").exists():
        raise ContractError("v0.34 development already complete")
    for slot in ORIGINS:
        _compose_slot(root, slot, derive=True, write=True)
    predictions = {f"{slot}_{candidate}": root / "predictions" / f"{slot}_{label}.csv" for slot in ORIGINS for candidate, label in CANDIDATES.items()}
    tables = {f"{candidate}_{slot}": root / "leaf_point_tables" / candidate / f"{slot}.npz" for candidate in CANDIDATES for slot in ORIGINS}
    views = {f"{candidate}_{slot}": root / "leaf_point_views" / f"{candidate}_{slot}.json" for candidate in CANDIDATES for slot in ORIGINS}
    attachments = {f"{candidate}_{slot}": V29 / "oob_attachments" / candidate / f"{slot}.npz" for candidate in CANDIDATES for slot in ORIGINS}
    atomic_write_json(root / "development_predictions_frozen.json", {
        "status": "BOTH_CANDIDATES_LEAF_POINT_TABLES_AND_PREDICTIONS_FROZEN_BEFORE_SCORING",
        "predictions": file_identities(predictions), "leaf_point_tables": file_identities(tables),
        "leaf_point_views": file_identities(views), "certified_source_attachments": file_identities(attachments),
        "platform_feedback_seen": False,
    })
    atomic_write_json(root / "development_complete.json", {
        "status": "PASS_G0_DEVELOPMENT_ZERO_FIT_LEAF_POINT_TABLES_FROZEN",
        "new_model_fits": 0, "new_tree_fits": 0, "new_preprocessor_fits": 0,
        "leaf_point_tables_A": 6, "leaf_point_tables_B": 6, "oob_attachments_reused": 12,
    })


def _write_report(root: Path, deltas: pd.DataFrame):
    lines = [
        "# optimization-v0.34 自动生成离线报告", "",
        "主口径为 `macro_origin_mean_wmape`；负 Δ 表示历史误差改善。历史标签已消费，OOB 不是独立验证集。", "",
        "| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in CANDIDATES.values():
        for reference in deltas.reference.unique():
            part = deltas.loc[(deltas.candidate == candidate) & (deltas.reference == reference)].set_index("scope")
            values = [part.loc[scope, "delta_E"] for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")]
            lines.append(f"| {candidate} | {reference} | " + " | ".join(f"{value:+.8f}" for value in values) + " |")
    lines.extend(["", "G0、已消费历史 G1 与平台反馈分别登记；树叶点分歧仅为诊断。", ""])
    with (root / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def score(root: Path):
    if (root / "offline_assessment.json").exists():
        raise ContractError("never overwrite v0.34 scoring evidence")
    candidates = {label: _candidate_errors(root, label) for label in CANDIDATES.values()}
    references = {
        V1: _reference_errors(V1), V21_REPLAY: _reference_errors(V21_REPLAY),
        PARENT: _source_errors(V30 / "all_errors.csv", PARENT, PARENT),
    }
    all_errors = pd.concat([*references.values(), *candidates.values()], ignore_index=True)
    all_errors.to_csv(root / "all_errors.csv", index=False)
    scorecard = pd.DataFrame([row for name, errors in {**references, **candidates}.items() for row in _score_rows(errors, name)])
    scorecard.to_csv(root / "canonical_scorecard.csv", index=False)
    summary = macro_origin_summary(scorecard); summary.to_csv(root / "canonical_summary.csv", index=False)
    pooled = exposure_pooled_summary(scorecard); pooled.to_csv(root / "exposure_pooled_summary.csv", index=False)
    isolation = verify_isolated_deltas(scorecard, summary, {CANDIDATE_A: "tap_iron", CANDIDATE_B: "tap_time_len"}, parent=PARENT)
    primary = summary.loc[summary.scope_type.isin(("HORIZON_MEAN", "GRID", "DEVELOPMENT"))].set_index(["algorithm", "scope"])
    rows = []
    for candidate in candidates:
        for reference in references:
            for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT"):
                left, right = primary.loc[(candidate, scope)], primary.loc[(reference, scope)]
                rows.append({"candidate": candidate, "reference": reference, "scope": scope, "aggregation": PRIMARY_AGGREGATION,
                             "E": float(left.E), "delta_E": float(left.E - right.E),
                             "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                             "delta_wmape_time": float(left.wmape_time - right.wmape_time)})
    deltas = pd.DataFrame(rows); deltas.to_csv(root / "canonical_deltas.csv", index=False)
    pooled_index = pooled.set_index(["algorithm", "scope"]); pooled_rows = []
    for candidate in candidates:
        for reference in references:
            for scope in ("H1", "H2", "H3", "H4", "DEV_LONG", "DEV_SHORT"):
                left, right = pooled_index.loc[(candidate, scope)], pooled_index.loc[(reference, scope)]
                pooled_rows.append({"candidate": candidate, "reference": reference, "scope": scope, "aggregation": POOLED_AGGREGATION,
                                    "delta_E": float(left.E - right.E), "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                                    "delta_wmape_time": float(left.wmape_time - right.wmape_time)})
    pd.DataFrame(pooled_rows).to_csv(root / "exposure_pooled_deltas.csv", index=False)
    all_errors.groupby(["candidate", "origin", "horizon", "spout_no"], as_index=False).agg(
        n=("sample_id", "size"), signed_bias_iron=("error_tap_iron", "mean"), signed_bias_time=("error_tap_time_len", "mean"),
    ).to_csv(root / "signed_bias_by_spout.csv", index=False)
    atomic_write_json(root / "bootstrap.json", {
        candidate: {reference: week_intervals(all_errors, candidate, reference, 1000, 2026) for reference in references}
        for candidate in candidates
    })
    atomic_write_json(root / "leaf_point_diagnostics.json", {
        candidate: {str(slot): read_json(root / "leaf_point_views" / f"{candidate}_{slot}.json") for slot in ORIGINS}
        for candidate in CANDIDATES
    })
    _write_report(root, deltas)
    atomic_write_json(root / "offline_assessment.json", {
        "status": "COMPLETED_CONSUMED_RETROSPECTIVE_NOT_A_PLATFORM_GATE",
        "primary_aggregation": PRIMARY_AGGREGATION, "diagnostic_aggregation": POOLED_AGGREGATION,
        "isolated_target_audit": isolation, "OOB_is_not_an_independent_validation_set": True,
        "tree_point_disagreement_is_not_calibrated_uncertainty": True,
        "both_candidates_retain_preregistered_platform_slot": True,
    })


def finalize(root: Path):
    if not (root / "offline_assessment.json").is_file() or (root / "packages_frozen_before_feedback.json").exists():
        raise ContractError("score v0.34 once before finalization")
    composed = _compose_slot(root, 12, derive=True, write=True)
    parent, receipts = composed["parent"], {}
    for key, candidate in CANDIDATES.items():
        prediction = validate_endpoint(pd.read_csv(root / "predictions" / f"12_{candidate}.csv", dtype=str, keep_default_na=False), label=candidate)
        if len(prediction) != registration()["test_a_rows"]:
            raise ContractError("v0.34 semantic package row count differs")
        unchanged = prediction.pred_tap_time_len.tolist() == parent.pred_tap_time_len.tolist() if key == "A" else prediction.pred_tap_iron.tolist() == parent.pred_tap_iron.tolist()
        if not unchanged:
            raise ContractError("v0.34 unchanged target differs from V30A")
        if prediction.equals(parent):
            receipts[key] = {"candidate": candidate, "status": "NO_OP_FINAL_PAYLOAD", "rows": len(prediction), "package_created": False}
            continue
        folder = root / "submissions" / candidate; folder.mkdir(parents=True, exist_ok=False)
        csv_path = folder / "result.csv"; payload = submission_bytes(prediction)
        with csv_path.open("xb") as handle: handle.write(payload)
        archive = folder / "Luqhhh_bf_tap_predict_prelim.zip"
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as handle: handle.write(csv_path, arcname="result.csv")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != payload:
                raise ContractError("v0.34 ZIP payload differs")
        receipts[key] = {"candidate": candidate, "status": "FROZEN", "rows": len(prediction), "package_created": True,
                         "result_sha256": file_sha256(csv_path), "zip_sha256": file_sha256(archive),
                         "unchanged_target": "time_exact_V30A" if key == "A" else "iron_exact_V30A",
                         "platform_uploads": 0, "platform_feedback": None}
    package_count = sum(item["package_created"] for item in receipts.values())
    atomic_write_json(root / "packages_frozen_before_feedback.json", {
        "status": "PASS_VALID_NON_NOOP_PACKAGES_FROZEN", "order": ["A", "B"], "receipts": receipts,
        "package_count": package_count, "no_op_count": 2 - package_count,
        "definitions_may_change_after_A_feedback": False, "third_candidate_generated": False,
        "platform_upload_budget": package_count, "platform_uploads": 0, "automatic_upload": False, "desktop_writes": 0,
    })


def _assert_frame(actual, expected, message):
    if not validate_endpoint(actual, label="actual").reset_index(drop=True).equals(validate_endpoint(expected, label="expected").reset_index(drop=True)):
        raise ContractError(message)


def cold(root: Path):
    if (root / "cold_validation.json").exists() or (root / "completion.json").exists():
        raise ContractError("never overwrite v0.34 cold/completion evidence")
    receipts, zero_fit = {"A": {}, "B": {}}, True
    for slot in SLOTS:
        for candidate in CANDIDATES:
            output = root / "cold" / candidate / f"{slot}.npz"; output.parent.mkdir(parents=True, exist_ok=True)
            receipt = json.loads(_worker("cold", "--root", root.resolve(), "--slot", slot, "--candidate", candidate, "--output", output.resolve()))
            receipts[candidate][str(slot)] = receipt; zero_fit &= not any(receipt["zero_fit"].values())
        composed = _compose_slot(root, slot, derive=False, write=False)
        _assert_frame(composed["A"], pd.read_csv(root / "predictions" / f"{slot}_{CANDIDATE_A}.csv", dtype=str, keep_default_na=False), "v0.34 cold A differs")
        _assert_frame(composed["B"], pd.read_csv(root / "predictions" / f"{slot}_{CANDIDATE_B}.csv", dtype=str, keep_default_na=False), "v0.34 cold B differs")
    if not zero_fit:
        raise ContractError("v0.34 cold restoration attempted a fit")
    package = read_json(root / "packages_frozen_before_feedback.json")
    for key, candidate in CANDIDATES.items():
        receipt = package["receipts"][key]
        if not receipt["package_created"]: continue
        folder = root / "submissions" / candidate; csv_path = folder / "result.csv"; archive = folder / "Luqhhh_bf_tap_predict_prelim.zip"
        if file_sha256(csv_path) != receipt["result_sha256"] or file_sha256(archive) != receipt["zip_sha256"]:
            raise ContractError("v0.34 frozen package identity changed")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != csv_path.read_bytes():
                raise ContractError("v0.34 cold ZIP reread differs")
    atomic_write_json(root / "fit_counts.json", {
        "new_base_model_attempted": 0, "new_base_model_completed": 0, "new_tree_attempted": 0, "new_tree_completed": 0,
        "new_preprocessor_attempted": 0, "new_preprocessor_completed": 0, "new_calibration_attempted": 0,
        "new_calibration_completed": 0, "certified_oob_attachment_reads_A": 7, "certified_oob_attachment_reads_B": 7,
        "supervised_leaf_point_tables_A": 7, "supervised_leaf_point_tables_B": 7,
        "new_candidate_packages": package["package_count"], "platform_uploads": 0, "third_candidate_generated": False,
    })
    atomic_write_json(root / "cold_validation.json", {
        "status": "PASS", "all_slots": list(SLOTS),
        "source_forests_responses_bootstrap_and_v029_selected_members_unchanged": True,
        "leaf_lower_median_tables_rederived_exact": True,
        "cross_tree_exact_binary64_mean": True,
        "full_reverse_chunk_subset_single_exact": True, "legacy_v029_switch_back_exact": True,
        "fit_attempts_during_cold": {"RandomForestRegressor": 0, "ExtraTreesRegressor": 0, "IronTargetForest": 0,
                                     "PartitionForest": 0, "old_QRF": 0, "preprocessor": 0, "CatBoost": 0,
                                     "LAD_or_calibration": 0},
        "trusted_private_bundles_only": True, "zip_and_csv_reread": True, "receipts": receipts,
    })
    atomic_write_json(root / "completion.json", {
        "status": "READY_FOR_EXPLICIT_PLATFORM_SUBMISSIONS",
        "G0_engineering": "PASS", "G1_model_quality": "REPORTED_SEPARATELY_NOT_A_GATE",
        "candidates": list(CANDIDATES.values()), "platform_order": ["A", "B"],
        "platform_uploads": 0, "platform_budget": package["package_count"],
        "packages_sha256": file_sha256(root / "packages_frozen_before_feedback.json"),
        "cold_validation_sha256": file_sha256(root / "cold_validation.json"),
        "fit_counts_sha256": file_sha256(root / "fit_counts.json"), "test_targets_read": False,
        "automatic_upload": False, "desktop_writes": 0, "third_candidate_generated": False,
    })


def run(root: Path):
    register(root); prepare(root); develop(root); score(root); finalize(root); cold(root)
