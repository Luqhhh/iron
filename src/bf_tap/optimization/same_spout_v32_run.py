"""Registered zero-fit v0.32 lifecycle for same-spout OOB responses."""
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
from .oob_pool_v31_run import (
    V26,
    V27,
    V28,
    V29,
    V30,
    V21_RESULT,
    V26_RESULT,
    V28_RESULT,
    V29I_RESULT,
    V29I_ZIP,
    V29T_RESULT,
    V29T_ZIP,
    V30A_RESULT,
    V30A_ZIP,
    _candidate_errors,
    _edge_identity_checks,
    _load_endpoint,
    _source_errors,
)
from .qrf_partition_v26_run import _replay as replay_qrf, _score_rows
from .same_spout_v32 import (
    ATTACHMENT_PROTOCOL,
    CANDIDATE_A,
    CANDIDATE_B,
    PARENT,
    POOLED_AGGREGATION,
    PRIMARY_AGGREGATION,
    SAME_SPOUT_PROTOCOL,
    TARGETS,
    V1,
    V21_REPLAY,
    V26A_PARENT,
    V28I,
    V29I,
    V29T,
    compose_iron_candidate,
    compose_time_candidate,
    exposure_pooled_summary,
    macro_origin_summary,
    verify_isolated_deltas,
)
from .structural_run import append_ledger


V32_WORKER = Path("workers/qrf_v032/worker.py")
WORKER_PYTHON = Path("workers/qrf_v015/.venv/bin/python")
ORIGINS = range(6, 12)
SLOTS = range(6, 13)
CANDIDATES = {"A": CANDIDATE_A, "B": CANDIDATE_B}
TARGET_META = {"A": ("tap_iron", "tonne"), "B": ("tap_time_len", "minutes")}


def registration():
    value = load_yaml("configs/optimization_v0_32/experiment.yaml")
    if value["branch"] != "optimization-v0.32-same-spout-oob-responses" or value["candidates"] != CANDIDATES:
        raise ContractError("v0.32 branch/candidate registration differs")
    if value["base_commit"] != "656be40fc19283972f6c757917d8caf460d75cd4":
        raise ContractError("v0.32 base commit registration differs")
    if value["protocol"] != SAME_SPOUT_PROTOCOL or value["oob_attachment_protocol"] != ATTACHMENT_PROTOCOL:
        raise ContractError("v0.32 protocol registration differs")
    rule = value["member_rule"]
    expected = {
        "known_query_nonempty_intersection": "use_same_spout_intersection",
        "known_query_empty_intersection": "use_original_selected_set_exact",
        "missing_query": "bypass_conditioning",
        "unknown_query": "bypass_conditioning",
        "numeric_token_coercion": "forbidden",
        "full_leaf_same_spout_search_after_empty_intersection": "forbidden",
        "neighboring_leaf_members": "forbidden",
        "skip_tree": "forbidden",
        "minimum_same_spout_count": 1,
        "new_set_nonempty_subset_of_original": "required",
    }
    if any(rule.get(key) != expected_value for key, expected_value in expected.items()):
        raise ContractError("v0.32 member rule registration differs")
    aggregation = value["aggregation_rule"]
    if (
        aggregation["tree_mass"] != "equal"
        or aggregation["within_tree_member_mass"] != "equal"
        or aggregation["occurrence_pooling_v031"] != "forbidden"
        or aggregation["median"] != "original_lower_weighted_median"
        or aggregation["exact_fallback_receives_conditioned_members"] != "required"
    ):
        raise ContractError("v0.32 equal-tree aggregation registration differs")
    expected_budget = {
        "new_model_or_tree_fits": 0,
        "new_tree_fits": 0,
        "new_preprocessing_lad_beta_lambda_bias_fits": 0,
        "conditional_read_protocol_A_development": 6,
        "conditional_read_protocol_A_final": 1,
        "conditional_read_protocol_A_total": 7,
        "conditional_read_protocol_B_development": 6,
        "conditional_read_protocol_B_final": 1,
        "conditional_read_protocol_B_total": 7,
        "certified_oob_attachment_reuse_development": 12,
        "certified_oob_attachment_reuse_final": 2,
        "certified_oob_attachment_reuse_total": 14,
        "aligned_spout_vectors_total": 14,
        "new_candidate_packages": 2,
        "new_candidate_platform_tests": 2,
        "automatic_uploads": 0,
        "restore_uploads": 0,
        "desktop_writes": 0,
        "public_pushes": 0,
        "third_candidate": 0,
    }
    if value["budget"] != expected_budget or value["platform_order"] != ["A", "B"]:
        raise ContractError("v0.32 budget/order registration differs")
    if value["platform_feedback_may_change_second_candidate"] is not False:
        raise ContractError("v0.32 B must remain frozen across A feedback")
    return value


def _source_files():
    return [
        Path("configs/optimization_v0_32/experiment.yaml"),
        Path("configs/optimization_v0_32/access_scope.yaml"),
        Path("src/bf_tap/optimization/same_spout_v32.py"),
        Path("src/bf_tap/optimization/same_spout_v32_run.py"),
        Path("scripts/optimization_v32_same_spout.py"),
        Path("scripts/optimization_v32_cold_check.py"),
        Path("tests/test_optimization_v32_same_spout.py"),
        Path("docs/optimization_v0_32/PLAN.md"),
        V32_WORKER,
        Path("workers/qrf_v032/same_spout.py"),
        Path("workers/qrf_v032/test_same_spout.py"),
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


def _worker(command: str, *arguments):
    return subprocess.check_output([str(WORKER_PYTHON), str(V32_WORKER), command, *map(str, arguments)], text=True)


def _registry_search():
    manifests = sorted(Path("local/runs").glob("**/*manifest*.json"))
    matches = []
    for path in manifests:
        text = path.read_text(encoding="utf-8", errors="ignore")
        reasons = [item for item in (*CANDIDATES.values(), SAME_SPOUT_PROTOCOL) if item in text]
        if reasons:
            matches.append({"path": str(path), "sha256": file_sha256(path), "matches": reasons})
    completed, incomplete = [], []
    for match in matches:
        root = Path(match["path"]).parent
        (completed if (root / "completion.json").is_file() else incomplete).append(match)
    return {"searched_manifest_count": len(manifests), "equivalent_matches": completed, "incomplete_equivalent_attempts": incomplete}


def register(root: Path):
    config = registration()
    scope = load_yaml("configs/optimization_v0_32/access_scope.yaml")
    if root.exists():
        raise ContractError("never overwrite a v0.32 run")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ContractError("clean preregistered working tree required")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != config["branch"]:
        raise ContractError("registered v0.32 branch required")
    subprocess.run(["git", "merge-base", "--is-ancestor", config["base_commit"], "HEAD"], check=True)
    registry = _registry_search()
    if registry["equivalent_matches"]:
        raise ContractError("equivalent completed v0.32 experiment already exists")

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
            raise ContractError(f"registered v0.32 source identity differs: {key}")

    evidence = {
        "v30_manifest": V30 / "manifest.json", "v30_completion": V30 / "completion.json",
        "v30_cold_validation": V30 / "cold_validation.json", "v30_packages_frozen": V30 / "packages_frozen_before_feedback.json",
        "v30_feedback": V30 / "platform_feedback_user_reported.json", "v30a_result": V30A_RESULT, "v30a_zip": V30A_ZIP,
        "v29_manifest": V29 / "manifest.json", "v29_completion": V29 / "completion.json",
        "v29_cold_validation": V29 / "cold_validation.json", "v29_packages_frozen": V29 / "packages_frozen_before_feedback.json",
        "v29_development_frozen": V29 / "development_predictions_frozen.json",
        "v29i_result": V29I_RESULT, "v29i_zip": V29I_ZIP, "v29t_result": V29T_RESULT, "v29t_zip": V29T_ZIP,
        "v28_manifest": V28 / "manifest.json", "v28_completion": V28 / "completion.json", "v28_errors": V28 / "all_errors.csv",
        "v28_parent_result": V28_RESULT, "v27_manifest": V27 / "manifest.json", "v27_completion": V27 / "completion.json",
        "v26_manifest": V26 / "manifest.json", "v26_completion": V26 / "completion.json", "v26_errors": V26 / "all_errors.csv",
        "v26a_result": V26_RESULT, "v26_models_complete_development": V26 / "development_models_complete.json",
        "v26_models_complete_final": V26 / "final_models_complete.json", "v27_models_complete_development": V27 / "development_models_complete.json",
        "v27_models_complete_final": V27 / "final_models_complete.json", "v22_errors": V22 / "all_errors.csv",
        "v23_v21_result": V21_RESULT, "protection": Path(scope["protection_contract"]),
    }
    for candidate in ("A", "B"):
        for slot in SLOTS:
            evidence[f"v29_attachment_{candidate}_{slot}"] = V29 / "oob_attachments" / candidate / f"{slot}.npz"
            evidence[f"v29_attachment_{candidate}_{slot}_metadata"] = V29 / "oob_attachments" / candidate / f"{slot}.npz.json"
            evidence[f"v29_prediction_{candidate}_{slot}"] = V29 / "worker_predictions" / candidate / f"{slot}.npz"
            evidence[f"v29_prediction_{candidate}_{slot}_metadata"] = V29 / "worker_predictions" / candidate / f"{slot}.npz.json"
    missing = [name for name, path in evidence.items() if not Path(path).is_file()]
    if missing:
        raise ContractError(f"missing registered v0.32 evidence: {missing}")

    worker_environment = json.loads(_worker("environment"))
    worker_sources = json.loads(_worker("identity"))
    worker_registration = json.loads(_worker("registration"))
    if worker_registration["same_spout_protocol"] != SAME_SPOUT_PROTOCOL:
        raise ContractError("v0.32 worker protocol differs")
    manifest = {
        "registration": config, "scope": scope, "protection": load_yaml(scope["protection_contract"]),
        "evidence": file_identities(evidence), "sources": file_identities({str(path): path for path in _source_files()}),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_environment(), "worker_environment": worker_environment,
        "worker_sources": worker_sources, "worker_registration": worker_registration,
        "candidate_targets": {
            "A": {"target": "tap_iron", "source": "V27I_ABS_QRF_DIRECT_IRON", "attachment": "v029_A"},
            "B": {"target": "tap_time_len", "source": "V26A_QRF_ABSOLUTE_SPLIT_TIME", "attachment": "v029_B"},
        },
        "source_runs": {name: str(path.resolve()) for name, path in {"v26": V26, "v27": V27, "v28": V28, "v29": V29, "v30": V30}.items()},
        "registry_search": registry, "holdout_consumed": True, "test_targets_read": False,
        "new_fit_budget": 0, "platform_feedback_may_change_second_candidate": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    atomic_write_json(root / "manifest.json", manifest)
    append_ledger(scope, root)
    atomic_write_json(root / "registration.json", {
        "status": "TWO_ZERO_FIT_SAME_SPOUT_CANDIDATES_FROZEN_BEFORE_DERIVATION",
        "candidates": CANDIDATES, "member_rule": config["member_rule"], "aggregation_rule": config["aggregation_rule"],
        "platform_order": ["A", "B"], "candidate_platform_budget": 2, "platform_uploads": 0,
        "definitions_may_change_after_A_feedback": False, "third_candidate_allowed": False,
    })
    return manifest


def prepare(root: Path):
    if (root / "p0_complete.json").exists():
        raise ContractError("v0.32 P0 already complete")
    audits = {}
    for slot in SLOTS:
        audits[str(slot)] = {"identities": _edge_identity_checks(slot)}
        for candidate in CANDIDATES:
            receipt = json.loads(_worker("audit", "--root", root.resolve(), "--slot", slot, "--candidate", candidate))
            if any(receipt["zero_fit"].values()):
                raise ContractError("v0.32 P0 attempted a fit")
            if not receipt["attachment_rederived_from_bootstrap_exact"] or receipt["attachment_trees_checked"] != 256:
                raise ContractError("v0.32 source attachment audit differs")
            audits[str(slot)][candidate] = receipt
    atomic_write_json(root / "p0_complete.json", {
        "status": "PASS_ZERO_FIT_SOURCE_FOREST_OOB_ATTACHMENT_AND_SPOUT_IDENTITY_AUDIT",
        "audits": audits, "registry_search": read_json(root / "manifest.json")["registry_search"],
        "new_model_fits": 0, "new_tree_fits": 0, "new_preprocessor_fits": 0,
        "new_calibration_fits": 0, "test_targets_read": False,
    })


def _write_prediction(path: Path, frame: pd.DataFrame):
    if path.exists():
        raise ContractError("never overwrite a v0.32 prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(submission_bytes(frame))


def _worker_prediction(root: Path, slot: int, candidate: str):
    path = root / "worker_predictions" / candidate / f"{slot}.npz"
    if not path.is_file():
        raise ContractError(f"missing v0.32 worker prediction: {path}")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if not {"ids", "query_spout", "median", "legacy_median"}.issubset(arrays):
        raise ContractError("v0.32 worker prediction fields differ")
    return path, arrays


def _compose_slot(root: Path, slot: int, *, derive: bool, write: bool):
    parent, v26a = _load_endpoint(slot, "parent"), _load_endpoint(slot, "v26a")
    outputs = {}
    for candidate in ("A", "B"):
        output = root / "worker_predictions" / candidate / f"{slot}.npz"
        if derive and not output.is_file():
            _worker("derive-predict", "--root", root.resolve(), "--slot", slot, "--candidate", candidate, "--output", output.resolve())
        path, arrays = _worker_prediction(root, slot, candidate)
        if arrays["ids"].tolist() != parent.sample_id.tolist():
            raise ContractError("v0.32 worker prediction IDs differ from parent")
        outputs[candidate] = arrays

    a = compose_iron_candidate(parent, v26a, outputs["A"]["median"])
    legacy_a = compose_iron_candidate(parent, v26a, outputs["A"]["legacy_median"])
    if not legacy_a.equals(parent) or a.pred_tap_time_len.tolist() != parent.pred_tap_time_len.tolist():
        raise ContractError("v0.32 A switch-back/isolation identity failed")

    replayed, replay_audit = replay_qrf(V26, slot, outputs["B"]["median"])
    b = compose_time_candidate(parent, replayed.pred_tap_time_len.to_numpy(dtype=float))
    legacy_replayed, legacy_replay_audit = replay_qrf(V26, slot, outputs["B"]["legacy_median"])
    legacy_b = compose_time_candidate(parent, legacy_replayed.pred_tap_time_len.to_numpy(dtype=float))
    if not legacy_b.equals(parent) or b.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist():
        raise ContractError("v0.32 B switch-back/isolation identity failed")

    audit = {
        "A": {"source": V26A_PARENT, "rule": "same_spout_selected_member_conditioning", "weight_each": 0.5,
              "unchanged_time_exact": True, "legacy_equal_tree_switch_back_exact": True},
        "B": {"rule": "same_spout_selected_member_conditioning", "replay": replay_audit,
              "legacy_replay": legacy_replay_audit, "unchanged_iron_exact": True,
              "gate_source": "recomputed_original_v015_effective_neighbors", "gate_uses_conditioned_support": False,
              "legacy_equal_tree_switch_back_exact": True},
    }
    if write:
        _write_prediction(root / "predictions" / f"{slot}_{CANDIDATE_A}.csv", a)
        _write_prediction(root / "predictions" / f"{slot}_{CANDIDATE_B}.csv", b)
        atomic_write_json(root / "predictions" / f"{slot}_audit.json", audit)
        for candidate in ("A", "B"):
            receipt = read_json(str(root / "worker_predictions" / candidate / f"{slot}.npz") + ".json")
            atomic_write_json(root / "condition_views" / f"{candidate}_{slot}.json", {
                "slot": slot, "candidate": CANDIDATES[candidate], "target": TARGET_META[candidate][0],
                "source_tree_count": 256, "attachment_path": receipt["attachment_path"],
                "attachment_sha256": receipt["attachment_sha256"], "training_spout": receipt["training_spout"],
                "preprocessor_vocabulary": receipt["preprocessor_vocabulary"],
                "legacy_equal_tree_switch_back_exact": receipt["legacy_equal_tree_switch_back_exact"],
                "changed_from_legacy_count": receipt["changed_from_legacy_count"], "diagnostics": receipt["diagnostics"],
            })
    return {"parent": parent, "A": a, "B": b}


def develop(root: Path):
    if (root / "development_complete.json").exists():
        raise ContractError("v0.32 development already complete")
    for slot in ORIGINS:
        _compose_slot(root, slot, derive=True, write=True)
    prediction_files = {f"{slot}_{candidate}": root / "predictions" / f"{slot}_{candidate}.csv" for slot in ORIGINS for candidate in CANDIDATES.values()}
    view_files = {f"{candidate}_{slot}": root / "condition_views" / f"{candidate}_{slot}.json" for slot in ORIGINS for candidate in CANDIDATES}
    attachments = {f"{candidate}_{slot}": V29 / "oob_attachments" / candidate / f"{slot}.npz" for slot in ORIGINS for candidate in CANDIDATES}
    atomic_write_json(root / "development_predictions_frozen.json", {
        "status": "BOTH_CANDIDATES_AND_SOURCE_ATTACHMENTS_FROZEN_BEFORE_SCORING",
        "predictions": file_identities(prediction_files), "condition_views": file_identities(view_files),
        "certified_source_attachments": file_identities(attachments), "platform_feedback_seen": False,
    })
    atomic_write_json(root / "development_complete.json", {
        "status": "PASS_G0_DEVELOPMENT_SAME_SPOUT_RESPONSES_FROZEN",
        "new_model_fits": 0, "new_tree_fits": 0, "new_preprocessor_fits": 0,
        "oob_attachments_reused": 12, "conditional_read_protocol_A": 6, "conditional_read_protocol_B": 6,
    })


def _write_report(root: Path, deltas: pd.DataFrame):
    lines = [
        "# optimization-v0.32 自动生成离线报告", "",
        "主口径为 `macro_origin_mean_wmape`。负 Δ 表示优于参照；历史标签已消费，OOB 叶响应不是独立验证集。", "",
        "| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in CANDIDATES.values():
        for reference in deltas.reference.unique():
            part = deltas.loc[(deltas.candidate == candidate) & (deltas.reference == reference)].set_index("scope")
            values = [part.loc[scope, "delta_E"] for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")]
            lines.append(f"| {candidate} | {reference} | " + " | ".join(f"{value:+.8f}" for value in values) + " |")
    lines.extend(["", "G0、已消费历史 G1 与平台反馈分别登记；跨铁口质量下降是实现不变量，不是提分证据。", ""])
    with (root / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def score(root: Path):
    if (root / "offline_assessment.json").exists():
        raise ContractError("never overwrite v0.32 scoring evidence")
    candidates = {candidate: _candidate_errors(root, candidate) for candidate in CANDIDATES.values()}
    references = {
        V1: _reference_errors(V1), V21_REPLAY: _reference_errors(V21_REPLAY),
        V28I: _source_errors(V28 / "all_errors.csv", "V28I_CB_QRF_EQUAL_BLEND", V28I),
        V29I: _source_errors(V29 / "all_errors.csv", V29I, V29I),
        V29T: _source_errors(V29 / "all_errors.csv", V29T, V29T),
        PARENT: _source_errors(V30 / "all_errors.csv", PARENT, PARENT),
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
                rows.append({"candidate": candidate, "reference": reference, "scope": scope,
                             "aggregation": PRIMARY_AGGREGATION, "E": float(left.E), "delta_E": float(left.E - right.E),
                             "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                             "delta_wmape_time": float(left.wmape_time - right.wmape_time)})
    deltas = pd.DataFrame(rows)
    deltas.to_csv(root / "canonical_deltas.csv", index=False)
    pooled_index = pooled.set_index(["algorithm", "scope"])
    pooled_rows = []
    for candidate in candidates:
        for reference in references:
            for scope in ("H1", "H2", "H3", "H4", "DEV_LONG", "DEV_SHORT"):
                left, right = pooled_index.loc[(candidate, scope)], pooled_index.loc[(reference, scope)]
                pooled_rows.append({"candidate": candidate, "reference": reference, "scope": scope,
                                    "aggregation": POOLED_AGGREGATION, "delta_E": float(left.E - right.E),
                                    "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                                    "delta_wmape_time": float(left.wmape_time - right.wmape_time)})
    pd.DataFrame(pooled_rows).to_csv(root / "exposure_pooled_deltas.csv", index=False)
    all_errors.groupby(["candidate", "origin", "horizon", "spout_no"], as_index=False).agg(
        n=("sample_id", "size"), signed_bias_iron=("error_tap_iron", "mean"), signed_bias_time=("error_tap_time_len", "mean"),
    ).to_csv(root / "signed_bias_by_spout.csv", index=False)
    atomic_write_json(root / "bootstrap.json", {
        candidate: {reference: week_intervals(all_errors, candidate, reference, 1000, 2026) for reference in references}
        for candidate in candidates
    })
    atomic_write_json(root / "condition_diagnostics.json", {
        candidate: {str(slot): read_json(root / "condition_views" / f"{candidate}_{slot}.json") for slot in ORIGINS}
        for candidate in CANDIDATES
    })
    _write_report(root, deltas)
    atomic_write_json(root / "offline_assessment.json", {
        "status": "COMPLETED_CONSUMED_RETROSPECTIVE_NOT_A_PLATFORM_GATE",
        "primary_aggregation": PRIMARY_AGGREGATION, "diagnostic_aggregation": POOLED_AGGREGATION,
        "isolated_target_audit": isolation, "OOB_is_not_an_independent_validation_set": True,
        "cross_spout_mass_reduction_is_not_accuracy_evidence": True,
        "both_candidates_retain_preregistered_platform_slot": True,
    })


def finalize(root: Path):
    if not (root / "offline_assessment.json").is_file() or (root / "packages_frozen_before_feedback.json").exists():
        raise ContractError("score v0.32 once before finalization")
    composed = _compose_slot(root, 12, derive=True, write=True)
    parent = composed["parent"]
    receipts = {}
    for key, candidate in CANDIDATES.items():
        prediction = validate_endpoint(pd.read_csv(root / "predictions" / f"12_{candidate}.csv", dtype=str, keep_default_na=False), label=candidate)
        if len(prediction) != registration()["test_a_rows"]:
            raise ContractError("v0.32 semantic package row count differs")
        unchanged = prediction.pred_tap_time_len.tolist() == parent.pred_tap_time_len.tolist() if key == "A" else prediction.pred_tap_iron.tolist() == parent.pred_tap_iron.tolist()
        if not unchanged:
            raise ContractError("v0.32 unchanged target differs from V30A")
        no_op = prediction.equals(parent)
        if no_op:
            receipts[key] = {"candidate": candidate, "status": "NO_OP_FINAL_PAYLOAD", "rows": len(prediction), "package_created": False}
            continue
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
                raise ContractError("v0.32 ZIP payload differs")
        receipts[key] = {"candidate": candidate, "status": "FROZEN", "rows": len(prediction), "package_created": True,
                         "result_sha256": file_sha256(csv_path), "zip_sha256": file_sha256(archive),
                         "unchanged_target": "time_exact_V30A" if key == "A" else "iron_exact_V30A",
                         "platform_uploads": 0, "platform_feedback": None}
    package_count = sum(item["package_created"] for item in receipts.values())
    atomic_write_json(root / "packages_frozen_before_feedback.json", {
        "status": "PASS_VALID_NON_NOOP_PACKAGES_FROZEN", "order": ["A", "B"], "receipts": receipts,
        "package_count": package_count, "no_op_count": 2 - package_count,
        "definitions_may_change_after_A_feedback": False, "third_candidate_generated": False,
        "platform_upload_budget": package_count, "platform_uploads": 0, "automatic_upload": False,
    })


def _assert_frame(actual, expected, message):
    left = validate_endpoint(actual, label="actual").reset_index(drop=True)
    right = validate_endpoint(expected, label="expected").reset_index(drop=True)
    if not left.equals(right):
        raise ContractError(message)


def cold(root: Path):
    if (root / "cold_validation.json").exists() or (root / "completion.json").exists():
        raise ContractError("never overwrite v0.32 cold/completion evidence")
    receipts = {"A": {}, "B": {}}
    zero_fit = True
    for slot in SLOTS:
        for candidate in ("A", "B"):
            output = root / "cold" / candidate / f"{slot}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            receipt = json.loads(_worker("cold", "--root", root.resolve(), "--slot", slot, "--candidate", candidate, "--output", output.resolve()))
            receipts[candidate][str(slot)] = receipt
            zero_fit &= not any(receipt["zero_fit"].values())
        composed = _compose_slot(root, slot, derive=False, write=False)
        _assert_frame(composed["A"], pd.read_csv(root / "predictions" / f"{slot}_{CANDIDATE_A}.csv", dtype=str, keep_default_na=False), "v0.32 cold A differs")
        _assert_frame(composed["B"], pd.read_csv(root / "predictions" / f"{slot}_{CANDIDATE_B}.csv", dtype=str, keep_default_na=False), "v0.32 cold B differs")
    if not zero_fit:
        raise ContractError("v0.32 cold restoration attempted a fit")
    package = read_json(root / "packages_frozen_before_feedback.json")
    for key, candidate in CANDIDATES.items():
        receipt = package["receipts"][key]
        if not receipt["package_created"]:
            continue
        folder = root / "submissions" / candidate
        csv_path, archive = folder / "result.csv", folder / "Luqhhh_bf_tap_predict_prelim.zip"
        if file_sha256(csv_path) != receipt["result_sha256"] or file_sha256(archive) != receipt["zip_sha256"]:
            raise ContractError("v0.32 frozen package identity changed")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != csv_path.read_bytes():
                raise ContractError("v0.32 cold ZIP reread differs")
    fit_counts = {
        "new_base_model_attempted": 0, "new_base_model_completed": 0, "new_tree_attempted": 0, "new_tree_completed": 0,
        "new_preprocessor_attempted": 0, "new_preprocessor_completed": 0, "new_calibration_attempted": 0,
        "new_calibration_completed": 0, "certified_oob_attachment_reads_A": 7,
        "certified_oob_attachment_reads_B": 7, "aligned_spout_vectors": 14,
        "new_oob_attachment_created": 0, "new_candidate_packages": package["package_count"],
        "platform_uploads": 0, "third_candidate_generated": False,
    }
    atomic_write_json(root / "fit_counts.json", fit_counts)
    atomic_write_json(root / "cold_validation.json", {
        "status": "PASS", "all_slots": list(SLOTS),
        "source_forests_responses_bootstrap_and_v029_selected_members_unchanged": True,
        "training_and_query_spout_ids_aligned": True,
        "same_spout_members_nonempty_subsets_or_original_fallback": True,
        "original_oob_and_same_spout_fallbacks_separate": True,
        "missing_and_unknown_queries_bypass": True,
        "source_checks": "full_reverse_chunk_subset_single_exact",
        "conditioned_checks": "full_reverse_chunk_subset_single_exact",
        "legacy_equal_tree_switch_back_exact": True,
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
        "cold_validation_sha256": file_sha256(root / "cold_validation.json"), "fit_counts_sha256": file_sha256(root / "fit_counts.json"),
        "test_targets_read": False, "automatic_upload": False, "desktop_writes": 0, "public_pushes": 0,
        "third_candidate_generated": False,
    })


def run(root: Path):
    register(root)
    prepare(root)
    develop(root)
    score(root)
    finalize(root)
    cold(root)
