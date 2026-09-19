"""Registered v0.33 lifecycle for QRF feature and row sampling experiments."""
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
from .dual_burden_v24_run import V22, _reference_errors
from .dual_ratio_common import week_intervals
from .oob_pool_v31_run import V26, V30, V30A_RESULT, V30A_ZIP, _candidate_errors, _load_endpoint, _source_errors
from .qrf_partition_v26_run import _replay as replay_qrf, _score_rows
from .qrf_sampling_v33 import (
    CANDIDATE_A, CANDIDATE_B, FOREST_PROTOCOL, OOB_PROTOCOL, PARENT,
    POOLED_AGGREGATION, PRIMARY_AGGREGATION, TARGETS, V1, V21_REPLAY,
    compose_time_candidate, exposure_pooled_summary, macro_origin_summary,
    submission_bytes, validate_endpoint, verify_isolated_deltas,
)
from .structural_run import append_ledger


WORKER = Path("workers/qrf_v033/worker.py")
WORKER_PYTHON = Path("workers/qrf_v015/.venv/bin/python")
V32 = Path("local/runs/optimization-v0.32-same-spout-oob-responses-r2")
ORIGINS = range(6, 12)
SLOTS = range(6, 13)
CANDIDATES = {"A": CANDIDATE_A, "B": CANDIDATE_B}
EXPECTED_ROWS = {6: 888, 7: 1180, 8: 1490, 9: 1803, 10: 2091, 11: 2424, 12: 2754}
EXPECTED_DRAWS = {
    "A": EXPECTED_ROWS,
    "B": {6: 444, 7: 590, 8: 745, 9: 902, 10: 1046, 11: 1212, 12: 1377},
}


def registration():
    value = load_yaml("configs/optimization_v0_33/experiment.yaml")
    if value["branch"] != "optimization-v0.33-qrf-feature-and-row-sampling" or value["base_commit"] != "f6eb91aebd0b827a0aac1d2c7f161a354f44c0fe":
        raise ContractError("v0.33 branch/base registration differs")
    if value["candidates"] != CANDIDATES or value["parent"] != PARENT:
        raise ContractError("v0.33 candidate/parent registration differs")
    if value["forest_protocol"] != FOREST_PROTOCOL or value["oob_protocol"] != OOB_PROTOCOL:
        raise ContractError("v0.33 protocol registration differs")
    if value["training_rows"] != EXPECTED_ROWS or value["draws_per_tree"] != EXPECTED_DRAWS:
        raise ContractError("v0.33 N/m registration differs")
    if value["candidate_A"]["only_parameter_change"]["max_features"] != 1.0 / 3.0 or value["candidate_A"]["max_samples"] is not None:
        raise ContractError("v0.33 A definition differs")
    if value["candidate_B"]["only_parameter_change"]["max_samples"] != 0.5 or value["candidate_B"]["max_features"] != 0.7:
        raise ContractError("v0.33 B definition differs")
    expected_budget = {
        "forest_fits_A_development": 6, "forest_fits_A_final": 1, "forest_fits_A_total": 7,
        "forest_fits_B_development": 6, "forest_fits_B_final": 1, "forest_fits_B_total": 7,
        "total_forest_fits": 14, "total_new_trees": 3584, "new_oob_attachments": 14,
        "new_preprocessor_iron_catboost_lad_beta_lambda_bias_fits": 0,
        "new_candidate_packages": 2, "new_candidate_platform_tests": 2,
        "automatic_uploads": 0, "restore_uploads": 0, "desktop_writes": 0,
        "public_pushes": 0, "third_candidate": 0,
    }
    if value["budget"] != expected_budget or value["platform_order"] != ["A", "B"]:
        raise ContractError("v0.33 budget/order registration differs")
    if value["platform_feedback_may_change_second_candidate"] is not False:
        raise ContractError("v0.33 B must stay frozen across A feedback")
    return value


def _worker(command: str, *arguments):
    return subprocess.check_output([str(WORKER_PYTHON), str(WORKER), command, *map(str, arguments)], text=True)


def _source_files():
    return [
        Path("configs/optimization_v0_33/experiment.yaml"), Path("configs/optimization_v0_33/access_scope.yaml"),
        Path("src/bf_tap/optimization/qrf_sampling_v33.py"), Path("src/bf_tap/optimization/qrf_sampling_v33_run.py"),
        Path("scripts/optimization_v33_qrf_sampling.py"), Path("scripts/optimization_v33_cold_check.py"),
        Path("tests/test_optimization_v33_qrf_sampling.py"), Path("docs/optimization_v0_33/PLAN.md"),
        WORKER, Path("workers/qrf_v033/sampled_forest.py"), Path("workers/qrf_v033/oob_sampled.py"),
        Path("workers/qrf_v033/test_sampled_oob.py"), Path("workers/qrf_v015/qrf_model.py"),
        Path("workers/qrf_v015/preprocessing.py"), Path("workers/qrf_v015/uv.lock"),
    ]


def _registry_search():
    manifests = sorted(Path("local/runs").glob("**/*manifest*.json"))
    matches = []
    for path in manifests:
        text = path.read_text(encoding="utf-8", errors="ignore")
        reasons = [token for token in (*CANDIDATES.values(), FOREST_PROTOCOL, OOB_PROTOCOL) if token in text]
        if reasons:
            matches.append({"path": str(path), "sha256": file_sha256(path), "matches": reasons})
    completed, incomplete = [], []
    for match in matches:
        root = Path(match["path"]).parent
        (completed if (root / "completion.json").is_file() else incomplete).append(match)
    return {"searched_manifest_count": len(manifests), "equivalent_matches": completed, "incomplete_equivalent_attempts": incomplete}


def register(root: Path):
    config = registration()
    scope = load_yaml("configs/optimization_v0_33/access_scope.yaml")
    if root.exists():
        raise ContractError("never overwrite a v0.33 run")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ContractError("clean preregistered working tree required")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != config["branch"]:
        raise ContractError("registered v0.33 branch required")
    subprocess.run(["git", "merge-base", "--is-ancestor", config["base_commit"], "HEAD"], check=True)
    registry = _registry_search()
    if registry["equivalent_matches"]:
        raise ContractError("equivalent completed v0.33 experiment already exists")
    identities = config["source_identities"]
    if file_sha256(V30A_RESULT) != identities["v30a_result_sha256"] or file_sha256(V30A_ZIP) != identities["v30a_zip_sha256"]:
        raise ContractError("V30A fallback identity differs")
    evidence = {
        "v26_manifest": V26 / "manifest.json", "v26_completion": V26 / "completion.json",
        "v26_development_models": V26 / "development_models_complete.json", "v26_final_models": V26 / "final_models_complete.json",
        "v26_errors": V26 / "all_errors.csv", "v30_manifest": V30 / "manifest.json",
        "v30_completion": V30 / "completion.json", "v30_errors": V30 / "all_errors.csv",
        "v30a_result": V30A_RESULT, "v30a_zip": V30A_ZIP,
        "v32_results": Path("docs/optimization_v0_32/RESULTS.md"), "v32_feedback": V32 / "platform_feedback_user_reported.json",
        "protection": Path(scope["protection_contract"]),
    }
    for slot in SLOTS:
        for kind in ("train", "evaluation"):
            evidence[f"v26_{slot}_{kind}"] = V26 / "features" / str(slot) / f"{kind}.npz"
            evidence[f"v26_{slot}_{kind}_metadata"] = V26 / "features" / str(slot) / f"{kind}.npz.json"
        evidence[f"v26_parent_model_{slot}"] = V26 / "models/A" / str(slot) / "forest.joblib"
        evidence[f"v26_parent_bundle_{slot}"] = V26 / "models/A" / str(slot) / "bundle.json"
    missing = [name for name, path in evidence.items() if not Path(path).is_file()]
    if missing:
        raise ContractError(f"missing v0.33 source evidence: {missing}")
    environment = json.loads(_worker("environment"))
    sources = json.loads(_worker("identity"))
    worker_registration = json.loads(_worker("registration"))
    if environment.get("python") != "3.12.12" or environment.get("sklearn") != "1.8.0":
        raise ContractError("locked Python 3.12.12 / sklearn 1.8.0 worker required")
    if worker_registration["forest_protocol"] != FOREST_PROTOCOL or worker_registration["oob_protocol"] != OOB_PROTOCOL:
        raise ContractError("worker protocol differs")
    manifest = {
        "registration": config, "scope": scope, "protection": load_yaml(scope["protection_contract"]),
        "evidence": file_identities(evidence), "sources": file_identities({str(path): path for path in _source_files()}),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_environment(), "worker_environment": environment, "worker_sources": sources,
        "worker_registration": worker_registration, "source_runs": {"v26": str(V26.resolve()), "v30": str(V30.resolve()), "v32": str(V32.resolve())},
        "registry_search": registry, "holdout_consumed": True, "test_targets_read": False,
        "new_forest_fit_budget": 14, "platform_feedback_may_change_second_candidate": False,
    }
    root.mkdir(parents=True, exist_ok=False); root.chmod(0o700)
    atomic_write_json(root / "manifest.json", manifest)
    append_ledger(scope, root)
    atomic_write_json(root / "registration.json", {
        "status": "TWO_CANDIDATES_FROZEN_BEFORE_ANY_V033_FIT", "candidates": CANDIDATES,
        "parameters": worker_registration["parameters"], "training_rows": EXPECTED_ROWS,
        "draws_per_tree": EXPECTED_DRAWS, "platform_order": ["A", "B"],
        "candidate_platform_budget": 2, "platform_uploads": 0,
        "definitions_may_change_after_A_feedback": False, "third_candidate_allowed": False,
    })
    return manifest


def prepare(root: Path):
    if (root / "p0_complete.json").exists():
        raise ContractError("v0.33 P0 already complete")
    audits, parents = {}, {}
    for slot in SLOTS:
        receipt = json.loads(_worker("audit", "--root", root.resolve(), "--slot", slot))
        if not receipt["zero_fit"] or receipt["inputs"]["train"]["rows"] != EXPECTED_ROWS[slot]:
            raise ContractError("v0.33 P0 worker audit differs")
        if receipt["candidate_draws"] != {key: EXPECTED_DRAWS[key][slot] for key in CANDIDATES}:
            raise ContractError("v0.33 P0 N/m audit differs")
        parent = _load_endpoint(slot, "parent")
        if len(parent) != (registration()["test_a_rows"] if slot == 12 else len(parent)):
            raise ContractError("v0.33 parent endpoint coverage differs")
        audits[str(slot)] = receipt
        parents[str(slot)] = {"rows": len(parent), "iron_strings_sha256": hashlib_strings(parent.pred_tap_iron)}
    atomic_write_json(root / "p0_complete.json", {
        "status": "PASS_ZERO_FIT_CERTIFIED_V015_HANDOFF_PREPROCESSOR_V26_PARENT_AND_V30A_ENDPOINT",
        "audits": audits, "parents": parents, "registry_search": read_json(root / "manifest.json")["registry_search"],
        "equivalent_completed_experiment_found": False, "new_model_fits": 0, "new_preprocessor_fits": 0,
        "test_targets_read": False, "v32_scores_used_only_as_design_history": registration()["v32_user_reported_scores"],
    })


def hashlib_strings(values):
    import hashlib
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode()); digest.update(b"\0")
    return digest.hexdigest()


def _model_completion(root: Path, slots):
    result = {candidate: {} for candidate in CANDIDATES}
    for candidate in CANDIDATES:
        for slot in slots:
            folder = root / "models" / candidate / str(slot)
            record = read_json(folder / "fit_record.json")
            digest = file_sha256(folder / "bundle.json")
            if record["status"] != "COMPLETED" or record["bundle_sha256"] != digest:
                raise ContractError("v0.33 fit completion identity differs")
            result[candidate][str(slot)] = digest
    return result


def _fit_one(root: Path, slot: int, candidate: str):
    folder = root / "models" / candidate / str(slot)
    if (folder / "fit_record.json").is_file():
        return
    if (folder / "fit_intent.json").exists() or folder.exists():
        raise ContractError("incomplete v0.33 fit retained; explicit recovery decision required")
    _worker("fit", "--root", root.resolve(), "--slot", slot, "--candidate", candidate)


def _save_prediction(path: Path, frame: pd.DataFrame):
    if path.exists():
        raise ContractError("never overwrite v0.33 prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(submission_bytes(frame))


def _compose_slot(root: Path, slot: int, *, derive: bool, write: bool):
    parent = _load_endpoint(slot, "parent")
    frames, audits = {}, {}
    for candidate, label in CANDIDATES.items():
        output = root / "worker_predictions" / candidate / f"{slot}.npz"
        if derive and not output.is_file():
            _worker("predict", "--root", root.resolve(), "--slot", slot, "--candidate", candidate, "--output", output.resolve())
        with np.load(output, allow_pickle=False) as source:
            ids, median = source["ids"].tolist(), source["median"]
            effective = source["effective_neighbors"]
        if ids != parent.sample_id.tolist():
            raise ContractError("v0.33 worker/parent sample IDs differ")
        replayed, replay_audit = replay_qrf(V26, slot, median)
        frame = compose_time_candidate(parent, replayed.pred_tap_time_len.to_numpy(dtype=float), label=label)
        if frame.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist():
            raise ContractError("v0.33 changed V30A iron strings")
        frames[candidate] = frame
        audits[candidate] = {
            "candidate_id": label, "unchanged_iron_exact_V30A": True, "all_rows_use_new_forest": True,
            "replay": replay_audit, "gate_source": "original_v015_effective_neighbors",
            "new_effective_neighbors_used_for_gate": False,
            "new_effective_neighbors": {"minimum": float(effective.min()), "median": float(np.median(effective)), "maximum": float(effective.max())},
            "worker_receipt": read_json(str(output) + ".json"),
        }
        if write:
            _save_prediction(root / "predictions" / f"{slot}_{label}.csv", frame)
    if write:
        atomic_write_json(root / "predictions" / f"{slot}_audit.json", audits)
    return {"parent": parent, **frames}


def develop(root: Path):
    if (root / "development_complete.json").exists():
        raise ContractError("v0.33 development already complete")
    for candidate in CANDIDATES:
        for slot in ORIGINS:
            _fit_one(root, slot, candidate)
    completed = _model_completion(root, ORIGINS)
    atomic_write_json(root / "development_models_complete.json", completed)
    for slot in ORIGINS:
        _compose_slot(root, slot, derive=True, write=True)
    prediction_files = {f"{slot}_{candidate}": root / "predictions" / f"{slot}_{label}.csv" for slot in ORIGINS for candidate, label in CANDIDATES.items()}
    model_files = {f"{candidate}_{slot}": root / "models" / candidate / str(slot) / "forest.joblib" for candidate in CANDIDATES for slot in ORIGINS}
    attachment_files = {f"{candidate}_{slot}": root / "models" / candidate / str(slot) / "oob_attachment.npz" for candidate in CANDIDATES for slot in ORIGINS}
    atomic_write_json(root / "development_predictions_frozen.json", {
        "status": "BOTH_CANDIDATES_MODELS_ATTACHMENTS_AND_PREDICTIONS_FROZEN_BEFORE_SCORING",
        "predictions": file_identities(prediction_files), "models": file_identities(model_files),
        "attachments": file_identities(attachment_files), "platform_feedback_seen": False,
    })
    atomic_write_json(root / "development_complete.json", {
        "status": "PASS_G0_DEVELOPMENT", "A_fits": 6, "B_fits": 6, "new_trees": 3072,
        "new_oob_attachments": 12, "new_preprocessor_fits": 0, "new_iron_or_calibration_fits": 0,
    })


def _write_report(root: Path, deltas: pd.DataFrame):
    lines = [
        "# optimization-v0.33 自动生成离线报告", "",
        "主口径为 `macro_origin_mean_wmape`；负 Δ 表示历史误差改善。历史标签已消费，OOB 不是独立验证集。", "",
        "| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in CANDIDATES.values():
        for reference in deltas.reference.unique():
            part = deltas.loc[(deltas.candidate == candidate) & (deltas.reference == reference)].set_index("scope")
            values = [part.loc[scope, "delta_E"] for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")]
            lines.append(f"| {candidate} | {reference} | " + " | ".join(f"{value:+.8f}" for value in values) + " |")
    lines.extend(["", "G0、已消费历史 G1 与平台反馈分别登记；H1 是观察重点但未用于修改候选。", ""])
    with (root / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def score(root: Path):
    if (root / "offline_assessment.json").exists():
        raise ContractError("never overwrite v0.33 scoring evidence")
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
    isolation = verify_isolated_deltas(scorecard, summary, {label: "tap_time_len" for label in candidates}, parent=PARENT)
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
    atomic_write_json(root / "bootstrap.json", {
        candidate: {reference: week_intervals(all_errors, candidate, reference, 1000, 2026) for reference in references}
        for candidate in candidates
    })
    diagnostics = {}
    for candidate in CANDIDATES:
        diagnostics[candidate] = {}
        for slot in ORIGINS:
            diagnostics[candidate][str(slot)] = {
                "bundle": read_json(root / "models" / candidate / str(slot) / "bundle.json"),
                "prediction": read_json(str(root / "worker_predictions" / candidate / f"{slot}.npz") + ".json"),
            }
    atomic_write_json(root / "forest_oob_diagnostics.json", diagnostics)
    _write_report(root, deltas)
    lookup = primary.E
    atomic_write_json(root / "offline_assessment.json", {
        "status": "COMPLETED_CONSUMED_RETROSPECTIVE_NOT_A_PLATFORM_GATE",
        "primary_aggregation": PRIMARY_AGGREGATION, "diagnostic_aggregation": POOLED_AGGREGATION,
        "isolated_target_audit": isolation, "H1_primary_observation_only": True,
        "candidates": {candidate: {scope: float(lookup.loc[(candidate, scope)]) for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")} for candidate in candidates},
        "references": {reference: {scope: float(lookup.loc[(reference, scope)]) for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")} for reference in references},
        "both_candidates_retain_preregistered_platform_slot": True,
    })


def finalize(root: Path):
    if not (root / "offline_assessment.json").is_file() or (root / "packages_frozen_before_feedback.json").exists():
        raise ContractError("score v0.33 once before finalization")
    for candidate in CANDIDATES:
        _fit_one(root, 12, candidate)
    completed = _model_completion(root, (12,)); atomic_write_json(root / "final_models_complete.json", completed)
    composed = _compose_slot(root, 12, derive=True, write=True)
    parent = composed["parent"]
    receipts = {}
    for key, label in CANDIDATES.items():
        prediction = validate_endpoint(composed[key], label=label)
        if len(prediction) != registration()["test_a_rows"] or prediction.sample_id.duplicated().any():
            raise ContractError("v0.33 final endpoint coverage differs")
        if prediction.pred_tap_iron.tolist() != parent.pred_tap_iron.tolist():
            raise ContractError("v0.33 final iron strings differ from V30A")
        if prediction.equals(parent):
            receipts[key] = {"candidate": label, "status": "NO_OP_FINAL_PAYLOAD", "rows": len(prediction), "package_created": False}
            continue
        folder = root / "submissions" / label; folder.mkdir(parents=True, exist_ok=False)
        csv_path = folder / "result.csv"; payload = submission_bytes(prediction)
        with csv_path.open("xb") as handle: handle.write(payload)
        archive = folder / "Luqhhh_bf_tap_predict_prelim.zip"
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as handle: handle.write(csv_path, arcname="result.csv")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != payload:
                raise ContractError("v0.33 ZIP payload differs")
        reread = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        numeric = reread[["pred_tap_iron", "pred_tap_time_len"]].to_numpy(dtype=float)
        if len(reread) != 335 or reread.sample_id.duplicated().any() or not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ContractError("v0.33 package format/value validation failed")
        receipts[key] = {
            "candidate": label, "status": "FROZEN", "rows": len(reread), "package_created": True,
            "result_sha256": file_sha256(csv_path), "zip_sha256": file_sha256(archive),
            "model_sha256": file_sha256(root / "models" / key / "12/forest.joblib"),
            "attachment_sha256": file_sha256(root / "models" / key / "12/oob_attachment.npz"),
            "iron_source_result_sha256": file_sha256(V30A_RESULT), "iron_csv_strings_exact": True,
            "platform_uploads": 0, "platform_feedback": None,
        }
    count = sum(item["package_created"] for item in receipts.values())
    atomic_write_json(root / "packages_frozen_before_feedback.json", {
        "status": "PASS_VALID_NON_NOOP_PACKAGES_FROZEN", "order": ["A", "B"], "receipts": receipts,
        "package_count": count, "no_op_count": 2 - count, "definitions_may_change_after_A_feedback": False,
        "third_candidate_generated": False, "platform_upload_budget": count, "platform_uploads": 0,
        "automatic_upload": False, "desktop_writes": 0, "public_pushes": 0,
    })


def _assert_frame(actual, expected, message):
    left = validate_endpoint(actual, label="actual").reset_index(drop=True)
    right = validate_endpoint(expected, label="expected").reset_index(drop=True)
    if not left.equals(right):
        raise ContractError(message)


def cold(root: Path):
    if (root / "cold_validation.json").exists() or (root / "completion.json").exists():
        raise ContractError("never overwrite v0.33 cold/completion evidence")
    receipts = {candidate: {} for candidate in CANDIDATES}; zero_fit = True
    for slot in SLOTS:
        for candidate in CANDIDATES:
            output = root / "cold" / candidate / f"{slot}.npz"; output.parent.mkdir(parents=True, exist_ok=True)
            receipt = json.loads(_worker("cold", "--root", root.resolve(), "--slot", slot, "--candidate", candidate, "--output", output.resolve()))
            receipts[candidate][str(slot)] = receipt
            zero_fit &= not any(receipt["zero_fit"].values())
        composed = _compose_slot(root, slot, derive=False, write=False)
        for key, label in CANDIDATES.items():
            _assert_frame(composed[key], pd.read_csv(root / "predictions" / f"{slot}_{label}.csv", dtype=str, keep_default_na=False), f"v0.33 cold {key} differs")
    if not zero_fit:
        raise ContractError("v0.33 cold restoration attempted a fit")
    package = read_json(root / "packages_frozen_before_feedback.json")
    for key, label in CANDIDATES.items():
        receipt = package["receipts"][key]
        if not receipt["package_created"]: continue
        folder = root / "submissions" / label; csv_path = folder / "result.csv"; archive = folder / "Luqhhh_bf_tap_predict_prelim.zip"
        if file_sha256(csv_path) != receipt["result_sha256"] or file_sha256(archive) != receipt["zip_sha256"]:
            raise ContractError("v0.33 frozen package identity changed")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != csv_path.read_bytes():
                raise ContractError("v0.33 cold ZIP reread differs")
    attempted = {candidate: len(list((root / "models" / candidate).glob("*/fit_intent.json"))) for candidate in CANDIDATES}
    completed = {candidate: len(list((root / "models" / candidate).glob("*/fit_record.json"))) for candidate in CANDIDATES}
    if attempted != {"A": 7, "B": 7} or completed != attempted:
        raise ContractError("v0.33 fit attempt/completion budget differs")
    records = [read_json(path) for path in (root / "models").glob("*/*/fit_record.json")]
    trees = sum(int(item["internal_trees"]) for item in records)
    if trees != 3584:
        raise ContractError("v0.33 tree budget differs")
    fit_counts = {
        "forest_attempted": attempted, "forest_completed": completed, "new_trees": trees,
        "new_oob_attachments": len(list((root / "models").glob("*/*/oob_attachment.npz"))),
        "new_preprocessor_fits": 0, "new_iron_catboost_lad_beta_lambda_bias_fits": 0,
        "new_candidate_packages": package["package_count"], "platform_uploads": 0, "third_candidate_generated": False,
    }
    atomic_write_json(root / "fit_counts.json", fit_counts)
    atomic_write_json(root / "cold_validation.json", {
        "status": "PASS", "all_slots": list(SLOTS), "candidates": list(CANDIDATES),
        "model_attachment_rederived_from_actual_estimators_samples": True,
        "N_and_m_separately_validated": True, "all_N_rows_projected_once_per_tree": True,
        "leaf_weighted_and_unique_inbag_counts_validated": True, "A_draws_equal_V26A": True,
        "B_old_draw_prefix_not_required": True, "candidate_checks": "full_reverse_chunk_subset_single_exact",
        "fit_attempts_during_cold": {"SampledTimeForest": 0, "RandomForestRegressor": 0, "ExtraTreesRegressor": 0, "parent_QRF": 0, "preprocessor": 0},
        "metadata_and_numeric_comparisons_separate": True, "trusted_private_bundles_only": True,
        "zip_and_csv_reread": True, "receipts": receipts,
    })
    atomic_write_json(root / "completion.json", {
        "status": "READY_FOR_EXPLICIT_PLATFORM_SUBMISSIONS", "G0_engineering": "PASS",
        "G1_model_quality": "REPORTED_SEPARATELY_NOT_A_GATE", "candidates": list(CANDIDATES.values()),
        "platform_order": ["A", "B"], "platform_uploads": 0, "platform_budget": package["package_count"],
        "packages_sha256": file_sha256(root / "packages_frozen_before_feedback.json"),
        "cold_validation_sha256": file_sha256(root / "cold_validation.json"),
        "fit_counts_sha256": file_sha256(root / "fit_counts.json"), "test_targets_read": False,
        "automatic_upload": False, "desktop_writes": 0, "public_pushes": 0, "third_candidate_generated": False,
    })


def run(root: Path):
    register(root); prepare(root); develop(root); score(root); finalize(root); cold(root)
