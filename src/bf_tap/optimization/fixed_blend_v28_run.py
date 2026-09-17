"""Registered zero-fit v0.28 lifecycle for two fixed equal endpoint blends."""
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
from .fixed_blend_v28 import (
    CANDIDATE_A,
    CANDIDATE_B,
    IRON_DONOR,
    PARENT,
    POOLED_AGGREGATION,
    PRIMARY_AGGREGATION,
    TARGETS,
    TIME_DONOR,
    cancellation_diagnostic,
    compose_candidate,
    exposure_pooled_summary,
    macro_origin_summary,
    submission_bytes,
    validate_endpoint,
    verify_isolated_deltas,
)
from .qrf_partition_v26 import roundtrip_six
from .qrf_partition_v26_run import _replay as replay_qrf, _score_rows
from .structural_run import append_ledger


V26 = Path("local/runs/optimization-v0.26-qrf-partition-tests-r1")
V27 = Path("local/runs/optimization-v0.27-qrf-iron-and-leaf-recency-r1")
V26_WORKER = Path("workers/qrf_v026/worker.py")
V27_WORKER = Path("workers/qrf_v027/worker.py")
WORKER_PYTHON = Path("workers/qrf_v015/.venv/bin/python")
ORIGINS = range(6, 12)
SLOTS = range(6, 13)
CANDIDATES = {"A": CANDIDATE_A, "B": CANDIDATE_B}
PARENT_SOURCE_ID = "V26A_QRF_ABSOLUTE_SPLIT_TIME"
IRON_DONOR_SOURCE_ID = "V27I_ABS_QRF_DIRECT_IRON"
PARENT_RESULT = V26 / "submissions" / PARENT_SOURCE_ID / "result.csv"
PARENT_ZIP = V26 / "submissions" / PARENT_SOURCE_ID / "Luqhhh_bf_tap_predict_prelim.zip"
IRON_DONOR_RESULT = V27 / "submissions" / IRON_DONOR_SOURCE_ID / "result.csv"
TIME_DONOR_RESULT = V23 / "recovery/result.csv"


def registration():
    value = load_yaml("configs/optimization_v0_28/experiment.yaml")
    if value["branch"] != "optimization-v0.28-fixed-equal-blends" or value["candidates"] != CANDIDATES:
        raise ContractError("v0.28 branch/candidate registration differs")
    expected = {
        "new_base_model_fits": 0,
        "new_tree_fits": 0,
        "new_preprocessor_fits": 0,
        "new_calibration_or_weight_fits": 0,
        "fixed_blend_algorithms": 2,
        "new_candidate_packages": 2,
        "new_candidate_platform_tests": 2,
    }
    if value["budget"] != expected or value["platform_order"] != ["A", "B"]:
        raise ContractError("v0.28 budget/order registration differs")
    for key, target in (("candidate_A", "tap_iron"), ("candidate_B", "tap_time_len")):
        candidate = value[key]
        if candidate["target"] != target or candidate["weight_left"] != .5 or candidate["weight_right"] != .5 or candidate["scope"] != "all_samples":
            raise ContractError("v0.28 fixed equal all-row definition differs")
    return value


def _source_files():
    return [
        Path("configs/optimization_v0_28/experiment.yaml"),
        Path("configs/optimization_v0_28/access_scope.yaml"),
        Path("src/bf_tap/optimization/fixed_blend_v28.py"),
        Path("src/bf_tap/optimization/fixed_blend_v28_run.py"),
        Path("scripts/optimization_v28_fixed_blends.py"),
        Path("scripts/optimization_v28_cold_check.py"),
        Path("tests/test_optimization_v28_fixed_blends.py"),
        Path("docs/optimization_v0_28/PLAN.md"),
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
        return PARENT_RESULT if slot == 12 else V26 / "predictions" / f"{slot}_{PARENT_SOURCE_ID}.csv"
    if role == "iron_donor":
        return IRON_DONOR_RESULT if slot == 12 else V27 / "predictions" / f"{slot}_{IRON_DONOR_SOURCE_ID}.csv"
    if role == "time_donor":
        return TIME_DONOR_RESULT if slot == 12 else V22 / "predictions" / str(slot) / "V21_REPLAY.csv"
    raise ContractError("unknown v0.28 endpoint role")


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
        reasons = [value for value in (*CANDIDATES.values(), "COMPLETE_ENDPOINT_EQUAL_BLEND_v028") if value in text]
        if reasons:
            matches.append({"path": str(path), "sha256": file_sha256(path), "matches": reasons})
    return {"searched_manifest_count": len(manifests), "equivalent_matches": matches}


def _worker(script: Path, command: str, *arguments):
    return subprocess.check_output([str(WORKER_PYTHON), str(script), command, *map(str, arguments)], text=True)


def register(root: Path):
    config = registration()
    scope = load_yaml("configs/optimization_v0_28/access_scope.yaml")
    if root.exists():
        raise ContractError("never overwrite a v0.28 run")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ContractError("clean preregistered working tree required")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != config["branch"]:
        raise ContractError("registered v0.28 branch required")
    subprocess.run(["git", "merge-base", "--is-ancestor", config["base_commit"], "HEAD"], check=True)
    registry = _registry_search()
    if registry["equivalent_matches"]:
        raise ContractError("equivalent v0.28 experiment already exists")
    identities = config["source_identities"]
    checks = {
        "v26a_result_sha256": PARENT_RESULT,
        "v26a_zip_sha256": PARENT_ZIP,
        "v27i_result_sha256": IRON_DONOR_RESULT,
        "v21_result_sha256": TIME_DONOR_RESULT,
    }
    for key, path in checks.items():
        if file_sha256(path) != identities[key]:
            raise ContractError(f"registered source identity differs: {key}")
    evidence = {
        "v26_manifest": V26 / "manifest.json",
        "v26_completion": V26 / "completion.json",
        "v26_errors": V26 / "all_errors.csv",
        "v26_parent_result": PARENT_RESULT,
        "v26_parent_zip": PARENT_ZIP,
        "v27_manifest": V27 / "manifest.json",
        "v27_completion": V27 / "completion.json",
        "v27_errors": V27 / "all_errors.csv",
        "v27_iron_result": IRON_DONOR_RESULT,
        "v22_errors": V22 / "all_errors.csv",
        "v23_v21_result": TIME_DONOR_RESULT,
        "v23_cold_validation": V23 / "cold_validation.json",
        "protection": Path(scope["protection_contract"]),
    }
    manifest = {
        "registration": config,
        "scope": scope,
        "protection": load_yaml(scope["protection_contract"]),
        "evidence": file_identities(evidence),
        "sources": file_identities({str(path): path for path in _source_files()}),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_environment(),
        "worker_environment": json.loads(_worker(V26_WORKER, "environment")),
        "registry_search": registry,
        "holdout_consumed": True,
        "test_targets_read": False,
        "source_model_training": False,
        "new_fit_budget": 0,
        "platform_feedback_may_change_second_candidate": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    atomic_write_json(root / "manifest.json", manifest)
    append_ledger(scope, root)
    atomic_write_json(root / "registration.json", {
        "status": "TWO_ZERO_FIT_CANDIDATES_FROZEN_BEFORE_COMPOSITION",
        "candidates": CANDIDATES,
        "definitions": {key: config[f"candidate_{key}"] for key in CANDIDATES},
        "platform_order": ["A", "B"],
        "candidate_platform_budget": 2,
        "platform_uploads": 0,
        "definitions_may_change_after_A_feedback": False,
    })
    return manifest


def prepare(root: Path):
    if (root / "p0_complete.json").exists():
        raise ContractError("v0.28 P0 already complete")
    source_receipts = {}
    changed_rows = {}
    for slot in SLOTS:
        parent = _load_endpoint(slot, "parent")
        iron = _load_endpoint(slot, "iron_donor")
        time = _load_endpoint(slot, "time_donor")
        if len(parent) != len(iron) or len(parent) != len(time):
            raise ContractError("v0.28 endpoint row counts differ")
        # Composition performs independent ID alignment and target-isolation checks.
        a = compose_candidate(parent, iron, changed_target="tap_iron")
        b = compose_candidate(parent, time, changed_target="tap_time_len")
        changed_rows[str(slot)] = {
            "A": int(np.sum(a.pred_tap_iron.to_numpy() != parent.pred_tap_iron.to_numpy())),
            "B": int(np.sum(b.pred_tap_time_len.to_numpy() != parent.pred_tap_time_len.to_numpy())),
        }
        if changed_rows[str(slot)]["A"] == 0 or changed_rows[str(slot)]["B"] == 0:
            raise ContractError("registered candidate degenerates to parent at submission precision")
        source_receipts[str(slot)] = {
            role: {"path": str(_source_path(slot, role)), "sha256": file_sha256(_source_path(slot, role)), "rows": len(parent)}
            for role in ("parent", "iron_donor", "time_donor")
        }
    atomic_write_json(root / "p0_complete.json", {
        "status": "PASS_ZERO_FIT_SOURCE_AND_REGISTRY_AUDIT",
        "source_receipts": source_receipts,
        "changed_rows": changed_rows,
        "registry_search": read_json(root / "manifest.json")["registry_search"],
        "source_endpoint_fits": 0,
        "new_model_fits": 0,
        "new_preprocessor_fits": 0,
        "new_calibration_or_weight_fits": 0,
        "test_targets_read": False,
    })


def _write_prediction(path: Path, value: pd.DataFrame):
    if path.exists():
        raise ContractError("never overwrite a v0.28 prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(submission_bytes(value))


def _compose_slot(root: Path, slot: int):
    parent = _load_endpoint(slot, "parent")
    a = compose_candidate(parent, _load_endpoint(slot, "iron_donor"), changed_target="tap_iron")
    b = compose_candidate(parent, _load_endpoint(slot, "time_donor"), changed_target="tap_time_len")
    _write_prediction(root / "predictions" / f"{slot}_{CANDIDATE_A}.csv", a)
    _write_prediction(root / "predictions" / f"{slot}_{CANDIDATE_B}.csv", b)


def develop(root: Path):
    if (root / "development_complete.json").exists():
        raise ContractError("v0.28 development already complete")
    for slot in ORIGINS:
        _compose_slot(root, slot)
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
        "new_base_model_fits": 0,
        "new_tree_fits": 0,
        "new_preprocessor_fits": 0,
        "new_calibration_or_weight_fits": 0,
        "fixed_blends": 2,
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


def _diagnostics(candidates, references):
    rows = []
    definitions = {
        CANDIDATE_A: ("tap_iron", references[PARENT], references[IRON_DONOR]),
        CANDIDATE_B: ("tap_time_len", references[PARENT], references[TIME_DONOR]),
    }
    for candidate, (target, left, right) in definitions.items():
        rounded = candidates[candidate]
        for unit, rounded_part in rounded.groupby("unit", sort=False):
            keys = rounded_part.sample_id.astype(str).tolist()
            left_part = left.loc[left.unit == unit].set_index("sample_id").loc[keys]
            right_part = right.loc[right.unit == unit].set_index("sample_id").loc[keys]
            if not np.array_equal(left_part[target].to_numpy(), right_part[target].to_numpy()) or not np.array_equal(left_part[target].to_numpy(), rounded_part[target].to_numpy()):
                raise ContractError("diagnostic source labels differ")
            item = cancellation_diagnostic(
                rounded_part[target].to_numpy(dtype=float),
                left_part[f"pred_{target}"].to_numpy(dtype=float),
                right_part[f"pred_{target}"].to_numpy(dtype=float),
                rounded_part[f"pred_{target}"].to_numpy(dtype=float),
            )
            item.update({"candidate": candidate, "target": target, "unit": unit, "origin": rounded_part.origin.iloc[0], "horizon": rounded_part.horizon.iloc[0]})
            rows.append(item)
    return pd.DataFrame(rows)


def _write_report(root: Path, deltas: pd.DataFrame):
    lines = [
        "# optimization-v0.28 自动生成离线报告",
        "",
        "主口径为 `macro_origin_mean_wmape`。负 Δ 表示优于对应参照；平台名额与已消费历史评价分开记录。",
        "",
        "| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate, donor in ((CANDIDATE_A, IRON_DONOR), (CANDIDATE_B, TIME_DONOR)):
        for reference in (PARENT, donor):
            part = deltas.loc[(deltas.candidate == candidate) & (deltas.reference == reference)].set_index("scope")
            values = [part.loc[scope, "delta_E"] for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT")]
            lines.append(f"| {candidate} | {reference} | " + " | ".join(f"{value:+.8f}" for value in values) + " |")
    lines.extend(["", "G0 工程状态与 G1 已消费历史质量分开登记；本报告不构成平台结果。", ""])
    with (root / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def score(root: Path):
    if (root / "offline_assessment.json").exists():
        raise ContractError("never overwrite v0.28 scoring evidence")
    candidates = {candidate: _candidate_errors(root, candidate) for candidate in CANDIDATES.values()}
    references = {
        "V1": _reference_errors("V1"),
        PARENT: _source_errors(V26 / "all_errors.csv", PARENT_SOURCE_ID, PARENT),
        TIME_DONOR: _source_errors(V22 / "all_errors.csv", "V21_REPLAY", TIME_DONOR),
        IRON_DONOR: _source_errors(V27 / "all_errors.csv", IRON_DONOR_SOURCE_ID, IRON_DONOR),
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
    for candidate, donor in ((CANDIDATE_A, IRON_DONOR), (CANDIDATE_B, TIME_DONOR)):
        for reference in (PARENT, donor, "V1"):
            for scope in ("H1", "H2", "H3", "H4", "J", "DEV_LONG", "DEV_SHORT"):
                left, right = primary.loc[(candidate, scope)], primary.loc[(reference, scope)]
                rows.append({
                    "candidate": candidate, "reference": reference, "scope": scope,
                    "aggregation": PRIMARY_AGGREGATION, "E": float(left.E),
                    "delta_E": float(left.E - right.E),
                    "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                    "delta_wmape_time": float(left.wmape_time - right.wmape_time),
                })
    deltas = pd.DataFrame(rows)
    deltas.to_csv(root / "canonical_deltas.csv", index=False)
    pooled_index = pooled.set_index(["algorithm", "scope"])
    pooled_rows = []
    for candidate in candidates:
        for scope in ("H1", "H2", "H3", "H4", "DEV_LONG", "DEV_SHORT"):
            left, right = pooled_index.loc[(candidate, scope)], pooled_index.loc[(PARENT, scope)]
            pooled_rows.append({
                "candidate": candidate, "reference": PARENT, "scope": scope,
                "aggregation": POOLED_AGGREGATION,
                "delta_E": float(left.E - right.E),
                "delta_wmape_iron": float(left.wmape_iron - right.wmape_iron),
                "delta_wmape_time": float(left.wmape_time - right.wmape_time),
            })
    pd.DataFrame(pooled_rows).to_csv(root / "exposure_pooled_deltas.csv", index=False)
    diagnostics = _diagnostics(candidates, references)
    diagnostics.to_csv(root / "error_cancellation_diagnostics.csv", index=False)
    bootstrap = {
        candidate: {reference: week_intervals(all_errors, candidate, reference, 1000, 2026) for reference in (PARENT, donor, "V1")}
        for candidate, donor in ((CANDIDATE_A, IRON_DONOR), (CANDIDATE_B, TIME_DONOR))
    }
    atomic_write_json(root / "bootstrap.json", bootstrap)
    _write_report(root, deltas)
    atomic_write_json(root / "offline_assessment.json", {
        "status": "COMPLETED_CONSUMED_RETROSPECTIVE_NOT_A_PLATFORM_GATE",
        "primary_aggregation": PRIMARY_AGGREGATION,
        "diagnostic_aggregation": POOLED_AGGREGATION,
        "isolated_target_audit": isolation,
        "maximum_cancellation_identity_residual": float(diagnostics.identity_residual.max()),
        "maximum_absolute_rounding_effect": float(diagnostics.rounding_effect_abs_error_sum.abs().max()),
        "both_candidates_retain_preregistered_platform_slot": True,
    })


def finalize(root: Path):
    if not (root / "offline_assessment.json").exists() or (root / "packages_frozen_before_feedback.json").exists():
        raise ContractError("score v0.28 once before finalization")
    _compose_slot(root, 12)
    parent = _load_endpoint(12, "parent")
    receipts = {}
    for key, candidate in CANDIDATES.items():
        prediction = validate_endpoint(pd.read_csv(root / "predictions" / f"12_{candidate}.csv", dtype=str, keep_default_na=False), label=candidate)
        changed = "pred_tap_iron" if key == "A" else "pred_tap_time_len"
        unchanged = "pred_tap_time_len" if key == "A" else "pred_tap_iron"
        if prediction[unchanged].tolist() != parent[unchanged].tolist():
            raise ContractError("v0.28 unchanged final target string differs from V26A")
        if prediction[changed].tolist() == parent[changed].tolist():
            raise ContractError("v0.28 candidate is identical to parent")
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
                raise ContractError("v0.28 ZIP payload differs")
        reread = validate_endpoint(pd.read_csv(csv_path, dtype=str, keep_default_na=False), label="package")
        if len(reread) != registration()["test_a_rows"]:
            raise ContractError("v0.28 package test_a coverage differs")
        receipts[key] = {
            "candidate": candidate,
            "rows": len(reread),
            "result_sha256": file_sha256(csv_path),
            "zip_sha256": file_sha256(archive),
            "unchanged_target": unchanged,
            "unchanged_target_csv_strings_exact": True,
            "platform_uploads": 0,
            "platform_feedback": None,
        }
    atomic_write_json(root / "packages_frozen_before_feedback.json", {
        "status": "PASS_TWO_COMPLETE_PACKAGES_FROZEN",
        "order": ["A", "B"],
        "receipts": receipts,
        "definitions_may_change_after_A_feedback": False,
        "third_candidate_generated": False,
        "platform_upload_budget": 2,
        "platform_uploads": 0,
    })


def _assert_frame_equal(actual: pd.DataFrame, expected: pd.DataFrame, message: str):
    left = validate_endpoint(actual, label="actual")
    right = validate_endpoint(expected, label="expected")
    if not left.equals(right):
        raise ContractError(message)


def _cold_composition_checks(parent: pd.DataFrame, donor: pd.DataFrame, expected: pd.DataFrame, target: str):
    full = compose_candidate(parent, donor, changed_target=target)
    _assert_frame_equal(full, expected, "cold full blend differs")
    reversed_value = compose_candidate(parent.iloc[::-1], donor.iloc[::-1], changed_target=target)
    _assert_frame_equal(reversed_value, expected.iloc[::-1].reset_index(drop=True), "cold reverse blend differs")
    indices = list(range(len(parent)))[::7]
    subset = compose_candidate(parent.iloc[indices], donor.iloc[indices], changed_target=target)
    _assert_frame_equal(subset, expected.iloc[indices].reset_index(drop=True), "cold subset blend differs")
    pieces = []
    for index in np.array_split(np.arange(len(parent)), 5):
        pieces.append(compose_candidate(parent.iloc[index], donor.iloc[index], changed_target=target))
    _assert_frame_equal(pd.concat(pieces, ignore_index=True), expected, "cold chunked blend differs")
    single = compose_candidate(parent.iloc[[0]], donor.iloc[[0]], changed_target=target)
    _assert_frame_equal(single, expected.iloc[[0]].reset_index(drop=True), "cold single blend differs")


def cold(root: Path):
    if (root / "cold_validation.json").exists() or (root / "completion.json").exists():
        raise ContractError("never overwrite v0.28 cold/completion evidence")
    receipts = {"v26a": {}, "v27i": {}, "v21": {}}
    zero_fit = True
    for slot in SLOTS:
        cold_dir = root / "cold"
        cold_dir.mkdir(parents=True, exist_ok=True)
        v26_path = cold_dir / "v26a" / f"{slot}.npz"
        old_path = cold_dir / "v21" / f"{slot}.npz"
        v27_path = cold_dir / "v27i" / f"{slot}.npz"
        v26_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.parent.mkdir(parents=True, exist_ok=True)
        v27_path.parent.mkdir(parents=True, exist_ok=True)
        _worker(V26_WORKER, "cold", "--root", V26.resolve(), "--slot", slot, "--candidate", "A", "--output", v26_path.resolve())
        _worker(V26_WORKER, "old-cold", "--root", V26.resolve(), "--slot", slot, "--output", old_path.resolve())
        _worker(V27_WORKER, "cold-a", "--root", V27.resolve(), "--slot", slot, "--output", v27_path.resolve())
        for role, path in (("v26a", v26_path), ("v21", old_path), ("v27i", v27_path)):
            info = read_json(str(path) + ".json")
            receipts[role][str(slot)] = info
            zero_fit &= not any(info["zero_fit"].values())
        parent = _load_endpoint(slot, "parent")
        iron_donor = _load_endpoint(slot, "iron_donor")
        time_donor = _load_endpoint(slot, "time_donor")
        with np.load(v26_path, allow_pickle=False) as raw_parent:
            reconstructed_parent, _ = replay_qrf(V26, slot, raw_parent["median"])
        with np.load(old_path, allow_pickle=False) as raw_v21:
            reconstructed_time, _ = replay_qrf(V26, slot, raw_v21["median"])
        with np.load(v27_path, allow_pickle=False) as raw_iron:
            reconstructed_iron = reconstructed_parent.copy()
            reconstructed_iron["pred_tap_iron"] = roundtrip_six(raw_iron["median"])
        _assert_frame_equal(reconstructed_parent, parent, "cold V26A source reconstruction differs")
        _assert_frame_equal(reconstructed_iron, iron_donor, "cold V27I source reconstruction differs")
        _assert_frame_equal(reconstructed_time, time_donor, "cold V21 source reconstruction differs")
        for key, donor, target in (("A", iron_donor, "tap_iron"), ("B", time_donor, "tap_time_len")):
            expected = pd.read_csv(root / "predictions" / f"{slot}_{CANDIDATES[key]}.csv", dtype=str, keep_default_na=False)
            _cold_composition_checks(parent, donor, expected, target)
    if not zero_fit:
        raise ContractError("v0.28 cold source restoration attempted a fit")
    package = read_json(root / "packages_frozen_before_feedback.json")
    for key, candidate in CANDIDATES.items():
        folder = root / "submissions" / candidate
        csv_path = folder / "result.csv"
        archive = folder / "Luqhhh_bf_tap_predict_prelim.zip"
        receipt = package["receipts"][key]
        if file_sha256(csv_path) != receipt["result_sha256"] or file_sha256(archive) != receipt["zip_sha256"]:
            raise ContractError("v0.28 frozen package identity changed")
        with ZipFile(archive) as handle:
            if handle.namelist() != ["result.csv"] or handle.read("result.csv") != csv_path.read_bytes():
                raise ContractError("v0.28 cold ZIP reread differs")
    fit_attempts = {
        "CatBoost": 0,
        "RandomForestRegressor": 0,
        "ExtraTreesRegressor": 0,
        "PartitionForest": 0,
        "IronTargetForest": 0,
        "old_QRF": 0,
        "preprocessor": 0,
        "LAD_or_calibration": 0,
    }
    atomic_write_json(root / "cold_validation.json", {
        "status": "PASS",
        "all_slots": list(SLOTS),
        "source_model_cold_audit_completed_by_this_lifecycle": True,
        "source_checks": "full_reverse_chunk_subset_single_exact",
        "blend_checks": "full_reverse_chunk_subset_single_exact",
        "fit_attempts_during_cold": fit_attempts,
        "trusted_private_bundles_only": True,
        "zip_and_csv_reread": True,
        "receipts": receipts,
    })
    atomic_write_json(root / "fit_counts.json", {
        "new_base_model_attempted": 0,
        "new_base_model_completed": 0,
        "new_tree_attempted": 0,
        "new_tree_completed": 0,
        "new_preprocessor_attempted": 0,
        "new_preprocessor_completed": 0,
        "new_calibration_or_weight_attempted": 0,
        "new_calibration_or_weight_completed": 0,
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
