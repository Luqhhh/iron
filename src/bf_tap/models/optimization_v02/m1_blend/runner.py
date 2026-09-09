from __future__ import annotations

from pathlib import Path

import pandas as pd

from ....artifacts import (
    atomic_write_json,
    code_identity,
    file_identities,
    verify_file_identities,
)
from ....config import load_yaml
from ....exceptions import ContractError
from ....io import read_csv, read_development_labels
from ....protection import load_protection_policy
from ..config import (
    validate_optimization_candidate_config,
    validate_optimization_common_config,
)
from ..run import initialize_optimization_run
from .model import select_blend_weights


def _prediction_paths(
    baseline_oof_dir: Path,
    folds: list[dict[str, object]],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for fold in folds:
        fold_id = str(fold["id"])
        fold_dir = baseline_oof_dir / fold_id
        result[f"{fold_id}_catboost"] = fold_dir / "predictions.csv"
        result[f"{fold_id}_B1"] = fold_dir / "predictions_B1.csv"
    return result


def _read_fold_prediction(path: Path, fold_id: str) -> pd.DataFrame:
    frame = read_csv(
        path,
        required_columns=["sample_id", "pred_tap_iron", "pred_tap_time_len"],
    )[["sample_id", "pred_tap_iron", "pred_tap_time_len"]]
    frame.insert(0, "fold_id", fold_id)
    return frame


def run_m1_selection(
    *,
    labels_path: str | Path,
    baseline_oof_dir: str | Path,
    common_config_path: str | Path,
    candidate_config_path: str | Path,
    protection_policy_path: str | Path,
    output_root: str | Path,
    run_id: str,
) -> Path:
    common = load_yaml(common_config_path)
    candidate = load_yaml(candidate_config_path)
    validate_optimization_common_config(common)
    validate_optimization_candidate_config(candidate)
    if candidate["candidate_id"] != "M1_BLEND":
        raise ContractError("M1 runner requires the M1_BLEND candidate config")
    policy = load_protection_policy(protection_policy_path)
    configured_boundary = pd.Timestamp(common["development_label_end_exclusive"])
    if configured_boundary != policy.development_label_end_exclusive:
        raise ContractError("optimization and protection label boundaries differ")

    baseline_root = Path(baseline_oof_dir)
    prediction_paths = _prediction_paths(baseline_root, common["folds"])
    manifest_paths = {"train_samples": Path(labels_path), **prediction_paths}
    manifest = file_identities(manifest_paths)
    destination = initialize_optimization_run(
        output_root,
        run_id,
        common_config=common,
        candidate_config=candidate,
        code_identity=code_identity(Path.cwd()),
        data_manifest=manifest,
    )

    safe_end = max(pd.Timestamp(fold["eval_end"]) for fold in common["folds"])
    try:
        labels = read_development_labels(
            labels_path,
            requested_end=safe_end,
            protection_policy=policy,
        )
        actual_parts: list[pd.DataFrame] = []
        catboost_parts: list[pd.DataFrame] = []
        b1_parts: list[pd.DataFrame] = []
        for fold in common["folds"]:
            fold_id = str(fold["id"])
            eval_start = pd.Timestamp(fold["eval_start"])
            eval_end = pd.Timestamp(fold["eval_end"])
            mask = (labels["reference_time"] >= eval_start) & (
                labels["reference_time"] < eval_end
            )
            actual = labels.loc[
                mask, ["sample_id", "tap_iron", "tap_time_len"]
            ].copy()
            if actual.empty:
                raise ContractError(f"{fold_id}: empty OOF labels")
            actual.insert(0, "fold_id", fold_id)
            actual_parts.append(actual)
            catboost_parts.append(
                _read_fold_prediction(prediction_paths[f"{fold_id}_catboost"], fold_id)
            )
            b1_parts.append(
                _read_fold_prediction(prediction_paths[f"{fold_id}_B1"], fold_id)
            )

        actual_oof = pd.concat(actual_parts, ignore_index=True)
        catboost_oof = pd.concat(catboost_parts, ignore_index=True)
        b1_oof = pd.concat(b1_parts, ignore_index=True)
        parameters = candidate["parameters"]
        selection = select_blend_weights(
            actual_oof,
            catboost_oof,
            b1_oof,
            weights=parameters["weights"],
            tie_tolerance=float(parameters["tie_tolerance"]),
        )
        selection_payload = {
            "weights": selection["weights"],
            "metrics": selection["metrics"],
            "grid": selection["grid"],
        }
        atomic_write_json(destination / "selection.json", selection_payload)
        selection["predictions"].to_csv(
            destination / "oof_predictions.csv",
            index=False,
            encoding="utf-8",
        )
        verify_file_identities(manifest)
        atomic_write_json(
            destination / "final_status.json",
            {
                "status": "PASS",
                "candidate_id": "M1_BLEND",
                "safe_label_end_exclusive": str(safe_end),
            },
        )
    except Exception as exc:
        atomic_write_json(
            destination / "final_status.json",
            {
                "status": "FAILED",
                "candidate_id": "M1_BLEND",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    return destination
