"""Real-data local M1-M4 OOF and frozen-selection confirmation.

Run with python -m bf_tap.models.optimization_v02.workflow --data-config ... --output ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ...artifacts import (atomic_write_json, build_inference_source_contract,
                         code_identity, file_identities, file_sha256,
                         runtime_environment, stable_digest, verify_file_identities)
from ...config import load_yaml
from ...exceptions import ContractError
from ...metrics import score_predictions
from ...protection import load_protection_policy, read_access_ledger
from ...splits import split_from_config, validate_split_definitions
from .config import validate_optimization_candidate_config, validate_optimization_common_config
from .derive import freeze_selection, replay_selection
from .folds import run_fold
from .inputs import load_inputs
from .m4_ensemble.model import POLICY as M4_POLICY
from .oof import score_oof_predictions


def _write_predictions(directory, predictions):
    directory.mkdir(exist_ok=False)
    for name, frame in predictions.items():
        frame.to_csv(directory / f"{name}.csv", index=False)


def _error_slices(actual, predictions, directory):
    records = []
    for name, pred in predictions.items():
        merged = actual.merge(pred, on=["fold_id", "sample_id"], validate="one_to_one")
        for target in ("tap_iron", "tap_time_len"):
            merged[f"abs_error_{target}"] = (merged[f"pred_{target}"] - merged[target]).abs()
        merged.to_csv(directory / f"{name}_errors.csv", index=False)
        for fold, part in merged.groupby("fold_id", sort=True):
            for spout, subgroup in part.groupby("spout_no", sort=True):
                metrics = score_predictions(subgroup, subgroup[[
                    "sample_id", "pred_tap_iron", "pred_tap_time_len"]])
                records.append({"candidate_id": name, "fold_id": fold, "spout_no": str(spout),
                                "rows": len(subgroup), "E": metrics["loss"],
                                "iron_wmape": metrics["iron"]["wmape"],
                                "time_wmape": metrics["time"]["wmape"]})
    pd.DataFrame(records).to_csv(directory / "errors_by_fold_spout.csv", index=False)


def _confirmation(inputs, destination, metadata, frozen):
    folds_config = load_yaml(Path(metadata["config_root"]) / "validation.yaml")
    folds = [fold for fold in validate_split_definitions(folds_config["folds"], inputs.protection)
             if fold.kind == "development"]
    if {fold.id for fold in folds} != {"DEV_LONG", "DEV_SHORT"}:
        raise ContractError("confirmation requires frozen DEV_LONG/DEV_SHORT")
    summaries = {}
    for fold in folds:
        actual, raw, _ = run_fold(inputs, fold, destination / "confirmation" / fold.id,
                                 metadata, [frozen["m2_half_life_days"]])
        derived = replay_selection(raw, frozen)
        _write_predictions(destination / "confirmation" / fold.id / "selected", derived)
        metrics = {name: score_oof_predictions(actual, pred)["pooled"]
                   for name, pred in {**raw, **derived}.items()}
        summaries[fold.id] = metrics
    return {"selection_sha256": frozen["selection_sha256"], "retuned": False,
            "scope": "stability_confirmation_not_independent_holdout", "folds": summaries}


def run_local_oof(data_config, output, *, confirm=True, config_root="configs",
                  ledger_path="local/manifests/protected_access.json"):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        config_root = Path(config_root).resolve()
        common = load_yaml(config_root / "optimization/v0.2/common.yaml")
        validate_optimization_common_config(common)
        configs = {name: load_yaml(config_root / f"optimization/v0.2/{name}.yaml")
                   for name in ("m1_blend", "m2_recency", "m3_residual", "m4_ensemble")}
        for config in configs.values():
            validate_optimization_candidate_config(config)
        protection = load_protection_policy(config_root / "protection.yaml")
        folds = validate_split_definitions(common["folds"], protection)
        ledger_before = read_access_ledger(ledger_path, protection)
        paths = load_yaml(data_config)["paths"]
        source_names = ("train_samples", "tap_history_train", "operation_hourly",
                        "burden_change", "data_dictionary")
        source_identities = file_identities({name: paths[name] for name in source_names})
        code = code_identity(Path.cwd())
        atomic_write_json(destination / "resolved_config.json",
                          {"common": common, "candidates": configs, "m4_activation_policy": M4_POLICY,
                           "confirm": confirm})
        atomic_write_json(destination / "run_state.json", {
            "status": "RUNNING", "sources": source_identities, "code_identity": code,
            "ledger_before": ledger_before, "protected_labels_read": False,
        })
        inputs = load_inputs(data_config, config_root, max(fold.eval_end for fold in folds))
        metadata = {
            "code_identity": code, "environment": runtime_environment(),
            "lockfile_sha256": file_sha256("uv.lock"), "config_root": str(config_root),
            "inference_source_contract": build_inference_source_contract(
                source_identities, semantic_contract_sha256=inputs.contract_digests["semantic_contract_sha256"]),
        }
        atomic_write_json(destination / "environment.json", metadata["environment"])
        atomic_write_json(destination / "input_audit.json", inputs.consistency)
        half_lives = configs["m2_recency"]["parameters"]["half_life_days"]
        all_actual, collected, audits = [], {}, {}
        for number, fold in enumerate(folds):
            actual, predictions, audit = run_fold(
                inputs, fold, destination / "folds" / fold.id, metadata, half_lives,
                determinism=number == 0)
            all_actual.append(actual)
            audits[fold.id] = audit
            for name, pred in predictions.items():
                collected.setdefault(name, []).append(pred)
        actual = pd.concat(all_actual, ignore_index=True)
        predictions = {name: pd.concat(parts, ignore_index=True) for name, parts in collected.items()}
        scores = {name: score_oof_predictions(actual, pred) for name, pred in predictions.items()}
        frozen, derived = freeze_selection(actual, predictions, half_lives, configs["m1_blend"])
        # Create the selection snapshot before reading any confirmation labels/results.
        atomic_write_json(destination / "frozen_selection.json", frozen)
        selection_file_digest = file_sha256(destination / "frozen_selection.json")
        predictions.update(derived)
        scores.update({name: score_oof_predictions(actual, pred) for name, pred in derived.items()})
        _write_predictions(destination / "oof", predictions)
        actual.to_csv(destination / "oof/actual.csv", index=False)
        atomic_write_json(destination / "oof_metrics.json", scores)
        atomic_write_json(destination / "oof_quality.json", frozen["quality"])
        _error_slices(actual, predictions, destination / "oof")
        summary = []
        for name, metrics in scores.items():
            for scope, values in {"pooled": metrics["pooled"], **metrics["folds"]}.items():
                summary.append({"candidate_id": name, "scope": scope, "E": values["loss"],
                                "iron_wmape": values["iron"]["wmape"], "time_wmape": values["time"]["wmape"]})
        pd.DataFrame(summary).to_csv(destination / "comparison.csv", index=False)
        confirmation = None
        if confirm:
            print("Selection frozen; starting DEV_LONG/DEV_SHORT confirmation.", flush=True)
            confirmation_inputs = load_inputs(data_config, config_root, protection.development_label_end_exclusive)
            confirmation = _confirmation(confirmation_inputs, destination, metadata, frozen)
            atomic_write_json(destination / "confirmation.json", confirmation)
        if file_sha256(destination / "frozen_selection.json") != selection_file_digest:
            raise ContractError("selection changed during confirmation")
        verify_file_identities(source_identities)
        if code_identity(Path.cwd())["source_snapshot_sha256"] != code["source_snapshot_sha256"]:
            raise ContractError("source configuration or implementation changed during run")
        ledger_after = read_access_ledger(ledger_path, protection)
        if stable_digest(ledger_after) != stable_digest(ledger_before):
            raise ContractError("protected access ledger changed during development run")
        quality = {}
        for name, evidence in frozen["quality"].items():
            # Do not silently turn the qualitative worst-fold/confirmation review into a new threshold.
            quality[name] = {
                "oof_quantitative_gates_pass": evidence["pass"],
                "status": ("REVIEW_REQUIRED" if evidence["pass"] else "FAIL"),
                "reason": ("worst_fold_and_confirmation_require_review" if evidence["pass"]
                           else "registered_oof_gate_failed"),
            }
        final = {
            "engineering_status": "PASS", "model_quality_status": (
                "REVIEW_REQUIRED" if any(q["oof_quantitative_gates_pass"] for q in quality.values()) else "FAIL"),
            "candidate_quality": quality, "protected_labels_read": False,
            "protected_ledger_unchanged": True, "oof_rows": len(actual),
            "selection_sha256": frozen["selection_sha256"], "m4_status": frozen["m4"]["status"],
            "confirmation_completed": confirmation is not None,
            "raw_models_roundtrip_verified": True,
        }
        atomic_write_json(destination / "final_status.json", final)
        print(f"Complete: {destination}; G0={final['engineering_status']} G1={final['model_quality_status']}", flush=True)
        return destination
    except Exception as exc:
        atomic_write_json(destination / "final_status.json",
                          {"engineering_status": "FAILED", "error_type": type(exc).__name__,
                           "error": str(exc), "protected_labels_read": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--ledger", default="local/manifests/protected_access.json")
    parser.add_argument("--no-confirm", action="store_true")
    args = parser.parse_args()
    run_local_oof(args.data_config, args.output, confirm=not args.no_confirm,
                  config_root=args.config_root, ledger_path=args.ledger)


if __name__ == "__main__":
    main()
