"""Evaluate at most four promoted single-target configurations, plus F-A/F-B.

Two target combinations are registered here before reading screening results:
CB-best-per-target and LG-best-per-target. No adaptive blend-weight search.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..artifacts import atomic_write_json, file_sha256, stable_digest, verify_file_identities
from ..config import load_yaml
from ..exceptions import ContractError
from .search_models import search_configurations, select_promotions
from .v3_evidence import acceptance, paired_week_intervals
from .v3_run import DevelopmentContext, TARGETS
from .validation import aggregate_grid, diagnostic_metrics, error_contributions, select_partitions, units_for_origin


def read(path):
    return json.loads(Path(path).read_text())


def run(ctx, reference_run: Path, screening_run: Path):
    for source, suite in ((reference_run, "reference"), (screening_run, "screening")):
        status = read(source / "final_status.json")
        if status.get("engineering_status") != "G0_EXECUTION_PASS" or status.get("suite") != suite:
            raise ContractError("grid source run is incomplete or has wrong suite")
        if stable_digest(read(source / "resolved_config.json")["inputs"]) != stable_digest(ctx.inputs):
            raise ContractError("grid source input identities disagree")
    promotions = read(screening_run / "promotions.json")
    expected = {(m, t) for m in ("tunable_catboost_l1", "lightgbm_l1") for t in TARGETS}
    if len(promotions) != 4 or {(r["model_type"], r["target"]) for r in promotions} != expected:
        raise ContractError("grid requires exactly one promoted model per family and target")
    configs = {c["id"]: c for c in search_configurations("configs/optimization_v0_3/models")}
    if promotions != select_promotions(read(screening_run / "candidate_metrics.json"), list(configs.values())):
        raise ContractError("promotions disagree with registered two-fold selection")
    for row in promotions:
        if configs[row["candidate"]]["model_type"] != row["model_type"]:
            raise ContractError("promoted model registry identity mismatch")
    reference = read(reference_run / "reference_selection.json")["C_ref"]
    metrics = read(reference_run / "candidate_metrics.json")
    reference_summary = aggregate_grid(metrics)
    expected_reference = min(("E12-raw", "E12-CVcal"), key=lambda c: (reference_summary[c]["J"], c != "E12-raw"))
    if reference != expected_reference:
        raise ContractError("C_ref disagrees with registered global reference selection")
    atomic_write_json(ctx.destination / "source_runs.json", {
        str(source): {name: file_sha256(source / name) for name in
                      ("resolved_config.json", "candidate_metrics.json", "final_status.json")}
        for source in (reference_run, screening_run)})
    errors = []
    units = [*ctx.screening, *[unit for origin in ctx.origins for unit in units_for_origin(
        origin, pd.Timestamp(ctx.validation["train_start"]))]]
    groups = {}
    for unit in units:
        groups.setdefault(unit.origin_id, []).append(unit)
    for origin, cells in groups.items():
        first = cells[0]
        train, _, _ = select_partitions(ctx.labels, first)
        evaluation = ctx.labels.loc[(ctx.labels.reference_time >= min(u.eval_start for u in cells))
                                    & (ctx.labels.reference_time < max(u.eval_end for u in cells))]
        predictions = {name: pd.DataFrame({"sample_id": evaluation.sample_id.astype(str)})
                       for name in ("CB-best-per-target", "LG-best-per-target")}
        for row in promotions:
            name = "CB-best-per-target" if row["model_type"] == "tunable_catboost_l1" else "LG-best-per-target"
            if origin in {"DEV_LONG", "DEV_SHORT"}:
                saved = pd.read_csv(screening_run / f"{origin}_{row['candidate']}_{row['target']}.csv", dtype={"sample_id": "string"}, float_precision="round_trip")
                values = pd.DataFrame({"sample_id": evaluation.sample_id.astype(str)}).merge(saved, on="sample_id", validate="one_to_one")
                if len(values) != len(evaluation) or set(values.sample_id) != set(evaluation.sample_id.astype(str)):
                    raise ContractError("screening prediction sample IDs disagree")
                predictions[name][f"pred_{row['target']}"] = values[f"pred_{row['target']}"].to_numpy()
            else:
                predictions[name][f"pred_{row['target']}"] = ctx.tunable(
                    configs[row["candidate"]], row["target"], train, evaluation, first.fit_cutoff, origin)
        for increment in ("F-A", "F-B"):
            predictions[increment] = ctx.frozen(train, evaluation, first.fit_cutoff, "E09", origin, increment)
        for unit in cells:
            _, actual, _ = select_partitions(ctx.labels, unit)
            for name, prediction in predictions.items():
                pred = prediction.loc[prediction.sample_id.astype(str).isin(actual.sample_id.astype(str))]
                metrics[unit.id]["candidates"][name] = diagnostic_metrics(actual, pred, ctx.feature_frame(actual, unit.fit_cutoff))
                path = ctx.destination / "units" / unit.id / name
                path.mkdir(parents=True)
                pred.to_csv(path / "predictions.csv", index=False)
                err = error_contributions(actual, pred)
                err.to_csv(path / "errors.csv", index=False)
                if unit.horizon > 0:
                    errors.append(err.assign(candidate=name, origin=origin, horizon=unit.horizon))
            if unit.horizon > 0:
                err = pd.read_csv(reference_run / "units" / unit.id / reference / "errors.csv", dtype={"sample_id": "string"}, float_precision="round_trip")
                errors.append(err.assign(candidate=reference, origin=origin, horizon=unit.horizon))
        atomic_write_json(ctx.destination / "candidate_metrics.json", metrics, overwrite=True)
        print(f"grid completed {origin}", flush=True)
    summary = aggregate_grid(metrics)
    gate = acceptance(metrics, summary, reference, load_yaml("configs/optimization_v0_3/acceptance.yaml"))
    atomic_write_json(ctx.destination / "grid_summary.json", summary)
    atomic_write_json(ctx.destination / "acceptance.json", gate)
    all_errors = pd.concat(errors, ignore_index=True)
    intervals = {name: paired_week_intervals(all_errors.loc[all_errors.candidate.isin([name, reference])], name, reference)
                 for name in predictions}
    atomic_write_json(ctx.destination / "paired_delta_by_horizon.json", intervals)
    verify_file_identities(ctx.inputs)
    atomic_write_json(ctx.destination / "final_status.json", {
        "engineering_status": "G0_EXECUTION_PASS", "model_quality_status": "G1_" + gate["status"] + "_DEVELOPMENT",
        "protected_labels_read": False, "total_target_fits": sum(r["actual_fit_count"] for r in ctx.fit_records),
        "feature_crosses_run": 0, "new_model_calibration_run": False,
        "remaining": ["optional_top_feature_crosses", "new_model_calibration_comparison", "OPT-10_lifecycle_and_release"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.local.yaml")
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--screening-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(DevelopmentContext(args.data_config, args.output), args.reference_run, args.screening_run)
    except Exception as exc:
        atomic_write_json(args.output / "final_status.json", {"status": "FAILED", "error": str(exc), "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
