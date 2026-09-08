"""The second and final registered feature/model cross; development labels only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json, file_identities, verify_file_identities
from ..config import load_yaml
from ..exceptions import ContractError
from .search_models import search_configurations, select_promotions
from .v3_evidence import acceptance, paired_week_intervals
from .v3_followup import fit_calibration, fit_cross_target, read, snapshot_sources
from .v3_remaining_features import ProfileContext
from .v3_run import TARGETS
from .validation import aggregate_grid, diagnostic_metrics, error_contributions, select_partitions, units_for_origin


def registration():
    cfg = load_yaml("configs/optimization_v0_3/last_cross.yaml")
    expected = {"protocol": "last-feature-cross-v1", "reference": "E12-raw", "feature": "F-C",
                "model_family": "tunable_catboost_l1", "target_configs": {"tap_iron": "CB08", "tap_time_len": "CB02"},
                "candidates": ["CB-FC-raw", "CB-FC-CVcal"], "calibration_protocol": "calibration-protocol-v1",
                "new_feature_crosses": 1, "cumulative_feature_crosses": 2,
                "additional_parameter_configurations": 0, "additional_target_combinations": 0,
                "additional_feature_candidates": 0, "protected_labels_read": False, "stop_after_this_cross": True}
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ContractError("last cross differs from bounded registration")
    return cfg


def run(ctx, cfg):
    roots = {key: Path(cfg[key]) for key in ("reference_run", "screening_run", "first_feature_run", "remaining_feature_run", "previous_cross_run")}
    sources = file_identities({str(p): p for root in roots.values() for p in root.rglob("*") if p.is_file()})
    for root in roots.values():
        resolved = read(root / "resolved_config.json")
        if (read(root / "final_status.json").get("engineering_status") != "G0_EXECUTION_PASS"
                or resolved["inputs"] != ctx.inputs or resolved["features"] != ctx.features
                or resolved["selection"] != ctx.selection or resolved["validation"] != ctx.validation):
            raise ContractError("last-cross source/config identity mismatch")
    if read(roots["remaining_feature_run"] / "known_index_manifest.json") != ctx.index_audit:
        raise ContractError("last-cross known-index identity mismatch")
    first = read(roots["first_feature_run"] / "grid_summary.json")
    remaining = read(roots["remaining_feature_run"] / "grid_summary.json")
    order = sorted(("F-A", "F-B", "F-C", "F-D"), key=lambda p: ({**first, **remaining}[p]["J"], p))
    if order[:2] != ["F-C", "F-B"] or remaining["F-C"]["J"] >= remaining["E09"]["J"]:
        raise ContractError("F-C is not the registered top-two incremental feature")
    if read(roots["previous_cross_run"] / "final_status.json")["new_feature_crosses"] != 1:
        raise ContractError("previous cross budget identity differs")
    configs = {c["id"]: c for c in search_configurations("configs/optimization_v0_3/models")}
    promotions = select_promotions(read(roots["screening_run"] / "candidate_metrics.json"), list(configs.values()))
    if {p["target"]: p["candidate"] for p in promotions if p["model_type"] == cfg["model_family"]} != cfg["target_configs"]:
        raise ContractError("last-cross target configurations differ from original promotion")
    metrics = read(roots["reference_run"] / "candidate_metrics.json")
    ref_summary = aggregate_grid(metrics)
    reference = min(("E12-raw", "E12-CVcal"), key=lambda c: (ref_summary[c]["J"], c != "E12-raw"))
    if reference != cfg["reference"]:
        raise ContractError("last-cross reference differs from global C_ref")
    groups = {}
    for unit in [*ctx.screening, *[u for o in ctx.origins for u in units_for_origin(o, pd.Timestamp(ctx.validation["train_start"]))]]:
        groups.setdefault(unit.origin_id, []).append(unit)
    errors, provenance = [], {}
    for origin, units in groups.items():
        cutoff = units[0].fit_cutoff
        evaluation = ctx.labels.loc[(ctx.labels.reference_time >= min(u.eval_start for u in units))
                                    & (ctx.labels.reference_time < max(u.eval_end for u in units))]
        raw = evaluation[["sample_id"]].copy()
        time_selection = None
        for target in TARGETS:
            config = configs[cfg["target_configs"][target]]
            model, iterations, selection_source = fit_cross_target(ctx, config, target, cutoff, origin,
                                                                   increment="F-C", candidate="CB-FC-raw")
            raw[f"pred_{target}"] = np.maximum(0, model.predict(ctx.X(evaluation, cutoff, increment="F-C")))
            if target == "tap_time_len":
                time_selection = (config, iterations, selection_source)
        config, iterations, selection_source = time_selection
        prov = fit_calibration(ctx, config, iterations, cutoff, origin, "CB-FC-CVcal", list(evaluation.sample_id),
                               increment="F-C", selection_source=selection_source)
        prov["independent_evaluation_cells"] = [u.id for u in units]
        provenance[origin] = prov
        calibrated = raw.copy()
        calibrated.pred_tap_time_len = np.maximum(0, calibrated.pred_tap_time_len - prov["median_prediction_minus_actual"]["tap_time_len"])
        for unit in units:
            _, actual, _ = select_partitions(ctx.labels, unit)
            for candidate, prediction in (("CB-FC-raw", raw), ("CB-FC-CVcal", calibrated)):
                part = prediction.loc[prediction.sample_id.astype(str).isin(actual.sample_id.astype(str))]
                metrics[unit.id]["candidates"][candidate] = diagnostic_metrics(actual, part, ctx.feature_frame(actual, cutoff))
                path = ctx.destination / "units" / unit.id / candidate
                path.mkdir(parents=True)
                part.to_csv(path / "predictions.csv", index=False)
                err = error_contributions(actual, part)
                err.to_csv(path / "errors.csv", index=False)
                for target in TARGETS:
                    err.nlargest(20, f"abs_error_{target}").to_csv(path / f"top20_{target}.csv", index=False)
                if unit.horizon:
                    errors.append(err.assign(candidate=candidate, origin=origin, horizon=unit.horizon))
            if unit.horizon:
                err = pd.read_csv(roots["reference_run"] / "units" / unit.id / reference / "errors.csv",
                                  dtype={"sample_id": "string"}, float_precision="round_trip")
                errors.append(err.assign(candidate=reference, origin=origin, horizon=unit.horizon))
        atomic_write_json(ctx.destination / "candidate_metrics.json", metrics, overwrite=True)
        atomic_write_json(ctx.destination / "calibration_provenance.json", provenance, overwrite=True)
        print(f"last cross completed {origin}", flush=True)
    summary = aggregate_grid(metrics)
    gate = acceptance(metrics, summary, reference, load_yaml("configs/optimization_v0_3/acceptance.yaml"))
    atomic_write_json(ctx.destination / "grid_summary.json", summary)
    atomic_write_json(ctx.destination / "acceptance.json", gate)
    errors = pd.concat(errors, ignore_index=True)
    pairs = [(candidate, reference) for candidate in cfg["candidates"]] + [("CB-FC-CVcal", "CB-FC-raw")]
    atomic_write_json(ctx.destination / "paired_delta_by_horizon.json", {
        f"{candidate}_vs_{ref}": paired_week_intervals(errors.loc[errors.candidate.isin([candidate, ref])], candidate, ref)
        for candidate, ref in pairs})
    verify_file_identities(sources)
    atomic_write_json(ctx.destination / "source_files.json", sources)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", nargs="+", required=True)
    parser.add_argument("--data-config", default="configs/data.local.yaml")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        snapshot = snapshot_sources(args.output)
        cfg = registration()
        atomic_write_json(args.output / "last_cross_registration.json", cfg)
        ctx = ProfileContext(args.data_config, args.output, args.sample_index)
        gate = run(ctx, cfg)
        verify_file_identities(ctx.inputs)
        verify_file_identities(ctx.index_audit["sources"])
        verify_file_identities(snapshot["files"])
        atomic_write_json(args.output / "final_status.json", {
            "engineering_status": "G0_EXECUTION_PASS", "model_quality_status": "G1_" + gate["status"] + "_DEVELOPMENT",
            "protected_labels_read": False, "total_target_fits": sum(r["actual_fit_count"] for r in ctx.fit_records),
            "new_feature_crosses": 1, "cumulative_feature_crosses": 2, "development_search_closed": True,
            "source_archive_sha256": snapshot["archive_sha256"]})
    except Exception as exc:
        atomic_write_json(args.output / "final_status.json", {"status": "FAILED", "error": str(exc), "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
