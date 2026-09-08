"""Registered follow-up: pre-origin calibration and one F-B/CatBoost cross."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..artifacts import atomic_write_json, code_identity, file_identities, file_sha256, stable_digest, verify_file_identities
from ..config import load_yaml
from ..exceptions import ContractError
from ..metrics import score_predictions
from .calibration import fit_time_calibration, inner_blocks
from .search_models import SingleTargetModel, search_configurations, select_promotions
from .v3_evidence import acceptance, paired_week_intervals
from .v3_run import DevelopmentContext, TARGETS
from .validation import aggregate_grid, diagnostic_metrics, error_contributions, select_partitions, units_for_origin


def read(path):
    return json.loads(Path(path).read_text())


def registration(path="configs/optimization_v0_3/followup.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    expected_candidates = {
        "CB-CVcal": {"base": "CB-best-per-target", "feature_increment": None, "calibration": "temporal_v1"},
        "LG-CVcal": {"base": "LG-best-per-target", "feature_increment": None, "calibration": "temporal_v1"},
        "CB-FB-raw": {"base": "CB-best-per-target", "feature_increment": "F-B", "calibration": "none"},
        "CB-FB-CVcal": {"base": "CB-best-per-target", "feature_increment": "F-B", "calibration": "temporal_v1"},
    }
    if (cfg.get("schema_version") != "optimization-v0.3-followup-v1" or cfg.get("reference") != "E12-raw"
            or cfg.get("candidates") != expected_candidates
            or cfg.get("target_configs") != {"CB-best-per-target": {"tap_iron": "CB08", "tap_time_len": "CB02"},
                                             "LG-best-per-target": {"tap_iron": "LG02", "tap_time_len": "LG01"}}
            or cfg.get("new_feature_crosses") != 1 or cfg.get("additional_parameter_configurations") != 0
            or cfg.get("additional_target_combinations") != 0 or cfg.get("additional_blend_weights") != 0
            or cfg.get("protected_access_authorized") is not False
            or cfg.get("selection_iteration_policy") != "reuse_verified_inner_selection_for_unchanged_features_reselect_for_cross"):
        raise ContractError("follow-up differs from the registered bounded experiment")
    return cfg


def snapshot_sources(destination: Path) -> dict:
    before = code_identity(Path.cwd())
    paths = [Path("pyproject.toml"), Path("uv.lock"), Path("AGENTS.md")]
    for root in ("src", "configs", "tests"):
        paths.extend(p for p in Path(root).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and not p.name.endswith(".local.yaml"))
    identities = file_identities({str(p): p for p in sorted(paths)})
    with tarfile.open(destination / "source_snapshot.tar.gz", "x:gz") as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path), recursive=False)
    verify_file_identities(identities)
    if before["source_snapshot_sha256"] != code_identity(Path.cwd())["source_snapshot_sha256"]:
        raise ContractError("source changed during snapshot")
    result = {"code": before, "files": identities, "archive_sha256": file_sha256(destination / "source_snapshot.tar.gz")}
    atomic_write_json(destination / "source_snapshot.json", result)
    return result


def model_identity(ctx, train, cutoff, blocks, increment, selection_source=None):
    split = {name: {"rows": len(value), "samples_sha256": stable_digest(list(value.sample_id.astype(str))),
                    "reference_min": str(value.reference_time.min()), "reference_max": str(value.reference_time.max()),
                    "label_available_min": str(value.label_available_at.min()), "label_available_max": str(value.label_available_at.max())}
             for name, value in blocks.items()}
    return {"fit_cutoff": str(cutoff), "history_cutoff": str(cutoff), "label_available_cutoff": str(cutoff),
            "train_samples_sha256": stable_digest(list(train.sample_id.astype(str))), "inner_split": split,
            "source_sha256": stable_digest(ctx.code), "data_sha256": stable_digest(ctx.inputs),
            "feature_increment": increment, "selected_iterations_source": selection_source}


def fit_cross_target(ctx, config, target, cutoff, origin, *, increment="F-B", candidate="CB-FB-raw"):
    if (increment, candidate) not in {("F-B", "CB-FB-raw"), ("F-C", "CB-FC-raw")}:
        raise ContractError("unregistered feature/model cross identity")
    blocks = inner_blocks(ctx.labels, cutoff)
    first = cutoff - pd.Timedelta(days=56)
    train, validation = blocks["selection_train"], blocks["selection_eval"]
    model = SingleTargetModel(config["model_type"], config["parameters"], target, config["patience"])
    model.fit(ctx.X(train, first, increment=increment), train[target],
              inner_validation=(ctx.X(validation, first, increment=increment), validation[target]))
    iterations = model.selected_iterations
    base = ctx.destination / "bundles" / origin / candidate / target
    inner_identity = model_identity(ctx, train, first, blocks, increment)
    inner_identity["selection_available_cutoff"] = str(cutoff - pd.Timedelta(days=28))
    model.save(base / "selection", identity=inner_identity)
    selection_source = file_sha256(base / "selection" / "bundle.json")
    ctx.record_fit(candidate=candidate, target=target, origin=origin, role="selection", actual_fit_count=1,
                   selected_iterations=iterations, config=config, identity=inner_identity)
    train = blocks["outer_train"]
    model = SingleTargetModel(config["model_type"], config["parameters"], target, config["patience"])
    model.fit(ctx.X(train, cutoff, increment=increment), train[target], iterations=iterations)
    identity = model_identity(ctx, train, cutoff, blocks, increment, selection_source)
    model.save(base / "outer", identity=identity)
    ctx.record_fit(candidate=candidate, target=target, origin=origin, role="outer_refit", actual_fit_count=1,
                   selected_iterations=iterations, config=config, identity=identity)
    return model, iterations, selection_source


def fit_calibration(ctx, config, iterations, cutoff, origin, candidate, evaluation_ids,
                    *, increment=None, selection_source):
    blocks = inner_blocks(ctx.labels, cutoff)
    inner = cutoff - pd.Timedelta(days=28)
    train, evaluation = blocks["calibration_train"], blocks["calibration_eval"]
    rows = evaluation[["sample_id", "reference_time", "label_available_at", "tap_time_len"]].copy()
    model_digest = None
    if len(evaluation) >= 100:
        model = SingleTargetModel(config["model_type"], config["parameters"], "tap_time_len", config["patience"])
        model.fit(ctx.X(train, inner, increment=increment), train.tap_time_len, iterations=iterations)
        identity = model_identity(ctx, train, inner, blocks, increment, selection_source)
        path = ctx.destination / "bundles" / origin / candidate / "calibration"
        model.save(path, identity=identity)
        model_digest = file_sha256(path / "bundle.json")
        ctx.record_fit(candidate=candidate, target="tap_time_len", origin=origin, role="calibration_refit",
                       actual_fit_count=1, selected_iterations=iterations, config=config, identity=identity)
        rows["pred_tap_time_len"] = np.maximum(0, model.predict(ctx.X(evaluation, inner, increment=increment)))
    else:
        # No estimator is needed for the registered zero fallback. Empty prediction
        # provenance is explicit; eligible label count is retained separately.
        rows = rows.iloc[:0].copy()
        rows["pred_tap_time_len"] = pd.Series(dtype=float)
    rows["prediction_fit_cutoff"] = inner
    rows["history_cutoff"] = inner
    provenance = fit_time_calibration(rows, outer_cutoff=cutoff, candidate=candidate,
                                      source_run=str(ctx.destination), origin_id=origin,
                                      pipeline_identity=stable_digest({"config": config, "increment": increment,
                                                                      "selection_source": selection_source, "code": ctx.code}),
                                      outer_sample_ids=evaluation_ids)
    provenance.update(selected_iterations=iterations, selected_iterations_source=selection_source,
                      calibration_model_bundle_sha256=model_digest, eligible_calibration_rows=len(evaluation))
    rows.to_csv(ctx.destination / f"calibration_oof_{origin}_{candidate}.csv", index=False)
    return provenance


def verified_iterations(ctx, config, target, origin, cutoff, source: Path, tracked_paths: set[Path]):
    bundle = source / "bundles" / origin / config["id"] / target
    for filename in ("bundle.json", "bundle_identity.json", "model.bin"):
        tracked_paths.add(bundle / filename)
    metadata = read(bundle / "bundle.json")
    if stable_digest(metadata) != read(bundle / "bundle_identity.json")["metadata_sha256"]:
        raise ContractError("source model metadata digest mismatch")
    if file_sha256(bundle / "model.bin") != metadata["model_sha256"]:
        raise ContractError("source model file digest mismatch")
    identity = metadata["identity"]
    blocks = inner_blocks(ctx.labels, cutoff)
    if (metadata["model_type"] != config["model_type"] or metadata["target"] != target
            or metadata["parameters"] != config["parameters"]
            or identity["fit_cutoff"] != str(cutoff) or identity["history_cutoff"] != str(cutoff)
            or identity["label_available_cutoff"] != str(cutoff)
            or identity["data_sha256"] != stable_digest(ctx.inputs)
            or identity["train_samples_sha256"] != stable_digest(list(blocks["outer_train"].sample_id.astype(str)))):
        raise ContractError("source iteration selection model/target/data/cutoff identity mismatch")
    for name in ("selection_train", "selection_eval"):
        if identity["inner_split"][name]["samples_sha256"] != stable_digest(list(blocks[name].sample_id.astype(str))):
            raise ContractError("source internal iteration-selection samples differ")
    registry_path = source / "registry.jsonl"
    tracked_paths.add(registry_path)
    records = [json.loads(line) for line in registry_path.read_text().splitlines()]
    matches = [r for r in records if r.get("candidate") == config["id"] and r.get("target") == target
               and r.get("origin") == origin and r.get("context") == "inner_selection"]
    if len(matches) != 1 or matches[0]["selected_iterations"] != metadata["selected_iterations"]:
        raise ContractError("source bundle iterations disagree with inner selection ledger")
    return int(metadata["selected_iterations"]), file_sha256(bundle / "bundle.json")


def run(ctx, cfg):
    reference_root, screen_root, grid_root = [Path(cfg[k]) for k in ("reference_run", "screening_run", "grid_run")]
    source_identities = file_identities({str(p): p for root in (reference_root, screen_root, grid_root)
                                       for p in root.rglob("*") if p.is_file()})
    tracked = set()
    for root in (reference_root, screen_root, grid_root):
        for filename in ("resolved_config.json", "candidate_metrics.json", "final_status.json"):
            tracked.add(root / filename)
        if read(root / "final_status.json").get("engineering_status") != "G0_EXECUTION_PASS":
            raise ContractError("source run did not complete")
        source = read(root / "resolved_config.json")
        if (stable_digest(source["inputs"]) != stable_digest(ctx.inputs) or source["features"] != ctx.features
                or source["selection"] != ctx.selection or source["validation"] != ctx.validation):
            raise ContractError("follow-up source data/feature/validation contract mismatch")
    metrics = read(reference_root / "candidate_metrics.json")
    ref_summary = aggregate_grid(metrics)
    reference = min(("E12-raw", "E12-CVcal"), key=lambda c: (ref_summary[c]["J"], c != "E12-raw"))
    if reference != cfg["reference"]:
        raise ContractError("registered reference differs from global selection")
    configs = {c["id"]: c for c in search_configurations("configs/optimization_v0_3/models")}
    promoted = select_promotions(read(screen_root / "candidate_metrics.json"), list(configs.values()))
    expected = {(base, target): cid for base, targets in cfg["target_configs"].items() for target, cid in targets.items()}
    received = {("CB-best-per-target" if p["model_type"] == "tunable_catboost_l1" else "LG-best-per-target", p["target"]): p["candidate"] for p in promoted}
    if expected != received:
        raise ContractError("registered follow-up target configurations differ from screening")
    source_grid = read(grid_root / "candidate_metrics.json")
    groups = {}
    for unit in [*ctx.screening, *[u for o in ctx.origins for u in units_for_origin(o, pd.Timestamp(ctx.validation["train_start"]))]]:
        groups.setdefault(unit.origin_id, []).append(unit)
    provenance, errors = {}, []
    for origin, units in groups.items():
        cutoff = units[0].fit_cutoff
        evaluation = ctx.labels.loc[(ctx.labels.reference_time >= min(u.eval_start for u in units))
                                    & (ctx.labels.reference_time < max(u.eval_end for u in units))]
        predictions = {}
        for base, cal_name in (("CB-best-per-target", "CB-CVcal"), ("LG-best-per-target", "LG-CVcal")):
            parts = []
            for unit in units:
                path = grid_root / "units" / unit.id / base / "predictions.csv"
                tracked.add(path)
                part = pd.read_csv(path, dtype={"sample_id": "string"}, float_precision="round_trip")
                _, actual, _ = select_partitions(ctx.labels, unit)
                measured = score_predictions(actual, part)
                if abs(measured["loss"] - source_grid[unit.id]["candidates"][base]["overall"]["loss"]) > 1e-14:
                    raise ContractError("source raw predictions disagree with recorded metrics")
                metrics[unit.id]["candidates"][base] = source_grid[unit.id]["candidates"][base]
                parts.append(part)
            raw = pd.concat(parts, ignore_index=True)
            predictions[base] = raw
            config = configs[cfg["target_configs"][base]["tap_time_len"]]
            source = screen_root if origin.startswith("DEV") else grid_root
            iterations, selection_source = verified_iterations(ctx, config, "tap_time_len", origin, cutoff, source, tracked)
            prov = fit_calibration(ctx, config, iterations, cutoff, origin, cal_name, list(evaluation.sample_id.astype(str)),
                                   selection_source=selection_source)
            prov["independent_evaluation_cells"] = [u.id for u in units]
            provenance[f"{origin}/{cal_name}"] = prov
            calibrated = raw.copy()
            calibrated["pred_tap_time_len"] = (raw.pred_tap_time_len - prov["median_prediction_minus_actual"]["tap_time_len"]).clip(lower=0)
            predictions[cal_name] = calibrated
        cross = pd.DataFrame({"sample_id": evaluation.sample_id.astype(str)})
        for target in TARGETS:
            config = configs[cfg["target_configs"]["CB-best-per-target"][target]]
            model, iterations, selection_source = fit_cross_target(ctx, config, target, cutoff, origin)
            cross[f"pred_{target}"] = np.maximum(0, model.predict(ctx.X(evaluation, cutoff, increment="F-B")))
            if target == "tap_time_len":
                prov = fit_calibration(ctx, config, iterations, cutoff, origin, "CB-FB-CVcal", list(evaluation.sample_id.astype(str)),
                                       increment="F-B", selection_source=selection_source)
                prov["independent_evaluation_cells"] = [u.id for u in units]
                provenance[f"{origin}/CB-FB-CVcal"] = prov
        predictions["CB-FB-raw"] = cross
        calibrated = cross.copy()
        calibrated["pred_tap_time_len"] = (cross.pred_tap_time_len - prov["median_prediction_minus_actual"]["tap_time_len"]).clip(lower=0)
        predictions["CB-FB-CVcal"] = calibrated
        for unit in units:
            _, actual, _ = select_partitions(ctx.labels, unit)
            for name, prediction in predictions.items():
                part = prediction.loc[prediction.sample_id.astype(str).isin(actual.sample_id.astype(str))]
                metrics[unit.id]["candidates"][name] = diagnostic_metrics(actual, part, ctx.feature_frame(actual, cutoff))
                path = ctx.destination / "units" / unit.id / name
                path.mkdir(parents=True)
                part.to_csv(path / "predictions.csv", index=False)
                err = error_contributions(actual, part)
                err.to_csv(path / "errors.csv", index=False)
                for target in TARGETS:
                    err.nlargest(20, f"abs_error_{target}").to_csv(path / f"top20_{target}.csv", index=False)
                if unit.horizon:
                    errors.append(err.assign(candidate=name, origin=origin, horizon=unit.horizon))
            if unit.horizon:
                path = reference_root / "units" / unit.id / reference / "errors.csv"
                tracked.add(path)
                err = pd.read_csv(path, dtype={"sample_id": "string"}, float_precision="round_trip")
                errors.append(err.assign(candidate=reference, origin=origin, horizon=unit.horizon))
        atomic_write_json(ctx.destination / "candidate_metrics.json", metrics, overwrite=True)
        atomic_write_json(ctx.destination / "calibration_provenance.json", provenance, overwrite=True)
        print(f"follow-up completed {origin}", flush=True)
    summary = aggregate_grid(metrics)
    gate = acceptance(metrics, summary, reference, load_yaml("configs/optimization_v0_3/acceptance.yaml"))
    atomic_write_json(ctx.destination / "grid_summary.json", summary)
    atomic_write_json(ctx.destination / "acceptance.json", gate)
    errors = pd.concat(errors, ignore_index=True)
    pairs = {name: reference for name in cfg["candidates"]}
    pairs.update({"CB-CVcal_vs_raw": "CB-best-per-target", "LG-CVcal_vs_raw": "LG-best-per-target",
                  "CB-FB-CVcal_vs_raw": "CB-FB-raw"})
    intervals = {}
    for label, comparator in pairs.items():
        candidate = label.removesuffix("_vs_raw")
        intervals[label] = paired_week_intervals(errors.loc[errors.candidate.isin([candidate, comparator])], candidate, comparator)
    atomic_write_json(ctx.destination / "paired_delta_by_horizon.json", intervals)
    verify_file_identities(source_identities)
    atomic_write_json(ctx.destination / "source_files.json", {str(p): source_identities[str(p)] for p in tracked})
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-config", default="configs/data.local.yaml")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        cfg = registration()
        snapshot = snapshot_sources(args.output)
        atomic_write_json(args.output / "followup_registration.json", cfg)
        ctx = DevelopmentContext(args.data_config, args.output)
        gate = run(ctx, cfg)
        verify_file_identities(ctx.inputs)
        verify_file_identities(snapshot["files"])
        atomic_write_json(args.output / "final_status.json", {
            "engineering_status": "G0_EXECUTION_PASS", "model_quality_status": "G1_" + gate["status"] + "_DEVELOPMENT",
            "protected_labels_read": False, "new_feature_crosses": 1, "calibrated_algorithms": 3,
            "total_target_fits": sum(r["actual_fit_count"] for r in ctx.fit_records),
            "source_archive_sha256": snapshot["archive_sha256"]})
    except Exception as exc:
        atomic_write_json(args.output / "final_status.json", {"status": "FAILED", "error": str(exc), "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
