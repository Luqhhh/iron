"""Development-only OPT-07/08 runner with temporal inner selection.

Run using ``python -m bf_tap.optimization.v3_run --help``.
All label reads go through the existing protected development reader.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json, code_identity, file_identities, stable_digest, verify_file_identities
from ..config import load_yaml, validate_data_paths, validate_frozen_contracts
from ..exceptions import ContractError
from ..features import build_features
from ..metrics import score_predictions, target_metrics
from ..models.sanity import MedianControls
from ..protection import load_protection_policy
from .calibration import fit_time_calibration, inner_blocks
from .config import load_experiment, load_feature_selection, load_validation
from .features import select_candidate_features
from .models import FrozenBaselineAdapter
from .process_change import add_process_change_features
from .run import _load_inputs
from .search_models import SingleTargetModel, search_configurations, select_promotions
from .temporal_increment import segmented_operation
from .validation import aggregate_grid, diagnostic_metrics, error_contributions, select_partitions, units_for_origin
from .v3_config import validate_registration

TARGETS = ("tap_iron", "tap_time_len")


class DevelopmentContext:
    def __init__(self, data_config: str, destination: Path):
        self.destination = destination
        self.registration = validate_registration()
        self.baseline = load_yaml("configs/baseline.yaml")
        self.features = load_yaml("configs/features.yaml")
        self.semantic = load_yaml("configs/data_contract.yaml")
        validate_frozen_contracts(self.baseline, self.features, self.semantic)
        self.policy = load_protection_policy("configs/protection.yaml")
        self.selection = load_feature_selection("configs/optimization_v0_2/features.yaml")
        _, candidates = load_experiment("configs/optimization_v0_2/experiment.yaml")
        self.candidates = {c.id: c for c in candidates}
        self.candidates["E09"] = self.candidates["E09_PROCESS_CHANGE_E02"]
        self.validation, self.screening, self.origins = load_validation(
            "configs/optimization_v0_3/validation.yaml", expected_timezone=self.policy.timezone)
        self.data = load_yaml(data_config)
        validate_data_paths(self.data, command="development")
        self.inputs = file_identities({k: self.data["paths"][k] for k in
                                     ("train_samples", "tap_history_train", "operation_hourly", "burden_change", "data_dictionary")})
        self.code = code_identity(Path.cwd())
        self.labels, self.history, self.operation, self.burden, self.consistency = _load_inputs(
            data_cfg=self.data, semantic_cfg=self.semantic, feature_cfg=self.features,
            protection=self.policy, requested_end=self.policy.development_label_end_exclusive)
        self.labels = self.labels.sort_values(["reference_time", "sample_id"], kind="mergesort")
        self.labels = self.labels.loc[self.labels.reference_time >= pd.Timestamp(self.validation["train_start"])]
        self.fit_records = []
        self.cache = {}
        atomic_write_json(destination / "resolved_config.json", {
            "optimization_id": "optimization-v0.3", "baseline": self.baseline,
            "registration": self.registration,
            "features": self.features, "selection": self.selection, "validation": self.validation,
            "inputs": self.inputs, "source": self.code, "development_only": True,
            "config_files": file_identities({str(p): p for p in Path("configs/optimization_v0_3").rglob("*.yaml")}),
            "search": search_configurations("configs/optimization_v0_3/models")})

    def feature_frame(self, samples, cutoff):
        key = (str(cutoff), stable_digest({
            "metadata": samples[["sample_id", "spout_no", "reference_time"]].astype(str).to_dict("records"),
            "index": list(samples.index),
        }))
        if key not in self.cache:
            result = build_features(samples, operation=self.operation, burden=self.burden,
                                    history=self.history, fit_cutoff=cutoff, config=self.features)
            self.cache[key] = add_process_change_features(result.X, self.selection["process_change"],
                                                         baseline_value_columns=self.features["operation"]["value_columns"])
        return self.cache[key]

    def X(self, samples, cutoff, candidate="E09", increment=None):
        frame = select_candidate_features(self.feature_frame(samples, cutoff), self.candidates[candidate], self.selection)
        if increment == "F-A":
            frame = frame.loc[:, ~frame.columns.str.startswith("burden__")]
        elif increment == "F-B":
            key = ("segments", stable_digest({
                "metadata": samples[["sample_id", "spout_no", "reference_time"]].astype(str).to_dict("records"),
                "index": list(samples.index),
            }))
            if key not in self.cache:
                self.cache[key] = segmented_operation(samples, self.operation,
                                                      self.features["operation"]["value_columns"])
            frame = pd.concat([frame, self.cache[key]], axis=1)
        elif increment is not None:
            raise ContractError("feature increment not authorized/implemented")
        return frame

    def record_fit(self, **record):
        self.fit_records.append(record)
        # Preserve each completed fit even if a later fit fails.
        with (self.destination / "registry.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def frozen(self, train, evaluation, cutoff, candidate, context, increment=None):
        model = FrozenBaselineAdapter(dict(self.baseline["parameters"]), ("spout_no",))
        model.fit(self.X(train, cutoff, candidate, increment), train[list(TARGETS)])
        self.record_fit(candidate=candidate, context=context, cutoff=str(cutoff), targets=list(TARGETS),
                        feature_increment=increment,
                        actual_fit_count=2, iterations={t: 800 for t in TARGETS}, rows=len(train),
                        train_samples_sha256=stable_digest(list(train.sample_id.astype(str))))
        result = model.predict_raw(self.X(evaluation, cutoff, candidate, increment)).clip(lower=0)
        result.index = evaluation.index
        result.insert(0, "sample_id", evaluation.sample_id.astype(str))
        return result

    def tunable(self, config, target, train, evaluation, cutoff, origin_id):
        blocks = inner_blocks(self.labels, cutoff)
        first = cutoff - pd.Timedelta(days=56)
        inner_train, inner_eval = blocks["selection_train"], blocks["selection_eval"]
        model = SingleTargetModel(config["model_type"], config["parameters"], target, config["patience"])
        model.fit(self.X(inner_train, first), inner_train[target],
                  inner_validation=(self.X(inner_eval, first), inner_eval[target]))
        count = model.selected_iterations
        split = {name: {"rows": len(value), "samples_sha256": stable_digest(list(value.sample_id.astype(str))),
                        "reference_min": str(value.reference_time.min()), "reference_max": str(value.reference_time.max()),
                        "label_available_min": str(value.label_available_at.min()), "label_available_max": str(value.label_available_at.max())}
                 for name, value in blocks.items()}
        self.record_fit(candidate=config["id"], target=target, origin=origin_id, context="inner_selection",
                        actual_fit_count=1, cutoff=str(first), history_cutoff=str(first),
                        selected_iterations=count, split=split, parameters=config["parameters"])
        model = SingleTargetModel(config["model_type"], config["parameters"], target, config["patience"])
        model.fit(self.X(train, cutoff), train[target], iterations=count)
        self.record_fit(candidate=config["id"], target=target, origin=origin_id, context="outer_refit",
                        actual_fit_count=1, cutoff=str(cutoff), selected_iterations=count, rows=len(train),
                        train_samples_sha256=stable_digest(list(train.sample_id.astype(str))))
        model.save(self.destination / "bundles" / origin_id / config["id"] / target,
                   identity={"fit_cutoff": str(cutoff), "history_cutoff": str(cutoff),
                             "label_available_cutoff": str(cutoff), "inner_split": split,
                             "train_samples_sha256": stable_digest(list(train.sample_id.astype(str))),
                             "source_sha256": stable_digest(self.code), "data_sha256": stable_digest(self.inputs)})
        return np.maximum(0, model.predict(self.X(evaluation, cutoff)))


def reference_run(ctx):
    metrics, provenance = {}, {}
    units = [*ctx.screening, *[unit for origin in ctx.origins for unit in units_for_origin(
        origin, pd.Timestamp(ctx.validation["train_start"]))]]
    grouped = {}
    for unit in units:
        grouped.setdefault(unit.origin_id, []).append(unit)
    for origin, cells in grouped.items():
        first = cells[0]
        train, _, _ = select_partitions(ctx.labels, first)
        evaluation = ctx.labels.loc[(ctx.labels.reference_time >= min(u.eval_start for u in cells))
                                    & (ctx.labels.reference_time < max(u.eval_end for u in cells))]
        predictions = {c: ctx.frozen(train, evaluation, first.fit_cutoff, c, origin)
                       for c in ("E00", "E09", "E04")}
        controls = MedianControls(min_group_count=20).fit(train)
        predictions.update({c: controls.predict(evaluation, c) for c in ("B0", "B1")})
        raw = predictions["E09"].copy()
        for t in TARGETS:
            raw[f"pred_{t}"] = .8 * predictions["E09"][f"pred_{t}"] + .2 * predictions["E04"][f"pred_{t}"]
        predictions["E12-raw"] = raw
        blocks = inner_blocks(ctx.labels, first.fit_cutoff)
        inner = first.fit_cutoff - pd.Timedelta(days=28)
        cal = blocks["calibration_eval"]
        if len(cal):
            parts = {c: ctx.frozen(blocks["calibration_train"], cal, inner, c, origin + ":calibration")
                     for c in ("E09", "E04")}
            cal_rows = cal[["sample_id", "reference_time", "label_available_at", "tap_time_len"]].copy()
            cal_rows["pred_tap_time_len"] = .8 * parts["E09"].pred_tap_time_len + .2 * parts["E04"].pred_tap_time_len
        else:
            cal_rows = cal[["sample_id", "reference_time", "label_available_at", "tap_time_len"]].copy()
            cal_rows["pred_tap_time_len"] = pd.Series(dtype=float)
        cal_rows["prediction_fit_cutoff"] = inner
        cal_rows["history_cutoff"] = inner
        prov = fit_time_calibration(cal_rows, outer_cutoff=first.fit_cutoff, candidate="E12-CVcal",
                                    source_run=str(ctx.destination), origin_id=origin,
                                    pipeline_identity=stable_digest({"code": ctx.code, "algorithm": "E12-CVcal-v1"}),
                                    outer_sample_ids=list(evaluation.sample_id.astype(str)))
        prov["independent_evaluation_cells"] = [u.id for u in cells]
        provenance[origin] = prov
        cal_rows.to_csv(ctx.destination / f"calibration_oof_{origin}.csv", index=False)
        calibrated = raw.copy()
        calibrated["pred_tap_time_len"] = (calibrated.pred_tap_time_len - prov["median_prediction_minus_actual"]["tap_time_len"]).clip(lower=0)
        predictions["E12-CVcal"] = calibrated
        for unit in cells:
            _, actual, _ = select_partitions(ctx.labels, unit)
            values = {}
            for name, prediction in predictions.items():
                pred = prediction.loc[prediction.sample_id.astype(str).isin(actual.sample_id.astype(str))]
                values[name] = diagnostic_metrics(actual, pred, ctx.feature_frame(actual, unit.fit_cutoff))
                path = ctx.destination / "units" / unit.id / name
                path.mkdir(parents=True)
                pred.to_csv(path / "predictions.csv", index=False)
                error_contributions(actual, pred).to_csv(path / "errors.csv", index=False)
            metrics[unit.id] = {"origin_id": origin, "horizon": unit.horizon, "candidates": values}
        atomic_write_json(ctx.destination / "candidate_metrics.json", metrics, overwrite=True)
        atomic_write_json(ctx.destination / "calibration_provenance.json", provenance, overwrite=True)
        print(f"reference completed {origin}", flush=True)
    summary = aggregate_grid(metrics)
    reference = min(("E12-raw", "E12-CVcal"), key=lambda c: (summary[c]["J"], c != "E12-raw"))
    atomic_write_json(ctx.destination / "grid_summary.json", summary)
    atomic_write_json(ctx.destination / "reference_selection.json", {"C_ref": reference, "scope": "global_development_selection", "J": summary[reference]["J"]})


def screening_run(ctx):
    configs = search_configurations("configs/optimization_v0_3/models")
    records = []
    for unit in ctx.screening:
        train, evaluation, _ = select_partitions(ctx.labels, unit)
        for config in configs:
            for target in TARGETS:
                values = ctx.tunable(config, target, train, evaluation, unit.fit_cutoff, unit.id)
                metric = asdict(target_metrics(evaluation[target], values))
                records.append({"fold": unit.id, "candidate": config["id"], "target": target,
                                "model_type": config["model_type"], **metric})
                pd.DataFrame({"sample_id": evaluation.sample_id, f"pred_{target}": values}).to_csv(
                    ctx.destination / f"{unit.id}_{config['id']}_{target}.csv", index=False)
                atomic_write_json(ctx.destination / "candidate_metrics.json", records, overwrite=True)
                print(f"screening {unit.id} {config['id']} {target}: {metric['wmape']:.6f}", flush=True)
    promotions = select_promotions(records, configs)
    atomic_write_json(ctx.destination / "promotions.json", promotions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.local.yaml")
    parser.add_argument("--suite", choices=["reference", "screening"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        ctx = DevelopmentContext(args.data_config, destination)
        (reference_run if args.suite == "reference" else screening_run)(ctx)
        verify_file_identities(ctx.inputs)
        atomic_write_json(destination / "final_status.json", {
            "engineering_status": "G0_EXECUTION_PASS", "model_quality_status": "G1_NOT_EVALUATED_FULL_GATE",
            "protected_labels_read": False, "suite": args.suite,
            "total_target_fits": sum(r["actual_fit_count"] for r in ctx.fit_records)})
    except Exception as exc:
        atomic_write_json(destination / "final_status.json", {"status": "FAILED", "error": str(exc), "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
