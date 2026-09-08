"""OPT-09 remaining two registered feature experiments, with explicit index input."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..artifacts import (atomic_write_json, build_inference_source_contract, file_identities,
                         file_sha256, runtime_environment, stable_digest, verify_file_identities)
from ..availability import freeze_history_origin
from ..config import load_yaml, validate_frozen_contracts
from ..exceptions import ContractError
from ..features import build_features
from ..models.baseline import DualTargetBaseline
from .alignment import aware
from .config import Candidate
from .features import select_candidate_features
from .indexed_features import INDEX_COLUMNS, burden_event_changes, known_index_features, load_known_index
from .models import FrozenBaselineAdapter
from .process_change import add_process_change_features
from .v3_evidence import acceptance, paired_week_intervals
from .v3_followup import read, snapshot_sources
from .v3_run import DevelopmentContext, TARGETS
from .validation import aggregate_grid, diagnostic_metrics, error_contributions, select_partitions, units_for_origin


def load_profiles():
    cfg = load_yaml("configs/optimization_v0_3/features_remaining.yaml")
    if (cfg["profile_id"] != "remaining-features-v1" or set(cfg["profiles"]) != {"F-C", "F-D"}
            or cfg["reference"] != "E12-raw" or cfg["feature_control"] != "E09"
            or cfg["model_parameters"] != "frozen_800"
            or cfg["additional_parameter_configurations"] != 0 or cfg["additional_target_combinations"] != 0
            or cfg["protected_labels_read"] is not False
            or cfg["profiles"]["F-C"]["columns"] != INDEX_COLUMNS
            or cfg["profiles"]["F-C"]["visibility"] != "strictly_before_current_reference_time"
            or cfg["profiles"]["F-C"]["windows_hours"] != [6, 24]
            or cfg["profiles"]["F-C"]["window_left"] != "exclusive"
            or cfg["profiles"]["F-C"]["sources_must_be_explicit"] is not True
            or cfg["profiles"]["F-D"] != {"kind": "burden_visible_event_changes", "delta": "latest_minus_previous_visible_event",
                                            "reuse_age_column": "burden__event_age_hours", "count_windows_hours": [24, 72],
                                            "window": "left_open_right_closed", "continuous_state_assumption": False}):
        raise ContractError("remaining feature registration differs from executable v1 profiles")
    return cfg


def extend_features(base, samples, *, profile, burden, burden_columns, index=None):
    if profile == "F-C":
        if index is None:
            raise ContractError("F-C requires explicit known-index metadata")
        extra = known_index_features(samples[INDEX_COLUMNS], index)
    elif profile == "F-D":
        extra = burden_event_changes(samples, burden, burden_columns)
    else:
        raise ContractError("unsupported remaining feature profile")
    return pd.concat([base, extra], axis=1)


class ProfileContext(DevelopmentContext):
    def __init__(self, data_config, destination, index_paths):
        super().__init__(data_config, destination)
        self.profiles = load_profiles()
        if (file_sha256("初赛数据集/高炉铁次预测-1.pdf") != self.profiles["rules_pdf_sha256"]
                or self.inputs["data_dictionary"]["sha256"] != self.profiles["dictionary_sha256"]):
            raise ContractError("rules/dictionary differ from reviewed source identities")
        roles = self.profiles["profiles"]["F-C"]["deployment_roles"]["development"]
        if {Path(p).resolve() for p in index_paths} != {Path(self.data["paths"][r]).resolve() for r in roles}:
            raise ContractError("development index paths must explicitly match registered source roles")
        self.index, self.index_audit = load_known_index(index_paths)
        self.profile_cache = {}
        atomic_write_json(destination / "profile_registration.json", self.profiles)
        atomic_write_json(destination / "known_index_manifest.json", self.index_audit)

    def X(self, samples, cutoff, candidate="E09", increment=None):
        if increment not in {"F-C", "F-D"}:
            return super().X(samples, cutoff, candidate, increment)
        key = (increment, str(cutoff), stable_digest({"samples": samples[INDEX_COLUMNS].astype(str).to_dict("records"),
                                                    "index": list(samples.index)}))
        if key not in self.profile_cache:
            base = super().X(samples, cutoff, candidate)
            self.profile_cache[key] = extend_features(base, samples, profile=increment, burden=self.burden,
                                                       burden_columns=self.features["burden"]["value_columns"], index=self.index)
        return self.profile_cache[key]

    def frozen(self, train, evaluation, cutoff, candidate, context, increment=None):
        if increment not in {"F-C", "F-D"}:
            return super().frozen(train, evaluation, cutoff, candidate, context, increment)
        self.policy.validate_development_read(cutoff)
        if (train.reference_time >= cutoff).any() or (train.label_available_at > cutoff).any():
            raise ContractError("profile training crossed its label cutoff")
        model = FrozenBaselineAdapter(dict(self.baseline["parameters"]), ("spout_no",))
        X = self.X(train, cutoff, candidate, increment)
        model.fit(X, train[list(TARGETS)])
        path = self.destination / "bundles" / context / increment
        save_profile_model(model, path, self, train, cutoff, candidate, increment, X)
        self.record_fit(candidate=increment, base_candidate=candidate, origin=context, actual_fit_count=2,
                        fit_cutoff=str(cutoff), history_cutoff=str(cutoff), label_available_cutoff=str(cutoff),
                        feature_columns_sha256=stable_digest(list(X.columns)),
                        train_samples_sha256=stable_digest(list(train.sample_id.astype(str))),
                        known_index_metadata_sha256=self.index_audit["metadata_sha256"] if increment == "F-C" else None,
                        model_bundle_sha256=file_sha256(path / "bundle.json"), parameters=self.baseline["parameters"])
        prediction = model.predict_raw(self.X(evaluation, cutoff, candidate, increment)).clip(lower=0)
        prediction.index = evaluation.index
        prediction.insert(0, "sample_id", evaluation.sample_id.astype(str))
        # Actual model restoration on the same complete evaluation batch.
        restored = ProfilePredictor(path)
        again = restored.predict(evaluation, self.operation, self.burden, self.index)
        if not prediction.equals(again):
            difference = (prediction[[f"pred_{t}" for t in TARGETS]] - again[[f"pred_{t}" for t in TARGETS]]).abs().max().max()
            if difference > 1e-12 or not prediction.sample_id.equals(again.sample_id):
                raise ContractError("restored complete feature profile prediction mismatch")
        return prediction


def save_profile_model(model, path, ctx, train, cutoff, candidate, profile, X):
    digests = validate_frozen_contracts(ctx.baseline, ctx.features, ctx.semantic)
    history = freeze_history_origin(ctx.history, cutoff)
    metadata = {
        "baseline_config": ctx.baseline, "semantic_contract": ctx.semantic, "feature_config": ctx.features,
        "contract_digests": digests,
        "inference_source_contract": build_inference_source_contract(ctx.inputs, semantic_contract_sha256=digests["semantic_contract_sha256"]),
        "training": {"mode": "optimization-development-profile", "candidate_id": profile,
                     "fit_cutoff": str(cutoff), "history_cutoff": str(cutoff), "label_available_cutoff": str(cutoff),
                     "train_start": str(train.reference_time.min()), "eligible_rows": len(train),
                     "eligible_sample_id_sha256": stable_digest(list(train.sample_id.astype(str))),
                     "input_identities": ctx.inputs, "protection_access": {"lifecycle": "development", "protected_access": False}},
        "code_identity": ctx.code, "environment": runtime_environment(), "lockfile_sha256": file_sha256("uv.lock"),
        "optimization_profile": {"schema_version": "profile-bundle-v1", "profile": profile,
                                 "base_candidate": asdict(ctx.candidates[candidate]), "feature_selection": ctx.selection,
                                 "registration": ctx.profiles,
                                 "training_index_identity": ctx.index_audit if profile == "F-C" else None,
                                 "feature_columns_sha256": stable_digest(list(X.columns))},
    }
    model.save(path, metadata=metadata, history_snapshot=history)
    atomic_write_json(path / "profile_identity.json", {"bundle_json_sha256": file_sha256(path / "bundle.json")})


class ProfilePredictor:
    """Rebuilds the same features from normalized authorized inputs, without labels."""
    def __init__(self, path):
        path = Path(path)
        if read(path / "profile_identity.json")["bundle_json_sha256"] != file_sha256(path / "bundle.json"):
            raise ContractError("profile bundle metadata identity mismatch")
        self.model = DualTargetBaseline.load(path)
        self.metadata = self.model.bundle_metadata_
        self.profile = self.metadata["optimization_profile"]
        if self.profile["schema_version"] != "profile-bundle-v1" or self.profile["profile"] not in {"F-C", "F-D"}:
            raise ContractError("unsupported profile bundle")
        training = self.metadata["training"]
        if training["candidate_id"] != self.profile["profile"] or not (training["fit_cutoff"] == training["history_cutoff"] == training["label_available_cutoff"]):
            raise ContractError("profile component/cutoff identity mismatch")
        validate_frozen_contracts(self.metadata["baseline_config"], self.metadata["feature_config"], self.metadata["semantic_contract"])
        self.history = self.model.load_history_snapshot()
        self.cutoff = aware(training["fit_cutoff"])

    def predict(self, samples, operation, burden, index=None):
        if (samples.reference_time < self.cutoff).any():
            raise ContractError("profile inference samples precede training cutoff")
        cfg = self.metadata["feature_config"]
        selection = self.profile["feature_selection"]
        frame = build_features(samples, operation=operation, burden=burden, history=self.history,
                               fit_cutoff=self.cutoff, config=cfg).X
        frame = add_process_change_features(frame, selection["process_change"], baseline_value_columns=cfg["operation"]["value_columns"])
        frame = select_candidate_features(frame, Candidate(**self.profile["base_candidate"]), selection)
        X = extend_features(frame, samples, profile=self.profile["profile"], burden=burden,
                            burden_columns=cfg["burden"]["value_columns"], index=index)
        if stable_digest(list(X.columns)) != self.profile["feature_columns_sha256"]:
            raise ContractError("profile feature schema identity mismatch")
        prediction = self.model.predict_raw(X).clip(lower=0)
        prediction.index = samples.index
        prediction.insert(0, "sample_id", samples.sample_id.astype(str))
        return prediction


def run(ctx, reference_root):
    source_identity = file_identities({str(p): p for p in reference_root.rglob("*") if p.is_file()})
    resolved = read(reference_root / "resolved_config.json")
    if (read(reference_root / "final_status.json").get("engineering_status") != "G0_EXECUTION_PASS"
            or resolved["inputs"] != ctx.inputs or resolved["features"] != ctx.features or resolved["selection"] != ctx.selection
            or resolved["validation"] != ctx.validation):
        raise ContractError("feature reference run/source/config identity mismatch")
    metrics = read(reference_root / "candidate_metrics.json")
    reference = read(reference_root / "reference_selection.json")["C_ref"]
    summary = aggregate_grid(metrics)
    if reference != "E12-raw" or reference != min(("E12-raw", "E12-CVcal"), key=lambda c: (summary[c]["J"], c != "E12-raw")):
        raise ContractError("feature reference differs from fixed global C_ref")
    groups = {}
    for unit in [*ctx.screening, *[u for o in ctx.origins for u in units_for_origin(o, pd.Timestamp(ctx.validation["train_start"]))]]:
        groups.setdefault(unit.origin_id, []).append(unit)
    errors = []
    for origin, units in groups.items():
        train, _, _ = select_partitions(ctx.labels, units[0])
        evaluation = ctx.labels.loc[(ctx.labels.reference_time >= min(u.eval_start for u in units))
                                    & (ctx.labels.reference_time < max(u.eval_end for u in units))]
        predictions = {p: ctx.frozen(train, evaluation, units[0].fit_cutoff, "E09", origin, p) for p in ("F-C", "F-D")}
        for unit in units:
            _, actual, _ = select_partitions(ctx.labels, unit)
            for profile, prediction in predictions.items():
                part = prediction.loc[prediction.sample_id.astype(str).isin(actual.sample_id.astype(str))]
                metrics[unit.id]["candidates"][profile] = diagnostic_metrics(actual, part, ctx.feature_frame(actual, unit.fit_cutoff))
                path = ctx.destination / "units" / unit.id / profile
                path.mkdir(parents=True)
                part.to_csv(path / "predictions.csv", index=False)
                err = error_contributions(actual, part)
                err.to_csv(path / "errors.csv", index=False)
                for target in TARGETS:
                    err.nlargest(20, f"abs_error_{target}").to_csv(path / f"top20_{target}.csv", index=False)
                if unit.horizon:
                    errors.append(err.assign(candidate=profile, origin=origin, horizon=unit.horizon))
            if unit.horizon:
                err = pd.read_csv(reference_root / "units" / unit.id / reference / "errors.csv", dtype={"sample_id": "string"}, float_precision="round_trip")
                errors.append(err.assign(candidate=reference, origin=origin, horizon=unit.horizon))
        atomic_write_json(ctx.destination / "candidate_metrics.json", metrics, overwrite=True)
        print(f"remaining features completed {origin}", flush=True)
    summary = aggregate_grid(metrics)
    gate = acceptance(metrics, summary, reference, load_yaml("configs/optimization_v0_3/acceptance.yaml"))
    atomic_write_json(ctx.destination / "grid_summary.json", summary)
    atomic_write_json(ctx.destination / "acceptance.json", gate)
    errors = pd.concat(errors, ignore_index=True)
    atomic_write_json(ctx.destination / "paired_delta_by_horizon.json", {
        p: paired_week_intervals(errors.loc[errors.candidate.isin([p, reference])], p, reference) for p in ("F-C", "F-D")})
    verify_file_identities(source_identity)
    atomic_write_json(ctx.destination / "source_files.json", source_identity)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", nargs="+", required=True)
    parser.add_argument("--data-config", default="configs/data.local.yaml")
    parser.add_argument("--reference-run", type=Path, default=Path("local/runs/optimization-v0.3-opt07-reference-r2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        snapshot = snapshot_sources(args.output)
        ctx = ProfileContext(args.data_config, args.output, args.sample_index)
        gate = run(ctx, args.reference_run)
        verify_file_identities(ctx.inputs)
        verify_file_identities(ctx.index_audit["sources"])
        verify_file_identities(snapshot["files"])
        atomic_write_json(args.output / "final_status.json", {
            "engineering_status": "G0_EXECUTION_PASS", "model_quality_status": "G1_" + gate["status"] + "_DEVELOPMENT",
            "protected_labels_read": False, "total_target_fits": sum(r["actual_fit_count"] for r in ctx.fit_records),
            "feature_experiments": ["F-C", "F-D"], "source_archive_sha256": snapshot["archive_sha256"]})
    except Exception as exc:
        atomic_write_json(args.output / "final_status.json", {"status": "FAILED", "error": str(exc), "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
