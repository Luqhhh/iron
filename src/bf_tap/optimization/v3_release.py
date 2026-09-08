"""Explicitly selected development release. No protected training or platform upload."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json, file_identities, file_sha256, stable_digest, verify_file_identities
from ..availability import freeze_history_origin
from ..config import load_yaml, validate_frozen_contracts
from ..data import normalize_event_source
from ..exceptions import ContractError
from ..features import build_features
from ..io import read_csv
from ..submission import pack_submission, validate_submission, write_submission
from .alignment import aware, stage_alignment
from .calibration import inner_blocks
from .config import Candidate
from .features import select_candidate_features
from .indexed_features import INDEX_COLUMNS, known_index_features, load_known_index, normalize_index
from .process_change import add_process_change_features
from .search_models import SingleTargetModel, search_configurations
from .v3_followup import fit_calibration, fit_cross_target, read, snapshot_sources
from .v3_remaining_features import ProfileContext
from .v3_run import TARGETS


def checked_relative(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ContractError("bundle component path escapes its directory or is missing")
    return path


def save_release(ctx, models, provenance, cutoff, output, selection, stages):
    output.mkdir(parents=True, exist_ok=False)
    components = {}
    for target, source in models.items():
        destination = output / target
        shutil.copytree(source, destination)
        components[target] = target
    history = freeze_history_origin(ctx.history, cutoff)
    history.to_csv(output / "history_snapshot.csv", index=False)
    atomic_write_json(output / "calibration_provenance.json", provenance)
    schema = {
        "schema_version": "development-fc-catboost-bundle-v1", "candidate": "CB-FC-CVcal",
        "lifecycle": "development", "selection": selection, "fit_cutoff": str(cutoff),
        "history_cutoff": str(cutoff), "label_available_cutoff": str(cutoff),
        "components": components, "calibration": "calibration_provenance.json",
        "target_configs": {t: next(c for c in search_configurations("configs/optimization_v0_3/models") if c["id"] == cid)
                           for t, cid in (("tap_iron", "CB08"), ("tap_time_len", "CB02"))},
        "baseline": ctx.baseline, "features": ctx.features, "semantic": ctx.semantic,
        "feature_selection": ctx.selection, "base_candidate": asdict(ctx.candidates["E09"]),
        "profile_registration": ctx.profiles,
        "source_inputs": ctx.inputs, "stages": stages, "code": ctx.code,
        "train_samples_sha256": stable_digest(list(inner_blocks(ctx.labels, cutoff)["outer_train"].sample_id.astype(str))),
        "files": {str(p.relative_to(output)): file_sha256(p) for p in output.rglob("*") if p.is_file()},
    }
    atomic_write_json(output / "bundle.json", schema)
    atomic_write_json(output / "bundle_identity.json", {"bundle_json_sha256": file_sha256(output / "bundle.json")})


class ReleasePredictor:
    def __init__(self, root):
        self.root = Path(root)
        m = self.manifest = read(self.root / "bundle.json")
        if file_sha256(self.root / "bundle.json") != read(self.root / "bundle_identity.json")["bundle_json_sha256"]:
            raise ContractError("release metadata identity mismatch")
        if (m["schema_version"] != "development-fc-catboost-bundle-v1" or m["candidate"] != "CB-FC-CVcal"
                or m["lifecycle"] != "development" or set(m["components"]) != set(TARGETS)
                or not (m["fit_cutoff"] == m["history_cutoff"] == m["label_available_cutoff"])):
            raise ContractError("release candidate/lifecycle/component/cutoff mismatch")
        self.cutoff = aware(m["fit_cutoff"])
        if self.cutoff > aware("2024-11-01T00:00:00+08:00"):
            raise ContractError("development release cannot include protected training")
        validate_frozen_contracts(m["baseline"], m["features"], m["semantic"])
        for name, digest in m["files"].items():
            if file_sha256(checked_relative(self.root, name)) != digest:
                raise ContractError("release component file identity mismatch")
        self.models = {}
        for target, directory in m["components"].items():
            for filename in ("model.bin", "bundle.json", "bundle_identity.json"):
                if f"{directory}/{filename}" not in m["files"]:
                    raise ContractError("unmanifested target component")
            model = SingleTargetModel.load(checked_relative(self.root, f"{directory}/bundle.json").parent)
            if (model.target != target or model.model_type != "tunable_catboost_l1"
                    or model.parameters != m["target_configs"][target]["parameters"]
                    or model.identity.get("feature_increment") != "F-C"
                    or any(model.identity[k] != str(self.cutoff) for k in ("fit_cutoff", "history_cutoff", "label_available_cutoff"))
                    or model.identity["train_samples_sha256"] != m["train_samples_sha256"]
                    or model.identity["data_sha256"] != stable_digest(m["source_inputs"])):
                raise ContractError("release target training identity mismatch")
            self.models[target] = model
        if "history_snapshot.csv" not in m["files"] or m["calibration"] not in m["files"]:
            raise ContractError("release lacks history/calibration provenance")
        self.history = pd.read_csv(self.root / "history_snapshot.csv", dtype={"sample_id": "string"})
        for column in ("reference_time", "tap_end_time", "available_at"):
            self.history[column] = pd.to_datetime(self.history[column], utc=True).dt.tz_convert("Asia/Shanghai")
        if (self.history.available_at > self.cutoff).any() or (self.history.reference_time >= self.cutoff).any():
            raise ContractError("release history crossed frozen cutoff")
        p = self.provenance = read(checked_relative(self.root, m["calibration"]))
        if (p["candidate"] != m["candidate"] or aware(p["outer_cutoff"]) != self.cutoff
                or aware(p["inner_fit_cutoff"]) != self.cutoff - pd.Timedelta(days=28)
                or p["calibration_fit_overlap"] or p["calibration_available_after_origin"]
                or p["median_prediction_minus_actual"]["tap_iron"] != 0
                or p["selected_iterations"] != self.models["tap_time_len"].selected_iterations
                or p["selected_iterations_source"] != self.models["tap_time_len"].identity["selected_iterations_source"]
                or (p["sample_count"] >= 100 and aware(p["label_available_max"]) >= self.cutoff)):
            raise ContractError("release calibration provenance mismatch")
        self.residual = float(p["median_prediction_minus_actual"]["tap_time_len"])
        if not np.isfinite(self.residual) or (p["sample_count"] < 100 and self.residual != 0):
            raise ContractError("invalid calibration or zero-fallback state")

    def predict(self, samples, operation, burden, index):
        samples = normalize_index(samples[INDEX_COLUMNS])
        if (samples.reference_time < self.cutoff).any():
            raise ContractError("release prediction precedes fit cutoff")
        m = self.manifest
        X = build_features(samples, operation=operation, burden=burden, history=self.history,
                           fit_cutoff=self.cutoff, config=m["features"]).X
        X = add_process_change_features(X, m["feature_selection"]["process_change"],
                                       baseline_value_columns=m["features"]["operation"]["value_columns"])
        X = select_candidate_features(X, Candidate(**m["base_candidate"]), m["feature_selection"])
        X = pd.concat([X, known_index_features(samples, index)], axis=1)
        result = samples[["sample_id"]].copy()
        for target, model in self.models.items():
            result[f"pred_{target}"] = np.maximum(0, model.predict(X))
        result.pred_tap_time_len = np.maximum(0, result.pred_tap_time_len - self.residual)
        validate_submission(result, samples.sample_id)
        return result


def predict_stage(bundle, data_config, stage, index_paths, destination):
    destination.mkdir(parents=True, exist_ok=False)
    predictor = ReleasePredictor(bundle)
    m = predictor.manifest
    data = load_yaml(data_config)
    if stage not in m["stages"]:
        raise ContractError("undeclared release stage")
    spec = m["stages"][stage]
    paths = data["paths"]
    roles = spec["index_roles"]
    if {Path(p).resolve() for p in index_paths} != {Path(paths[r]).resolve() for r in roles}:
        raise ContractError("stage index roles must be supplied explicitly")
    inputs = file_identities({r: paths[r] for r in [*roles, "operation_hourly", "burden_change"]})
    if inputs != spec["inputs"]:
        raise ContractError("stage/source identity mismatch")
    samples = normalize_index(read_csv(paths[f"{stage}_samples"], usecols=INDEX_COLUMNS, time_columns=["reference_time"]))
    index, index_audit = load_known_index(index_paths)
    frames = {}
    for kind, role in (("operation", "operation_hourly"), ("burden", "burden_change")):
        mapping = m["semantic"]["sources"][kind]
        frames[kind] = normalize_event_source(read_csv(paths[role]), event_time_column=mapping["event_time_column"],
            available_at_column=mapping["available_at_column"], missing_markers=mapping["missing_markers"],
            value_columns=m["features"][kind]["value_columns"])
    result = predictor.predict(samples, frames["operation"], frames["burden"], index)
    chunks = pd.concat([predictor.predict(part, frames["operation"], frames["burden"], index)
                        for part in (samples.iloc[::2], samples.iloc[1::2]) if len(part)]).sort_index()
    if not np.array_equal(result[TARGET_PRED].to_numpy(), chunks[TARGET_PRED].to_numpy()):
        raise ContractError("release batch/chunk prediction mismatch")
    write_submission(result, samples.sample_id, destination / "result.csv")
    result.to_csv(destination / "unrounded_predictions.csv", index=False)
    verify_file_identities(inputs)
    atomic_write_json(destination / "prediction_manifest.json", {
        "candidate": m["candidate"], "stage": stage, "bundle_sha256": file_sha256(Path(bundle) / "bundle.json"),
        "inputs": inputs, "index_manifest": index_audit, "rows": len(result), "chunk_max_difference": 0.0,
        "result_sha256": file_sha256(destination / "result.csv"), "protected_labels_read": False})
    atomic_write_json(destination / "stage_alignment.json", stage_alignment(
        component="dual_target_fc_calibrated", stage=stage, fit_cutoff=predictor.cutoff,
        history_cutoff=predictor.cutoff, label_available_cutoff=predictor.cutoff, reference_times=samples.reference_time,
        identity={"source": stable_digest(m["code"]), "data": stable_digest(inputs), "candidate": m["candidate"],
                  "component": file_sha256(Path(bundle) / "bundle.json")},
        development_cells={h: [f"O2024{month:02d}_H{h}" for month in range(6, 11) if month + h - 1 <= 10] for h in range(1, 5)}))
    return result


TARGET_PRED = [f"pred_{t}" for t in TARGETS]


def train_release(output, data_config, index_paths):
    output.mkdir(parents=True, exist_ok=False)
    snapshot = snapshot_sources(output)
    selection = load_yaml("configs/optimization_v0_3/release_selection.yaml")
    if (selection["active_candidate"] != "CB-FC-CVcal" or selection["lifecycle"] != "development"
            or selection["selection_basis"] != "explicit_user_override_of_failed_J_margin"
            or selection["protected_access_authorized"] or selection["platform_upload_authorized"]
            or selection["approval"] != "user_confirmed_replace_current_candidate_and_generate_platform_test_package"):
        raise ContractError("explicit development release selection is missing")
    evidence = Path(selection["development_evidence_run"])
    evidence_ids = file_identities({str(p): p for p in evidence.rglob("*") if p.is_file()})
    gate = read(evidence / "acceptance.json")["candidates"][selection["active_candidate"]]
    if read(evidence / "final_status.json")["engineering_status"] != "G0_EXECUTION_PASS" or gate["pass"] or gate["checks"]["J"]:
        raise ContractError("release override evidence differs from reviewed result")
    ctx = ProfileContext(data_config, output, index_paths)
    old = read(evidence / "resolved_config.json")
    if any(old[k] != value for k, value in (("inputs", ctx.inputs), ("features", ctx.features), ("selection", ctx.selection), ("search", search_configurations("configs/optimization_v0_3/models")))):
        raise ContractError("release data/features/parameters differ from development evidence")
    cutoff = ctx.policy.validate_development_read(selection["fit_cutoff"])
    if cutoff != aware("2024-11-01T00:00:00+08:00"):
        raise ContractError("this release has a fixed development cutoff")
    stages = {}
    for stage in ("test_a", "test_b", "test_c"):
        roles = ctx.profiles["profiles"]["F-C"]["deployment_roles"][stage]
        stages[stage] = {"index_roles": roles, "inputs": file_identities({r: ctx.data["paths"][r] for r in [*roles, "operation_hourly", "burden_change"]})}
    models, estimators = {}, {}
    configs = {c["id"]: c for c in search_configurations("configs/optimization_v0_3/models")}
    for target, cid in (("tap_iron", "CB08"), ("tap_time_len", "CB02")):
        model, iterations, source = fit_cross_target(ctx, configs[cid], target, cutoff, "RELEASE", increment="F-C", candidate="CB-FC-raw")
        estimators[target] = model
        models[target] = output / "bundles" / "RELEASE" / "CB-FC-raw" / target / "outer"
        print(f"release target fitted {target}: {iterations} iterations", flush=True)
    queries = normalize_index(read_csv(ctx.data["paths"]["test_a_samples"], usecols=INDEX_COLUMNS, time_columns=["reference_time"]))
    provenance = fit_calibration(ctx, configs["CB02"], iterations, cutoff, "RELEASE", "CB-FC-CVcal", list(queries.sample_id), increment="F-C", selection_source=source)
    save_release(ctx, models, provenance, cutoff, output / "bundle", selection, stages)
    atomic_write_json(output / "selection_override.json", {"selection": selection, "original_gate": gate, "evidence": evidence_ids})
    # Independent in-memory prediction from the training objects, then cold-process raw-input restoration.
    stage_index_paths = [ctx.data["paths"][r] for r in stages["test_a"]["index_roles"]]
    stage_index, _ = load_known_index(stage_index_paths)
    X = pd.concat([ctx.X(queries, cutoff), known_index_features(queries, stage_index)], axis=1)
    expected = queries[["sample_id"]].copy()
    for target, model in estimators.items():
        expected[f"pred_{target}"] = np.maximum(0, model.predict(X))
    expected.pred_tap_time_len = np.maximum(0, expected.pred_tap_time_len - provenance["median_prediction_minus_actual"]["tap_time_len"])
    subprocess.run([sys.executable, "-m", "bf_tap.optimization.v3_release", "predict", "--bundle", str(output / "bundle"),
                    "--data-config", data_config, "--stage", "test_a", "--sample-index", *stage_index_paths,
                    "--output", str(output / "test_a")], check=True)
    restored = pd.read_csv(output / "test_a" / "unrounded_predictions.csv", dtype={"sample_id": "string"}, float_precision="round_trip")
    validate_submission(restored, expected.sample_id)
    difference = float(np.max(np.abs(expected[TARGET_PRED].to_numpy() - restored[TARGET_PRED].to_numpy())))
    if difference > 1e-12:
        raise ContractError("cold-process release predictions differ from training objects")
    verify_file_identities(ctx.inputs)
    verify_file_identities(evidence_ids)
    verify_file_identities(snapshot["files"])
    for spec in stages.values():
        verify_file_identities(spec["inputs"])
    zip_path = pack_submission(output / "test_a" / "result.csv", stage="test_a", team_name="Luqhhh",
                               output_dir=output / "submission", expected_ids=expected.sample_id)
    atomic_write_json(output / "final_status.json", {
        "engineering_status": "G0_DEVELOPMENT_RELEASE_PASS", "model_quality_status": "G1_FAIL_DEVELOPMENT_USER_OVERRIDE",
        "candidate": "CB-FC-CVcal", "fit_cutoff": str(cutoff), "training_rows": len(inner_blocks(ctx.labels, cutoff)["outer_train"]),
        "single_target_fits": sum(r["actual_fit_count"] for r in ctx.fit_records),
        "selected_iterations": {t: m.selected_iterations for t, m in estimators.items()},
        "time_calibration": provenance["median_prediction_minus_actual"]["tap_time_len"],
        "cold_process_max_difference": difference, "test_a_rows": len(expected),
        "zip_path": str(zip_path), "zip_sha256": file_sha256(zip_path),
        "bundle_sha256": file_sha256(output / "bundle" / "bundle.json"),
        "source_archive_sha256": snapshot["archive_sha256"], "protected_labels_read": False, "platform_uploaded": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-config", default="configs/data.local.yaml")
    parser.add_argument("--sample-index", nargs="+", required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--stage", choices=["test_a", "test_b", "test_c"])
    args = parser.parse_args()
    if args.output.exists():
        raise ContractError("refusing to overwrite a release run")
    try:
        if args.command == "train":
            train_release(args.output, args.data_config, args.sample_index)
        else:
            if args.bundle is None or args.stage is None:
                raise ContractError("prediction requires bundle and declared stage")
            predict_stage(args.bundle, args.data_config, args.stage, args.sample_index, args.output)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output / "final_status.json", {"status": "FAILED", "error": str(exc), "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
