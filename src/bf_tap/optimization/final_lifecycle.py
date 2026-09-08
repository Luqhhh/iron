"""Frozen E12-raw holdout report and final training, without search or tuning.

freeze/prepare use metadata and development targets only. report/final require
separate manifest-bound authorization and consume an append-only access ledger.
predict restores a composite bundle and opens only metadata/process inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import (atomic_write_json, build_inference_source_contract,
                        file_identities, file_sha256, runtime_environment,
                        stable_digest, validate_inference_source_contract,
                        verify_file_identities)
from ..availability import freeze_history_origin
from ..config import load_yaml, validate_frozen_contracts
from ..exceptions import ContractError, ProtectedLabelError
from ..features import build_features
from ..io import parse_local_time, read_csv
from ..metrics import score_predictions
from ..models.baseline import DualTargetBaseline
from ..offline import _load_process_sources
from ..protection import load_protection_policy
from ..schema import validate_cross_table_consistency, validate_history, validate_samples
from ..submission import pack_submission, write_submission
from .alignment import aware, stage_alignment
from .config import Candidate, load_experiment, load_feature_selection
from .features import select_candidate_features
from .lifecycle import authorize_protected_operation, PURPOSES
from .process_change import add_process_change_features
from .run import _load_inputs

IDENTITY = "optimization-v0.3-r2/E12-raw/final-v1"
COMPONENTS = {"E09_PROCESS_CHANGE_E02": .8, "E04": .2}
TARGETS = ("tap_iron", "tap_time_len")
META = ["sample_id", "tap_no", "spout_no", "reference_time"]
INPUTS = ("train_samples", "tap_history_train", "operation_hourly", "burden_change",
          "data_dictionary", "test_a_samples", "test_b_samples", "test_c_samples")


def read_json(path):
    return json.loads(Path(path).read_text())


def algorithm():
    baseline, features, semantic = [load_yaml(f"configs/{p}.yaml")
                                    for p in ("baseline", "features", "data_contract")]
    contracts = validate_frozen_contracts(baseline, features, semantic)
    _, candidates = load_experiment("configs/optimization_v0_2/experiment.yaml")
    return {"candidate": "E12-raw", "baseline": baseline, "features": features,
            "semantic": semantic, "contract_digests": contracts,
            "selection": load_feature_selection("configs/optimization_v0_2/features.yaml"),
            "components": {c.id: asdict(c) for c in candidates if c.id in COMPONENTS},
            "weights": COMPONENTS, "component_lower_bound": 0., "output_lower_bound": 0.,
            "calibration": {"algorithm": "none", "tap_iron": 0., "tap_time_len": 0.},
            "fit_eligibility": "reference_time < cutoff and label_available_at <= cutoff",
            "fit_order": "stable reference_time, sample_id",
            "history": "freeze at each model cutoff; per-sample as-of builder",
            "decision": "NO_ACCEPTED_CHALLENGER_USE_PREVIOUSLY_SELECTED_C_REF",
            "holdout_h1_guard": {"max_E_delta": 0., "max_target_wmape_delta": .003}}


def source_files():
    paths = [*Path("src/bf_tap").rglob("*.py"), Path("uv.lock"), Path("pyproject.toml")]
    paths.extend(Path(p) for p in ("configs/baseline.yaml", "configs/features.yaml",
                 "configs/data_contract.yaml", "configs/protection.yaml",
                 "configs/optimization_v0_2/features.yaml", "configs/optimization_v0_2/experiment.yaml"))
    return {str(p): p for p in sorted(paths)}


def sample_metadata(path):
    frame = read_csv(path, usecols=META, required_columns=META, time_columns=["reference_time"])
    validate_samples(frame, labeled=False)
    return frame


def label_metadata(paths, semantic):
    samples = sample_metadata(paths["train_samples"])
    available = semantic["targets"]["available_at_column"]
    history = read_csv(paths["tap_history_train"], usecols=[*META, available],
                       required_columns=[*META, available], time_columns=["reference_time", available])
    validate_samples(history, labeled=False)
    left, right = samples.set_index("sample_id").sort_index(), history.set_index("sample_id").sort_index()
    if not left.equals(right[left.columns]):
        raise ContractError("official sample/history metadata coverage or identities differ")
    result = samples.merge(history[["sample_id", available]].rename(columns={available: "label_available_at"}),
                           on="sample_id", validate="one_to_one")
    if result.label_available_at.isna().any() or (result.label_available_at < result.reference_time).any():
        raise ContractError("invalid label availability mapping")
    return result


def eligibility_identity(meta):
    return {"rows": len(meta), "sample_ids_sha256": stable_digest(meta.sample_id.astype(str).tolist()),
            "reference_min": str(meta.reference_time.min()), "reference_max": str(meta.reference_time.max()),
            "label_available_max": str(meta.label_available_at.max())}


def freeze(data_config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = load_yaml(data_config)["paths"]
    policy = load_protection_policy("configs/protection.yaml")
    inputs = file_identities({key: paths[key] for key in INPUTS})
    stages = {}
    for stage in ("test_a", "test_b", "test_c"):
        samples = sample_metadata(paths[stage + "_samples"])
        stages[stage] = {"rows": len(samples), "start": str(samples.reference_time.min()),
                         "end": str(samples.reference_time.max()),
                         "sample_ids_sha256": stable_digest(samples.sample_id.astype(str).tolist())}
    cutoff = aware(stages["test_a"]["start"])
    if cutoff < policy.protected_end:
        raise ContractError("test_a starts before protected training interval ends")
    a = algorithm()
    metadata = label_metadata(paths, a["semantic"])
    if (metadata.reference_time >= policy.protected_end).any():
        raise ContractError("official training source contains rows outside the frozen training interval")
    final_meta = eligible(metadata, cutoff)
    report_meta = final_meta.loc[final_meta.reference_time >= policy.protected_start]
    manifest = {"schema_version": "frozen-algorithm-v3", "algorithm_identity": IDENTITY,
                "policy_digest": policy.digest, "algorithm": a, "algorithm_sha256": stable_digest(a),
                "inputs": inputs, "source_files": file_identities(source_files()),
                "stages": stages, "final_fit_cutoff": str(cutoff),
                "final_eligibility": eligibility_identity(final_meta),
                "report_eligibility": eligibility_identity(report_meta),
                "holdout_origins": {f"HOLDOUT_H{h}": str(policy.protected_start - pd.DateOffset(months=h-1))
                                    for h in range(1, 5)},
                "protected_interval": [str(policy.protected_start), str(policy.protected_end)],
                "challenger": None, "fallback": "E12-raw", "selection_task": "FROZEN_REFERENCE_ONLY",
                "protected_labels_read": False}
    verify_file_identities(inputs)
    atomic_write_json(output / "frozen_manifest.json", manifest)
    return output / "frozen_manifest.json"


def validate_manifest(path):
    manifest = read_json(path)
    policy = load_protection_policy("configs/protection.yaml")
    expected = algorithm()
    if (manifest.get("schema_version") != "frozen-algorithm-v3"
            or manifest.get("algorithm_identity") != IDENTITY
            or manifest.get("algorithm_sha256") != stable_digest(expected)
            or stable_digest(manifest.get("algorithm")) != stable_digest(expected)
            or manifest.get("policy_digest") != policy.digest
            or manifest.get("challenger") is not None or manifest.get("fallback") != "E12-raw"
            or manifest.get("selection_task") != "FROZEN_REFERENCE_ONLY"
            or set(manifest.get("inputs", {})) != set(INPUTS)
            or set(manifest.get("source_files", {})) != set(source_files())
            or manifest.get("protected_interval") != [str(policy.protected_start), str(policy.protected_end)]
            or manifest.get("holdout_origins") != {
                f"HOLDOUT_H{h}": str(policy.protected_start - pd.DateOffset(months=h-1)) for h in range(1, 5)}):
        raise ProtectedLabelError("frozen manifest or current algorithm configuration mismatch")
    verify_file_identities(manifest["source_files"])
    verify_file_identities(manifest["inputs"])
    times = sample_metadata(manifest["inputs"]["test_a_samples"]["path"]).reference_time
    if (aware(manifest["final_fit_cutoff"]) != times.min()
            or times.min() < policy.protected_end):
        raise ProtectedLabelError("final cutoff differs from actual earliest test reference time")
    return manifest, policy


def eligible(labels, cutoff):
    cutoff = aware(cutoff)
    selected = labels.loc[(labels.reference_time < cutoff) & (labels.label_available_at <= cutoff)].copy()
    if selected.empty:
        raise ContractError("no eligible training labels")
    return selected.sort_values(["reference_time", "sample_id"], kind="mergesort")


def _read_eligible_rows(path, ids):
    # Select row positions using metadata before parsing either target column.
    metadata = read_csv(path, usecols=["sample_id"], required_columns=["sample_id"])
    skip = set((metadata.index[~metadata.sample_id.isin(ids)] + 1).tolist())
    rows = pd.read_csv(path, dtype={"sample_id": "string"}, skiprows=lambda line: line in skip)
    if set(rows.sample_id) != set(ids):
        raise ProtectedLabelError("authorized sample IDs differ from loaded labels")
    rows["reference_time"] = parse_local_time(rows.reference_time, "reference_time")
    return rows


def protected_inputs(manifest_path, authorization, ledger, lifecycle):
    manifest, policy = validate_manifest(manifest_path)
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    # Serialize validation+consumption across processes and disallow a second
    # lifecycle read of this manifest even under a different approval ID.
    with Path(str(ledger) + ".operation.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        previous = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        if any(r.get("lifecycle") == lifecycle and r.get("algorithm_identity") == IDENTITY for r in previous):
            raise ProtectedLabelError("this frozen procedure already consumed this lifecycle")
        access = authorize_protected_operation(
            policy=policy, lifecycle=lifecycle, purpose=PURPOSES.get(lifecycle),
            frozen_manifest=manifest_path, authorization=authorization, ledger=ledger,
            algorithm_identity=IDENTITY)
    paths = {k: v["path"] for k, v in manifest["inputs"].items()}
    mapping = manifest["algorithm"]["semantic"]
    available = mapping["targets"]["available_at_column"]
    meta = label_metadata(paths, mapping)
    cutoff = aware(manifest["final_fit_cutoff"])
    selected = eligible(meta, cutoff)
    # Official training-period labels only; never add actual test outcomes.
    selected = selected.loc[selected.reference_time < policy.protected_end]
    if lifecycle == "holdout_scoring":
        selected = selected.loc[selected.reference_time >= policy.protected_start]
    identity_key = "report_eligibility" if lifecycle == "holdout_scoring" else "final_eligibility"
    if eligibility_identity(selected) != manifest[identity_key]:
        raise ProtectedLabelError("eligible samples differ from the pre-frozen metadata identity")
    labels = _read_eligible_rows(paths["train_samples"], selected.sample_id)
    labels = labels.merge(selected[["sample_id", "label_available_at"]], on="sample_id", validate="one_to_one")
    history = _read_eligible_rows(paths["tap_history_train"], selected.sample_id)
    for column in {mapping["sources"]["history"]["end_time_column"], available}:
        history[column] = parse_local_time(history[column], column)
    history["available_at"] = history[available]
    validate_samples(labels, labeled=True)
    validate_history(history)
    consistency = validate_cross_table_consistency(labels, history)
    verify_file_identities(manifest["inputs"])
    return labels, history, access, consistency


def feature_frame(samples, history, operation, burden, cutoff, a):
    base = build_features(samples, history=history, operation=operation, burden=burden,
                          fit_cutoff=aware(cutoff), config=a["features"])
    return add_process_change_features(base.X, a["selection"]["process_change"],
                                        baseline_value_columns=a["features"]["operation"]["value_columns"])


def component_features(frame, name, a):
    return select_candidate_features(frame, Candidate(**a["components"][name]), a["selection"])


def train_bundle(labels, history, operation, burden, cutoff, manifest, manifest_sha, output, lifecycle, access=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    a = manifest["algorithm"]
    cutoff = aware(cutoff)
    train = eligible(labels, cutoff)
    snapshot = freeze_history_origin(history, cutoff)
    X = feature_frame(train, history, operation, burden, cutoff, a)
    process_contract = build_inference_source_contract(manifest["inputs"],
                           semantic_contract_sha256=a["contract_digests"]["semantic_contract_sha256"])
    training = {"mode": lifecycle, "candidate_id": "E12-raw", "fit_cutoff": str(cutoff),
                "history_cutoff": str(cutoff), "label_available_cutoff": str(cutoff),
                "eligible_rows": len(train), "history_rows": len(snapshot),
                "reference_min": str(train.reference_time.min()), "reference_max": str(train.reference_time.max()),
                "label_available_max": str(train.label_available_at.max()),
                "sample_ids_sha256": stable_digest(train.sample_id.astype(str).tolist()),
                "access": access, "frozen_manifest_sha256": manifest_sha}
    components = {}
    for name in COMPONENTS:
        model = DualTargetBaseline(a["baseline"]["parameters"], tuple(a["baseline"]["categorical_features"]))
        model.fit(component_features(X, name, a), train[list(TARGETS)])
        model.save(output / name, history_snapshot=snapshot, metadata={
            "baseline_config": a["baseline"], "feature_config": a["features"], "semantic_contract": a["semantic"],
            "contract_digests": a["contract_digests"], "inference_source_contract": process_contract,
            "training": {**training, "component": name}, "code_identity": manifest["source_files"],
            "environment": runtime_environment(), "lockfile_sha256": manifest["source_files"]["uv.lock"]["sha256"]})
        components[name] = file_sha256(output / name / "bundle.json")
        atomic_write_json(output / f"fit_{name}.json", {**training, "component": name,
            "actual_target_fits": 2, "parameters_by_target": {t: a["baseline"]["parameters"] for t in TARGETS},
            "selected_iterations": {t: 800 for t in TARGETS}, "iteration_selection": "frozen_no_early_stopping"})
        print(f"fitted {lifecycle} {cutoff} {name}: {len(train)} rows", flush=True)
    atomic_write_json(output / "bundle.json", {"schema_version": "e12-final-composite-v1", "algorithm": a,
                      "training": training, "components": components, "inference_source_contract": process_contract,
                      "frozen_manifest_sha256": manifest_sha})
    atomic_write_json(output / "bundle_identity.json", {"sha256": file_sha256(output / "bundle.json")})
    return training


class Predictor:
    def __init__(self, root):
        self.root = Path(root)
        self.m = read_json(self.root / "bundle.json")
        if read_json(self.root / "bundle_identity.json")["sha256"] != file_sha256(self.root / "bundle.json"):
            raise ContractError("composite bundle metadata hash mismatch")
        a = self.m["algorithm"]
        if (self.m["schema_version"] != "e12-final-composite-v1" or set(self.m["components"]) != set(COMPONENTS)
                or a["weights"] != COMPONENTS or a["candidate"] != "E12-raw"
                or a["calibration"] != {"algorithm": "none", "tap_iron": 0., "tap_time_len": 0.}):
            raise ContractError("unsupported frozen composite algorithm")
        validate_frozen_contracts(a["baseline"], a["features"], a["semantic"])
        self.models, self.history = {}, None
        cutoff = aware(self.m["training"]["fit_cutoff"])
        for name, digest in self.m["components"].items():
            if file_sha256(self.root / name / "bundle.json") != digest:
                raise ContractError("composite component hash mismatch")
            model = DualTargetBaseline.load(self.root / name)
            training = model.bundle_metadata_["training"]
            if (training["component"] != name or any(training[k] != self.m["training"][k] for k in self.m["training"])
                    or model.parameters != a["baseline"]["parameters"]):
                raise ContractError("component training identity mismatch")
            history = model.load_history_snapshot()
            validate_history(history)
            if len(freeze_history_origin(history, cutoff)) != len(history):
                raise ContractError("bundle history crosses fit cutoff")
            if self.history is not None and not self.history.equals(history):
                raise ContractError("component history snapshots differ")
            self.models[name], self.history = model, history

    def predict(self, samples, operation, burden):
        validate_samples(samples, labeled=False)
        cutoff = aware(self.m["training"]["fit_cutoff"])
        if (samples.reference_time < cutoff).any():
            raise ContractError("prediction precedes model cutoff")
        X = feature_frame(samples, self.history, operation, burden, cutoff, self.m["algorithm"])
        result = pd.DataFrame({"sample_id": samples.sample_id.astype("string").to_numpy()})
        values = []
        for name, weight in COMPONENTS.items():
            raw = self.models[name].predict_raw(component_features(X, name, self.m["algorithm"]))
            values.append(weight * np.maximum(raw.to_numpy(), 0.))
        blended = sum(values)
        for i, t in enumerate(TARGETS):
            result[f"pred_{t}"] = np.maximum(blended[:, i], 0.)
        return result


def prepare(manifest_path, output):
    manifest, policy = validate_manifest(manifest_path)
    paths = {k: v["path"] for k, v in manifest["inputs"].items()}
    a = manifest["algorithm"]
    labels, history, op, burden, _ = _load_inputs(data_cfg={"paths": paths}, semantic_cfg=a["semantic"],
                     feature_cfg=a["features"], protection=policy, requested_end=policy.development_label_end_exclusive)
    samples = sample_metadata(paths["train_samples"])
    samples = samples.loc[(samples.reference_time >= policy.protected_start) & (samples.reference_time < policy.protected_end)]
    outputs = {}
    for unit, cutoff in manifest["holdout_origins"].items():
        directory = output / unit
        directory.mkdir()
        train_bundle(labels, history, op, burden, cutoff, manifest, file_sha256(manifest_path),
                     directory / "bundle", "development")
        pred = Predictor(directory / "bundle").predict(samples, op, burden)
        pred.to_csv(directory / "predictions.csv", index=False)
        outputs[unit] = {"predictions_sha256": file_sha256(directory / "predictions.csv"),
                         "bundle_sha256": file_sha256(directory / "bundle/bundle.json"), "cutoff": cutoff}
    verify_file_identities(manifest["inputs"])
    atomic_write_json(output / "prepared.json", {"status": "PASS", "frozen_manifest_sha256": file_sha256(manifest_path),
                      "outputs": outputs, "protected_labels_read": False, "single_target_fits": 16})


def report(manifest_path, prepared, authorization, ledger, output):
    manifest, policy = validate_manifest(manifest_path)
    info = read_json(prepared / "prepared.json")
    if (info.get("status") != "PASS" or info.get("frozen_manifest_sha256") != file_sha256(manifest_path)
            or set(info["outputs"]) != set(manifest["holdout_origins"])):
        raise ProtectedLabelError("holdout predictions were not prepared under this frozen manifest")
    for unit, value in info["outputs"].items():
        if (file_sha256(prepared / unit / "predictions.csv") != value["predictions_sha256"]
                or file_sha256(prepared / unit / "bundle/bundle.json") != value["bundle_sha256"]
                or value["cutoff"] != manifest["holdout_origins"][unit]):
            raise ProtectedLabelError("prepared holdout predictions changed")
    labels, _, access, consistency = protected_inputs(manifest_path, authorization, ledger, "holdout_scoring")
    metrics = {}
    for unit in info["outputs"]:
        pred = pd.read_csv(prepared / unit / "predictions.csv", dtype={"sample_id": "string"})
        pred = pred.loc[pred.sample_id.isin(labels.sample_id)]
        metrics[unit] = score_predictions(labels, pred)
    atomic_write_json(output / "holdout_report.json", {
        "status": "PASS_REPORT", "candidate": "E12-raw", "challenger": None,
        "decision": "USE_PRE_FROZEN_C_REF_NO_CHALLENGER", "metrics": metrics,
        "protected_labels_read": True, "holdout_consumed": True, "access": access,
        "frozen_manifest_sha256": file_sha256(manifest_path), "prepared_sha256": file_sha256(prepared / "prepared.json"),
        "sample_ids_sha256": stable_digest(labels.sample_id.astype(str).tolist()), "rows": len(labels),
        "available_at_max": str(labels.label_available_at.max()), "consistency": consistency,
        "independent_protected_months": 1, "G1": "REFERENCE_REPORT_NOT_NEW_CANDIDATE_ACCEPTANCE"})


def final_train(manifest_path, report_path, authorization, ledger, output):
    manifest, _ = validate_manifest(manifest_path)
    report_value = read_json(report_path)
    if (report_value.get("frozen_manifest_sha256") != file_sha256(manifest_path)
            or report_value.get("status") != "PASS_REPORT"
            or report_value.get("decision") != "USE_PRE_FROZEN_C_REF_NO_CHALLENGER"):
        raise ProtectedLabelError("final training requires the frozen reference report")
    # Bind the report's access entry to the actual append-only ledger.
    entries = [json.loads(line) for line in Path(ledger).read_text().splitlines()]
    if report_value["access"] not in entries:
        raise ProtectedLabelError("holdout report is not bound to the access ledger")
    labels, history, access, consistency = protected_inputs(manifest_path, authorization, ledger, "final_training")
    paths = {k: v["path"] for k, v in manifest["inputs"].items()}
    op, burden, _ = _load_process_sources(paths, manifest["algorithm"]["semantic"], manifest["algorithm"]["features"])
    training = train_bundle(labels, history, op, burden, manifest["final_fit_cutoff"], manifest,
                            file_sha256(manifest_path), output / "bundle", "final_training", access)
    verify_file_identities(manifest["inputs"])
    atomic_write_json(output / "training_manifest.json", {"status": "PASS_FIT", "training": training,
                      "consistency": consistency, "single_target_fits": 4,
                      "protected_labels_read": True, "holdout_report_sha256": file_sha256(report_path),
                      "bundle_sha256": file_sha256(output / "bundle/bundle.json")})


def predict(bundle, data_config, stage, output):
    predictor = Predictor(bundle)
    paths = load_yaml(data_config)["paths"]
    inputs = file_identities({k: paths[k] for k in (stage + "_samples", "operation_hourly", "burden_change")})
    a = predictor.m["algorithm"]
    validate_inference_source_contract(predictor.m["inference_source_contract"], inputs,
                                       semantic_contract_sha256=a["contract_digests"]["semantic_contract_sha256"])
    samples = sample_metadata(paths[stage + "_samples"])
    op, burden, _ = _load_process_sources(paths, a["semantic"], a["features"])
    result = predictor.predict(samples, op, burden)
    repeat = predictor.predict(samples, op, burden)
    if not result.equals(repeat):
        raise ContractError("same-bundle repeated predictions differ")
    result.to_csv(output / "predictions_raw.csv", index=False)
    write_submission(result, samples.sample_id, output / "result.csv")
    archive = pack_submission(output / "result.csv", stage=stage, team_name="Luqhhh", output_dir=output,
                              expected_ids=samples.sample_id)
    verify_file_identities(inputs)
    training = predictor.m["training"]
    alignment = stage_alignment(component="E12-raw", stage=stage, fit_cutoff=training["fit_cutoff"],
                  history_cutoff=training["history_cutoff"], label_available_cutoff=training["label_available_cutoff"],
                  reference_times=samples.reference_time,
                  identity={"source": predictor.m["frozen_manifest_sha256"], "data": stable_digest(inputs),
                            "candidate": stable_digest(a), "component": file_sha256(Path(bundle) / "bundle.json")},
                  development_cells={h: [f"O2024{m:02d}_H{h}" for m in range(6, 11) if m+h-1 <= 10]
                                     for h in range(1, 5)})
    atomic_write_json(output / "stage_alignment.json", alignment)
    (output / "README.txt").write_text(
        "E12-raw / final_training. Official scoring aligns by sample_id.\n"
        "Internal checks additionally enforce sample-table row order. ZIP contains only result.csv.\n"
        "No test targets used; predictions restore frozen models without fitting.\n", encoding="utf-8")
    atomic_write_json(output / "release_manifest.json", {"status": "PASS", "candidate": "E12-raw", "stage": stage,
                      "training": training, "inputs": inputs, "rows": len(result),
                      "bundle_sha256": file_sha256(Path(bundle) / "bundle.json"),
                      "frozen_manifest_sha256": predictor.m["frozen_manifest_sha256"],
                      "raw_sha256": file_sha256(output / "predictions_raw.csv"),
                      "csv_sha256": file_sha256(output / "result.csv"), "zip_sha256": file_sha256(archive),
                      "same_bundle_max_abs_diff": 0., "prediction_label_files_read": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "prepare", "report", "final", "predict"])
    parser.add_argument("--data-config", default="configs/data.local.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--stage", choices=["test_a", "test_b", "test_c"], default="test_a")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.data_config, args.output))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        auth = read_json(args.authorization) if args.authorization else None
        if args.command == "prepare":
            prepare(args.manifest, args.output)
        elif args.command == "report":
            report(args.manifest, args.prepared, auth, args.ledger, args.output)
        elif args.command == "final":
            final_train(args.manifest, args.report, auth, args.ledger, args.output)
        else:
            predict(args.bundle, args.data_config, args.stage, args.output)
        atomic_write_json(args.output / "final_status.json", {"status": "PASS", "command": args.command})
    except Exception as exc:
        atomic_write_json(args.output / "final_status.json", {"status": "FAILED", "command": args.command,
                          "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
