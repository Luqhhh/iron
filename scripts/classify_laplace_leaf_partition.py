"""One frozen, zero-fit development shortlist; never scientific promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from bf_tap_r2.candidate_tiers import classify_candidates
from bf_tap_r2.joint_support_audit import local_output, sha, write_new
from bf_tap_r2.laplace_leaf_partition_experiment import (
    SPEC, COLD_STATUS, REFERENCE_MATERIAL, OLD, inventory, validate_spec,
    original, describe as native_describe, validate_development_units)
from bf_tap_r2.metrics import wmape

TRIAGE = "configs/laplace_leaf_partition/TRIAGE.yaml"
POLICY = "configs/candidate_tiers.yaml"
NATIVE_GIT_COMMIT = "be282454cbb213aff84df3df13ce2d00367e0841"
FROZEN_TRIAGE_SEMANTIC_SHA256 = "d2612b78dd26ab74e3a0586e5337456346e922d1b64d82ad422acf11183d7d20"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def expected_artifacts(units):
    names = {"manifest.json", "data.npz", "ledger.jsonl"}
    for seed in (42, 3407):
        names.update((f"folds-{seed}.npz", f"oof-{seed}.npz"))
    for unit in units:
        key = unit["key"]
        names.add(f"{key}-query.npz")
        for role in ("selector", "refit"):
            names.update((f"{key}-{role}.pkl", f"{key}-{role}-trace.json"))
    return names


def require_native_complete(root, run, spec):
    """Validate the complete native terminal, without loading any model."""
    validate_spec(spec)
    manifest, result, cold = (read_json(run/name) for name in
                              ("manifest.json", "result.json", "cold-readback.json"))
    if (manifest["action"] != "development" or manifest["identity"] != spec["identity"]
            or manifest["spec"] != spec or manifest["sources"] != inventory(root)
            or manifest["git_commit"] != NATIVE_GIT_COMMIT):
        raise ValueError("Exact frozen native science identity required")
    input_names = {"复赛_train/train_samples.csv", "复赛_train/train_features.csv",
                   f"{REFERENCE_MATERIAL}/MANIFEST.json",
                   f"{REFERENCE_MATERIAL}/oof/original/q75-development-reference.npz"}
    if (set(manifest["input_sha256"]) != input_names
            or manifest["input_sha256"] != {name:sha(root/name) for name in input_names}):
        raise ValueError("Complete native input identity required")
    if (result["stage"] != "development" or result["model_fits"] != 20
            or result["formal_promoted"] is not False
            or any(result[field] != 0 for field in
                   ("packages", "full_fits", "confirmation_seeds_consumed", "agent_uploads"))):
        raise ValueError("Native two-seed development only; no prior release or promotion")
    validate_development_units(result["units"], spec)
    expected_models = {}
    for unit in result["units"]:
        epoch = unit["selected_epoch"]
        if (isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch <= 12000
                or unit["control_key"] != f"s{unit['seed']}-f{unit['fold']}-D3_LONG"
                or unit["selector_cap_hit"] != (epoch==12000)):
            raise ValueError("Complete native control/checkpoint roles required")
        expected_models[unit["key"]+"-selector"] = 12000
        expected_models[unit["key"]+"-refit"] = epoch
    models = {model["key"]:model["metadata"] for model in result["models"]}
    tree_count = sum(metadata["fitted_tree_count"] for metadata in models.values())
    if (len(result["models"]) != 20 or set(models) != set(expected_models)
            or any(models[key]["fitted_tree_count"] != epochs for key, epochs in expected_models.items())
            or result["tree_fits"] != tree_count or tree_count > 240000
            or result["original_control_models_reused"] != 20):
        raise ValueError("Twenty complete native selector/refit identities required")
    if (cold["status"] != COLD_STATUS or cold["checked_models"] != 20
            or cold["checked_tree_fits"] != tree_count
            or cold["maximum_cold_difference"] != 0 or cold["new_fits"] != 0
            or cold["model_fits_in_audit"] != 0 or cold["packages"] != 0
            or cold["residual_target_node_statistics_checked"] is not True
            or cold["node_statistic_absolute_tolerance"] != spec["node_statistic_absolute_tolerance"]
            or cold["manifest_sha256"] != sha(run/"manifest.json")
            or cold["result_sha256"] != sha(run/"result.json")):
        raise ValueError("Exact complete zero-fit native cold receipt required")
    if set(result["output_sha256"]) != expected_artifacts(result["units"]):
        raise ValueError("Complete native output inventory required")
    for name, digest in result["output_sha256"].items():
        if sha(run/name) != digest:
            raise ValueError("Hash-bound native output changed")
    return manifest, result, cold


def validate_triage(triage, science_spec, policy_sha256):
    semantic = hashlib.sha256(json.dumps(triage, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    candidate, reference = science_spec["candidate"], science_spec["reference"]
    if (semantic != FROZEN_TRIAGE_SEMANTIC_SHA256
            or triage["policy_sha256"] != policy_sha256
            or triage["split_seeds"] != [42, 3407] or triage["folds"] != 5
            or triage["reference_by_target"] != {"tap_iron":reference}
            or triage["candidates"] != {"tap_iron":[candidate]}
            or triage["tie_preference_by_target"] != {"tap_iron":[candidate]}
            or triage["new_official_sample_label_parses"] != 0
            or triage["official_file_hash_reads_permitted"] is not True):
        raise ValueError("Frozen one-candidate prospective triage/policy required")


def describe(y, prediction, folds, spout):
    y, prediction, folds, spout = (np.asarray(array) for array in (y, prediction, folds, spout))
    if (y.ndim != 1 or not len(y) or prediction.shape != y.shape or folds.shape != y.shape
            or spout.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all()
            or (prediction < 0).any() or not np.abs(y).sum()
            or set(folds.tolist()) != set(range(5)) or set(spout.tolist()) != {1, 2}):
        raise ValueError("Finite aligned complete five-fold/two-spout vectors required")
    return dict(wmape=wmape(y, prediction),
        by_fold={str(f):wmape(y[folds==f], prediction[folds==f]) for f in range(5)},
        by_spout={str(s):wmape(y[spout==s], prediction[spout==s]) for s in (1, 2)})


def load_predictions(run, q, control_run):
    """Rebuild every seed OOF from its exact matched outer query units."""
    expected_rows = len(q["targets"])
    predictions = {}
    for seed in (42, 3407):
        with np.load(run/f"oof-{seed}.npz", allow_pickle=False) as archive:
            prediction = np.asarray(archive["prediction"], dtype=float).copy()
        if prediction.shape != (expected_rows,) or not np.isfinite(prediction).all():
            raise ValueError("Complete finite native OOF prediction required")
        folds = np.asarray(q[f"fold-{seed}"])
        if folds.shape != (expected_rows,) or set(folds.tolist()) != set(range(5)):
            raise ValueError("Complete bound outer folds required")
        rebuilt, coverage = np.full(expected_rows, np.nan), np.zeros(expected_rows, dtype=int)
        for fold in range(5):
            key = f"s{seed}-f{fold}-D3_L1_PARTITION"
            outer, query = np.flatnonzero(folds!=fold), np.flatnonzero(folds==fold)
            with np.load(run/f"{key}-query.npz", allow_pickle=False) as archive:
                parts = {name:archive[name].copy() for name in
                         ("prediction", "outer", "query", "fit", "calibration", "inner_folds")}
            np.testing.assert_array_equal(parts["outer"], outer)
            np.testing.assert_array_equal(parts["query"], query)
            inner = parts["inner_folds"]
            if (inner.shape != outer.shape or not np.issubdtype(inner.dtype, np.integer)
                    or set(inner.tolist()) != set(range(5))):
                raise ValueError("Complete bound calibration roles required")
            np.testing.assert_array_equal(parts["fit"], outer[inner!=0])
            np.testing.assert_array_equal(parts["calibration"], outer[inner==0])
            # original() hash-binds this complete native control before here.
            with np.load(control_run/f"s{seed}-f{fold}-D3_LONG-query.npz", allow_pickle=False) as control:
                for name in ("outer", "query", "fit", "calibration", "inner_folds"):
                    np.testing.assert_array_equal(parts[name], control[name])
            if parts["prediction"].shape != query.shape or not np.isfinite(parts["prediction"]).all():
                raise ValueError("Complete finite native unit prediction required")
            rebuilt[query] = parts["prediction"]
            coverage[query] += 1
        if not np.all(coverage==1):
            raise ValueError("Each outer OOF row must occur exactly once")
        np.testing.assert_array_equal(rebuilt, prediction)
        predictions[seed] = rebuilt
    return predictions


def build_metrics(q, predictions, triage):
    if set(predictions) != {42, 3407}:
        raise ValueError("Both complete split-seed OOF vectors required")
    targets = np.asarray(q["targets"], dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2 or not np.isfinite(targets).all():
        raise ValueError("Aligned bound reference targets required")
    y, spout = targets[:, 0], q["spout"]
    reference = triage["reference_by_target"]["tap_iron"]
    pool = triage["candidates"]["tap_iron"]
    if len(pool) != 1:
        raise ValueError("Exactly one frozen candidate required")
    candidate = pool[0]
    metrics = {"tap_iron": {reference:{}, candidate:{}}}
    for seed in (42, 3407):
        parent_array = np.asarray(q[f"current-{seed}"], dtype=float)
        prediction = np.asarray(predictions[seed], dtype=float)
        if parent_array.shape != targets.shape or prediction.shape != y.shape:
            raise ValueError("Reference/native vector identity mismatch")
        parent, folds = parent_array[:, 0], q[f"fold-{seed}"]
        metrics["tap_iron"][reference][str(seed)] = describe(y, parent, folds, spout)
        blend = .8*parent+.2*prediction
        metrics["tap_iron"][candidate][str(seed)] = describe(y, blend, folds, spout)
    return metrics


def recalculate(root, run, science_spec, triage, result):
    old, q = original(root, science_spec)
    control_run = root/OLD
    control_inputs = {str(control_run/name):science_spec[field] for name, field in (
        ("manifest.json", "original_manifest_sha256"),
        ("result.json", "original_result_sha256"),
        ("cold-readback.json", "original_cold_sha256"))}
    for name, digest in old["output_sha256"].items():
        path = str(control_run/name)
        if path in control_inputs and control_inputs[path] != digest:
            raise ValueError("Original frozen control manifest digest differs")
        control_inputs[path] = digest
    predictions = load_predictions(run, q, root/OLD)
    recomputed = native_describe(q, predictions, old)
    if (recomputed != result["metrics"]
            or float(np.mean([m["gain_vs_Q75"] for m in recomputed.values()])) != result["mean_gain_vs_Q75"]
            or float(np.mean([m["gain_vs_original_long"] for m in recomputed.values()])) != result["mean_gain_vs_original_long"]
            or all(m["gain_vs_Q75"]>0 for m in recomputed.values()) != result["eligible_for_separately_frozen_confirmation"]):
        raise ValueError("Independently complete native metrics/decision differ")
    return build_metrics(q, predictions, triage), control_inputs


def classify(metrics, triage, policy, native_result):
    report = classify_candidates(metrics, triage, policy)
    report.update(formal_scientific_promotion=False, confirmation_seeds_consumed=0,
        full_fits=0, new_fits=0, deserializations=0, model_deserializations=0,
        new_official_sample_label_parses=0, official_file_hash_reads_permitted=True,
        packages=0, agent_uploads=0,
        automatically_scheduled=False,
        original_phase_eligible_for_separately_frozen_confirmation=native_result["eligible_for_separately_frozen_confirmation"],
        interpretation="Two-seed development shortlist only; no four-seed scientific promotion, platform claim, automatic queue change, or release. Original phase decision unchanged.")
    return report


def source_inventory(root):
    paths = (TRIAGE, POLICY, "docs/laplace_leaf_partition/TRIAGE.md",
             "src/bf_tap_r2/candidate_tiers.py", "src/bf_tap_r2/weak_models.py",
             "src/bf_tap_r2/models.py", "src/bf_tap_r2/metrics.py", __file__)
    return {**inventory(root), **{
        str(Path(path).relative_to(root) if Path(path).is_absolute() else path):sha(root/path)
        for path in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    run, out = local_output(root, args.run), local_output(root, args.output)
    sources = source_inventory(root)
    science_spec = read_json(root/SPEC)
    triage = yaml.safe_load((root/TRIAGE).read_text(encoding="utf-8"))
    policy = yaml.safe_load((root/POLICY).read_text(encoding="utf-8"))
    validate_triage(triage, science_spec, sha(root/POLICY))
    manifest, result, cold = require_native_complete(root, run, science_spec)
    inputs = {str(run/name):sha(run/name) for name in ("manifest.json", "result.json", "cold-readback.json")}
    inputs.update({str(run/name):digest for name, digest in result["output_sha256"].items()})
    inputs.update({str(root/name):digest for name, digest in manifest["input_sha256"].items()})
    metrics, control_inputs = recalculate(root, run, science_spec, triage, result)
    inputs.update(control_inputs)
    report = classify(metrics, triage, policy, result)
    if source_inventory(root) != sources or any(sha(Path(name)) != digest for name, digest in inputs.items()):
        raise ValueError("Frozen triage source/native inputs changed during zero-fit readback")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/"manifest.json", dict(sources=sources, input_sha256=inputs,
        native_git_commit=NATIVE_GIT_COMMIT, native_cold_status=cold["status"],
        new_fits=0, deserializations=0, model_deserializations=0,
        new_official_sample_label_parses=0, official_file_hash_reads_permitted=True,
        packages=0, agent_uploads=0))
    write_new(out/"metrics.json", metrics)
    write_new(out/"result.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__=="__main__":
    main()
