"""Frozen residual-L1 partition trial with complete, no-fit cold auditing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from .data import FEATURES
from .joint_support_audit import local_output, sha, write_new
from .laplace_epoch_experiment import synthetic_arrays, save_arrays, folds_for
from .laplace_leaf_experiment import check_prediction, resource, log, scalar_gain
from . import laplace_leaf_long_experiment as previous
from .laplace_leaf_partition import ResidualL1PartitionRegressor
from .metrics import wmape
from .submission import deny_training_reads
from .v2_release import load_v2

SPEC = "configs/laplace_leaf_partition/SPEC.json"
OLD = "local/runs/laplace-leaf-long-20261002/development-r1"
RUN = "local/runs/laplace-leaf-partition-20261002"
REFERENCE_MATERIAL = "local/runs/laplace-q75-review-20261002/intake-r1/material"
EXTRA = (SPEC, "docs/laplace_leaf_partition/PREREGISTRATION.md",
         "src/bf_tap_r2/laplace_leaf_partition.py", __file__,
         "scripts/run_laplace_leaf_partition_sequence.py")
# Bind every preregistered field, not only the recipe's most visible knobs.
FROZEN_SPEC_SEMANTIC_SHA256 = "ffebb61dfe595f935946588f63ba1fc612119eabcc5db5f414b9f907a0740c7d"
COLD_STATUS = "passed_residual_L1_partitions_every_leaf_complete_trajectories_cold_OOF_and_ledger"
NODE_STATISTIC_ABSOLUTE_TOLERANCE = 1e-12


def inventory(root):
    return {**previous.inventory(root), **{
        str(Path(p).relative_to(root) if Path(p).is_absolute() else p): sha(root/p)
        for p in EXTRA}}


def validate_spec(spec):
    semantic = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (semantic != FROZEN_SPEC_SEMANTIC_SHA256
            or spec["node_statistic_absolute_tolerance"] != NODE_STATISTIC_ABSOLUTE_TOLERANCE):
        raise ValueError("Frozen residual-L1 partition scope differs")


def verify_reference_identity(root, spec):
    # These digests were independently checked against the original hash-bound
    # intake ZIP, not copied from a possibly replaced self-declaring manifest.
    files = {"reference_material_manifest_sha256": f"{REFERENCE_MATERIAL}/MANIFEST.json",
             "reference_development_oof_sha256": f"{REFERENCE_MATERIAL}/oof/original/q75-development-reference.npz"}
    for field, name in files.items():
        if sha(root/name) != spec[field]:
            raise ValueError("Original Q75 material manifest or complete OOF changed")


def original(root, spec):
    """Bind the complete native long control before accepting any of its files."""
    run = root/OLD
    manifest = json.loads((run/"manifest.json").read_text())
    result = json.loads((run/"result.json").read_text())
    cold = json.loads((run/"cold-readback.json").read_text())
    previous.validate_spec(manifest["spec"])
    if (sha(run/"manifest.json") != spec["original_manifest_sha256"]
            or sha(run/"result.json") != spec["original_result_sha256"]
            or sha(run/"cold-readback.json") != spec["original_cold_sha256"]
            or manifest["sources"] != previous.inventory(root)
            or manifest["action"] != "development" or result["stage"] != "development"
            or cold["status"] != "passed_exact_original_prefix_complete_leaf_updates_trajectories_cold_OOF_and_ledger"
            or cold["checked_models"] != 20 or cold["prefix_models_checked"] != 10
            or cold["maximum_cold_difference"] != 0 or cold["new_fits"] != 0
            or cold["manifest_sha256"] != sha(run/"manifest.json")
            or cold["result_sha256"] != sha(run/"result.json")
            or result["model_fits"] != 20):
        raise ValueError("Complete native cold-audited long control required")
    for name, digest in result["output_sha256"].items():
        if sha(run/name) != digest:
            raise ValueError("Bound long-control artifact changed")
    verify_reference_identity(root, spec)
    _, q = previous.original(root, manifest["spec"])
    if manifest["spec"]["reference_zip_sha256"] != spec["reference_zip_sha256"]:
        raise ValueError("Residual partition reference package differs")
    return result, q


def fit_model(out, key, x, y, epochs, calibration=None):
    log(out, dict(event="model_started", key=key, epochs=epochs, fit_rows=len(y)))
    model = ResidualL1PartitionRegressor().fit(
        x, y, epochs=epochs, validation=calibration, on_epoch=resource)
    with (out/f"{key}.pkl").open("xb") as handle:
        pickle.dump(model, handle, protocol=5)
    item = dict(key=key, metadata=model.metadata())
    write_new(out/f"{key}-trace.json", dict(metadata=model.metadata(), history=model.history_))
    log(out, dict(event="model_completed", **item))
    return model, item


def engineering(out, spec):
    x, y = synthetic_arrays()
    save_arrays(out/"data.npz", x=x, y=y)
    models = []
    short, meta = fit_model(out, "SHORT-selector", x[:108], y[:108], 300, (x[108:144], y[108:144]))
    models.append(meta)
    long, meta = fit_model(out, "LONG-selector", x[:108], y[:108], 600, (x[108:144], y[108:144]))
    models.append(meta)
    # This is the new recipe's internal short/long prefix, not the old sign recipe.
    previous.check_prefix(long, short, 300)
    units = []
    for name, selector in (("SHORT", short), ("LONG", long)):
        refit, meta = fit_model(out, name+"-refit", x[:144], y[:144], selector.selected_epoch_)
        models.append(meta)
        save_arrays(out/f"{name}-query.npz", prediction=refit.predict(x[144:]))
        units.append(dict(key=name, selected_epoch=selector.selected_epoch_))
    return dict(stage="engineering", models=models, units=units,
                prefix_models_checked=1, official_sample_reads=0)


def describe(q, predictions, old):
    if set(predictions) != {42, 3407}:
        raise ValueError("Both complete registered seed vectors required")
    y, spout, result = q["targets"][:, 0], q["spout"], {}
    for seed, prediction in predictions.items():
        parent, folds = q[f"current-{seed}"][:, 0], q[f"fold-{seed}"]
        gain = scalar_gain(y, parent, prediction)
        result[str(seed)] = dict(
            iron_wmape=wmape(y, prediction), gain_vs_Q75=gain,
            gain_vs_original_long=gain-old["metrics"][str(seed)]["gain_vs_Q75"],
            folds={str(f): scalar_gain(y[folds==f], parent[folds==f], prediction[folds==f]) for f in range(5)},
            spouts={str(s): scalar_gain(y[spout==s], parent[spout==s], prediction[spout==s]) for s in (1, 2)})
    return result


def matched_parts(run, key, outer, query, fitting, calibration, inner):
    with np.load(run/f"{key}-query.npz", allow_pickle=False) as parts:
        for name, value in dict(outer=outer, query=query, fit=fitting,
                                calibration=calibration, inner_folds=inner).items():
            np.testing.assert_array_equal(parts[name], value)


def validate_development_units(units, spec):
    expected = [(f"s{seed}-f{fold}-D3_L1_PARTITION", seed, fold)
                for seed in spec["split_seeds"] for fold in range(5)]
    observed = [(unit.get("key"), unit.get("seed"), unit.get("fold")) for unit in units]
    if observed != expected:
        raise ValueError("Registered keys and seed/fold roles must match exactly")


def development(root, out, spec):
    old, q = original(root, spec)
    frame = load_v2(root/"复赛_train", "train", 2754)
    x = np.column_stack((frame[list(FEATURES)].to_numpy(float), frame.spout_no==1, frame.spout_no==2))
    y, ids = frame.tap_iron.to_numpy(float), frame.sample_id.to_numpy(str)
    np.testing.assert_array_equal(ids, q["ids"])
    np.testing.assert_array_equal(x[:, :21], q["numeric"])
    np.testing.assert_array_equal(frame.spout_no.to_numpy(int), q["spout"])
    np.testing.assert_array_equal(frame[["tap_iron", "tap_time_len"]].to_numpy(float), q["targets"])
    save_arrays(out/"data.npz", x=x, y=y, ids=ids)
    models, units, predictions = [], [], {}
    for seed in spec["split_seeds"]:
        folds = folds_for(frame, seed)
        np.testing.assert_array_equal(folds, q[f"fold-{seed}"])
        predictions[seed] = np.full(len(y), np.nan)
        save_arrays(out/f"folds-{seed}.npz", folds=folds)
        for fold in range(5):
            key, control_key = f"s{seed}-f{fold}-D3_L1_PARTITION", f"s{seed}-f{fold}-D3_LONG"
            outer, query = np.flatnonzero(folds!=fold), np.flatnonzero(folds==fold)
            inner = folds_for(frame.iloc[outer], 27001)
            fitting, calibration = outer[inner!=0], outer[inner==0]
            matched_parts(root/OLD, control_key, outer, query, fitting, calibration, inner)
            selector, meta = fit_model(out, key+"-selector", x[fitting], y[fitting], 12000,
                                       (x[calibration], y[calibration]))
            models.append(meta)
            epoch = selector.selected_epoch_
            refit, meta = fit_model(out, key+"-refit", x[outer], y[outer], epoch)
            models.append(meta)
            prediction = refit.predict(x[query])
            predictions[seed][query] = prediction
            save_arrays(out/f"{key}-query.npz", prediction=prediction, outer=outer, query=query,
                        fit=fitting, calibration=calibration, inner_folds=inner)
            units.append(dict(key=key, control_key=control_key, seed=seed, fold=fold,
                              selected_epoch=epoch, selector_cap_hit=epoch==12000))
            del selector, refit
    for seed, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError("Incomplete residual-L1 partition OOF")
        save_arrays(out/f"oof-{seed}.npz", prediction=prediction)
    metrics = describe(q, predictions, old)
    return dict(stage="development", models=models, units=units, metrics=metrics,
        mean_gain_vs_Q75=float(np.mean([m["gain_vs_Q75"] for m in metrics.values()])),
        mean_gain_vs_original_long=float(np.mean([m["gain_vs_original_long"] for m in metrics.values()])),
        eligible_for_separately_frozen_confirmation=all(m["gain_vs_Q75"]>0 for m in metrics.values()),
        original_control_models_reused=20, prefix_models_checked=0, official_sample_reads=1)


def verify_node_statistics(tree, z, residual):
    """Witness signed residual/L1 node statistics without fitting a tree."""
    residual = np.asarray(residual, float)
    if residual.shape != (len(z),) or not np.isfinite(residual).all():
        raise ValueError("Invalid independent residual witness")
    path = tree.decision_path(z).tocsc()
    if path.shape != (len(z), tree.tree_.node_count):
        raise ValueError("Saved training-node path shape differs")
    for node in range(tree.tree_.node_count):
        rows = path.indices[path.indptr[node]:path.indptr[node+1]]
        if not len(rows) or len(rows) != tree.tree_.n_node_samples[node]:
            raise ValueError("Training node sample count differs")
        ordered = np.sort(residual[rows])
        median = float(ordered[(len(ordered)-1)//2]+ordered[len(ordered)//2])/2
        impurity = float(np.abs(residual[rows]-median).mean())
        value = np.asarray(tree.tree_.value[node]).ravel()
        if (len(value) != 1 or not np.isfinite(value).all()
                or not np.isfinite(tree.tree_.impurity[node])
                or abs(float(value[0])-median) > NODE_STATISTIC_ABSOLUTE_TOLERANCE
                or abs(float(tree.tree_.impurity[node])-impurity) > NODE_STATISTIC_ABSOLUTE_TOLERANCE):
            raise ValueError("Residual-target node median or L1 impurity witness differs")


def checked_model(out, key, x, y, query, query_y=None):
    """Audit saved structures/updates independently, never fitting another tree."""
    resource()
    # The caller first hash-binds all artifacts produced by this native runner.
    with (out/f"{key}.pkl").open("rb") as handle:
        model = pickle.load(handle)
    resource()
    if type(model) is not ResidualL1PartitionRegressor or model.depth != 3:
        raise ValueError("Native residual-L1 partition model required")
    trace = json.loads((out/f"{key}-trace.json").read_text())
    if model.metadata() != trace["metadata"] or model.history_ != trace["history"]:
        raise ValueError("Model trace identity differs")
    xs = np.where(x.std(axis=0)>0, x.std(axis=0), 1.)
    np.testing.assert_array_equal(model.x_mean_, x.mean(axis=0))
    np.testing.assert_array_equal(model.x_scale_, xs)
    if model.y_median_ != float(np.median(y)) or model.y_scale_ != (float(y.std()) or 1.) or model.fit_rows_ != len(y):
        raise ValueError("Training-only normalization differs")
    z, v = (x-model.x_mean_)/model.x_scale_, (query-model.x_mean_)/model.x_scale_
    sy, mu, vm = (y-model.y_median_)/model.y_scale_, np.zeros(len(y)), np.zeros(len(query))
    previous_loss = float(np.abs(sy).mean())
    if model.initial_train_mae_ != previous_loss:
        raise ValueError("Initial training loss differs")
    initial_cal = float(np.abs(query_y-model.y_median_).mean()) if query_y is not None else None
    if model.initial_calibration_mae_ != initial_cal:
        raise ValueError("Initial calibration differs")
    if not 0 <= len(model.trees_) <= 12000:
        raise ValueError("Saved tree horizon differs")
    for epoch, (tree, values, event) in enumerate(zip(model.trees_, model.leaf_values_, model.history_, strict=True), 1):
        if (tree.max_depth != 3 or tree.min_samples_leaf != 20 or tree.random_state != 42
                or tree.criterion != "absolute_error"):
            raise ValueError("Frozen residual-L1 tree parameters differ")
        leaves, residual = tree.apply(z), sy-mu
        if values.shape != (tree.tree_.node_count,):
            raise ValueError("Frozen leaf-value shape differs")
        # Witness the fitted residual target, including internal nodes. This
        # does not refit or claim to reconstruct every greedy split decision.
        verify_node_statistics(tree, z, residual)
        for leaf in np.unique(leaves):
            ordered = np.sort(residual[leaves==leaf])
            # Independent sorted-order statistic, not the fitting helper.
            median = float(ordered[(len(ordered)-1)//2]+ordered[len(ordered)//2])/2
            if values[leaf] != median:
                raise ValueError("Independent leaf median differs")
        update = .05*values[leaves]
        mu += update
        vm += .05*values[tree.apply(v)]
        loss = float(np.abs(sy-mu).mean())
        expected = dict(epoch=epoch, train_mae_normalized=loss,
                        maximum_abs_update=float(np.max(np.abs(update))),
                        nonzero_update_rows=int(np.count_nonzero(update)))
        if query_y is not None:
            expected["calibration_mae"] = float(np.abs(query_y-(vm*model.y_scale_+model.y_median_)).mean())
        if event != expected or not np.isfinite(loss) or loss > previous_loss+1e-12:
            raise ValueError("Independent complete L1 trajectory differs")
        previous_loss = loss
        if epoch % 100 == 0:
            resource()
    if query_y is not None:
        losses = [initial_cal, *[event["calibration_mae"] for event in model.history_]]
        selected = min(range(len(losses)), key=lambda i: (losses[i], i))
        if not np.isfinite(losses).all() or model.selected_epoch_ != selected:
            raise ValueError("Independent checkpoint selection differs")
    elif model.selected_epoch_ != len(model.trees_):
        raise ValueError("Fresh full-outer refit horizon differs")
    return model, vm*model.y_scale_+model.y_median_


def verify_ledger(out, result, checked, expected_count, budget):
    ledger = [json.loads(line) for line in (out/"ledger.jsonl").read_text().splitlines()]
    starts = [a for a in ledger if a["event"]=="model_started"]
    ends = [a for a in ledger if a["event"]=="model_completed"]
    keys = [a["key"] for a in starts]
    registered = {item["key"]: item["metadata"] for item in result["models"]}
    tree_count = sum(metadata["fitted_tree_count"] for metadata in checked.values())
    if (keys != [a["key"] for a in ends] or len(keys) != expected_count
            or len(set(keys)) != expected_count or len(checked) != expected_count
            or len(result["models"]) != expected_count or registered != checked
            or result["model_fits"] != expected_count or result["tree_fits"] != tree_count
            or tree_count > budget or len(ledger) != 2*expected_count):
        raise ValueError("Closed actual residual-partition model/tree ledger required")
    for start, end in zip(starts, ends, strict=True):
        metadata = checked[start["key"]]
        if (start["epochs"] != metadata["fitted_tree_count"]
                or start["fit_rows"] != metadata["fit_rows"] or end["metadata"] != metadata):
            raise ValueError("Started/completed/saved model identity differs")
    return tree_count


def verify(root, out, spec):
    manifest, result = (json.loads((out/name).read_text()) for name in ("manifest.json", "result.json"))
    if (manifest["sources"] != inventory(root) or manifest["spec"] != spec
            or manifest["identity"] != spec["identity"]
            or manifest["action"] not in ("engineering", "development")
            or result["stage"] != manifest["action"]
            or manifest["input_sha256"] != {p:sha(root/p) for p in manifest["input_sha256"]}):
        raise ValueError("Residual partition frozen identity changed")
    for name, digest in result["output_sha256"].items():
        if sha(out/name) != digest:
            raise ValueError("Residual partition output identity changed")
    with np.load(out/"data.npz", allow_pickle=False) as data:
        x, y = data["x"], data["y"]
        ids = data["ids"] if manifest["action"]=="development" else None
    old, q = original(root, spec) if manifest["action"]=="development" else (None, None)
    if old is not None:
        frame = load_v2(root/"复赛_train", "train", 2754)
        np.testing.assert_array_equal(x, np.column_stack((frame[list(FEATURES)].to_numpy(float), frame.spout_no==1, frame.spout_no==2)))
        np.testing.assert_array_equal(y, frame.tap_iron.to_numpy(float))
        np.testing.assert_array_equal(ids, q["ids"])
        expected_keys = [f"s{seed}-f{fold}-D3_L1_PARTITION" for seed in spec["split_seeds"] for fold in range(5)]
    else:
        sx, sy = synthetic_arrays()
        np.testing.assert_array_equal(x, sx)
        np.testing.assert_array_equal(y, sy)
        expected_keys = ["SHORT", "LONG"]
    if [unit["key"] for unit in result["units"]] != expected_keys:
        raise ValueError("Complete registered unit pool required")
    if old is not None:
        validate_development_units(result["units"], spec)
    sys.addaudithook(deny_training_reads)
    predictions, selectors, checked = {}, {}, {}
    for unit in result["units"]:
        key = unit["key"]
        if old is None:
            fitting, calibration, outer, query = np.arange(108), np.arange(108,144), np.arange(144), np.arange(144,180)
        else:
            folds = folds_for(frame, unit["seed"])
            np.testing.assert_array_equal(folds, q[f"fold-{unit['seed']}"])
            with np.load(out/f"folds-{unit['seed']}.npz", allow_pickle=False) as saved_folds:
                np.testing.assert_array_equal(saved_folds["folds"], folds)
            outer, query = np.flatnonzero(folds!=unit["fold"]), np.flatnonzero(folds==unit["fold"])
            inner = folds_for(frame.iloc[outer], 27001)
            fitting, calibration = outer[inner!=0], outer[inner==0]
            control_key = f"s{unit['seed']}-f{unit['fold']}-D3_LONG"
            if unit["control_key"] != control_key:
                raise ValueError("Matched control unit differs")
            matched_parts(root/OLD, control_key, outer, query, fitting, calibration, inner)
            matched_parts(out, key, outer, query, fitting, calibration, inner)
        selector, _ = checked_model(out, key+"-selector", x[fitting], y[fitting], x[calibration], y[calibration])
        refit, independent = checked_model(out, key+"-refit", x[outer], y[outer], x[query])
        checked[key+"-selector"], checked[key+"-refit"] = selector.metadata(), refit.metadata()
        horizon = 12000 if old is not None else (300 if key=="SHORT" else 600)
        if (selector.selected_epoch_ != unit["selected_epoch"]
                or refit.selected_epoch_ != selector.selected_epoch_
                or len(refit.trees_) != selector.selected_epoch_ or len(selector.trees_) != horizon
                or (old is not None and unit["selector_cap_hit"] != (selector.selected_epoch_==12000))):
            raise ValueError("Frozen checkpoint/depth/horizon mismatch")
        with np.load(out/f"{key}-query.npz", allow_pickle=False) as file:
            saved = file["prediction"]
        check_prediction(refit, x[query], independent, saved)
        if old is None:
            selectors[key] = selector
        else:
            predictions.setdefault(unit["seed"], np.full(len(y), np.nan))[query] = saved
        del refit
    if old is None:
        previous.check_prefix(selectors["LONG"], selectors["SHORT"], 300)
    else:
        for seed, prediction in predictions.items():
            if not np.isfinite(prediction).all():
                raise ValueError("Incomplete independent OOF")
            with np.load(out/f"oof-{seed}.npz", allow_pickle=False) as file:
                np.testing.assert_array_equal(prediction, file["prediction"])
        metrics = describe(q, predictions, old)
        if (metrics != result["metrics"]
                or float(np.mean([m["gain_vs_Q75"] for m in metrics.values()])) != result["mean_gain_vs_Q75"]
                or float(np.mean([m["gain_vs_original_long"] for m in metrics.values()])) != result["mean_gain_vs_original_long"]
                or all(m["gain_vs_Q75"]>0 for m in metrics.values()) != result["eligible_for_separately_frozen_confirmation"]):
            raise ValueError("Independent complete partition-control-Q75 metrics differ")
    count, budget = (4, 1800) if old is None else (20, 240000)
    trees = verify_ledger(out, result, checked, count, budget)
    return dict(status=COLD_STATUS, checked_models=count, checked_tree_fits=trees,
                maximum_cold_difference=0., new_fits=0, model_fits_in_audit=0,
                residual_target_node_statistics_checked=True,
                node_statistic_absolute_tolerance=NODE_STATISTIC_ABSOLUTE_TOLERANCE,
                greedy_split_retraining_performed=False,
                prefix_models_checked=1 if old is None else 0,
                original_control_models_reused=0 if old is None else 20,
                manifest_sha256=sha(out/"manifest.json"), result_sha256=sha(out/"result.json"), packages=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("engineering", "development", "verify"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--engineering")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    spec = json.loads((root/SPEC).read_text())
    validate_spec(spec)
    out = local_output(root, args.output)
    if args.action=="verify":
        receipt = verify(root, out, spec)
        write_new(out/"cold-readback.json", receipt)
        print(json.dumps(receipt), flush=True)
        return
    if any(os.environ.get(k)!="1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Single numerical threads required before import")
    sources = inventory(root)
    if args.action=="development":
        engineering_run = local_output(root, args.engineering)
        em = json.loads((engineering_run/"manifest.json").read_text())
        er = json.loads((engineering_run/"result.json").read_text())
        ec = json.loads((engineering_run/"cold-readback.json").read_text())
        if (em["sources"] != sources or em["spec"] != spec or em["action"] != "engineering"
                or ec["checked_models"] != 4 or ec["status"] != COLD_STATUS
                or ec["maximum_cold_difference"] != 0 or ec["new_fits"] != 0
                or ec["prefix_models_checked"] != 1
                or ec["manifest_sha256"] != sha(engineering_run/"manifest.json")
                or ec["result_sha256"] != sha(engineering_run/"result.json")):
            raise ValueError("Residual partition G0 admission failed")
        for name, digest in er["output_sha256"].items():
            if sha(engineering_run/name) != digest:
                raise ValueError("Original G0 artifact changed")
        original(root, spec)
    out.mkdir(parents=True, exist_ok=False)
    input_names = ("复赛_train/train_samples.csv", "复赛_train/train_features.csv",
                   f"{REFERENCE_MATERIAL}/MANIFEST.json",
                   f"{REFERENCE_MATERIAL}/oof/original/q75-development-reference.npz")
    inputs = {p:sha(root/p) for p in input_names} if args.action=="development" else {}
    write_new(out/"manifest.json", dict(identity=spec["identity"], spec=spec, action=args.action,
        sources=sources, input_sha256=inputs,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        started_ns=time.time_ns(), pid=os.getpid(), python=sys.version, numpy=np.__version__))
    try:
        result = engineering(out, spec) if args.action=="engineering" else development(root, out, spec)
        if inventory(root) != sources or inputs != {p:sha(root/p) for p in inputs}:
            raise ValueError("Residual partition identity changed during fit")
        count, budget = (4, 1800) if args.action=="engineering" else (20, 240000)
        result.update(model_fits=len(result["models"]),
            tree_fits=sum(m["metadata"]["fitted_tree_count"] for m in result["models"]),
            observed_rss_mib=resource(), formal_promoted=False, confirmation_seeds_consumed=0,
            full_fits=0, packages=0, agent_uploads=0, completed_ns=time.time_ns(), platform_score=None)
        if result["model_fits"] != count or result["tree_fits"] > budget:
            raise ValueError("Frozen residual partition fit count exceeded")
        result["output_sha256"] = {p.name:sha(p) for p in out.iterdir() if p.is_file()}
        write_new(out/"result.json", result)
        print(json.dumps({k:v for k,v in result.items() if k not in ("models", "units", "output_sha256")}), flush=True)
    except Exception as exc:
        write_new(out/"FAILED.json", dict(type=type(exc).__name__, error=str(exc), automatic_retry=False))
        raise


if __name__=="__main__":
    main()
