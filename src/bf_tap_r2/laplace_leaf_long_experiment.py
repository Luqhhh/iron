"""Frozen single-arm LAD horizon extension with exact original-prefix controls."""
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
from .laplace_leaf_experiment import (inventory as previous_inventory, reference,
    checked_model, check_prediction, resource, log, scalar_gain)
from .laplace_leaf_median import LeafMedianRegressor
from .laplace_leaf_long import LeafMedianLongRegressor
from .metrics import wmape
from .submission import deny_training_reads
from .v2_release import load_v2

SPEC = "configs/laplace_leaf_long/SPEC.json"
OLD = "local/runs/laplace-leaf-median-20261002/development-r1"
EXTRA = (SPEC, "docs/laplace_leaf_long/PREREGISTRATION.md",
         "src/bf_tap_r2/laplace_leaf_long.py", __file__,
         "scripts/run_laplace_leaf_long_sequence.py")


def inventory(root):
    return {**previous_inventory(root), **{str(Path(p).relative_to(root) if Path(p).is_absolute() else p): sha(root/p) for p in EXTRA}}


def validate_spec(spec):
    semantic = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if semantic != "951f23c4e4f241f68b4244f3c6de59cf05b7f46b149bce79ebed168fad182701":
        raise ValueError("Frozen long-prefix scope differs")


def original(root, spec):
    run = root/OLD
    manifest = json.loads((run/"manifest.json").read_text())
    result = json.loads((run/"result.json").read_text())
    cold = json.loads((run/"cold-readback.json").read_text())
    if (sha(run/"manifest.json") != spec["original_manifest_sha256"]
            or sha(run/"result.json") != spec["original_result_sha256"]
            or sha(run/"cold-readback.json") != spec["original_cold_sha256"]
            or manifest["sources"] != previous_inventory(root) or cold["checked_models"] != 40
            or cold["manifest_sha256"] != sha(run/"manifest.json")
            or cold["result_sha256"] != sha(run/"result.json") or cold["maximum_cold_difference"] != 0):
        raise ValueError("Original complete native cold-audited run required")
    for name, digest in result["output_sha256"].items():
        if sha(run/name) != digest:
            raise ValueError("Original control artifact changed")
    q, material = reference(root, manifest["spec"])
    if sha(material/"submission/EMA_TIME_Q75/Luqhhh_bf_tap_predict_round2.zip") != spec["reference_zip_sha256"]:
        raise ValueError("Long-stage current reference differs")
    return result, q


def check_prefix(long, short, epochs):
    if len(short.trees_) != epochs or len(long.trees_) < epochs or long.history_[:epochs] != short.history_:
        raise ValueError("Original complete trajectory prefix differs")
    for attr in ("x_mean_", "x_scale_", "y_median_", "y_scale_", "fit_rows_",
                 "initial_train_mae_", "initial_calibration_mae_"):
        np.testing.assert_equal(getattr(long, attr), getattr(short, attr))
    for left, right, lv, rv in zip(long.trees_[:epochs], short.trees_, long.leaf_values_[:epochs], short.leaf_values_, strict=True):
        # Exact semantic tree state, not irrelevant bytes in structured-array padding.
        if left.get_params() != right.get_params():
            raise ValueError("Original tree parameters prefix differs")
        left_state, right_state = left.tree_.__getstate__(), right.tree_.__getstate__()
        if left_state.keys() != right_state.keys():
            raise ValueError("Original tree state schema prefix differs")
        for name, value in left_state.items():
            np.testing.assert_equal(value, right_state[name])
        np.testing.assert_array_equal(lv, rv)


def fit_model(out, key, model, x, y, epochs, calibration=None, prefix=3000):
    log(out, dict(event="model_started", key=key, epochs=epochs, fit_rows=len(y)))
    kwargs = dict(epochs=epochs, validation=calibration, on_epoch=resource)
    if isinstance(model, LeafMedianLongRegressor):
        kwargs["prefix_epochs"] = prefix
    model.fit(x, y, **kwargs)
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
    short, meta = fit_model(out, "SHORT-selector", LeafMedianRegressor(3), x[:108], y[:108], 300, (x[108:144], y[108:144]))
    models.append(meta)
    long, meta = fit_model(out, "LONG-selector", LeafMedianLongRegressor(), x[:108], y[:108], 600, (x[108:144], y[108:144]), prefix=300)
    models.append(meta)
    check_prefix(long, short, 300)
    units = []
    for name, selector, model in (("SHORT", short, LeafMedianRegressor(3)), ("LONG", long, LeafMedianLongRegressor())):
        refit, meta = fit_model(out, name+"-refit", model, x[:144], y[:144], selector.selected_epoch_, prefix=300)
        models.append(meta)
        save_arrays(out/f"{name}-query.npz", prediction=refit.predict(x[144:]))
        units.append(dict(key=name, selected_epoch=selector.selected_epoch_))
    return dict(stage="engineering", models=models, units=units, prefix_models_checked=1)


def describe(q, predictions, old):
    y, spout, result = q["targets"][:, 0], q["spout"], {}
    for seed, prediction in predictions.items():
        parent, folds = q[f"current-{seed}"][:, 0], q[f"fold-{seed}"]
        gain = scalar_gain(y, parent, prediction)
        result[str(seed)] = dict(iron_wmape=wmape(y, prediction), gain_vs_Q75=gain,
            gain_long_minus_short=gain-old["metrics"][str(seed)]["arms"]["MEDIAN_D3"]["gain"],
            folds={str(f):scalar_gain(y[folds==f], parent[folds==f], prediction[folds==f]) for f in range(5)},
            spouts={str(s):scalar_gain(y[spout==s], parent[spout==s], prediction[spout==s]) for s in (1,2)})
    return result


def development(root, out, spec):
    old, q = original(root, spec)
    frame = load_v2(root/"复赛_train", "train", 2754)
    x = np.column_stack((frame[list(FEATURES)].to_numpy(float), frame.spout_no==1, frame.spout_no==2))
    y, ids = frame.tap_iron.to_numpy(float), frame.sample_id.to_numpy(str)
    np.testing.assert_array_equal(ids, q["ids"])
    np.testing.assert_array_equal(x[:, :21], q["numeric"])
    np.testing.assert_array_equal(frame[["tap_iron", "tap_time_len"]].to_numpy(float), q["targets"])
    save_arrays(out/"data.npz", x=x, y=y, ids=ids)
    models, units, predictions = [], [], {}
    for seed in spec["split_seeds"]:
        folds = folds_for(frame, seed)
        np.testing.assert_array_equal(folds, q[f"fold-{seed}"])
        predictions[seed] = np.full(len(y), np.nan)
        save_arrays(out/f"folds-{seed}.npz", folds=folds)
        for fold in range(5):
            key = f"s{seed}-f{fold}-D3_LONG"
            control_key = f"s{seed}-f{fold}-MEDIAN_D3"
            outer, query = np.flatnonzero(folds!=fold), np.flatnonzero(folds==fold)
            inner = folds_for(frame.iloc[outer], 27001)
            fitting, calibration = outer[inner!=0], outer[inner==0]
            with np.load(root/OLD/f"{control_key}-query.npz", allow_pickle=False) as parts:
                for name, value in dict(outer=outer, query=query, fit=fitting, calibration=calibration, inner_folds=inner).items():
                    np.testing.assert_array_equal(parts[name], value)
            selector, meta = fit_model(out, key+"-selector", LeafMedianLongRegressor(), x[fitting], y[fitting], 12000, (x[calibration], y[calibration]))
            models.append(meta)
            with (root/OLD/f"{control_key}-selector.pkl").open("rb") as handle:
                control = pickle.load(handle)
            check_prefix(selector, control, 3000)
            epoch = selector.selected_epoch_
            refit, meta = fit_model(out, key+"-refit", LeafMedianLongRegressor(), x[outer], y[outer], epoch)
            models.append(meta)
            prediction = refit.predict(x[query])
            predictions[seed][query] = prediction
            save_arrays(out/f"{key}-query.npz", prediction=prediction, outer=outer, query=query,
                        fit=fitting, calibration=calibration, inner_folds=inner)
            units.append(dict(key=key, control_key=control_key, seed=seed, fold=fold,
                              selected_epoch=epoch, selector_cap_hit=epoch==12000))
            del selector, refit, control
    for seed, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError("Incomplete long OOF")
        save_arrays(out/f"oof-{seed}.npz", prediction=prediction)
    metrics = describe(q, predictions, old)
    return dict(stage="development", models=models, units=units, metrics=metrics,
        mean_gain_vs_Q75=float(np.mean([m["gain_vs_Q75"] for m in metrics.values()])),
        mean_gain_long_minus_short=float(np.mean([m["gain_long_minus_short"] for m in metrics.values()])),
        eligible_for_separately_frozen_confirmation=all(m["gain_vs_Q75"]>0 for m in metrics.values()),
        original_control_models_reused=20, prefix_models_checked=10)


def verify(root, out, spec):
    manifest, result = (json.loads((out/name).read_text()) for name in ("manifest.json", "result.json"))
    if manifest["sources"] != inventory(root) or manifest["input_sha256"] != {p:sha(root/p) for p in manifest["input_sha256"]}:
        raise ValueError("Long-stage frozen identity changed")
    for name, digest in result["output_sha256"].items():
        if sha(out/name) != digest:
            raise ValueError("Long output identity changed")
    with np.load(out/"data.npz", allow_pickle=False) as data:
        x, y = data["x"], data["y"]
        ids = data["ids"] if manifest["action"]=="development" else None
    old, q = original(root, spec) if manifest["action"]=="development" else (None, None)
    if old is not None:
        frame = load_v2(root/"复赛_train", "train", 2754)
        np.testing.assert_array_equal(x, np.column_stack((frame[list(FEATURES)].to_numpy(float), frame.spout_no==1, frame.spout_no==2)))
        np.testing.assert_array_equal(y, frame.tap_iron.to_numpy(float))
        np.testing.assert_array_equal(ids, q["ids"])
    else:
        sx, sy = synthetic_arrays()
        np.testing.assert_array_equal(x, sx)
        np.testing.assert_array_equal(y, sy)
    sys.addaudithook(deny_training_reads)
    predictions, selectors = {}, {}
    for unit in result["units"]:
        key = unit["key"]
        if old is None:
            fitting, calibration, outer, query = np.arange(108), np.arange(108,144), np.arange(144), np.arange(144,180)
        else:
            folds = folds_for(frame, unit["seed"])
            np.testing.assert_array_equal(folds, q[f"fold-{unit['seed']}"])
            outer, query = np.flatnonzero(folds!=unit["fold"]), np.flatnonzero(folds==unit["fold"])
            inner = folds_for(frame.iloc[outer], 27001)
            fitting, calibration = outer[inner!=0], outer[inner==0]
            with np.load(out/f"{key}-query.npz", allow_pickle=False) as parts:
                for name, value in dict(outer=outer, query=query, fit=fitting, calibration=calibration, inner_folds=inner).items():
                    np.testing.assert_array_equal(parts[name], value)
        selector, _ = checked_model(out, key+"-selector", x[fitting], y[fitting], x[calibration], y[calibration])
        refit, independent = checked_model(out, key+"-refit", x[outer], y[outer], x[query])
        if (selector.selected_epoch_ != unit["selected_epoch"] or refit.selected_epoch_ != selector.selected_epoch_
                or selector.depth != 3 or refit.depth != 3 or len(selector.trees_) != (12000 if old is not None else (300 if key=="SHORT" else 600))):
            raise ValueError("Frozen checkpoint/depth/horizon mismatch")
        with np.load(out/f"{key}-query.npz", allow_pickle=False) as file:
            saved = file["prediction"]
        check_prediction(refit, x[query], independent, saved)
        if old is None:
            selectors[key] = selector
        else:
            with (root/OLD/f"{unit['control_key']}-selector.pkl").open("rb") as handle:
                control = pickle.load(handle)
            check_prefix(selector, control, 3000)
            predictions.setdefault(unit["seed"], np.full(len(y), np.nan))[query] = saved
    if old is None:
        check_prefix(selectors["LONG"], selectors["SHORT"], 300)
    else:
        for seed, prediction in predictions.items():
            with np.load(out/f"oof-{seed}.npz", allow_pickle=False) as file:
                np.testing.assert_array_equal(prediction, file["prediction"])
        if describe(q, predictions, old) != result["metrics"]:
            raise ValueError("Independent complete long-short-Q75 metrics differ")
    ledger = [json.loads(line) for line in (out/"ledger.jsonl").read_text().splitlines()]
    starts = [a["key"] for a in ledger if a["event"]=="model_started"]
    ends = [a["key"] for a in ledger if a["event"]=="model_completed"]
    count = 2*len(result["units"])
    if starts != ends or len(starts)!=count or len(set(starts))!=count or count!=result["model_fits"]:
        raise ValueError("Closed actual long-model ledger required")
    return dict(status="passed_exact_original_prefix_complete_leaf_updates_trajectories_cold_OOF_and_ledger",
                checked_models=count, maximum_cold_difference=0., new_fits=0,
                prefix_models_checked=1 if old is None else 10, manifest_sha256=sha(out/"manifest.json"),
                result_sha256=sha(out/"result.json"), packages=0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action", choices=("engineering", "development", "verify"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--engineering")
    args=parser.parse_args()
    root=Path.cwd().resolve()
    spec=json.loads((root/SPEC).read_text())
    validate_spec(spec)
    out=local_output(root,args.output)
    if args.action=="verify":
        receipt=verify(root,out,spec)
        write_new(out/"cold-readback.json",receipt)
        print(json.dumps(receipt),flush=True)
        return
    if any(os.environ.get(k)!="1" for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")):
        raise ValueError("Single numerical threads required before import")
    sources=inventory(root)
    if args.action=="development":
        engineering_run=local_output(root,args.engineering)
        em=json.loads((engineering_run/"manifest.json").read_text())
        ec=json.loads((engineering_run/"cold-readback.json").read_text())
        if (em["sources"]!=sources or ec["checked_models"]!=4
                or ec["status"]!="passed_exact_original_prefix_complete_leaf_updates_trajectories_cold_OOF_and_ledger"
                or ec["maximum_cold_difference"]!=0 or ec["new_fits"]!=0
                or ec["manifest_sha256"]!=sha(engineering_run/"manifest.json")
                or ec["result_sha256"]!=sha(engineering_run/"result.json")):
            raise ValueError("Long-stage G0 admission failed")
        original(root,spec)
    out.mkdir(parents=True,exist_ok=False)
    inputs={p:sha(root/p) for p in ("复赛_train/train_samples.csv","复赛_train/train_features.csv")} if args.action=="development" else {}
    write_new(out/"manifest.json",dict(identity=spec["identity"],spec=spec,action=args.action,sources=sources,
        input_sha256=inputs,git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        started_ns=time.time_ns(),pid=os.getpid(),python=sys.version,numpy=np.__version__))
    try:
        result=engineering(out,spec) if args.action=="engineering" else development(root,out,spec)
        if inventory(root)!=sources or inputs!={p:sha(root/p) for p in inputs}:
            raise ValueError("Long-stage identity changed during fit")
        count,budget=(4,1800) if args.action=="engineering" else (20,240000)
        result.update(model_fits=len(result["models"]),tree_fits=sum(m["metadata"]["fitted_tree_count"] for m in result["models"]),
                      observed_rss_mib=resource(),formal_promoted=False,confirmation_seeds_consumed=0,
                      full_fits=0,packages=0,agent_uploads=0,completed_ns=time.time_ns(),platform_score=None)
        if result["model_fits"]!=count or result["tree_fits"]>budget:
            raise ValueError("Frozen long fit count exceeded")
        result["output_sha256"]={p.name:sha(p) for p in out.iterdir() if p.is_file()}
        write_new(out/"result.json",result)
        print(json.dumps({k:v for k,v in result.items() if k not in ("models","units","output_sha256")}),flush=True)
    except Exception as exc:
        write_new(out/"FAILED.json",dict(type=type(exc).__name__,error=str(exc),automatic_retry=False))
        raise


if __name__=="__main__":
    main()
