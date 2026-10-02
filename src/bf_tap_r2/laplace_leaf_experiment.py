"""Frozen LAD leaf-median G0, full development, and no-fit cold reconstruction."""
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
import psutil

from .data import FEATURES
from .joint_support_audit import local_output, sha, write_new
from .laplace_epoch_experiment import inventory as old_inventory, synthetic_arrays, folds_for, save_arrays
from .laplace_leaf_median import LeafMedianRegressor, best_epoch
from .metrics import wmape
from .submission import deny_training_reads
from .v2_release import load_v2

SPEC = "configs/laplace_leaf_median/SPEC.json"
EXTRA = (SPEC, "docs/laplace_leaf_median/PREREGISTRATION.md",
         "src/bf_tap_r2/laplace_leaf_median.py", "src/bf_tap_r2/laplace_leaf_experiment.py",
         "scripts/run_laplace_leaf_sequence.py")
INTAKE = "local/runs/laplace-q75-review-20261002/intake-r1"
REVIEW = "local/runs/laplace-q75-review-20261002/review-r1"
OLD = "local/runs/laplace-epoch-20261002/public-v2-r1"


def validate_spec(spec):
    semantic = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if semantic != "40e19898afc8523fbf8ef7c6b62e6291d097d17722afe46222bcf7f7e6700155":
        raise ValueError("Frozen leaf-median specification differs")


def inventory(root):
    return {**old_inventory(root), **{p:sha(root/p) for p in EXTRA}}


def resource():
    rss = psutil.Process().memory_info().rss/1024**2
    if rss > 1536:
        raise MemoryError("Frozen worker RSS gate")
    return rss


def log(out, event):
    with (out/"ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(utc_ns=time.time_ns(), **event), allow_nan=False)+"\n")


def fit(out, key, depth, x, y, epochs, validation=None):
    log(out, dict(event="model_started", key=key, depth=depth, epochs=epochs, fit_rows=len(y)))
    model = LeafMedianRegressor(depth).fit(x, y, epochs=epochs, validation=validation, on_epoch=resource)
    with (out/f"{key}.pkl").open("xb") as f:
        pickle.dump(model, f, protocol=5)
    item = dict(key=key, metadata=model.metadata())
    write_new(out/f"{key}-trace.json", dict(metadata=model.metadata(), history=model.history_))
    log(out, dict(event="model_completed", **item))
    return model, item


def reference(root, spec):
    intake, review = root/INTAKE, root/REVIEW
    if sha(intake/"result.json") != spec["intake_result_sha256"] or sha(review/"result.json") != spec["matched_review_result_sha256"]:
        raise ValueError("Matched material receipt changed")
    ir = json.loads((intake/"result.json").read_text())
    material = Path(ir["material"])
    m = json.loads((material/"MANIFEST.json").read_text())
    for name, item in m["files"].items():
        if sha(material/name) != item["sha256"]:
            raise ValueError("Bound material changed")
    if sha(material/"submission/EMA_TIME_Q75/Luqhhh_bf_tap_predict_round2.zip") != spec["reference_zip_sha256"]:
        raise ValueError("Wrong reference package")
    for item in ir["data"]:
        if sha(root/item["path"]) != item["raw_sha256"]:
            raise ValueError("Original data changed")
    with np.load(material/"oof/original/q75-development-reference.npz", allow_pickle=False) as f:
        arrays = {k:f[k].copy() for k in f.files}
    return arrays, material


def engineering(out, spec):
    x, y = synthetic_arrays()
    save_arrays(out/"data.npz", x=x, y=y)
    models, units = [], []
    for arm, depth in spec["recipes"].items():
        selector, meta = fit(out, arm+"-selector", depth, x[:108], y[:108], 300, (x[108:144], y[108:144]))
        models.append(meta)
        refit, meta = fit(out, arm+"-refit", depth, x[:144], y[:144], selector.selected_epoch_)
        models.append(meta)
        save_arrays(out/f"{arm}-query.npz", prediction=refit.predict(x[144:]))
        unit = dict(key=arm, arm=arm, depth=depth, selected_epoch=selector.selected_epoch_)
        write_new(out/f"{arm}-unit.json", unit)
        units.append(unit)
    return dict(stage="engineering", models=models, units=units, official_sample_reads=0)


def scalar_gain(y, parent, pred):
    blend = .8*parent+.2*pred
    if not np.isfinite(blend).all() or (blend < 0).any():
        raise ValueError("Invalid blend; never clipped")
    gain = 50*(wmape(y, parent)-wmape(y, blend))
    independent = 50*(sum(abs(float(a)-float(b)) for a, b in zip(y, parent))-
                      sum(abs(float(a)-float(b)) for a, b in zip(y, blend)))/sum(abs(float(a)) for a in y)
    if abs(gain-independent) > 1e-12:
        raise ValueError("Independent scalar gain differs")
    return gain


def metrics(y, q, spout, vectors, spec):
    result = {}
    for seed in spec["split_seeds"]:
        parent, folds = q[f"current-{seed}"], q[f"fold-{seed}"]
        result[str(seed)] = dict(reference_local_score=100-50*(wmape(y, parent[:,0])+wmape(q["targets"][:,1], parent[:,1])), arms={})
        for arm in spec["recipes"]:
            pred = vectors[seed, arm]
            result[str(seed)]["arms"][arm] = dict(iron_wmape=wmape(y, pred), rows=len(y),
                gain=scalar_gain(y, parent[:,0], pred),
                folds={str(f):scalar_gain(y[folds==f], parent[folds==f,0], pred[folds==f]) for f in range(5)},
                spouts={str(s):scalar_gain(y[spout==s], parent[spout==s,0], pred[spout==s]) for s in (1,2)})
    return result


def development(root, out, spec):
    q, material = reference(root, spec)
    frame = load_v2(root/"复赛_train", "train", 2754)
    x = np.column_stack((frame[list(FEATURES)].to_numpy(float), frame.spout_no==1, frame.spout_no==2))
    y, ids = frame.tap_iron.to_numpy(float), frame.sample_id.to_numpy(str)
    np.testing.assert_array_equal(ids, q["ids"])
    np.testing.assert_array_equal(x[:,:21], q["numeric"])
    np.testing.assert_array_equal(frame[["tap_iron","tap_time_len"]].to_numpy(float), q["targets"])
    save_arrays(out/"data.npz", x=x, y=y, ids=ids)
    models, units, vectors = [], [], {}
    roles = json.loads((material/"oof/partitions.json").read_text())["units"]
    for seed in spec["split_seeds"]:
        folds = folds_for(frame, seed)
        np.testing.assert_array_equal(folds, q[f"fold-{seed}"])
        save_arrays(out/f"folds-{seed}.npz", folds=folds)
        for arm in spec["recipes"]:
            vectors[seed, arm] = np.full(len(y), np.nan)
        for fold in range(5):
            outer, query = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            r = next(r for r in roles if r["split_seed"]==seed and r["outer_fold"]==fold)
            if r["outer_training_ids"] != ids[outer].tolist() or r["outer_query_ids"] != ids[query].tolist():
                raise ValueError("Matched outer partition differs")
            inner = folds_for(frame.iloc[outer], 27001)
            fitting, calibration = outer[inner!=0], outer[inner==0]
            for arm, depth in spec["recipes"].items():
                key = f"s{seed}-f{fold}-{arm}"
                selector, meta = fit(out, key+"-selector", depth, x[fitting], y[fitting], 3000, (x[calibration], y[calibration]))
                models.append(meta)
                epoch = selector.selected_epoch_
                refit, meta = fit(out, key+"-refit", depth, x[outer], y[outer], epoch)
                models.append(meta)
                pred = refit.predict(x[query])
                vectors[seed,arm][query] = pred
                save_arrays(out/f"{key}-query.npz", prediction=pred, outer=outer, query=query,
                            fit=fitting, calibration=calibration, inner_folds=inner)
                unit = dict(key=key, arm=arm, depth=depth, seed=seed, fold=fold, selected_epoch=epoch,
                            selector_cap_hit=epoch==3000, exactly_zero_update_epochs=selector.metadata()["exactly_zero_update_epochs"])
                write_new(out/f"{key}-unit.json", unit)
                units.append(unit)
                del selector, refit
    for (seed, arm), pred in vectors.items():
        if not np.isfinite(pred).all():
            raise ValueError("Incomplete OOF")
        save_arrays(out/f"oof-{seed}-{arm}.npz", prediction=pred)
    scores = metrics(y, q, frame.spout_no.to_numpy(int), vectors, spec)
    mean = {arm:float(np.mean([scores[str(s)]["arms"][arm]["gain"] for s in spec["split_seeds"]])) for arm in spec["recipes"]}
    eligible = [arm for arm in spec["candidate_tie_order"] if all(scores[str(s)]["arms"][arm]["gain"]>0 for s in spec["split_seeds"])]
    return dict(stage="development", models=models, units=units, metrics=scores, mean_gain=mean,
                finalist=max(eligible, key=lambda a:mean[a]) if eligible else None,
                eligible_for_separately_frozen_confirmation=eligible, official_sample_reads=1)


def checked_model(out, key, x, y, query, query_y=None):
    # External pickle is never accepted: caller verifies our private output hashes.
    with (out/f"{key}.pkl").open("rb") as f:
        model = pickle.load(f)
    trace = json.loads((out/f"{key}-trace.json").read_text())
    xs = np.where(x.std(axis=0)>0, x.std(axis=0), 1.)
    if model.metadata()!=trace["metadata"] or model.history_!=trace["history"]:
        raise ValueError("Model trace identity differs")
    np.testing.assert_array_equal(model.x_mean_, x.mean(axis=0))
    np.testing.assert_array_equal(model.x_scale_, xs)
    if model.y_median_!=float(np.median(y)) or model.y_scale_!=(float(y.std()) or 1.) or model.fit_rows_!=len(y):
        raise ValueError("Training-only normalization differs")
    z, v = (x-model.x_mean_)/model.x_scale_, (query-model.x_mean_)/model.x_scale_
    sy, mu, vm = (y-model.y_median_)/model.y_scale_, np.zeros(len(y)), np.zeros(len(query))
    previous = float(np.abs(sy).mean())
    if model.initial_train_mae_ != previous:
        raise ValueError("Initial training loss differs")
    initial_cal = float(np.abs(query_y-model.y_median_).mean()) if query_y is not None else None
    if model.initial_calibration_mae_ != initial_cal:
        raise ValueError("Initial calibration differs")
    for epoch,(tree, values, event) in enumerate(zip(model.trees_,model.leaf_values_,model.history_,strict=True),1):
        if tree.max_depth!=model.depth or tree.min_samples_leaf!=20 or tree.random_state!=42 or tree.criterion!="squared_error":
            raise ValueError("Frozen tree parameters differ")
        leaves, residual = tree.apply(z), sy-mu
        for leaf in np.unique(leaves):
            # Independent sorted-order statistic, not the fitting helper.
            ordered = np.sort(residual[leaves==leaf])
            median = float(ordered[(len(ordered)-1)//2]+ordered[len(ordered)//2])/2
            if values[leaf] != median:
                raise ValueError("Independent leaf median differs")
        update = .05*values[leaves]
        mu += update
        vm += .05*values[tree.apply(v)]
        loss = float(np.abs(sy-mu).mean())
        expected = dict(epoch=epoch,train_mae_normalized=loss,maximum_abs_update=float(np.max(np.abs(update))),nonzero_update_rows=int(np.count_nonzero(update)))
        if query_y is not None:
            expected["calibration_mae"] = float(np.abs(query_y-(vm*model.y_scale_+model.y_median_)).mean())
        if event!=expected or loss>previous+1e-12:
            raise ValueError("Independent complete trajectory differs")
        previous=loss
    if query_y is not None and model.selected_epoch_!=best_epoch(model.history_,initial_cal):
        raise ValueError("Independent checkpoint selection differs")
    return model, vm*model.y_scale_+model.y_median_


def check_prediction(model, query, independent, saved):
    p = model.predict(query)
    reverse = model.predict(query[::-1])[::-1]
    chunk = np.concatenate([model.predict(query[i:i+37]) for i in range(0,len(query),37)])
    for value in (independent,saved,reverse,chunk):
        np.testing.assert_array_equal(p,value)
    if len(model.predict(query[:0])):
        raise ValueError("Empty prediction mismatch")


def verify(root, out, spec):
    m = json.loads((out/"manifest.json").read_text())
    r = json.loads((out/"result.json").read_text())
    if m["sources"]!=inventory(root) or m["input_sha256"]!={name:sha(root/name) for name in m["input_sha256"]}:
        raise ValueError("Source/input identity changed")
    for name,h in r["output_sha256"].items():
        if sha(out/name)!=h:
            raise ValueError("Original output changed")
    with np.load(out/"data.npz",allow_pickle=False) as d:
        x,y = d["x"],d["y"]
        ids = d["ids"] if m["action"]=="development" else None
    q = None
    if m["action"]=="development":
        q,_ = reference(root,spec)
        frame=load_v2(root/"复赛_train","train",2754)
        expected_x=np.column_stack((frame[list(FEATURES)].to_numpy(float),frame.spout_no==1,frame.spout_no==2))
        np.testing.assert_array_equal(x,expected_x)
        np.testing.assert_array_equal(y,frame.tap_iron.to_numpy(float))
        np.testing.assert_array_equal(ids,q["ids"])
    else:
        sx,sy=synthetic_arrays()
        np.testing.assert_array_equal(x,sx)
        np.testing.assert_array_equal(y,sy)
    sys.addaudithook(deny_training_reads)
    vectors, count = {}, 0
    for unit in r["units"]:
        key=unit["key"]
        if m["action"]=="engineering":
            fit_idx,calibration,outer,query=np.arange(108),np.arange(108,144),np.arange(144),np.arange(144,180)
        else:
            seed,fold=unit["seed"],unit["fold"]
            folds=folds_for(frame,seed)
            np.testing.assert_array_equal(folds,q[f"fold-{seed}"])
            outer,query=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
            inner=folds_for(frame.iloc[outer],27001)
            fit_idx,calibration=outer[inner!=0],outer[inner==0]
            with np.load(out/f"{key}-query.npz",allow_pickle=False) as a:
                for name,value in dict(outer=outer,query=query,fit=fit_idx,calibration=calibration,inner_folds=inner).items():
                    np.testing.assert_array_equal(a[name],value)
        selector,_=checked_model(out,key+"-selector",x[fit_idx],y[fit_idx],x[calibration],y[calibration])
        refit, independent=checked_model(out,key+"-refit",x[outer],y[outer],x[query])
        if selector.selected_epoch_!=unit["selected_epoch"] or refit.selected_epoch_!=unit["selected_epoch"] or selector.depth!=unit["depth"] or refit.depth!=unit["depth"]:
            raise ValueError("Selector/refit identity differs")
        with np.load(out/f"{key}-query.npz",allow_pickle=False) as a:
            saved=a["prediction"].copy()
        check_prediction(refit,x[query],independent,saved)
        if m["action"]=="development":
            vectors.setdefault((unit["seed"],unit["arm"]),np.full(len(y),np.nan))[query]=saved
        count+=2
    if m["action"]=="development":
        for (seed,arm),prediction in vectors.items():
            with np.load(out/f"oof-{seed}-{arm}.npz",allow_pickle=False) as a:
                np.testing.assert_array_equal(prediction,a["prediction"])
        if metrics(y,q,frame.spout_no.to_numpy(int),vectors,spec)!=r["metrics"]:
            raise ValueError("Complete independent metrics differ")
    ledger=[json.loads(line) for line in (out/"ledger.jsonl").read_text().splitlines()]
    starts=[a["key"] for a in ledger if a["event"]=="model_started"]
    ends=[a["key"] for a in ledger if a["event"]=="model_completed"]
    if count!=r["model_fits"] or len(starts)!=count or starts!=ends or len(set(starts))!=count:
        raise ValueError("Actual fitting ledger differs")
    return dict(status="passed_source_data_partitions_every_leaf_median_complete_traces_cold_predictions_and_metrics",
                checked_models=count,maximum_cold_difference=0.,new_fits=0,model_fits_in_audit=0,
                manifest_sha256=sha(out/"manifest.json"),result_sha256=sha(out/"result.json"),packages=0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("engineering","development","verify"))
    p.add_argument("--output",required=True)
    p.add_argument("--engineering")
    args=p.parse_args()
    root=Path.cwd().resolve()
    out=local_output(root,args.output)
    spec=json.loads((root/SPEC).read_text())
    validate_spec(spec)
    if args.action=="verify":
        receipt=verify(root,out,spec)
        write_new(out/"cold-readback.json",receipt)
        print(json.dumps(receipt),flush=True)
        return
    if any(os.environ.get(k)!="1" for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")):
        raise ValueError("Single numerical threads required before import")
    sources=inventory(root)
    if args.action=="development":
        engineering_out=local_output(root,args.engineering)
        em=json.loads((engineering_out/"manifest.json").read_text())
        ec=json.loads((engineering_out/"cold-readback.json").read_text())
        if em["sources"]!=sources or ec["checked_models"]!=4 or ec["manifest_sha256"]!=sha(engineering_out/"manifest.json") or ec["result_sha256"]!=sha(engineering_out/"result.json"):
            raise ValueError("Original engineering admission failed")
        reference(root,spec)
    out.mkdir(parents=True,exist_ok=False)
    inputs={name:sha(root/name) for name in ("复赛_train/train_samples.csv","复赛_train/train_features.csv")} if args.action=="development" else {}
    write_new(out/"manifest.json",dict(identity=spec["identity"],spec=spec,action=args.action,sources=sources,
              input_sha256=inputs,git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              started_ns=time.time_ns(),pid=os.getpid(),python=sys.version,numpy=np.__version__))
    try:
        result=engineering(out,spec) if args.action=="engineering" else development(root,out,spec)
        if inventory(root)!=sources or inputs!={name:sha(root/name) for name in inputs}:
            raise ValueError("Frozen identity changed during fitting")
        result.update(model_fits=len(result["models"]),tree_fits=sum(a["metadata"]["fitted_tree_count"] for a in result["models"]),
                      observed_rss_mib=resource(),formal_promoted=False,confirmation_seeds_consumed=0,full_fits=0,packages=0,agent_uploads=0,
                      completed_ns=time.time_ns(),platform_score=None)
        expected,budget=(4,1200) if args.action=="engineering" else (40,120000)
        if result["model_fits"]!=expected or result["tree_fits"]>budget:
            raise ValueError("Frozen fit counts differ")
        result["output_sha256"]={p.name:sha(p) for p in out.iterdir() if p.is_file()}
        write_new(out/"result.json",result)
        print(json.dumps({k:v for k,v in result.items() if k not in ("models","units","output_sha256")}),flush=True)
    except Exception as exc:
        write_new(out/"FAILED.json",dict(type=type(exc).__name__,error=str(exc),automatic_retry=False))
        raise


if __name__=="__main__":
    main()
