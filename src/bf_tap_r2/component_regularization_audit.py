"""Fresh-process saved-state and fixed-replacement audit; no new fits."""
from pathlib import Path
import argparse
import json
import math

import numpy as np
import torch
import yaml

from .component_regularization import ComponentRegressor
from .component_regularization_run import SPEC, RUN_ROOT, RECIPE, outputs, model_class
from .data import FEATURES, TARGETS
from .v3_4_bags import group_safe_inner_folds
from .v3_6_networks import NumericPreprocessor
from .v5_library import load_v5_training_frame, fold_vector
from .v5_spec import load_v5_spec
from .v7_periodic import digest, file_hash, write_new
from .v49_run import verify_hashes, verified_unit, unit_id, verify_reference_cache, read_reference


def verify_saved(path, training, y, arm, settings, mechanisms, validation=None, expected_epoch=None, model_type=ComponentRegressor):
    model=model_type.load(path);saved=model.saved;trace=saved["trace"]
    if saved["arm"]!=arm or saved["settings"]!=settings or saved["mechanisms"]!=mechanisms or saved["recipe"]!=RECIPE:
        raise ValueError("Saved recipe identity mismatch")
    if trace["fit_ids_digest"]!=digest(training.sample_id.tolist()) or trace["fit_rows"]!=len(training):
        raise ValueError("Saved fit row identity mismatch")
    x=training[list(FEATURES)].to_numpy(float)
    np.testing.assert_array_equal(saved["preprocessing"]["means"],x.mean(axis=0))
    np.testing.assert_array_equal(saved["preprocessing"]["stds"],x.std(axis=0))
    if saved["preprocessing"]!=NumericPreprocessor(structure="raw_tabm").fit(training).metadata():
        raise ValueError("Train-only preprocessing mismatch")
    np.testing.assert_array_equal(saved["mean"],np.asarray(y,float).mean(axis=0))
    np.testing.assert_array_equal(saved["std"],np.asarray(y,float).std(axis=0))
    if any(v.is_floating_point() and (v.dtype!=torch.float32 or not torch.isfinite(v).all()) for v in saved["state"].values()):
        raise ValueError("Invalid saved dtype/value")
    history=trace["history"];batches=math.ceil(len(training)/settings["batch_size"])
    if [r["epoch"] for r in history]!=list(range(1,trace["stopped_epoch"]+1)):
        raise ValueError("Incomplete epoch trace")
    if not 1<=trace["selected_epoch"]<=trace["stopped_epoch"]<=settings["max_epochs"]:
        raise ValueError("Invalid epoch range")
    for row in history:
        if row["updates"]!=batches or row["gradient_evaluations"]!=batches*(2 if arm=="SAM" else 1):
            raise ValueError("Update/gradient count mismatch")
        if any(not np.isfinite(v) or v<0 for v in row.values()):raise ValueError("Invalid training trace")
    if trace["updates"]!=batches*len(history) or trace["gradient_evaluations"]!=sum(r["gradient_evaluations"] for r in history):
        raise ValueError("Total update mismatch")
    if validation is not None:
        best,selected,stale=float("inf"),0,0
        for row in history:
            if stale>=settings["patience"]:raise ValueError("Training beyond patience")
            if row["validation_mae"]<best-settings["min_delta"]:
                best,selected,stale=row["validation_mae"],row["epoch"],0
            else:stale+=1
        if selected!=trace["selected_epoch"]:raise ValueError("Incorrect selected epoch")
        if trace["stopped_epoch"]<settings["max_epochs"] and stale!=settings["patience"]:
            raise ValueError("Premature stopping")
        query=validation.drop(columns=[t for t in TARGETS if t in validation])
        # Raw-space recomputation differs slightly from the original float32
        # standardized arithmetic; use the actual standardized tensor instead.
        vx,vc=model._inputs(query)
        vy=torch.as_tensor((validation[outputs("tap_iron" if len(saved["mean"])==2 else "tap_time_len")].to_numpy()-model.mean_)/model.std_,dtype=torch.float32)
        with torch.no_grad():actual=float((model.model_(vx,vc).mean(1)-vy).abs().mean())
        if actual!=best:raise ValueError("Selected checkpoint does not reproduce validation metric")
    else:
        if trace["selected_epoch"]!=trace["stopped_epoch"] or (expected_epoch is not None and trace["selected_epoch"]!=expected_epoch):
            raise ValueError("Refit epoch differs from selection")
    if hasattr(model,"audit_fit"):model.audit_fit(training)
    return model


def run(root,out,spec_path=SPEC):
    root=Path(root).resolve();out=(root/out).resolve()
    spec=yaml.safe_load((root/spec_path).read_text())
    if not out.is_relative_to((root/spec.get("run_root",RUN_ROOT)).resolve()):raise ValueError("Private audit required")
    manifest=json.loads((out/"manifest.json").read_text())
    expected=dict(manifest);expected.pop("identity")
    if digest(expected)!=manifest["identity"] or manifest["spec_sha256"]!=file_hash(root/spec_path):
        raise ValueError("Manifest identity mismatch")
    verify_hashes(root,manifest["source_hashes"]);verify_hashes(root,manifest["data_hashes"])
    if not manifest["development"]:verify_reference_cache(root,spec)
    frame=load_v5_training_frame(root);summary=json.loads((out/"summary.json").read_text())
    differences=[];models=0;base={};columns={};native_differences=[]
    for seed in manifest["seeds"]:
        fv=fold_vector(root,frame,seed,load_v5_spec(root))
        if digest(fv.tolist())!=manifest["fold_hashes"][str(seed)]:raise ValueError("Fold identity mismatch")
        base[seed]={t:np.full(len(frame),np.nan) for t in TARGETS}
        for t,arms in manifest["candidates"].items():
            for arm in arms:columns[seed,t,arm]=np.full(len(frame),np.nan)
        for fold in range(5):
            mask=fv==fold;training=frame.loc[~mask].reset_index(drop=True)
            query=frame.loc[mask,["sample_id","spout_no",*FEATURES]].reset_index(drop=True)
            refkey=f"reference-s{seed}-f{fold}"
            verified_unit(out/refkey,unit_id(manifest,refkey))
            refs=dict(np.load(out/refkey/"predictions.npz"))
            np.testing.assert_array_equal(refs["query_ids"],query.sample_id.to_numpy(dtype=str))
            np.testing.assert_array_equal(refs["tap_iron"],.5*refs["v36_iron"]+.5*refs["v12_iron"])
            np.testing.assert_array_equal(refs["tap_time_len"],.2*refs["v36_time"]+.3*refs["n_time"]+.5*refs["v7_time"])
            if not manifest["development"]:
                cached,_=read_reference(root/spec["reference_cache"]["directory"]/f"reference-outer-s{seed}-f{fold}",training,query,spec)
                for k in cached:np.testing.assert_array_equal(refs[k],cached[k])
            for target in TARGETS:base[seed][target][mask]=refs[target]
            for target,arms in manifest["candidates"].items():
                for arm in arms:
                    init_seeds=[42,1042] if seed==42 and fold==0 and not manifest["development"] and spec["diagnostics"].get("initialization_enabled",True) else [42]
                    for init in init_seeds:
                        key=f"{target}-{arm}-s{seed}-f{fold}"+(f"-init{init}" if init!=42 else "")
                        verified_unit(out/key,unit_id(manifest,key))
                        meta=json.loads((out/key/"metadata.json").read_text())
                        settings=dict(spec["training"][target],random_seed=init)
                        mechanisms=spec["mechanisms"]
                        if arm=="BASE" and not manifest["development"] and "base_replay_cache" in spec:
                            source_spec=spec["base_replay_cache"]["source_spec"]
                            cached=json.loads((root/spec["base_replay_cache"]["directory"]/"manifest.json").read_text())
                            if file_hash(root/source_spec)!=cached["source_hashes"][source_spec]:raise ValueError("BASE source spec mismatch")
                            mechanisms=yaml.safe_load((root/source_spec).read_text())["mechanisms"]
                        inner=group_safe_inner_folds(training,seed=settings["inner_seed"])["fold"]
                        fitting=training.loc[inner!=0].reset_index(drop=True);validation=training.loc[inner==0]
                        # Match native y[inner != 0] array layout: rebuilding a
                        # DataFrame array changes NumPy reduction order.
                        fitting_y=training[outputs(target)].to_numpy()[inner!=0]
                        selector=verify_saved(out/key/"selection.pt",fitting,fitting_y,arm,settings,mechanisms,validation,model_type=model_class(arm,spec))
                        model=verify_saved(out/key/"refit.pt",training,training[outputs(target)].to_numpy(),arm,settings,mechanisms,expected_epoch=selector.saved["trace"]["selected_epoch"],model_type=model_class(arm,spec))
                        for phase,m in [("selection",selector),("refit",model)]:
                            if m.saved["trace"]!=meta["model"]["traces"][phase]:raise ValueError("Metadata trace mismatch")
                        with np.load(out/key/"predictions.npz") as saved:
                            pred=saved["prediction"].copy()
                            np.testing.assert_array_equal(saved["query_ids"],query.sample_id.to_numpy(dtype=str))
                        np.testing.assert_array_equal(model.predict(query),pred)
                        variants=[model.predict(query.iloc[::-1])[::-1],np.concatenate([model.predict(query.iloc[i:i+37]) for i in range(0,len(query),37)])]
                        difference=max(float(np.max(np.abs(v-pred))) for v in variants)
                        if difference>spec["preflight"]["cold_predict_atol"]:raise ValueError("Order/chunk mismatch")
                        differences.append(difference);models+=2
                        if init==42:
                            old=refs["v12_iron" if target=="tap_iron" else "v7_time"]
                            columns[seed,target,arm][mask]=refs[target]+.5*(pred[:,0]-old)
                            if arm=="BASE":
                                np.testing.assert_array_equal(pred[:,0],old);native_differences.append(float(np.max(np.abs(pred[:,0]-old))))
    gains={}
    for row in summary["records"]:
        target,arm=row["target"],row["arm"];y=frame[target].to_numpy();actual=[]
        for seed in manifest["seeds"]:
            p=columns[seed,target,arm]
            if not np.isfinite(p).all():raise ValueError("Incomplete coverage")
            gain=float(50*(np.abs(y-base[seed][target]).sum()-np.abs(y-p).sum())/np.abs(y).sum())
            if abs(gain-row["seeds"][str(seed)]["gain"])>1e-10:raise ValueError("Score arithmetic mismatch")
            other=next(t for t in TARGETS if t!=target)
            score=100-50*(np.abs(y-p).sum()/np.abs(y).sum()+np.abs(frame[other].to_numpy()-base[seed][other]).sum()/np.abs(frame[other].to_numpy()).sum())
            if abs(score-row["seeds"][str(seed)]["candidate_score"])>1e-10:raise ValueError("Package score mismatch")
            actual.append(gain)
        gains[target,arm]=actual
    selected={}
    for target in TARGETS:
        eligible=[a for a in manifest["candidates"][target] if a in spec["promotion"]["eligible_recipes"]
                  and min(gains[target,a])>0 and np.mean(gains[target,a])>=spec["promotion"]["development_mean_gain_ge"]
                  and np.mean(gains[target,a])>np.mean(gains[target,"BASE"])]
        selected[target]=min(eligible,key=lambda a:(-np.mean(gains[target,a]),spec["tie_preference_by_target"][target].index(a))) if eligible else None
    if selected!=summary["selected_for_confirmation"]:raise ValueError("Selection mismatch")
    if manifest["development"]:
        from .v5_resolution import paired_summary
        dev=root/manifest["development"]
        if file_hash(dev/"summary.json")!=manifest["development_summary_sha256"]:raise ValueError("Development changed")
        old=json.loads((dev/"summary.json").read_text())
        for decision in summary["four_seed_decisions"]:
            target,arm=decision["target"],decision["arm"]
            previous=next(r for r in old["records"] if (r["target"],r["arm"])==(target,arm))
            combined=list(v["gain"] for v in previous["seeds"].values())+gains[target,arm]
            paired=paired_summary(combined)
            score=float(np.mean([v["candidate_score"] for v in previous["seeds"].values()]))
            passed=all(v>0 for v in combined) and paired["lcb95"]>0 and score>=spec["promotion"]["local_working_gate"]
            if decision["promoted"]!=passed:raise ValueError("Four-seed decision mismatch")
    report={"status":"passed","cold_models":models,"native_replays":len(native_differences),
        "native_max_difference":max(native_differences),"full_batch_cold_difference":0.,
        "maximum_order_chunk_difference":max(differences),"selection_verified":True,
        "summary_sha256":file_hash(out/"summary.json"),"manifest_sha256":file_hash(out/"manifest.json"),
        "auditor_sha256":file_hash(Path(__file__)),"new_fits":0,"packages":0}
    write_new(out/"audit.json",report);print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--spec",default=SPEC);a=p.parse_args();run(Path.cwd(),a.output,a.spec)
