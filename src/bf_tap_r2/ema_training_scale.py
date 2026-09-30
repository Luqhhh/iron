"""Frozen paired training-size diagnostic, using the original scientific trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .data import FEATURES, TARGETS
from .v7_periodic import digest, file_hash, write_new

SPEC = "configs/ema_training_scale/SPEC.json"


def event(path, payload):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(dict(time_ns=time.time_ns(), **payload), allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_files(root, files):
    root = Path(root).resolve()
    for name, expected in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != expected:
            raise ValueError(f"Frozen file identity changed: {name}")


def sources(workspace):
    workspace = Path(workspace)
    paths = list((workspace/"src").rglob("*.py")) + list((workspace/"tests").glob("*.py"))
    for extension in ("*.yaml", "*.json"):
        paths += list((workspace/"configs").rglob(extension))
    paths += [workspace/p for p in ("uv.lock", "pyproject.toml", SPEC,
              "docs/ema_training_scale/PREREGISTRATION.md", "scripts/ema_training_scale.py")]
    return {str(p.relative_to(workspace)):file_hash(p) for p in sorted(set(paths))}


def nested_subsets(frame, fractions=(.25,.5,.75,1.), seed=961046):
    """Select whole numerical duplicate groups, without consulting any target."""
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("Unique training IDs required")
    if tuple(fractions) != (.25,.5,.75,1.):
        raise ValueError("Frozen fractions required")
    keys = pd.util.hash_pandas_object(frame[list(FEATURES)],index=False).astype(str).to_numpy()
    strata = {}
    for group in sorted(set(keys)):
        spouts = tuple(sorted(int(s) for s in frame.loc[keys==group,"spout_no"].unique()))
        strata.setdefault(spouts,[]).append(group)
    rng = np.random.default_rng(seed)
    ranked = {s:list(rng.permutation(groups)) for s,groups in sorted(strata.items())}
    result = {}
    for fraction in fractions:
        selected = set()
        for groups in ranked.values():
            selected.update(groups[:math.ceil(fraction*len(groups))])
        ids = frame.loc[np.isin(keys,list(selected)),"sample_id"].tolist()
        if not ids:
            raise ValueError("Empty training subset")
        result[str(fraction)] = ids
    return result


def assert_isolated(training, query):
    if set(training.sample_id) & set(query.sample_id):
        raise ValueError("Training/query ID overlap")
    train_keys = set(pd.util.hash_pandas_object(training[list(FEATURES)],index=False))
    query_keys = set(pd.util.hash_pandas_object(query[list(FEATURES)],index=False))
    if train_keys & query_keys:
        raise ValueError("Training/query duplicate-group overlap")
    if any(t in query for t in TARGETS):
        raise ValueError("Prediction query contains targets")


def runtime(spec):
    from .v49_run import check_runtime
    import torch
    if sys.version_info[:2] != (3,12):
        raise ValueError("Locked Python3.12 required")
    versions = check_runtime(spec)
    torch.set_num_threads(1)
    return versions


def require_serial(spec):
    root=Path(spec["previous_release_root"])
    terminal=json.loads((root/"completion-event.json").read_text())
    audit=json.loads((root/"terminal-verification.json").read_text())
    if (terminal["status"] != "completed" or audit["status"] != "passed"
            or audit["terminal_sha256"] != file_hash(root/"completion-event.json")
            or audit["manifest_sha256"] != file_hash(root/"manifest.json")):
        raise ValueError("Successful audited previous terminal required")
    values=subprocess.check_output(["systemctl","--user","show",spec["previous_service"],
        "--property=MainPID,ActiveState,SubState,ExecMainStatus"],text=True)
    parsed=dict(line.split("=",1) for line in values.splitlines())
    if parsed.get("MainPID") != "0" or parsed.get("ExecMainStatus") != "0" or parsed.get("ActiveState") != "inactive":
        raise ValueError("Previous training service still active or unsuccessful")
    return dict(service=spec["previous_service"],properties=parsed,terminal_sha256=file_hash(root/"completion-event.json"),
                audit_sha256=file_hash(root/"terminal-verification.json"))


def require_memory(spec):
    import psutil
    available = psutil.virtual_memory().available/2**20
    if available < spec["max_worker_rss_mib"]*2:
        raise ValueError("Insufficient currently available memory")
    return available


def context(workspace, manifest):
    from .v5_library import load_v5_training_frame,fold_vector
    from .v5_spec import load_v5_spec
    workspace=Path(workspace)
    verify_files(workspace,manifest["sources"])
    spec=manifest["spec"];main=Path(spec["main_root"])
    verify_files(main,spec["inputs"])
    runtime(spec)
    frame=load_v5_training_frame(main)
    folds={s:fold_vector(main,frame,s,load_v5_spec(main)) for s in spec["split_seeds"]}
    if {str(s):digest(v.tolist()) for s,v in folds.items()} != manifest["fold_digests"]:
        raise ValueError("Frozen outer folds changed")
    return spec,frame,folds


def prepare(workspace, receipt_path):
    import yaml
    from .v5_library import load_v5_training_frame,fold_vector
    from .v5_spec import load_v5_spec
    workspace=Path(workspace).resolve();spec=json.loads((workspace/SPEC).read_text())
    main=Path(spec["main_root"]);out=Path(spec["output"]).resolve()
    if not out.is_relative_to(main/"local/runs") or out.exists():
        raise ValueError("Fresh private output required")
    verify_files(main,spec["inputs"])
    original=yaml.safe_load((main/"configs/strong_component_regularization/SPEC.yaml").read_text())
    if (spec["training"] != original["training"]["tap_time_len"] or spec["mechanisms"] != original["mechanisms"]
            or spec["split_seeds"] != [42,3407] or spec["held_fold"] != 0
            or spec["arms"] != ["BASE","EMA","SAM"] or spec["fractions"] != [.25,.5,.75,1.]
            or spec["subset_seed"] != 961046 or spec["workers"] != 1
            or spec["new_estimators"] != 18 or spec["new_optimizer_runs"] != 36
            or any(spec[k] != 0 for k in ("full_data_fits","packages","desktop_writes","agent_uploads"))):
        raise ValueError("Original scientific protocol or execution scope changed")
    admitted=json.loads((main/"local/runs/strong-component-regularization/preflight-r1/report.json").read_text())
    if (admitted["status"] != "passed" or file_hash(main/"local/runs/strong-component-regularization/preflight-r1/report.json")
            != spec["original_preflight_sha256"]):
        raise ValueError("Original full-size resource admission changed")
    old=json.loads((Path(spec["old_development"])/"manifest.json").read_text())
    old_audit=json.loads((Path(spec["old_development"])/"audit.json").read_text())
    if old_audit["status"] != "passed" or old_audit["manifest_sha256"] != file_hash(Path(spec["old_development"])/"manifest.json"):
        raise ValueError("Audited original model cache required")
    # Compare all original scientific dependencies, without expanding its frozen inventory.
    verify_files(workspace,old["source_hashes"])
    versions=runtime(spec);serial=require_serial(spec);available=require_memory(spec)
    current=json.loads((main/"EVIDENCE_STATUS.json").read_text())["round2_current_platform_best"]
    if any(current[k] != v for k,v in spec["reference"].items()):
        raise ValueError("Current platform reference changed before launch")
    receipt=json.loads(Path(receipt_path).read_text())
    snapshot=sources(workspace)
    if receipt.get("status") != "passed" or receipt.get("sources") != snapshot or receipt.get("full_suite") is not True:
        raise ValueError("Exact-source passed full locked suite required")
    dirty=subprocess.check_output(["git","status","--porcelain","--",*snapshot],cwd=workspace,text=True)
    if dirty.strip():
        raise ValueError("Commit tested scientific sources before preparation")
    frame=load_v5_training_frame(main)
    folds={s:fold_vector(main,frame,s,load_v5_spec(main)) for s in spec["split_seeds"]}
    plan={}
    for seed,fv in folds.items():
        training=frame.loc[fv!=0].reset_index(drop=True)
        query=frame.loc[fv==0,["sample_id","spout_no",*FEATURES]].reset_index(drop=True)
        assert_isolated(training,query)
        subset=nested_subsets(training,spec["fractions"],spec["subset_seed"])
        plan[str(seed)]=dict(query_ids=query.sample_id.tolist(),subsets=subset,
                            training_pool_rows=len(training),query_rows=len(query))
    out.mkdir(parents=True)
    manifest=dict(spec=spec,sources=snapshot,versions=versions,plan=plan,
                  fold_digests={str(s):digest(fv.tolist()) for s,fv in folds.items()},
                  original_manifest_sha256=file_hash(Path(spec["old_development"])/"manifest.json"),
                  receipt_path=str(Path(receipt_path).resolve()),receipt_sha256=file_hash(receipt_path),
                  serial_admission=serial,available_memory_mib=available,workspace=str(workspace),
                  source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=workspace,text=True).strip())
    write_new(out/"manifest.json",manifest)
    return out


def task_frames(frame,plan,seed,fraction):
    ids=plan[str(seed)]["subsets"][str(fraction)]
    indexed=frame.set_index("sample_id",drop=False)
    training=indexed.loc[ids].reset_index(drop=True)
    query=indexed.loc[plan[str(seed)]["query_ids"],["sample_id","spout_no",*FEATURES]].reset_index(drop=True)
    assert_isolated(training,query)
    return training,query


def unit_key(seed,fraction,arm):
    return f"s{seed}-q{int(fraction*100):03d}-{arm}"


def fit_worker(workspace,out,seed,fraction,arm):
    from .component_regularization import ComponentRegressor
    from .component_regularization_run import RECIPE
    out=Path(out);manifest=json.loads((out/"manifest.json").read_text())
    spec,frame,_=context(workspace,manifest)
    if seed not in spec["split_seeds"] or fraction not in (.25,.5,.75) or arm not in spec["arms"]:
        raise ValueError("Unregistered training task")
    directory=out/"units"/unit_key(seed,fraction,arm)
    directory.mkdir(exist_ok=False)
    event(out/"events.jsonl",dict(event="unit_started",seed=seed,fraction=fraction,arm=arm))
    start=time.monotonic()
    try:
        available=require_memory(spec)
        training,query=task_frames(frame,manifest["plan"],seed,fraction)
        class RecordedRegressor(ComponentRegressor):
            def _train(self,*args,**kwargs):
                phase="selection" if kwargs.get("validation") is not None or (len(args)>3 and args[3] is not None) else "refit"
                event(directory/"events.jsonl",dict(event="optimizer_started",phase=phase))
                try:
                    result=super()._train(*args,**kwargs)
                except BaseException as exc:
                    event(directory/"events.jsonl",dict(event="optimizer_failed",phase=phase,error=repr(exc)))
                    raise
                event(directory/"events.jsonl",dict(event="optimizer_completed",phase=phase))
                return result
        model=RecordedRegressor(RECIPE,spec["training"],arm,spec["mechanisms"],directory)
        model.fit(training,training[[spec["target"]]].to_numpy())
        prediction=model.predict(query)
        train_prediction=model.predict(training.drop(columns=[t for t in TARGETS if t in training]))
        with (directory/"predictions.npz").open("xb") as stream:
            np.savez_compressed(stream,prediction=prediction,query_ids=query.sample_id.to_numpy(str),
                train_prediction=train_prediction,training_ids=training.sample_id.to_numpy(str))
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if peak>spec["max_worker_rss_mib"]:
            raise ValueError("Original worker memory gate failed")
        write_new(directory/"metadata.json",dict(seed=seed,fraction=fraction,arm=arm,model=model.metadata_,
                  query_ids_digest=digest(query.sample_id.tolist()),training_ids_digest=digest(training.sample_id.tolist()),
                  seconds=time.monotonic()-start,peak_rss_mib=peak,available_memory_at_entry_mib=available))
        context(workspace,manifest)
        write_new(directory/"complete.json",dict(seed=seed,fraction=fraction,arm=arm,
            manifest_sha256=file_hash(out/"manifest.json"),
            hashes={p.name:file_hash(p) for p in directory.iterdir() if p.name != "events.jsonl"},new_optimizer_runs=2))
        event(out/"events.jsonl",dict(event="unit_completed",seed=seed,fraction=fraction,arm=arm))
    except BaseException as exc:
        event(out/"events.jsonl",dict(event="unit_failed",seed=seed,fraction=fraction,arm=arm,error=repr(exc)))
        raise


def verify_one(manifest,directory,training,query,seed,fraction,arm,reused=False):
    from .component_regularization_audit import verify_saved
    from .v3_4_bags import group_safe_inner_folds
    spec=manifest["spec"];directory=Path(directory)
    complete=json.loads((directory/"complete.json").read_text())
    verify_files(directory,complete["hashes"])
    y=training[[spec["target"]]].to_numpy()
    inner=group_safe_inner_folds(training,seed=spec["training"]["inner_seed"])["fold"]
    fit=training.loc[inner!=0].reset_index(drop=True)
    valid=training.loc[inner==0]
    selector=verify_saved(directory/"selection.pt",fit,y[inner!=0],arm,spec["training"],spec["mechanisms"],valid)
    model=verify_saved(directory/"refit.pt",training,y,arm,spec["training"],spec["mechanisms"],
                       expected_epoch=selector.saved["trace"]["selected_epoch"])
    saved=np.load(directory/"predictions.npz",allow_pickle=False)
    np.testing.assert_array_equal(saved["query_ids"],query.sample_id.to_numpy(str))
    prediction=model.predict(query)
    np.testing.assert_array_equal(prediction,saved["prediction"])
    errors=[float(np.max(np.abs(model.predict(query.iloc[::-1])[::-1]-prediction))),
            float(np.max(np.abs(np.concatenate([model.predict(query.iloc[i:i+37]) for i in range(0,len(query),37)])-prediction)))]
    if max(errors)>spec["order_chunk_atol"]:
        raise ValueError("Original order/chunk tolerance failed")
    train_pred=model.predict(training.drop(columns=[t for t in TARGETS if t in training]))
    if not reused:
        np.testing.assert_array_equal(saved["training_ids"],training.sample_id.to_numpy(str))
        np.testing.assert_array_equal(saved["train_prediction"],train_pred)
        ledger=[json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
        if [(r["event"],r["phase"]) for r in ledger] != [(e,p) for p in ("selection","refit")
                for e in ("optimizer_started","optimizer_completed")]:
            raise ValueError("Expected exactly two closed optimizer reservations")
    target=training[spec["target"]].to_numpy(float)
    trace=selector.saved["trace"];selected=trace["history"][trace["selected_epoch"]-1]
    return dict(seed=seed,fraction=fraction,arm=arm,training_rows=len(training),query_rows=len(query),
                prediction=prediction[:,0],selected_epoch=trace["selected_epoch"],stopped_epoch=trace["stopped_epoch"],
                inner_train_mae_standardized=selected["training_eval_mae"],inner_validation_mae_standardized=selected["validation_mae"],
                inner_gap_standardized=selected["validation_mae"]-selected["training_eval_mae"],
                refit_training_mae=float(np.abs(target-train_pred[:,0]).mean()),
                selector_updates=trace["updates"],refit_updates=model.saved["trace"]["updates"],
                gradient_evaluations=trace["gradient_evaluations"]+model.saved["trace"]["gradient_evaluations"],
                cap_hit=trace["stopped_epoch"]==spec["training"]["max_epochs"],
                maximum_order_chunk_difference=max(errors),cold_difference=0.,reused=reused)


def reuse_admission(workspace,out):
    out=Path(out);manifest=json.loads((out/"manifest.json").read_text())
    spec,frame,_=context(workspace,manifest);rows=[]
    for seed in spec["split_seeds"]:
        training,query=task_frames(frame,manifest["plan"],seed,1.)
        for arm in spec["arms"]:
            directory=Path(spec["old_development"])/f"tap_time_len-{arm}-s{seed}-f0"
            row=verify_one(manifest,directory,training,query,seed,1.,arm,reused=True)
            row.pop("prediction");rows.append(row)
    require_serial(spec);require_memory(spec)
    write_new(out/"reuse-admission.json",dict(status="passed",rows=rows,saved_models=12,new_fits=0,
                  manifest_sha256=file_hash(out/"manifest.json")))


def audit(workspace,out):
    out=Path(out);manifest=json.loads((out/"manifest.json").read_text())
    spec,frame,folds=context(workspace,manifest);rows=[];max_error=0.
    # Forbid any fit call in this process while replaying saved states.
    from .component_regularization import ComponentRegressor
    def no_fit(*args,**kwargs):
        raise AssertionError("Audit attempted fitting")
    ComponentRegressor.fit=no_fit
    for seed in spec["split_seeds"]:
        pool=frame.loc[folds[seed]!=0].reset_index(drop=True)
        if nested_subsets(pool,spec["fractions"],spec["subset_seed"])!=manifest["plan"][str(seed)]["subsets"]:
            raise ValueError("Subset plan changed")
        eval_y=frame.loc[folds[seed]==0,spec["target"]].to_numpy(float)
        for fraction in spec["fractions"]:
            training,query=task_frames(frame,manifest["plan"],seed,fraction)
            per_arm={}
            for arm in spec["arms"]:
                reused=fraction==1.
                directory=(Path(spec["old_development"])/f"tap_time_len-{arm}-s{seed}-f0" if reused
                           else out/"units"/unit_key(seed,fraction,arm))
                if not reused:
                    complete=json.loads((directory/"complete.json").read_text())
                    if complete["manifest_sha256"]!=file_hash(out/"manifest.json"):
                        raise ValueError("Unit belongs to another manifest")
                row=verify_one(manifest,directory,training,query,seed,fraction,arm,reused)
                prediction=row.pop("prediction")
                if prediction.shape!=eval_y.shape or not np.isfinite(prediction).all():
                    raise ValueError("Invalid held-out prediction")
                mae=float(np.abs(eval_y-prediction).mean());wmape=float(np.abs(eval_y-prediction).sum()/eval_y.sum())
                row.update(held_out_mae=mae,held_out_wmape=wmape,refit_gap_raw=mae-row["refit_training_mae"],
                    by_spout={str(s):dict(rows=int((query.spout_no==s).sum()),
                        mae=float(np.abs(eval_y[query.spout_no==s]-prediction[query.spout_no==s]).mean()))
                        for s in sorted(query.spout_no.unique())})
                per_arm[arm]=(row,prediction)
                event(out/"audit-events.jsonl",dict(event="saved_unit_verified",seed=seed,fraction=fraction,arm=arm))
            base,base_pred=per_arm["BASE"]
            for arm,(row,pred) in per_arm.items():
                reductions=[abs(float(y-b))-abs(float(y-c)) for y,b,c in zip(eval_y,base_pred,pred)]
                oracle=50*math.fsum(reductions)/math.fsum(eval_y)
                score_gain=50*(base["held_out_wmape"]-row["held_out_wmape"])
                max_error=max(max_error,abs(oracle-score_gain))
                row.update(mae_reduction_vs_BASE=base["held_out_mae"]-row["held_out_mae"],
                           score_gain_vs_BASE=score_gain,
                           by_spout_mae_reduction_vs_BASE={s:base["by_spout"][s]["mae"]-v["mae"] for s,v in row["by_spout"].items()})
                rows.append(row)
    if max_error>1e-10:
        raise ValueError("Independent MAE/WMAPE arithmetic disagrees")
    if len(rows)!=24 or sum(not r["reused"] for r in rows)!=18:
        raise ValueError("Incomplete diagnostic coverage")
    contrasts=[]
    for seed in spec["split_seeds"]:
        for arm in ("EMA","SAM"):
            full=next(r for r in rows if r["seed"]==seed and r["fraction"]==1 and r["arm"]==arm)
            for fraction in (.25,.5,.75):
                small=next(r for r in rows if r["seed"]==seed and r["fraction"]==fraction and r["arm"]==arm)
                contrasts.append(dict(seed=seed,arm=arm,small_fraction=fraction,
                    full_minus_small_method_gain=full["score_gain_vs_BASE"]-small["score_gain_vs_BASE"]))
    context(workspace,manifest)
    write_new(out/"summary.json",dict(G0="passed",G1="fixed_two_fold_mechanism_diagnostic_not_promotion",
        rows=rows,scale_interactions=contrasts,reference=spec["reference"],new_estimators=18,new_optimizer_runs=36,
        reused_units=6,saved_models_checked=48,independent_arithmetic_max_difference=max_error,
        packages=0,desktop_writes=0,agent_uploads=0,historical_decisions_unchanged=True))
    write_new(out/"audit.json",dict(status="passed",manifest_sha256=file_hash(out/"manifest.json"),
        summary_sha256=file_hash(out/"summary.json"),new_fits=0,saved_models=48,
        max_arithmetic_difference=max_error,full_batch_cold_difference=0.,
        maximum_order_chunk_difference=max(r["maximum_order_chunk_difference"] for r in rows),
        artifact_hashes={str(p.relative_to(out)):file_hash(p) for p in sorted((out/"units").rglob("*")) if p.is_file()}))


def supervise(workspace,out):
    out=Path(out);manifest=json.loads((out/"manifest.json").read_text())
    spec,_,_=context(workspace,manifest);require_serial(spec)
    write_new(out/"run-start.json",dict(time_ns=time.time_ns(),pid=os.getpid(),manifest_sha256=file_hash(out/"manifest.json")))
    (out/"units").mkdir(exist_ok=False)
    script=Path(workspace)/"scripts/ema_training_scale.py"
    next_observation=time.monotonic()+spec["interval_seconds"]
    def stage(name,command):
        nonlocal next_observation
        with (out/f"{name}.log").open("x") as stream:
            child=subprocess.Popen(command,cwd=workspace,stdout=stream,stderr=subprocess.STDOUT)
            event(out/"workflow-events.jsonl",dict(event="stage_started",stage=name,pid=child.pid))
            while True:
                try:
                    code=child.wait(timeout=max(0.,next_observation-time.monotonic()))
                    break
                except subprocess.TimeoutExpired:
                    events=[json.loads(line) for line in (out/"events.jsonl").read_text().splitlines()] if (out/"events.jsonl").exists() else []
                    event(out/"scheduled-observations.jsonl",dict(stage=name,pid=child.pid,
                        completed_new_units=sum(r["event"]=="unit_completed" for r in events)))
                    next_observation=time.monotonic()+spec["interval_seconds"]
            event(out/"workflow-events.jsonl",dict(event="stage_completed",stage=name,returncode=code))
            if code:
                raise RuntimeError(f"{name} failed; no retry")
    try:
        stage("reuse-admission",[sys.executable,str(script),"reuse","--out",str(out)])
        for fraction in (.25,.5,.75):
            for seed in spec["split_seeds"]:
                for arm in spec["arms"]:
                    stage(unit_key(seed,fraction,arm),[sys.executable,str(script),"worker","--out",str(out),
                        "--seed",str(seed),"--fraction",str(fraction),"--arm",arm])
        stage("independent-audit",[sys.executable,str(script),"audit","--out",str(out)])
        write_new(out/"completion-event.json",dict(status="completed",time_ns=time.time_ns(),
            audit_sha256=file_hash(out/"audit.json"),new_estimators=18,new_optimizer_runs=36,packages=0))
    except BaseException as exc:
        write_new(out/"completion-event.json",dict(status="failed",time_ns=time.time_ns(),error=repr(exc)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","supervise","worker","reuse","audit"])
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--receipt",type=Path)
    parser.add_argument("--seed",type=int)
    parser.add_argument("--fraction",type=float)
    parser.add_argument("--arm")
    args=parser.parse_args()
    if args.mode=="prepare":
        print(prepare(args.workspace,args.receipt))
    elif args.mode=="worker":fit_worker(args.workspace,args.out,args.seed,args.fraction,args.arm)
    elif args.mode=="reuse":reuse_admission(args.workspace,args.out)
    elif args.mode=="audit":audit(args.workspace,args.out)
    else:supervise(args.workspace,args.out)
