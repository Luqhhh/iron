"""Frozen fixed-weight replacement runner; all recoverable artifacts private."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import yaml

from .candidate_tiers import classify_candidates
from .component_regularization import ComponentRegressor, replacement, paired_row_bootstrap
from .data import FEATURES, TARGETS
from .v5_library import load_v5_training_frame, fold_vector
from .v5_spec import load_v5_spec
from .v5_resolution import paired_summary
from .v7_periodic import digest, file_hash, write_new
from .v49_run import (append_event, check_runtime, verify_hashes, verify_reference_cache,
                      verified_unit, unit_id, read_reference, metric_detail)

SPEC = "configs/strong_component_regularization/SPEC.yaml"
RUN_ROOT = "local/runs/strong-component-regularization"
RECIPE = {"backbone":"tabm","frequency":.01}


def sources(root):
    paths=list((root/"src/bf_tap_r2").glob("*.py"))+list((root/"configs").rglob("*.yaml"))
    paths += [root/"uv.lock",root/"pyproject.toml",root/"docs/strong_component_regularization/PREREGISTRATION.md"]
    paths += [root/"tests/test_component_regularization.py"]
    return {str(p.relative_to(root)):file_hash(p) for p in paths}


def outputs(target):
    return list(TARGETS) if target=="tap_iron" else ["tap_time_len"]


def choose(records, spec):
    selected={}
    for target in TARGETS:
        rows=[r for r in records if r["target"]==target]
        base=next(r for r in rows if r["arm"]=="BASE")
        eligible=[r for r in rows if r["arm"] in ("EMA","SAM")
            and min(v["gain"] for v in r["seeds"].values())>0
            and r["paired"]["mean"]>=spec["promotion"]["development_mean_gain_ge"]
            and r["paired"]["mean"]>base["paired"]["mean"]]
        selected[target]=min(eligible,key=lambda r:(-r["paired"]["mean"],r["arm"]!="EMA"))["arm"] if eligible else None
    return selected


def fit_unit(root, out, frame, fv, seed, fold, target, arm, spec, manifest, train_seed=42):
    suffix="" if train_seed==42 else f"-init{train_seed}"
    key=f"{target}-{arm}-s{seed}-f{fold}{suffix}"
    directory=out/key;identity=unit_id(manifest,key)
    cached=verified_unit(directory,identity)
    if cached:return {"key":key,"cached":True}
    directory.mkdir(exist_ok=False)
    append_event(directory/"events.jsonl",{"event":"start","identity":identity})
    start=time.monotonic()
    try:
        training=frame.loc[fv!=fold].reset_index(drop=True)
        query=frame.loc[fv==fold,["sample_id","spout_no",*FEATURES]].reset_index(drop=True)
        settings=dict(spec["training"][target],random_seed=train_seed)
        model=ComponentRegressor(RECIPE,settings,arm,spec["mechanisms"],directory)
        model.fit(training,training[outputs(target)].to_numpy())
        prediction=model.predict(query)
        with (directory/"predictions.npz").open("xb") as stream:
            np.savez_compressed(stream,prediction=prediction,query_ids=query.sample_id.to_numpy(dtype=str))
        metadata={"target":target,"arm":arm,"seed":seed,"fold":fold,"training_seed":train_seed,
                  "model":model.metadata_,"seconds":time.monotonic()-start,
                  "fit_ids_digest":digest(training.sample_id.tolist())}
        if arm=="BASE" and train_seed==42:
            with np.load(out/f"reference-s{seed}-f{fold}"/"predictions.npz") as cache:
                expected=cache["v12_iron" if target=="tap_iron" else "v7_time"]
            delta=float(np.max(np.abs(prediction[:,0]-expected)))
            metadata["native_replay_max_difference"]=delta
            if not np.array_equal(prediction[:,0],expected):
                write_new(directory/"replay_failure.json",metadata)
                raise ValueError(f"Native BASE replay failed: {delta}")
        write_new(directory/"metadata.json",metadata)
        result={"identity":identity,"hashes":{p.name:file_hash(p) for p in directory.iterdir() if p.suffix in (".pt",".npz",".json")},
                "key":key,"seconds":metadata["seconds"]}
        write_new(directory/"complete.json",result)
        append_event(directory/"events.jsonl",{"event":"complete","seconds":metadata["seconds"]})
        return result
    except BaseException as exc:
        append_event(directory/"events.jsonl",{"event":"failed","error":repr(exc)})
        raise


def references(root,out,frame,folds,spec,manifest,confirmation):
    for seed,fv in folds.items():
        for fold in range(5):
            key=f"reference-s{seed}-f{fold}";directory=out/key
            identity=unit_id(manifest,key)
            if verified_unit(directory,identity):continue
            training=frame.loc[fv!=fold].reset_index(drop=True)
            query=frame.loc[fv==fold,["sample_id","spout_no",*FEATURES]].reset_index(drop=True)
            directory.mkdir(exist_ok=False)
            if confirmation:
                from .v30_reference import fit_b0
                values,metadata=fit_b0(root,training,query,spec)
            else:
                source=root/spec["reference_cache"]["directory"]/f"reference-outer-s{seed}-f{fold}"
                values,metadata=read_reference(source,training,query,spec)
            write_new(directory/"metadata.json",metadata)
            with (directory/"predictions.npz").open("xb") as stream:
                np.savez_compressed(stream,**values,query_ids=query.sample_id.to_numpy(dtype=str))
            write_new(directory/"complete.json",{"identity":identity,"hashes":{p.name:file_hash(p) for p in directory.iterdir()}})


def collect(out,frame,folds,candidates):
    base={};members={}
    for seed,fv in folds.items():
        base[seed]={t:np.full(len(frame),np.nan) for t in [*TARGETS,"v12_iron","v7_time"]}
        for target,arms in candidates.items():
            for arm in arms:members[seed,target,arm]=np.full(len(frame),np.nan)
        for fold in range(5):
            mask=fv==fold
            with np.load(out/f"reference-s{seed}-f{fold}/predictions.npz") as saved:
                if not np.array_equal(saved["query_ids"],frame.loc[mask,"sample_id"].to_numpy(dtype=str)):
                    raise ValueError("Reference row identity")
                for t in base[seed]:base[seed][t][mask]=saved[t]
            for target,arms in candidates.items():
                for arm in arms:
                    with np.load(out/f"{target}-{arm}-s{seed}-f{fold}/predictions.npz") as saved:
                        if not np.array_equal(saved["query_ids"],frame.loc[mask,"sample_id"].to_numpy(dtype=str)):
                            raise ValueError("Candidate row identity")
                        members[seed,target,arm][mask]=saved["prediction"][:,0]
    return base,members


def summarize(out,frame,folds,spec,candidates):
    base,members=collect(out,frame,folds,candidates)
    metrics={t:{"CURRENT_FIXED_REPLACEMENT":{}} for t in TARGETS};records=[]
    for target,arms in candidates.items():
        y=frame[target].to_numpy();other=next(t for t in TARGETS if t!=target)
        old_key="v12_iron" if target=="tap_iron" else "v7_time"
        for seed,fv in folds.items():
            metrics[target]["CURRENT_FIXED_REPLACEMENT"][str(seed)]=metric_detail(y,base[seed][target],fv,frame.spout_no.to_numpy())
        for arm in arms:
            metrics[target][arm]={};rows={};combined=[]
            for seed,fv in folds.items():
                member=members[seed,target,arm]
                candidate=replacement(base[seed][target],base[seed][old_key],member)
                combined.append(candidate)
                details=metric_detail(y,candidate,fv,frame.spout_no.to_numpy())
                metrics[target][arm][str(seed)]=details
                original=metrics[target]["CURRENT_FIXED_REPLACEMENT"][str(seed)]["wmape"]
                gain=50*(original-details["wmape"])
                current_score=100-50*(original+np.abs(frame[other].to_numpy()-base[seed][other]).sum()/np.abs(frame[other].to_numpy()).sum())
                raw_gain=50*(np.abs(y-base[seed][old_key]).sum()-np.abs(y-member).sum())/np.abs(y).sum()
                contribution=np.abs(y-base[seed][target])-np.abs(y-candidate)
                positive=contribution.clip(0)
                top=max(1,int(np.ceil(.01*len(y))))
                high=np.zeros(len(y),bool)
                for f in range(5):high[fv==f]=y[fv==f]>=np.quantile(y[fv!=f],spec["diagnostics"]["top_target_quantile"])
                rows[str(seed)]={"gain":gain,"candidate_score":current_score+gain,"reference_score":current_score,
                    "standalone_component_score_gain":float(raw_gain),
                    "standalone_wmape":float(np.abs(y-member).sum()/np.abs(y).sum()),
                    "positive_folds_descriptive":sum(details["by_fold"][str(f)]<metrics[target]["CURRENT_FIXED_REPLACEMENT"][str(seed)]["by_fold"][str(f)] for f in range(5)),
                    "top_one_percent_share_of_positive_error_reduction":float(np.sort(positive)[-top:].sum()/positive.sum()) if positive.sum()>0 else None,
                    "top_target_decile_error_change":float(-contribution[high].sum()),
                    "maximum_row_error_reduction":float(contribution.max()),
                    "replacement_wmape":details["wmape"]}
            bootstrap=paired_row_bootstrap(y,np.stack([base[s][target] for s in folds]),np.stack(combined),
                spec["diagnostics"]["bootstrap_replicates"],spec["diagnostics"]["bootstrap_seed"])
            records.append({"target":target,"arm":arm,"seeds":rows,
                "paired":paired_summary([v["gain"] for v in rows.values()]),"bootstrap":bootstrap})
    selected=choose(records,spec)
    tiers=(classify_candidates(metrics,spec,yaml.safe_load(Path("configs/candidate_tiers.yaml").read_text()))
           if set(folds)==set(spec["split_seeds"]) and candidates==spec["candidates"] else None)
    overlap=[]
    for target,arms in candidates.items():
        if not {"EMA","SAM"}.issubset(arms):continue
        y=frame[target].to_numpy();old_key="v12_iron" if target=="tap_iron" else "v7_time"
        for seed in folds:
            effects=[]
            for arm in ("EMA","SAM"):
                pred=replacement(base[seed][target],base[seed][old_key],members[seed,target,arm])
                effects.append(np.abs(y-base[seed][target])-np.abs(y-pred))
            corr=float(np.corrcoef(effects)[0,1]) if all(np.std(v)>0 for v in effects) else None
            overlap.append({"target":target,"seed":seed,"row_error_reduction_correlation":corr,
                            "both_improve_row_fraction":float(np.mean((effects[0]>0)&(effects[1]>0)))})
    return {"records":records,"metrics":metrics,"selected_for_confirmation":selected,"tiers":tiers,
            "mechanism_row_overlap":overlap,
            "packages":0,"agent_uploads":0,"release_authorized":False}


def run(root,out,development=None):
    root=Path(root).resolve();out=(root/out).resolve()
    if not out.is_relative_to(root/RUN_ROOT):raise ValueError("Private output required")
    spec=yaml.safe_load((root/SPEC).read_text());versions=check_runtime(spec)
    preflight=json.loads((root/RUN_ROOT/"preflight-r1/report.json").read_text())
    if preflight["status"]!="passed" or preflight["spec_sha256"]!=file_hash(root/SPEC):
        raise ValueError("Successful matching preflight required")
    verify_hashes(root,preflight["source_hashes"])
    seeds=spec["split_seeds"];candidates=spec["candidates"]
    old=None
    if development:
        dev=(root/development).resolve();audit=json.loads((dev/"audit.json").read_text())
        if audit["status"]!="passed" or audit["summary_sha256"]!=file_hash(dev/"summary.json"):
            raise ValueError("Audited development required")
        old=json.loads((dev/"summary.json").read_text());selected=old["selected_for_confirmation"]
        if not any(selected.values()):
            print(json.dumps({"status":"no_development_finalist","new_fits":0}),flush=True);return
        candidates={t:(["BASE",a] if a else ["BASE"]) for t,a in selected.items()}
        seeds=spec["confirmation_seeds"]
    else:verify_reference_cache(root,spec)
    frame=load_v5_training_frame(root)
    folds={s:fold_vector(root,frame,s,load_v5_spec(root)) for s in seeds}
    manifest={"source_hashes":sources(root),"spec_sha256":file_hash(root/SPEC),"versions":versions,
        "data_hashes":{str(p.relative_to(root)):file_hash(p) for p in (root/"复赛_train").glob("*.csv")},
        "fold_hashes":{str(s):digest(f.tolist()) for s,f in folds.items()},"seeds":seeds,"candidates":candidates,
        "development":str(development) if development else None,
        "development_summary_sha256":file_hash(root/development/"summary.json") if development else None,
        "starting_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True,cwd=root).strip()}
    manifest["identity"]=digest(manifest)
    if out.exists():raise ValueError("Run exists; preserve evidence, no implicit restart")
    out.mkdir(parents=True,exist_ok=False);write_new(out/"manifest.json",manifest)
    references(root,out,frame,folds,spec,manifest,bool(development))
    phases=[[(s,f,t,"BASE",42) for s in seeds for f in range(5) for t in TARGETS],
            [(s,f,t,a,42) for s in seeds for f in range(5) for t,arms in candidates.items() for a in arms if a!="BASE"]]
    if not development:
        phases.append([(42,0,t,a,1042) for t in TARGETS for a in spec["recipes"]])
    for phase,tasks in enumerate(phases):
        failed=[]
        with ProcessPoolExecutor(max_workers=spec["budget"]["candidate_workers"]) as pool:
            jobs={pool.submit(fit_unit,root,out,frame,folds[s],s,f,t,a,spec,manifest,init):(s,f,t,a,init) for s,f,t,a,init in tasks}
            for job in as_completed(jobs):
                identity=jobs[job]
                try:
                    result=job.result();append_event(out/"fit_ledger.jsonl",{"event":"complete",**result})
                except Exception as exc:
                    failed.append(identity);append_event(out/"fit_ledger.jsonl",{"event":"failed","unit":identity,"error":repr(exc)})
        if failed:raise RuntimeError(f"Failed phase{phase} preserved: {failed}")
        print(json.dumps({"event":"phase_complete","phase":phase,"units":len(tasks)}),flush=True)
    verify_hashes(root,manifest["source_hashes"])
    result=summarize(out,frame,folds,spec,candidates)
    if old:
        result["four_seed_decisions"]=[]
        for target,arm in old["selected_for_confirmation"].items():
            if arm is None:continue
            previous=next(r for r in old["records"] if (r["target"],r["arm"])==(target,arm))
            current=next(r for r in result["records"] if (r["target"],r["arm"])==(target,arm))
            rows={**previous["seeds"],**current["seeds"]};paired=paired_summary([v["gain"] for v in rows.values()])
            dev_score=float(np.mean([v["candidate_score"] for v in previous["seeds"].values()]))
            failed=[]
            if paired["positive"]!=4:failed.append("not_all_four_seeds_positive")
            if paired["lcb95"]<=0:failed.append("nonpositive_seed_lcb95")
            if dev_score<spec["promotion"]["local_working_gate"]:failed.append("local_working_gate")
            result["four_seed_decisions"].append({"target":target,"arm":arm,"paired":paired,"seeds":rows,
                "development_score":dev_score,"failed_conditions":failed,"promoted":not failed,"release_authorized":False})
    else:
        diagnostic=[]
        base,members=collect(out,frame,folds,candidates)
        for target in TARGETS:
            mask=folds[42]==0;y=frame.loc[mask,target].to_numpy()
            for arm in spec["recipes"]:
                with np.load(out/f"{target}-{arm}-s42-f0-init1042/predictions.npz") as saved:p=saved["prediction"][:,0]
                original=members[42,target,arm][mask]
                ref=base[42][target][mask];old_key="v12_iron" if target=="tap_iron" else "v7_time"
                old_component=base[42][old_key][mask]
                before=replacement(ref,old_component,original);after=replacement(ref,old_component,p)
                diagnostic.append({"target":target,"arm":arm,"mean_absolute_prediction_change":float(np.abs(p-original).mean()),
                    "standalone_score_change":float(50*(np.abs(y-original).sum()-np.abs(y-p).sum())/np.abs(y).sum()),
                    "fixed_replacement_score_change":float(50*(np.abs(y-before).sum()-np.abs(y-after).sum())/np.abs(y).sum()),
                    "scope":"single_prespecified_fold_descriptive_only"})
        result["initialization_diagnostics"]=diagnostic
    write_new(out/"summary.json",result)
    print(json.dumps({"status":"complete","selected":result["selected_for_confirmation"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--development",type=Path)
    a=p.parse_args();run(Path.cwd(),a.output,a.development)
