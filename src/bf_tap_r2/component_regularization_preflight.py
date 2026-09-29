"""Full-shape synthetic admission and independent fresh-process inference."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import yaml

from .component_regularization import ComponentRegressor
from .component_regularization_audit import verify_saved
from .component_regularization_run import SPEC, RUN_ROOT, RECIPE, sources, outputs
from .data import FEATURES, TARGETS
from .v12_joint import JointRegressor
from .v7_periodic import PeriodicRegressor, file_hash, write_new
from .v49_run import check_runtime, verify_reference_cache, append_event


def synthetic():
    rng=np.random.default_rng(52001);x=rng.normal(size=(2754,len(FEATURES)))
    z=3*x[:,0]+2*np.sin(x[:,1])+x[:,2]*x[:,3]+.2*rng.normal(size=len(x))
    frame=pd.DataFrame(x,columns=FEATURES)
    frame["sample_id"]=[f"synthetic-regularization-{i}" for i in range(len(x))]
    frame["spout_no"]=rng.integers(1,3,len(x))
    frame["tap_iron"]=500+10*z;frame["tap_time_len"]=100+2*z
    return frame.iloc[:2204].reset_index(drop=True),frame.iloc[2204:].reset_index(drop=True)


def fit(out,target,arm,spec):
    directory=out/f"{target}-{arm}";directory.mkdir(exist_ok=False)
    training,query=synthetic();y=training[outputs(target)].to_numpy();settings=spec["training"][target]
    start=time.monotonic()
    m=ComponentRegressor(RECIPE,settings,arm,spec["mechanisms"],directory)
    m._initialize(training,y);m._train(training,y,settings["max_epochs"])
    pred=m.predict(query.drop(columns=list(TARGETS)))
    seconds=time.monotonic()-start
    with (directory/"prediction.npy").open("xb") as stream:np.save(stream,pred,allow_pickle=False)
    yq=query[outputs(target)].to_numpy();mae=np.abs(yq-pred).mean(0)
    constant=np.abs(yq-np.median(y,axis=0)).mean(0)
    row={"target":target,"arm":arm,"seconds":seconds,"mae":mae.tolist(),"constant_mae":constant.tolist(),
         "peak_rss_mib":m.traces["refit"]["peak_rss_mib"],"updates":m.traces["refit"]["updates"],
         "gradient_evaluations":m.traces["refit"]["gradient_evaluations"]}
    if arm=="BASE":
        start=time.monotonic()
        native=JointRegressor(RECIPE,settings) if target=="tap_iron" else PeriodicRegressor(RECIPE,settings)
        yy=y if target=="tap_iron" else y[:,0]
        native._initialize(training,yy);native._train(training,yy,settings["max_epochs"])
        native_pred=native.predict(query.drop(columns=list(TARGETS)))
        if target!="tap_iron":native_pred=native_pred[:,None]
        np.testing.assert_array_equal(native_pred,pred)
        for k,v in native.model_.state_dict().items():
            import torch
            if not torch.equal(v,m.model_.state_dict()[k]):raise ValueError("Full-shape native trajectory mismatch")
        row["native_replay_difference"]=0.;row["native_control_seconds"]=time.monotonic()-start
    write_new(directory/"result.json",row)
    return row


def cold(root,out):
    spec=yaml.safe_load((root/SPEC).read_text());training,query=synthetic();rows=[]
    for target in TARGETS:
        for arm in spec["recipes"]:
            d=out/f"{target}-{arm}"
            m=verify_saved(d/"refit.pt",training,training[outputs(target)].to_numpy(),arm,spec["training"][target],spec["mechanisms"],expected_epoch=240)
            expected=np.load(d/"prediction.npy",allow_pickle=False)
            query_clean=query.drop(columns=list(TARGETS))
            np.testing.assert_array_equal(m.predict(query_clean),expected)
            variants=[m.predict(query_clean.iloc[::-1])[::-1],np.concatenate([m.predict(query_clean.iloc[i:i+37]) for i in range(0,len(query_clean),37)])]
            diff=max(float(np.max(np.abs(v-expected))) for v in variants)
            if diff>spec["preflight"]["cold_predict_atol"]:raise ValueError("Synthetic cold/chunk mismatch")
            rows.append({"target":target,"arm":arm,"full_batch_difference":0.,"order_chunk_difference":diff})
    print(json.dumps(rows))


def run(root):
    spec=yaml.safe_load((root/SPEC).read_text());check_runtime(spec);verify_reference_cache(root,spec)
    out=root/RUN_ROOT/"preflight-r1";out.mkdir(parents=True,exist_ok=False)
    frozen=sources(root);write_new(out/"manifest.json",{"spec_sha256":file_hash(root/SPEC),"source_hashes":frozen})
    rows=[]
    with ProcessPoolExecutor(max_workers=spec["budget"]["candidate_workers"]) as pool:
        jobs={pool.submit(fit,out,t,a,spec):(t,a) for t in TARGETS for a in spec["recipes"]}
        for job in as_completed(jobs):
            try:row=job.result();rows.append(row);append_event(out/"events.jsonl",{"event":"complete",**row})
            except BaseException as exc:
                append_event(out/"events.jsonl",{"event":"failed","unit":jobs[job],"error":repr(exc)});raise
    completed=subprocess.run([sys.executable,"-m","bf_tap_r2.component_regularization_preflight","--cold",str(out)],cwd=root,check=True,text=True,capture_output=True)
    differences=json.loads(completed.stdout)
    peak=max(r["peak_rss_mib"] for r in rows)
    projection=2*sum(r["seconds"] for r in rows)*22/spec["budget"]["candidate_workers"]/3600
    checks={"learnability":all(all(a<b for a,b in zip(r["mae"],r["constant_mae"])) for r in rows),
            "native_replay":all(r.get("native_replay_difference",0.)==0 for r in rows),
            "runtime":projection<=spec["preflight"]["max_projected_hours"],
            "worker_memory":peak<=spec["preflight"]["max_worker_rss_mib"],
            "available_memory":4*peak+1024<psutil.virtual_memory().available/1024**2,
            "sources_unchanged":sources(root)==frozen}
    report={"status":"passed" if all(checks.values()) else "failed","checks":checks,"results":rows,
            "projected_hours":projection,"cold_differences":differences,"source_hashes":frozen,
            "spec_sha256":file_hash(root/SPEC),"synthetic_new_refits":6,"synthetic_native_controls":2,"official_fits":0}
    write_new(out/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="source_hashes"}),flush=True)
    if report["status"]!="passed":raise ValueError("Synthetic admission failed")


if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--cold":cold(Path.cwd(),Path(sys.argv[2]))
    else:run(Path.cwd())
