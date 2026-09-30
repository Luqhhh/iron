"""One full-size synthetic worst-case probe; no official labels or reference fits."""
from pathlib import Path
import argparse
import json
import os
import resource
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import psutil
from .data import FEATURES
from .t2g_model import canonical,file_hash
from .t2g_freeze import ROOT,verify_manifest,write_new

ARMS=("LEARNED_GRAPH","DENSE_CONTROL")

def synthetic_data():
    rng=np.random.default_rng(57001)
    x=rng.standard_normal((2755,21))
    epsilon=rng.standard_normal(2755)
    f=pd.DataFrame(x,columns=FEATURES)
    f["spout_no"]=np.arange(2755)%2+1
    f["sample_id"]=[f"SYNTH_T2G_{i:05}" for i in range(2755)]
    f["tap_time_len"]=10+2*np.sin(x[:,0])+x[:,1]*x[:,2]+.5*x[:,3]+.1*epsilon
    return f

def resource_admission(graph_seconds,dense_seconds,graph_mib,dense_mib,available_mib):
    values=(graph_seconds,dense_seconds,graph_mib,dense_mib,available_mib)
    if any(not np.isfinite(v) or v<=0 for v in values): raise ValueError("Finite positive resource witnesses required")
    peak=max(graph_mib,dense_mib); projected=20*(graph_seconds+dense_seconds)/4*1.5+300
    required=4*peak+1024
    passed=projected<=7200 and peak<=1536 and available_mib>=required
    return {"status":"passed" if passed else "failed","projected_seconds":projected,
        "maximum_worker_rss_mib":peak,"required_available_mib":required,"available_mib":available_mib}

def learning_check(y_train,y_query,prediction):
    train,query,pred=(np.asarray(v,float) for v in (y_train,y_query,prediction))
    if (train.ndim!=1 or not len(train) or query.shape!=pred.shape or not len(query)
        or not all(np.isfinite(v).all() for v in (train,query,pred))):
        raise ValueError("Invalid synthetic learning witness")
    mae=float(np.abs(pred-query).mean()); constant=float(np.abs(np.median(train)-query).mean())
    return {"status":"passed" if mae<constant else "failed","query_mae":mae,"median_constant_mae":constant}

def reserve_probe(root,manifest_digest):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    write_new(root/"probe.started.json",{"engineering_sha256":manifest_digest,
        "planned_paths":2,"planned_optimizer_starts":4,"retry_authorized":False})

def probe_arm(manifest,output,arm):
    from .t2g_model import fit_partition,load_state,predict_loaded
    from .t2g_audit import compare_predictions,functional_predict
    manifest=Path(manifest); verify_manifest(manifest)
    if arm not in ARMS: raise ValueError("Unknown synthetic arm")
    output=Path(output).resolve()
    if output!=manifest.parent/"g0-r1"/arm: raise ValueError("Frozen synthetic output differs")
    reservation=json.loads((ROOT/"local/t2g-graph-v1/probe.started.json").read_text())
    if reservation["engineering_sha256"]!=file_hash(manifest): raise ValueError("Synthetic reservation differs")
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    f=synthetic_data();train=f.iloc[:2204].reset_index(drop=True)
    query=f.iloc[2204:].drop(columns="tap_time_len").reset_index(drop=True)
    def before(stage):
        write_new(output/(stage+".started.json"),{"arm":arm,"stage":stage,"engineering_sha256":file_hash(manifest)})
    r=fit_partition(train,query,"tap_time_len",arm,output/"model",before,force_epochs=240)
    model,prep,meta=load_state(r.model_dir)
    if meta["selector_rows"]!=1763 or meta["refit_epochs"]!=240 or len(meta["trace"])!=240:
        raise ValueError("Worst-case synthetic resource path incomplete")
    cold=predict_loaded(model,prep,meta,query);compare_predictions(r.prediction,cold,exact=True)
    comparisons=[]
    for v in (predict_loaded(model,prep,meta,query.iloc[::-1])[::-1],
        np.concatenate([predict_loaded(model,prep,meta,query.iloc[i:i+37]) for i in range(0,len(query),37)]),
        np.concatenate([predict_loaded(model,prep,meta,query.iloc[i:i+1]) for i in range(len(query))]),
        functional_predict(model.state_dict(),meta["preprocessor"],meta["target_mean"],meta["target_std"],query,arm)):
        comparisons.append(compare_predictions(cold,v))
    learning=learning_check(train.tap_time_len.to_numpy(),f.tap_time_len.iloc[2204:].to_numpy(),cold)
    report={"arm":arm,"optimizer_starts":2,"outer_paths":1,"worker_seconds":time.monotonic()-started,
        "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "learning":learning,"cold_full_difference":0.,"cold_variants":comparisons,
        "init_digest":r.init_digest,"batch_digest":r.batch_digest,"engineering_sha256":file_hash(manifest)}
    write_new(output/"worker.complete.json",report)
    return report

def run_preflight(manifest,output):
    manifest=Path(manifest).resolve();m=verify_manifest(manifest)
    if m["kind"]!="t2g-engineering-v1": raise ValueError("Synthetic probe requires engineering manifest")
    output=Path(output).resolve()
    if output!=manifest.parent/"g0-r1": raise ValueError("One canonical synthetic probe required")
    for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        if os.environ.get(n)!="1": raise ValueError("Set thread environment to1")
    reserve_probe(ROOT/"local/t2g-graph-v1",file_hash(manifest))
    output.mkdir(parents=True,exist_ok=False)
    reports={}
    try:
        for arm in ARMS:
            log=output/(arm+".log")
            started=time.monotonic()
            with log.open("x") as stream:
                subprocess.run([sys.executable,"-m","bf_tap_r2.t2g_preflight","--probe-arm",arm,
                    "--manifest",str(manifest),"--output",str(output/arm)],check=True,timeout=920,
                    stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,env=os.environ.copy())
            report=json.loads((output/arm/"worker.complete.json").read_text())
            report["complete_path_seconds"]=time.monotonic()-started
            reports[arm]=report
        graph,dense=reports["LEARNED_GRAPH"],reports["DENSE_CONTROL"]
        if graph["init_digest"]!=dense["init_digest"] or graph["batch_digest"]!=dense["batch_digest"]:
            raise ValueError("Synthetic paired initialization/batches differ")
        admission=resource_admission(graph["complete_path_seconds"],dense["complete_path_seconds"],
            graph["peak_rss_mib"],dense["peak_rss_mib"],psutil.virtual_memory().available/2**20)
        status="passed" if admission["status"]=="passed" and all(r["learning"]["status"]=="passed" for r in reports.values()) else "failed"
        result={"status":status,"engineering_sha256":file_hash(manifest),"outer_paths":2,
            "optimizer_starts":4,"arms":reports,"resource":admission,"official_fits":0}
        write_new(output/"preflight.json",result)
        write_new(output/"preflight.complete.json",{"sha256":file_hash(output/"preflight.json")})
        return result
    except BaseException as e:
        counts=sum((output/arm/(stage+".started.json")).exists() for arm in ARMS for stage in ("selector","refit"))
        write_new(output/"preflight.failed.json",{"status":"failed","type":type(e).__name__,
            "message":str(e),"optimizer_starts":counts,"official_fits":0,"retry_authorized":False})
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--probe-arm",choices=ARMS)
    a=p.parse_args()
    print(canonical(probe_arm(a.manifest,a.output,a.probe_arm) if a.probe_arm else run_preflight(a.manifest,a.output)))

if __name__=="__main__": main()
