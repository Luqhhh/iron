"""One-shot SCARF development and conditional confirmation, never packaging/uploading."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
from .data import FEATURES
from .scarf_protocol import Ledger,BUDGETS,ARMS,sha,digest,write_new,verify_tree,decision,bootstrap

THREAD_KEYS=("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")

def runtime():
    versions={}
    for name in ("torch","numpy","pandas","scipy","scikit-learn","psutil","PyYAML"):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return dict(python=sys.version,packages=versions,threads={k:os.environ.get(k) for k in THREAD_KEYS})

def verify_frozen(root,frozen):
    verify_tree(root,frozen["source_hashes"]); verify_tree(root,frozen["reference"]["frozen_files"])
    if runtime()!=frozen["runtime"]: raise ValueError("Frozen runtime or thread setting changed")

def assert_threads():
    if any(os.environ.get(k)!="1" for k in THREAD_KEYS) or sys.version_info[:2]!=(3,12): raise ValueError("Python3.12 and four single-thread variables required")
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)

def source_hashes(root):
    paths=list((root/"src/bf_tap_r2").glob("*.py"))+list((root/"tests").glob("*.py"))
    paths+=[root/"configs/scarf_reg_v1/SPEC.json",root/"uv.lock",root/"pyproject.toml"]
    return {str(p.relative_to(root)):sha(p) for p in paths}

def feature_frame(frame): return frame.drop(columns=[c for c in ("tap_iron","tap_time_len") if c in frame]).copy()

def partitions(frame,folds,seeds):
    from .scarf_model import inner_partition
    units={}
    for s in seeds:
        for f in range(5):
            train=np.flatnonzero(folds[s]!=f); query=np.flatnonzero(folds[s]==f)
            a,b=inner_partition(feature_frame(frame.iloc[train]).reset_index(drop=True))
            units[f"s{s}-f{f}"]=(train,query,train[a],train[b])
    return units

def save_inputs(path,frame,units):
    arrays=dict(x=frame[list(FEATURES)].to_numpy(float),spouts=frame.spout_no.to_numpy(int),ids=frame.sample_id.astype(str).to_numpy(dtype=str),
        y=frame[["tap_iron","tap_time_len"]].to_numpy(float))
    for u,(train,query,inner,cal) in units.items():
        for name,a in zip(("train","query","inner","cal"),(train,query,inner,cal)): arrays[name+"__"+u]=a
    with Path(path).open("xb") as f: np.savez(f,**arrays)

def run_phase(root,runroot,phase,frame,units,frozen):
    from .scarf_model import fit_state,TARGETS
    verify_frozen(root,frozen); directory=runroot/(phase+"-r1")
    directory.mkdir(exist_ok=False)
    ledger=Ledger(runroot/"ledger")
    if any(ledger.counts(phase)["reserved"].values()): raise ValueError("Consumed phase cannot restart")
    write_new(directory/"started.json",dict(phase=phase,time=time.time(),pid=os.getpid(),preflight_sha256=sha(runroot/"preflight.json") if phase!="engineering" else None))
    save_inputs(directory/"inputs.npz",frame,units)
    write_new(directory/"manifest.json",dict(phase=phase,units=list(units),input_sha256=sha(directory/"inputs.npz"),source_hashes=frozen["source_hashes"],runtime=frozen["runtime"],budgets=BUDGETS[phase]))
    y=frame[list(TARGETS)].to_numpy(float); predictors=feature_frame(frame)
    for unit,(train,query,inner,cal) in units.items():
        verify_frozen(root,frozen); d=directory/unit; d.mkdir(exist_ok=False); preds={"ids":frame.iloc[query].sample_id.astype(str).to_numpy(dtype=str)}
        for arm in ARMS:
            selected,sm=fit_state(predictors.iloc[inner],y[inner],predictors.iloc[cal],y[cal],arm=arm,epochs=240,selector=True,
                path=d/(arm+"-selector.npz"),ledger=ledger,phase=phase,key=unit+":"+arm+":selector")
            refit,rm=fit_state(predictors.iloc[train],y[train],predictors.iloc[query],None,arm=arm,epochs=sm["selected_epoch"],selector=False,
                path=d/(arm+"-refit.npz"),ledger=ledger,phase=phase,key=unit+":"+arm+":refit")
            preds[arm+"-selector"]=selected;preds[arm+"-refit"]=refit
        with (d/"predictions.npz").open("xb") as f: np.savez(f,**preds)
        write_new(d/"complete.json",dict(time=time.time(),hashes={p.name:sha(p) for p in d.iterdir() if p.is_file()}))
        print(phase,unit,"complete",ledger.counts(phase),flush=True)
    verify_frozen(root,frozen)
    with (directory/"cold-audit.log").open("x") as out:
        subprocess.run([sys.executable,"-m","bf_tap_r2.scarf_audit",str(directory)],cwd=root,stdout=out,stderr=subprocess.STDOUT,check=True)
    verify_frozen(root,frozen)
    write_new(directory/"finished.json",dict(status="passed_complete_cold_audit",time=time.time(),counts=ledger.counts(phase),audit_sha256=sha(directory/"audit.json")))
    return directory

def independent_decision(gains,controls,confirmation):
    # Independent gate arithmetic, cross-checked against protocol decision.
    from scipy.stats import t as student
    n=4 if confirmation else 2; a=np.asarray(gains,float); b=np.asarray(controls,float)
    if len(a)!=n or len(b)!=n or not np.isfinite([a,b]).all(): raise ValueError("Independent gate lacks complete seeds")
    mean=float(sum(a)/n); mech=float(sum(a-b)/n)
    lcb=float(mean-student.ppf(.95,n-1)*np.sqrt(sum((a-mean)**2)/(n-1))/np.sqrt(n)) if confirmation else None
    accepted=all(v>0 for v in a) and mech>0 and (not confirmation or lcb>0)
    d=decision(a,b,confirmation=confirmation)
    if bool(d["formal_promoted"])!=(accepted and confirmation) or bool(d["selected_for_confirmation"])!=(accepted and not confirmation): raise ValueError("Independent gate disagrees")
    if confirmation and abs(d["seed_lcb95"]-lcb)>1e-12: raise ValueError("LCB arithmetic disagrees")
    d["independent_recalculation_passed"]=True
    return d

def evaluate(runroot,phase,frame,folds,parents,de3,seeds,development=None):
    directory=runroot/(phase+"-r1"); manifest=json.loads((directory/"manifest.json").read_text()); audit=json.loads((directory/"audit.json").read_text())
    if audit["status"]!="passed" or audit["inputs_sha256"]!=sha(directory/"inputs.npz") or manifest["input_sha256"]!=sha(directory/"inputs.npz"): raise ValueError("Audit/input mismatch")
    y=frame.tap_iron.to_numpy(float); rows={}; endpoints={}; gains=[]; controls=[]
    for s in seeds:
        columns={arm:np.full(len(frame),np.nan) for arm in ARMS}
        for f in range(5):
            u=f"s{s}-f{f}"; p=directory/u
            if sha(p/"complete.json")!=audit["units_sha256"][u]: raise ValueError("Unit changed after audit")
            complete=json.loads((p/"complete.json").read_text()); verify_tree(p,complete["hashes"])
            q=folds[s]==f
            with np.load(p/"predictions.npz",allow_pickle=False) as a:
                if not np.array_equal(a["ids"],frame.loc[q,"sample_id"].astype(str).to_numpy()): raise ValueError("OOF row mismatch")
                for arm in ARMS: columns[arm][q]=a[arm+"-refit"][:,0]
        rows[str(s)]={}; endpoints[s]={}
        for arm,v in columns.items():
            if not np.isfinite(v).all(): raise ValueError("Incomplete OOF coverage")
            endpoint=.8*parents[s]+.2*v
            if not np.isfinite(endpoint).all() or (endpoint<0).any(): raise ValueError("Endpoint invalid; no post hoc clipping")
            error=np.abs(y-endpoint); gain=float(50*(np.abs(y-parents[s]).sum()-error.sum())/y.sum())
            endpoints[s][arm]=endpoint
            fold_gains=[float(50*(np.abs(y[q]-parents[s][q]).sum()-error[q].sum())/y[q].sum()) for q in (folds[s]==f for f in range(5))]
            rows[str(s)][arm]=dict(current_Q75_iron_gain=gain,historical_B0_iron_gain=gain,
                DE3_iron_gain=float(50*(np.abs(y-de3[s]).sum()-error.sum())/y.sum()),member_wmape=float(np.abs(y-v).sum()/y.sum()),
                endpoint_wmape=float(error.sum()/y.sum()),fold_gains_descriptive=fold_gains)
            with (directory/f"endpoint-{arm}-s{s}.npy").open("xb") as out: np.save(out,endpoint,allow_pickle=False)
        gains.append(rows[str(s)]["SCARF_REG"]["current_Q75_iron_gain"]);controls.append(rows[str(s)]["SUPERVISED_CONTROL"]["current_Q75_iron_gain"])
    all_gains=gains;all_controls=controls
    if development:
        dev=json.loads((development/"evaluation.json").read_text())
        all_gains=dev["seed_gains"]+gains;all_controls=dev["control_gains"]+controls
    d=independent_decision(all_gains,all_controls,phase=="confirmation")
    write_new(directory/"evaluation.json",dict(rows=rows,seed_gains=gains,control_gains=controls,decision=d,current_platform_score_user_reported=96.392,
        complete_Q75_package_local_score_available=False,reference_fits=0,
        bootstrap=bootstrap(y,[parents[s] for s in seeds],[endpoints[s]["SCARF_REG"] for s in seeds],frame.spout_no.to_numpy()),
        endpoint_hashes={p.name:sha(p) for p in directory.glob("endpoint-*.npy")}))
    write_new(directory/"independent-decision.json",d)
    return d

def g0_frame():
    r=np.random.default_rng(61001); n=2754; x=r.normal(size=(n,21)); sp=np.arange(n)%2+1
    # Include true duplicate groups to exercise the production mask and split logic.
    for i in range(0,200,2): x[i+1]=x[i]
    frame=pd.DataFrame(x,columns=FEATURES);frame["spout_no"]=sp;frame["sample_id"]=[f"synthetic-{i:05}" for i in range(n)]
    frame["tap_iron"]=200+20*(x[:,0]+.25*x[:,1]**2+.15*x[:,2]*x[:,3])
    frame["tap_time_len"]=50+5*(x[:,4]+.3*x[:,5]**2)
    from .splits import make_folds
    fv=make_folds(frame,42).set_index("sample_id").loc[frame.sample_id,"fold"].to_numpy()
    return frame,partitions(frame,{42:fv},[42])["s42-f0"]

def preflight(root,runroot):
    from .scarf_reference import load
    runroot.mkdir(exist_ok=False)
    spec=json.loads((root/"configs/scarf_reg_v1/SPEC.json").read_text())
    if spec["budgets"]!=BUDGETS or spec["release_authorized"] is not False: raise ValueError("Frozen spec changed")
    checks=root/"local/scarf-reg-20261001/tests-complete.json"; test=json.loads(checks.read_text())
    if test["status"]!="passed" or test["source_hashes"]!=source_hashes(root): raise ValueError("Required complete tests missing or stale")
    for path,h in test["logs"].items():
        if sha(root/path)!=h: raise ValueError("Test evidence changed")
    frame,folds,parents,de3,ref=load(root,runroot/"ledger")
    frozen=dict(source_hashes=source_hashes(root),runtime=runtime(),reference=ref,tests_complete_sha256=sha(checks),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),spec_sha256=sha(root/"configs/scarf_reg_v1/SPEC.json"))
    write_new(runroot/"freeze.json",frozen)
    syn,p=g0_frame(); g0=run_phase(root,runroot,"engineering",syn,{"synthetic-pair":p},frozen)
    write_new(runroot/"preflight.json",dict(**frozen,G0="passed",engineering_audit_sha256=sha(g0/"audit.json"),
        engineering_finished_sha256=sha(g0/"finished.json"),engineering_inputs_sha256=sha(g0/"inputs.npz"),new_G0_optimizer_runs=6,
        previous_research_optimizer_runs=6,development_optimizer_budget=60,confirmation_optimizer_budget=60,official_fits_started=0,reference_fits=0))
    print("G0 passed, formal preflight frozen",flush=True)

def execute(root,runroot):
    from .scarf_reference import load
    frozen=json.loads((runroot/"preflight.json").read_text());verify_frozen(root,frozen)
    for name,key in (("audit.json","engineering_audit_sha256"),("finished.json","engineering_finished_sha256"),("inputs.npz","engineering_inputs_sha256")):
        if sha(runroot/"engineering-r1"/name)!=frozen[key]: raise ValueError("G0 changed")
    write_new(runroot/"controller-started.json",dict(time=time.time(),pid=os.getpid(),preflight_sha256=sha(runroot/"preflight.json")))
    try:
        frame,folds,parents,de3,ref=load(root,runroot/"ledger")
        if ref!=frozen["reference"]: raise ValueError("Reference identity changed")
        dev=run_phase(root,runroot,"development",frame,partitions(frame,folds,[42,3407]),frozen)
        d=evaluate(runroot,"development",frame,folds,parents,de3,[42,3407])
        if d["selected_for_confirmation"]:
            run_phase(root,runroot,"confirmation",frame,partitions(frame,folds,[7777,12011]),frozen)
            d=evaluate(runroot,"confirmation",frame,folds,parents,de3,[7777,12011],development=dev)
        verify_frozen(root,frozen)
        write_new(runroot/"controller-finished.json",dict(status="completed",time=time.time(),decision=d,
            phase_counts={p:Ledger(runroot/"ledger").counts(p) for p in BUDGETS},reference_fits=0,packages=0,uploads=0))
    except BaseException as e:
        write_new(runroot/"controller-failed.json",dict(status="failed_no_retry",time=time.time(),error=repr(e),traceback=traceback.format_exc(),
            phase_counts={p:Ledger(runroot/"ledger").counts(p) for p in BUDGETS}));raise

def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["preflight","execute"]);args=p.parse_args()
    assert_threads();root=Path.cwd().resolve();runroot=root/"local/runs/scarf-reg-v1"
    if args.action=="preflight": preflight(root,runroot)
    else: execute(root,runroot)

if __name__=="__main__": main()
