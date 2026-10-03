"""Exclusive native constant-LR head bootstrap MSE experiment experiment. No fullfit or package entrypoint."""
from pathlib import Path
import argparse,importlib.metadata,importlib.util,json,os,resource,shutil,subprocess,sys,time,traceback
import numpy as np
import pandas as pd
import yaml
from .data import FEATURES
from .fixed_bootstrap_protocol import BUDGETS,CANDIDATE,CONTROL,SEEDS,Ledger,sha,digest,write_new,verify_tree,decide,select_target
from .tabm_swa_run import runtime,assert_threads

RUN_REL="local/runs/tabm-fixed-bootstrap-v1"
PRIVATE_REL="local/fixed-bootstrap-20261003"
SPEC_REL="configs/tabm_fixed_bootstrap_v1/SPEC.json"
BRANCH="codex/tabm-fixed-bootstrap-v1"
ORIGINAL="local/runs/tabm-time-swa-v1"
RAW_BASE="local/runs/tabm-head-bootstrap-v1"
CONFIRM="local/runs/tabm-head-bootstrap-v1"
BUNDLE="local/incoming/swa-confirmation-271828-314159-20261002-r1"
def original(root,phase):
    return Path(root)/(CONFIRM if phase=="confirmation" else RAW_BASE)/(phase+"-r1")
def sources(root):
    paths=list((root/"src/bf_tap_r2").glob("*.py"))+list((root/"tests").glob("*.py"))
    paths += [root/SPEC_REL,root/"configs/round2_v12/SPEC.yaml",root/"configs/protection.yaml",root/"uv.lock",root/"pyproject.toml",root/"docs/tabm_fixed_bootstrap_v1/STRATEGY.md"]
    paths += [root/PRIVATE_REL/p for p in ["start_once.py","monitor_once.py","launch_start_once.py","supervise_controller_once_r1.py","verify_formal_launch_once_r1.py","verify_final_once_r1.py"] if (root/PRIVATE_REL/p).exists()]
    return {str(p.relative_to(root)):sha(p) for p in paths}
def branch(root):
    if subprocess.check_output(["git","branch","--show-current"],cwd=root,text=True).strip()!=BRANCH:
        raise ValueError("Branch differs; no automatic switch")
def check(root,frozen):
    branch(root);verify_tree(root,frozen["source_hashes"]);verify_tree(root,frozen["reference_files"])
    if runtime()!=frozen["runtime"]:raise ValueError("Frozen runtime changed")
def ledger(root):
    return Ledger(Path(root)/RUN_REL/"ledger",BUDGETS)
def active_freeze(root):return Path(root)/RUN_REL/"freeze.json"

def development_iron(root,seed,ids,folds):
    root=Path(root);receipt=json.loads((root/"local/next-direction-20261001/native-iron-reference-r1.json").read_text())
    col=receipt["columns"][str(seed)];path=root/f"local/next-direction-20261001/parent-iron-seed-{seed}.npy"
    if receipt["status"]!="passed_zero_fit_native_parent_iron_binding" or receipt["current_platform_representative"]!="EMA_TIME_Q75" or sha(path)!=col["file_sha256"] or digest(np.asarray(folds).tolist())!=col["fold_digest"] or len(ids)!=col["rows"]:
        raise ValueError("Native Q75 iron anchor or same-seed fold identity changed")
    with np.load(root/RAW_BASE/"development-r1/inputs.npz",allow_pickle=False) as data:
        np.testing.assert_array_equal(data["ids"],ids)
    iron=np.load(path,allow_pickle=False)
    if iron.shape!=np.asarray(ids).shape or not np.isfinite(iron).all() or (iron<0).any():raise ValueError("Invalid native Q75 iron")
    return iron.copy()

def append_access(root,freeze):
    p=Path(root)/RUN_REL/"ledger/label-access.jsonl";p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(dict(time=time.time(),event="before_official_round2_cached_label_read",freeze_sha256=sha(freeze),
                               protection_sha256=sha(Path(root)/"configs/protection.yaml"),protected_labels_requested=False))+"\n")
        f.flush();os.fsync(f.fileno())
def phase_data(root,phase):
    if phase!="engineering":append_access(root,active_freeze(root))
    with np.load(original(root,phase)/"inputs.npz",allow_pickle=False) as a:
        frame=pd.DataFrame(a["x"],columns=FEATURES);frame["spout_no"]=a["spouts"];frame["sample_id"]=a["ids"]
        y=a["y"].copy();units={}
        unit_names=["synthetic-pair"] if phase=="engineering" else [f"s{s}-f{f}" for s in ((42,3407) if phase=="development" else (271828,314159)) for f in range(5)]
        for unit in unit_names:units[unit]={name:a[name+"__"+unit].copy() for name in ("train","query","inner","cal")}
    return frame,y,units
def feature_frame(frame):
    return frame.drop(columns=[c for c in ["tap_iron","tap_time_len"] if c in frame])
def readonly_verifier(root):
    p=root/BUNDLE/"scripts/verify_swa_continuation_handoff.py"
    old=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:
        spec=importlib.util.spec_from_file_location("time_swa_verified_reference",p)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    finally:sys.dont_write_bytecode=old
    return m
def parents(root):
    m=readonly_verifier(root);v=m.verify_bundle(root/BUNDLE,"4f7f04e00f8f5d79dad26e3b66c5e5f6d5789c9993dca250ce6482393d015e9f")
    result={}
    for phase,seeds in [("development",(42,3407)),("confirmation",(271828,314159))]:
        frame,y,units=phase_data(root,phase);ids=frame.sample_id.astype(str).to_numpy();folds=np.full(len(ids),-1,int)
        if len(ids)!=2754 or len(set(ids))!=2754:raise ValueError("Incomplete native rows")
        for s in seeds:
            folds.fill(-1);count=np.zeros(len(ids),int)
            for f in range(5):q=units[f"s{s}-f{f}"]["query"];folds[q]=f;count[q]+=1
            if not np.all(count==1):raise ValueError("Incomplete fold coverage")
            p,fv=m.reindex_reference(v,s,ids,folds);np.testing.assert_array_equal(fv,folds)
            manifest=v["manifest"];binding=json.loads((root/BUNDLE/"refs/original-binding.json").read_text())
            source=binding["original_confirmation_source" if phase=="confirmation" else "original_development_source"]
            iron=np.full(len(ids),np.nan);coverage=np.zeros(len(ids),int)
            native_map={value:i for i,value in enumerate(ids)}
            for fold in range(5):
                name=("" if phase=="confirmation" else "SHORT_SPAN-")+f"s{s}-f{fold}"
                payload=manifest["original_path_to_payload"][source+"/"+name+"/predictions.npz"]
                with np.load(root/BUNDLE/payload,allow_pickle=False) as z:
                    qids=z["query_ids"].astype(str);ix=np.array([native_map[q] for q in qids]);col=z["iron" if phase=="confirmation" else "iron_reference"]
                    if not np.all(folds[ix]==fold) or len(set(ix))!=len(ix):raise ValueError("Iron reference same-seed fold differs")
                    if phase=="confirmation":np.testing.assert_array_equal(col,z["tap_iron"])
                    iron[ix]=col;coverage[ix]+=1
            if not np.all(coverage==1) or not np.isfinite(iron).all() or (iron<0).any():raise ValueError("Incomplete native Q75 iron reference")
            if phase=="development":
                # The shared iron_reference field is not the byte-anchored native
                # iron column used by Q75. Keep that auxiliary field untouched.
                iron=development_iron(root,s,ids,folds)
            else:
                # Cold-audited original reference recipe, unchanged Q75 iron.
                for fold in range(5):
                    payload=manifest["original_path_to_payload"][source+f"/s{s}-f{fold}/predictions.npz"]
                    with np.load(root/BUNDLE/payload,allow_pickle=False) as z:
                        np.testing.assert_array_equal(z["iron"],.5*z["v36_iron"]+.5*z["v12_iron"])
            result[s]=np.column_stack([iron,p])
    return result
def write_predictions(path,ids,cal_ids,control,candidate):
    ids=np.asarray(ids,dtype=str);cal_ids=np.asarray(cal_ids,dtype=str)
    control=np.asarray(control,dtype=float);candidate=np.asarray(candidate,dtype=float)
    if ids.ndim!=1 or cal_ids.ndim!=1 or control.shape!=(len(ids),2) or candidate.shape!=(len(ids),2) or not np.isfinite([control,candidate]).all():
        raise ValueError("Prediction column identity differs")
    if len(np.unique(ids))!=len(ids) or len(np.unique(cal_ids))!=len(cal_ids):
        raise ValueError("Duplicate prediction IDs")
    with Path(path).open("xb") as stream:
        np.savez(stream,ids=ids,cal_ids=cal_ids,**{CONTROL:control,CANDIDATE:candidate})

def train_unit(root,phase,unit):
    from .fixed_bootstrap_model import FixedBootstrapRegressor
    run=root/RUN_REL;freeze_path=active_freeze(root);frozen=json.loads(freeze_path.read_text());check(root,frozen)
    frame,y,units=phase_data(root,phase);ix=units[unit]
    directory=run/(phase+"-r1")/unit;directory.mkdir(exist_ok=False)
    # Cached original partitions must agree with the actual fitter's inner grouping.
    from .v3_4_bags import group_safe_inner_folds
    iv=np.asarray(group_safe_inner_folds(frame.iloc[ix["train"]].reset_index(drop=True),seed=42)["fold"])
    np.testing.assert_array_equal(ix["train"][iv!=0],ix["inner"]);np.testing.assert_array_equal(ix["train"][iv==0],ix["cal"])
    groups=pd.util.hash_pandas_object(frame[list(FEATURES)],index=False).to_numpy()
    if set(groups[ix["train"]])&set(groups[ix["query"]]) or set(groups[ix["inner"]])&set(groups[ix["cal"]]):
        raise ValueError("Duplicate partition leakage")
    source=original(root,phase)/unit
    original_complete=json.loads((source/"complete.json").read_text());verify_tree(source,original_complete["hashes"])
    with np.load(source/"partitions.npz",allow_pickle=False) as p:
        for k in ix:np.testing.assert_array_equal(p[k],ix[k])
    control_source=source/"HEAD_BOOTSTRAP"
    from .component_regularization import ComponentRegressor
    cached=ComponentRegressor.load(control_source/"refit.pt")
    if cached.saved["mechanisms"]!={"bootstrap_rate":1.0,"bootstrap_seed_offset":1000003,"loss":"fixed_denominator_per_row_head_poisson_mse"}:raise ValueError("Raw BASE mechanism differs")
    control=cached.predict(feature_frame(frame.iloc[ix["query"]]));del cached
    write_new(directory/"control-reuse.json",dict(source_directory=str(control_source.relative_to(root)),
                     original_complete_sha256=sha(source/"complete.json"),original_inputs_sha256=sha(original(root,phase)/"inputs.npz"),
                     original_audit_sha256=sha(original(root,phase)/"audit.json"),
                     model_hashes={name:sha(control_source/name) for name in ["selection.pt","refit.pt"]},
                     new_control_fits=0,original_identity={"source_directory":str(control_source.relative_to(root)),"phase":phase,"unit":unit}))
    settings=yaml.safe_load((root/"configs/round2_v12/SPEC.yaml").read_text())["training"]
    target=directory/CANDIDATE;target.mkdir()
    model=FixedBootstrapRegressor(settings,target,ledger(root),phase,unit).fit(frame.iloc[ix["train"]].reset_index(drop=True),y[ix["train"]])
    predicted=model.predict(feature_frame(frame.iloc[ix["query"]]));write_new(target/"metadata.json",model.metadata_);del model
    with (directory/"partitions.npz").open("xb") as stream:np.savez(stream,**ix)
    write_predictions(directory/"predictions.npz",frame.iloc[ix["query"]].sample_id.astype(str).to_numpy(),frame.iloc[ix["cal"]].sample_id.astype(str).to_numpy(),control,predicted)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if rss>1024:raise ValueError("Worker RSS exceeded")
    write_new(directory/"complete.json",dict(phase=phase,unit=unit,peak_rss_mib=rss,
                           hashes={str(p.relative_to(directory)):sha(p) for p in directory.rglob("*") if p.is_file()}))
    check(root,frozen)
def run_phase(root,phase,frozen):
    run=root/RUN_REL;directory=run/(phase+"-r1");directory.mkdir(exist_ok=False)
    if any(ledger(root).counts(phase)["reserved"].values()):raise ValueError("Consumed phase cannot retry")
    frame,y,units=phase_data(root,phase)
    shutil.copyfile(original(root,phase)/"inputs.npz",directory/"inputs.npz")
    settings=yaml.safe_load((root/"configs/round2_v12/SPEC.yaml").read_text())["training"]
    write_new(directory/"manifest.json",dict(phase=phase,units=list(units),input_sha256=sha(directory/"inputs.npz"),
                        source_hashes=frozen["source_hashes"],settings=settings,spec=json.loads((root/SPEC_REL).read_text()),
                        candidate=CANDIDATE,control=CONTROL,control_fits=0,budgets=BUDGETS[phase]))
    write_new(directory/"started.json",dict(time=time.time(),pid=os.getpid()))
    for unit in units:
        check(root,frozen)
        with (directory/f"{unit}-worker.log").open("x") as log:
            subprocess.run([sys.executable,"-u","-m","bf_tap_r2.fixed_bootstrap_run","worker","--phase",phase,"--unit",unit],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(phase,unit,"complete",flush=True)
    check(root,frozen)
    with (directory/"cold-audit.log").open("x") as log:
        subprocess.run([sys.executable,"-m","bf_tap_r2.fixed_bootstrap_audit",str(directory)],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    check(root,frozen);write_new(directory/"finished.json",dict(status="passed_complete_cold_audit",counts=ledger(root).counts(phase),audit_sha256=sha(directory/"audit.json")))
    return directory
def evaluate(root,phase,parent):
    from scipy.stats import t
    import math
    run=root/RUN_REL;d=run/(phase+"-r1");frame,y,units=phase_data(root,phase)
    audit=json.loads((d/"audit.json").read_text())
    if audit["status"]!="passed" or audit["input_sha256"]!=sha(d/"inputs.npz"):raise ValueError("Missing complete cold audit")
    seeds=(42,3407) if phase=="development" else (271828,314159);rows={};vectors={};decisions={};independent={}
    for s in seeds:
        columns={arm:np.full(y.shape,np.nan) for arm in (CANDIDATE,CONTROL)};coverage=np.zeros(len(frame),int)
        for f in range(5):
            u=f"s{s}-f{f}";query=units[u]["query"]
            if sha(d/u/"complete.json")!=audit["units_sha256"][u]:raise ValueError("Audit unit changed")
            with np.load(d/u/"predictions.npz",allow_pickle=False) as a:
                np.testing.assert_array_equal(a["ids"],frame.iloc[query].sample_id.astype(str).to_numpy())
                for arm in columns:columns[arm][query]=a[arm]
            coverage[query]+=1
        if not np.all(coverage==1):raise ValueError("Incomplete OOF")
        rows[str(s)]={}
        for j,target_name in enumerate(("tap_iron","tap_time_len")):
            target=y[:,j];rows[str(s)][target_name]={}
            for arm,member in columns.items():
                p=.8*parent[s][:,j]+.2*member[:,j]
                if not np.isfinite(p).all() or (p<0).any():raise ValueError("Invalid endpoint; no clipping")
                g=float(50*(np.abs(target-parent[s][:,j]).sum()-np.abs(target-p).sum())/target.sum())
                rows[str(s)][target_name][arm]=dict(gain=g,endpoint_wmape=float(np.abs(target-p).sum()/target.sum()),member_wmape=float(np.abs(target-member[:,j]).sum()/target.sum()))
                with (d/f"endpoint-{target_name}-{arm}-s{s}.npy").open("xb") as stream:np.save(stream,p,allow_pickle=False)
    old=json.loads((run/"development-r1/evaluation.json").read_text()) if phase=="confirmation" else None
    for target_name in ("tap_iron","tap_time_len"):
        gains=[rows[str(s)][target_name][CANDIDATE]["gain"] for s in seeds];controls=[rows[str(s)][target_name][CONTROL]["gain"] for s in seeds]
        vectors[target_name]=dict(seed_gains=gains,control_gains=controls)
        ag=(old["vectors"][target_name]["seed_gains"]+gains) if old else gains
        ac=(old["vectors"][target_name]["control_gains"]+controls) if old else controls
        dec=decide(ag,ac,phase=="confirmation");decisions[target_name]=dec
        n=len(ag);mean=math.fsum(ag)/n;mech=math.fsum(a-b for a,b in zip(ag,ac))/n
        lcb=mean-float(t.ppf(.95,n-1))*math.sqrt(math.fsum((a-mean)**2 for a in ag)/(n-1))/math.sqrt(n) if n==4 else None
        passed=all(a>0 for a in ag) and mech>0 and (n==2 or lcb>0)
        if abs(mean-dec["mean_gain"])>1e-12 or abs(mech-dec["mechanism_mean"])>1e-12 or bool(dec["formal_promoted"])!=(n==4 and passed) or bool(dec["selected_for_confirmation"])!=(n==2 and passed):raise ValueError("Independent decision differs")
        if n==4 and abs(lcb-dec["seed_lcb95"])>1e-12:raise ValueError("Independent LCB differs")
        independent[target_name]=dict(mean_gain=mean,mechanism_mean=mech,seed_lcb95=lcb,passed=passed)
    chosen=old["selected_target"] if old else select_target(decisions)
    if old:
        for target_name,dec in decisions.items():
            if target_name!=chosen:
                dec["mathematical_four_seed_gate_passed_descriptive"]=dec["formal_promoted"]
                dec["formal_promoted"]=False
                dec["classification"]="descriptive_unselected_target_no_promotion"
    candidates=[t for t in ("tap_time_len",) if decisions[t]["mean_gain"]>0]
    explore=max(candidates,key=lambda t:decisions[t]["mean_gain"]) if candidates else None
    summary=dict(selected_target=chosen,selected_for_confirmation=[chosen] if chosen and not old else [],formal_promoted=bool(old and chosen and decisions[chosen]["formal_promoted"]),exploration_target=explore,targets=decisions,release_authorized=False)
    write_new(d/"evaluation.json",dict(rows=rows,vectors=vectors,selected_target=chosen,decision=summary,reference="EMA_TIME_Q75",current_platform_score_user_reported=96.392,complete_package_local_score_available=False))
    write_new(d/"independent-decision.json",dict(status="passed_scalar_seed_gate_recomputation",targets=independent,selected_target=chosen,new_fits=0))
    return summary

def admit(root):
    root=Path(root).resolve();run=root/RUN_REL
    if run.exists():raise ValueError("Existing run cannot restart")
    branch(root);spec=json.loads((root/SPEC_REL).read_text())
    if spec["identity"]!="TABM_FIXED_BOOTSTRAP_V1" or spec["loss"]!={"rate":1.0,"seed_offset":1000003,"per_row_head":True,"shared_two_target_weight":True,"denominator":"batch_rows*16*2","zero_batch":"no_resample_no_skip_AdamW_step","evaluation":"unweighted_head_mean","draw_policy":"once_per_actual_fit_row_head"} or spec["targets"]!=["tap_time_len"] or spec["budgets"]!=BUDGETS or spec["alpha"]!=.2 or spec["development_seeds"]!=[42,3407] or spec["confirmation_seeds"]!=[271828,314159] or spec["release_authorized"]:
        raise ValueError("Frozen specification differs")
    tests=json.loads((root/PRIVATE_REL/"tests-complete-r1.json").read_text())
    if tests["status"]!="passed" or tests["source_hashes"]!=sources(root):raise ValueError("Missing/stale full tests")
    verify_tree(root,tests["logs"])
    deps={}
    for phase in ("engineering","development","confirmation"):
        d=original(root,phase);a=json.loads((d/"audit.json").read_text())
        if a["status"]!="passed" or a.get("new_fits",0)!=0:raise ValueError("Original control audit missing")
        for p in d.rglob("*"):
            if p.is_file():deps[str(p.relative_to(root))]=sha(p)
    for old in (root/RAW_BASE,root/CONFIRM):
        f=json.loads((old/"preflight.json").read_text());verify_tree(root,f["source_hashes"]);verify_tree(root,f["reference_files"])
        deps.update(f["reference_files"])
        for name in ("preflight.json","freeze.json","ledger/fits.jsonl"):
            p=old/name
            if p.exists():deps[str(p.relative_to(root))]=sha(p)
    for p in (root/BUNDLE).rglob("*"):
        if p.is_file():deps[str(p.relative_to(root))]=sha(p)
    for name in ["local/next-direction-20261001/native-iron-reference-r1.json", "local/next-direction-20261001/parent-iron-seed-42.npy", "local/next-direction-20261001/parent-iron-seed-3407.npy"]:
        deps[name]=sha(root/name)
    frozen=dict(source_hashes=sources(root),reference_files=deps,runtime=runtime(),spec_sha256=sha(root/SPEC_REL),
                source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                tests_receipt_sha256=sha(root/PRIVATE_REL/"tests-complete-r1.json"),new_reference_fits=0)
    check(root,frozen);run.mkdir(parents=True,exist_ok=False);write_new(run/"freeze.json",frozen)
    append_access(root,run/"freeze.json");p=parents(root)
    write_new(run/"q75-binding.json",dict(candidate="EMA_TIME_Q75",verified_seeds=list(SEEDS),frozen_bundle_manifest_sha256=sha(root/BUNDLE/"HANDOFF_MANIFEST.json"),new_fits=0))
    g0=run_phase(root,"engineering",frozen)
    write_new(run/"preflight.json",dict(**frozen,G0="passed",engineering_audit_sha256=sha(g0/"audit.json"),budgets=BUDGETS))
    print("Production G0 passed; formal ready",flush=True)
def execute(root):
    root=Path(root).resolve();run=root/RUN_REL
    frozen=json.loads((run/"preflight.json").read_text());check(root,frozen)
    if frozen["G0"]!="passed" or frozen["engineering_audit_sha256"]!=sha(run/"engineering-r1/audit.json"):raise ValueError("G0 missing")
    write_new(run/"controller-started.json",dict(time=time.time(),pid=os.getpid(),preflight_sha256=sha(run/"preflight.json")))
    try:
        append_access(root,run/"preflight.json");p=parents(root)
        run_phase(root,"development",frozen);d=evaluate(root,"development",p)
        if d["selected_for_confirmation"]:
            run_phase(root,"confirmation",frozen);d=evaluate(root,"confirmation",p)
        check(root,frozen)
        write_new(run/"controller-finished.json",dict(status="completed",time=time.time(),decision=d,
                              counts={phase:ledger(root).counts(phase) for phase in BUDGETS},reference_fits=0,fullfit=0,packages=0,uploads=0))
    except BaseException as e:
        write_new(run/"controller-failed.json",dict(status="failed_preserved",time=time.time(),error=repr(e),traceback=traceback.format_exc(),
                              counts={phase:ledger(root).counts(phase) for phase in BUDGETS}));raise
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=["admit","execute","worker"]);parser.add_argument("--phase");parser.add_argument("--unit");args=parser.parse_args();assert_threads();root=Path.cwd().resolve()
    if args.action=="admit":admit(root)
    elif args.action=="execute":execute(root)
    else:train_unit(root,args.phase,args.unit)
