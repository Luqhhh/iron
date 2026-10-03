"""Immutable zero-fit correction/fusion replay and separate scalar audit."""
from pathlib import Path
import argparse, importlib.util, json, math, os, resource, sys, time
import numpy as np
from scipy.stats import t
from bf_tap_r2.tabm_swa_protocol import sha, verify_tree, write_new
from bf_tap_r2.tabm_correction_fusion import affine, assemble, choose

SPEC="configs/tabm_correction_fusion_v1/SPEC.json"
OUT="local/runs/tabm-correction-fusion-v1"
BUNDLE="local/incoming/swa-confirmation-271828-314159-20261002-r1"
SEEDS=(42,3407,271828,314159)
METHODS={"HEAD_BOOTSTRAP":("local/runs/tabm-head-bootstrap-v1","local/head-bootstrap-20261003"),"JOINT_COV_MSE":("local/runs/joint-cov-mse-v1","local/joint-cov-20261003")}
CONTROL="NATIVE_MSE_CONTROL"

def threads():
    for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        if os.environ.get(n)!="1":raise ValueError("Single numeric threads required")

def load(path):return json.loads(path.read_text())

def access(out,freeze,source):
    with (out/"label-access.jsonl").open("a") as stream:
        stream.write(json.dumps({"event":"before_cached_round2_label_read","time":time.time(),"source":source,"freeze_sha256":sha(freeze),"protection_sha256":load(freeze)["frozen_files"]["configs/protection.yaml"],"protected_november_labels_requested":False})+"\n")
        stream.flush();os.fsync(stream.fileno())

def prior_identity(root):
    evidence={};old_ledgers={}
    for method,(run_rel,aux_rel) in METHODS.items():
        run=root/run_rel;aux=root/aux_rel;pre=load(run/"preflight.json");final=load(aux/"final-verification-r1.json")
        if final["actual_exit_code"]!=0 or final["preflight_sha256"]!=sha(run/"preflight.json") or load(run/"controller-finished.json")["status"]!="completed":raise ValueError("Prior terminal identity differs")
        for category in ("source_hashes","reference_files"):
            verify_tree(root,pre[category]);evidence.update(pre[category])
        for p in run.rglob("*"):
            if p.is_file():evidence[str(p.relative_to(root))]=sha(p)
        for name in ("final-verification-r1.json","controller-actual-exit-r1.json"):
            evidence[str((aux/name).relative_to(root))]=sha(aux/name)
        for p in (run/"ledger").rglob("*"):
            if p.is_file():old_ledgers[str(p.relative_to(root))]=sha(p)
        for phase in ("development","confirmation"):
            d=run/(phase+"-r1");audit=load(d/"audit.json")
            if audit["status"]!="passed" or audit["new_fits"]!=0 or audit["new_optimizers"]!=0 or audit["input_sha256"]!=sha(d/"inputs.npz"):raise ValueError("Original audit input differs")
            if len(audit["units_sha256"])!=10:raise ValueError("Original outer population incomplete")
            for unit,digest in audit["units_sha256"].items():
                u=d/unit
                if sha(u/"complete.json")!=digest:raise ValueError("Audited unit identity differs")
                verify_tree(u,load(u/"complete.json")["hashes"])
                reuse=load(u/"control-reuse.json")
                # Existing source partition/model identities remain verbatim in this receipt.
                if not reuse:raise ValueError("Control identity absent")
    for p in (root/BUNDLE).rglob("*"):
        if p.is_file():evidence[str(p.relative_to(root))]=sha(p)
    for rel in (SPEC,"configs/protection.yaml","configs/candidate_tiers.yaml","scripts/tabm_correction_fusion/replay.py","src/bf_tap_r2/tabm_correction_fusion.py","tests/test_tabm_correction_fusion.py"):
        evidence[rel]=sha(root/rel)
    verify_tree(root,evidence)
    return evidence,old_ledgers

def readonly_reference(root):
    prior=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:
        s=importlib.util.spec_from_file_location("verified_correction_reference",root/BUNDLE/"scripts/verify_swa_continuation_handoff.py")
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        v=m.verify_bundle(root/BUNDLE,"4f7f04e00f8f5d79dad26e3b66c5e5f6d5789c9993dca250ce6482393d015e9f")
        return m,v
    finally:sys.dont_write_bytecode=prior

def columns(root,method,phase,seed,ids):
    d=root/METHODS[method][0]/(phase+"-r1");folds=np.full(len(ids),-1,int);records={method:[],CONTROL:[]}
    for fold in range(5):
        u=d/f"s{seed}-f{fold}"
        with np.load(u/"partitions.npz",allow_pickle=False) as a:q=a["query"]
        if q.ndim!=1 or (q<0).any() or (q>=len(ids)).any() or np.any(folds[q]!=-1):raise ValueError("Repeated/outside query partition")
        folds[q]=fold
        with np.load(u/"predictions.npz",allow_pickle=False) as a:
            for arm in records:
                if a[arm].shape!=(len(q),2):raise ValueError("Two-target cache shape differs")
                records[arm].append((q,a["ids"],a[arm][:,1],fold))
    return {arm:assemble(ids,folds,v) for arm,v in records.items()},folds

def gain(y,parent,pred):return float(50*np.sum(np.abs(y-parent)-np.abs(y-pred))/np.sum(y))

def summary(values):
    a=np.asarray(values);mean=float(a.mean())
    return {"seed_gains":a.tolist(),"mean_gain":mean,"minimum_gain":float(a.min()),"all_seed_positive":bool((a>0).all()),"seed_lcb95_descriptive":float(mean-t.ppf(.95,3)*a.std(ddof=1)/2),"formal_promoted":False}

def produce(root):
    threads();out=root/OUT
    if out.exists():raise ValueError("Existing batch cannot be overwritten/restarted")
    spec=load(root/SPEC)
    if spec["split_seeds"]!=list(SEEDS) or spec["formal_promotion_allowed"] or len(spec["formulas"])!=5:raise ValueError("Frozen candidate protocol differs")
    evidence,ledgers=prior_identity(root)
    out.mkdir(parents=True,exist_ok=False)
    import importlib.metadata
    write_new(out/"preflight.json",{"identity":spec["identity"],"frozen_files":evidence,"old_ledgers":ledgers,"runtime":{"python":sys.version,"numpy":np.__version__,"scipy":importlib.metadata.version("scipy")},"new_fits":0,"new_optimizers":0,"new_model_inferences":0})
    access(out,out/"preflight.json",[rel+"/"+phase+"-r1/inputs.npz" for rel,_ in METHODS.values() for phase in ("development","confirmation")])
    arrays=[]
    for rel,_ in METHODS.values():
        for phase in ("development","confirmation"):
            with np.load(root/rel/(phase+"-r1")/"inputs.npz",allow_pickle=False) as a:arrays.append((a["ids"],a["y"][:,1],a["spouts"]))
    ids,y,spouts=arrays[0]
    for a in arrays[1:]:
        for x,z in zip(arrays[0],a):np.testing.assert_array_equal(x,z)
    if len(ids)!=2754 or len(set(ids))!=2754 or not np.isfinite(y).all() or y.sum()<=0:raise ValueError("Invalid full population")
    verifier,ref=readonly_reference(root);rows={};values={n:[] for n in [*spec["candidate_order"],*spec["controls"]]};correlations={}
    for seed in SEEDS:
        phase="development" if seed in (42,3407) else "confirmation"
        b,folds=columns(root,"HEAD_BOOTSTRAP",phase,seed,ids);c,other=columns(root,"JOINT_COV_MSE",phase,seed,ids)
        np.testing.assert_array_equal(folds,other);np.testing.assert_array_equal(b[CONTROL],c[CONTROL])
        parent,rf=verifier.reindex_reference(ref,seed,ids,folds);np.testing.assert_array_equal(folds,rf)
        cols=[parent,b["HEAD_BOOTSTRAP"],c["JOINT_COV_MSE"],b[CONTROL]]
        np.savez(out/f"seed-{seed}-inputs.npz",ids=ids,folds=folds,y=y,spouts=spouts,parent=cols[0],bootstrap=cols[1],covariance=cols[2],base=cols[3])
        coefficients=dict(spec["formulas"],BOOT_A20=[.8,.2,0,0],COV_A20=[.8,0,.2,0],BASE_A20=[.8,0,0,.2])
        rows[str(seed)]={}
        for name,coeff in coefficients.items():
            pred=affine(cols,coeff);g=gain(y,parent,pred);values[name].append(g)
            rows[str(seed)][name]={"gain":g,"mean_abs_change":float(np.abs(pred-parent).mean()),"max_abs_change":float(np.abs(pred-parent).max()),"fold_gains":[gain(y[folds==f],parent[folds==f],pred[folds==f]) for f in range(5)]}
            with (out/f"{name}-s{seed}.npy").open("xb") as stream:np.save(stream,pred,allow_pickle=False)
            if name in ("BOOT_A20","COV_A20","BASE_A20"):
                method="JOINT_COV_MSE" if name=="COV_A20" else "HEAD_BOOTSTRAP";arm=CONTROL if name=="BASE_A20" else method
                old=np.load(root/METHODS[method][0]/(phase+"-r1")/f"endpoint-tap_time_len-{arm}-s{seed}.npy",allow_pickle=False)
                np.testing.assert_allclose(pred,old,rtol=0,atol=1e-12)
        correlations[str(seed)]=float(np.corrcoef(cols[1]-cols[3],cols[2]-cols[3])[0,1])
    records={n:summary(v) for n,v in values.items()}
    for n in spec["candidate_order"]:
        records[n]["paired_control_mean_differences"]={c:float(np.mean(np.asarray(values[n])-values[c])) for c in spec["controls"]}
    selected=choose(records,spec["candidate_order"])
    verify_tree(root,evidence);rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if rss>1024:raise ValueError("RSS ceiling exceeded")
    write_new(out/"evaluation.json",{"status":"completed_zero_fit_exploration","records":records,"rows":rows,"correction_correlations":correlations,"exploration_selected":selected,"formal_promoted":False,"release_authorized":False,"new_fits":0,"new_optimizers":0,"new_model_inferences":0,"old_ledgers_unchanged":True,"peak_rss_mib":rss})
    write_new(out/"production-finished.json",{"evaluation_sha256":sha(out/"evaluation.json"),"artifact_hashes":{str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.suffix in (".npz",".npy")},"status":"awaiting_independent_arithmetic_audit"})
    print(json.dumps({"records":records,"selected":selected,"correlations":correlations,"rss":rss},indent=2))

def audit(root):
    threads();out=root/OUT;freeze=load(out/"preflight.json");verify_tree(root,freeze["frozen_files"]);spec=load(root/SPEC);evaluation=load(out/"evaluation.json")
    if (out/"independent-audit.json").exists():raise ValueError("Existing independent audit cannot be repeated")
    production=load(out/"production-finished.json")
    if production["evaluation_sha256"]!=sha(out/"evaluation.json"):raise ValueError("Production report changed")
    verify_tree(out,production["artifact_hashes"])
    access(out,out/"preflight.json",[str(out/f"seed-{s}-inputs.npz") for s in SEEDS])
    all_values={n:[] for n in evaluation["records"]};hashes={};maximum=0.
    for seed in SEEDS:
        with np.load(out/f"seed-{seed}-inputs.npz",allow_pickle=False) as a:
            y=a["y"];p=a["parent"];vectors=[p,a["bootstrap"],a["covariance"],a["base"]]
            if len(set(a["ids"]))!=2754 or set(a["folds"])!=set(range(5)):raise ValueError("Independent closed population invalid")
            coefficients=dict(spec["formulas"],BOOT_A20=[.8,.2,0,0],COV_A20=[.8,0,.2,0],BASE_A20=[.8,0,0,.2])
            mass=math.fsum(float(v) for v in y)
            for n,coeff in coefficients.items():
                pred=np.load(out/f"{n}-s{seed}.npy",allow_pickle=False)
                independent=np.array([math.fsum(float(c)*float(v[i]) for c,v in zip(coeff,vectors)) for i in range(len(y))])
                maximum=max(maximum,float(np.abs(independent-pred).max()))
                np.testing.assert_allclose(independent,pred,rtol=0,atol=1e-12)
                g=50*math.fsum(float(abs(v-parent)-abs(v-est)) for v,parent,est in zip(y,p,independent))/mass
                if abs(g-evaluation["rows"][str(seed)][n]["gain"])>1e-12:raise ValueError("Independent gain differs")
                all_values[n].append(g);hashes[f"{n}-s{seed}.npy"]=sha(out/f"{n}-s{seed}.npy")
    independently_selected=[]
    eligible=[n for n in spec["candidate_order"] if math.fsum(all_values[n])/4>0]
    if eligible:independently_selected=[sorted(eligible,key=lambda n:(-math.fsum(all_values[n])/4,-min(all_values[n]),spec["candidate_order"].index(n)))[0]]
    if independently_selected!=evaluation["exploration_selected"]:raise ValueError("Independent selection differs")
    for n,gs in all_values.items():
        mean=math.fsum(gs)/4;lcb=mean-float(t.ppf(.95,3))*math.sqrt(math.fsum((g-mean)**2 for g in gs)/3)/2
        saved=evaluation["records"][n]
        if abs(mean-saved["mean_gain"])>1e-12 or abs(lcb-saved["seed_lcb95_descriptive"])>1e-12 or saved["formal_promoted"]:raise ValueError("Independent summary differs")
    verify_tree(root,freeze["frozen_files"]);verify_tree(root,freeze["old_ledgers"])
    write_new(out/"independent-audit.json",{"status":"passed","independent_process_scalar_arithmetic":True,"maximum_arithmetic_difference":maximum,"evaluation_sha256":sha(out/"evaluation.json"),"prediction_hashes":hashes,"old_ledgers_unchanged":True,"selection":independently_selected,"formal_promoted":False,"new_fits":0,"new_optimizers":0,"new_model_inferences":0})
    print(json.dumps({"status":"passed","selection":independently_selected,"max_difference":maximum}))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["produce","audit"]);args=parser.parse_args()
    (produce if args.mode=="produce" else audit)(Path.cwd())
