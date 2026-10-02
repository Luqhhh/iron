"""Zero-fit post-selection SWA delta diagnostic; immutable existing model evidence."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, resource, time
import numpy as np
from scipy.stats import t
from bf_tap_r2.tabm_swa_protocol import sha, verify_tree, write_new
from bf_tap_r2.candidate_tiers import classify_candidates
import yaml

SEEDS=(42,3407,271828,314159)
SPEC_PATH="configs/swa_delta_v1/SPEC.json"
DEV="local/runs/tabm-time-swa-v1/development-r1"
CONF="local/runs/tabm-time-swa-confirm-271828-314159-20261001/confirmation-r1"
BUNDLE="local/incoming/swa-confirmation-271828-314159-20261002-r1"
OUT="local/runs/swa-delta-v1/diagnostic-r2"

def import_readonly(path):
    import sys
    original=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        spec=importlib.util.spec_from_file_location("swa_delta_verified_reference",path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        return m
    finally:
        sys.dont_write_bytecode=original

def endpoint(parent,swa,base,weight):
    p,s,b=(np.asarray(v,dtype=float) for v in (parent,swa,base))
    if p.ndim!=1 or p.shape!=s.shape or p.shape!=b.shape:
        raise ValueError("Column shape identity differs")
    if not np.isfinite([p,s,b]).all() or not np.isfinite(weight):
        raise ValueError("Inputs must be finite")
    result=p+weight*(s-b)
    if not np.isfinite(result).all() or (result<0).any():
        raise ValueError("Invalid negative or nonfinite affine endpoint")
    return result

def gain(y,parent,prediction):
    y=np.asarray(y,float)
    if y.ndim!=1 or not np.isfinite(y).all() or y.sum()<=0:
        raise ValueError("Invalid target mass")
    return float(50*(np.abs(y-parent).sum()-np.abs(y-prediction).sum())/y.sum())

def oof(directory,seed,ids,folds,fold_count=5):
    result={arm:np.full(len(ids),np.nan) for arm in ("BASE","SWA_WINDOW10")}
    count=np.zeros(len(ids),int)
    for fold in range(fold_count):
        unit=Path(directory)/f"s{seed}-f{fold}"
        with np.load(unit/"partitions.npz",allow_pickle=False) as a:
            query=a["query"]
        if query.ndim!=1 or not np.issubdtype(query.dtype,np.integer) or len(np.unique(query))!=len(query) or (query<0).any() or (query>=len(ids)).any():
            raise ValueError("Invalid query identity")
        with np.load(unit/"predictions.npz",allow_pickle=False) as a:
            np.testing.assert_array_equal(a["ids"],ids[query])
            for arm in result:
                v=a[arm]
                if v.shape!=query.shape or not np.isfinite(v).all():
                    raise ValueError("Invalid cached prediction")
                result[arm][query]=v
        count[query]+=1
    if not np.all(count==1):
        raise ValueError("OOF coverage must be exactly one")
    for fold in range(fold_count):
        with np.load(Path(directory)/f"s{seed}-f{fold}"/"partitions.npz",allow_pickle=False) as a:
            if set(a["query"])!=set(np.flatnonzero(folds==fold)):
                raise ValueError("Query fold identity differs")
    return result

def scores(y,pred,folds,spouts):
    error=np.abs(y-pred)
    def ratio(mask):
        return float(error[mask].sum()/y[mask].sum())
    return dict(wmape=float(error.sum()/y.sum()),
                by_fold={str(f):ratio(folds==f) for f in range(5)},
                by_spout={str(s):ratio(spouts==s) for s in np.unique(spouts)})

def assert_threads():
    for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        if os.environ.get(name)!="1":
            raise ValueError("Single numeric threads required")

def run(root):
    root=Path(root).resolve();out=root/OUT
    if out.exists():
        raise ValueError("Existing diagnostic cannot be overwritten or restarted")
    assert_threads()
    spec=json.loads((root/SPEC_PATH).read_text())
    if spec["weights"]!=[.1,.2,.3] or spec["split_seeds"]!=list(SEEDS) or spec["formal_promotion_allowed"]:
        raise ValueError("Frozen diagnostic specification changed")
    evidence={};ledgers={}
    for phase in (DEV,CONF):
        directory=root/phase
        for p in directory.rglob("*"):
            if p.is_file():evidence[str(p.relative_to(root))]=sha(p)
        audit=json.loads((directory/"audit.json").read_text())
        if audit["status"]!="passed" or audit["new_fits"]!=0:
            raise ValueError("Original cold audit missing")
        ledger=directory.parent/"ledger/fits.jsonl"
        evidence[str(ledger.relative_to(root))]=sha(ledger)
        ledgers[str(ledger.relative_to(root))]=sha(ledger)
        preflight=directory.parent/"preflight.json"
        frozen=json.loads(preflight.read_text())
        verify_tree(root,frozen["source_hashes"])
        verify_tree(root,frozen["reference"]["frozen_files"])
        evidence[str(preflight.relative_to(root))]=sha(preflight)
    for p in (root/BUNDLE).rglob("*"):
        if p.is_file():evidence[str(p.relative_to(root))]=sha(p)
    for rel in (SPEC_PATH,"configs/candidate_tiers.yaml","configs/protection.yaml","scripts/swa_delta/diagnose.py"):
        evidence[rel]=sha(root/rel)
    verify_tree(root,evidence)
    out.mkdir(parents=True,exist_ok=False)
    import importlib.metadata,sys
    freeze=dict(identity=spec["identity"],frozen_files=evidence,old_fit_ledgers=ledgers,
                runtime={"python":sys.version,"numpy":np.__version__,"scipy":importlib.metadata.version("scipy")},
                observed_seeds=list(SEEDS),validation_status="post_selection_exploration",
                new_fits=0,new_optimizer_runs=0,new_model_inferences=0)
    write_new(out/"freeze.json",freeze)
    # Append durable label authorization before reading the official round2 cached y.
    with (out/"label-access.jsonl").open("x") as f:
        f.write(json.dumps(dict(time=time.time(),event="before_cached_round2_development_label_read",
                   protection_sha256=evidence["configs/protection.yaml"],
                   freeze_sha256=sha(out/"freeze.json"),protected_november_labels_requested=False,
                   source_inputs=[DEV+"/inputs.npz",CONF+"/inputs.npz"]))+"\n")
        f.flush();os.fsync(f.fileno())
    with np.load(root/DEV/"inputs.npz",allow_pickle=False) as a:
        ids=a["ids"];y=a["y"][:,1];spouts=a["spouts"]
    with np.load(root/CONF/"inputs.npz",allow_pickle=False) as a:
        np.testing.assert_array_equal(ids,a["ids"])
        np.testing.assert_array_equal(y,a["y"][:,1])
        np.testing.assert_array_equal(spouts,a["spouts"])
    if len(ids)!=2754 or len(np.unique(ids))!=2754:
        raise ValueError("Incomplete official rows")
    verifier_path=root/BUNDLE/"scripts/verify_swa_continuation_handoff.py"
    m=import_readonly(verifier_path)
    verified=m.verify_bundle(root/BUNDLE,"4f7f04e00f8f5d79dad26e3b66c5e5f6d5789c9993dca250ce6482393d015e9f")
    rows={};metrics={"tap_time_len":{k:{} for k in ["Q75",*spec["candidate_order"]]}}
    for seed in SEEDS:
        directory=root/(DEV if seed in (42,3407) else CONF)
        folds=np.full(len(ids),-1,int)
        for fold in range(5):
            with np.load(directory/f"s{seed}-f{fold}"/"partitions.npz",allow_pickle=False) as a:
                folds[a["query"]]=fold
        parent,reference_folds=m.reindex_reference(verified,seed,ids,folds)
        np.testing.assert_array_equal(folds,reference_folds)
        columns=oof(directory,seed,ids,folds)
        rows[str(seed)]={}
        if seed in (42,3407):
            metrics["tap_time_len"]["Q75"][str(seed)]=scores(y,parent,folds,spouts)
        np.savez(out/f"seed-{seed}-inputs.npz",ids=ids,folds=folds,y=y,spouts=spouts,parent=parent,**columns)
        for name,weight in zip(spec["candidate_order"],spec["weights"]):
            pred=endpoint(parent,columns["SWA_WINDOW10"],columns["BASE"],weight)
            row=scores(y,pred,folds,spouts)
            row.update(gain=gain(y,parent,pred),max_abs_correction=float(np.abs(pred-parent).max()),
                       mean_abs_correction=float(np.abs(pred-parent).mean()))
            rows[str(seed)][name]=row
            if seed in (42,3407):metrics["tap_time_len"][name][str(seed)]=scores(y,pred,folds,spouts)
            with (out/f"{name}-seed-{seed}.npy").open("xb") as stream:
                np.save(stream,pred,allow_pickle=False)
    records={}
    for name in spec["candidate_order"]:
        gains=np.asarray([rows[str(s)][name]["gain"] for s in SEEDS])
        records[name]=dict(seed_gains=gains.tolist(),mean_gain=float(gains.mean()),
                          minimum_gain=float(gains.min()),
                          seed_lcb95_descriptive=float(gains.mean()-t.ppf(.95,3)*gains.std(ddof=1)/2),
                          all_seed_positive=bool((gains>0).all()),
                          formal_promoted=False,evidence="post_selection_exploration")
    eligible=[n for n in spec["candidate_order"] if records[n]["mean_gain"]>0]
    selected=sorted(eligible,key=lambda n:(-records[n]["mean_gain"],-records[n]["minimum_gain"],spec["candidate_order"].index(n)))[:1]
    tier_spec=dict(split_seeds=[42,3407],folds=5,candidates={"tap_time_len":spec["candidate_order"]},
                   reference_by_target={"tap_time_len":"Q75"},
                   tie_preference_by_target={"tap_time_len":spec["candidate_order"]})
    tier=classify_candidates(metrics,tier_spec,yaml.safe_load((root/"configs/candidate_tiers.yaml").read_text()))
    tier["interpretation"]="Historical two-seed descriptive triage only; every candidate remains post-selection exploration, no promotion"
    verify_tree(root,evidence)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if rss>1024:raise ValueError("Resource limit exceeded")
    report=dict(status="completed_zero_fit_exploration",records=records,rows=rows,
                exploration_selected=selected,development_tier_description=tier,
                formal_promoted=False,release_authorized=False,new_fits=0,new_optimizer_runs=0,
                new_model_inferences=0,old_fit_ledgers_unchanged=True,
                row_population=2754,seed_order=list(SEEDS),peak_rss_mib=rss,
                cross_split_prediction_averaging=False,packages=0,uploads=0)
    write_new(out/"evaluation.json",report)
    write_new(out/"finished.json",dict(status=report["status"],evaluation_sha256=sha(out/"evaluation.json"),
                                     source_hashes_unchanged=True,old_ledgers_unchanged=True,time=time.time()))
    print(json.dumps({"records":records,"selected":selected,"peak_rss_mib":rss},indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd());args=parser.parse_args()
    run(args.root)