"""Frozen joint-density GMR engineering/development and independent NPZ audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil
import scipy
import sklearn
from sklearn.mixture import _base, _gaussian_mixture

from .data import FEATURES
from .joint_support_audit import local_output, sha, write_new
from .joint_density_gmr import JointDensityGMR, independent_predict
from .metrics import wmape
from .splits import make_folds
from .v2_release import load_v2

SPEC = "configs/joint_density_gmr/SPEC.json"
RUN = "local/runs/joint-density-gmr-20261003"
MATERIAL = "local/runs/laplace-q75-review-20261002/intake-r1/material"
SPEC_DIGEST = "ab6148ce8419ec83f592e924f725c3cf3b672ab357173db83870f381e46d87a5"
COLD_STATUS = "passed_all_native_starts_fit_only_normalization_likelihood_selection_conditional_median_full_OOF_ledger"


def validate_spec(spec):
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != SPEC_DIGEST:
        raise ValueError("Frozen GMR specification changed")


def inventory(root):
    names = [SPEC, "docs/joint_density_gmr/PREREGISTRATION.md",
             "scripts/run_joint_density_gmr_sequence.py", "pyproject.toml", "uv.lock"]
    names += [str(p.relative_to(root)) for p in sorted((root/"src").rglob("*.py"))]
    return {name: sha(root/name) for name in names}


def runtime():
    if sys.version_info[:3] != (3, 12, 12) or np.__version__ != "2.2.6" or sklearn.__version__ != "1.8.0":
        raise ValueError("Existing locked numerical environment required")
    if any(os.environ.get(key) != "1" for key in
           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Single numerical threads required before import")
    return dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, psutil=psutil.__version__,
                installed_source_sha256={module.__file__:sha(Path(module.__file__))
                    for module in (_base, _gaussian_mixture)})


def resource(_event=None):
    rss = psutil.Process().memory_info().rss/1024**2
    if rss > 1536:
        raise MemoryError("Frozen GMR RSS gate exceeded")
    return rss


def log(out, event):
    with (out/"ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(utc_ns=time.time_ns(), **event), allow_nan=False)+"\n")


def save_arrays(path, **arrays):
    with path.open("xb") as handle:
        np.savez(handle, **arrays)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key:arrays[key].copy() for key in arrays.files}


def reference(root, spec):
    paths = {f"{MATERIAL}/MANIFEST.json":spec["reference_material_manifest_sha256"],
             f"{MATERIAL}/oof/original/q75-development-reference.npz":spec["reference_development_oof_sha256"],
             f"{MATERIAL}/submission/EMA_TIME_Q75/Luqhhh_bf_tap_predict_round2.zip":spec["reference_zip_sha256"],
             "local/runs/laplace-q75-review-20261002/intake-r1/result.json":spec["intake_result_sha256"],
             "local/runs/laplace-q75-review-20261002/review-r1/result.json":spec["matched_review_result_sha256"]}
    if any(sha(root/name) != digest for name, digest in paths.items()):
        raise ValueError("Original matched Q75 reference identity changed")
    manifest = json.loads((root/MATERIAL/"MANIFEST.json").read_text(encoding="utf-8"))
    for name, item in manifest["files"].items():
        if sha(root/MATERIAL/name) != item["sha256"]:
            raise ValueError("Original Q75 material artifact changed")
    inputs = {name:sha(root/name) for name in ("复赛_train/train_samples.csv", "复赛_train/train_features.csv")}
    intake = json.loads((root/"local/runs/laplace-q75-review-20261002/intake-r1/result.json").read_text())
    for item in intake["data"]:
        if sha(root/item["path"]) != item["raw_sha256"]:
            raise ValueError("Authorized V2 data no longer match original intake")
    inputs.update(paths)
    roles_name = f"{MATERIAL}/oof/partitions.json"
    inputs[roles_name] = sha(root/roles_name)
    roles = json.loads((root/roles_name).read_text())["units"]
    return load_arrays(root/MATERIAL/"oof/original/q75-development-reference.npz"), roles, inputs


def frame_arrays(frame):
    if set(frame.spout_no.unique()) != {1, 2}:
        raise ValueError("Only matched spouts 1/2 allowed; never silently encode 3/4")
    return np.column_stack((frame[list(FEATURES)].to_numpy(float), frame.spout_no==1, frame.spout_no==2))


def matched_data(root, spec):
    q, roles, inputs = reference(root, spec)
    frame = load_v2(root/"复赛_train", "train", 2754)
    x, y, ids = frame_arrays(frame), frame.tap_time_len.to_numpy(float), frame.sample_id.to_numpy(str)
    for observed, original in ((ids,q["ids"]),(x[:,:21],q["numeric"]),
            (frame.spout_no.to_numpy(),q["spout"]),(frame[["tap_iron","tap_time_len"]].to_numpy(float),q["targets"])):
        np.testing.assert_array_equal(observed, original)
    for seed in spec["split_seeds"]:
        folds = make_folds(frame, seed).set_index("sample_id").loc[frame.sample_id,"fold"].to_numpy(int)
        np.testing.assert_array_equal(folds,q[f"fold-{seed}"])
        for fold in range(5):
            outer, query = np.flatnonzero(folds!=fold), np.flatnonzero(folds==fold)
            role = [r for r in roles if r["split_seed"]==seed and r["outer_fold"]==fold]
            if (len(role)!=1 or role[0]["outer_training_ids"]!=ids[outer].tolist()
                    or role[0]["outer_query_ids"]!=ids[query].tolist()):
                raise ValueError("Original outer partitions differ")
    return x,y,ids,q,inputs


def fit_one(out, key, k, x, y, query_x, outer, query):
    log(out, dict(event="procedure_started",key=key,k=k,fit_rows=len(y)))
    def on_event(event):
        log(out, dict(**event,key=key))
        resource()
    model = JointDensityGMR(k).fit(x,y,on_event=on_event,on_iteration=resource)
    state, metadata = model.export_state(), model.metadata()
    save_arrays(out/f"{key}-state.npz",**state)
    write_new(out/f"{key}-metadata.json",metadata)
    prediction = model.predict(query_x)
    if prediction.shape!=(len(query),) or not np.isfinite(prediction).all():
        raise ValueError("Invalid native conditional median; no clipping")
    save_arrays(out/f"{key}-query.npz",outer=outer,query=query,prediction=prediction)
    log(out,dict(event="procedure_completed",key=key,metadata=metadata))
    return dict(key=key,k=k,metadata=metadata),prediction


def engineering(out,spec):
    x,y=synthetic_data(spec)
    save_arrays(out/"data.npz",x=x,y=y)
    units=[]
    learning={}
    for k in spec["components"]:
        unit,pred=fit_one(out,f"K{k}-engineering",k,x[:180],y[:180],x[180:],np.arange(180),np.arange(180,240))
        units.append(unit)
        learning[f"K{k}"]=dict(query_MAE=float(np.abs(y[180:]-pred).mean()),
            fit_median_constant_query_MAE=float(np.abs(y[180:]-np.median(y[:180])).mean()))
    if not all(m["query_MAE"]<m["fit_median_constant_query_MAE"] for m in learning.values()):
        raise ValueError("Synthetic GMR learning gate failed; no development admission")
    return dict(stage="engineering",units=units,learning=learning,official_label_parses=0)


def synthetic_data(spec):
    rng=np.random.default_rng(spec["engineering_seed"])
    x=np.zeros((240,23))
    x[:,:4]=rng.normal(size=(240,4))
    x[:,21]=np.arange(240)%2
    x[:,22]=1-x[:,21]
    regime=x[:,0]>0
    y=100+12*x[:,1]+np.where(regime,8*x[:,2],-5*x[:,2])+rng.normal(size=240)
    return x,y


def gain(y,parent,pred):
    blend=.8*parent+.2*pred
    if not np.isfinite(blend).all() or (blend<0).any():
        raise ValueError("Invalid fixed blend; never clipped")
    value=50*(wmape(y,parent)-wmape(y,blend))
    scalar=50*(math.fsum(abs(float(a)-float(b)) for a,b in zip(y,parent))-
               math.fsum(abs(float(a)-float(b)) for a,b in zip(y,blend)))/math.fsum(abs(float(a)) for a in y)
    if abs(value-scalar)>1e-12:
        raise ValueError("Independent scalar fixed gain differs")
    return value


def describe(q,vectors):
    metrics={}
    y,spout=q["targets"][:,1],q["spout"]
    for seed in (42,3407):
        parent,folds=q[f"current-{seed}"][:,1],q[f"fold-{seed}"]
        arms={}
        for k in (1,8):
            pred=vectors[seed,k]
            if len(pred)!=2754 or not np.isfinite(pred).all():
                raise ValueError("Only complete registered OOF vectors allowed")
            arms[f"K{k}"]=dict(single_time_wmape=wmape(y,pred),gain_vs_Q75=gain(y,parent,pred),
                fold_gains={str(f):gain(y[folds==f],parent[folds==f],pred[folds==f]) for f in range(5)},
                spout_gains={str(s):gain(y[spout==s],parent[spout==s],pred[spout==s]) for s in (1,2)})
        metrics[str(seed)]=dict(arms=arms,K8_minus_K1_gain=arms["K8"]["gain_vs_Q75"]-arms["K1"]["gain_vs_Q75"])
    return metrics


def development(root,out,spec):
    x,y,ids,q,_=matched_data(root,spec)
    save_arrays(out/"data.npz",x=x,y=y,ids=ids,spout=q["spout"])
    units,vectors=[],{}
    for seed in spec["split_seeds"]:
        folds=q[f"fold-{seed}"]
        save_arrays(out/f"folds-{seed}.npz",folds=folds)
        for k in spec["components"]:
            vectors[seed,k]=np.full(len(y),np.nan)
        for fold in range(5):
            outer,query=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
            for k in spec["components"]:
                key=f"K{k}-s{seed}-f{fold}"
                unit,pred=fit_one(out,key,k,x[outer],y[outer],x[query],outer,query)
                unit.update(seed=seed,fold=fold)
                units.append(unit)
                vectors[seed,k][query]=pred
        for k in spec["components"]:
            save_arrays(out/f"oof-K{k}-{seed}.npz",prediction=vectors[seed,k])
    metrics=describe(q,vectors)
    return dict(stage="development",units=units,metrics=metrics,
        eligible_for_separately_frozen_confirmation=all(m["arms"]["K8"]["gain_vs_Q75"]>0 for m in metrics.values()),
        mean_K8_gain_vs_Q75=float(np.mean([m["arms"]["K8"]["gain_vs_Q75"] for m in metrics.values()])))


def independent_score(z,weights,means,covariances):
    terms=[]
    d=z.shape[1]
    for w,m,c in zip(weights,means,covariances):
        sign,logdet=np.linalg.slogdet(c)
        if sign!=1 or w<=0:
            raise ValueError("Invalid joint covariance or prior")
        delta=z-m
        quadratic=np.sum(delta*np.linalg.solve(c,delta.T).T,axis=1)
        terms.append(math.log(float(w))-.5*(d*math.log(2*math.pi)+logdet+quadratic))
    rows=np.stack(terms,axis=1)
    maximum=rows.max(axis=1)
    return float(np.mean(maximum+np.log(np.exp(rows-maximum[:,None]).sum(axis=1))))


def audit_state(state,metadata,x,y,k,spec):
    if (metadata["fit_rows"]!=len(y) or metadata["n_components"]!=k
            or metadata["native_fit_count"]!=5 or metadata["native_kmeans_count"]!=5
            or len(metadata["starts"])!=5):
        raise ValueError("Native GMR metadata differs")
    xmean,xscale=x.mean(axis=0),x.std(axis=0)
    xscale=np.where(xscale>0,xscale,1.)
    ymean,yscale=y.mean(),y.std()
    yscale=yscale if yscale>0 else 1.
    for key,value in dict(x_mean=xmean,x_scale=xscale,y_mean=ymean,y_scale=yscale).items():
        np.testing.assert_array_equal(state[key],value)
    d=x.shape[1]+1
    for key,shape in dict(weights=(5,k),means=(5,k,d),covariances=(5,k,d,d),n_iters=(5,),
                          lower_bounds=(5,),final_fit_scores=(5,),converged=(5,)).items():
        if state[key].shape!=shape or not np.isfinite(state[key]).all():
            raise ValueError("Incomplete native-start numeric state")
    if not state["converged"].all() or (state["n_iters"]<1).any() or (state["n_iters"]>2000).any():
        raise ValueError("Native convergence/iteration budget differs")
    if (state["converged"].dtype!=np.dtype(bool) or state["n_iters"].dtype.kind not in "iu"
            or state["selected_start"].shape!=() or state["selected_start"].dtype.kind not in "iu"):
        raise ValueError("Native numeric state types differ")
    np.testing.assert_allclose(state["weights"].sum(axis=1),np.ones(5),rtol=0,atol=1e-12)
    for key,value in dict(random_states=np.arange(42,47),fit_rows=len(y),native_fit_count=5,
            native_kmeans_count=5,reg_covar=.01,max_iter=2000,tol=1e-5).items():
        np.testing.assert_array_equal(state[key],value)
    trace=state["em_lower_bounds"]
    if trace.shape!=(5,2000) or not np.isfinite(trace).all():
        raise ValueError("Complete native iteration trace absent")
    z=np.column_stack(((x-xmean)/xscale,(y-ymean)/yscale))
    for i,item in enumerate(metadata["starts"]):
        if (item["start_index"]!=i or item["random_state"]!=42+i
                or item["n_iter"]!=int(state["n_iters"][i]) or not item["converged"]
                or item["final_fit_score"]!=float(state["final_fit_scores"][i])
                or item["lower_bound"]!=float(state["lower_bounds"][i])):
            raise ValueError("Native-start trace/state mismatch")
        n=int(state["n_iters"][i])
        if (float(trace[i,n-1])!=item["lower_bound"] or (trace[i,n:]!=0).any()
                or item["em_lower_bounds"]!=trace[i,:n].tolist()):
            raise ValueError("Native iteration history length/final bound differs")
        for c in state["covariances"][i]:
            np.testing.assert_allclose(c,c.T,rtol=0,atol=1e-12)
            np.linalg.cholesky(c)
        score=independent_score(z,state["weights"][i],state["means"][i],state["covariances"][i])
        if abs(score-item["final_fit_score"])>spec["cold_likelihood_absolute_tolerance"]:
            raise ValueError("Independent native final fit likelihood differs")
        resource()
    selected=int(np.argmax(state["final_fit_scores"]))
    if int(state["selected_start"])!=selected or metadata["selected_start"]!=selected:
        raise ValueError("Fit-only start choice differs")


def audit_ledger(out,units):
    ledger=[json.loads(line) for line in (out/"ledger.jsonl").read_text().splitlines()]
    keys=[u["key"] for u in units]
    native=("native_fit_started","native_kmeans_started","native_kmeans_completed","native_fit_completed")
    allowed={"procedure_started","procedure_completed",*native}
    if any(entry["key"] not in keys or entry["event"] not in allowed for entry in ledger):
        raise ValueError("Unknown ledger key/event; no hidden native fits")
    if any(sum(entry["event"]==event for entry in ledger)!=5*len(units) for event in native):
        raise ValueError("Global native EM/KMeans ledger count differs")
    for event in ("procedure_started","procedure_completed"):
        entries=[entry["key"] for entry in ledger if entry["event"]==event]
        if entries!=keys or len(set(entries))!=len(keys):
            raise ValueError("Procedure ledger incomplete or duplicated")
    for unit in units:
        key=unit["key"]
        events=[entry for entry in ledger if entry["key"]==key]
        if [entry["event"] for entry in events]!=["procedure_started"]+list(native)*5+["procedure_completed"]:
            raise ValueError("Native ledger order differs")
        if events[0]["fit_rows"]!=unit["metadata"]["fit_rows"] or events[-1]["metadata"]!=unit["metadata"]:
            raise ValueError("Procedure ledger metadata differs")
        for event in native:
            selected=[entry for entry in events if entry["event"]==event]
            entries=[entry["start_index"] for entry in selected]
            if entries!=list(range(5)):
                raise ValueError("Native EM/KMeans ledger incomplete or duplicated")
            if any(entry["random_state"]!=42+entry["start_index"]
                    or entry["fit_rows"]!=unit["metadata"]["fit_rows"] for entry in selected):
                raise ValueError("Native ledger fit-only identity differs")


def verify(root,out,spec):
    manifest=json.loads((out/"manifest.json").read_text())
    result=json.loads((out/"result.json").read_text())
    if manifest["spec"]!=spec or manifest["sources"]!=inventory(root) or manifest["runtime"]!=runtime():
        raise ValueError("Frozen cold source/environment differs")
    if manifest["action"]!=result["stage"]:
        raise ValueError("Native stage identity differs")
    for name,digest in {**manifest["inputs"],**{str(out/p):v for p,v in result["output_sha256"].items()}}.items():
        path=Path(name) if Path(name).is_absolute() else root/name
        if sha(path)!=digest:
            raise ValueError("Cold input/output identity changed")
    data=load_arrays(out/"data.npz")
    x,y=data["x"],data["y"]
    if result["stage"]=="development":
        mx,my,mids,q,_=matched_data(root,spec)
        for a,b in ((x,mx),(y,my),(data["ids"],mids)):
            np.testing.assert_array_equal(a,b)
        expected=[(f"K{k}-s{s}-f{f}",k,s,f) for s in (42,3407) for f in range(5) for k in (1,8)]
    elif result["stage"]=="engineering":
        sx,sy=synthetic_data(spec)
        np.testing.assert_array_equal(x,sx)
        np.testing.assert_array_equal(y,sy)
        q=None
        expected=[(f"K{k}-engineering",k,None,None) for k in (1,8)]
    else:
        raise ValueError("Unknown native stage")
    if [(u["key"],u["k"],u.get("seed"),u.get("fold")) for u in result["units"]]!=expected:
        raise ValueError("Full unit pool/order differs")
    vectors,counts={},{}
    for s in (42,3407):
        for k in (1,8):
            vectors[s,k]=np.full(len(y),np.nan)
            counts[s,k]=np.zeros(len(y),int)
    maximum=0.
    learning={}
    for unit in result["units"]:
        key,k=unit["key"],unit["k"]
        parts=load_arrays(out/f"{key}-query.npz")
        if q is None:
            outer,query=np.arange(180),np.arange(180,240)
        else:
            s,f=unit["seed"],unit["fold"]
            folds=q[f"fold-{s}"]
            np.testing.assert_array_equal(load_arrays(out/f"folds-{s}.npz")["folds"],folds)
            outer,query=np.flatnonzero(folds!=f),np.flatnonzero(folds==f)
        np.testing.assert_array_equal(parts["outer"],outer)
        np.testing.assert_array_equal(parts["query"],query)
        metadata=json.loads((out/f"{key}-metadata.json").read_text())
        if metadata!=unit["metadata"]:
            raise ValueError("Native metadata identity differs")
        state=load_arrays(out/f"{key}-state.npz")
        audit_state(state,metadata,x[outer],y[outer],k,spec)
        pred=independent_predict(state,x[query])
        if pred.shape!=parts["prediction"].shape:
            raise ValueError("Independent conditional median shape differs")
        difference=float(np.max(np.abs(pred-parts["prediction"])))
        if not np.isfinite(pred).all() or difference>spec["cold_prediction_absolute_tolerance"]:
            raise ValueError("Independent conditional median differs")
        maximum=max(maximum,difference)
        # Query permutation/chunk never enter any normalization or EM fit.
        np.testing.assert_array_equal(independent_predict(state,x[query][::-1])[::-1],pred)
        split=len(query)//2
        np.testing.assert_array_equal(np.r_[independent_predict(state,x[query][:split]),
            independent_predict(state,x[query][split:])],pred)
        if q is not None:
            vectors[s,k][query]=parts["prediction"]
            counts[s,k][query]+=1
        else:
            learning[f"K{k}"]=dict(query_MAE=float(np.abs(y[query]-parts["prediction"]).mean()),
                fit_median_constant_query_MAE=float(np.abs(y[query]-np.median(y[outer])).mean()))
        resource()
    audit_ledger(out,result["units"])
    n=len(expected)
    iterations=sum(sum(int(start["n_iter"]) for start in u["metadata"]["starts"]) for u in result["units"])
    if (result["training_procedures"]!=n or result["native_EM_fits"]!=5*n
            or result["native_KMeans_fits"]!=5*n or result["actual_EM_iterations"]!=iterations):
        raise ValueError("Actual native counts do not close")
    if q is not None:
        for s in (42,3407):
            for k in (1,8):
                if not (counts[s,k]==1).all():
                    raise ValueError("OOF row coverage differs")
                np.testing.assert_array_equal(vectors[s,k],load_arrays(out/f"oof-K{k}-{s}.npz")["prediction"])
        metrics=describe(q,vectors)
        if (metrics!=result["metrics"] or all(m["arms"]["K8"]["gain_vs_Q75"]>0 for m in metrics.values())
                !=result["eligible_for_separately_frozen_confirmation"]
                or float(np.mean([m["arms"]["K8"]["gain_vs_Q75"] for m in metrics.values()]))!=result["mean_K8_gain_vs_Q75"]):
            raise ValueError("Complete cold metrics/decision differ")
    elif (learning!=result["learning"] or not all(
            m["query_MAE"]<m["fit_median_constant_query_MAE"] for m in learning.values())):
        raise ValueError("Independent synthetic learning gate differs")
    if inventory(root)!=manifest["sources"] or runtime()!=manifest["runtime"]:
        raise ValueError("Source changed during audit")
    return dict(status=COLD_STATUS,checked_procedures=n,checked_native_states=5*n,
        checked_native_KMeans_fits=5*n,actual_EM_iterations=iterations,
        maximum_cold_difference=maximum,absolute_prediction_tolerance=spec["cold_prediction_absolute_tolerance"],
        new_fits=0,model_deserializations=0,EM_retraining_performed=False,
        manifest_sha256=sha(out/"manifest.json"),result_sha256=sha(out/"result.json"),packages=0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("engineering","development","verify"))
    parser.add_argument("--output",required=True)
    parser.add_argument("--engineering")
    args=parser.parse_args()
    root=Path.cwd().resolve()
    spec=json.loads((root/SPEC).read_text())
    validate_spec(spec)
    env=runtime()
    out=local_output(root,args.output)
    if args.action=="verify":
        receipt=verify(root,out,spec)
        write_new(out/"cold-readback.json",receipt)
        print(json.dumps(receipt),flush=True)
        return
    sources=inventory(root)
    inputs={}
    if args.action=="development":
        engineering_run=local_output(root,args.engineering)
        em=json.loads((engineering_run/"manifest.json").read_text())
        er=json.loads((engineering_run/"result.json").read_text())
        ec=json.loads((engineering_run/"cold-readback.json").read_text())
        if (em["spec"]!=spec or em["sources"]!=sources or em["runtime"]!=env
                or em["action"]!="engineering" or ec["status"]!=COLD_STATUS
                or ec["checked_procedures"]!=2 or ec["checked_native_states"]!=10 or ec["new_fits"]!=0
                or ec["maximum_cold_difference"]>spec["cold_prediction_absolute_tolerance"]
                or ec["manifest_sha256"]!=sha(engineering_run/"manifest.json")
                or ec["result_sha256"]!=sha(engineering_run/"result.json")):
            raise ValueError("Complete engineering cold admission failed")
        for name,digest in er["output_sha256"].items():
            if sha(engineering_run/name)!=digest:
                raise ValueError("Bound G0 output changed")
        _,_,inputs=reference(root,spec)
        inputs.update({str(engineering_run/name):sha(engineering_run/name)
            for name in ("manifest.json","result.json","cold-readback.json")})
        inputs.update({str(engineering_run/name):digest for name,digest in er["output_sha256"].items()})
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/"manifest.json",dict(identity=spec["identity"],spec=spec,action=args.action,
        sources=sources,inputs=inputs,runtime=env,started_ns=time.time_ns(),pid=os.getpid(),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()))
    try:
        result=engineering(out,spec) if args.action=="engineering" else development(root,out,spec)
        if inventory(root)!=sources or runtime()!=env or any(sha(root/p)!=v for p,v in inputs.items()):
            raise ValueError("Frozen source/input/runtime changed during fit")
        n=2 if args.action=="engineering" else 20
        if len(result["units"])!=n:
            raise ValueError("Native procedure pool incomplete")
        audit_ledger(out,result["units"])
        result.update(training_procedures=n,native_EM_fits=5*n,native_KMeans_fits=5*n,
            actual_EM_iterations=sum(sum(int(s["n_iter"]) for s in u["metadata"]["starts"]) for u in result["units"]),
            formal_promoted=False,confirmation_seeds_consumed=0,full_fits=0,packages=0,agent_uploads=0,
            observed_rss_mib=resource(),platform_score=None,completed_ns=time.time_ns())
        result["output_sha256"]={p.name:sha(p) for p in out.iterdir() if p.is_file()}
        write_new(out/"result.json",result)
        print(json.dumps({k:v for k,v in result.items() if k not in ("units","output_sha256")}),flush=True)
    except Exception as exc:
        write_new(out/"FAILED.json",dict(type=type(exc).__name__,error=str(exc),automatic_retry=False))
        raise


if __name__=="__main__":
    main()
