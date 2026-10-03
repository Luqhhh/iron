"""Independent cold audit: no fit or optimizer; original raw checkpoint metric and independent training-only weight/order replay."""
from pathlib import Path
import json,resource
import numpy as np

from .bootstrap_ema_protocol import BUDGETS,CANDIDATE,CONTROL,expected_mechanisms,Ledger,sha,digest,write_new,verify_tree,child
from .data import FEATURES

def verify_selection(trace,settings,actual):
    best=float("inf");selected=0;stale=0
    for number,row in enumerate(trace["history"],1):
        if row["epoch"]!=number or stale>=settings["patience"]:
            raise ValueError("Unexpected selector epoch")
        if row["validation_mae"]<best-settings["min_delta"]:
            best=row["validation_mae"];selected=row["epoch"];stale=0
        else:stale+=1
    if selected!=trace["selected_epoch"] or len(trace["history"])!=trace["stopped_epoch"] or len(trace["history"])>settings["max_epochs"]:
        raise ValueError("Selected epoch differs")
    if not np.isfinite(actual) or abs(actual-best)>1e-6:
        raise ValueError("Checkpoint metric differs from selected epoch")
    return selected

def load_predictions(path):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={"ids","cal_ids",CONTROL,CANDIDATE}:raise ValueError("Unexpected prediction fields")
        return {k:z[k].copy() for k in z.files}

def verify_bootstrap(trace,ids,settings):
    import hashlib
    ids=np.asarray(ids)
    if trace["bootstrap_fit_ids_digest"]!=digest(ids.tolist()):raise ValueError("Bootstrap fit identity differs")
    heads=settings.get("tabm_k",settings.get("k"));batch=settings["batch_size"]
    rng=np.random.Generator(np.random.PCG64(int(settings["random_seed"])+1000003));order=np.random.default_rng(settings["random_seed"])
    h=hashlib.sha256();draws=total=zeros=zero_batches=0
    for row in trace["history"]:
        indices=order.permutation(len(ids))
        for start in range(0,len(ids),batch):
            ix=indices[start:start+batch];weights=rng.poisson(1.0,(len(ix),heads)).astype('<i8')
            h.update(digest(ids[ix].tolist()).encode('ascii'));h.update(weights.tobytes())
            draws+=weights.size;total+=int(weights.sum());zeros+=int(np.count_nonzero(weights==0));zero_batches+=int(np.count_nonzero(weights)==0)
        expected=dict(draws=draws,weight_sum=total,zero_weights=zeros,zero_batches=zero_batches,weight_and_order_sha256=h.hexdigest())
        if row["bootstrap"]!=expected:raise ValueError("Training-only Poisson weight/order certificate differs")
    if trace.get("bootstrap_seed",int(settings["random_seed"])+1000003)!=int(settings["random_seed"])+1000003:raise ValueError("Bootstrap RNG seed differs")

def replay_ema(directory,updates,parameter_names,beta):
    d=Path(directory)
    if not 0<=beta<1 or updates<1:raise ValueError("Invalid EMA recurrence")
    with np.load(d/"initial.npz",allow_pickle=False) as a:state={k:a[k].astype(np.float64) for k in a.files}
    if not all(np.isfinite(v).all() for v in state.values()) or not set(parameter_names)<=set(state):raise ValueError("Invalid initial state")
    for number in range(1,updates+1):
        with np.load(d/f"step-{number:06d}.npz",allow_pickle=False) as a:
            if set(a.files)!=set(state):raise ValueError("Raw state fields differ")
            for k,old in state.items():
                raw=a[k].astype(np.float64)
                if raw.shape!=old.shape or not np.isfinite(raw).all():raise ValueError("Raw state shape/finite identity differs")
                state[k]=beta*old+(1-beta)*raw if k in parameter_names else raw
    return state

def audit(directory):
    import torch
    from .component_regularization import ComponentRegressor
    d=Path(directory).resolve();root=d.parents[3]
    manifest=json.loads((d/"manifest.json").read_text())
    verify_tree(root,manifest["source_hashes"])
    phase=manifest["phase"];settings=manifest["settings"]
    if manifest["budgets"]!=BUDGETS[phase]:raise ValueError("Budget identity differs")
    from .bootstrap_ema_run import append_access,active_freeze
    freeze_path=d.parent/"preflight.json" if (d.parent/"preflight.json").exists() else active_freeze(root)
    append_access(root,freeze_path)
    with np.load(d/"inputs.npz",allow_pickle=False) as a:
        x={k:a[k].copy() for k in a.files}
    if sha(d/"inputs.npz")!=manifest["input_sha256"]:raise ValueError("Input hash differs")
    frame=np.asarray(x["x"]);y=x["y"]
    import pandas as pd
    f=pd.DataFrame(frame,columns=FEATURES);f["spout_no"]=x["spouts"];f["sample_id"]=x["ids"]
    groups=pd.util.hash_pandas_object(f[list(FEATURES)],index=False).to_numpy()
    def forbidden(*args,**kw):raise RuntimeError("Cold audit fit/optimizer forbidden")
    ComponentRegressor.fit=forbidden;torch.optim.AdamW=forbidden;torch.optim.Adam=forbidden
    differences=[];models=0
    for unit in manifest["units"]:
        q=d/unit;c=json.loads((q/"complete.json").read_text());verify_tree(q,c["hashes"])
        with np.load(q/"partitions.npz",allow_pickle=False) as a:
            part={name:a[name] for name in ("train","query","inner","cal")}
        for name,indices in part.items():np.testing.assert_array_equal(indices,x[name+"__"+unit])
        train,query,inner,cal=[part[name] for name in ("train","query","inner","cal")]
        if not np.array_equal(np.sort(np.r_[inner,cal]),np.sort(train)) or set(groups[train])&set(groups[query]) or set(groups[inner])&set(groups[cal]):
            raise ValueError("Partition leakage")
        pred_path=q/"predictions.npz"
        output=load_predictions(pred_path)
        np.testing.assert_array_equal(output["ids"],x["ids"][query]);np.testing.assert_array_equal(output["cal_ids"],x["ids"][cal])
        reuse=json.loads((q/"control-reuse.json").read_text())
        control=child(root,reuse["source_directory"])
        if reuse["new_control_fits"]!=0:raise ValueError("Control reuse count differs")
        for name,h in reuse["model_hashes"].items():
            if sha(control/name)!=h:raise ValueError("Reused control model changed")
        if sha(control.parent/"complete.json")!=reuse["original_complete_sha256"]:raise ValueError("Control source identity changed")
        old=json.loads((control.parent/"complete.json").read_text());verify_tree(control.parent,old["hashes"])
        for arm in (CONTROL,CANDIDATE):
            source=control if arm==CONTROL else q/CANDIDATE
            selected=None
            for stage,fit,held in [("selection",inner,cal),("refit",train,query)]:
                model=ComponentRegressor.load(source/(stage+".pt"));saved=model.saved;trace=saved["trace"]
                expected=expected_mechanisms(arm,phase)
                if saved["settings"]!=settings or saved["recipe"]!={"backbone":"tabm","frequency":.01} or saved["arm"]!="BASE" or saved["mechanisms"]!=expected:
                    raise ValueError("Scientific recipe differs")
                if arm==CANDIDATE:
                    for row in trace["history"]:
                        if row.get("learning_rate")!=settings["learning_rate"] or row.get("training_objective")!="fixed_denominator_per_row_head_poisson_mse":
                            raise ValueError("Candidate objective/LR trace differs")
                if trace["fit_ids_digest"]!=digest(x["ids"][fit].tolist()) or trace["fit_rows"]!=len(fit):
                    raise ValueError("Fit identity differs")
                vocabulary={int(s):i+1 for i,s in enumerate(sorted(f.iloc[fit].spout_no.unique()))}
                if {int(k):v for k,v in saved["preprocessing"]["spout_vocabulary"].items()}!=vocabulary or saved["preprocessing"]["n_spout_categories"]!=len(vocabulary)+1:
                    raise ValueError("Vocabulary differs")
                for a,b in [(saved["preprocessing"]["means"],frame[fit].mean(0)),(saved["preprocessing"]["stds"],frame[fit].std(0)),(saved["mean"],y[fit].mean(0)),(saved["std"],y[fit].std(0))]:
                    np.testing.assert_allclose(a,b,rtol=1e-14,atol=1e-12)
                verify_bootstrap(trace,x["ids"][fit],settings)
                if arm==CANDIDATE:
                    if trace["ema_beta"]!=.99 or any(row.get("ema_beta")!=.99 or row["ema_cumulative_updates"]!=sum(v["updates"] for v in trace["history"][:row["epoch"]]) for row in trace["history"]):raise ValueError("EMA trace differs")
                    archive=source/trace["ema_raw_archive"]
                    if len(list(archive.glob("step-*.npz")))!=trace["ema_archive_updates"]:raise ValueError("EMA raw archive coverage differs")
                    expected_step=sum(row["updates"] for row in trace["history"][:trace["selected_epoch"]])
                    if trace["ema_saved_update"]!=expected_step or trace["ema_archive_updates"]!=trace["updates"]:raise ValueError("EMA saved step identity differs")
                    if set(trace["ema_parameter_names"])!=set(dict(model.model_.named_parameters())):raise ValueError("EMA parameter identity differs")
                    independently_averaged=replay_ema(archive,expected_step,trace["ema_parameter_names"],.99)
                    for k,v in saved["state"].items():np.testing.assert_allclose(v.numpy(),independently_averaged[k],rtol=1e-5,atol=5e-7)
                expected_updates=(len(fit)+settings["batch_size"]-1)//settings["batch_size"]
                if any(row["updates"]!=expected_updates or row["gradient_evaluations"]!=expected_updates for row in trace["history"]):raise ValueError("Update count differs")
                held_frame=f.iloc[held]
                if stage=="selection":
                    vx,vc=model._inputs(held_frame);target=torch.as_tensor((y[held]-model.mean_)/model.std_,dtype=torch.float32)
                    with torch.no_grad():
                        error=(model.model_(vx,vc).mean(1)-target).abs()
                        actual=float(error.mean())
                    selected=verify_selection(trace,settings,actual)
                elif trace["selected_epoch"]!=selected or trace["stopped_epoch"]!=selected or len(trace["history"])!=selected:
                    raise ValueError("Fresh refit epoch differs")
                full=model.predict(held_frame);reverse=model.predict(held_frame.iloc[::-1])[::-1]
                chunks=np.concatenate([model.predict(held_frame.iloc[i:i+37]) for i in range(0,len(held_frame),37)])
                differences += [float(np.abs(full-reverse).max()),float(np.abs(full-chunks).max())]
                if stage=="refit":
                    differences.append(float(np.abs(full-output[arm]).max()))
                    if phase=="engineering" and any(np.mean(np.abs(full[:,j]-y[query,j]))>=np.mean(np.abs(np.median(y[train,j])-y[query,j])) for j in range(2)):
                        raise ValueError("Synthetic two-target learnability failed")
                models+=1;del model
    counts=Ledger(d.parent/"ledger",BUDGETS).counts(phase)
    if counts["reserved"]!=BUDGETS[phase] or counts["completed"]!=counts["reserved"]:
        raise ValueError("Budget incomplete")
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if max(differences)>5e-4 or rss>1024:raise ValueError("Cold prediction/resource check failed")
    write_new(d/"audit.json",dict(status="passed",cold_models=models,reused_control_states=models//2,new_fits=0,new_optimizers=0,
                         max_difference=max(differences),training_only_Poisson_weights_order_independently_replayed=True,ema_raw_update_float64_recurrence_independently_replayed=True,
                         peak_rss_mib=rss,counts=counts,input_sha256=sha(d/"inputs.npz"),manifest_sha256=sha(d/"manifest.json"),
                         units_sha256={u:sha(d/u/"complete.json") for u in manifest["units"]}))
if __name__=="__main__":
    import argparse,torch
    p=argparse.ArgumentParser();p.add_argument("directory");a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);audit(a.directory)
