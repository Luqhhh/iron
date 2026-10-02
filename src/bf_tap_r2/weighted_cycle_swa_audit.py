"""Independent cold audit: sparse cycle trajectory and original uniform SWA control."""
from pathlib import Path
import json,resource
import numpy as np

from .weighted_cycle_swa_protocol import BUDGETS,CANDIDATE,CONTROL,Ledger,sha,digest,write_new,verify_tree,child
from .data import FEATURES
from .weighted_cycle_swa_protocol import MECHANISMS,CONTROL_MECHANISMS,epoch_lr,verify_cycle_window,verify_cycle_selection

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

def load_predictions(path,legacy_sha256=None):
    import io,pickle,zipfile
    class StringsOnly(pickle.Unpickler):
        def find_class(self,module,name):
            allowed={("numpy._core.multiarray","_reconstruct"):np._core.multiarray._reconstruct,
                     ("numpy","ndarray"):np.ndarray,("numpy","dtype"):np.dtype}
            if (module,name) not in allowed:
                raise ValueError("Unapproved pickle global")
            return allowed[(module,name)]
    path=Path(path)
    if legacy_sha256 is not None and sha(path)!=legacy_sha256:
        raise ValueError("Legacy prediction identity differs")
    result={}
    with zipfile.ZipFile(path) as z:
        expected={name+".npy" for name in ("ids","cal_ids",CONTROL,CANDIDATE)}
        if set(z.namelist())!=expected:
            raise ValueError("Unexpected prediction fields")
        for entry in z.namelist():
            name=entry[:-4];raw=z.read(entry);stream=io.BytesIO(raw)
            version=np.lib.format.read_magic(stream)
            shape,order,dtype=np.lib.format._read_array_header(stream,version)
            if dtype.hasobject:
                if legacy_sha256 is None or name not in ("ids","cal_ids"):
                    raise ValueError("Unapproved legacy object field")
                if len(shape)!=1 or shape[0]>2754:
                    raise ValueError("Legacy ID shape differs")
                a=StringsOnly(stream).load()
                if not isinstance(a,np.ndarray) or a.shape!=shape or not all(isinstance(v,str) for v in a):
                    raise ValueError("Legacy IDs must contain strings only")
                result[name]=np.asarray(a,dtype=str)
            else:
                result[name]=np.load(io.BytesIO(raw),allow_pickle=False)
    return result

def verify_repair(root,manifest,manifest_path,bridge_path):
    b=json.loads(Path(bridge_path).read_text());allowed={
        "src/bf_tap_r2/cycle_swa_run.py","src/bf_tap_r2/cycle_swa_audit.py",
        "tests/test_cycle_swa.py","local/swa-cycle-tail-20261002/monitor_once.py"}
    if b.get("status")!="user_authorized_zero_fit_serialization_repair" or b.get("original_manifest_sha256")!=sha(manifest_path):
        raise ValueError("Invalid recovery bridge")
    changed={p for p,h in manifest["source_hashes"].items() if sha(child(root,p))!=h}
    if changed!=set(b["changes"]) or not changed<=allowed:
        raise ValueError("Unexpected recovered source")
    for p,c in b["changes"].items():
        if manifest["source_hashes"][p]!=c["original_sha256"] or sha(child(root,c["original_archive"]))!=c["original_sha256"] or sha(child(root,p))!=c["current_sha256"]:
            raise ValueError("Recovery source/archive identity differs")
    verify_tree(root,b["original_prediction_files"])
    return b

def audit(directory,source_recovery=None):
    import torch
    from .component_regularization import ComponentRegressor
    from .tabm_swa_audit import verify_window
    d=Path(directory).resolve();root=d.parents[3]
    manifest=json.loads((d/"manifest.json").read_text())
    if source_recovery is None:
        verify_tree(root,manifest["source_hashes"]);bridge=None
    else:
        bridge=verify_repair(root,manifest,d/"manifest.json",source_recovery)
    phase=manifest["phase"];settings=manifest["settings"]
    if manifest["budgets"]!=BUDGETS[phase]:raise ValueError("Budget identity differs")
    with np.load(d/"inputs.npz",allow_pickle=False) as a:
        x={k:a[k].copy() for k in a.files}
    if sha(d/"inputs.npz")!=manifest["input_sha256"]:raise ValueError("Input hash differs")
    frame=np.asarray(x["x"]);y=x["y"]
    import pandas as pd
    f=pd.DataFrame(frame,columns=FEATURES);f["spout_no"]=x["spouts"];f["sample_id"]=x["ids"]
    groups=pd.util.hash_pandas_object(f[list(FEATURES)],index=False).to_numpy()
    def forbidden(*args,**kw):raise RuntimeError("Cold audit fit/optimizer forbidden")
    ComponentRegressor.fit=forbidden;torch.optim.AdamW=forbidden;torch.optim.Adam=forbidden
    differences=[];window_differences=[];models=0
    for unit in manifest["units"]:
        q=d/unit;c=json.loads((q/"complete.json").read_text());verify_tree(q,c["hashes"])
        with np.load(q/"partitions.npz",allow_pickle=False) as a:
            part={name:a[name] for name in ("train","query","inner","cal")}
        for name,indices in part.items():np.testing.assert_array_equal(indices,x[name+"__"+unit])
        train,query,inner,cal=[part[name] for name in ("train","query","inner","cal")]
        if not np.array_equal(np.sort(np.r_[inner,cal]),np.sort(train)) or set(groups[train])&set(groups[query]) or set(groups[inner])&set(groups[cal]):
            raise ValueError("Partition leakage")
        pred_path=q/"predictions.npz"
        legacy=bridge["original_prediction_files"].get(str(pred_path.relative_to(root))) if bridge is not None else None
        output=load_predictions(pred_path,legacy_sha256=legacy)
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
                expected=CONTROL_MECHANISMS if arm==CONTROL else MECHANISMS
                if saved["settings"]!=settings or saved["recipe"]!={"backbone":"tabm","frequency":.01} or saved["arm"]!="BASE" or saved["mechanisms"]!=expected:
                    raise ValueError("Scientific recipe differs")
                if trace["fit_ids_digest"]!=digest(x["ids"][fit].tolist()) or trace["fit_rows"]!=len(fit):
                    raise ValueError("Fit identity differs")
                vocabulary={int(s):i+1 for i,s in enumerate(sorted(f.iloc[fit].spout_no.unique()))}
                if {int(k):v for k,v in saved["preprocessing"]["spout_vocabulary"].items()}!=vocabulary or saved["preprocessing"]["n_spout_categories"]!=len(vocabulary)+1:
                    raise ValueError("Vocabulary differs")
                for a,b in [(saved["preprocessing"]["means"],frame[fit].mean(0)),(saved["preprocessing"]["stds"],frame[fit].std(0)),(saved["mean"],y[fit].mean(0)),(saved["std"],y[fit].std(0))]:
                    np.testing.assert_allclose(a,b,rtol=1e-14,atol=1e-12)
                witness=torch.load(source/(stage+"-window.pt"),map_location="cpu",weights_only=True)
                window_differences.append(verify_cycle_window(saved,witness))
                held_frame=f.iloc[held]
                if stage=="selection":
                    vx,vc=model._inputs(held_frame);target=torch.as_tensor((y[held]-model.mean_)/model.std_,dtype=torch.float32)
                    with torch.no_grad():
                        error=(model.model_(vx,vc).mean(1)-target).abs()
                        actual=float(error.mean())
                    selected=verify_cycle_selection(trace,settings,actual)
                elif trace["selected_epoch"]!=selected or trace["stopped_epoch"]!=selected or len(trace["history"])!=selected:
                    raise ValueError("Fresh refit epoch differs")
                if stage=="refit":
                    for epoch,row in enumerate(trace["history"],1):
                        if row["epoch"]!=epoch or abs(row["learning_rate"]-epoch_lr(epoch))>1e-15 or "validation_mae" in row:
                            raise ValueError("Fresh refit cycle/LR differs")
                full=model.predict(held_frame)[:,1];reverse=model.predict(held_frame.iloc[::-1])[:,1][::-1]
                chunks=np.concatenate([model.predict(held_frame.iloc[i:i+37])[:,1] for i in range(0,len(held_frame),37)])
                differences += [float(np.abs(full-reverse).max()),float(np.abs(full-chunks).max())]
                if stage=="refit":
                    differences.append(float(np.abs(full-output[arm]).max()))
                    if phase=="engineering" and np.mean(np.abs(full-y[query,1]))>=np.mean(np.abs(np.median(y[train,1])-y[query,1])):
                        raise ValueError("Synthetic time learnability failed")
                models+=1;del model
    counts=Ledger(d.parent/"ledger",BUDGETS).counts(phase)
    if counts["reserved"]!=BUDGETS[phase] or counts["completed"]!=counts["reserved"]:
        raise ValueError("Budget incomplete")
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if max(differences)>5e-4 or rss>1024:raise ValueError("Cold prediction/resource check failed")
    write_new(d/"audit.json",dict(status="passed",cold_models=models,reused_control_states=models//2,new_fits=0,new_optimizers=0,
                         max_difference=max(differences),max_window_parameter_difference=max(window_differences),
                         peak_rss_mib=rss,counts=counts,input_sha256=sha(d/"inputs.npz"),manifest_sha256=sha(d/"manifest.json"),
                         source_recovery_sha256=sha(source_recovery) if source_recovery else None,
                         units_sha256={u:sha(d/u/"complete.json") for u in manifest["units"]}))
if __name__=="__main__":
    import argparse,torch
    p=argparse.ArgumentParser();p.add_argument("directory");p.add_argument("--source-recovery");a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);audit(a.directory,a.source_recovery)
