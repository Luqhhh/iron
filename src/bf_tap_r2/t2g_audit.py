"""Fresh-process cold audit, with a second functional forward implementation."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from .data import FEATURES,TARGETS
from .t2g_model import canonical,file_hash,predict_loaded,load_state,state_digest
from .v3_4_bags import group_safe_inner_folds

def compare_predictions(expected,observed,exact=False):
    e,o=np.asarray(expected,float),np.asarray(observed,float)
    if e.shape!=o.shape or not np.isfinite(e).all() or not np.isfinite(o).all():
        raise ValueError("Prediction shape/finite mismatch")
    if exact and not np.array_equal(e,o): raise ValueError("Full-batch warm/cold mismatch")
    delta=float(np.max(np.abs(e-o),initial=0))
    relative=delta/max(1.,float(np.max(np.abs(e),initial=0)))
    if delta>5e-4 or relative>1e-6: raise ValueError("Cold absolute/relative tolerance exceeded")
    return {"absolute":delta,"relative":relative}

def functional_forward(state,numeric,category,arm):
    import torch
    from torch.nn import functional as f
    if arm not in ("LEARNED_GRAPH","DENSE_CONTROL"): raise ValueError("Unknown arm")
    with torch.no_grad():
        xnum=torch.as_tensor(numeric,dtype=torch.float32)
        cat=torch.as_tensor(category,dtype=torch.int64)
        x=torch.cat((state["readout"].expand(len(cat),1,64),
            xnum.unsqueeze(-1)*state["numeric_weight"]+state["numeric_bias"],
            (state["category.weight"][cat]+state["category_bias"]).unsqueeze(1)),1)
        def norm(z,prefix):
            return f.layer_norm(z,(64,),state[prefix+".weight"],state[prefix+".bias"],1e-5)
        def linear(z,prefix): return f.linear(z,state[prefix+".weight"],state[prefix+".bias"])
        for layer in (0,1):
            b=f"blocks.{layer}"; a=b+".attention"; n=len(cat); k=x.shape[1]
            z=norm(x,b+".norm_attention")
            q=linear(z,a+".q").reshape(n,k,4,16).transpose(1,2)
            v=linear(z,a+".v").reshape(n,k,4,16).transpose(1,2)
            query=q[:,:,:1] if layer==1 else q
            logits=(query*state[a+".relation"][None,:,None,:])@q.transpose(-1,-2)/4
            h=f.normalize(state[a+".column_head"],dim=-1,eps=1e-12)
            t=f.normalize(state[a+".column_tail"],dim=-1,eps=1e-12)
            feature=(torch.sigmoid(h@t.transpose(-1,-2)+state[a+".topology_bias"])>.5).float()
            if arm=="DENSE_CONTROL": feature=torch.ones_like(feature)
            for i in range(22): feature[:,i,i]=1
            mask=torch.zeros(4,23,23); mask[:,0,1:]=1; mask[:,1:,1:]=feature
            if layer==1: mask=mask[:,:1]
            attention=torch.softmax(logits-10000*(1-mask[None]),-1)
            mixed=(attention@v).transpose(1,2).reshape(n,1 if layer==1 else k,64)
            residual=x[:,:1] if layer==1 else x
            x=residual+linear(mixed,a+".out")
            z=linear(norm(x,b+".norm_ffn"),b+".ffn_in")
            value,gate=z.chunk(2,-1)
            x=x+linear(value*f.relu(gate),b+".ffn_out")
        return linear(f.relu(norm(x[:,0],"norm")),"head").squeeze(-1).numpy()

def functional_predict(state,preprocessor,mean,std,query,arm):
    x=query.loc[:,FEATURES].to_numpy(float)
    x=((x-np.asarray(preprocessor["means"]))/np.asarray(preprocessor["stds"])).astype(np.float32)
    vocab={int(k):int(v) for k,v in preprocessor["spout_vocabulary"].items()}
    cat=np.array([vocab.get(int(v),0) for v in query.spout_no],dtype=np.int64)
    return functional_forward(state,x,cat,arm).astype(float)*std+mean

def audit_unit(directory,record,frame,folds,manifest_digest):
    import torch
    from .t2g_model import EpochSelector
    from .t2g_core import make_model
    directory=Path(directory); key=record["key"]
    if file_hash(directory/"complete.json")!=record["complete_sha256"]:
        raise ValueError("Unit external complete anchor differs")
    complete=json.loads((directory/"complete.json").read_text())
    if complete["key"]!=key or complete["manifest_digest"]!=manifest_digest:
        raise ValueError("Unit phase identity differs")
    expected_files={"weights.pt","metadata.json","selector-selected.pt","hashes.json","predictions.npz"}
    if set(complete["hashes"])!=expected_files: raise ValueError("Unit artifact schema differs")
    for name,sha in complete["hashes"].items():
        if file_hash(directory/name)!=sha: raise ValueError("Unit artifact differs")
    mask=folds[key["seed"]]==key["fold"]
    train=frame.loc[~mask].reset_index(drop=True)
    query=frame.loc[mask,["sample_id","spout_no",*FEATURES]].reset_index(drop=True)
    ids=query.sample_id.astype(str).tolist()
    if complete["fit_ids"]!=train.sample_id.astype(str).tolist() or complete["query_ids"]!=ids:
        raise ValueError("Saved fit/query IDs differ")
    with np.load(directory/"predictions.npz",allow_pickle=False) as p:
        if set(p.files)!={"query_ids","prediction"} or p["query_ids"].tolist()!=ids:
            raise ValueError("Query prediction IDs/schema differ")
        warm=p["prediction"].copy()
    model,prep,meta=load_state(directory)
    if meta["target"]!=key["target"] or meta["arm"]!=key["arm"]:
        raise ValueError("Saved target/arm differs")
    y=train[key["target"]].to_numpy(float)
    means=train.loc[:,FEATURES].to_numpy(float).mean(0)
    stds=train.loc[:,FEATURES].to_numpy(float).std(0)
    if (not np.array_equal(prep.means_,means) or not np.array_equal(prep.stds_,stds)
        or meta["target_mean"]!=float(y.mean()) or meta["target_std"]!=float(y.std())
        or meta["fit_ids"]!=complete["fit_ids"] or meta["query_ids"]!=ids):
        raise ValueError("Outer transformations/partition identity differ")
    inner=group_safe_inner_folds(train,seed=42)["fold"]!=0
    fitting=train.loc[inner]; calibration=train.loc[~inner,["sample_id","spout_no",*FEATURES]]
    if (meta["inner_fit_ids"]!=fitting.sample_id.astype(str).tolist()
        or meta["inner_calibration_ids"]!=calibration.sample_id.astype(str).tolist()):
        raise ValueError("Selector partition differs")
    ip=meta["selector_preprocessor"]
    if (not np.array_equal(np.asarray(ip["means"]),fitting.loc[:,FEATURES].to_numpy(float).mean(0))
        or not np.array_equal(np.asarray(ip["stds"]),fitting.loc[:,FEATURES].to_numpy(float).std(0))):
        raise ValueError("Selector preprocessing differs")
    expected_vocab={str(v):i for i,v in enumerate(sorted(set(train.spout_no.astype(int))),1)}
    inner_vocab={str(v):i for i,v in enumerate(sorted(set(fitting.spout_no.astype(int))),1)}
    if meta["preprocessor"]["spout_vocabulary"]!=expected_vocab or ip["spout_vocabulary"]!=inner_vocab:
        raise ValueError("Train-only vocabulary differs")
    s=EpochSelector()
    trace=meta["trace"]
    if not trace or len(trace)>240: raise ValueError("Selector epoch budget differs")
    for epoch,row in enumerate(trace,1):
        if row["epoch"]!=epoch: raise ValueError("Noncontiguous epoch trace")
        s.update(epoch,row["scaled_mae"])
        if s.stop and epoch!=len(trace): raise ValueError("Ignored early stopping")
    if (meta["selected_epoch"]!=s.selected_epoch or meta["refit_epochs"]!=s.selected_epoch
        or len(meta["refit_losses"])!=s.selected_epoch or meta["optimizer_starts"]!=2):
        raise ValueError("Selected epoch/fresh-refit identity differs")
    if len(trace)<240 and not s.stop: raise ValueError("Unjustified selector stop")
    expected_init=make_model(key["arm"],prep.n_spout_categories_)
    if state_digest(expected_init.state_dict())!=meta["refit_init_digest"]:
        raise ValueError("Fresh refit initialization identity differs")
    rng=np.random.default_rng(42); h=hashlib.sha256()
    for _ in range(s.selected_epoch): h.update(rng.permutation(len(train)).astype("<i8").tobytes())
    if h.hexdigest()!=meta["batch_digest"]: raise ValueError("Frozen refit batch identity differs")
    cold=predict_loaded(model,prep,meta,query); compare_predictions(warm,cold,exact=True)
    deltas=[]
    variants=[predict_loaded(model,prep,meta,query.iloc[::-1])[::-1],
        np.concatenate([predict_loaded(model,prep,meta,query.iloc[i:i+37]) for i in range(0,len(query),37)]),
        np.concatenate([predict_loaded(model,prep,meta,query.iloc[i:i+1]) for i in range(len(query))])]
    for variant in variants: deltas.append(compare_predictions(warm,variant))
    independent=functional_predict(model.state_dict(),meta["preprocessor"],meta["target_mean"],meta["target_std"],query,key["arm"])
    deltas.append(compare_predictions(warm,independent))
    selected_state=torch.load(directory/"selector-selected.pt",map_location="cpu",weights_only=True)
    iy=fitting[key["target"]].to_numpy(float)
    selected_prediction=functional_predict(selected_state,ip,float(iy.mean()),float(iy.std()),calibration,key["arm"])
    metric=float(np.abs(selected_prediction-train.loc[~inner,key["target"]].to_numpy(float)).mean()/iy.std())
    if abs(metric-s.best)>1e-6: raise ValueError("Saved selector checkpoint metric differs")
    return {"key":key,"status":"passed","cold_full_difference":0.,"variants":deltas,"optimizer_starts":0}

def audit_phase(manifest,output):
    from .t2g_freeze import verify_manifest
    from .t2g_run import validate_completion,write_new
    from .t2g_reference import verify_reference
    output=Path(output); m=verify_manifest(Path(manifest))
    if (output/"audit.json").exists(): raise FileExistsError("Audit already exists; verify existing evidence")
    phase=json.loads((output/"phase.json").read_text())
    if phase["manifest_digest"]!=file_hash(manifest): raise ValueError("Phase manifest differs")
    reference=verify_reference(m["reference"]["native_root"],m["reference"]["overlay_root"])
    if reference.identity!=m["reference"]: raise ValueError("Current reference identity differs")
    complete=json.loads((output/"complete.json").read_text())
    validate_completion(phase["phase"],phase["selected_targets"],complete["records"])
    reports=[audit_unit(output/r["directory"],r,reference.frame,reference.folds,phase["manifest_digest"])
             for r in complete["records"]]
    paired={}
    for r in complete["records"]:
        key=r["key"]; meta=json.loads((output/r["directory"]/"metadata.json").read_text())
        pair=(key["target"],key["seed"],key["fold"])
        values=(meta["refit_init_digest"],meta["selector_init_digest"])
        if pair in paired and paired[pair]!=values: raise ValueError("Paired initialization differs")
        paired[pair]=values
    report={"status":"passed","phase":phase["phase"],"complete_sha256":file_hash(output/"complete.json"),
        "manifest_digest":phase["manifest_digest"],"units":reports,"audit_optimizer_starts":0}
    write_new(output/"audit.json",report)
    write_new(output/"audit.complete.json",{"audit_sha256":file_hash(output/"audit.json")})
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); print(canonical(audit_phase(a.manifest,a.output)))

if __name__=="__main__": main()
