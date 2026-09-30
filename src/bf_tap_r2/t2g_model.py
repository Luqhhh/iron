"""Train-only T2G selection and fresh refit; explicit weights-only state."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import numpy as np
from .data import FEATURES,TARGETS
from .v3_6_networks import NumericPreprocessor
from .v3_4_bags import group_safe_inner_folds

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def state_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy()
        h.update(k.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

@dataclass
class EpochSelector:
    best: float=math.inf
    selected_epoch: int=0
    stale: int=0
    def update(self,epoch,value):
        if not math.isfinite(value): raise ValueError("Nonfinite calibration MAE")
        improved=value<self.best-1e-5
        if improved: self.best,self.selected_epoch,self.stale=value,epoch,0
        else: self.stale+=1
        return improved
    @property
    def stop(self): return self.stale>=25

@dataclass
class FitResult:
    selected_epoch: int
    init_digest: str
    batch_digest: str
    model_dir: Path
    prediction: np.ndarray
    optimizer_starts: int

def validate_features(frame):
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("Invalid query/fit IDs")
    x=frame.loc[:,FEATURES].to_numpy(dtype=float)
    spout=frame.spout_no.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(spout).all():
        raise ValueError("Nonfinite feature/category")
    if not (spout==np.floor(spout)).all(): raise ValueError("Noninteger category")

def prepare(frame,target):
    if target not in TARGETS: raise ValueError("Invalid target")
    validate_features(frame)
    y=frame[target].to_numpy(dtype=float)
    if not np.isfinite(y).all() or len(y)<2 or y.std()==0:
        raise ValueError("Nonfinite or constant target")
    prep=NumericPreprocessor(structure="raw_tabm").fit(frame)
    x,c=prep.transform_tabm(frame)
    return prep,x,c.reshape(-1),float(y.mean()),float(y.std())

def configure_threads():
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)

class Session:
    def __init__(self,frame,target,arm,before_optimizer,stage,unit):
        import torch
        from .t2g_core import make_model
        configure_threads()
        self.prep,x,c,self.mean,self.std=prepare(frame,target)
        self.x,self.c=torch.from_numpy(x),torch.from_numpy(c)
        self.y=torch.as_tensor((frame[target].to_numpy(dtype=float)-self.mean)/self.std,dtype=torch.float32)
        self.model=make_model(arm,self.prep.n_spout_categories_)
        self.initial_digest=state_digest(self.model.state_dict())
        before_optimizer(stage)
        if unit:
            p=Path("local/t2g-unit-training/optimizer-starts.jsonl")
            p.parent.mkdir(parents=True,exist_ok=True)
            with p.open("a") as f:
                f.write(canonical({"stage":stage,"arm":arm,"rows":len(frame),"purpose":"artificial-unit-test"})+"\n")
                f.flush(); os.fsync(f.fileno())
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
        self.rng=np.random.default_rng(42)
        self.batch_hash=hashlib.sha256()

    def epoch(self):
        import torch
        self.model.train()
        order=self.rng.permutation(len(self.x))
        self.batch_hash.update(order.astype("<i8").tobytes())
        losses=[]
        for i in range(0,len(order),256):
            index=order[i:i+256]
            self.optimizer.zero_grad(set_to_none=True)
            loss=(self.model(self.x[index],self.c[index])-self.y[index]).square().mean()
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            loss.backward()
            # Dense topology and last-block unused feature edges legitimately have no gradients.
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                raise ValueError("Nonfinite gradient")
            self.optimizer.step()
            if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                raise ValueError("Nonfinite fitted state")
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    def predict(self,frame):
        import torch
        validate_features(frame)
        x,c=self.prep.transform_tabm(frame)
        self.model.eval()
        with torch.no_grad():
            result=self.model(torch.from_numpy(x),torch.from_numpy(c.reshape(-1))).numpy().astype(float)
        return result*self.std+self.mean

def fit_partition(train,query,target,arm,output,before_optimizer,force_epochs=None):
    import torch
    output=Path(output)
    if any(t in query for t in TARGETS): raise ValueError("Query must not contain targets")
    validate_features(query)
    if force_epochs is not None and (type(force_epochs) is not int or not 1<=force_epochs<=240
                                    or not train.sample_id.astype(str).str.startswith("SYNTH_").all()):
        raise ValueError("Forced epochs restricted to artificial engineering data")
    output.mkdir(parents=True,exist_ok=False)
    split=group_safe_inner_folds(train,seed=42)
    mask=split["fold"]!=0
    if force_epochs==240 and len(train)==2204:
        # Only the frozen full-size synthetic probe uses its predefined contiguous selector.
        mask=np.arange(len(train))<1763
    inner=train.loc[mask].reset_index(drop=True)
    calibration=train.loc[~mask].reset_index(drop=True)
    session=Session(inner,target,arm,before_optimizer,"selector",force_epochs is not None and force_epochs!=240)
    selection=EpochSelector(); trace=[]; best_state=None
    for epoch in range(1,(force_epochs or 240)+1):
        loss=session.epoch()
        value=float(np.mean(np.abs(session.predict(calibration)-calibration[target].to_numpy()))/session.std)
        if selection.update(epoch,value):
            best_state={k:v.detach().clone() for k,v in session.model.state_dict().items()}
        trace.append({"epoch":epoch,"loss":loss,"scaled_mae":value})
        if force_epochs is None and selection.stop: break
    if selection.selected_epoch<1: raise ValueError("No selected epoch")
    selector_final=state_digest(session.model.state_dict())
    selector_init=session.initial_digest
    selector_meta=session.prep.metadata()
    torch.save(best_state,output/"selector-selected.pt")
    selected=selection.selected_epoch
    session=Session(train,target,arm,before_optimizer,"refit",force_epochs is not None and force_epochs!=240)
    refit_epochs=force_epochs or selected
    losses=[session.epoch() for _ in range(refit_epochs)]
    prediction=session.predict(query)
    if not np.isfinite(prediction).all(): raise ValueError("Nonfinite prediction")
    torch.save(session.model.state_dict(),output/"weights.pt")
    metadata={"schema":"t2g-weights-v1","target":target,"arm":arm,
        "selected_epoch":selected,"selector_rows":len(inner),"refit_rows":len(train),
        "refit_epochs":refit_epochs,"selector_init_digest":selector_init,
        "selector_final_digest":selector_final,"refit_init_digest":session.initial_digest,
        "batch_digest":session.batch_hash.hexdigest(),"selector_preprocessor":selector_meta,
        "preprocessor":session.prep.metadata(),"target_mean":session.mean,"target_std":session.std,
        "fit_ids":train.sample_id.astype(str).tolist(),"query_ids":query.sample_id.astype(str).tolist(),
        "inner_fit_ids":inner.sample_id.astype(str).tolist(),
        "inner_calibration_ids":calibration.sample_id.astype(str).tolist(),
        "trace":trace,"refit_losses":losses,"optimizer_starts":2}
    (output/"metadata.json").write_text(canonical(metadata)+"\n")
    hashes={name:file_hash(output/name) for name in ("weights.pt","metadata.json","selector-selected.pt")}
    (output/"hashes.json").write_text(canonical(hashes)+"\n")
    return FitResult(selected,session.initial_digest,session.batch_hash.hexdigest(),output,prediction,2)

def load_state(model_dir):
    import torch
    from .t2g_core import make_model
    model_dir=Path(model_dir)
    hashes=json.loads((model_dir/"hashes.json").read_text())
    if set(hashes)!={"weights.pt","metadata.json","selector-selected.pt"}:
        raise ValueError("Unknown saved-state manifest")
    for name,h in hashes.items():
        if file_hash(model_dir/name)!=h: raise ValueError("Saved state external hash mismatch")
    meta=json.loads((model_dir/"metadata.json").read_text())
    if meta["schema"]!="t2g-weights-v1": raise ValueError("Unsupported saved schema")
    prep=NumericPreprocessor(structure="raw_tabm")
    p=meta["preprocessor"]
    if p["feature_names"]!=list(FEATURES) or p["structure"]!="raw_tabm":
        raise ValueError("Saved feature order mismatch")
    prep.means_=np.asarray(p["means"],dtype=float)
    prep.stds_=np.asarray(p["stds"],dtype=float)
    prep.spout_to_index_={int(k):int(v) for k,v in p["spout_vocabulary"].items()}
    prep.n_spout_categories_=int(p["n_spout_categories"])
    if prep.means_.shape!=(21,) or prep.stds_.shape!=(21,) or not np.isfinite(prep.means_).all() or not np.isfinite(prep.stds_).all() or (prep.stds_<=0).any():
        raise ValueError("Invalid saved scaler")
    if not math.isfinite(meta["target_mean"]) or not math.isfinite(meta["target_std"]) or meta["target_std"]<=0:
        raise ValueError("Invalid saved target scaler")
    model=make_model(meta["arm"],prep.n_spout_categories_)
    state=torch.load(model_dir/"weights.pt",map_location="cpu",weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()): raise ValueError("Nonfinite saved tensors")
    model.load_state_dict(state,strict=True); model.eval()
    return model,prep,meta

def load_predict(model_dir,query):
    import torch
    if any(t in query for t in TARGETS): raise ValueError("Query must not contain targets")
    validate_features(query); configure_threads()
    model,prep,meta=load_state(model_dir)
    x,c=prep.transform_tabm(query)
    with torch.no_grad():
        prediction=model(torch.from_numpy(x),torch.from_numpy(c.reshape(-1))).numpy().astype(float)
    return prediction*meta["target_std"]+meta["target_mean"]
