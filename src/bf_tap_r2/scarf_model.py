"""Fixed SCARF regression adaptation. Donor pools and all statistics are fit-part-only."""
from pathlib import Path
import json
import resource
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .data import FEATURES
from .v3_6_networks import NumericPreprocessor
from .v3_4_bags import group_safe_inner_folds
from .scarf_protocol import write_new,sha
TARGETS=("tap_iron","tap_time_len")

def feature_only(frame):
    if any(t in frame for t in TARGETS): raise ValueError("Predictor/preprocessor inputs must not contain targets")
    if not np.isfinite(frame[list(FEATURES)].to_numpy(float)).all(): raise ValueError("Nonfinite feature")
    return frame

class Preprocessor:
    def fit(self,frame):
        self.native=NumericPreprocessor(structure="raw_tabm").fit(feature_only(frame))
        self.mean=self.native.means_.copy(); self.std=self.native.stds_.copy()
        self.vocab=np.asarray(sorted(self.native.spout_to_index_),dtype=np.int64)
        return self
    def transform(self,frame):
        frame=feature_only(frame); raw=frame[list(FEATURES)].to_numpy(float)
        numeric=((raw-self.mean)/self.std).astype(np.float32)
        sp=frame.spout_no.to_numpy(int)
        cats=(sp[:,None]==self.vocab[None,:]).astype(np.float32)
        return np.concatenate([numeric,cats],axis=1)

def network(dim):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        layers=[]; previous=dim
        for _ in range(4): layers.extend([nn.Linear(previous,256),nn.ReLU()]); previous=256
        encoder=nn.Sequential(*layers)
        head=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,2))
        net=nn.Sequential(encoder,head)
        projection=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,256))
    return net,projection

def corrupt(batch,spouts,bank,bank_spouts,generator):
    n=len(batch); mask=torch.rand((n,21),generator=generator).argsort(1)<0
    columns=torch.rand((n,21),generator=generator).argsort(1)[:,:12]
    mask.scatter_(1,columns,True); out=batch.clone()
    for s in torch.unique(spouts):
        rows=torch.where(spouts==s)[0]; pool=torch.where(bank_spouts==s)[0]
        if not len(pool): raise ValueError("Donor category absent from fitting partition")
        donors=pool[torch.randint(len(pool),(len(rows),21),generator=generator)]
        replacements=bank[donors,torch.arange(21)[None,:]]
        out[rows,:21]=torch.where(mask[rows],replacements,batch[rows,:21])
    return out,mask

def info_nce(anchor,view,groups):
    logits=F.normalize(anchor,dim=1)@F.normalize(view,dim=1).T
    same=groups[:,None]==groups[None,:]; same.fill_diagonal_(False)
    logits=logits.masked_fill(same,-torch.inf)
    return F.cross_entropy(logits,torch.arange(len(logits)))

def inner_partition(frame):
    fold=group_safe_inner_folds(frame,seed=42)["fold"]
    return np.flatnonzero(fold!=0),np.flatnonzero(fold==0)

def save_state(path,net,projection,prep,ymean,ystd):
    arrays={"xmean":prep.mean,"xstd":prep.std,"vocab":prep.vocab,"ymean":ymean,"ystd":ystd}
    arrays.update({"net__"+k:v.detach().numpy() for k,v in net.state_dict().items()})
    arrays.update({"projection__"+k:v.detach().numpy() for k,v in projection.state_dict().items()})
    with Path(path).open("xb") as f: np.savez(f,**arrays)

def state_input(state,frame):
    frame=feature_only(frame)
    raw=frame[list(FEATURES)].to_numpy(float)
    numeric=((raw-state["xmean"])/state["xstd"]).astype(np.float32)
    cats=(frame.spout_no.to_numpy(int)[:,None]==state["vocab"][None,:]).astype(np.float32)
    return np.concatenate([numeric,cats],axis=1)

def predict_state(path,frame):
    # Functional forward: no network initialization, optimizer or training path.
    with np.load(path,allow_pickle=False) as s:
        x=torch.from_numpy(state_input(s,frame))
        with torch.no_grad():
            for prefix,indices in (("0.",(0,2,4,6)),("1.",(0,2))):
                for i in indices:
                    x=F.linear(x,torch.from_numpy(s[f"net__{prefix}{i}.weight"]),torch.from_numpy(s[f"net__{prefix}{i}.bias"]))
                    if prefix=="0." or i==0: x=F.relu(x)
            return x.numpy().astype(float)*s["ystd"]+s["ymean"]

def numpy_predict(path,frame):
    with np.load(path,allow_pickle=False) as s:
        x=state_input(s,frame)
        for prefix,indices in (("0.",(0,2,4,6)),("1.",(0,2))):
            for i in indices:
                x=x@s[f"net__{prefix}{i}.weight"].T+s[f"net__{prefix}{i}.bias"]
                if prefix=="0." or i==0: x=np.maximum(x,0)
        return x.astype(float)*s["ystd"]+s["ymean"]

def fit_state(frame,y,query,cal_y,*,arm,epochs,selector,path,ledger,phase,key):
    feature_only(frame); feature_only(query)
    if arm not in ("SUPERVISED_CONTROL","SCARF_REG"): raise ValueError("Unknown frozen arm")
    y=np.asarray(y,float)
    if y.shape!=(len(frame),2) or not np.isfinite(y).all(): raise ValueError("Two finite fitting targets required")
    groups=pd.util.hash_pandas_object(frame[list(FEATURES)],index=False).to_numpy()
    qgroups=pd.util.hash_pandas_object(query[list(FEATURES)],index=False).to_numpy()
    if set(groups)&set(qgroups): raise ValueError("Duplicate group leaks to query")
    if selector and (epochs!=240 or np.asarray(cal_y).shape!=(len(query),2)): raise ValueError("Full frozen selector required")
    if not 1<=epochs<=240: raise ValueError("Frozen epoch budget")
    ledger.reserve(phase,"state",key); started=time.perf_counter()
    prep=Preprocessor().fit(frame); x=torch.from_numpy(prep.transform(frame)); q=torch.from_numpy(prep.transform(query))
    ym=y.mean(0); ys=y.std(0); 
    if (ys<=0).any(): raise ValueError("Constant response")
    yt=torch.from_numpy(((y-ym)/ys).astype(np.float32)); cy=None if cal_y is None else torch.from_numpy(((cal_y-ym)/ys).astype(np.float32))
    net,projection=network(x.shape[1]); sp=torch.tensor(frame.spout_no.to_numpy(int)); gid=torch.tensor(pd.factorize(groups)[0],dtype=torch.long)
    ssl_trace=[]; supervised_trace=[]
    if arm=="SCARF_REG":
        okey=key+":ssl"; ledger.reserve(phase,"optimizer",okey)
        optimizer=torch.optim.Adam(list(net[0].parameters())+list(projection.parameters()),lr=.001)
        gen=torch.Generator().manual_seed(54191)
        for _ in range(100):
            order=torch.randperm(len(x),generator=gen); losses=[]
            for ids in order.split(128):
                view,_=corrupt(x[ids],sp[ids],x,sp,gen)
                loss=info_nce(projection(net[0](x[ids])),projection(net[0](view)),gid[ids])
                if not torch.isfinite(loss): raise ValueError("Nonfinite contrastive loss")
                optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            ssl_trace.append(float(np.mean(losses)))
        ledger.complete(phase,"optimizer",okey)
    okey=key+":supervised"; ledger.reserve(phase,"optimizer",okey)
    optimizer=torch.optim.Adam(net.parameters(),lr=.001); gen=torch.Generator().manual_seed(42)
    best=float("inf"); selected=epochs; best_state=None
    for epoch in range(1,epochs+1):
        order=torch.randperm(len(x),generator=gen); losses=[]
        for ids in order.split(128):
            loss=F.mse_loss(net(x[ids]),yt[ids])
            if not torch.isfinite(loss): raise ValueError("Nonfinite supervised loss")
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        if selector:
            with torch.no_grad(): score=float((net(q)-cy).abs().mean())
            supervised_trace.append(score)
            if score<best:
                best=score; selected=epoch; best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
        else: supervised_trace.append(float(np.mean(losses)))
    if selector: net.load_state_dict(best_state)
    ledger.complete(phase,"optimizer",okey)
    save_state(path,net,projection,prep,ym,ys)
    prediction=predict_state(path,query)
    if not np.isfinite(prediction).all(): raise ValueError("Nonfinite prediction")
    meta=dict(arm=arm,selector=selector,supervised_epochs=epochs,selected_epoch=selected,ssl_epochs=len(ssl_trace),ssl_trace=ssl_trace,
        supervised_trace=supervised_trace,fit_ids=frame.sample_id.astype(str).tolist(),query_ids=query.sample_id.astype(str).tolist(),
        fit_group_digest=sha_array(groups),state_sha256=sha(path),seconds=time.perf_counter()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    if meta["peak_rss_mib"]>1024: raise ValueError("Frozen single-worker RSS admission exceeded")
    write_new(Path(path).with_suffix(".json"),meta); ledger.complete(phase,"state",key)
    return prediction,meta

def sha_array(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
