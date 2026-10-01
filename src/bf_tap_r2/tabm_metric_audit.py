"""New-process native checkpoint cold audit, no optimizer or fit calls."""
from pathlib import Path
import json,resource
import numpy as np
import pandas as pd
import torch
from .component_regularization import ComponentRegressor
from .data import FEATURES
from .tabm_metric_protocol import sha,digest,verify_tree,write_new,Ledger,BUDGETS

def features(frame):return frame.drop(columns=[c for c in ("tap_iron","tap_time_len") if c in frame])
def audit(directory):
 d=Path(directory);manifest=json.loads((d/"manifest.json").read_text());x=np.load(d/"inputs.npz",allow_pickle=False)
 frame=pd.DataFrame(x["x"],columns=FEATURES);frame["spout_no"]=x["spouts"];frame["sample_id"]=x["ids"];y=x["y"];groups=pd.util.hash_pandas_object(frame[list(FEATURES)],index=False).to_numpy();diffs=[];models=0
 def forbidden(*a,**k):raise RuntimeError("Fit/optimizer forbidden in independent audit")
 torch.optim.AdamW=forbidden;torch.optim.Adam=forbidden;ComponentRegressor.fit=forbidden
 for unit in manifest["units"]:
  q=d/unit;c=json.loads((q/"complete.json").read_text());verify_tree(q,c["hashes"]);part=np.load(q/"partitions.npz");train,query,inner,cal=[part[k] for k in ("train","query","inner","cal")]
  for name,indices in zip(("train","query","inner","cal"),(train,query,inner,cal)):np.testing.assert_array_equal(indices,x[name+"__"+unit])
  if not np.array_equal(np.sort(np.r_[inner,cal]),np.sort(train)) or set(groups[train])&set(groups[query]) or set(groups[inner])&set(groups[cal]):raise ValueError("Partition leak")
  out=np.load(q/"predictions.npz");np.testing.assert_array_equal(out["ids"],x["ids"][query]);np.testing.assert_array_equal(out["cal_ids"],x["ids"][cal])
  for arm in ("BASE","TARGET_METRIC"):
   selected=None
   for stage,fit,held in [("selection",inner,cal),("refit",train,query)]:
    model=ComponentRegressor.load(q/arm/(stage+".pt"));saved=model.saved;t=saved["trace"]
    if saved["settings"]!=manifest["settings"] or saved["recipe"]!={"backbone":"tabm","frequency":.01} or saved["arm"]!="BASE" or saved["mechanisms"]!={"target_metric_weight":0. if arm=="BASE" else .02} or t["fit_ids_digest"]!=digest(x["ids"][fit].tolist()):raise ValueError("Native model fit identity differs")
    vocabulary={int(v):i+1 for i,v in enumerate(sorted(frame.iloc[fit].spout_no.unique()))}
    if {int(k):v for k,v in saved["preprocessing"]["spout_vocabulary"].items()}!=vocabulary or saved["preprocessing"]["n_spout_categories"]!=len(vocabulary)+1:raise ValueError("Vocabulary differs")
    raw=frame.iloc[fit][list(FEATURES)].to_numpy(float)
    for actual,expected in [(saved["preprocessing"]["means"],raw.mean(0)),(saved["preprocessing"]["stds"],raw.std(0)),(saved["mean"],y[fit].mean(0)),(saved["std"],y[fit].std(0))]:np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-12)
    if stage=="selection":
     best=float("inf");selected=0;stale=0
     for row in t["history"]:
      if stale>=manifest["settings"]["patience"]:raise ValueError("Extra selector epochs")
      if row["validation_mae"]<best-manifest["settings"]["min_delta"]:best=row["validation_mae"];selected=row["epoch"];stale=0
      else:stale+=1
     if selected!=t["selected_epoch"] or len(t["history"])!=t["stopped_epoch"] or len(t["history"])>manifest["settings"]["max_epochs"]:raise ValueError("Selected epoch differs")
     vx,vc=model._inputs(features(frame.iloc[held]));yt=torch.as_tensor((y[held]-model.mean_)/model.std_,dtype=torch.float32)
     with torch.no_grad():actual=float((model.model_(vx,vc).mean(1)-yt).abs().mean())
     if abs(actual-best)>1e-6:raise ValueError("Checkpoint differs from chosen epoch")
    elif t["selected_epoch"]!=selected or t["stopped_epoch"]!=selected or len(t["history"])!=selected:raise ValueError("Fresh refit epoch differs")
    f=features(frame.iloc[held]);b=model.predict(f)[:,0];reverse=model.predict(f.iloc[::-1])[:,0][::-1];chunk=np.concatenate([model.predict(f.iloc[i:i+37])[:,0] for i in range(0,len(f),37)])
    diffs.extend([float(abs(b-reverse).max()),float(abs(b-chunk).max())])
    if stage=="refit":
     diffs.append(float(abs(b-out[arm]).max()))
     if manifest["phase"]=="engineering" and np.mean(abs(b-y[query,0]))>=np.mean(abs(np.median(y[train,0])-y[query,0])):raise ValueError("Production synthetic learnability failed")
    models+=1;del model
 counts=Ledger(d.parent/"ledger").counts(manifest["phase"])
 if counts["reserved"]!=BUDGETS[manifest["phase"]] or counts["completed"]!=counts["reserved"]:raise ValueError("Budget incomplete")
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 if max(diffs)>5e-4 or rss>1024:raise ValueError("Cold/resource admission failed")
 write_new(d/"audit.json",dict(status="passed",cold_models=models,max_difference=max(diffs),peak_rss_mib=rss,counts=counts,new_fits=0,input_sha256=sha(d/"inputs.npz"),manifest_sha256=sha(d/"manifest.json"),units_sha256={u:sha(d/u/"complete.json") for u in manifest["units"]}))
if __name__=="__main__":
 import sys
 torch.set_num_threads(1);torch.set_num_interop_threads(1);audit(sys.argv[1])
