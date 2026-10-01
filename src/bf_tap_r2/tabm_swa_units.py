"""Exactly one candidate pair per unit, with original audited BASE identity reuse."""
from pathlib import Path
import json,resource
import numpy as np
import pandas as pd
from .component_regularization import ComponentRegressor
from .tabm_metric_models import MetricRegressor
from .tabm_swa_model import SWARegressor
from .tabm_swa_protocol import sha,digest,write_new,verify_tree
from .tabm_swa_reference import controls
from .data import FEATURES,TARGETS
from .v3_4_bags import group_safe_inner_folds

def features(frame):return frame.drop(columns=[c for c in TARGETS if c in frame])
def reuse(root,phase,unit,frame,indices,settings):
 catalogue,_=controls(root);old=Path(root)/"local/runs/tabm-target-metric-v1"/(phase+"-r1");saved=np.load(old/"inputs.npz",allow_pickle=False)
 for key,actual in (("x",frame[list(FEATURES)].to_numpy(float)),("spouts",frame.spout_no.to_numpy(int)),("ids",frame.sample_id.astype(str).to_numpy(dtype=str)),("y",frame[list(TARGETS)].to_numpy(float))):np.testing.assert_array_equal(saved[key],actual)
 for key,ix in indices.items():np.testing.assert_array_equal(saved[key+"__"+unit],ix)
 source=old/unit/"BASE";records=[r for r in catalogue["records"] if r["phase"]==phase and r["unit"]==unit]
 if len(records)!=2:raise ValueError("Control source identity incomplete")
 for r in records:
  p=source/(r["stage"]+".pt")
  if str(source.relative_to(root))!=r["source_directory"] or sha(p)!=r["source_sha256"]:raise ValueError("Reused BASE state differs")
  model=ComponentRegressor.load(p)
  if model.saved["settings"]!=settings or model.saved["recipe"]!={"backbone":"tabm","frequency":.01} or model.saved["mechanisms"]!={"target_metric_weight":0.}:raise ValueError("BASE recipe differs")
  del model
 cache=np.load(Path(root)/"local/tabm-swa-20261001"/(phase+"-BASE-time-r1.npz"),allow_pickle=False);key=unit+"__refit";np.testing.assert_array_equal(cache[key+"__ids"],frame.iloc[indices["query"]].sample_id.astype(str).to_numpy(dtype=str))
 return cache[key].copy(),dict(source_directory=str(source.relative_to(root)),records=records,old_inputs_sha256=sha(old/"inputs.npz"),old_audit_sha256=sha(old/"audit.json"),new_fits=0)

def train_unit(root,frame,train,query,directory,spec,settings,ledger,phase,unit):
 d=Path(directory);d.mkdir(exist_ok=False);training=frame.iloc[train].reset_index(drop=True);iv=np.asarray(group_safe_inner_folds(training,seed=settings["inner_seed"])["fold"]);inner=train[iv!=0];cal=train[iv==0]
 indices=dict(train=train,query=query,inner=inner,cal=cal);groups=pd.util.hash_pandas_object(frame[list(FEATURES)],index=False).to_numpy()
 if set(groups[train])&set(groups[query]) or set(groups[inner])&set(groups[cal]):raise ValueError("Duplicate group leak")
 pred={"ids":frame.iloc[query].sample_id.astype(str).to_numpy(dtype=str),"cal_ids":frame.iloc[cal].sample_id.astype(str).to_numpy(dtype=str)}
 if phase in ("engineering","development"):
  pred["BASE"],receipt=reuse(root,phase,unit,frame,indices,settings);write_new(d/"BASE-reuse.json",receipt)
 else:
  ad=d/"BASE";ad.mkdir();model=MetricRegressor(settings,ad,ledger,phase,unit,"BASE").fit(features(training),training[list(TARGETS)].to_numpy(float));pred["BASE"]=model.predict(features(frame.iloc[query]))[:,1];write_new(ad/"metadata.json",model.metadata_);del model
 ad=d/"SWA_WINDOW10";ad.mkdir();model=SWARegressor(settings,ad,ledger,phase,unit).fit(features(training),training[list(TARGETS)].to_numpy(float));pred["SWA_WINDOW10"]=model.predict(features(frame.iloc[query]))[:,1];write_new(ad/"metadata.json",model.metadata_);del model
 with (d/"predictions.npz").open("xb") as f:np.savez(f,**pred)
 with (d/"partitions.npz").open("xb") as f:np.savez(f,**indices)
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 if rss>1024:raise ValueError("Worker memory exceeded")
 write_new(d/"complete.json",dict(phase=phase,unit=unit,peak_rss_mib=rss,hashes={str(p.relative_to(d)):sha(p) for p in d.rglob("*") if p.is_file()}))
