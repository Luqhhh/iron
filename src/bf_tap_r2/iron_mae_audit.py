"""Independent cold model/statistics/partition audit. This module never fits a model."""
from pathlib import Path
import json
import resource
import numpy as np
import pandas as pd
import torch
from .data import FEATURES
from .iron_mae_protocol import sha,write_new,Ledger,ARMS,BUDGETS
from .iron_mae_model import predict_state,numpy_predict,TARGETS

def audit_phase(directory):
    directory=Path(directory); manifest=json.loads((directory/"manifest.json").read_text())
    data=np.load(directory/"inputs.npz",allow_pickle=False)
    frame=pd.DataFrame(data["x"],columns=FEATURES);frame["spout_no"]=data["spouts"];frame["sample_id"]=data["ids"].astype(str)
    y=data["y"]; all_groups=pd.util.hash_pandas_object(frame[list(FEATURES)],index=False).to_numpy()
    differences=[]; models=0
    # No optimizer initialization or model fitting is permitted in this process.
    def forbidden(*args,**kwargs): raise RuntimeError("Optimizer forbidden in cold audit")
    torch.optim.Adam=forbidden; torch.optim.AdamW=forbidden
    for unit in manifest["units"]:
        d=directory/unit; meta=json.loads((d/"complete.json").read_text())
        verify=meta["hashes"]
        for n,h in verify.items():
            if sha(d/n)!=h: raise ValueError("Model artifact changed")
        ix=data[f"query__{unit}"]; train=data[f"train__{unit}"]; inner=data[f"inner__{unit}"]; cal=data[f"cal__{unit}"]
        if not np.array_equal(np.sort(np.concatenate([inner,cal])),np.sort(train)) or set(all_groups[train])&set(all_groups[ix]) or set(all_groups[inner])&set(all_groups[cal]): raise ValueError("Partition/group leak")
        predictions=np.load(d/"predictions.npz",allow_pickle=False)
        if not np.array_equal(predictions["ids"],data["ids"][ix]): raise ValueError("Query ordering differs")
        for arm in ARMS:
            sm=json.loads((d/(arm+"-selector.json")).read_text()); rm=json.loads((d/(arm+"-refit.json")).read_text())
            selected=int(np.argmin(sm["supervised_trace"])+1)
            if sm["supervised_epochs"]!=240 or len(sm["supervised_trace"])!=240 or sm["selected_epoch"]!=selected or rm["supervised_epochs"]!=selected or rm["selected_epoch"]!=selected: raise ValueError("Selector/refit contract differs")
            for stage,fit,q,record in (("selector",inner,cal,sm),("refit",train,ix,rm)):
                expected_ssl=0
                if record["ssl_epochs"]!=expected_ssl or len(record["ssl_trace"])!=expected_ssl or record["arm"]!=arm or record["selector"]!=(stage=="selector"): raise ValueError("Frozen arm changed")
                path=d/(arm+"-"+stage+".npz")
                if sha(path)!=record["state_sha256"] or record["fit_ids"]!=data["ids"][fit].tolist() or record["query_ids"]!=data["ids"][q].tolist(): raise ValueError("Fit/query state identity differs")
                with np.load(path,allow_pickle=False) as state:
                    if not np.allclose(state["xmean"],frame.iloc[fit][list(FEATURES)].to_numpy(float).mean(0),rtol=0,atol=1e-12) or not np.allclose(state["xstd"],frame.iloc[fit][list(FEATURES)].to_numpy(float).std(0),rtol=0,atol=1e-12): raise ValueError("Input statistics not training-only")
                    if not np.array_equal(state["vocab"],np.unique(data["spouts"][fit])) or not np.allclose(state["ymean"],y[fit].mean(0),rtol=0,atol=1e-12) or not np.allclose(state["ystd"],y[fit].std(0),rtol=0,atol=1e-12): raise ValueError("Response/category statistics differ")
                query=frame.iloc[q]; pred=predict_state(path,query)
                checks=[numpy_predict(path,query),predict_state(path,query.iloc[::-1])[::-1],np.concatenate([predict_state(path,query.iloc[i:i+1]) for i in range(len(query))]),
                    np.concatenate([predict_state(path,query.iloc[i:i+37]) for i in range(0,len(query),37)]),predictions[arm+"-"+stage]]
                maximum=max(float(np.max(np.abs(pred-v))) for v in checks)
                if not np.isfinite(pred).all() or maximum>5e-4: raise ValueError("Cold functional/NumPy/batch invariance failed")
                if manifest["phase"]=="engineering":
                    constant=np.median(y[fit],axis=0)
                    if not float(np.abs(pred-y[q]).mean())<float(np.abs(constant-y[q]).mean()): raise ValueError("Synthetic learnability failed")
                differences.append(maximum);models+=1
    ledger=Ledger(directory.parent/"ledger"); counts=ledger.counts(manifest["phase"])
    if counts["reserved"]!=BUDGETS[manifest["phase"]] or counts["completed"]!=counts["reserved"]: raise ValueError("Phase budget/coverage incomplete")
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if rss>1024: raise ValueError("Cold resource admission exceeded")
    write_new(directory/"audit.json",dict(status="passed",phase=manifest["phase"],cold_models=models,maximum_difference=max(differences),peak_rss_mib=rss,
        counts=counts,new_optimizer_runs=0,new_reference_fits=0,inputs_sha256=sha(directory/"inputs.npz"),manifest_sha256=sha(directory/"manifest.json"),
        units_sha256={u:sha(directory/u/"complete.json") for u in manifest["units"]}))

if __name__=="__main__":
    import sys
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    audit_phase(sys.argv[1])
