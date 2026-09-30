"""Cold inference tests use artificial tensors, never official fits."""
import numpy as np
import pandas as pd
import pytest
torch=pytest.importorskip("torch")
from bf_tap_r2.data import FEATURES
from bf_tap_r2.t2g_core import make_model
from bf_tap_r2.t2g_audit import functional_forward,compare_predictions
from bf_tap_r2.t2g_model import load_predict,fit_partition

@pytest.mark.parametrize("arm",["LEARNED_GRAPH","DENSE_CONTROL"])
def test_independent_forward(arm):
    model=make_model(arm,3).eval()
    torch.set_num_threads(1)
    x=torch.randn(7,21); c=torch.arange(7)%3
    with torch.no_grad(): expected=model(x,c).numpy()
    got=functional_forward(model.state_dict(),x.numpy(),c.numpy(),arm)
    compare_predictions(expected,got,exact=False)

def test_tolerances_require_both_bounds():
    compare_predictions(np.array([1000.]),np.array([1000.0004]),exact=False)
    with pytest.raises(ValueError): compare_predictions(np.array([.1]),np.array([.1001]),exact=False)
    with pytest.raises(ValueError): compare_predictions(np.array([1000.]),np.array([1000.0006]),exact=False)
    with pytest.raises(ValueError): compare_predictions(np.array([1.]),np.array([1.+1e-7]),exact=True)

@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    rng=np.random.default_rng(919)
    f=pd.DataFrame(rng.normal(size=(40,21)),columns=FEATURES)
    f["sample_id"]=[f"SYNTH_{i}" for i in range(40)]; f["spout_no"]=np.arange(40)%2+1
    f["tap_iron"]=100+f.air_volume
    q=f.iloc[:8].drop(columns="tap_iron")
    r=fit_partition(f,q,"tap_iron","LEARNED_GRAPH",tmp_path_factory.mktemp("audit")/"model",lambda s:None,force_epochs=1)
    return q,r

def test_cold_batch_orders(saved):
    q,r=saved
    assert np.array_equal(r.prediction,load_predict(r.model_dir,q))
    for got in (load_predict(r.model_dir,q.iloc[::-1])[::-1],
                np.concatenate([load_predict(r.model_dir,q.iloc[i:i+3]) for i in range(0,len(q),3)]),
                np.concatenate([load_predict(r.model_dir,q.iloc[i:i+1]) for i in range(len(q))])):
        compare_predictions(r.prediction,got,exact=False)

def test_tensor_hash_tamper(saved,tmp_path):
    import shutil
    q,r=saved; p=tmp_path/"tampered"; shutil.copytree(r.model_dir,p)
    (p/"weights.pt").write_bytes(b"substitution")
    with pytest.raises(ValueError): load_predict(p,q)

def test_nan_prediction_rejected():
    with pytest.raises(ValueError): compare_predictions(np.array([1.]),np.array([np.nan]),exact=False)

def test_loaded_model_inference_does_not_reinitialize(saved,monkeypatch):
    from bf_tap_r2.t2g_model import load_state,predict_loaded,state_digest
    import bf_tap_r2.t2g_core as core
    q,r=saved
    model,prep,meta=load_state(r.model_dir)
    before=state_digest(model.state_dict())
    def forbidden(*args,**kwargs): raise AssertionError("Inference rebuilt a model")
    monkeypatch.setattr(core,"make_model",forbidden)
    assert np.array_equal(r.prediction,predict_loaded(model,prep,meta,q))
    assert state_digest(model.state_dict())==before


def test_real_selector_refit_cold_unit(tmp_path):
    import json
    from bf_tap_r2.t2g_audit import audit_unit
    from bf_tap_r2.t2g_model import file_hash,canonical
    from pathlib import Path
    rng=np.random.default_rng(1900)
    f=pd.DataFrame(rng.normal(size=(40,21)),columns=FEATURES)
    f["sample_id"]=[f"SYNTH_AUDIT_{i}" for i in range(40)]
    f["spout_no"]=np.arange(40)%2+1; f["tap_iron"]=100+f.air_volume
    folds=np.where(np.arange(40)<8,0,1)
    train=f.loc[folds!=0].reset_index(drop=True)
    q=f.loc[folds==0,["sample_id","spout_no",*FEATURES]].reset_index(drop=True)
    def event(stage):
        p=Path("local/t2g-unit-training/optimizer-starts.jsonl")
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("a") as stream: stream.write(canonical({"purpose":"artificial-cold-audit-unit","stage":stage,"rows":32})+"\n")
    output=tmp_path/"unit"
    r=fit_partition(train,q,"tap_iron","LEARNED_GRAPH",output,event)
    np.savez(output/"predictions.npz",query_ids=q.sample_id.to_numpy(dtype=str),prediction=r.prediction)
    key={"phase":"development","target":"tap_iron","arm":"LEARNED_GRAPH","seed":42,"fold":0}
    complete={"key":key,"manifest_digest":"a"*64,"fit_ids":train.sample_id.tolist(),"query_ids":q.sample_id.tolist(),
        "hashes":{p.name:file_hash(p) for p in output.iterdir() if p.is_file()}}
    (output/"complete.json").write_text(json.dumps(complete))
    record={**complete,"complete_sha256":file_hash(output/"complete.json")}
    report=audit_unit(output,record,f,{42:folds},"a"*64)
    assert report["status"]=="passed"
    assert report["optimizer_starts"]==0
    changed=f.copy(); changed.loc[8,"tap_iron"]+=1
    with pytest.raises(ValueError): audit_unit(output,record,changed,{42:folds},"a"*64)
