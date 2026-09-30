"""Isolation and serialization of frozen T2G partition training."""
import json
import numpy as np
import pandas as pd
import pytest
torch=pytest.importorskip("torch")
from bf_tap_r2.data import FEATURES
from bf_tap_r2.t2g_model import EpochSelector, fit_partition, load_predict, prepare

def frame(n=40):
    rng=np.random.default_rng(910)
    f=pd.DataFrame(rng.normal(size=(n,21)),columns=FEATURES)
    f["spout_no"]=np.arange(n)%2+1
    f["sample_id"]=[f"SYNTH_{i:04}" for i in range(n)]
    f["tap_time_len"]=10+f.air_volume+f.oxygen*.2
    return f

def test_train_only_unknown_category():
    f=frame(); q=f.iloc[:4].drop(columns="tap_time_len").copy()
    q.spout_no=99; q.air_volume+=1000
    prep,x,c,mean,std=prepare(f,"tap_time_len")
    saved=prep.means_.copy()
    xn,cn=prep.transform_tabm(q)
    assert (cn==0).all()
    assert np.array_equal(prep.means_,saved)
    assert xn.dtype==np.float32 and c.dtype==np.int64
    assert abs(mean-f.tap_time_len.mean())<1e-12

@pytest.mark.parametrize("values",[np.ones(40),np.full(40,np.nan)])
def test_invalid_target_scale(values):
    f=frame(); f.tap_time_len=values
    with pytest.raises(ValueError): prepare(f,"tap_time_len")

def test_earliest_epoch_patience():
    s=EpochSelector()
    assert s.update(1,1.) and s.selected_epoch==1
    assert not s.update(2,1.-1e-6) and s.selected_epoch==1
    assert s.update(3,.9) and s.selected_epoch==3
    for epoch in range(4,29): s.update(epoch,.9)
    assert s.stop and s.selected_epoch==3
    with pytest.raises(ValueError): s.update(29,float("nan"))

@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    root=tmp_path_factory.mktemp("t2g_pair")
    f=frame(); q=f.iloc[:7].drop(columns="tap_time_len")
    results=[]
    counts=[]
    for arm in ("LEARNED_GRAPH","DENSE_CONTROL"):
        r=fit_partition(f,q,"tap_time_len",arm,root/arm,counts.append,force_epochs=2)
        results.append(r)
    assert counts==["selector","refit","selector","refit"]
    return f,q,results

def test_fresh_refit(pair):
    f,q,results=pair
    for r in results:
        m=json.loads((r.model_dir/"metadata.json").read_text())
        assert m["refit_rows"]==40 and m["selector_rows"]<40
        assert m["selector_final_digest"]!=m["refit_init_digest"]
        assert m["refit_init_digest"]==r.init_digest
        assert m["refit_epochs"]==2 and r.optimizer_starts==2
        assert m["preprocessor"]["means"]==f.loc[:,FEATURES].mean().tolist()

def test_pair_batch_identity(pair):
    a,b=pair[2]
    assert a.init_digest==b.init_digest and a.batch_digest==b.batch_digest

def test_weights_only_roundtrip(pair):
    f,q,results=pair
    for r in results:
        assert np.array_equal(r.prediction,load_predict(r.model_dir,q))
        saved=r.prediction.copy()
        reverse=load_predict(r.model_dir,q.iloc[::-1])[::-1]
        assert np.max(np.abs(saved-reverse))<=5e-4

def test_query_targets_and_existing_output_rejected(tmp_path,pair):
    f,q,results=pair
    with pytest.raises(ValueError):
        fit_partition(f,f.iloc[:2],"tap_time_len","LEARNED_GRAPH",tmp_path/"bad",lambda s:None)
    with pytest.raises(FileExistsError):
        fit_partition(f,q,"tap_time_len","LEARNED_GRAPH",results[0].model_dir,lambda s:None)
