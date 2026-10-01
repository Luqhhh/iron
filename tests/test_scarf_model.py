import importlib
import numpy as np
import pandas as pd
import pytest
torch=pytest.importorskip("torch")
from bf_tap_r2.data import FEATURES

def mod(): return importlib.import_module("bf_tap_r2.scarf_model")

def frame(n=30):
    r=np.random.default_rng(12)
    f=pd.DataFrame(r.normal(size=(n,21)),columns=FEATURES)
    f["spout_no"]=np.arange(n)%2+1; f["sample_id"]=[str(i) for i in range(n)]
    return f

def test_corruption_exact_columns_same_spout_training_bank():
    m=mod(); x=torch.arange(40*23,dtype=torch.float32).reshape(40,23)
    sp=torch.arange(40)%2; g=torch.Generator().manual_seed(2)
    out,mask=m.corrupt(x[:8],sp[:8],x,sp,g)
    assert torch.equal(out[:,21:],x[:8,21:])
    assert torch.equal(mask.sum(1),torch.ones(8,dtype=torch.int64)*12)
    for i,j in torch.nonzero(mask): assert out[i,j].item() in x[sp==sp[i],j].tolist()
    a,_=m.corrupt(x[:8],sp[:8],x,sp,torch.Generator().manual_seed(2))
    assert torch.equal(out,a)

def test_infonce_duplicate_negative_mask_numpy_and_gradient():
    m=mod(); z=torch.tensor([[1.,0.],[0.,1.],[.6,.8]],requires_grad=True)
    w=torch.tensor([[1.,0.],[0.,1.],[.8,.6]],requires_grad=True)
    groups=torch.tensor([0,0,1]); loss=m.info_nce(z,w,groups)
    a=z.detach().numpy(); b=w.detach().numpy(); q=(a/np.linalg.norm(a,axis=1)[:,None])@(b/np.linalg.norm(b,axis=1)[:,None]).T
    q[0,1]=q[1,0]=-np.inf
    expected=np.mean(np.log(np.exp(q).sum(1))-np.diag(q))
    assert loss.item()==pytest.approx(expected,abs=1e-6)
    loss.backward(); assert torch.isfinite(z.grad).all()
    assert m.info_nce(z,w,torch.zeros(3,dtype=torch.long)).item()==pytest.approx(0.)

def test_train_only_transform_unknown_and_label_rejection():
    m=mod(); f=frame(); p=m.Preprocessor().fit(f)
    q=f.iloc[:2].copy(); q["spout_no"]=99
    x=p.transform(q); assert not x[:,21:].any()
    assert np.max(np.abs(p.transform(f)[:,:21].mean(0)))<1e-6
    q["tap_iron"]=1
    with pytest.raises(ValueError): p.transform(q)
    assert np.allclose(p.mean,f[list(FEATURES)].to_numpy().mean(0))

def test_matched_init_and_pickle_free_cold_forward(tmp_path):
    m=mod(); f=frame(); p=m.Preprocessor().fit(f)
    net,proj=m.network(23); net2,_=m.network(23)
    assert all(torch.equal(a,b) for a,b in zip(net.parameters(),net2.parameters()))
    path=tmp_path/"state.npz"
    m.save_state(path,net,proj,p,np.array([200.,50.]),np.array([20.,5.]))
    a=m.predict_state(path,f); b=m.numpy_predict(path,f)
    assert np.max(np.abs(a-b))<5e-4
    assert np.max(np.abs(a-m.predict_state(path,f.iloc[::-1])[::-1]))<5e-4
    assert np.max(np.abs(a-np.concatenate([m.predict_state(path,f.iloc[i:i+1]) for i in range(len(f))])))<5e-4
    with np.load(path,allow_pickle=False) as s: assert all(s[k].dtype!=object for k in s.files)

def test_partition_group_disjoint(tmp_path):
    m=mod(); f=frame(100); f.loc[1,list(FEATURES)]=f.loc[0,list(FEATURES)].values
    inner,cal=m.inner_partition(f)
    groups=pd.util.hash_pandas_object(f[list(FEATURES)],index=False).to_numpy()
    assert not set(groups[inner])&set(groups[cal])
