import importlib
import numpy as np
import pytest
torch=pytest.importorskip("torch")
def mod():return importlib.import_module("bf_tap_r2.iron_mae_model")
def test_single_target_small_deterministic_network():
 m=mod();a,_=m.network(23);b,_=m.network(23)
 assert a(torch.zeros(4,23)).shape==(4,1)
 assert sum(p.numel() for p in a.parameters())<11000
 assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
def test_loss_alignment_and_outlier_gradient():
 m=mod();p=torch.tensor([[0.],[0.]],requires_grad=True);y=torch.tensor([[1.],[10.]])
 loss=m.supervised_loss(p,y,"COMPACT_MAE");assert loss.item()==5.5
 loss.backward();assert torch.equal(p.grad,torch.tensor([[-.5],[-.5]]))
 assert m.supervised_loss(p,y,"MSE_CONTROL").item()==50.5
 with pytest.raises(ValueError):m.supervised_loss(p,y,"unknown")

def test_saved_one_target_cold_roundtrip_without_optimizer(tmp_path,monkeypatch):
 m=mod()
 import pandas as pd
 from bf_tap_r2.data import FEATURES
 rng=np.random.default_rng(1);frame=pd.DataFrame(rng.normal(size=(20,21)),columns=FEATURES)
 frame["spout_no"]=np.arange(20)%2+1;frame["sample_id"]=[str(i) for i in range(20)]
 prep=m.Preprocessor().fit(frame);net,projection=m.network(23);path=tmp_path/"model.npz"
 ym=np.array([500.]);ys=np.array([50.]);m.save_state(path,net,projection,prep,ym,ys)
 expected=net(torch.from_numpy(prep.transform(frame))).detach().numpy()*ys+ym
 def forbidden(*a,**k):raise AssertionError("cold prediction must not instantiate optimizer")
 monkeypatch.setattr(torch.optim,"Adam",forbidden)
 actual=m.predict_state(path,frame)
 assert actual.shape==(20,1)
 np.testing.assert_allclose(actual,expected,atol=5e-4,rtol=0)
 np.testing.assert_allclose(m.numpy_predict(path,frame),actual,atol=5e-4,rtol=0)
 unknown=frame.copy();unknown["spout_no"]=99
 assert np.all(prep.transform(unknown)[:,21:]==0)
 with pytest.raises(ValueError):prep.transform(frame.assign(tap_time_len=1))
def test_duplicate_group_rejected_before_budget_consumption(tmp_path):
 m=mod()
 import pandas as pd
 from bf_tap_r2.data import FEATURES
 from bf_tap_r2.iron_mae_protocol import Ledger
 frame=pd.DataFrame(np.ones((3,21)),columns=FEATURES);frame["sample_id"]=["a","b","c"];frame["spout_no"]=1
 ledger=Ledger(tmp_path/"ledger")
 with pytest.raises(ValueError,match="Duplicate group"):
  m.fit_state(frame,np.ones((3,1)),frame,None,arm="COMPACT_MAE",epochs=1,selector=False,path=tmp_path/"unused.npz",ledger=ledger,phase="development",key="u")
 assert ledger.counts("development")["reserved"]=={"state":0,"optimizer":0}
