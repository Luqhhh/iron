from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import pytest
import torch
import yaml

from test_round2_v12 import sample
from bf_tap_r2.data import TARGETS
from bf_tap_r2.component_regularization import ComponentRegressor
from bf_tap_r2.component_regularization_audit import verify_saved
from bf_tap_r2.component_regularization_run import RECIPE,choose
from bf_tap_r2.component_augmentation import AugmentedRegressor,CoupledNetwork,neighbor_index,interpolate
from bf_tap_r2.v12_joint import make_network
from bf_tap_r2.component_replay_cache import verify_original_sources


def spec():return yaml.safe_load(Path('configs/component_augmentation_representation/SPEC.yaml').read_text())


def test_neighbors_exclude_self_cross_spout_resolve_ties_and_singletons():
    x=np.array([[0.],[1.],[-1.],[.1],[10.]])
    ids=['z','b','a','q','only'];spout=[1,1,1,2,3]
    n,c=neighbor_index(x,spout,ids,8)
    assert n[0,:2].tolist()==[2,1]
    assert c.tolist()==[2,2,2,0,0]
    for i in range(3):assert i not in n[i,:c[i]] and all(spout[j]==spout[i] for j in n[i,:c[i]])
    for i in [3,4]:assert n[i,0]==i
    permutation=np.array([4,2,1,0,3]);nn,cc=neighbor_index(x[permutation],np.array(spout)[permutation],np.array(ids)[permutation],8)
    for i,old in enumerate(permutation):assert permutation[nn[i,:cc[i]]].tolist()==n[old,:c[old]].tolist()


def test_interpolation_uses_same_lambda_for_features_and_both_targets():
    x=torch.tensor([[0.,2.],[4.,6.]]);y=torch.tensor([[10.,20.],[30.,60.]])
    lam=torch.tensor([.25,.75]);mx,my=interpolate(x,y,(x.flip(0),y.flip(0)),lam)
    torch.testing.assert_close(mx,torch.tensor([[3.,5.],[3.,5.]]))
    torch.testing.assert_close(my,torch.tensor([[25.,50.],[25.,50.]]))


def test_zero_mix_weight_replays_base_and_validation_never_enters_neighbors(tmp_path):
    frame,settings=sample();y=frame[list(TARGETS)].to_numpy();mech=spec()['mechanisms']
    base=ComponentRegressor(RECIPE,settings,'BASE',mech).fit(frame,y)
    new=AugmentedRegressor(RECIPE,settings,'D-LMIX',dict(mech,mixup_loss_weight=0.),tmp_path).fit(frame,y)
    query=frame.drop(columns=list(TARGETS));np.testing.assert_array_equal(base.predict(query),new.predict(query))
    assert new.neighbors.max()<len(frame)
    inner=torch.load(tmp_path/'selection.pt',weights_only=True)
    assert inner['trace']['fit_rows']<len(frame)
    assert inner['auxiliary']['fit_ids_digest']==inner['trace']['fit_ids_digest']


def test_coupled_initial_function_rng_rank_and_fixed_learned_pair():
    frame,settings=sample();mech=spec()['mechanisms'];torch.manual_seed(42)
    native=make_network(RECIPE,settings,4,2);before=torch.get_rng_state()
    fixed=CoupledNetwork(deepcopy(native),mech,False);learned=CoupledNetwork(deepcopy(native),mech,True)
    assert torch.equal(before,torch.get_rng_state())
    x=torch.randn(13,21);cat=torch.ones(13,1,dtype=torch.long)
    native.eval();fixed.eval();learned.eval()
    torch.testing.assert_close(native(x,cat),fixed(x,cat),atol=0,rtol=0)
    torch.testing.assert_close(fixed(x,cat),learned(x,cat),atol=0,rtol=0)
    assert torch.linalg.matrix_rank((learned.u@learned.v).double(),atol=1e-6).item()<=4
    assert not fixed.u.requires_grad and not fixed.v.requires_grad
    with torch.no_grad():learned.extra_weight.fill_(.05)
    learned(x,cat).square().mean().backward()
    assert learned.u.grad.abs().sum()>0 and learned.v.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in learned.native.num_module.parameters())


@pytest.mark.parametrize('arm',['D-LMIX','R-FIXED','R-LEARNED'])
def test_adapter_saved_inference_audit_and_tamper(tmp_path,arm):
    frame,settings=sample();mech=spec()['mechanisms'];train=frame.iloc[:80];query=frame.iloc[80:].drop(columns=list(TARGETS))
    model=AugmentedRegressor(RECIPE,settings,arm,mech,tmp_path).fit(train,train[list(TARGETS)].to_numpy())
    cold=verify_saved(tmp_path/'refit.pt',train,train[list(TARGETS)].to_numpy(),arm,settings,mech,
                      expected_epoch=model.metadata_['selected_epoch'],model_type=AugmentedRegressor)
    np.testing.assert_array_equal(cold.predict(query),model.predict(query))
    variants=[cold.predict(query.iloc[::-1])[::-1],np.concatenate([cold.predict(query.iloc[i:i+3]) for i in range(0,len(query),3)])]
    for v in variants:np.testing.assert_allclose(v,model.predict(query),atol=5e-4,rtol=0)
    payload=torch.load(tmp_path/'refit.pt',weights_only=True)
    if arm=='D-LMIX':payload['auxiliary']['neighbor_digest']='tampered'
    elif arm=='R-FIXED':payload['state']['u'][0,0]+=1
    else:payload['trace']['updates']+=1
    torch.save(payload,tmp_path/'bad.pt')
    with pytest.raises(ValueError):verify_saved(tmp_path/'bad.pt',train,train[list(TARGETS)].to_numpy(),arm,settings,mech,model_type=AugmentedRegressor)


def test_mixup_separate_dropout_stream_and_live_supervision():
    frame,settings=sample();mech=spec()['mechanisms'];y=frame[list(TARGETS)].to_numpy()
    m=AugmentedRegressor(RECIPE,settings,'D-LMIX',mech);m._initialize(frame,y)
    x,c=m._inputs(frame);target=torch.as_tensor((y-m.mean_)/m.std_,dtype=torch.float32)
    m.prepare_training(frame,x,c,target);m.model_.train();idx=np.arange(10)
    before=torch.get_rng_state();m.model_(x[idx],c[idx]);after=torch.get_rng_state()
    torch.set_rng_state(before);loss=m.training_loss(x,c,target,idx)
    assert torch.equal(after,torch.get_rng_state()) and m.auxiliary['synthetic_rows']==10
    loss.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.model_.parameters())


def test_original_source_fallback_rejects_tamper(tmp_path,monkeypatch):
    p=tmp_path/'model.py';p.write_bytes(b'changed')
    monkeypatch.setattr('subprocess.check_output',lambda *a,**k:b'original')
    verify_original_sources(tmp_path,{'model.py':hashlib.sha256(b'original').hexdigest()},'frozen')
    with pytest.raises(ValueError):verify_original_sources(tmp_path,{'model.py':'0'*64},'frozen')


def test_new_pool_tie_order_and_negative_seed_rejection():
    s=spec();rows=[]
    for t in TARGETS:
        for a,g in [('BASE',[0,0]),('D-LMIX',[.02,.02]),('R-FIXED',[.02,.02]),('R-LEARNED',[.05,-.001])]:
            rows.append({'target':t,'arm':a,'seeds':{str(i):{'gain':v} for i,v in enumerate(g)},'paired':{'mean':np.mean(g)}})
    assert set(choose(rows,s).values())=={'D-LMIX'}
