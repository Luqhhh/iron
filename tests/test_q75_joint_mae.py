import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from bf_tap_r2.data import FEATURES
from bf_tap_r2.joint_mae_model import ARMS, TARGET_ORDER, JointNetwork, JointMAERegressor, joint_loss, numpy_predict
from bf_tap_r2.q75_joint_mae import CANDIDATES, admission, joint_cold_state, validate_spec
from bf_tap_r2.v3_6_networks import NumericPreprocessor
from bf_tap_r2.v33_mixture import MixtureNetwork


def settings():
    return yaml.safe_load(Path('configs/round2_v33/SPEC.yaml').read_text())['training']


def test_loss_value_and_gradient_match_independent_laplace_distribution():
    y=torch.tensor([[-3.,.2],[4.,2.]],dtype=torch.float64)
    p=torch.tensor([[-1.,.4],[2.,3.]],dtype=torch.float64,requires_grad=True)
    value=joint_loss(y,p)
    expected=-torch.distributions.Laplace(p,.5).log_prob(y).sum(1).mean()
    torch.testing.assert_close(value,expected)
    torch.testing.assert_close(torch.autograd.grad(value,p)[0],2*torch.sign(p-y)/len(y))
    assert torch.autograd.gradcheck(lambda v:joint_loss(y,v),(p,))


def test_initial_target_functions_match_original_and_each_other():
    s=settings();x=torch.randn((7,len(FEATURES)+3),dtype=torch.float64)
    torch.manual_seed(42);original=MixtureNetwork('GAUSS1',s,3).double()
    expected=original(x)[1]
    for arm in ARMS:
        torch.manual_seed(42);model=JointNetwork(arm,s,3).double()
        torch.testing.assert_close(model(x),expected.repeat(1,2),rtol=0,atol=0)
        if arm=='SEPARATE_MAE':
            assert not ({p.data_ptr() for p in model.networks[0].parameters()} &
                        {p.data_ptr() for p in model.networks[1].parameters()})


def test_shared_trunk_gradient_is_sum_and_separate_has_no_cross_target_path():
    s=settings();x=torch.randn((7,len(FEATURES)+3),dtype=torch.float64)
    y=torch.randn((7,2),dtype=torch.float64)
    torch.manual_seed(42);shared=JointNetwork('SHARED_MAE',s,3).double()
    torch.manual_seed(42);separate=JointNetwork('SEPARATE_MAE',s,3).double()
    joint_loss(y,shared(x)).backward();joint_loss(y,separate(x)).backward()
    for name,value in shared.networks[0].named_parameters():
        if not name.startswith('head.'):
            a=dict(separate.networks[0].named_parameters())[name].grad
            b=dict(separate.networks[1].named_parameters())[name].grad
            torch.testing.assert_close(value.grad,a+b,rtol=1e-10,atol=1e-12)
    separate.zero_grad(set_to_none=True)
    separate(x)[:,0].sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in separate.networks[1].parameters())
    assert not shared.networks[0].head.weight.grad[2].any()


@pytest.mark.parametrize('arm',ARMS)
def test_persistence_numpy_forward_target_order_and_train_only_statistics(tmp_path,arm):
    rng=np.random.default_rng(987)
    frame=pd.DataFrame(rng.normal(size=(47,len(FEATURES))),columns=FEATURES)
    frame['sample_id']=[f'synthetic-{i}' for i in range(len(frame))];frame['spout_no']=np.arange(len(frame))%2+1
    frame['tap_iron']=20+frame.iloc[:,0]*2;frame['tap_time_len']=100+frame.iloc[:,1]*10
    model=JointMAERegressor(arm,settings())
    model.preprocessor_=NumericPreprocessor(structure='raw_mlp').fit(frame)
    y=frame[list(TARGET_ORDER)].to_numpy();model.mean_=y.mean(0);model.std_=y.std(0)
    model.model_=JointNetwork(arm,settings(),3).double()
    with torch.no_grad():
        model.model_.networks[0].head.bias.add_(torch.tensor([.1,.2,.3]))
    model.fit_ids_=frame.sample_id.tolist();model.selected_epoch_=model.stopped_epoch_=1;model.history_=[]
    path=tmp_path/'model.pt';model.save(path)
    query=frame.drop(columns=list(TARGET_ORDER)).copy();query.loc[0,'spout_no']=999
    query.iloc[0,0]+=100
    prediction=model.predict(query)
    loaded=JointMAERegressor.load(path)
    np.testing.assert_array_equal(prediction,loaded.predict(query))
    payload=torch.load(path,map_location='cpu',weights_only=True)
    np.testing.assert_allclose(prediction,numpy_predict(payload,query),rtol=0,atol=1e-10)
    _,diff=joint_cold_state(path,frame,query,model.metadata(),settings(),prediction)
    assert diff<1e-10
    with pytest.raises(ValueError,match='Query labels'):joint_cold_state(path,frame,frame,model.metadata(),settings(),prediction)
    payload['target_order']=list(reversed(TARGET_ORDER));bad=tmp_path/'bad.pt';torch.save(payload,bad)
    with pytest.raises(ValueError,match='target order'):JointMAERegressor.load(bad)


@pytest.mark.parametrize('y,p',[(torch.ones(3),torch.ones(3)),(torch.ones((2,2)),torch.ones((2,1))),
                              (torch.ones((2,2)),torch.full((2,2),float('nan')))])
def test_invalid_shape_or_nonfinite_loss_rejected(y,p):
    with pytest.raises(ValueError):joint_loss(y,p)


def vectors():
    return {c:{'42':.01,'3407':.02} for c in CANDIDATES}


def test_gate_requires_paired_mechanism_and_ready_comparisons():
    gains,ready,paired=vectors(),vectors(),vectors()
    assert admission(gains,ready,paired)['confirmation_finalist']==CANDIDATES[0]
    paired[CANDIDATES[0]]['3407']=-.001
    assert admission(gains,ready,paired)['confirmation_finalist']==CANDIDATES[1]
    ready[CANDIDATES[1]]['42']=0.
    assert admission(gains,ready,paired)['confirmation_finalist'] is None
    assert not admission(gains,ready,paired)['formal_promoted']


def test_incomplete_or_nonpositive_development_cannot_advance():
    gains=vectors();gains[CANDIDATES[0]]['42']=0.;gains[CANDIDATES[1]]['3407']=-.01
    assert admission(gains,vectors(),vectors())['confirmation_finalist'] is None
    gains[CANDIDATES[0]].pop('42')
    with pytest.raises(ValueError,match='Complete'):admission(gains,vectors(),vectors())


def test_scope_budget_and_target_order_frozen():
    spec=json.loads(Path('configs/q75_joint_mae/SPEC.json').read_text());validate_spec(spec)
    for key,value in [('optimizer_runs',42),('engineering_optimizer_runs',6),('target_order',list(reversed(TARGET_ORDER))),('monitor_seconds',1800)]:
        with pytest.raises(ValueError,match='scope or budget'):validate_spec(dict(spec,**{key:value}))
