import math
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2 import de3_independent_batches as run
from bf_tap_r2.de3_independent_batches_model import JointIndependentBatchRegressor
from bf_tap_r2.ema_independent_batches_model import member_loss, member_orders
from bf_tap_r2.data import FEATURES, TARGETS


def test_joint_head_row_output_alignment_and_loss_gradient():
    import torch
    from bf_tap_r2.v12_joint import make_network
    torch.set_num_threads(1)
    pred=torch.arange(12,dtype=torch.float64).reshape(2,3,2).requires_grad_()
    target=torch.flip(pred.detach(),[1])+torch.tensor([1.,-2.])
    expected=math.fsum((float(pred.detach()[i,j,k])-float(target[i,j,k]))**2
        for i in range(2) for j in range(3) for k in range(2))/12
    loss=member_loss(pred,target);assert float(loss.detach())==expected;loss.backward()
    torch.testing.assert_close(pred.grad,2*(pred.detach()-target)/12,rtol=0,atol=5e-16)
    with pytest.raises(ValueError):member_loss(pred,target[:,:,0:1])
    spec=run.read(run.SPEC);torch.manual_seed(964505)
    model=make_network(spec['recipe'],spec['training'],2,2).eval()
    x=torch.randn(31,len(FEATURES));cat=(torch.arange(31)%2).reshape(-1,1)
    orders=member_orders(np.random.default_rng(964505),31,16)
    np.testing.assert_array_equal(np.sort(orders,axis=0),np.tile(np.arange(31)[:,None],(1,16)))
    assert len({tuple(x) for x in orders.T})==16
    with torch.no_grad():
        shared=model(x,cat);independent=model(x[orders[:7]],cat[orders[:7]])
    expected=torch.stack([shared[orders[:7,j],j] for j in range(16)],dim=1)
    torch.testing.assert_close(independent,expected,rtol=1e-5,atol=1e-6)


def test_replacement_is_incremental_to_current_de3_and_fixed_pool():
    old=np.array([[80.,180.],[100.,200.],[120.,220.]])
    new=old+[2.,-4.]
    values=run.columns([100.,200.],old,new)
    np.testing.assert_array_equal(values['JOINT_IBATCH_IRON_A100'],[101.,198.])
    np.testing.assert_allclose(values['JOINT_IBATCH_IRON_A20'],[100.2,199.6],rtol=0,atol=1e-12)
    assert run.choose({'JOINT_IBATCH_IRON_A100':{'42':-.1,'3407':.2},
        'JOINT_IBATCH_IRON_A20':{'42':.01,'3407':.02}})=='JOINT_IBATCH_IRON_A20'
    assert run.choose({k:{'42':.1,'3407':.2} for k in run.CANDIDATES})=='JOINT_IBATCH_IRON_A100'
    with pytest.raises(ValueError):run.choose({k:{'42':.1} for k in run.CANDIDATES})
    with pytest.raises(ValueError):run.columns([100.,200.],old,new[:2])
    with pytest.raises(ValueError,match='clipping'):run.columns([100.,200.],old,new-1000.)


def test_partial_coverage_cannot_read_quality_and_controller_stops_first_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'RUN',tmp_path)
    touched=[]
    monkeypatch.setattr(run,'reference_arrays',lambda *_:touched.append(True))
    with pytest.raises(FileNotFoundError):run.report()
    assert not touched
    tasks=run.tasks();assert len(tasks)==72
    assert sum(x[0]=='worker' for x in tasks)==sum(x[0]=='cold' for x in tasks)==30
    assert {x[3] for x in tasks if x[0]=='worker'}=={42,104729,130363}
    with pytest.raises(FileExistsError):run.prepare('missing')
    run.write(tmp_path/'manifest.json',{'files':{}})
    launched=[]
    def launch(*a,**kw):launched.append(a[0]);return SimpleNamespace(pid=1001,returncode=None)
    monkeypatch.setattr(run.subprocess,'Popen',launch)
    monkeypatch.setattr(run.os,'wait4',lambda *_:(1001,3<<8,SimpleNamespace(ru_maxrss=800*1024)))
    with pytest.raises(RuntimeError,match='Scientific child failed'):run.controller()
    assert len(launched)==1 and 'bf_tap_r2.de3_independent_batches' in launched[0]
    assert run.read(tmp_path/'execution/terminal.json')['status']=='failed'
    assert not (tmp_path/'report.json').exists()


def test_joint_worker_passes_only_own_two_targets_and_label_free_features(tmp_path,monkeypatch):
    spec=run.read(run.SPEC)
    training=pd.DataFrame({name:np.arange(20,dtype=float) for name in FEATURES})
    training['sample_id']=[f'synthetic-joint-{i}' for i in range(20)];training['spout_no']=1
    training['tap_iron']=1000+np.arange(20);training['tap_time_len']=100+np.arange(20)
    query=training.iloc[:3].drop(columns=list(TARGETS)).copy();query['sample_id']=['q0','q1','q2']
    plan={'training_ids':training.sample_id.tolist(),'query_ids':query.sample_id.tolist()}
    run.validate_frames(training,query,plan)
    with pytest.raises(ValueError):run.validate_frames(training,training.iloc[:3],plan)
    with pytest.raises(ValueError):JointIndependentBatchRegressor(spec['recipe'],spec['training'])
    settings=dict(spec['training'],batch_order='independent_without_replacement')
    model=JointIndependentBatchRegressor(spec['recipe'],settings)
    with pytest.raises(ValueError,match='two native'):model.fit(training,np.zeros((20,1)))
    touched=[]
    def fake_fit(self,frame,y):
        assert not set(TARGETS)&set(frame.columns)
        np.testing.assert_array_equal(y,training[list(TARGETS)].to_numpy())
        touched.append(True);raise RuntimeError('synthetic pre-fit sentinel')
    monkeypatch.setattr(run,'RUN',tmp_path)
    monkeypatch.setattr(run,'unit_frames',lambda *_:(spec,training,query))
    monkeypatch.setattr(JointIndependentBatchRegressor,'fit',fake_fit)
    with pytest.raises(RuntimeError,match='pre-fit sentinel'):run.worker(42,0,104729)
    assert touched==[True]
    assert not list(tmp_path.rglob('*-optimizer-start.json'))
    assert (tmp_path/'s42-f0-init104729/failure.json').exists()


def test_optimizer_budget_fails_before_third_real_construction(tmp_path):
    import torch
    calls=[]
    def fake(*args,**kwargs):calls.append(True);return SimpleNamespace(step=lambda:None)
    identity={'source_directory':str(tmp_path),'split_seed':-1,'trial_id':'mock_budget'}
    with patch.object(torch.optim,'AdamW',fake):
        with run.optimizer_ledger(tmp_path,identity) as counts:
            torch.optim.AdamW([]).step();torch.optim.AdamW([]).step()
            with pytest.raises(ValueError,match='budget'):torch.optim.AdamW([])
    assert len(calls)==2
    run.verify_counts(counts,{'selection':{'updates':1},'refit':{'updates':1}})


def test_synthetic_full_shape_joint_fit_once():
    """Explicit env opt-in prevents accidental consumption by routine pytest."""
    location=os.environ.get('IRON_DE3_IBATCH_SYNTHETIC_DIRECTORY')
    if not location:pytest.skip('Requires a fresh explicitly budgeted engineering directory')
    d=Path(location).resolve();d.mkdir(parents=True,exist_ok=False)
    spec=run.read(run.SPEC);rng=np.random.default_rng(spec['engineering_budget']['synthetic_seed'])
    frame=pd.DataFrame(rng.normal(size=(2204,len(FEATURES))),columns=FEATURES)
    frame['sample_id']=[f'synthetic-joint-{i}' for i in range(len(frame))]
    frame['spout_no']=np.arange(len(frame))%2+1
    frame['tap_iron']=1000+15*frame[FEATURES[0]]+rng.normal(size=len(frame))
    frame['tap_time_len']=100+2*frame[FEATURES[1]]+.01*frame.tap_iron+rng.normal(size=len(frame))
    query=frame.iloc[:37].drop(columns=list(TARGETS)).copy()
    query['sample_id']=[f'query-joint-{i}' for i in range(len(query))]
    for name,value in [('training',frame),('query',query)]:
        with (d/(name+'.pkl')).open('xb') as f:pickle.dump(value,f,protocol=5)
    settings=dict(spec['training'],max_epochs=2,batch_order='independent_without_replacement')
    identity={'source_directory':str(d),'split_seed':-1,'trial_id':'SYNTHETIC_JOINT_INDEPENDENT'}
    with run.optimizer_ledger(d,identity) as counts:
        model=JointIndependentBatchRegressor(spec['recipe'],settings,d)
        model.fit(frame.drop(columns=list(TARGETS)),frame[list(TARGETS)].to_numpy())
    run.verify_counts(counts,model.traces)
    prediction=model.predict(query)
    run.save_arrays(d/'predictions.npz',ids=query.sample_id.to_numpy(str),prediction=prediction)
    run.write(d/'warm.json',dict(scope='synthetic_only',pid=os.getpid(),settings=settings,counts=counts,
        source_sha256=run.sha(run.__file__),model_sha256=run.sha(run.MODEL),
        files={str(d/n):run.sha(d/n) for n in ('training.pkl','query.pkl','selection.pt','refit.pt','predictions.npz')},
        peak_rss_mib=run.memory()))
    assert prediction.shape==(37,2) and np.isfinite(prediction).all()
    assert counts['constructors']==['selection','refit']
    assert all(t['stopped_epoch']<=2 for t in model.traces.values())
    assert model.arm=='BASE' and model.mechanisms=={}
