import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2 import ema_mean5 as m
from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.ema_nested_residual import split_training
from bf_tap_r2.v3_4_bags import group_safe_inner_folds


def test_equal_members_arithmetic_and_prespecified_pool():
    q = np.array([100., 200., 300.])
    members = np.array([[10,20,30],[20,10,40],[30,30,50],[40,50,10],[50,40,20]],float)
    old, new = m.columns(q,members)
    np.testing.assert_array_equal(old,q+.75*(members[:3].mean(0)-members[0]))
    np.testing.assert_allclose(new,old+.75*(members.mean(0)-members[:3].mean(0)),rtol=0,atol=1e-13)
    assert m.OLD_INITS+m.NEW_INITS == (42,1042,2042,3042,4042)
    with pytest.raises(ValueError,match='five'):m.columns(q,members[:4])
    with pytest.raises(ValueError,match='five'):m.columns(q,np.full((5,3),np.nan))
    with pytest.raises(ValueError,match='no clipping'):m.columns(np.zeros(3),members[[4,1,2,3,0]])


def test_confirmation_uses_complete_current_reference_gains():
    assert m.eligible({'42':.001,'3407':.002})
    assert not m.eligible({'42':.001,'3407':0})
    assert not m.eligible({'42':-.001,'3407':.002})
    for g in ({'42':1}, {'42':float('nan'),'3407':1}, {'42':1,'3407':1,'9':1}):
        with pytest.raises(ValueError,match='complete'):m.eligible(g)
    y = np.array([[500.,100.],[500.,100.]])
    gain = m.scalar_score(y,y[:,0],[101.,101.])-m.scalar_score(y,y[:,0],[102.,102.])
    assert gain == .5  # One percent time WMAPE means half a total score point.


def test_worker_isolation_ignores_query_labels_and_preserves_duplicate_groups():
    rng=np.random.default_rng(137)
    frame=pd.DataFrame(rng.normal(size=(100,len(FEATURES))),columns=FEATURES)
    frame['sample_id']=[f'synthetic-{i:03}' for i in range(len(frame))]
    frame['spout_no']=np.arange(len(frame))%2+1
    for t in TARGETS:frame[t]=rng.normal(size=len(frame))
    frame.loc[3,list(FEATURES)]=frame.loc[0,list(FEATURES)].to_numpy()
    fv=group_safe_inner_folds(frame,seed=42)['fold'];assert fv[0]==fv[3]
    for fold in range(5):
        t,q=split_training(frame,fv,fold)
        changed=frame.copy();changed.loc[fv==fold,list(TARGETS)]=1e90
        t2,q2=split_training(changed,fv,fold)
        pd.testing.assert_frame_equal(t,t2);pd.testing.assert_frame_equal(q,q2)
        plan=dict(training_ids=t.sample_id.tolist(),query_ids=q.sample_id.tolist())
        m.validate_frames(t,q,plan)
        bad=q.copy();bad[TARGETS[0]]=0
        with pytest.raises(ValueError,match='isolation'):m.validate_frames(t,bad,plan)
        bad=q.copy();bad.loc[0,'sample_id']=t.sample_id.iloc[0]
        with pytest.raises(ValueError,match='isolation'):m.validate_frames(t,bad,plan)


def test_optimizer_records_before_native_construction_and_rejects_third(tmp_path):
    import torch
    identity=dict(source_directory=str(tmp_path),split_seed=42,fold=0,trial_id='EMA_INIT3042')
    calls=[]
    class FakeOptimizer:
        def __init__(self):
            role=('selection','refit')[len(calls)]
            recorded=json.loads((tmp_path/(role+'-optimizer-start.json')).read_text())
            assert recorded['identity']==identity and recorded['role']==role
            calls.append(role)
        def step(self):return 17
    with patch.object(torch.optim,'AdamW',FakeOptimizer),m.optimizer_ledger(tmp_path,identity) as counts:
        a=torch.optim.AdamW();assert a.step()==17;a.step()
        b=torch.optim.AdamW();b.step()
        with pytest.raises(ValueError,match='before construction'):torch.optim.AdamW()
    assert calls==['selection','refit']
    m.verify_counts(counts,{'selection':{'updates':2},'refit':{'updates':1}})
    with pytest.raises(ValueError,match='inventory'):
        m.verify_counts(counts,{'selection':{'updates':3},'refit':{'updates':1}})
    assert not list(tmp_path.glob('*.pt'))


def test_fixed_task_inventory_and_no_quality_before_all_models(tmp_path,monkeypatch):
    ts=m.tasks();assert len(ts)==52
    assert sum(x[0]=='worker' for x in ts)==sum(x[0]=='cold' for x in ts)==20
    assert sum(x[0]=='reuse' for x in ts)==10
    assert [x[0] for x in ts[-2:]]==['report','audit']
    monkeypatch.setattr(m,'RUN',tmp_path)
    (tmp_path/'execution').mkdir()
    with pytest.raises(FileNotFoundError):m.verify_model_events()
    for i,(stage,seed,fold,init) in enumerate(ts[:50]):
        key=f'{i:03d}-{stage}-s{seed}-f{fold}-i{init}'
        m.write(tmp_path/'execution'/(key+'-terminal.json'),dict(task=key,exit_code=0,peak_rss_mib=800))
    m.verify_model_events()
    key='049-cold-s3407-f4-i4042';p=tmp_path/'execution'/(key+'-terminal.json')
    p.write_text(json.dumps(dict(task=key,exit_code=1,peak_rss_mib=800)))
    with pytest.raises(ValueError,match='successful'):m.verify_model_events()


def test_existing_run_not_restarted_before_reading_any_input(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'RUN',tmp_path)
    with pytest.raises(FileExistsError,match='consumed'):m.prepare('missing-checks.json')
    with pytest.raises(ValueError,match='Undeclared'):m.worker(42,0,42)


def test_controller_preserves_first_failure_without_launching_next_child(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'RUN',tmp_path)
    m.write(tmp_path/'manifest.json',dict(files={}))
    launched=[]
    def launch(*args,**kwargs):
        launched.append(args[0]);return SimpleNamespace(pid=1001,returncode=None)
    monkeypatch.setattr(m.subprocess,'Popen',launch)
    monkeypatch.setattr(m.os,'wait4',lambda pid,flag:(pid,3<<8,SimpleNamespace(ru_maxrss=800*1024)))
    with pytest.raises(RuntimeError,match='child failed'):m.controller()
    assert len(launched)==1
    terminal=json.loads((tmp_path/'execution/terminal.json').read_text())
    assert terminal['status']=='failed' and terminal['automatic_retry'] is False
    assert terminal['events'][0]['exit_code']==3
    assert not (tmp_path/'report.json').exists()
