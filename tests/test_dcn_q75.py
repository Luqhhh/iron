import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES
from bf_tap_r2 import dcn_cross as native
from bf_tap_r2.dcn_q75_run import decisions,synthetic,fit_arm,audit_arm,sha


def test_quality_gate_requires_complete_two_seed_and_matched_advantage():
    spec={'development_mean_gain_minimum':.01,'minimum_local_working_score':96.25}
    rows={'42':dict(gain=.01,mechanism_gain=.003,working_score=96.3),
          '3407':dict(gain=.012,mechanism_gain=.002,working_score=96.4)}
    assert decisions({'tap_iron':rows},spec)['tap_iron']['eligible']
    rows['3407']['gain']=-.001
    assert not decisions({'tap_iron':rows},spec)['tap_iron']['eligible']
    rows['3407']['gain']=.012;rows['3407']['mechanism_gain']=-.004
    assert not decisions({'tap_iron':rows},spec)['tap_iron']['eligible']
    del rows['3407']
    with pytest.raises(ValueError):decisions({'tap_iron':rows},spec)


def test_fixed_synthetic_has_official_size_and_four_spouts():
    f,c,t,q,y=synthetic()
    assert (len(f),len(c),len(t),len(q),len(y))==(1762,441,2203,551,551)
    assert set(t.spout_no)=={1,2,3,4}
    assert not set(t.sample_id)&set(q.sample_id)
    assert not {'tap_iron','tap_time_len'}&set(q.columns)
    assert set(f.sample_id)|set(c.sample_id)==set(t.sample_id)


def test_recorded_native_training_is_identical_and_cold_never_initializes_optimizer(tmp_path,monkeypatch):
    f,c,t,q,_=synthetic();f=f.iloc[:64];c=c.iloc[:32];t=pd.concat([f,c]);q=q.iloc[:17]
    spec=json.loads(Path('configs/dcn_q75_development/SPEC.json').read_text())['training']
    settings=dict(spec,max_epochs=8,patience=3)
    original=native.CrossRegressor
    unit=tmp_path/'pair';unit.mkdir()
    pred,meta=fit_arm(tmp_path,unit,f,c,t,q,'tap_iron','CROSS',settings,dict(trial_id='test'))
    assert native.CrossRegressor is original
    refit,selector,expected,cp=native.fit_partition(f,c,t,q,'tap_iron','CROSS',settings)
    np.testing.assert_array_equal(pred,expected)
    assert meta['selector']['selected_epoch']==selector.selected_epoch_
    events=[json.loads(v) for v in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [v['phase'] for v in events if v['event']=='optimizer_started']==['selection','refit']
    assert sum(v['event']=='procedure_completed' for v in events)==1
    hashes={str(p.relative_to(unit)):sha(p) for p in unit.rglob('*') if p.is_file()}
    monkeypatch.setattr(native.CrossRegressor,'initialize',lambda *a,**k:(_ for _ in ()).throw(AssertionError('no fitting')))
    audit,p=audit_arm(unit,f,c,t,q,'tap_iron','CROSS',settings,hashes)
    assert audit['maximum_difference']<1e-10
    np.testing.assert_array_equal(p,pred)
