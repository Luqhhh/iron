import json
import numpy as np
import pandas as pd
import pytest
from bf_tap.exceptions import ContractError
from bf_tap.features.history import build_history_features
from bf_tap.optimization.history_stable import stable_features
from bf_tap.optimization.history_component_v5 import transform, MAIN, AUX, AGES
from bf_tap.optimization.component_followup import reserve_fit


def inputs():
    t=pd.Timestamp('2024-07-01',tz='Asia/Shanghai')
    times=[t-pd.Timedelta(days=41-i) for i in range(1,41)]+[t,t+pd.Timedelta(days=1)]
    history=pd.DataFrame({'sample_id':[f'h{i}' for i in range(40)]+['self','future'],
        'tap_no':range(42),'spout_no':[1]*42,'reference_time':times,'tap_end_time':times,
        'available_at':times,'tap_iron':list(range(1,41))+[9999,8888],
        'tap_time_len':[2*i for i in range(1,41)]+[9999,8888]})
    samples=pd.DataFrame({'sample_id':['self','early'],'spout_no':[1,2],
                         'reference_time':[t,t-pd.Timedelta(days=100)]})
    raw,_=build_history_features(samples,history,fit_cutoff=t)
    raw.insert(0,'spout_no',samples.spout_no.astype(str))
    r2=stable_features(raw,samples,history,t,'R2')
    return t,samples,history,r2


def test_T1_eight_deltas_exclude_current_future_and_preserve_R2():
    t,s,h,r=inputs();base=r.drop(columns=AGES);x=transform(base,s,h,t,'T1',MAIN)
    assert len(x.columns)==len(base.columns)+8
    pd.testing.assert_frame_equal(x[base.columns],base)
    for group in ('all','spout'):
        assert x.loc[0,f'history__{group}__tap_iron__mean10_minus_median30']==10.
        assert x.loc[0,f'history__{group}__tap_iron__median30_minus_median100']==5.
        assert x.loc[0,f'history__{group}__tap_time_len__mean10_minus_median30']==20.
    assert x.iloc[1,-8:].isna().all()
    assert not any(c in x for c in AGES)


def test_T2_only_removes_auxiliary_ages():
    t,s,h,r=inputs();x=transform(r,s,h,t,'T2',AUX)
    pd.testing.assert_frame_equal(x,r.drop(columns=AGES))
    assert x.loc[0,'history__all__tap_iron__last100_count']==40.


@pytest.mark.parametrize('variant,component',[('T1',AUX),('T2',MAIN),('T3',MAIN)])
def test_wrong_component_variant_rejected(variant,component):
    t,s,h,r=inputs()
    with pytest.raises(ContractError):transform(r,s,h,t,variant,component)


def test_fit_budget_counts_attempts_and_rejects_repeated_slot(tmp_path):
    p=tmp_path/'budget.jsonl'
    assert reserve_fit(p,{'slot':'T1/June/O','schema_sha256':'one'},limit=4)==2
    with pytest.raises(ContractError,match='already attempted'):
        reserve_fit(p,{'slot':'T1/June/O','schema_sha256':'changed'},limit=4)
    assert reserve_fit(p,{'slot':'T2/June/H'},limit=4)==4
    with pytest.raises(ContractError,match='exhausted'):reserve_fit(p,{'slot':'T1/July/O'},limit=4)
    assert sum(json.loads(x)['target_fits'] for x in p.read_text().splitlines())==4


def test_OPT16_refuses_failed_gate_without_creating_output(tmp_path):
    from bf_tap.optimization.component_release_v5 import release
    run=tmp_path/'run';run.mkdir()
    (run/'acceptance.json').write_text(json.dumps({'selected_STAGE_A':None}))
    (run/'final_status.json').write_text(json.dumps({'G0':'PASS_OPT15'}))
    out=tmp_path/'out'
    with pytest.raises(ContractError,match='passing challenger'):release(run,'missing.yaml',out)
    assert not out.exists()


@pytest.mark.parametrize('defect',['digest','weights','variant','components'])
def test_mixed_bundle_identity_guards(tmp_path,defect):
    from bf_tap.optimization.component_release_v5 import Predictor
    from bf_tap.artifacts import file_sha256
    m={'schema':'v5-component-composite-v1','candidate':'T1','weights':{MAIN:.8,AUX:.2},
       'calibration':'none','implementation':{},'components':{MAIN:{'variant':'R2'},AUX:{'variant':'R2'}}}
    if defect=='weights':m['weights'][MAIN]=.75
    if defect=='components':del m['components'][AUX]
    p=tmp_path/'composite.json';p.write_text(json.dumps(m))
    (tmp_path/'identity.json').write_text(json.dumps({'sha256':'wrong' if defect=='digest' else file_sha256(p)}))
    with pytest.raises(ContractError):Predictor(tmp_path)
