import copy
import numpy as np
import pandas as pd
import pytest
from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.component_export import META, units
from bf_tap.optimization.qrf_time_run import (pack, fit_counts, acceptance, CANDIDATE, DIAGNOSTIC, registration, outer_samples)
from bf_tap.optimization.v13_common import zero_fit


def fixture_frames():
    cutoff=pd.Timestamp('2024-06-01',tz='Asia/Shanghai')
    samples=pd.DataFrame({'sample_id':['a','b'],'spout_no':['1','2'],
        'reference_time':[cutoff-pd.Timedelta(hours=2),cutoff-pd.Timedelta(hours=1)]})
    x=pd.DataFrame({'x':[1.,np.nan],'spout_no':['1','2']})
    history=samples.copy();history['tap_time_len']=[10.,20.];history['available_at']=samples.reference_time+pd.Timedelta(minutes=10)
    return cutoff,samples,x,history


def test_fixed_registration_calendar_and_no_capacity_change():
    reg=registration()
    assert reg['forest_parameters']['n_estimators']==256 and reg['quantile']==.5
    assert reg['budget']['LAD']==0
    grid=units(reg)
    assert sum(v[2]==2 for v in grid)==5
    assert grid[-2][1].month==7 and grid[-1][1].month==9


def test_train_eval_handoffs_separate(tmp_path):
    c,s,x,h=fixture_frames()
    info=pack(tmp_path/'train.npz',x,s,c,h)
    with np.load(tmp_path/'train.npz',allow_pickle=False) as f:
        assert f['y'].tolist()==[10.,20.] and f['ids'].tolist()==info['ids']
    s.reference_time += pd.Timedelta(days=1)
    pack(tmp_path/'eval.npz',x,s,c)
    with np.load(tmp_path/'eval.npz',allow_pickle=False) as f:
        assert 'y' not in f and 'available_ns' not in f
    with pytest.raises(ContractError,match='overwrite'): pack(tmp_path/'train.npz',x,s,c)


@pytest.mark.parametrize('fault',['future_ref','future_available','duplicate','misaligned_labels','Inf','dtype','overflow','missing_available','negative_y','feature_order'])
def test_illegal_training_payload_rejected(tmp_path,fault):
    c,s,x,h=fixture_frames()
    if fault=='future_ref': h.loc[0,'reference_time']=c
    elif fault=='future_available': h.loc[0,'available_at']=c+pd.Timedelta(seconds=1)
    elif fault=='duplicate': s.loc[1,'sample_id']='a'
    elif fault=='misaligned_labels': h=h.iloc[::-1]
    elif fault=='Inf': x.loc[0,'x']=np.inf
    elif fault=='dtype': x['x']=['bad','data']
    elif fault=='overflow': x.loc[0,'x']=1e100
    elif fault=='missing_available': h.loc[0,'available_at']=pd.NaT
    elif fault=='negative_y': h.loc[0,'tap_time_len']=-1.
    elif fault=='feature_order': x=x.iloc[::-1]
    with pytest.raises(ContractError): pack(tmp_path/'bad.npz',x,s,c,h)


def synthetic_metrics(delta=-.001):
    reg=registration();metrics={}
    for unit,cutoff,h,start,end in units(reg):
        metrics[unit]={'horizon':h,'candidates':{'V1':{'overall':{'loss':.2}},
            CANDIDATE:{'overall':{'loss':.2+delta}},DIAGNOSTIC:{'overall':{'loss':.1}}}}
    summary={'V1':{'J':.2},CANDIDATE:{'J':.2+delta}}
    return reg,metrics,summary


def test_acceptance_diagnostic_never_selectable_and_iron_exact():
    reg,m,s=synthetic_metrics()
    result=acceptance(m,s,reg,True,True)
    assert result['historical_quality_passed'] and not result['diagnostic_releasable']
    assert not result['ready_challenger'] and not result['official_data_identity_verified']
    assert result['H2_improved']==5 and result['recent_H2_improved']==3
    assert not acceptance(m,s,reg,True,False)['historical_quality_passed']


@pytest.mark.parametrize('gate',['H2','J','single','DEV','H3','engineering'])
def test_each_hard_gate_is_required(gate):
    reg,m,s=synthetic_metrics()
    if gate=='H2':
        for v in m.values(): v['candidates'][CANDIDATE]['overall']['loss']=.2
    elif gate=='J': s[CANDIDATE]['J']=.2003
    elif gate=='single': m['O202406_H2']['candidates'][CANDIDATE]['overall']['loss']=.202
    elif gate=='DEV': m['DEV_LONG']['candidates'][CANDIDATE]['overall']['loss']=.201
    elif gate=='H3':
        for v in m.values():
            if v['horizon']==3: v['candidates'][CANDIDATE]['overall']['loss']=.201
    assert not acceptance(m,s,reg,gate!='engineering',True)['historical_quality_passed']


def test_budget_counts_attempts_and_refuses_seventh(tmp_path):
    for i in range(7):
        d=tmp_path/'models'/str(i);d.mkdir(parents=True);(d/'forest_intent.json').write_text('{}')
    with pytest.raises(ContractError,match='budget'): fit_counts(tmp_path)


def test_old_fit_interfaces_remain_disabled():
    from bf_tap.models.baseline import DualTargetBaseline
    from bf_tap.optimization import structural
    with zero_fit() as count:
        from bf_tap.models.baseline import FROZEN_PARAMETERS
        with pytest.raises(ContractError): DualTargetBaseline(FROZEN_PARAMETERS).fit(None,None)
        with pytest.raises(ContractError): structural.lad_coefficient(None,None)
    assert count['attempted_target_fits']==1 and count['attempted_calibration_fits']==1


def test_metadata_recovered_from_oof_not_prediction_columns(tmp_path):
    src=tmp_path/'source';(src/'oof').mkdir(parents=True);(src/'predictions').mkdir()
    samples=pd.DataFrame({'sample_id':['a','b'],'spout_no':['1','2'],
        'reference_time':['2024-06-01T01:00:00+08:00','2024-06-02T01:00:00+08:00']})
    samples.to_csv(src/'oof/6.csv',index=False)
    pd.DataFrame({'sample_id':['b','a'],'pred_tap_iron':[1.,2.]}).to_csv(src/'predictions/6_inputs.csv',index=False)
    manifest={'registration':{'source_v8':str(src),'origins':{6:1}}}
    out=outer_samples(manifest,6)
    assert list(out)==META and out.sample_id.tolist()==['b','a']
    pd.DataFrame({'sample_id':['wrong']}).to_csv(src/'predictions/6_inputs.csv',index=False)
    with pytest.raises(ContractError,match='IDs differ'): outer_samples(manifest,6)
