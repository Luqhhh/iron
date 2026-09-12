import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from bf_tap.artifacts import stable_digest, build_inference_source_contract, validate_inference_source_contract
from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization import structural
from bf_tap.optimization.component_export import META, PRED
from bf_tap.optimization.horizon_calibration import (CANDIDATE,CONTROL,calendar_window,month_start,
    verify_bank,verify_pair,select_matched,predict_time,lad_certificate,ScalarBudget,
    no_tree_or_dual_fit,acceptance,require_official_identity,read_bank)
from bf_tap.optimization.v13_common import zero_fit
from bf_tap.optimization.validation import aggregate_grid


def pair(n=105):
    reference=pd.date_range('2024-05-01 01:00',periods=n,freq='3h',tz='Asia/Shanghai')
    records={str(m):{'fit_cutoff':str(month_start(m)),'reference_max':str(month_start(m)-pd.Timedelta(hours=4)),
        'label_available_max':str(month_start(m)-pd.Timedelta(hours=2)),
        'history_available_max':str(month_start(m)-pd.Timedelta(hours=2))} for m in (4,5)}
    banks=[]
    for horizon,month in ((2,4),(1,5)):
        record=records[str(month)]
        b=pd.DataFrame({'sample_id':[f'i{x:03d}' for x in range(n)],'spout_no':[1]*n,
            'reference_time':reference,'label_available_at':reference+pd.Timedelta(hours=2),
            'pred_tap_iron':np.arange(n)+200.,'pred_tap_time_len':np.arange(n)+100.,'pred_rate':2.,
            'fold_cutoff':month_start(month),'train_reference_max':pd.Timestamp(record['reference_max']),
            'train_available_max':pd.Timestamp(record['label_available_max']),
            'history_available_max':pd.Timestamp(record['history_available_max']),
            'evaluation_month_start':month_start(5),'evaluation_month_end':month_start(6),'horizon':horizon,
            'fold_identity_sha256':stable_digest(record)})
        banks.append(b)
    history=banks[0][META].copy();history['available_at']=banks[0].label_available_at
    history['tap_iron']=300.;history['tap_time_len']=110.
    return (*banks,history,records)


@pytest.mark.parametrize('month,days',[(4,31),(5,30),(6,31),(7,31),(8,30),(9,31),(10,30)])
def test_h2_is_target_calendar_month(month,days):
    a,b=calendar_window(month_start(month),2)
    assert a.month==month+1 and (b-a).days==days


@pytest.mark.parametrize('cutoff',['2024-04-01','2024-04-02 00:00+08:00','2025-04-01 00:00+08:00'])
def test_non_original_or_naive_cutoff_rejected(cutoff):
    with pytest.raises(ContractError):calendar_window(cutoff,2)


def test_genuine_h2_h1_pair_and_same_available_labels():
    h2,h1,history,records=pair()
    assert verify_bank(h2,2,records) and verify_bank(h1,1,records) and verify_pair(h2,h1)
    a,b=select_matched(h2,h1,history,month_start(6))
    assert a.sample_id.tolist()==b.sample_id.tolist() and np.array_equal(a.tap_time_len,b.tap_time_len)
    assert (a.fold_cutoff==month_start(4)).all() and (b.fold_cutoff==month_start(5)).all()


@pytest.mark.parametrize('mutation',['fake_cutoff','train_future','history_future','outside_month','null','duplicate','target','identity'])
def test_forecast_certificate_rejects_leakage_or_forgery(mutation):
    h2,_,_,records=pair()
    if mutation=='fake_cutoff':h2['fold_cutoff']=month_start(5)
    elif mutation=='train_future':h2.loc[0,'train_reference_max']=month_start(4)
    elif mutation=='history_future':h2.loc[0,'history_available_max']=month_start(5)
    elif mutation=='outside_month':h2.loc[0,'reference_time']=month_start(6)
    elif mutation=='null':h2.loc[0,'label_available_at']=pd.NaT
    elif mutation=='duplicate':h2.loc[0,'sample_id']=h2.loc[1,'sample_id']
    elif mutation=='target':h2['tap_time_len']=1.
    else:h2.loc[0,'fold_identity_sha256']='forged'
    with pytest.raises(ContractError):verify_bank(h2,2,records)


@pytest.mark.parametrize('column',['sample_id','spout_no','reference_time','label_available_at'])
def test_pair_changes_rejected(column):
    h2,h1,_,_=pair()
    if column=='sample_id':h1.loc[0,column]='other'
    elif column=='spout_no':h1.loc[0,column]=2
    else:h1.loc[0,column]+=pd.Timedelta(minutes=1)
    with pytest.raises(ContractError):verify_pair(h2,h1)


def test_month_end_availability_filters_both_banks_identically():
    h2,h1,h,_=pair()
    h2.loc[104,'label_available_at']=h1.loc[104,'label_available_at']=month_start(6)+pd.Timedelta(minutes=1)
    h=h.iloc[:104]
    a,b=select_matched(h2,h1,h,month_start(6))
    assert len(a)==len(b)==104 and 'i104' not in set(a.sample_id)


@pytest.mark.parametrize('case',['insufficient','history_future','history_reference_future','missing_label','label_mismatch','incomplete_window'])
def test_invalid_coefficient_sources_fail(case):
    h2,h1,h,_=pair()
    if case=='insufficient':h2,h1,h=h2.iloc[:99],h1.iloc[:99],h.iloc[:99]
    elif case=='history_future':h.loc[0,'available_at']=month_start(6)+pd.Timedelta(minutes=1)
    elif case=='history_reference_future':h.loc[0,'reference_time']=month_start(6)
    elif case=='missing_label':h=h.iloc[1:]
    elif case=='label_mismatch':h.loc[0,'tap_time_len']=np.nan
    else:h2['evaluation_month_end']=h1['evaluation_month_end']=month_start(7)
    with pytest.raises(ContractError):select_matched(h2,h1,h,month_start(6))


def test_exact_iron_original_direction_and_rate_fallback():
    inputs=pd.DataFrame({'sample_id':['a','b','c'],'pred_tap_iron':[200.,300.,400.],
        'pred_tap_time_len':[80.,100.,110.],'pred_rate':[2.,0.,1e-6]})
    old=pd.DataFrame({'sample_id':['c','a','b'],'pred_tap_iron':[999.,777.,888.],'pred_tap_time_len':[99.,88.,77.]})
    p,fallback=predict_time(inputs,old,.5)
    assert p.pred_tap_iron.tolist()==[777.,888.,999.] and p.pred_tap_time_len.tolist()==[90.,100.,110.] and fallback==2
    changed=old.copy();changed[PRED[1]]=10000.
    q,_=predict_time(inputs,changed,.5)
    assert p.equals(q)
    inputs2=inputs.copy();inputs2.loc[0,PRED[0]]=100.
    r,_=predict_time(inputs2,old,.5)
    assert r.loc[0,PRED[1]]==65. and r.loc[0,PRED[0]]==777.


@pytest.mark.parametrize('beta',[-1.,1.01,np.nan,np.inf])
def test_illegal_coefficient(beta):
    h2,_,_,_=pair()
    with pytest.raises(ContractError):predict_time(h2[structural.INPUT],h2[['sample_id',*PRED]],beta)


def test_serialized_order_chunk_subset_single_exact(tmp_path):
    h2,_,_,_=pair();base=h2[structural.INPUT];old=h2[['sample_id',*PRED]]
    p,_=predict_time(base,old,.1234567890123456)
    for selected in (base.iloc[::-1],base.iloc[::7],base.iloc[[0]],base.iloc[:31]):
        q,_=predict_time(selected,old.loc[old.sample_id.isin(selected.sample_id)],.1234567890123456)
        assert q.set_index('sample_id').equals(p.set_index('sample_id').loc[q.sample_id])
    path=tmp_path/'prediction.csv';p.to_csv(path,index=False)
    assert read_bank(path).equals(p)


def test_scalar_budget_success_resume_no_refit_and_twelve_attempts(tmp_path,monkeypatch):
    h2,h1,h,_=pair();a,b=select_matched(h2,h1,h,month_start(6));budget=ScalarBudget(tmp_path)
    for m in range(6,12):
        for role,x in ((CANDIDATE,a),(CONTROL,b)):budget.fit(role,m,x)
    assert sum(v['attempted'] for v in budget.counts().values())==12
    def deny(*args):raise AssertionError('must reuse stored coefficient')
    monkeypatch.setattr(structural,'lad_coefficient',deny)
    saved=budget.fit(CANDIDATE,6,a)
    assert lad_certificate(a,saved['beta'])==saved['certificate']
    with pytest.raises(ContractError):budget.fit(CANDIDATE,12,a)
    a2=a.copy();a2.loc[a2.index[0],'tap_time_len']+=1
    with pytest.raises(ContractError):budget.fit(CANDIDATE,6,a2)


def test_failed_attempt_is_accounted_and_cannot_retry(tmp_path,monkeypatch):
    h2,h1,h,_=pair();a,_=select_matched(h2,h1,h,month_start(6));budget=ScalarBudget(tmp_path)
    def fail(*args):raise ContractError('synthetic failed primitive')
    monkeypatch.setattr(structural,'lad_coefficient',fail)
    with pytest.raises(ContractError):budget.fit(CANDIDATE,6,a)
    assert budget.counts()[CANDIDATE]=={'attempted':1,'completed':0}
    with pytest.raises(ContractError):budget.fit(CANDIDATE,6,a)


@pytest.mark.parametrize('direction,actual,expected',[([0.,0.],[1.,2.],0.),([1.,1.],[.2,.8],.2),([1.,1.],[2.,3.],1.),([1.,1.],[-2.,-1.],0.)])
def test_original_lad_smallest_tie_and_zero(direction,actual,expected):
    assert structural.lad_coefficient(actual,[0.,0.],direction)==expected


def test_certificate_rejects_non_smallest_tied_coefficient():
    selected=pd.DataFrame({'sample_id':['a','b'],'pred_tap_iron':[2.,2.],
        'pred_tap_time_len':[1.,1.],'pred_rate':[1.,1.],'tap_time_len':[1.2,1.8]})
    assert lad_certificate(selected,structural.lad_coefficient(selected.tap_time_len,[1.,1.],[1.,1.]))['smallest_minimizer_verified']
    with pytest.raises(ContractError):lad_certificate(selected,.5)


def test_tree_and_implicit_two_target_fit_are_rejected():
    from bf_tap.models.baseline import DualTargetBaseline,FROZEN_PARAMETERS
    with no_tree_or_dual_fit() as counts:
        with pytest.raises(ContractError):DualTargetBaseline(FROZEN_PARAMETERS).fit(None,None)
        with pytest.raises(ContractError):structural.fit_correction(None,None)
    assert counts=={'attempted_target_fits':1,'attempted_dual_calibration_fits':1}


def test_cold_fit_guards_include_scalar_budget(tmp_path):
    h2,h1,h,_=pair();a,_=select_matched(h2,h1,h,month_start(6))
    with zero_fit() as counts:
        with pytest.raises(ContractError):ScalarBudget(tmp_path).fit(CANDIDATE,6,a)
    assert counts['attempted_calibration_fits']==1 and counts['completed_calibration_fits']==0


def metric(loss):
    return {'loss':loss,'iron':{'wmape':.2,'actual_sum':1000.},'time':{'wmape':2*loss-.2,'actual_sum':1000.}}


def grid(delta=-.001):
    metrics={}
    for month,n in {6:4,7:4,8:4,9:3,10:2,11:1}.items():
        for horizon in range(1,n+1):
            metrics[f'O2024{month:02d}_H{horizon}']={'horizon':horizon,'candidates':
                {'V1':{'overall':metric(.2)},CANDIDATE:{'overall':metric(.2+delta if horizon==2 else .2)},
                 CONTROL:{'overall':metric(.2)}}}
    for name in ('DEV_LONG','DEV_SHORT'):metrics[name]={'horizon':None,'candidates':
        {'V1':{'overall':metric(.2)},CANDIDATE:{'overall':metric(.2)},CONTROL:{'overall':metric(.2)}}}
    return metrics


def test_full_grid_equal_horizon_J_h2_recent_months_and_no_release():
    reg=load_yaml('configs/optimization_v0_14/experiment.yaml');metrics=grid();summary=aggregate_grid(metrics)
    assert abs(summary[CANDIDATE]['J']-.19975)<1e-12
    result=acceptance(metrics,summary,reg,True,True)
    assert result['historical_quality_passed'] and result['H2_improved']==5 and result['recent_H2_improved']==3
    assert result['status']=='DEV_ACCEPTED_PENDING_OFFICIAL_IDENTITY' and not result['ready_challenger']
    with pytest.raises(ContractError):require_official_identity(result)


@pytest.mark.parametrize('case',['small_gain','control_better','iron','engineering','H1','single_origin','DEV'])
def test_no_failed_candidate_can_be_challenger(case):
    reg=load_yaml('configs/optimization_v0_14/experiment.yaml');metrics=grid();iron=True;engineering=True
    if case=='small_gain':metrics=grid(-.0001)
    elif case=='control_better':
        for v in metrics.values():v['candidates'][CONTROL]['overall']=metric(.19)
    elif case=='iron':iron=False
    elif case=='engineering':engineering=False
    elif case=='H1':
        for k,v in metrics.items():
            if k.endswith('_H1'):v['candidates'][CANDIDATE]['overall']=metric(.201)
    elif case=='single_origin':metrics['O202406_H2']['candidates'][CANDIDATE]['overall']=metric(.202)
    else:metrics['DEV_LONG']['candidates'][CANDIDATE]['overall']=metric(.201)
    result=acceptance(metrics,aggregate_grid(metrics),reg,engineering,iron)
    assert result['status']=='FAIL_CLOSE_V7_RETAIN_V1' and not result['historical_quality_passed'] and not result['ready_challenger']


def test_changed_official_process_source_is_blocked():
    identities={k:{'sha256':'a'*64,'bytes':123} for k in ('operation_hourly','burden_change')}
    contract=build_inference_source_contract(identities,semantic_contract_sha256='b'*64)
    revised=copy.deepcopy(identities);revised['operation_hourly']['sha256']='c'*64
    with pytest.raises(ContractError):validate_inference_source_contract(contract,revised,semantic_contract_sha256='b'*64)


def test_serialized_manifest_origin_keys_restore_integer_model_identity(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('v14_runner',Path('scripts/optimization_v14_h2_calibration.py'))
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    (tmp_path/'manifest.json').write_text(json.dumps({'registration':{'origins':{6:4,7:4,8:4,9:3,10:2,11:1}}}))
    manifest=runner.execution_manifest(tmp_path)
    assert manifest['registration']['origins']=={6:4,7:4,8:4,9:3,10:2,11:1}
    assert {m:object() for m in range(6,12)}[next(iter(manifest['registration']['origins']))] is not None
