import numpy as np
import pandas as pd
import pytest
from bf_tap.exceptions import ContractError
from bf_tap.optimization.component_export import forbid_fit, ComponentFeatures, load_verified_component
from bf_tap.optimization.component_ablation import fixed_combinations, representation_effect, validate_frame, gates
from bf_tap.models.baseline import DualTargetBaseline


def parts():
    return {k:pd.DataFrame({'sample_id':['b','a'],'pred_tap_iron':[v,v+1.],
                           'pred_tap_time_len':[v*2,v*2+1.]}) for k,v in [('O0',4.),('H0',6.),('OR',8.),('HR',10.)]}


def test_fixed_combinations_and_reordering():
    p=parts();x=fixed_combinations(p)
    y=fixed_combinations({k:v.iloc[::-1] for k,v in p.items()})
    for c in x:
        pd.testing.assert_frame_equal(x[c].sort_values('sample_id').reset_index(drop=True),y[c].sort_values('sample_id').reset_index(drop=True))
    assert x['P1'].set_index('sample_id').loc['b','pred_tap_iron']==pytest.approx(7.6)
    assert x['P2'].set_index('sample_id').loc['b','pred_tap_iron']==pytest.approx(5.2)
    pd.testing.assert_frame_equal(x['P3'],p['OR'])


@pytest.mark.parametrize('defect',['missing','duplicate','nan','inf','negative','id'])
def test_component_rejections(defect):
    p=parts()
    if defect=='missing': del p['H0']
    if defect=='duplicate': p['H0'].loc[1,'sample_id']='b'
    if defect=='id': p['H0'].loc[1,'sample_id']='c'
    if defect in ('nan','inf','negative'): p['H0'].loc[0,'pred_tap_iron']={'nan':np.nan,'inf':np.inf,'negative':-1.}[defect]
    with pytest.raises(ContractError): fixed_combinations(p)


def test_representation_identity():
    d=representation_effect({'E12-raw':.2,'P1':.17,'P2':.21,'R2':.18})
    assert d['D_O']+d['D_H']==pytest.approx(d['joint'])
    assert d['D_O']==pytest.approx(-.03)
    assert d['D_H']==pytest.approx(.01)


def test_fit_disabled():
    counter={'attempted_target_fits':0}
    with forbid_fit(counter),pytest.raises(ContractError,match='forbids fit'):
        DualTargetBaseline.fit(None,None,None)
    assert counter['attempted_target_fits']==1


def test_missing_model_never_retrained(tmp_path):
    with pytest.raises(ContractError,match='missing'):
        load_verified_component({'path':str(tmp_path),'bundle_sha256':'x'},None,None,None)


def test_builder_refuses_label_columns():
    b=ComponentFeatures(None,None,None)
    with pytest.raises(ContractError,match='metadata only'):
        b.X(pd.DataFrame({'sample_id':['x'],'spout_no':[1],'reference_time':[pd.Timestamp('2024-07-01')],'tap_iron':[1.]}),{},None)


def test_zero_denominator():
    p=parts()['O0'];p['tap_iron']=0.;p['tap_time_len']=2.;p['reference_time']='2024-07-01';p['spout_no']=1
    with pytest.raises(ContractError,match='denominator'): validate_frame(p,labeled=True)


def test_stage_a_and_general_are_distinct():
    from bf_tap.config import load_yaml
    from bf_tap.optimization.component_export import units
    from bf_tap.optimization.validation import aggregate_grid
    registration=load_yaml('configs/optimization_v0_5/experiment.yaml')
    metrics={}
    for unit,cutoff,horizon,start,end in units(registration):
        values={}
        for c in ['R2','E12-raw','P1','P2','P3','B0','B1']:
            loss=.2
            if c in ('B0','B1'): loss=.3
            if c=='P1' and horizon==1: loss-=.002
            if c=='P2' and horizon==1: loss+=(-.005 if cutoff.month<=8 else .0001)
            if c=='P3' and horizon==1: loss-=.002
            if c=='P3' and horizon==2: loss+=.0016
            values[c]={'overall':{'loss':loss,'iron':{'wmape':loss},'time':{'wmape':loss}}}
        metrics[unit]={'horizon':horizon,'candidates':values}
    result=gates(metrics,aggregate_grid(metrics),registration)
    assert result['selected_STAGE_A']=='P1'
    assert result['candidates']['P1']['STAGE_A']['passed']
    assert not result['candidates']['P1']['GENERAL']['passed']
    assert not result['candidates']['P2']['STAGE_A']['checks']['improved_H1_origins']
    assert not result['candidates']['P3']['STAGE_A']['checks']['far_horizon_regression']


@pytest.mark.parametrize('defect',['future','current'])
def test_prediction_history_boundary(defect):
    t=pd.Timestamp('2024-07-01',tz='Asia/Shanghai')
    samples=pd.DataFrame({'sample_id':['x'],'spout_no':[1],'reference_time':[t]})
    history=pd.DataFrame({'sample_id':['x' if defect=='current' else 'h'],
                          'available_at':[t+pd.Timedelta(days=1) if defect=='future' else t]})
    with pytest.raises(ContractError): ComponentFeatures(None,None,None).X(samples,{'cutoff':str(t)},history)


@pytest.mark.parametrize('defect',['cutoff','source','component'])
def test_verified_loader_rejects_wrong_identity(tmp_path,monkeypatch,defect):
    from types import SimpleNamespace
    from bf_tap.artifacts import file_sha256
    p=tmp_path/'bundle.json';p.write_text('{}')
    t='2024-07-01T00:00:00+08:00'
    tr={'fit_cutoff':t,'history_cutoff':t,'label_available_cutoff':t,'variant':'R2','component':'E04'}
    md={'training':tr,'inference_source_contract':{'x':1},'contract_digests':{}}
    if defect=='cutoff': tr['fit_cutoff']='2024-08-01T00:00:00+08:00'
    if defect=='component': tr['component']='E09_PROCESS_CHANGE_E02'
    monkeypatch.setattr(DualTargetBaseline,'load',lambda _:SimpleNamespace(bundle_metadata_=md))
    identity={'path':str(tmp_path),'bundle_sha256':file_sha256(p),'cutoff':t,'variant':'R2','component':'E04'}
    with pytest.raises(ContractError): load_verified_component(identity,{'contract_digests':{}},{'x':2},None)
