import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from bf_tap.exceptions import ContractError
from bf_tap.optimization.recency_model import weights,weight_audit,RecencyModel,original_schema
from bf_tap.optimization.recency_candidate import (IRON,TIME,BOTH,CANDIDATES,PARTS,isolated_directions,
    outputs,validate_outputs,fit_coefficients,acceptance,choose,additive)
from bf_tap.optimization.recency_run import no_estimators,archive,matrix_digest
from bf_tap.optimization.rate_model import schema
from bf_tap.optimization.residual_stack import ResidualFitBudget
from bf_tap.optimization.component_export import PRED,units
from bf_tap.optimization.validation import aggregate_grid
from bf_tap.models.baseline import DualTargetBaseline,FROZEN_PARAMETERS
from bf_tap.config import load_yaml


def metadata(ages=(60.,120.,180.)):
    cutoff=pd.Timestamp('2024-11-01',tz='Asia/Shanghai')
    ref=[cutoff-pd.Timedelta(days=d) for d in ages]
    m=pd.DataFrame(dict(sample_id=[str(i) for i in range(len(ages))],reference_time=ref,available_at=[t+pd.Timedelta(minutes=30) for t in ref],spout_no=['1' if i%2 else '2' for i in range(len(ages))]))
    return m,cutoff


def test_weights_fixed_days_hours_mean_and_reference_only():
    m,c=metadata();w,d=weights(m,c)
    np.testing.assert_array_equal(d.raw_weight,[.5,.25,.125]);assert w.mean()==pytest.approx(1.)
    m.available_at+=pd.Timedelta(hours=4)
    assert weights(m,c)[0].equals(w)
    m,c=metadata((.5,1.,1.5));_,d=weights(m,c)
    np.testing.assert_allclose(d.raw_weight,np.exp2(-np.array([.5,1.,1.5])/60),rtol=0,atol=0)


def test_weights_id_shuffle_and_ess_are_diagnostics():
    m,c=metadata();w,d=weights(m,c)
    shuffled=weights(m.iloc[::-1],c)[0]
    pd.testing.assert_series_equal(w.sort_index(),shuffled.sort_index())
    audit=weight_audit(m,w);assert 0<audit['global_summary']['ESS_over_n']<=1
    assert sum(audit['global_summary']['month_weight_share'].values())==pytest.approx(1.)
    assert any(abs(v['mean']-1)>1e-3 for v in audit['by_spout'].values())


@pytest.mark.parametrize('bad', ['duplicate','missing_id','missing_time','future_ref','at_cutoff','future_available','available_before_ref','naive'])
def test_weight_invalid_metadata_rejected(bad):
    m,c=metadata()
    if bad=='duplicate':m.loc[1,'sample_id']=m.loc[0,'sample_id']
    if bad=='missing_id':m.loc[0,'sample_id']=None
    if bad=='missing_time':m.loc[0,'reference_time']=pd.NaT
    if bad=='future_ref':m.loc[0,'reference_time']=c+pd.Timedelta(hours=1)
    if bad=='at_cutoff':m.loc[0,'reference_time']=c
    if bad=='future_available':m.loc[0,'available_at']=c+pd.Timedelta(seconds=1)
    if bad=='available_before_ref':m.loc[0,'available_at']=m.loc[0,'reference_time']-pd.Timedelta(seconds=1)
    if bad=='naive':m.reference_time=m.reference_time.dt.tz_localize(None)
    with pytest.raises(ContractError):weights(m,c)


def inputs():
    old=pd.DataFrame(dict(sample_id=['a','b','c'],pred_tap_iron=[100.,200.,300.],pred_tap_time_len=[5.,6.,7.],pred_rate=[10.,0.,1e-6]))
    base=pd.DataFrame(dict(sample_id=['a','b','c'],base_iron=[80.,180.,280.],base_time=[4.,5.,6.]))
    direction,fallback=isolated_directions(old,base)
    parts=old.merge(base,on='sample_id').merge(direction,on='sample_id');parts['direct_iron']=70.;parts['direct_time']=3.
    original=pd.DataFrame(dict(sample_id=['c','b','a'],pred_tap_iron=[302.,204.,106.],pred_tap_time_len=[17.,16.,15.]))
    return old,base,parts[PARTS],original


def test_isolated_directions_do_not_cross_new_inputs_and_fallback():
    old,base,parts,v=inputs();d,n=isolated_directions(old,base)
    np.testing.assert_array_equal(d.direction_iron,[-30.,0.,0.]);np.testing.assert_array_equal(d.direction_time,[6.,0.,0.]);assert n==2
    changed=base.copy();changed.base_time*=9
    np.testing.assert_array_equal(isolated_directions(old,changed)[0].direction_iron,d.direction_iron)
    changed=base.copy();changed.base_iron*=9
    np.testing.assert_array_equal(isolated_directions(old,changed)[0].direction_time,d.direction_time)
    shuffled=isolated_directions(old,base.iloc[::-1])[0];pd.testing.assert_frame_equal(d,shuffled)


def test_exact_candidate_targets_both_composition_reversal_subset():
    _,_,parts,v=inputs();beta={'tap_iron':.5,'tap_time_len':.5}
    result=outputs(parts,v,beta);assert validate_outputs(result,v)
    assert result[IRON].pred_tap_iron.tolist()==[65.,180.,280.]
    assert result[TIME].pred_tap_time_len.tolist()==[7.,5.,6.]
    rev=outputs(parts.iloc[::-1],v,beta);sub=outputs(parts.iloc[[1]],v.loc[v.sample_id=='b'],beta)
    for c in CANDIDATES:
        pd.testing.assert_frame_equal(result[c],rev[c].iloc[::-1].reset_index(drop=True))
        pd.testing.assert_frame_equal(result[c].iloc[[1]].reset_index(drop=True),sub[c])
    parts.loc[0,'direction_iron']+=1
    with pytest.raises(ContractError,match='cross-input'):outputs(parts,v,beta)


def test_no_estimators_rejects_base_and_lad():
    import bf_tap.optimization.recency_candidate as candidate
    with no_estimators() as counts:
        with pytest.raises(ContractError):DualTargetBaseline(FROZEN_PARAMETERS).fit(None,None)
        with pytest.raises(ContractError):candidate.fit_coefficients(None,None)
    assert counts=={'attempted_target_fits':1,'attempted_LAD_fits':1}


@pytest.mark.model
def test_weight_alignment_persistence_original_schema_and_fit_budget(tmp_path):
    m,c=metadata(tuple(range(1,9)));w,_=weights(m,c)
    x=pd.DataFrame(dict(spout_no=pd.Series(['1','2']*4,dtype='string'),x=np.arange(8,dtype=float)));x.index=w.index
    y=pd.Series(np.arange(8,dtype=float)+10,index=w.index,name='tap_time_len');expected=schema(x)
    for altered in (w.iloc[::-1],w*0,w*np.nan,w*np.inf):
        with ResidualFitBudget(1) as b:
            with pytest.raises(ContractError):RecencyModel().fit(x,y,altered,m,c,y.name,expected,b)
            assert b.completed==0
    with ResidualFitBudget(1) as b:
        with pytest.raises(ContractError):RecencyModel().fit(x,y.iloc[::-1],w,m,c,y.name,expected,b)
        model=RecencyModel().fit(x,y,w,m,c,y.name,expected,b)
        model.save(tmp_path/'model',dict(synthetic=True));restored=RecencyModel.load(tmp_path/'model')
        np.testing.assert_array_equal(model.predict(x),restored.predict(x))
        np.testing.assert_array_equal(model.predict(x),restored.predict(x.iloc[::-1])[::-1])
        np.testing.assert_array_equal(model.predict(x)[::2],restored.predict(x.iloc[::2]))
        assert b.completed==1
        with pytest.raises(ContractError):DualTargetBaseline(FROZEN_PARAMETERS).fit(x,None)
    for column in ('trajectory__air_volume__6h__slope','weight','reference_time'):
        dirty=x.copy();dirty[column]=1.
        with pytest.raises(ContractError):original_schema(dirty,schema(dirty))
    assert matrix_digest(x)==matrix_digest(x.copy())


@pytest.mark.model
def test_synthetic_all_ones_matches_unweighted_small_model():
    x=pd.DataFrame({'x':np.arange(12,dtype=float)});y=np.arange(12,dtype=float)**.5
    params=dict(iterations=6,depth=2,loss_function='MAE',random_seed=2026,verbose=False,allow_writing_files=False,thread_count=2)
    a=CatBoostRegressor(**params).fit(x,y);b=CatBoostRegressor(**params).fit(x,y,sample_weight=np.ones(len(x)))
    np.testing.assert_array_equal(a.predict(x),b.predict(x))


def test_causal_lad_earlier_available_only_and_smallest_tie():
    _,_,parts,v=inputs();parts=parts.iloc[[0,1,2]].copy();cutoff=pd.Timestamp('2024-06-01',tz='Asia/Shanghai');old=cutoff-pd.Timedelta(days=20);fold=cutoff-pd.Timedelta(days=31)
    for column,value in [('reference_time',old),('fold_cutoff',fold),('label_available_at',old),('train_reference_max',fold-pd.Timedelta(hours=1)),('train_available_max',fold),('history_available_max',fold)]:parts[column]=value
    parts['tap_iron']=[65.,180.,280.];parts['tap_time_len']=[7.,5.,6.]
    parts.loc[2,'label_available_at']=cutoff+pd.Timedelta(seconds=1)
    beta,used=fit_coefficients(parts,cutoff,minimum=2);assert beta=={'tap_iron':.5,'tap_time_len':.5};assert len(used)==2
    parts.loc[0,'train_available_max']=cutoff
    with pytest.raises(ContractError):fit_coefficients(parts,cutoff,minimum=2)


def grid():
    reg=load_yaml('configs/optimization_v0_12/experiment.yaml');metrics={}
    for unit,_,h,_,_ in units(reg):
        metrics[unit]=dict(horizon=h,candidates={})
        for c,i,t in [('V1',.2,.2),(IRON,.198,.2),(TIME,.2,.198),(BOTH,.198,.198)]:
            metrics[unit]['candidates'][c]={'overall':dict(loss=(i+t)/2,iron={'wmape':i},time={'wmape':t})}
    return reg,metrics


def test_grid_additivity_full_gates_and_changed_target_protection():
    reg,metrics=grid();summary=aggregate_grid(metrics);assert additive(metrics,summary)['passed']
    g=acceptance(metrics,summary,reg,True,True);assert all(x['passed'] for x in g['candidates'].values());assert g['selected']==BOTH
    assert acceptance(metrics,summary,reg,False,True)['selected'] is None
    for unit,row in metrics.items():
        row['candidates'][IRON]['overall']=dict(loss=.2001,iron={'wmape':.2002},time={'wmape':.2})
        row['candidates'][BOTH]['overall']=dict(loss=.1991,iron={'wmape':.2002},time={'wmape':.198})
    g=acceptance(metrics,aggregate_grid(metrics),reg,True,True)
    assert not g['candidates'][BOTH]['checks']['H1_iron'] and not g['candidates'][BOTH]['passed']


def test_selection_near_tie_target_count_h1_then_lexicographic():
    r={IRON:dict(passed=True,J=.10009,H1_E=.1,changed_targets=['tap_iron']),TIME:dict(passed=True,J=.10008,H1_E=.1,changed_targets=['tap_time_len']),BOTH:dict(passed=True,J=.1,H1_E=.09,changed_targets=['tap_iron','tap_time_len'])}
    assert choose(r)==IRON
    r[TIME]['H1_E']=.099;assert choose(r)==TIME
    r[IRON]['passed']=r[TIME]['passed']=False;assert choose(r)==BOTH
    r[BOTH]['passed']=False;assert choose(r) is None


def test_archive_restores_exact_timezone_instants(tmp_path):
    p=tmp_path/'times.csv';pd.DataFrame(dict(sample_id=['x'],available_at=['2024-06-01 00:00:00+08:00'])).to_csv(p,index=False)
    t=archive(p).available_at
    assert str(t.dtype)=='datetime64[ns, Asia/Shanghai]'
    assert t.iloc[0]==pd.Timestamp('2024-05-31 16:00:00',tz='UTC')


def test_weight_shuffle_exact_for_irregular_real_valued_ages():
    m,c=metadata(tuple(np.linspace(.1,270.3,137)))
    expected=weights(m,c)[0].sort_index()
    got=weights(m.sample(frac=1,random_state=2026),c)[0].sort_index()
    pd.testing.assert_series_equal(expected,got,check_exact=True)
