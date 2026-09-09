import pandas as pd
import numpy as np
import pytest
from bf_tap.exceptions import ContractError
from bf_tap.optimization.refresh_factorial import score_factorial, decomposition, Context
from bf_tap.optimization.snapshot_ensemble import blend
from bf_tap.optimization.history_stable import stable_features


def fixture():
    return pd.DataFrame([dict(eval_month='2024-07',sample_id=s,model_state=m,history_state=h,
        tap_iron=10.,tap_time_len=20.,pred_tap_iron=p,pred_tap_time_len=p*2)
        for m,h,p in [('old','old',8),('old','new',9),('new','old',11),('new','new',12)] for s in ['a','b']])


def test_score_order_and_identity():
    x=fixture()
    assert score_factorial(x)==score_factorial(x.sample(frac=1,random_state=3))
    d=score_factorial(x)['2024-07']['decomposition']['E']
    assert d['model']+d['history']==pytest.approx(d['joint'])


@pytest.mark.parametrize('bad',['missing','duplicate','label','nan','negative','state'])
def test_reject(bad):
    x=fixture()
    if bad=='missing': x=x.iloc[:-1]
    if bad=='duplicate': x=pd.concat([x,x.iloc[:1]])
    if bad=='label': x.loc[0,'tap_iron']=11.
    if bad=='nan': x.loc[0,'pred_tap_iron']=np.nan
    if bad=='negative': x.loc[0,'pred_tap_iron']=-1.
    if bad=='state': x.loc[0,'model_state']='future'
    with pytest.raises(ContractError): score_factorial(x)


def test_predict_never_fits():
    ctx=Context.__new__(Context)
    ctx.models={}
    with pytest.raises(ContractError,match='cannot fit'): ctx.predict(None,'missing','missing')


def test_blend_identity():
    a=pd.DataFrame({'sample_id':['b','a'],'pred_tap_iron':[2.,4.],'pred_tap_time_len':[6.,8.]})
    assert blend(a,a.iloc[::-1]).equals(a.sort_values('sample_id').reset_index(drop=True))
    with pytest.raises(ContractError): blend(a,a.iloc[:1])


def test_r1_counts():
    x=pd.DataFrame({'history__all__tap_iron__latest':[1.], 'history__all__tap_iron__last3_mean':[2.],
                    'history__all__tap_iron__last3_count':[3.], 'history__all__tap_iron__last10_mean':[4.]})
    out=stable_features(x,None,None,None,'R1')
    assert list(out)==['history__all__tap_iron__last3_count','history__all__tap_iron__last10_mean']


def test_stable_centers_asof_and_current_exclusion():
    t=pd.Timestamp('2024-07-01',tz='Asia/Shanghai')
    samples=pd.DataFrame({'sample_id':['self','early'],'spout_no':['1','2'],
                          'reference_time':[t,t-pd.Timedelta(days=10)]})
    history=pd.DataFrame({'sample_id':['a','self','future'],'spout_no':['1']*3,
        'reference_time':[t-pd.Timedelta(days=2),t-pd.Timedelta(days=1),t],
        'available_at':[t-pd.Timedelta(days=1),t,t+pd.Timedelta(days=1)],
        'tap_iron':[10.,1000.,9999.],'tap_time_len':[20.,2000.,9999.]})
    out=stable_features(pd.DataFrame({'spout_no':['1','2']}),samples,history,t,'R2')
    assert out.loc[0,'history__all__tap_iron__last30_median']==10.
    assert out.loc[0,'history__spout__tap_iron__last100_count']==1.
    assert np.isnan(out.loc[1,'history__all__tap_iron__last30_median'])
    assert out.loc[1,'history__all__tap_iron__last30_count']==0.


@pytest.mark.parametrize('case',['future','current','forged','valid'])
def test_snapshot_boundaries(case):
    from bf_tap.optimization.refresh_factorial import validate_snapshot
    t=pd.Timestamp('2024-07-01',tz='Asia/Shanghai')
    official=pd.DataFrame({'sample_id':['a'],'tap_no':[1],'spout_no':[1],
        'reference_time':[t-pd.Timedelta(days=2)],'tap_end_time':[t-pd.Timedelta(days=1)],
        'available_at':[t-pd.Timedelta(days=1)],'tap_iron':[10.],'tap_time_len':[20.]})
    samples=pd.DataFrame({'sample_id':['b'],'reference_time':[t]})
    snapshot=official.copy()
    if case=='future': snapshot.loc[0,'available_at']=t+pd.Timedelta(days=1)
    if case=='current': samples.loc[0,'sample_id']='a'
    if case=='forged': snapshot.loc[0,'tap_iron']=99.
    if case=='valid': validate_snapshot(snapshot,official,t,samples)
    else:
        with pytest.raises(ContractError): validate_snapshot(snapshot,official,t,samples)


def test_public_cache_recomputes_all_history_and_preserves_order(monkeypatch):
    import bf_tap.optimization.refresh_factorial as module
    from bf_tap.features.history import build_history_features
    t=pd.Timestamp('2024-07-01',tz='Asia/Shanghai')
    h=pd.DataFrame({'sample_id':['a','b'],'tap_no':[1,2],'spout_no':[1,1],
        'reference_time':[t-pd.Timedelta(days=4),t-pd.Timedelta(days=2)],
        'tap_end_time':[t-pd.Timedelta(days=3),t-pd.Timedelta(days=1)],
        'available_at':[t-pd.Timedelta(days=3),t-pd.Timedelta(days=1)],
        'tap_iron':[10.,30.],'tap_time_len':[20.,60.]})
    samples=pd.DataFrame({'sample_id':['x','y'],'spout_no':[1,2],'reference_time':[t,t]},index=[8,4])
    calls=[]
    def build(s,history,op,burden,cutoff,a):
        calls.append(len(s))
        hx,_=build_history_features(s,history,fit_cutoff=cutoff)
        return pd.concat([s[['spout_no']],hx],axis=1)
    monkeypatch.setattr(module,'feature_frame',build)
    monkeypatch.setattr(module,'component_features',lambda f,c,a:f)
    ctx=Context.__new__(Context)
    ctx.history=h;ctx.op=None;ctx.burden=None;ctx.cache={};ctx.public_features=None;ctx.public_metadata={}
    ctx.a={'features':{'history':{'last_k_mean':[3,10]}}}
    old=ctx.X(samples,t-pd.Timedelta(days=2),'E04')
    new=ctx.X(samples.iloc[::-1],t,'E04')
    assert calls==[2]
    assert old.loc[8,'history__all__tap_iron__last3_count']==1.
    assert new.loc[8,'history__all__tap_iron__last3_count']==2.
    assert new.loc[8,'history__spout__tap_iron__last3_mean']==20.
    expected=build(samples.iloc[::-1],h,None,None,t,None)
    pd.testing.assert_frame_equal(new,expected)
