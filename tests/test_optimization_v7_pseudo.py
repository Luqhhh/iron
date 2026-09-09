import numpy as np
import pandas as pd
import pytest
from bf_tap.exceptions import ContractError
from bf_tap.optimization.pseudo_history import rollout
from bf_tap.optimization.component_export import META, PRED
from bf_tap.optimization.history_stable import stable_features
from bf_tap.features.history import build_history_features

T = pd.Timestamp('2024-06-01', tz='Asia/Shanghai')

class Builder:
    a = {'features': {'history': {'last_k_mean': [3, 10]}}}
    def X(self, samples, entry, history):
        f,_ = build_history_features(samples, history, fit_cutoff=pd.Timestamp(entry['cutoff']))
        return stable_features(f,samples,history,pd.Timestamp(entry['cutoff']),'R2')

class Model:
    def predict_raw(self,X):
        # A visible completion produces a detectable feedback change.
        count=X['history__all__tap_iron__last30_count'].to_numpy()
        return pd.DataFrame({PRED[0]:100+count,PRED[1]:np.full(len(X),10.)},index=X.index)

def fixture():
    base=pd.DataFrame([dict(sample_id='base',tap_no='base',spout_no='1',
        reference_time=T-pd.Timedelta(hours=2),tap_end_time=T-pd.Timedelta(hours=1),
        available_at=T-pd.Timedelta(hours=1),tap_iron=50.,tap_time_len=20.)])
    samples=pd.DataFrame({'sample_id':['a','b','c','d'],'tap_no':['a','b','c','d'],
        'spout_no':['1']*4,'reference_time':[T,T,T+pd.Timedelta(minutes=9),T+pd.Timedelta(minutes=10)]})
    entries={r:dict(cutoff=str(T),bundle_sha256=r,history_snapshot_sha256='history') for r in ('OR','HR')}
    return samples,entries,{r:(Model(),base.copy()) for r in entries},Builder()

def test_completion_boundary_timestamp_snapshot_and_reversal():
    s,e,l,b=fixture();original=l['OR'][1].copy(deep=True)
    u0,u1,audit,pseudo=rollout(s,e,l,b)
    assert [r['pseudo_history_rows_visible'] for r in audit]==[0,0,0,2]
    assert [r['pseudo_history_depth'] for r in audit]==[0,0,0,1]
    assert u1.set_index('sample_id').loc['d',PRED[0]]==103.
    assert np.array_equal(u0.iloc[:2][PRED],u1.iloc[:2][PRED])
    assert pseudo.iloc[0].available_at==T+pd.Timedelta(minutes=10)
    assert audit[0]['pseudo_history_sha256']==audit[1]['pseudo_history_sha256']
    reverse=rollout(s.iloc[::-1],e,l,b)
    pd.testing.assert_frame_equal(u1,reverse[1]);assert audit==reverse[2]
    pd.testing.assert_frame_equal(original,l['OR'][1])
    assert set(pseudo.sample_id)==set(s.sample_id)
    assert pseudo.tap_iron.tolist()==u1[PRED[0]].tolist()

@pytest.mark.parametrize('column',['tap_iron','tap_time_len','label_available_at'])
def test_labels_rejected(column):
    s,e,l,b=fixture();s[column]=999
    with pytest.raises(ContractError,match='metadata only'):rollout(s,e,l,b)

def test_future_base_rejected():
    s,e,l,b=fixture();l['OR'][1]['available_at']=T+pd.Timedelta(minutes=1)
    with pytest.raises(ContractError,match='base boundary'):rollout(s,e,l,b)

def test_duplicate_ids_rejected():
    s,e,l,b=fixture();s.loc[1,'sample_id']='a'
    with pytest.raises(ContractError,match='duplicate'):rollout(s,e,l,b)

def test_zero_duration_still_cannot_affect_same_timestamp():
    s,e,l,b=fixture()
    class Zero(Model):
        def predict_raw(self,X):
            p=super().predict_raw(X);p[PRED[1]]=0.;return p
    l={r:(Zero(),h) for r,(_,h) in l.items()}
    _,_,audit,_=rollout(s,e,l,b)
    assert [r['pseudo_history_rows_visible'] for r in audit]==[0,0,2,3]

def test_later_metadata_does_not_change_prefix():
    s,e,l,b=fixture();full=rollout(s,e,l,b)
    short=rollout(s.iloc[:3],e,l,b)
    pd.testing.assert_frame_equal(full[1].iloc[:3],short[1])
    assert full[2][:3]==short[2]

def test_all_strict_gates_are_required():
    from copy import deepcopy
    from bf_tap.config import load_yaml
    from bf_tap.optimization.pseudo_evaluation import gate
    reg=load_yaml('configs/optimization_v0_7/experiment.yaml')
    def metric(loss):return {'candidates':{'U0':{'overall':{'loss':.2}},'U1':{'overall':{'loss':loss}}}}
    metrics={f'O2024{m:02d}_H1':dict(horizon=1,**metric(.198)) for m in range(6,12)}
    metrics.update({u:dict(horizon=None,**metric(.2)) for u in ('DEV_LONG','DEV_SHORT')})
    summary={c:{'J':v,'horizons':{f'H{h}':{'mean_loss':v,'iron_mean_wmape':v,'time_mean_wmape':v} for h in range(1,5)}} for c,v in [('U0',.2),('U1',.198)]}
    assert gate(metrics,summary,reg)['passed']
    bad=deepcopy(metrics);bad['O202410_H1']['candidates']['U1']['overall']['loss']=.2
    assert not gate(bad,summary,reg)['checks']['recent_H1']
    bad=deepcopy(summary);bad['U1']['horizons']['H1']['mean_loss']=.1991
    assert not gate(metrics,bad,reg)['checks']['H1_mean']
    bad=deepcopy(summary);bad['U1']['horizons']['H3']['mean_loss']=.2016
    assert not gate(metrics,bad,reg)['checks']['H234']
    bad=deepcopy(summary);bad['U1']['J']=.200001
    assert not gate(metrics,bad,reg)['checks']['extended_J']
    bad=deepcopy(summary);bad['U1']['horizons']['H1']['time_mean_wmape']=.2011
    assert not gate(metrics,bad,reg)['checks']['H1_targets']
    bad=deepcopy(metrics);bad['DEV_LONG']['candidates']['U1']['overall']['loss']=.2016
    assert not gate(bad,summary,reg)['checks']['DEV']
