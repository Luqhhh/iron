from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.component_export import PRED
from bf_tap.optimization.dual_ratio import V2,V3,predict_dual,fit_time,acceptance,choose
from bf_tap.optimization.dual_ratio_common import week_intervals
from bf_tap.optimization.target_ablation import reconstruct


def fixture():
    base=pd.DataFrame([['a',100.,20.],['b',90.,30.]],columns=['sample_id',*PRED])
    v1=pd.DataFrame([['a',150.,17.5],['b',95.,25.]],columns=['sample_id',*PRED])
    q=pd.DataFrame([['a',.1],['b',0.]],columns=['sample_id','pred_inverse_rate'])
    return base,v1,q


def test_exact_iron_and_time_uses_R2_not_corrected_iron_or_reciprocal():
    b,v,q=fixture();p=predict_dual(b,v,q,.5)
    np.testing.assert_array_equal(p[V2][PRED[0]],v[PRED[0]])
    np.testing.assert_array_equal(p[V3][PRED[0]],b[PRED[0]])
    np.testing.assert_array_equal(p[V2][PRED[1]],[15.,15.])
    np.testing.assert_array_equal(p[V2][PRED[1]],p[V3][PRED[1]])
    for c in p:pd.testing.assert_frame_equal(p[c],predict_dual(b.iloc[::-1],v,q.iloc[::-1],.5)[c])
    np.testing.assert_array_equal(predict_dual(b,v,q,0)[V3][PRED],b[PRED])


def test_W0_exact_alignment():
    b,v,q=fixture();w=reconstruct({'R2':b.iloc[::-1],'V1':v})
    np.testing.assert_array_equal(w[PRED[0]],b[PRED[0]])
    np.testing.assert_array_equal(w[PRED[1]],v[PRED[1]])


@pytest.mark.parametrize('kind',['label','duplicate','negative','nan','beta'])
def test_invalid_inverse_inference(kind):
    b,v,q=fixture();beta=.5
    if kind=='label':b['tap_time_len']=99
    if kind=='duplicate':q.loc[1,'sample_id']='a'
    if kind=='negative':q.loc[0,'pred_inverse_rate']=-1
    if kind=='nan':q.loc[0,'pred_inverse_rate']=np.nan
    if kind=='beta':beta=1.1
    with pytest.raises(ContractError):predict_dual(b,v,q,beta)


def test_causal_time_LAD_excludes_unavailable_labels():
    b,v,q=fixture();t=pd.Timestamp('2024-05-01',tz='Asia/Shanghai');cutoff=t+pd.DateOffset(months=1)
    oof=b.merge(q,on='sample_id');oof['pred_rate']=10.;oof['reference_time']=[t,t+pd.Timedelta(days=30)]
    oof['label_available_at']=[t+pd.Timedelta(minutes=20),cutoff+pd.Timedelta(days=1)]
    oof['fold_cutoff']=t;oof['train_reference_max']=t-pd.Timedelta(days=1)
    oof['train_available_max']=t;oof['history_available_max']=t
    oof['tap_iron']=[100.,999.];oof['tap_time_len']=[12.,999.]
    beta,used=fit_time(oof,cutoff,1);assert beta==.8 and used.sample_id.tolist()==['a']
    oof.loc[1,'tap_time_len']=1e9;assert fit_time(oof,cutoff,1)[0]==beta
    oof.loc[0,'train_available_max']=t+pd.Timedelta(seconds=1)
    with pytest.raises(ContractError,match='temporal leakage'):fit_time(oof,cutoff,1)


def gate_fixture():
    values={'R2':(.2001,.2002,.2),'V1':(.2,.2,.2),V2:(.199,.2,.198),V3:(.1991,.2002,.198)}
    summary={c:dict(J=e,horizons={f'H{h}':dict(mean_loss=e,iron_mean_wmape=i,time_mean_wmape=t) for h in range(1,5)}) for c,(e,i,t) in values.items()}
    metrics={f'O2024{m:02d}_H1':dict(horizon=1,candidates={c:{'overall':{'loss':e}} for c,(e,_,_) in values.items()}) for m in range(6,12)}
    for unit in ('DEV_LONG','DEV_SHORT'):metrics[unit]=dict(horizon=None,candidates={c:{'overall':{'loss':e}} for c,(e,_,_) in values.items()})
    return metrics,summary,load_yaml('configs/optimization_v0_9/experiment.yaml')


def test_V1_gates_and_V3_preference_only_among_passes():
    m,s,r=gate_fixture();result=acceptance(m,s,r,{V2:True,V3:True})
    assert result['selected']==V3
    result=acceptance(m,s,r,{V2:True,V3:False});assert result['selected']==V2
    bad=deepcopy(s);bad[V2]['horizons']['H1']['time_mean_wmape']=.1996
    assert not acceptance(m,bad,r,{V2:True,V3:True})['candidates'][V2]['checks']['H1_time']
    bad=deepcopy(s);bad[V2]['horizons']['H4']['mean_loss']=.2006
    assert not acceptance(m,bad,r,{V2:True,V3:True})['candidates'][V2]['checks']['H234_E']
    bad=deepcopy(m)
    for month in (9,10):bad[f'O2024{month:02d}_H1']['candidates'][V2]['overall']['loss']=.2
    assert not acceptance(bad,s,r,{V2:True,V3:True})['candidates'][V2]['checks']['recent_H1']
    rows={V2:dict(passed=True,J=.19),V3:dict(passed=True,J=.19021)}
    assert choose(rows)==V2
    rows[V3]['J']=.19019;assert choose(rows)==V3
    rows[V3]['passed']=False;assert choose(rows)==V2


def test_paired_week_bootstrap_recomputes_ratios_and_targets():
    rows=[]
    for unit,h in [('H1',1),('H2',2),('H3',3),('H4',4),('DEV_LONG',None),('DEV_SHORT',None)]:
        for c,err in [('V1',0.),('R2',.1)]:
            rows.append(dict(unit=unit,horizon=h,candidate=c,sample_id=unit,week='2024-W01',tap_iron=100.,tap_time_len=10.,abs_error_tap_iron=100*err,abs_error_tap_time_len=10*err))
    errors=pd.DataFrame(rows);result=week_intervals(errors,'V1','R2',repetitions=10)
    assert result['valid_repetitions']==10
    for unit in ('H1','H2','H3','H4','J','DEV_LONG','DEV_SHORT'):
        for target in ('E','iron_wmape','time_wmape'):assert result['deltas'][unit][target]['median']==pytest.approx(-.1)
    errors.loc[0,'tap_iron']=101.
    with pytest.raises(ContractError,match='unpaired'):week_intervals(errors,'V1','R2',repetitions=10)
