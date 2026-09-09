"""OPT-22 fixed symmetric ratio inference and V1-relative acceptance."""
import numpy as np
import pandas as pd
from ..exceptions import ContractError
from .structural import lad_coefficient, select_oof

PRED = ['pred_tap_iron', 'pred_tap_time_len']
V2 = 'V2_DUAL_RATIO'
V3 = 'V3_TIME_STRUCTURAL'


def aligned(frame, columns):
    if list(frame) != columns or frame.empty or frame.isna().any().any() or frame.sample_id.duplicated().any():
        raise ContractError('complete unique prediction-only inputs required')
    numeric=frame[columns[1:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric<0).any():
        raise ContractError('finite nonnegative predictions required')
    return frame.set_index('sample_id').sort_index()


def predict_dual(base, v1, inverse, beta):
    if not np.isfinite(beta) or not 0<=beta<=1:raise ContractError('time coefficient must be in [0,1]')
    b=aligned(base,['sample_id',*PRED]);v=aligned(v1,['sample_id',*PRED])
    q=aligned(inverse,['sample_id','pred_inverse_rate'])
    if not b.index.equals(v.index) or not b.index.equals(q.index):raise ContractError('structural prediction IDs differ')
    structural_time=q.pred_inverse_rate.to_numpy()*b.pred_tap_iron.to_numpy()
    t=np.maximum(0.,b.pred_tap_time_len.to_numpy()+beta*(structural_time-b.pred_tap_time_len.to_numpy()))
    if not np.isfinite(t).all():raise ContractError('inverse-rate direction overflow')
    result={}
    for name,iron in ((V2,v.pred_tap_iron),(V3,b.pred_tap_iron)):
        result[name]=pd.DataFrame({'sample_id':b.index.to_numpy(),PRED[0]:iron.to_numpy(),PRED[1]:t})
    return result


def fit_time(oof, cutoff, minimum=100):
    selected=select_oof(oof,cutoff,minimum)
    if 'pred_inverse_rate' not in selected:raise ContractError('missing inverse-rate OOF prediction')
    q=selected.pred_inverse_rate.to_numpy(dtype=float)
    if not np.isfinite(q).all() or (q<0).any():raise ContractError('invalid inverse-rate OOF')
    direction=q*selected.pred_tap_iron.to_numpy()-selected.pred_tap_time_len.to_numpy()
    beta=lad_coefficient(selected.tap_time_len,selected.pred_tap_time_len,direction)
    return beta,selected


def acceptance(metrics,summary,reg,iron_exact):
    policy=reg['acceptance'];reference=summary['V1'];result={}
    for name in (V2,V3):
        s=summary[name]
        d={h:s['horizons'][h]['mean_loss']-reference['horizons'][h]['mean_loss'] for h in ('H1','H2','H3','H4')}
        target={h:{t:s['horizons'][h][t+'_mean_wmape']-reference['horizons'][h][t+'_mean_wmape'] for t in ('iron','time')} for h in d}
        origins={u:v['candidates'][name]['overall']['loss']-v['candidates']['V1']['overall']['loss'] for u,v in metrics.items() if v['horizon']==1}
        if len(origins)!=6:raise ContractError('six H1 origins required')
        dev={u:metrics[u]['candidates'][name]['overall']['loss']-metrics[u]['candidates']['V1']['overall']['loss'] for u in ('DEV_LONG','DEV_SHORT')}
        checks=dict(J_improvement=reference['J']-s['J']>=policy['J_min_improvement'],
            H1_E=d['H1']<=policy['H1_E_max_regression'],
            H1_time=-target['H1']['time']>=policy['H1_time_min_improvement'],
            H1_origins=sum(x<=0 for x in origins.values())>=policy['H1_min_nonworse_origins'],
            recent_H1=sum(origins[f'O2024{m:02d}_H1']<0 for m in policy['recent_H1_months'])>=policy['recent_H1_min_improved'],
            H234_E=max(d[h] for h in ('H2','H3','H4'))<=policy['H234_E_max_regression'],
            H234_time=max(target[h]['time'] for h in ('H2','H3','H4'))<=policy['H234_time_max_regression'],
            iron=iron_exact[name] and (name==V3 or max(target[h]['iron'] for h in d)<=policy['V2_iron_max_horizon_regression_vs_V1']),
            DEV=max(dev.values())<=policy['DEV_max_regression'])
        result[name]=dict(passed=all(checks.values()),checks=checks,J=s['J'],J_delta_vs_V1=s['J']-reference['J'],
            horizon_deltas=d,target_horizon_deltas=target,H1_origin_deltas=origins,DEV_deltas=dev,
            iron_exact_reference='R2' if name==V3 else 'V1',iron_exact=iron_exact[name])
    selected=choose(result,reg['selection']['prefer_V3_when_abs_J_difference_exclusive'])
    return dict(candidates=result,selected=selected,status='PASS' if selected else 'FAIL_CLOSE_RATIO_STRUCTURE_EXTENSION',reference='V1',scope='RETROSPECTIVE_POST_HOLDOUT_CONSUMPTION')


def choose(results,tolerance=.0002):
    passed=[k for k in (V2,V3) if results[k]['passed']]
    if not passed:return None
    if len(passed)==2 and abs(results[V2]['J']-results[V3]['J'])<tolerance:return V3
    return min(passed,key=lambda k:results[k]['J'])
