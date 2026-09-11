"""Three registered target ablations with isolated directions and fixed selection."""
import numpy as np
import pandas as pd
from ..exceptions import ContractError
from .component_export import PRED
from .structural import select_oof,lad_coefficient

IRON='V6I_RECENCY60_IRON';TIME='V6T_RECENCY60_TIME';BOTH='V6B_RECENCY60_BOTH'
CANDIDATES={IRON:('tap_iron',),TIME:('tap_time_len',),BOTH:('tap_iron','tap_time_len')}
PARTS=['sample_id',*PRED,'pred_rate','direct_iron','direct_time','base_iron','base_time','direction_iron','direction_time']


def isolated_directions(old,base,rate_floor=1e-6):
    if list(old)!=['sample_id',*PRED,'pred_rate'] or list(base)!=['sample_id','base_iron','base_time']:
        raise ContractError('prediction-only old endpoint/base inputs required')
    for x in (old,base):
        if x.empty or x.sample_id.isna().any() or x.sample_id.duplicated().any():raise ContractError('unique prediction IDs required')
        if not np.isfinite(x.iloc[:,1:].to_numpy(dtype=float)).all() or (x.iloc[:,1:].to_numpy(dtype=float)<0).any():raise ContractError('finite nonnegative predictions required')
    o=old.set_index('sample_id');b=base.set_index('sample_id')
    if set(o.index)!=set(b.index):raise ContractError('base IDs differ')
    b=b.loc[o.index];usable=o.pred_rate.to_numpy()>rate_floor
    di=np.zeros(len(o));dt=np.zeros(len(o))
    di[usable]=(o.pred_rate*o[PRED[1]]-b.base_iron).to_numpy()[usable]
    dt[usable]=o[PRED[0]].to_numpy()[usable]/o.pred_rate.to_numpy()[usable]-b.base_time.to_numpy()[usable]
    if not np.isfinite([di,dt]).all():raise ContractError('nonfinite isolated direction')
    return pd.DataFrame(dict(sample_id=o.index,direction_iron=di,direction_time=dt)),int((~usable).sum())


def fit_coefficients(oof,cutoff,minimum=100):
    selected=select_oof(oof,cutoff,minimum)
    if set(PARTS)-set(selected):raise ContractError('incomplete recency OOF provenance')
    beta={target:lad_coefficient(selected[target],selected['base_'+short],selected['direction_'+short])
          for target,short in (('tap_iron','iron'),('tap_time_len','time'))}
    return beta,selected


def outputs(parts,original_v1,beta):
    if list(parts)!=PARTS or set(beta)!=set(('tap_iron','tap_time_len')) or not all(np.isfinite(v) and 0<=v<=1 for v in beta.values()):
        raise ContractError('registered parts and two fixed LAD coefficients required')
    if list(original_v1)!=['sample_id',*PRED] or original_v1.sample_id.duplicated().any() or set(parts.sample_id)!=set(original_v1.sample_id):raise ContractError('V1 IDs differ')
    if parts.sample_id.duplicated().any() or parts.sample_id.isna().any() or not np.isfinite(parts.iloc[:,1:].to_numpy()).all():raise ContractError('invalid candidate inputs')
    expected,_=isolated_directions(parts[['sample_id',*PRED,'pred_rate']],parts[['sample_id','base_iron','base_time']])
    if not np.array_equal(expected[['direction_iron','direction_time']],parts[['direction_iron','direction_time']]):raise ContractError('direction cross-input or order violation')
    v1=original_v1.set_index('sample_id').loc[parts.sample_id].reset_index()
    corrected={target:np.maximum(0.,parts['base_'+short].to_numpy()+beta[target]*parts['direction_'+short].to_numpy())
               for target,short in (('tap_iron','iron'),('tap_time_len','time'))}
    result={}
    for candidate,changed in CANDIDATES.items():
        p=v1.copy()
        for t in changed:p['pred_'+t]=corrected[t]
        if not np.isfinite(p[PRED].to_numpy()).all():raise ContractError('nonfinite corrected prediction')
        result[candidate]=p
    validate_outputs(result,original_v1)
    return result


def validate_outputs(result,v1):
    if set(result)!=set(CANDIDATES):raise ContractError('exactly three registered outputs required')
    a={c:p.set_index('sample_id').sort_index() for c,p in result.items()};v=v1.set_index('sample_id').sort_index()
    if any(not p.index.equals(v.index) for p in a.values()):raise ContractError('candidate identity mismatch')
    for left,right,target in ((a[IRON],v,PRED[1]),(a[TIME],v,PRED[0]),(a[BOTH],a[IRON],PRED[0]),(a[BOTH],a[TIME],PRED[1])):
        if not np.array_equal(left[target],right[target]):raise ContractError('unchanged target/composition equality failed')
    return True


def additive(metrics,summary,tolerance=1e-12):
    errors={}
    for unit,v in metrics.items():
        loss={c:x['overall']['loss'] for c,x in v['candidates'].items()}
        errors[unit]=abs((loss[BOTH]-loss['V1'])-(loss[IRON]-loss['V1'])-(loss[TIME]-loss['V1']))
    errors['J']=abs((summary[BOTH]['J']-summary['V1']['J'])-(summary[IRON]['J']-summary['V1']['J'])-(summary[TIME]['J']-summary['V1']['J']))
    if max(errors.values())>tolerance:raise ContractError('E/J target additivity failed')
    return dict(passed=True,tolerance=tolerance,max_error=max(errors.values()),errors=errors)


def acceptance(metrics,summary,reg,engineering,identity):
    policy=reg['acceptance'];checks={};results={}
    if len([u for u in metrics.values() if u['horizon']==1])!=6 or len([u for u in metrics.values() if u['horizon'] is not None])!=18:raise ContractError('six-origin 18-cell grid required')
    algebra=additive(metrics,summary,policy['algebra_tolerance'])
    for candidate,changed in CANDIDATES.items():
        current,ref=summary[candidate],summary['V1']
        delta={h:{k:current['horizons'][h][k]-ref['horizons'][h][k] for k in ('mean_loss','iron_mean_wmape','time_mean_wmape')} for h in ('H1','H2','H3','H4')}
        origins={u:v['candidates'][candidate]['overall']['loss']-v['candidates']['V1']['overall']['loss'] for u,v in metrics.items() if v['horizon']==1}
        dev={u:metrics[u]['candidates'][candidate]['overall']['loss']-metrics[u]['candidates']['V1']['overall']['loss'] for u in ('DEV_LONG','DEV_SHORT')}
        c=dict(engineering_and_causal=bool(engineering),identity_and_composition=bool(identity),additive=algebra['passed'],
            J=current['J']-ref['J']<=policy['J_delta_max'],H1_E=delta['H1']['mean_loss']<=policy['H1_E_delta_max'],
            H1_origins=sum(x<0 for x in origins.values())>=policy['H1_min_strict_improved'],
            recent_H1=sum(origins[f'O2024{m:02d}_H1']<0 for m in policy['recent_months'])>=policy['recent_min_strict_improved'])
        for target in changed:
            short='iron' if target=='tap_iron' else 'time'
            c['H1_'+short]=delta['H1'][short+'_mean_wmape']<=policy['H1_changed_target_delta_max']
            for h in ('H2','H3','H4'):c[h+'_'+short]=delta[h][short+'_mean_wmape']<=policy['H234_changed_target_delta_max']
        for h in ('H2','H3','H4'):c[h+'_E']=delta[h]['mean_loss']<=policy['H234_E_delta_max']
        for u,d in dev.items():c[u]=d<=policy['DEV_E_delta_max']
        results[candidate]=dict(passed=all(c.values()),checks=c,changed_targets=list(changed),J=current['J'],
            H1_E=current['horizons']['H1']['mean_loss'],J_delta=current['J']-ref['J'],horizon_deltas=delta,H1_origin_deltas=origins,DEV_deltas=dev)
    selected=choose(results,reg['selection']['J_near_tie'])
    return dict(status='PASS_ONE_SELECTED' if selected else 'FAIL_NO_RELEASE',selected=selected,candidates=results,
                algebra=algebra,scope='CONSUMED_RETROSPECTIVE',selection_rule=reg['selection'])


def choose(results,near_tie=.0001):
    eligible=[(c,r) for c,r in results.items() if r['passed']]
    if not eligible:return None
    best=min(r['J'] for _,r in eligible)
    near=[(c,r) for c,r in eligible if r['J']<=best+near_tie]
    return min(near,key=lambda item:(len(item[1]['changed_targets']),item[1]['H1_E'],item[0]))[0]
