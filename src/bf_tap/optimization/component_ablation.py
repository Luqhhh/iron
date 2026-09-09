"""OPT-14 fixed component combinations, complementary errors and two distinct gates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..artifacts import atomic_write_json, file_sha256, stable_digest, verify_file_identities
from ..exceptions import ContractError
from ..io import parse_local_time
from ..metrics import score_predictions
from .component_export import META, PRED, ROLES, read_json, units
from .snapshot_ensemble import blend
from .validation import aggregate_grid, error_contributions
from .v3_evidence import paired_week_intervals

TARGETS = ['tap_iron','tap_time_len']
CANDIDATES = ('P1','P2','P3')


def validate_frame(frame, *, labeled=False):
    frame=frame.copy()
    cols=['sample_id',*PRED] + (META[1:]+TARGETS if labeled else [])
    if set(cols)-set(frame) or frame.empty or frame[cols].isna().any().any():
        raise ContractError('incomplete aligned scoring frame')
    frame['sample_id']=frame.sample_id.astype(str)
    if frame.sample_id.duplicated().any():
        raise ContractError('duplicate sample ID')
    numeric=frame[PRED+(TARGETS if labeled else [])].to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric<0).any():
        raise ContractError('nonfinite or negative scoring inputs')
    if labeled:
        frame['reference_time']=parse_local_time(frame.reference_time,'reference_time')
        if (frame[TARGETS].sum()<=0).any():
            raise ContractError('zero target denominator')
    return frame


def fixed_combinations(parts):
    if set(parts)!=set(ROLES):
        raise ContractError('missing or unexpected component role')
    clean={k:validate_frame(v)[['sample_id',*PRED]] for k,v in parts.items()}
    reference=set(clean['O0'].sample_id)
    if any(set(p.sample_id)!=reference for p in clean.values()):
        raise ContractError('component sample sets differ')
    return {**clean,'E12-raw':blend(clean['O0'],clean['H0'],.8),
        'R2':blend(clean['OR'],clean['HR'],.8),'P1':blend(clean['OR'],clean['H0'],.8),
        'P2':blend(clean['O0'],clean['HR'],.8),'P3':clean['OR'].copy()}


def representation_effect(loss):
    a,b,c,d=[loss[k] for k in ('E12-raw','P1','P2','R2')]
    result=dict(D_O=((b-a)+(d-c))/2,D_H=((c-a)+(d-b))/2,joint=d-a,interaction=d-b-c+a)
    if not np.isclose(result['D_O']+result['D_H'],result['joint'],atol=1e-14):
        raise ContractError('representation decomposition identity failed')
    return result


def complement(actual,predictions):
    y=actual.set_index('sample_id').sort_index()
    p={k:v.set_index('sample_id').sort_index() for k,v in predictions.items()}
    metrics={k:score_predictions(actual,v) for k,v in predictions.items()}
    result={}
    for target,key in zip(TARGETS,('iron','time')):
        denominator=float(y[target].sum())
        eo=p['OR']['pred_'+target]-y[target]
        eh=p['HR']['pred_'+target]-y[target]
        eb=p['R2']['pred_'+target]-y[target]
        result[target]=dict(D_help=metrics['R2'][key]['wmape']-metrics['OR'][key]['wmape'],
            opposite_error_fraction=float((eo*eh<0).mean()),
            cancellation_abs_sum=float((.8*eo.abs()+.2*eh.abs()-eb.abs()).sum()),
            denominator=denominator,
            representation=representation_effect({k:metrics[k][key]['wmape'] for k in ('E12-raw','P1','P2','R2')}))
    result['E']=representation_effect({k:metrics[k]['loss'] for k in ('E12-raw','P1','P2','R2')})
    return result


def gates(metrics,summary,registration):
    reference=summary['R2'];old=summary['E12-raw']
    h1=[k for k,v in metrics.items() if v['horizon']==1]
    if len(h1)!=6:
        raise ContractError('STAGE_A requires six H1 origins')
    results={}
    for candidate in CANDIDATES:
        s=summary[candidate]
        delta={h:s['horizons'][h]['mean_loss']-reference['horizons'][h]['mean_loss'] for h in ('H1','H2','H3','H4')}
        origin_delta={k:metrics[k]['candidates'][candidate]['overall']['loss']-metrics[k]['candidates']['R2']['overall']['loss'] for k in h1}
        dev={k:metrics[k]['candidates'][candidate]['overall']['loss']-metrics[k]['candidates']['R2']['overall']['loss'] for k in ('DEV_LONG','DEV_SHORT')}
        better=min(metrics['DEV_LONG']['candidates'][c]['overall']['loss'] for c in ('B0','B1'))
        long_beats=metrics['DEV_LONG']['candidates'][candidate]['overall']['loss']<better
        target_h1={t:s['horizons']['H1'][t+'_mean_wmape']-reference['horizons']['H1'][t+'_mean_wmape'] for t in ('iron','time')}
        target_general={t:float(np.mean([s['horizons'][h][t+'_mean_wmape']-reference['horizons'][h][t+'_mean_wmape'] for h in delta])) for t in ('iron','time')}
        p=registration['stage_a'];g=registration['general']
        stage=dict(H1_improvement=-delta['H1']>=p['min_H1_improvement'],
            improved_H1_origins=sum(x<0 for x in origin_delta.values())>=p['min_improved_H1_origins'],
            J_regression=s['J']-reference['J']<=p['max_J_regression'],
            far_horizon_regression=max(delta[h] for h in ('H2','H3','H4'))<=p['max_far_horizon_regression'],
            H1_target_regression=max(target_h1.values())<=p['max_H1_target_regression'],
            DEV_regression=max(dev.values())<=p['max_DEV_regression'],DEV_LONG_beats_control=long_beats)
        general=dict(J_improvement=reference['J']-s['J']>=g['min_J_improvement'],
            improved_horizons=sum(x<0 for x in delta.values())>=g['min_improved_horizons'],
            horizon_regression=max(delta.values())<=g['max_any_horizon_regression'],
            target_regression=max(target_general.values())<=g['per_target_mean_wmape_max_regression'],
            DEV_LONG_beats_control=long_beats,DEV_SHORT_regression=dev['DEV_SHORT']<=g['dev_short_max_regression'])
        results[candidate]=dict(STAGE_A=dict(passed=all(stage.values()),checks=stage),
            GENERAL=dict(passed=all(general.values()),checks=general),J=s['J'],H1=s['horizons']['H1']['mean_loss'],
            H1_improvement_vs_R2=-delta['H1'],J_delta_vs_R2=s['J']-reference['J'],
            H1_delta_vs_E12_raw=s['horizons']['H1']['mean_loss']-old['horizons']['H1']['mean_loss'],
            J_delta_vs_E12_raw=s['J']-old['J'],horizon_deltas=delta,H1_origin_deltas=origin_delta,
            H1_target_deltas=target_h1,target_general_deltas=target_general,DEV_deltas=dev)
    accepted=[k for k,v in results.items() if v['STAGE_A']['passed']]
    selected=min(accepted,key=lambda c:(results[c]['H1'],results[c]['J'],1 if c=='P3' else 2,c)) if accepted else None
    return dict(reference='R2',candidates=results,selected_STAGE_A=selected,
                next_step='OPT16_ELIGIBLE_NO_AUTOMATIC_RELEASE' if selected else 'OPT15_ELIGIBLE_NOT_EXECUTED',
                scope='RETROSPECTIVE_POST_HOLDOUT_CONSUMPTION_NOT_INDEPENDENT_CONFIRMATION')


def run(export,output):
    output.mkdir(parents=True,exist_ok=False)
    try:
        status=read_json(export/'final_status.json');manifest=read_json(export/'manifest.json')
        if status.get('G0')!='PASS_COMPONENT_EXPORT' or status.get('attempted_target_fits')!=0:
            raise ContractError('successful zero-fit export required')
        if file_sha256(export/'component_predictions.csv')!=status['predictions_sha256']:
            raise ContractError('export prediction digest mismatch')
        verify_file_identities(manifest['evidence'])
        registration=dict(manifest['registration'])
        registration['origins']={int(m):n for m,n in registration['origins'].items()}
        source=Path(registration['source_run'])
        atomic_write_json(output/'scoring_manifest.json',dict(export_manifest_sha256=file_sha256(export/'manifest.json'),
            scorer_sha256=file_sha256(__file__),registration=registration,holdout_consumed=True))
        # Explicit scoring-stage access entry; source targets remain outside feature construction.
        ledger=Path(manifest['access_scope']['new_ledger'])
        import fcntl
        with ledger.open('a') as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX)
            f.write(json.dumps(dict(purpose='OPT14_existing_error_targets_scoring',
                export_manifest_sha256=file_sha256(export/'manifest.json'),output=str(output),holdout_consumed=True))+'\n')
        exported=pd.read_csv(export/'component_predictions.csv',dtype={'sample_id':str})
        if set(exported.unit)!={u[0] for u in units(registration)}:
            raise ContractError('missing or extra evaluation units')
        metrics,diagnostics,alignment,all_errors,actuals={},{},[],[],[]
        for unit,cutoff,horizon,start,end in units(registration):
            actual=validate_frame(pd.read_csv(source/'units'/unit/'R2'/'errors.csv',dtype={'sample_id':str}),labeled=True)
            if not ((actual.reference_time>=start)&(actual.reference_time<end)).all():
                raise ContractError('scoring unit time boundary mismatch')
            actuals.append(actual[META+TARGETS])
            group=exported.loc[exported.unit==unit]
            if set(group.role)!=set(ROLES):
                raise ContractError('missing role')
            parts={}
            for role,part in group.groupby('role'):
                part=part.copy();part['reference_time']=parse_local_time(part.reference_time,'reference_time')
                if not (pd.to_datetime(part.fit_cutoff,utc=True)==cutoff).all():
                    raise ContractError('export cutoff mismatch')
                expected=actual.set_index('sample_id').sort_index()
                received=part.set_index('sample_id').sort_index()
                if not expected.index.equals(received.index) or not expected[META[1:]].equals(received[META[1:]]):
                    raise ContractError('export sample metadata mismatch')
                raw=part[['raw_'+c for c in PRED]].to_numpy(float)
                if not np.isfinite(raw).all() or not np.array_equal(np.maximum(raw,0),part[PRED].to_numpy(float)):
                    raise ContractError('component clipping order mismatch')
                parts[role]=part[['sample_id',*PRED]]
            predictions=fixed_combinations(parts)
            for endpoint in ('E12-raw','R2','B0','B1'):
                old=validate_frame(pd.read_csv(source/'units'/unit/endpoint/'errors.csv',dtype={'sample_id':str}),labeled=True)
                x=actual.set_index('sample_id').sort_index();y=old.set_index('sample_id').sort_index()
                if not x.index.equals(y.index) or not x[META[1:]+TARGETS].equals(y[META[1:]+TARGETS]):
                    raise ContractError('reference IDs, metadata or labels disagree')
                if endpoint in ('B0','B1'):
                    predictions[endpoint]=old[['sample_id',*PRED]]
                else:
                    new=predictions[endpoint].set_index('sample_id').sort_index()
                    diffs=(new[PRED]-y[PRED]).abs()
                    maximum=float(diffs.to_numpy().max())
                    if maximum>1e-8:
                        raise ContractError('A00/A11 reconstruction differs from existing endpoint')
                    for sid,row in diffs.iterrows():
                        alignment.append(dict(unit=unit,endpoint=endpoint,sample_id=sid,**row.to_dict()))
            metrics[unit]=dict(horizon=horizon,origin_id=f'O2024{cutoff.month:02d}',candidates={})
            destination=output/'units'/unit;destination.mkdir(parents=True)
            for candidate,pred in predictions.items():
                metrics[unit]['candidates'][candidate]={'overall':score_predictions(actual,pred)}
                errors=error_contributions(actual,pred)
                errors['candidate'],errors['origin'],errors['horizon'],errors['unit']=candidate,f'O2024{cutoff.month:02d}',horizon,unit
                errors.to_csv(destination/f'{candidate}_errors.csv',index=False)
                all_errors.append(errors)
            diagnostics[unit]={'overall':complement(actual,predictions),'by_month':{},'by_spout':{}}
            for kind,groups in [('by_month',actual.reference_time.dt.strftime('%Y-%m')),('by_spout',actual.spout_no.astype(str))]:
                for value in sorted(groups.unique()):
                    a=actual.loc[groups==value];subset={c:p.loc[p.sample_id.isin(a.sample_id)] for c,p in predictions.items()}
                    diagnostics[unit][kind][str(value)]=dict(complement=complement(a,subset),
                        metrics={c:score_predictions(a,p) for c,p in subset.items()})
        combined_actual=pd.concat(actuals)
        if (combined_actual.groupby('sample_id')[META[1:]+TARGETS].nunique()>1).any().any():
            raise ContractError('cross-origin label or metadata inconsistency')
        summary=aggregate_grid(metrics)
        acceptance=gates(metrics,summary,registration)
        errors=pd.concat(all_errors,ignore_index=True)
        grid=errors.loc[errors.horizon.notna()].copy();grid['horizon']=grid.horizon.astype(int)
        table=[]
        for unit,v in metrics.items():
            for candidate,c in v['candidates'].items():
                s=c['overall'];table.append(dict(unit=unit,horizon=v['horizon'],candidate=candidate,
                    E=s['loss'],**{f'{t}_{k}':x for t in ('iron','time') for k,x in s[t].items()}))
        pd.DataFrame(table).to_csv(output/'all_units.csv',index=False)
        pd.DataFrame(table).loc[lambda f:f.horizon==1].to_csv(output/'h1_six_origins.csv',index=False)
        errors.loc[pd.to_datetime(errors.reference_time).dt.month.isin([9,10,11])].to_csv(output/'recent_three_months.csv',index=False)
        pd.DataFrame(alignment).to_csv(output/'endpoint_alignment_by_sample.csv',index=False)
        atomic_write_json(output/'endpoint_alignment.json',dict(units=20,endpoint_checks=40,max_abs_difference=max(max(r[c] for c in PRED) for r in alignment)))
        atomic_write_json(output/'metrics.json',metrics);atomic_write_json(output/'component_complementarity.json',diagnostics)
        atomic_write_json(output/'extended_18_summary.json',summary)
        legacy={k:v for k,v in metrics.items() if k.startswith('O') and int(k[5:7])+v['horizon']<=11}
        atomic_write_json(output/'legacy_14_summary.json',aggregate_grid(legacy))
        atomic_write_json(output/'acceptance.json',acceptance)
        intervals={c:paired_week_intervals(grid.loc[grid.candidate.isin([c,'R2'])],c,'R2',**registration['bootstrap']) for c in CANDIDATES}
        atomic_write_json(output/'paired_week_intervals.json',intervals)
        verify_file_identities(manifest['evidence'])
        atomic_write_json(output/'final_status.json',dict(G0='PASS_OPT14',selected_STAGE_A=acceptance['selected_STAGE_A'],
            G1_STAGE_A='PASS' if acceptance['selected_STAGE_A'] else 'FAIL',
            target_fits=0,holdout_consumed=True,active_release_unchanged=True,
            test_a_used_for_selection=False,export_manifest_sha256=file_sha256(export/'manifest.json')))
    except Exception as exc:
        atomic_write_json(output/'final_status.json',dict(status='FAILED',error=str(exc)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.export,args.output)

if __name__=='__main__': main()
