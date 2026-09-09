"""OPT-17 zero-fit cold replay, sample-wise calendar routing and frozen gates."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..artifacts import atomic_write_json, file_identities, file_sha256, verify_file_identities, build_inference_source_contract
from ..config import load_yaml
from ..exceptions import ContractError
from ..io import read_csv
from ..offline import _load_process_sources
from ..metrics import score_predictions
from .component_export import META,PRED,read_json,units,forbid_fit,ComponentFeatures
from .component_ablation import validate_frame
from .final_lifecycle import algorithm
from .validation import aggregate_grid,error_contributions
from .horizon_router import TARGETS, horizons, route, load_component, predict_parts


def append_access(output,purpose):
    import fcntl
    p=Path(load_yaml('configs/optimization_v0_6/access_scope.yaml')['new_ledger']);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        f.write(json.dumps(dict(purpose=purpose,output=str(output),manifest_sha256=file_sha256(output/'manifest.json'),holdout_consumed=True))+'\n')


def deployment_gate(metrics,summary,reg,h4_exact):
    s,r=summary['S1'],summary['R2'];p=reg['deployment_gate']
    delta={h:s['horizons'][h]['mean_loss']-r['horizons'][h]['mean_loss'] for h in ('H1','H2','H3','H4')}
    targets={h:{t:s['horizons'][h][t+'_mean_wmape']-r['horizons'][h][t+'_mean_wmape'] for t in ('iron','time')} for h in delta}
    origins={u:v['candidates']['S1']['overall']['loss']-v['candidates']['R2']['overall']['loss'] for u,v in metrics.items() if v['horizon']==1}
    if len(origins)!=6:raise ContractError('six H1 origins required')
    dev={u:metrics[u]['candidates']['S1']['overall']['loss']-metrics[u]['candidates']['R2']['overall']['loss'] for u in ('DEV_LONG','DEV_SHORT')}
    checks=dict(H1_improvement=-delta['H1']>=p['H1_min_improvement'],H1_origins=sum(v<0 for v in origins.values())>=p['H1_min_improved_origins'],
        H2_H3=max(delta[h] for h in ('H2','H3'))<=p['H2_H3_max_regression'],H4_exact=h4_exact and delta['H4']==0,
        J_improvement=r['J']-s['J']>=p['J_min_improvement'],target_horizon=max(v for row in targets.values() for v in row.values())<=p['target_horizon_max_regression'],DEV=max(dev.values())<=p['DEV_max_regression'])
    return dict(passed=all(checks.values()),checks=checks,J=s['J'],J_delta_vs_R2=s['J']-r['J'],horizon_deltas=delta,target_horizon_deltas=targets,H1_origin_deltas=origins,DEV_deltas=dev,scope='RETROSPECTIVE_POST_HOLDOUT_CONSUMPTION')


def replay(data_config,output):
    output.mkdir(parents=True,exist_ok=False);counter={'attempted_target_fits':0}
    try:
        reg=load_yaml('configs/optimization_v0_6/experiment.yaml');scope=load_yaml('configs/optimization_v0_6/access_scope.yaml')
        export=Path(reg['source_export']);old=read_json(export/'manifest.json');scores=Path(reg['source_scores'])
        identities=read_json(export/'component_identities.json');a=algorithm();paths=load_yaml(data_config)['paths']
        if set(paths)-{'operation_hourly','burden_change','data_dictionary'}:raise ContractError('replay config must have no label/test paths')
        if read_json(export/'final_status.json')['attempted_target_fits']!=0 or read_json(scores/'final_status.json')['G0']!='PASS_OPT14':raise ContractError('verified v5 replay required')
        evidence={str(p):p for p in [export/'manifest.json',export/'component_identities.json',scores/'final_status.json']}
        for unit,*_ in units(reg):
            for c in ('P2','R2'):p=scores/'units'/unit/f'{c}_errors.csv';evidence[str(p)]=p
        sources=file_identities({str(p):p for p in [*Path('src/bf_tap').rglob('*.py'),*Path('configs/optimization_v0_6').glob('*.yaml')]})
        manifest=dict(registration=reg,sources=sources,evidence=file_identities(evidence),inputs=file_identities(paths),
            old_ledger_sha256=file_sha256(scope['old_ledger']),active_sha256=file_sha256('configs/optimization_v0_4/active_release.yaml'),
            holdout_consumed=True,label_paths_in_inference_config=False)
        atomic_write_json(output/'manifest.json',manifest);append_access(output,'OPT17_existing_error_targets_scoring_and_zero_fit_replay')
        verify_file_identities(old['evidence'])
        contract=build_inference_source_contract(manifest['inputs'],semantic_contract_sha256=a['contract_digests']['semantic_contract_sha256'])
        op,burden,_=_load_process_sources(paths,a['semantic'],a['features']);builder=ComponentFeatures(a,op,burden)
        metrics={};align=[];rows=[];actuals=[];h4_exact=True;diagnostics={}
        with forbid_fit(counter):
            loaded={k:load_component(v,a,contract) for k,v in identities.items() if v['role'] in ('O0','OR','HR')}
            for unit,cutoff,horizon,start,end in units(reg):
                src=scores/'units'/unit/'R2_errors.csv'
                samples=read_csv(src,usecols=META,time_columns=['reference_time'])[META]
                if not ((samples.reference_time>=start)&(samples.reference_time<end)).all():raise ContractError('unit time boundary')
                h=horizons(samples,cutoff)
                if horizon is not None and not (h==horizon).all():raise ContractError('calendar/grid horizon mismatch')
                entries={role:identities[f'{cutoff.month}/{role}'] for role in ('O0','OR','HR')}
                parts=predict_parts(samples,entries,{role:loaded[f'{cutoff.month}/{role}'] for role in entries},builder)
                actual=validate_frame(pd.read_csv(src,dtype={'sample_id':str}),labeled=True);actuals.append(actual[META+TARGETS])
                expected=actual.set_index('sample_id').sort_index()
                for name in ('P2','R2'):
                    archive=validate_frame(pd.read_csv(scores/'units'/unit/f'{name}_errors.csv',dtype={'sample_id':str}),labeled=True).set_index('sample_id').sort_index()
                    fresh=parts[name].set_index('sample_id').sort_index()
                    if not expected.index.equals(archive.index) or not expected[META[1:]+TARGETS].equals(archive[META[1:]+TARGETS]):raise ContractError('endpoint identity/label mismatch')
                    maximum=float(np.abs(fresh[PRED]-archive[PRED]).to_numpy().max())
                    if maximum>=reg['endpoint_tolerance_exclusive']:raise ContractError('endpoint failed reproduction')
                    align.append(dict(unit=unit,endpoint=name,max_abs_difference=maximum))
                parts['S1']=route(samples,cutoff,parts)
                mask=h.to_numpy()==4
                h4_exact &= np.array_equal(parts['S1'][PRED].to_numpy()[mask],parts['R2'][PRED].to_numpy()[mask])
                dest=output/'units'/unit;dest.mkdir(parents=True)
                metrics[unit]=dict(horizon=horizon,origin_id=f'O2024{cutoff.month:02d}',candidates={})
                for name,pred in parts.items():
                    metrics[unit]['candidates'][name]={'overall':score_predictions(actual,pred)}
                    errors=error_contributions(actual,pred);errors['candidate']=name;errors['origin']=f'O2024{cutoff.month:02d}';errors['horizon']=horizon;errors['calendar_horizon']=h.to_numpy();errors['unit']=unit
                    errors.to_csv(dest/f'{name}_errors.csv',index=False);rows.append(errors)
                    for key,g in [('month',actual.reference_time.dt.strftime('%Y-%m')),('spout',actual.spout_no.astype(str))]:
                        for value in g.unique():
                            subset=actual.loc[g==value];diagnostics[f'{unit}/{name}/{key}/{value}']=score_predictions(subset,pred.loc[pred.sample_id.isin(subset.sample_id)])
                print(f'OPT17 cold replay {unit}; fits=0',flush=True)
        all_actual=pd.concat(actuals)
        if (all_actual.groupby('sample_id')[META[1:]+TARGETS].nunique()>1).any().any():raise ContractError('cross-origin labels/metadata differ')
        summary=aggregate_grid(metrics);gate=deployment_gate(metrics,summary,reg,h4_exact)
        atomic_write_json(output/'metrics.json',metrics);atomic_write_json(output/'summary.json',summary);atomic_write_json(output/'acceptance.json',gate)
        atomic_write_json(output/'endpoint_alignment.json',dict(checks=align,max_abs_difference=max(x['max_abs_difference'] for x in align),H4_exact=h4_exact))
        atomic_write_json(output/'group_metrics.json',diagnostics)
        pd.concat(rows,ignore_index=True).to_csv(output/'all_errors.csv',index=False)
        verify_file_identities(manifest['sources']);verify_file_identities(manifest['evidence']);verify_file_identities(manifest['inputs'])
        if file_sha256(scope['old_ledger'])!=manifest['old_ledger_sha256'] or file_sha256('configs/optimization_v0_4/active_release.yaml')!=manifest['active_sha256']:raise ContractError('old lifecycle changed')
        atomic_write_json(output/'final_status.json',dict(G0='PASS_OPT17',G1='PASS' if gate['passed'] else 'FAIL',**counter,completed_target_fits=0,units=20,active_R2_unchanged=True))
    except Exception as e:
        atomic_write_json(output/'failure.json',dict(error=str(e),**counter));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-config',required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args();replay(a.data_config,a.output)
