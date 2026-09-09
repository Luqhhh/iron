"""Complete origin trajectories first; archived labels enter only the scoring phase."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..artifacts import atomic_write_json, file_identities, file_sha256, verify_file_identities, build_inference_source_contract
from ..config import load_yaml
from ..exceptions import ContractError
from ..io import read_csv
from ..metrics import score_predictions
from ..offline import _load_process_sources
from .component_export import META, PRED, ComponentFeatures, forbid_fit, read_json, units
from .final_lifecycle import algorithm
from .horizon_router import load_component
from .pseudo_history import rollout
from .validation import aggregate_grid, error_contributions


def gate(metrics, summary, reg):
    a,b = summary['U1'],summary['U0']
    delta = {h:a['horizons'][h]['mean_loss']-b['horizons'][h]['mean_loss'] for h in ('H1','H2','H3','H4')}
    origins = {u:v['candidates']['U1']['overall']['loss']-v['candidates']['U0']['overall']['loss'] for u,v in metrics.items() if v['horizon']==1}
    if len(origins)!=6: raise ContractError('six H1 origins required')
    targets = {t:a['horizons']['H1'][t+'_mean_wmape']-b['horizons']['H1'][t+'_mean_wmape'] for t in ('iron','time')}
    dev = {u:metrics[u]['candidates']['U1']['overall']['loss']-metrics[u]['candidates']['U0']['overall']['loss'] for u in ('DEV_LONG','DEV_SHORT')}
    checks = dict(H1_mean=-delta['H1']>=reg['H1_min_improvement'],
        H1_origins=sum(v<0 for v in origins.values())>=reg['H1_min_improved_origins'],
        recent_H1=all(origins[f'O2024{m:02d}_H1']<0 for m in reg['recent_H1_must_improve']),
        H1_targets=max(targets.values())<=reg['H1_target_max_regression'],
        H234=max(delta[h] for h in ('H2','H3','H4'))<=reg['H234_max_regression'],
        extended_J=a['J']-b['J']<=reg['J_max_regression'],DEV=max(dev.values())<=reg['DEV_max_regression'])
    return dict(passed=all(checks.values()), checks=checks, horizon_deltas=delta,
        H1_origin_deltas=origins,H1_target_deltas=targets,DEV_deltas=dev,J=a['J'],J_delta=a['J']-b['J'],
        scope='RETROSPECTIVE_POST_HOLDOUT_CONSUMPTION')


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    counter={'attempted_target_fits':0}
    try:
        reg=load_yaml('configs/optimization_v0_7/experiment.yaml')
        scope=load_yaml('configs/optimization_v0_7/access_scope.yaml')
        load_yaml(scope['protection_contract'])
        active=load_yaml('configs/optimization_v0_4/active_release.yaml')
        if file_sha256(active['test_a_zip'])!=active['test_a_zip_sha256']:raise ContractError('R2 ZIP changed')
        scores=Path(reg['source_scores']);export=Path(reg['source_export'])
        identities=read_json(export/'component_identities.json')
        paths=load_yaml('local/runs/optimization-v0.4-r2-challenger-r1/cold_data.yaml')['paths']
        paths={k:v for k,v in paths.items() if k!='test_a_samples'}
        metadata_path=load_yaml('configs/data.local.yaml')['paths']['train_samples']
        evidence=[scores/'final_status.json',export/'component_identities.json',export/'manifest.json',Path(scope['protection_contract']),Path('configs/optimization_v0_4/active_release.yaml'),Path(active['test_a_zip'])]
        evidence += [scores/'units'/u/'R2_errors.csv' for u,*_ in units(reg)]
        manifest=dict(registration=reg,access_scope=scope,sources=file_identities({str(p):p for p in [*Path('src/bf_tap').rglob('*.py'),*Path('configs/optimization_v0_7').glob('*.yaml')]}),
            evidence=file_identities({str(p):p for p in evidence}),inputs=file_identities(paths),
            metadata=file_identities({'metadata_only':metadata_path}),old_ledger_sha256=file_sha256(scope['old_ledger']),
            labels_in_inference=False,test_labels_read=False)
        atomic_write_json(output/'manifest.json',manifest)
        import fcntl
        ledger=Path(scope['new_ledger']);ledger.parent.mkdir(parents=True,exist_ok=True)
        with ledger.open('a') as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX)
            f.write(json.dumps(dict(output=str(output),purpose=scope['authorization'],manifest_sha256=file_sha256(output/'manifest.json'),holdout_consumed=True))+'\n')
        a=algorithm();contract=build_inference_source_contract(manifest['inputs'],semantic_contract_sha256=a['contract_digests']['semantic_contract_sha256'])
        op,burden,_=_load_process_sources(paths,a['semantic'],a['features'])
        builder=ComponentFeatures(a,op,burden)
        metadata=read_csv(metadata_path,usecols=META+['tap_no'],time_columns=['reference_time'])
        trajectories={};drift={}
        with forbid_fit(counter):
            for month,n in reg['origins'].items():
                entries={r:identities[f'{month}/{r}'] for r in ('OR','HR')}
                loaded={r:load_component(e,a,contract) for r,e in entries.items()}
                cutoff=pd.Timestamp(entries['OR']['cutoff']);end=cutoff+pd.DateOffset(months=n)
                samples=metadata.loc[(metadata.reference_time>=cutoff)&(metadata.reference_time<end),META+['tap_no']].copy()
                u0,u1,audit,pseudo=rollout(samples,entries,loaded,builder)
                reverse=rollout(samples.iloc[::-1],entries,loaded,builder)
                for expected,actual in zip((u0,u1),reverse[:2]):
                    if not expected.set_index('sample_id').sort_index().equals(actual.set_index('sample_id').sort_index()):raise ContractError('CSV reversal changes output')
                if audit!=reverse[2] or not pseudo.equals(reverse[3]):raise ContractError('CSV reversal changes history')
                trajectories[month]=(u0,u1)
                dest=output/'origins'/str(month);dest.mkdir(parents=True)
                u0.to_csv(dest/'U0.csv',index=False);u1.to_csv(dest/'U1.csv',index=False)
                pseudo.to_csv(dest/'pseudo_rows.csv',index=False)
                with (dest/'audit.jsonl').open('w') as f:
                    for row in audit:f.write(json.dumps(row)+'\n')
                def distributions(frame):
                    return {t:{str(q):float(frame[t].quantile(q)) for q in (.05,.5,.95)} for t in ('tap_iron','tap_time_len')}
                drift[str(month)]=dict(rows=len(pseudo),max_depth=int(pseudo.depth.max()),
                    pseudo_targets=distributions(pseudo),monthly={m:distributions(g) for m,g in pseudo.groupby(pseudo.reference_time.dt.strftime('%Y-%m'))},
                    max_pending=max(r['predicted_completion_overlap'] for r in audit),
                    max_visible=max(r['pseudo_history_rows_visible'] for r in audit),
                    delta_abs_max=np.abs(np.array([r['prediction_delta'] for r in audit])).max(axis=0).tolist(),
                    delta_signed_mean=np.array([r['prediction_delta'] for r in audit]).mean(axis=0).tolist())
                builder.cache.clear()
                print(f'OPT19 origin {month}: {len(samples)} rows, full {n}-month rollout and reversal PASS; fits=0',flush=True)
        # No scoring target has been read before ALL origin trajectories complete.
        atomic_write_json(output/'rollout_complete.json',dict(origins=6,reversal=True,**counter))
        metrics={};alignment=[]
        for unit,cutoff,horizon,start,end in units(reg):
            actual=read_csv(scores/'units'/unit/'R2_errors.csv',time_columns=['reference_time'])
            if actual.sample_id.duplicated().any() or not ((actual.reference_time>=start)&(actual.reference_time<end)).all():raise ContractError('invalid scoring identity/boundary')
            expected=metadata.loc[(metadata.reference_time>=start)&(metadata.reference_time<end)].set_index('sample_id').sort_index()
            truth=actual.set_index('sample_id').sort_index()
            if not expected.index.equals(truth.index) or not expected[META[1:]].equals(truth[META[1:]]):raise ContractError('scoring metadata mismatch')
            metrics[unit]=dict(horizon=horizon,candidates={});dest=output/'units'/unit;dest.mkdir(parents=True)
            for name,pred in zip(('U0','U1'),trajectories[cutoff.month]):
                pred=pred.loc[pred.sample_id.isin(actual.sample_id)]
                if name=='U0':
                    maximum=float(np.abs(pred.set_index('sample_id').sort_index()[PRED]-truth[PRED]).to_numpy().max())
                    if maximum>=1e-8:raise ContractError('frozen R2 endpoint failed')
                    alignment.append(dict(unit=unit,max_abs_difference=maximum))
                metrics[unit]['candidates'][name]={'overall':score_predictions(actual,pred)}
                error_contributions(actual,pred).to_csv(dest/f'{name}_errors.csv',index=False)
        summary=aggregate_grid(metrics);acceptance=gate(metrics,summary,reg)
        for name,value in [('metrics',metrics),('summary',summary),('acceptance',acceptance),('drift',drift),('endpoint_alignment',alignment)]:atomic_write_json(output/f'{name}.json',value)
        for name in ('sources','evidence','inputs','metadata'):verify_file_identities(manifest[name])
        if file_sha256(scope['old_ledger'])!=manifest['old_ledger_sha256']:raise ContractError('old ledger changed')
        atomic_write_json(output/'final_status.json',dict(G0='PASS_OPT19',G1='NUMERIC_PASS_REQUIRES_DRIFT_REVIEW' if acceptance['passed'] else 'FAIL_CLOSE_PSEUDO_HISTORY',**counter,completed_target_fits=0,units=20,active_R2_unchanged=True))
        print(json.dumps(acceptance,indent=2),flush=True)
    except Exception as e:
        atomic_write_json(output/'failure.json',dict(error=str(e),**counter));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
