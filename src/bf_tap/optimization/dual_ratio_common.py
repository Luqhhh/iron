"""Shared identity, scoring and week-cluster diagnostics for OPT-21/22."""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ..artifacts import atomic_write_json,file_identities,file_sha256,stable_digest,verify_file_identities
from ..config import load_yaml
from ..exceptions import ContractError
from ..metrics import score_predictions
from .component_export import META,PRED,read_json,units
from .component_ablation import validate_frame
from .validation import aggregate_grid,error_contributions


def frame(path,usecols=None):
    return pd.read_csv(path,float_precision='round_trip',dtype={'sample_id':str},usecols=usecols)


def registry():return load_yaml('configs/optimization_v0_9/experiment.yaml')


def record_access(output,purpose):
    import fcntl
    scope=load_yaml('configs/optimization_v0_9/access_scope.yaml');path=Path(scope['new_ledger'])
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX);f.seek(0);prior=f.read().splitlines()
        f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),purpose=purpose,authorization=scope['authorization'],
            manifest_sha256=file_sha256(output/'manifest.json'),previous_sha256=stable_digest(prior[-1]) if prior else None,holdout_consumed=True))+'\n')


def freeze(output,purpose,extra=(),inputs=None):
    reg=registry();scope=load_yaml('configs/optimization_v0_9/access_scope.yaml')
    load_yaml(scope['protection_contract']);source=Path(reg['source_v8'])
    if read_json(source/'final_status.json')['G0']!='PASS_OPT20':raise ContractError('verified V1 source required')
    if read_json(Path(reg['source_v8_cold'])/'validation.json')['status']!='PASS':raise ContractError('V1 cold audit required')
    old=read_json(source/'manifest.json')
    for key in ('inputs','sources','evidence'):verify_file_identities(old[key])
    evidence=[source/'manifest.json',source/'final_status.json',source/'summary.json',source/'training_records.json',
        Path(reg['source_v8_cold'])/'validation.json',Path(scope['protection_contract']),*map(Path,extra)]
    for name in ('active_release','fallback_release'):
        path=Path(reg[name]);release=load_yaml(path)
        if file_sha256(release['test_a_zip'])!=release['test_a_zip_sha256']:raise ContractError('incumbent archive changed')
        evidence.extend([path,Path(release['test_a_zip'])])
    for unit,*_ in units(reg):
        evidence.extend(source/'units'/unit/f'{c}_errors.csv' for c in ('U0','U1'))
    sources=[*Path('src/bf_tap').rglob('*.py'),*Path('configs/optimization_v0_9').glob('*.yaml'),Path('docs/optimization_v0_9/PLAN.md')]
    manifest=dict(purpose=purpose,registration=reg,scope=scope,evidence=file_identities({str(p):p for p in evidence}),
        sources=file_identities({str(p):p for p in sources}),inputs=file_identities(inputs or {}),
        old_ledger_sha256=file_sha256(scope['old_ledger']),test_inputs_used_for_selection=False)
    atomic_write_json(output/'manifest.json',manifest);record_access(output,purpose)
    return manifest


def verify_manifest(manifest):
    for key in ('evidence','sources','inputs'):verify_file_identities(manifest[key])
    if file_sha256(manifest['scope']['old_ledger'])!=manifest['old_ledger_sha256']:raise ContractError('old protected ledger changed')


def score(output,reg,provider):
    """provider returns complete prediction-only frames for each registered unit."""
    source=Path(reg['source_v8']);metrics={};groups={};all_errors=[];actuals=[]
    exact={'V2_DUAL_RATIO':True,'V3_TIME_STRUCTURAL':True};table=[]
    for unit,cutoff,horizon,start,end in units(reg):
        actual=validate_frame(frame(source/'units'/unit/'U0_errors.csv'),labeled=True)
        v1=validate_frame(frame(source/'units'/unit/'U1_errors.csv'),labeled=True)
        a=actual.set_index('sample_id').sort_index();v=v1.set_index('sample_id').sort_index()
        if not a.index.equals(v.index) or not a[META[1:]+['tap_iron','tap_time_len']].equals(v[META[1:]+['tap_iron','tap_time_len']]):raise ContractError('R2/V1 label or metadata mismatch')
        if not ((actual.reference_time>=start)&(actual.reference_time<end)).all():raise ContractError('unit boundary mismatch')
        actuals.append(actual[META+['tap_iron','tap_time_len']])
        parts={'R2':actual[['sample_id',*PRED]],'V1':v1[['sample_id',*PRED]]}
        parts.update(provider(unit,cutoff,actual[META],parts))
        metrics[unit]=dict(horizon=horizon,origin_id=f'O2024{cutoff.month:02d}',candidates={});groups[unit]={}
        dest=output/'units'/unit;dest.mkdir(parents=True)
        for candidate,pred in parts.items():
            pred=validate_frame(pred)
            if candidate in exact:
                reference=v if candidate=='V2_DUAL_RATIO' else a
                x=pred.set_index('sample_id').sort_index()
                exact[candidate]&=x.index.equals(reference.index) and np.array_equal(x.pred_tap_iron.to_numpy(),reference.pred_tap_iron.to_numpy())
            metric=score_predictions(actual,pred);metrics[unit]['candidates'][candidate]={'overall':metric}
            errors=error_contributions(actual,pred);errors['candidate']=candidate;errors['origin']=f'O2024{cutoff.month:02d}';errors['horizon']=horizon;errors['unit']=unit
            errors.to_csv(dest/f'{candidate}_errors.csv',index=False);all_errors.append(errors)
            table.append(dict(unit=unit,candidate=candidate,horizon=horizon,E=metric['loss'],**{f'{t}_{k}':value for t in ('iron','time') for k,value in metric[t].items()}))
            groups[unit][candidate]={str(spout):score_predictions(part,pred.loc[pred.sample_id.isin(part.sample_id)]) for spout,part in actual.groupby('spout_no')}
    all_actual=pd.concat(actuals)
    if (all_actual.groupby('sample_id')[META[1:]+['tap_iron','tap_time_len']].nunique()>1).any().any():raise ContractError('inconsistent repeated sample labels/metadata')
    summary=aggregate_grid(metrics);errors=pd.concat(all_errors,ignore_index=True)
    atomic_write_json(output/'metrics.json',metrics);atomic_write_json(output/'summary.json',summary);atomic_write_json(output/'spout_metrics.json',groups)
    errors.to_csv(output/'all_errors.csv',index=False);pd.DataFrame(table).to_csv(output/'all_units.csv',index=False)
    pd.DataFrame(table).loc[lambda x:x.horizon==1].to_csv(output/'h1_origins.csv',index=False)
    return metrics,summary,errors,exact


def week_intervals(errors,candidate,reference,repetitions=1000,seed=2026):
    """Paired calendar-week multiplicities, including target and DEV intervals."""
    parts={c:errors.loc[errors.candidate==c].copy() for c in (candidate,reference)}
    identity=['unit','sample_id','week','tap_iron','tap_time_len']
    for p in parts.values():
        if p.empty or p.duplicated(['unit','sample_id']).any():raise ContractError('duplicate/missing bootstrap identities')
    left,right=[p[identity].sort_values(['unit','sample_id']).reset_index(drop=True) for p in parts.values()]
    if not left.equals(right):raise ContractError('unpaired bootstrap data')
    weeks=sorted(right.week.unique());unit_names=sorted(right.unit.unique())
    index=pd.MultiIndex.from_product([weeks,unit_names],names=['week','unit'])
    targets=['tap_iron','tap_time_len'];errorcols=['abs_error_'+t for t in targets]
    def tensor(p,columns):return p.groupby(['week','unit'])[columns].sum().reindex(index,fill_value=0).to_numpy().reshape(len(weeks),-1)
    weights=np.random.default_rng(seed).multinomial(len(weeks),np.full(len(weeks),1/len(weeks)),size=repetitions)
    denom=(weights@tensor(parts[reference],targets)).reshape(repetitions,len(unit_names),2)
    valid=(denom>0).all(axis=(1,2))
    if not valid.any():raise ContractError('no valid bootstrap replicates')
    delta=(weights@(tensor(parts[candidate],errorcols)-tensor(parts[reference],errorcols))).reshape(repetitions,len(unit_names),2)[valid]/denom[valid]
    lookup=parts[reference][['unit','horizon']].drop_duplicates().set_index('unit').horizon.to_dict()
    def interval(x):return dict(zip(('p025','median','p975'),np.quantile(x,[.025,.5,.975]).tolist()))
    def report(x):return {'E':interval(x.mean(axis=1)),'iron_wmape':interval(x[:,0]),'time_wmape':interval(x[:,1])}
    results={};horizons=[]
    for h in range(1,5):
        mask=[i for i,u in enumerate(unit_names) if lookup[u]==h]
        if not mask:raise ContractError('incomplete bootstrap horizon grid')
        value=delta[:,mask].mean(axis=1);horizons.append(value);results[f'H{h}']=report(value)
    results['J']=report(np.mean(horizons,axis=0))
    for unit in ('DEV_LONG','DEV_SHORT'):results[unit]=report(delta[:,unit_names.index(unit)])
    return dict(candidate=candidate,reference=reference,seed=seed,requested_repetitions=repetitions,valid_repetitions=int(valid.sum()),deltas=results,
        scope='SHARED_CALENDAR_WEEK_CONSUMED_RETROSPECTIVE_NOT_SELECTION_ADJUSTED')
