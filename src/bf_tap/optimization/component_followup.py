"""OPT-15: the two pre-registered component fits, only after OPT-14 gate failure."""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from ..artifacts import atomic_write_json, file_sha256, file_identities, stable_digest, verify_file_identities, runtime_environment, build_inference_source_contract
from ..config import load_yaml
from ..exceptions import ContractError
from ..models.baseline import DualTargetBaseline
from ..offline import _load_process_sources
from ..schema import validate_samples
from .component_export import META, PRED, ComponentFeatures, load_verified_component, units, read_json, forbid_fit
from .component_ablation import gates, validate_frame
from .final_lifecycle import algorithm, label_metadata, eligible, _read_eligible_rows, source_files
from .refresh_factorial import Context, stamp, snapshot_identity
from .history_component_v5 import transform, MAIN, AUX
from .snapshot_ensemble import blend
from .validation import aggregate_grid, error_contributions
from .v3_evidence import paired_week_intervals
from ..metrics import score_predictions

TARGETS=['tap_iron','tap_time_len']
COMPONENTS={'T1':MAIN,'T2':AUX}


def reserve_fit(path,identity,limit=24):
    """Reserve two target calls before fitting; even failed attempts consume budget."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX);f.seek(0)
        previous=[json.loads(x) for x in f.read().splitlines()]
        used=sum(r['target_fits'] for r in previous)
        if any(r['identity']['slot']==identity['slot'] for r in previous):
            raise ContractError('component/cutoff was already attempted; reuse or repair explicitly')
        if used+2>limit:
            raise ContractError('registered single-target fit budget exhausted')
        f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),identity=identity,target_fits=2,
                                total_reserved=used+2),sort_keys=True)+'\n');f.flush()
        return used+2


def training_features(builder,samples,history,cutoff,variant,component):
    if list(samples)!=META or samples.sample_id.duplicated().any():
        raise ContractError('training feature input must contain only unique metadata')
    base=Context.X(builder,samples,cutoff,component,'R2',history)
    return transform(base,samples,history,cutoff,variant,component)


def fit_component(builder,labels,history,cutoff,variant,output,manifest,budget_ledger,limit=24):
    component=COMPONENTS[variant]
    train=eligible(labels,cutoff)
    snapshot=history.loc[history.available_at<=cutoff].copy()
    X=training_features(builder,train[META],snapshot,cutoff,variant,component)
    schema=[dict(name=c,dtype=str(X[c].dtype),categorical=c=='spout_no') for c in X]
    identity=dict(slot=f'{variant}/{cutoff}/{component}',component=component,variant=variant,cutoff=str(cutoff),
        schema_sha256=stable_digest(schema),history_sha256=snapshot_identity(snapshot),
        training_samples_sha256=stable_digest(train.sample_id.astype(str).tolist()),
        source_sha256=stable_digest(manifest['sources']),inputs_sha256=stable_digest(manifest['inputs']))
    reserve_fit(budget_ledger,identity,limit)
    model=DualTargetBaseline(builder.a['baseline']['parameters'],('spout_no',))
    model.fit(X,train[TARGETS])
    training=dict(fit_cutoff=str(cutoff),history_cutoff=str(cutoff),label_available_cutoff=str(cutoff),
        model_fit_cutoff=str(cutoff),training_label_available_cutoff=str(cutoff),
        training_history_policy='original_per_sample_asof_and_fit_cutoff',variant=variant,component=component,
        rows=len(train),sample_ids_sha256=identity['training_samples_sha256'],cache_identity=identity)
    model.save(output,history_snapshot=snapshot,metadata=dict(baseline_config=builder.a['baseline'],
        feature_config=builder.a['features'],semantic_contract=builder.a['semantic'],
        contract_digests=builder.a['contract_digests'],inference_source_contract=manifest['source_contract'],
        training=training,code_identity=manifest['sources'],environment=runtime_environment(),
        lockfile_sha256=file_sha256('uv.lock')))
    restored=DualTargetBaseline.load(output)
    if not np.array_equal(model.predict_raw(X).to_numpy(),restored.predict_raw(X).to_numpy()):
        raise ContractError('saved component predictions differ from training object')
    return restored,{**identity,'path':str(output),'bundle_sha256':file_sha256(output/'bundle.json'),
        'feature_schema':schema,'completed_target_fits':2,'training_rows':len(train)}


def append_access(ledger,manifest_path,purpose):
    with Path(ledger).open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),purpose=purpose,
            authorization='user_continue_v0_5_OPT15_OPT16_20260909',
            manifest_sha256=file_sha256(manifest_path),holdout_consumed=True))+'\n');f.flush()


def run(data_config,output):
    output.mkdir(parents=True,exist_ok=False)
    completed=0
    try:
        cfg=load_yaml('configs/optimization_v0_5/opt15.yaml')
        reg=load_yaml(cfg['registration'])
        old=Path(cfg['opt14_scores']);export=Path(cfg['opt14_export'])
        gate14=read_json(old/'acceptance.json');status14=read_json(old/'final_status.json')
        if status14.get('G0')!='PASS_OPT14' or gate14['selected_STAGE_A'] is not None or any(v['STAGE_A']['passed'] for v in gate14['candidates'].values()):
            raise ContractError('OPT-15 requires completed OPT-14 with no STAGE_A winner')
        em=read_json(export/'manifest.json');es=read_json(export/'final_status.json')
        if file_sha256(export/'component_predictions.csv')!=es['predictions_sha256']:
            raise ContractError('cached component predictions changed')
        if read_json('EVIDENCE_STATUS.json')['holdout_consumed'] is not True:
            raise ContractError('holdout must remain consumed')
        verify_file_identities(em['inputs']);verify_file_identities(em['evidence'])
        paths=load_yaml(data_config)['paths']
        inputs=file_identities({k:paths[k] for k in em['inputs']})
        if any(inputs[k]['sha256']!=em['inputs'][k]['sha256'] for k in inputs):
            raise ContractError('data identity changed since OPT-14')
        a=algorithm()
        sources=file_identities({**source_files(),**{str(p):p for p in Path('configs/optimization_v0_5').glob('*.yaml')}})
        contract=build_inference_source_contract(inputs,semantic_contract_sha256=a['contract_digests']['semantic_contract_sha256'])
        manifest=dict(config=cfg,registration=reg,inputs=inputs,sources=sources,source_contract=contract,
            opt14_acceptance_sha256=file_sha256(old/'acceptance.json'),
            opt14_export_manifest_sha256=file_sha256(export/'manifest.json'),
            opt14_prediction_sha256=file_sha256(export/'component_predictions.csv'),
            reference_metrics_sha256=file_sha256(old/'metrics.json'),
            active_release_sha256=file_sha256('configs/optimization_v0_4/active_release.yaml'),
            old_ledger_sha256=file_sha256(em['access_scope']['old_ledger']),holdout_consumed=True,
            test_a_used_for_selection=False)
        atomic_write_json(output/'manifest.json',manifest)
        append_access(cfg['access_ledger'],output/'manifest.json',cfg['purpose'])
        metadata=label_metadata(paths,a['semantic'])
        metadata=metadata.loc[metadata.reference_time>=pd.Timestamp(cfg['training_reference_start'])]
        selected=eligible(metadata,pd.Timestamp(cfg['training_available_cutoff']))
        labels=_read_eligible_rows(paths['train_samples'],selected.sample_id).merge(
            selected[['sample_id','label_available_at']],on='sample_id',validate='one_to_one')
        validate_samples(labels,labeled=True)
        labels=labels.sort_values(['reference_time','sample_id'],kind='mergesort')
        op,burden,_=_load_process_sources(paths,a['semantic'],a['features'])
        builder=ComponentFeatures(a,op,burden)
        loaded={};load_details={}
        counter={'attempted_target_fits':0}
        with forbid_fit(counter):
            for key,identity in em['inventory'].items():
                if identity['variant']=='R2':
                    loaded[key]=load_verified_component(identity,a,contract,metadata)
                    load_details[key]=loaded[key][2]
        # Cross-check the newly authorized targets against the old legal training snapshot.
        h=loaded['11/OR'][1].set_index('sample_id').sort_index()
        y=labels.set_index('sample_id').sort_index()
        if not h.index.equals(y.index) or not np.allclose(h[TARGETS],y[TARGETS],rtol=0,atol=1e-9):
            raise ContractError('official training labels differ from saved authorized history')
        atomic_write_json(output/'reused_components.json',load_details)
        models={};new_ids={}
        for variant in ('T1','T2'):
            role='OR' if variant=='T1' else 'HR'
            for month in reg['origins']:
                cutoff=stamp(month);history=loaded[f'{month}/{role}'][1]
                path=output/'models'/variant/cutoff.strftime('%Y%m%dT%H%M%S')/COMPONENTS[variant]
                model,identity=fit_component(builder,labels,history,cutoff,variant,path,manifest,
                    cfg['fit_budget_ledger'],cfg['max_development_target_fits'])
                key=f'{month}/{variant}';models[key]=model;new_ids[key]=identity;completed+=2
                with (output/'registry.jsonl').open('a') as f:f.write(json.dumps(identity)+'\n')
                print(f'fit {variant} {cutoff}: {identity["training_rows"]} rows; completed target fits={completed}',flush=True)
        atomic_write_json(output/'new_components.json',new_ids)
        metrics=copy.deepcopy(read_json(old/'metrics.json'))
        cache=pd.read_csv(export/'component_predictions.csv',dtype={'sample_id':str})
        diagnostics={};all_errors=[];predictions=[]
        with forbid_fit(counter):
            for unit,cutoff,horizon,start,end in units(reg):
                source=Path(reg['source_run'])/'units'/unit/'R2'/'errors.csv'
                actual=validate_frame(pd.read_csv(source,dtype={'sample_id':str}),labeled=True)
                if not ((actual.reference_time>=start)&(actual.reference_time<end)).all():
                    raise ContractError('evaluation time boundaries differ')
                dest=output/'units'/unit;dest.mkdir(parents=True)
                for variant in ('T1','T2'):
                    component=COMPONENTS[variant];role='OR' if variant=='T1' else 'HR'
                    history=loaded[f'{cutoff.month}/{role}'][1]
                    if set(history.sample_id.astype(str))&set(actual.sample_id):
                        raise ContractError('current evaluation ID in history')
                    X=training_features(builder,actual[META],history,cutoff,variant,component)
                    model=models[f'{cutoff.month}/{variant}']
                    raw=model.predict_raw(X)
                    if not np.array_equal(raw.to_numpy(),model.predict_raw(X.iloc[::-1]).iloc[::-1].to_numpy()):
                        raise ContractError('new component changes under row reversal')
                    pred=raw.clip(lower=0);pred.insert(0,'sample_id',actual.sample_id)
                    reuse=cache.loc[(cache.unit==unit)&(cache.role==cfg['reuse_components'][variant])].copy()
                    reuse=validate_frame(reuse)
                    if not (pd.to_datetime(reuse.fit_cutoff,utc=True)==cutoff).all() or set(reuse.sample_id)!=set(actual.sample_id):
                        raise ContractError('reused prediction identity differs')
                    reuse=reuse[['sample_id',*PRED]]
                    final=blend(pred,reuse,.8) if variant=='T1' else blend(reuse,pred,.8)
                    metrics[unit]['candidates'][variant]={'overall':score_predictions(actual,final)}
                    errors=error_contributions(actual,final)
                    errors['candidate'],errors['origin'],errors['horizon'],errors['unit']=variant,f'O2024{cutoff.month:02d}',horizon,unit
                    errors.to_csv(dest/f'{variant}_errors.csv',index=False);all_errors.append(errors)
                    component_rows=actual[META].copy()
                    for c in PRED:component_rows['raw_'+c]=raw[c].to_numpy();component_rows[c]=pred[c].to_numpy()
                    component_rows['unit'],component_rows['variant']=unit,variant;predictions.append(component_rows)
                    for kind,groups in [('month',actual.reference_time.dt.strftime('%Y-%m')),('spout',actual.spout_no.astype(str))]:
                        for value in groups.unique():
                            part=actual.loc[groups==value]
                            diagnostics[f'{unit}/{variant}/{kind}/{value}']=score_predictions(part,final.loc[final.sample_id.isin(part.sample_id)])
                print(f'evaluated T1/T2 {unit}',flush=True)
        summary=aggregate_grid(metrics)
        acceptance=gates(metrics,summary,reg,candidates=('T1','T2'))
        acceptance['next_step']='OPT16_ELIGIBLE' if acceptance['selected_STAGE_A'] else 'V05_CLOSED_KEEP_R2'
        atomic_write_json(output/'metrics.json',metrics);atomic_write_json(output/'extended_18_summary.json',summary)
        atomic_write_json(output/'acceptance.json',acceptance);atomic_write_json(output/'group_metrics.json',diagnostics)
        pd.concat(predictions,ignore_index=True).to_csv(output/'new_component_predictions.csv',index=False)
        errors=pd.concat(all_errors,ignore_index=True)
        rows=[]
        for unit,v in metrics.items():
            for c in ('E12-raw','R2','T1','T2'):
                s=v['candidates'][c]['overall'];rows.append(dict(unit=unit,horizon=v['horizon'],candidate=c,E=s['loss'],
                    **{f'{t}_{k}':value for t in ('iron','time') for k,value in s[t].items()}))
        table=pd.DataFrame(rows);table.to_csv(output/'all_units.csv',index=False)
        table.loc[table.horizon==1].to_csv(output/'h1_six_origins.csv',index=False)
        errors.loc[pd.to_datetime(errors.reference_time).dt.month.isin([9,10,11])].to_csv(output/'recent_three_months.csv',index=False)
        legacy={k:v for k,v in metrics.items() if k.startswith('O') and int(k[5:7])+v['horizon']<=11}
        atomic_write_json(output/'legacy_14_summary.json',aggregate_grid(legacy))
        refs=[]
        for unit,cutoff,horizon,*_ in units(reg):
            if horizon is None:continue
            f=pd.read_csv(Path(reg['source_run'])/'units'/unit/'R2'/'errors.csv',dtype={'sample_id':str})
            f['candidate'],f['origin'],f['horizon']='R2',f'O2024{cutoff.month:02d}',horizon;refs.append(f)
        grid=pd.concat([errors.loc[errors.horizon.notna()],*refs],ignore_index=True);grid['horizon']=grid.horizon.astype(int)
        intervals={v:paired_week_intervals(grid.loc[grid.candidate.isin([v,'R2'])],v,'R2',**reg['bootstrap']) for v in ('T1','T2')}
        atomic_write_json(output/'paired_week_intervals.json',intervals)
        verify_file_identities(inputs);verify_file_identities(sources);verify_file_identities(em['evidence'])
        if file_sha256('configs/optimization_v0_4/active_release.yaml')!=manifest['active_release_sha256']:
            raise ContractError('active release changed')
        if file_sha256(em['access_scope']['old_ledger'])!=manifest['old_ledger_sha256']:
            raise ContractError('old protected ledger changed')
        atomic_write_json(output/'final_status.json',dict(G0='PASS_OPT15',G1_STAGE_A='PASS' if acceptance['selected_STAGE_A'] else 'FAIL',
            selected_STAGE_A=acceptance['selected_STAGE_A'],completed_target_fits=completed,
            inference_fit_attempts=counter['attempted_target_fits'],test_a_used_for_selection=False,
            active_release_unchanged=True,holdout_consumed=True))
    except Exception as exc:
        atomic_write_json(output/'final_status.json',dict(status='FAILED',error=str(exc),completed_target_fits=completed))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-config',default='configs/data.local.yaml')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.data_config,a.output)

if __name__=='__main__':main()
