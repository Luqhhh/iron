"""OPT-20 fixed development run: causal OOF, rate fits and structural LAD correction."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..artifacts import (atomic_write_json, file_sha256, file_identities, stable_digest,
                        verify_file_identities, build_inference_source_contract, runtime_environment)
from ..config import load_yaml
from ..exceptions import ContractError
from ..io import parse_local_time, read_csv
from ..models.baseline import DualTargetBaseline
from ..offline import _load_process_sources
from ..schema import validate_samples,validate_history,validate_cross_table_consistency
from ..metrics import score_predictions
from .component_export import META,PRED,ComponentFeatures,read_json,units,forbid_fit
from .final_lifecycle import algorithm,label_metadata,eligible,_read_eligible_rows
from .refresh_factorial import Context,stamp
from .horizon_router import load_component
from .snapshot_ensemble import blend
from .structural import INPUT,fit_correction,apply_correction,directions
from .rate_model import RateModel
from .validation import aggregate_grid,error_contributions
from .pseudo_evaluation import gate


def append_ledger(scope,output):
    import fcntl
    path=Path(scope['new_ledger']);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX);f.seek(0);prior=f.read().splitlines()
        f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),purpose=scope['authorization'],
            manifest_sha256=file_sha256(output/'manifest.json'),output=str(output),
            previous_sha256=stable_digest(prior[-1]) if prior else None,holdout_consumed=True))+'\n')


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    counts=dict(attempted_target_fits=0,completed_target_fits=0,rate_fits=0,early_R2_target_fits=0,structural_LAD_target_fits=0)
    try:
        reg=load_yaml('configs/optimization_v0_8/experiment.yaml');scope=load_yaml('configs/optimization_v0_8/access_scope.yaml')
        protection=load_yaml(scope['protection_contract']);a=algorithm()
        paths=load_yaml('configs/data.local.yaml')['paths'];paths={k:paths[k] for k in ('train_samples','tap_history_train','operation_hourly','burden_change','data_dictionary')}
        scores=Path(reg['source_scores']);export=Path(reg['source_export'])
        if read_json(scores/'final_status.json')['G0']!='PASS_OPT14':raise ContractError('verified endpoint scores required')
        identities=read_json(export/'component_identities.json')
        active=load_yaml('configs/optimization_v0_4/active_release.yaml')
        if file_sha256(active['test_a_zip'])!='e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf':raise ContractError('R2 ZIP changed')
        meta=label_metadata(paths,a['semantic']);last=stamp(11)
        evidence=[export/'component_identities.json',scores/'final_status.json',Path(scope['protection_contract']),Path('configs/optimization_v0_4/active_release.yaml'),Path(active['test_a_zip'])]
        evidence += [scores/'units'/u/'R2_errors.csv' for u,*_ in units(reg)]
        source_paths=[*Path('src/bf_tap').rglob('*.py'),*Path('configs/optimization_v0_8').glob('*.yaml'),*Path('tests').glob('test_optimization_v8*.py'),Path('uv.lock'),Path('docs/optimization_v0_8/PLAN.md')]
        manifest=dict(registration=reg,scope=scope,protection=protection,algorithm=a,
            inputs=file_identities(paths),evidence=file_identities({str(p):p for p in evidence}),
            sources=file_identities({str(p):p for p in source_paths}),
            old_ledger_sha256=file_sha256(scope['old_ledger']),environment=runtime_environment(),
            training_reference_end_exclusive=str(last),test_inputs_read=False)
        atomic_write_json(output/'manifest.json',manifest);append_ledger(scope,output)
        selected=eligible(meta,last)
        labels=_read_eligible_rows(paths['train_samples'],selected.sample_id).merge(selected[['sample_id','label_available_at']],on='sample_id',validate='one_to_one')
        history=_read_eligible_rows(paths['tap_history_train'],selected.sample_id)
        for col in ('reference_time','tap_end_time'):history[col]=parse_local_time(history[col],col)
        history['available_at']=history.tap_end_time
        validate_samples(labels,labeled=True);validate_history(history);validate_cross_table_consistency(labels,history)
        if (labels.reference_time>=last).any() or (labels.label_available_at>last).any():raise ContractError('training label scope exceeded')
        op,burden,_=_load_process_sources(paths,a['semantic'],a['features'])
        builder=ComponentFeatures(a,op,burden)
        contract=build_inference_source_contract(manifest['inputs'],semantic_contract_sha256=a['contract_digests']['semantic_contract_sha256'])
        models={};oof_parts=[];training_records=[]
        components={'OR':'E09_PROCESS_CHANGE_E02','HR':'E04'}
        for month in reg['oof_months']:
            cutoff=stamp(month);train=eligible(labels,cutoff)
            h=history.loc[(history.reference_time<cutoff)&(history.available_at<=cutoff)].copy()
            validate_history(h)
            if set(h.sample_id)!=set(train.sample_id):raise ContractError('training/history identities differ')
            train_meta=train[META]
            training=dict(fit_cutoff=str(cutoff),history_cutoff=str(cutoff),label_available_cutoff=str(cutoff),
                variant='R2',rows=len(train),sample_ids_sha256=stable_digest(train.sample_id.astype(str).tolist()),
                reference_max=str(train.reference_time.max()),label_available_max=str(train.label_available_at.max()),
                history_available_max=str(h.available_at.max()),training_history_policy='original_per_sample_asof')
            loaded={};entries={}
            for role,component in components.items():
                if month>=6:
                    entry=identities[f'{month}/{role}'];model,snapshot=load_component(entry,a,contract)
                    if model.bundle_metadata_['training']['sample_ids_sha256']!=training['sample_ids_sha256']:raise ContractError('reused R2 training identity differs')
                else:
                    X=Context.X(builder,train_meta,cutoff,component,'R2',h)
                    model=DualTargetBaseline(a['baseline']['parameters'],('spout_no',))
                    counts['attempted_target_fits']+=2;model.fit(X,train[['tap_iron','tap_time_len']])
                    counts['completed_target_fits']+=2;counts['early_R2_target_fits']+=2
                    directory=output/'models'/str(month)/role
                    model.save(directory,history_snapshot=h,metadata=dict(baseline_config=a['baseline'],feature_config=a['features'],
                        semantic_contract=a['semantic'],contract_digests=a['contract_digests'],inference_source_contract=contract,
                        training=dict(**training,component=component),code_identity=manifest['sources'],
                        environment=manifest['environment'],lockfile_sha256=file_sha256('uv.lock')))
                    restored=DualTargetBaseline.load(directory)
                    if not np.array_equal(model.predict_raw(X).to_numpy(),restored.predict_raw(X).to_numpy()):raise ContractError('early R2 roundtrip changed')
                    model=restored;snapshot=model.load_history_snapshot()
                    entry=dict(path=str(directory),cutoff=str(cutoff),component=component,variant='R2',role=role,
                        bundle_sha256=file_sha256(directory/'bundle.json'),history_snapshot_sha256=file_sha256(directory/'history_snapshot.csv'))
                entries[role]=entry;loaded[role]=(model,snapshot)
            X=Context.X(builder,train_meta,cutoff,reg['rate']['component'],'R2',h)
            rate=RateModel();counts['attempted_target_fits']+=1
            rate.fit(X,train.tap_iron,train.tap_time_len,a['baseline']['parameters'])
            counts['completed_target_fits']+=1;counts['rate_fits']+=1
            directory=output/'models'/str(month)/'rate'
            rate.save(directory,dict(parameters=a['baseline']['parameters'],training=training,
                inference_source_contract=contract,manifest_sha256=file_sha256(output/'manifest.json'),
                rate_spec=reg['rate'],algorithm_contract=a['contract_digests']),h)
            restored=RateModel.load(directory)
            if not np.array_equal(rate.predict(X),restored.predict(X)):raise ContractError('rate roundtrip changed')
            record=dict(**training,month=month,rate_bundle_sha256=file_sha256(directory/'bundle.json'),
                rate_excluded_zero_duration=rate.excluded_zero_duration,rate_training_rows=rate.training_rows,base_components=entries)
            training_records.append(record)
            models[month]=(entries,loaded,restored,h)
            samples=meta.loc[(meta.reference_time>=cutoff)&(meta.reference_time<cutoff+pd.DateOffset(months=1)),META]
            pred=predict_inputs(samples,models[month],builder,reg)
            oof=samples.merge(pred,on='sample_id',validate='one_to_one').merge(meta[['sample_id','label_available_at']],on='sample_id',validate='one_to_one')
            for column,value in [('fold_cutoff',cutoff),('train_reference_max',train.reference_time.max()),('train_available_max',train.label_available_at.max()),('history_available_max',h.available_at.max())]:oof[column]=value
            oof_parts.append(oof)
            dest=output/'oof';dest.mkdir(exist_ok=True);oof.to_csv(dest/f'{month}.csv',index=False)
            atomic_write_json(output/'models'/str(month)/'fit_record.json',record)
            builder.cache.clear()
            print(f'OPT20 cutoff {month}: rate trained ({rate.training_rows} rows), causal OOF {len(oof)}; model fits={counts["completed_target_fits"]}',flush=True)
        if counts['rate_fits']!=8 or counts['early_R2_target_fits']!=8:raise ContractError('registered fit count differs')
        atomic_write_json(output/'training_records.json',training_records)
        oof=pd.concat(oof_parts,ignore_index=True)
        # Only earlier official training labels can join correction training. November remains unparsed.
        oof_labeled=oof.merge(labels[['sample_id','tap_iron','tap_time_len']],on='sample_id',validate='one_to_one')
        predictions={};coefficients={};inference_counter={'attempted_target_fits':0}
        with forbid_fit(inference_counter):
            for month,n in reg['origins'].items():
                cutoff=stamp(month);alpha,used=fit_correction(oof_labeled,cutoff,reg['correction']['minimum_OOF_rows'],reg['rate']['unusable_predicted_rate_max'])
                counts['structural_LAD_target_fits']+=2
                used.to_csv(output/'oof'/f'correction_training_{month}.csv',index=False)
                coefficients[str(month)]=dict(alpha=alpha,fit_cutoff=str(cutoff),OOF_rows=len(used),
                    OOF_ids_sha256=stable_digest(used.sample_id.astype(str).tolist()),
                    OOF_reference_max=str(used.reference_time.max()),OOF_label_available_max=str(used.label_available_at.max()),
                    OOF_fold_cutoffs=sorted(used.fold_cutoff.astype(str).unique().tolist()))
                dest=output/'corrections';dest.mkdir(exist_ok=True);atomic_write_json(dest/f'{month}.json',coefficients[str(month)])
                alpha=read_json(dest/f'{month}.json')['alpha']
                samples=meta.loc[(meta.reference_time>=cutoff)&(meta.reference_time<cutoff+pd.DateOffset(months=n)),META]
                parts=predict_inputs(samples,models[month],builder,reg)
                corrected=apply_correction(parts,alpha,reg['rate']['unusable_predicted_rate_max'])
                reverse=predict_inputs(samples.iloc[::-1],models[month],builder,reg)
                if not parts.set_index('sample_id').sort_index().equals(reverse.set_index('sample_id').sort_index()):raise ContractError('input reversal changes base/rate')
                reverse=apply_correction(reverse,alpha,reg['rate']['unusable_predicted_rate_max'])
                if not corrected.set_index('sample_id').sort_index().equals(reverse.set_index('sample_id').sort_index()):raise ContractError('input reversal changes correction')
                predictions[month]=(parts,corrected)
                dest=output/'predictions';dest.mkdir(exist_ok=True)
                parts.to_csv(dest/f'{month}_inputs.csv',index=False);corrected.to_csv(dest/f'{month}_V1.csv',index=False)
                print(f'OPT20 outer {month}: OOF rows={len(used)}, alpha={alpha}, reversal PASS',flush=True)
        atomic_write_json(output/'predictions_complete.json',dict(**counts,inference_fit_attempts=inference_counter['attempted_target_fits']))
        metrics={};alignment=[];diagnostics={}
        for unit,cutoff,horizon,start,end in units(reg):
            actual=read_csv(scores/'units'/unit/'R2_errors.csv',time_columns=['reference_time'])
            expected=meta.loc[(meta.reference_time>=start)&(meta.reference_time<end),META].set_index('sample_id').sort_index()
            truth=actual.set_index('sample_id').sort_index()
            if actual.sample_id.duplicated().any() or not expected.index.equals(truth.index) or not expected[META[1:]].equals(truth[META[1:]]):raise ContractError('scoring identity mismatch')
            parts,v1=predictions[cutoff.month];base=parts[['sample_id',*PRED]]
            metrics[unit]=dict(horizon=horizon,candidates={});dest=output/'units'/unit;dest.mkdir(parents=True)
            for name,pred in [('U0',base),('U1',v1)]:
                pred=pred.loc[pred.sample_id.isin(actual.sample_id)]
                if name=='U0':
                    diff=float(np.abs(pred.set_index('sample_id').sort_index()[PRED]-truth[PRED]).to_numpy().max())
                    if diff>=1e-8:raise ContractError('R2 endpoint reproduction failed')
                    alignment.append(dict(unit=unit,max_abs_difference=diff))
                metrics[unit]['candidates'][name]={'overall':score_predictions(actual,pred)}
                error_contributions(actual,pred).to_csv(dest/f'{name}_errors.csv',index=False)
            p=parts.loc[parts.sample_id.isin(actual.sample_id)].set_index('sample_id').sort_index()
            corrected=v1.set_index('sample_id').loc[p.index]
            rate=p.pred_rate;valid=truth.tap_time_len>0
            diagnostics[unit]=dict(unusable_rate_rows=int((rate<=reg['rate']['unusable_predicted_rate_max']).sum()),
                predicted_rate_quantiles={str(q):float(rate.quantile(q)) for q in (.05,.5,.95)},
                rate_MAE_positive_duration=float(np.abs(rate[valid]-truth.tap_iron[valid]/truth.tap_time_len[valid]).mean()),
                delta_mean=(corrected[PRED]-p[PRED]).mean().to_dict(),delta_abs_max=(corrected[PRED]-p[PRED]).abs().max().to_dict())
        summary=aggregate_grid(metrics);acceptance=gate(metrics,summary,reg)
        acceptance['candidate']='V1_RATE_STRUCTURAL';acceptance['alias']='U1 denotes V1 in shared gate/score artifacts'
        for name,data in [('metrics',metrics),('summary',summary),('acceptance',acceptance),('coefficients',coefficients),('diagnostics',diagnostics),('endpoint_alignment',alignment)]:atomic_write_json(output/f'{name}.json',data)
        for key in ('inputs','evidence','sources'):verify_file_identities(manifest[key])
        if file_sha256(scope['old_ledger'])!=manifest['old_ledger_sha256']:raise ContractError('old ledger changed')
        atomic_write_json(output/'final_status.json',dict(G0='PASS_OPT20',G1='PASS_REQUIRES_FINAL_RELEASE_VALIDATION' if acceptance['passed'] else 'FAIL_CLOSE_V1',
            **counts,inference_fit_attempts=0,units=20,active_R2_unchanged=True,test_inputs_read=False))
        print(json.dumps(acceptance,indent=2),flush=True)
    except Exception as exc:
        atomic_write_json(output/'failure.json',dict(error=str(exc),**counts));raise


def predict_inputs(samples,model_tuple,builder,reg):
    entries,loaded,rate,rate_history=model_tuple;parts={}
    for role,(model,history) in loaded.items():
        X=builder.X(samples[META],entries[role],history)
        p=model.predict_raw(X).clip(lower=0);p.insert(0,'sample_id',samples.sample_id.to_numpy());parts[role]=p
    result=blend(parts['OR'],parts['HR'],.8)
    X=builder.X(samples[META],entries['OR'],rate_history)
    r=pd.DataFrame(dict(sample_id=samples.sample_id.to_numpy(),pred_rate=rate.predict(X)))
    return result.merge(r,on='sample_id',validate='one_to_one')[INPUT]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
