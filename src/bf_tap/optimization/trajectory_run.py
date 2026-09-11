"""OPT-24: zero-fit P0, eight fixed direct-time fits, six causal time LAD fits."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ..artifacts import (atomic_write_json, file_identities, file_sha256, stable_digest,
                         verify_file_identities, runtime_environment)
from ..config import load_yaml
from ..exceptions import ContractError
from ..features.trajectory import COLUMNS, VALUES, WINDOWS, STATS, build_trajectory_features, append_trajectory
from ..models.baseline import DualTargetBaseline
from ..offline import _load_process_sources
from .component_export import META, PRED, ComponentFeatures, read_json, units, forbid_fit
from .dual_ratio_common import frame, score, week_intervals, verify_manifest
from .rate_model import RateModel, schema
from .refresh_factorial import Context, stamp, snapshot_identity
from .residual_stack import ResidualFitBudget
from .structural import INPUT, apply_correction
from .structural_run import append_ledger, predict_inputs
from .trajectory_candidate import CANDIDATE, TimeModel, fit_time_coefficient, predict, acceptance

DATES = ['reference_time', 'label_available_at', 'fold_cutoff', 'train_reference_max',
         'train_available_max', 'history_available_max']


def registration():
    reg = load_yaml('configs/optimization_v0_11/experiment.yaml')
    feature = load_yaml(reg['features'])
    if (feature['value_columns'] != list(VALUES) or feature['windows_hours'] != list(WINDOWS)
            or feature['statistics'] != list(STATS) or feature['slope_minimum_distinct_finite_times'] != 3
            or feature['step_maximum_gap_hours'] != 1.5 or feature['step_minimum_valid_pairs'] != 2):
        raise ContractError('fixed trajectory profile differs')
    if reg['candidate'] != CANDIDATE or reg['oof_months'] != list(range(4,12)) or reg['budget']['development_time_CatBoost_fits'] != 8 or reg['budget']['development_time_LAD_fits'] != 6:
        raise ContractError('registered OPT24 identity/budget differs')
    return reg


def dated(path):
    result = frame(path)
    for column in DATES:
        if column in result:
            result[column] = pd.to_datetime(result[column])
    return result


def equal(left, right, columns, tolerance=1e-10):
    a, b = [x.set_index('sample_id').sort_index() for x in (left, right)]
    if a.index.duplicated().any() or not a.index.equals(b.index):
        raise ContractError('prediction identity differs')
    diff = float(np.abs(a[columns].to_numpy()-b[columns].to_numpy()).max())
    if not np.isfinite(diff) or diff > tolerance:
        raise ContractError(f'prediction reproduction differs: {diff}')
    return diff


def freeze(output):
    reg = registration(); scope = load_yaml('configs/optimization_v0_11/access_scope.yaml')
    protection = load_yaml(scope['protection_contract'])
    if not scope['holdout_consumed'] or not read_json('EVIDENCE_STATUS.json')['holdout_consumed']:
        raise ContractError('consumed retrospective scope required')
    source = Path(reg['source_v8']); old = read_json(source/'manifest.json')
    for key in ('sources', 'inputs', 'evidence'):
        verify_file_identities(old[key])
    if read_json(source/'final_status.json')['G0'] != 'PASS_OPT20':
        raise ContractError('completed V1 source required')
    evidence = [source/'manifest.json', source/'final_status.json', source/'training_records.json',
                Path(reg['source_v8_cold'])/'validation.json', Path(scope['protection_contract'])]
    if read_json(evidence[-2])['status'] != 'PASS':
        raise ContractError('old V1 cold evidence required')
    inventory = {}
    for month in reg['oof_months']:
        directory = source/'models'/str(month)
        record = read_json(directory/'fit_record.json'); evidence.append(directory/'fit_record.json')
        evidence.append(source/'oof'/f'{month}.csv')
        inventory[str(month)] = record
        for entry in record['base_components'].values():
            root = Path(entry['path'])
            if file_sha256(root/'bundle.json') != entry['bundle_sha256']:
                raise ContractError('missing/changed reused base bundle; restore original, no fit')
            md = read_json(root/'bundle.json')
            if set(COLUMNS) & {x['name'] for x in md['feature_schema']}:
                raise ContractError('trajectory feature already registered in old schema')
            evidence += [root/'bundle.json', root/'history_snapshot.csv']
            evidence += list(root.glob('*.cbm'))
        if file_sha256(directory/'rate'/'bundle.json') != record['rate_bundle_sha256']:
            raise ContractError('missing/changed original rate bundle; no replacement fit')
        evidence += list((directory/'rate').glob('*'))
    for month in reg['origins']:
        evidence += [source/'predictions'/f'{month}_inputs.csv',source/'predictions'/f'{month}_V1.csv',source/'corrections'/f'{month}.json']
    for unit, *_ in units(reg):
        evidence += [source/'units'/unit/f'{c}_errors.csv' for c in ('U0','U1')]
    for name in ('active_release','fallback_release'):
        path = Path(reg[name]); release = load_yaml(path)
        if file_sha256(release['test_a_zip']) != release['test_a_zip_sha256']:
            raise ContractError('original archive differs')
        evidence += [path, Path(release['test_a_zip'])]
        evidence.append(Path(release['bundle']).parent/'cold_predictions.csv')
        evidence.append(Path(release['bundle']).parent/('cold_data_repaired.yaml' if name == 'active_release' else 'cold_data.yaml'))
        evidence += [p for p in Path(release['bundle']).rglob('*') if p.is_file()]
    paths = {k:v['path'] for k,v in old['inputs'].items() if k in ('operation_hourly','burden_change','data_dictionary')}
    sources = [*Path('src/bf_tap').rglob('*.py'), *Path('configs/optimization_v0_11').glob('*.yaml'),
               *Path('tests').glob('test_trajectory*.py'), Path('scripts/optimization_v11_cold_check.py'),
               Path('docs/optimization_v0_11/PLAN.md'), Path('uv.lock')]
    manifest = dict(registration=reg, scope=scope, protection=protection, algorithm=old['algorithm'],
        inventory=inventory, sources=file_identities({str(p):p for p in sources}),
        inputs=file_identities(paths), evidence=file_identities({str(p):p for p in evidence}),
        old_ledger_sha256=file_sha256(scope['old_ledger']), environment=runtime_environment(),
        test_inputs_used_for_selection=False, official_target_columns_read=False,
        holdout_consumed=True, timestamp_contract_status='ASSUMED')
    atomic_write_json(output/'manifest.json', manifest)
    append_ledger(scope, output)
    return manifest


def builder_for(manifest):
    paths = {k:v['path'] for k,v in manifest['inputs'].items()}
    a = manifest['algorithm']; op, burden, _ = _load_process_sources(paths, a['semantic'], a['features'])
    return ComponentFeatures(a, op, burden)


def load_fold(manifest, month):
    reg = manifest['registration']; source = Path(reg['source_v8'])
    record = manifest['inventory'][str(month)]
    entries, loaded = record['base_components'], {}
    cutoff = stamp(month); a = manifest['algorithm']
    for role, entry in entries.items():
        root = Path(entry['path'])
        if file_sha256(root/'bundle.json') != entry['bundle_sha256']:
            raise ContractError('reused bundle hash changed')
        model = DualTargetBaseline.load(root); md = model.bundle_metadata_; tr = md['training']
        if (model.parameters != a['baseline']['parameters'] or md['feature_config'] != a['features']
                or md['contract_digests'] != a['contract_digests'] or tr['sample_ids_sha256'] != record['sample_ids_sha256']
                or any(pd.Timestamp(tr[k]) != cutoff for k in ('fit_cutoff','history_cutoff','label_available_cutoff'))
                or tr.get('component',entry['component']) != entry['component'] or tr.get('variant','raw') != 'R2'
                or entry['role'] != role or entry['variant'] != 'R2'
                or entry['component'] != {'OR':'E09_PROCESS_CHANGE_E02','HR':'E04'}[role]):
            raise ContractError('reused training/feature identity differs')
        relevant = {k:v for k,v in md['code_identity'].items() if k.startswith(('src/bf_tap/features/', 'src/bf_tap/models/', 'configs/')) or k in (
            'src/bf_tap/optimization/history_stable.py','src/bf_tap/optimization/features.py',
            'src/bf_tap/optimization/process_change.py','src/bf_tap/optimization/final_lifecycle.py','src/bf_tap/availability.py')}
        verify_file_identities(relevant)
        history = model.load_history_snapshot().sort_values(['reference_time','sample_id'],kind='mergesort').reset_index(drop=True)
        if (stable_digest(history.sample_id.tolist()) != tr['sample_ids_sha256'] or len(history) != record['rows']
                or (history.reference_time >= cutoff).any() or (history.available_at > cutoff).any()):
            raise ContractError('reused history identity/availability differs')
        if 'schema_sha256' in entry and stable_digest(model.feature_schema_) != entry['schema_sha256']:
            raise ContractError('registered feature schema differs')
        loaded[role] = model, history
    rate_root = source/'models'/str(month)/'rate'; rate = RateModel.load(rate_root)
    if file_sha256(rate_root/'bundle.json') != record['rate_bundle_sha256'] or rate.metadata_['training']['sample_ids_sha256'] != record['sample_ids_sha256'] or pd.Timestamp(rate.metadata_['training']['fit_cutoff']) != cutoff:
        raise ContractError('rate identity differs')
    history = frame(rate_root/'history_snapshot.csv')
    for c in ('reference_time','tap_end_time','available_at'):
        history[c] = pd.to_datetime(history[c])
    if any(snapshot_identity(h) != snapshot_identity(history) for _,h in loaded.values()):
        raise ContractError('base and rate original histories differ')
    if rate.schema_ != loaded['OR'][0].feature_schema_:
        raise ContractError('rate and E09 schemas differ')
    oof = dated(source/'oof'/f'{month}.csv')
    if set(oof) & {'tap_iron','tap_time_len'} or set(oof.sample_id) & set(history.sample_id):
        raise ContractError('OOF labels/identity leakage')
    for c, value in zip(DATES[2:], (cutoff, history.reference_time.max(),history.available_at.max(),history.available_at.max())):
        if not (oof[c] == value).all():
            raise ContractError('OOF certificate differs from saved fold')
    return (entries,loaded,rate,history), oof


def extra_for(builder, samples):
    return build_trajectory_features(samples,builder.op,contract_value_columns=builder.a['features']['operation']['value_columns'])


def new_features(builder, samples, fold, *, training=False):
    entries, loaded, _, _ = fold
    h = loaded['OR'][1]; entry = entries['OR']
    base = (Context.X(builder,samples,stamp(pd.Timestamp(entry['cutoff']).month),entry['component'],'R2',h)
            if training else builder.X(samples,entry,h))
    if schema(base) != loaded['OR'][0].feature_schema_:
        raise ContractError('original E09 matrix schema differs')
    extra, audit = extra_for(builder,samples)
    result = append_trajectory(base,extra)
    return result, audit


def p0(output, manifest, builder):
    reg = manifest['registration']; source = Path(reg['source_v8']); counter={'attempted_target_fits':0}
    folds, oofs, checks, audits = {}, {}, [], {}
    # Old release reproduction only; no test feature distribution is audited or selected.
    import subprocess, sys
    release_checks = []
    for name, script in (('active_release','scripts/optimization_v8_cold_predict.py'),
                         ('fallback_release','scripts/optimization_v4_cold_predict.py')):
        release=load_yaml(reg[name]); root=Path(release['bundle']).parent
        config=root/('cold_data_repaired.yaml' if name=='active_release' else 'cold_data.yaml')
        dest=output/f'p0_{name}_predictions.csv'
        subprocess.run([sys.executable,script,'--bundle',release['bundle'],'--data-config',str(config),'--output',str(dest)],check=True)
        delta=equal(frame(dest),frame(root/'cold_predictions.csv'),PRED)
        release_checks.append(dict(release=name,rows=len(frame(dest)),max_delta=delta))
    with forbid_fit(counter):
        for month in reg['oof_months']:
            fold, oof = load_fold(manifest,month); folds[month], oofs[month] = fold,oof
            fresh = predict_inputs(oof[META],fold,builder,reg)
            diff = equal(fresh,oof,PRED+['pred_rate'])
            x,audit = new_features(builder,fold[1]['OR'][1][META],fold,training=True)
            signal = x[COLUMNS]; stats={c:dict(finite_rate=float(np.isfinite(signal[c]).mean()),
                finite_unique=int(signal[c].dropna().nunique()),minimum=float(signal[c].min()) if signal[c].notna().any() else None,
                maximum=float(signal[c].max()) if signal[c].notna().any() else None) for c in COLUMNS}
            audits[str(month)] = dict(rows=len(x),columns=stats,
                spans={c:audit[c].describe().to_dict() for c in audit if c!='sample_id'},
                old_columns_exact=True,old_schema_sha256=stable_digest(fold[1]['OR'][0].feature_schema_),
                new_schema_sha256=stable_digest(schema(x)))
            if not any(signal[c].notna().any() for c in COLUMNS if not c.endswith('valid_pair_count')):
                raise ContractError('data cannot form registered trajectories; block without substitution')
            checks.append(dict(month=month,OOF_rows=len(oof),OOF_max_delta=diff,training_rows=len(x),
                training_ids_sha256=stable_digest(fold[1]['OR'][1].sample_id.tolist())))
            builder.cache.clear()
            print(f'OPT24 P0 fold {month}: old OOF and feature audit PASS; zero fits',flush=True)
        combined = pd.concat(oofs.values(),ignore_index=True)
        for month, count in reg['origins'].items():
            cutoff=stamp(month)
            samples=combined.loc[(combined.reference_time>=cutoff)&(combined.reference_time<cutoff+pd.DateOffset(months=count)),META]
            parts=predict_inputs(samples,folds[month],builder,reg)
            equal(parts,dated(source/'predictions'/f'{month}_inputs.csv'),PRED+['pred_rate'])
            alpha=read_json(source/'corrections'/f'{month}.json')['alpha']
            v1=apply_correction(parts,alpha,reg['rate_floor'])
            delta=equal(v1,dated(source/'predictions'/f'{month}_V1.csv'),PRED)
            checks.append(dict(outer_month=month,rows=len(samples),V1_max_delta=delta))
            builder.cache.clear()
    atomic_write_json(output/'feature_audit.json',audits)
    atomic_write_json(output/'p0.json',dict(status='PASS',checks=checks,release_checks=release_checks,**counter,
        existing_registry_same_definition=False,old_feature_semantics_unchanged=True,test_inputs_used=False))
    return folds,combined


def candidate_inputs(builder, samples, fold, model):
    old = predict_inputs(samples,fold,builder,registration())
    x,_=new_features(builder,samples,fold)
    hr=fold[1]['HR']; x04=builder.X(samples,fold[0]['HR'],hr[1])
    t04=hr[0].predict_raw(x04)[PRED[1]].clip(lower=0).to_numpy()
    time=pd.Series(0.8*model.predict(x)+(1.-0.8)*t04,index=samples.sample_id)
    result=old.copy(); result[PRED[1]]=time.loc[result.sample_id].to_numpy()
    return result


def counts(output, budget, lad, *, final=False, **extra):
    events=output/'fit_count_events';events.mkdir(exist_ok=True)
    value=dict(development_time_CatBoost_attempted=budget.attempted,
        development_time_CatBoost_completed=budget.completed,development_time_LAD=lad,
        forbidden_fits=budget.forbidden,new_iron_E04_rate_q_residual_fits=0,
        final_time_CatBoost=0,final_time_LAD=0,challengers=0,**extra)
    atomic_write_json(events/f'{len(list(events.glob("*.json"))):03d}.json',value)
    if final:
        atomic_write_json(output/'fit_counts.json',value)


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    budget=ResidualFitBudget(8); lad=0; counter={'attempted_target_fits':0}
    try:
        manifest=freeze(output); reg=manifest['registration']; builder=builder_for(manifest)
        folds,combined=p0(output,manifest,builder)
        models, new_oofs, records = {}, [], []
        with budget:
            for month in reg['oof_months']:
                fold=folds[month]; history=fold[1]['OR'][1]
                x,_=new_features(builder,history[META],fold,training=True)
                # Persist the intent before calling fit. Failed attempts consume budget.
                dest=output/'models'/str(month); dest.mkdir(parents=True,exist_ok=False)
                training=dict(cutoff=str(stamp(month)),rows=len(history),sample_ids_sha256=stable_digest(history.sample_id.tolist()),
                    reference_max=str(history.reference_time.max()),available_max=str(history.available_at.max()),
                    original_OR=fold[0]['OR'],original_HR=fold[0]['HR'],feature_schema_sha256=stable_digest(schema(x)),
                    manifest_sha256=file_sha256(output/'manifest.json'),target='tap_time_len')
                atomic_write_json(dest/'fit_intent.json',training)
                model=TimeModel().fit(x,history.tap_time_len,budget)
                counts(output,budget,lad)
                model.save(dest/'time',training); restored=TimeModel.load(dest/'time')
                if not np.array_equal(model.predict(x),restored.predict(x)):
                    raise ContractError('time model persistence differs')
                models[month]=restored
                with forbid_fit(counter):
                    original=combined.loc[combined.fold_cutoff==stamp(month)].copy()
                    parts=candidate_inputs(builder,original[META],fold,restored)
                    original[PRED[1]]=parts.set_index('sample_id').loc[original.sample_id,PRED[1]].to_numpy()
                    target=output/'oof';target.mkdir(exist_ok=True);original.to_csv(target/f'{month}.csv',index=False)
                    new_oofs.append(original)
                records.append(dict(**training,model_bundle_sha256=file_sha256(dest/'time'/'bundle.json'),
                    OOF_sha256=file_sha256(target/f'{month}.csv')))
                atomic_write_json(dest/'fit_record.json',records[-1])
                builder.cache.clear()
                print(f'OPT24 new time fold {month}: {len(history)} rows; fits={budget.completed}/8',flush=True)
        atomic_write_json(output/'training_records.json',records)
        new_oof=pd.concat(new_oofs,ignore_index=True); predictions={}; paired={}; iron_exact=True
        with forbid_fit(counter):
            for month,n in reg['origins'].items():
                fold=folds[month]; history=fold[1]['OR'][1]; cutoff=stamp(month)
                labeled=new_oof.merge(history[['sample_id','tap_iron','tap_time_len','available_at']],on='sample_id',validate='one_to_one')
                if not (labeled.label_available_at==labeled.available_at).all():
                    raise ContractError('coefficient label availability differs')
                alpha,used=fit_time_coefficient(labeled,cutoff,reg['minimum_OOF_rows'],reg['rate_floor']);lad+=1
                counts(output,budget,lad)
                dest=output/'corrections';dest.mkdir(exist_ok=True)
                used.to_csv(dest/f'{month}_OOF.csv',index=False)
                atomic_write_json(dest/f'{month}.json',dict(alpha_T=alpha,cutoff=str(cutoff),rows=len(used),
                    OOF_ids_sha256=stable_digest(used.sample_id.tolist()),OOF_sha256=file_sha256(dest/f'{month}_OOF.csv'),
                    reference_max=str(used.reference_time.max()),available_max=str(used.label_available_at.max()),
                    source_history_sha256=fold[0]['OR']['history_snapshot_sha256']))
                samples=combined.loc[(combined.reference_time>=cutoff)&(combined.reference_time<cutoff+pd.DateOffset(months=n)),META]
                parts=candidate_inputs(builder,samples,fold,models[month])
                original=dated(Path(reg['source_v8'])/'predictions'/f'{month}_V1.csv')
                pred,fallback=predict(parts,original,alpha,reg['rate_floor'])
                reverse=candidate_inputs(builder,samples.iloc[::-1],fold,models[month])
                rev,_=predict(reverse,original.iloc[::-1],alpha,reg['rate_floor'])
                equal(pred,rev,PRED)
                equal(pred,original,[PRED[0]],tolerance=0.)
                dest=output/'predictions';dest.mkdir(exist_ok=True)
                parts.to_csv(dest/f'{month}_inputs.csv',index=False);pred.to_csv(dest/f'{month}_V5.csv',index=False)
                predictions[month]=pred
                delta=pred.set_index('sample_id').loc[parts.sample_id,PRED[1]].to_numpy()-parts[PRED[1]].to_numpy()
                paired[str(month)]=dict(alpha_T=alpha,unusable_rate_rows=fallback,
                    correction_delta_quantiles=dict(zip(('p05','median','p95'),np.quantile(delta,[.05,.5,.95]).tolist())),
                    prediction_delta_vs_V1_quantiles=dict(zip(('p05','median','p95'),np.quantile(
                        pred.set_index('sample_id').sort_index()[PRED[1]]-original.set_index('sample_id').sort_index()[PRED[1]], [.05,.5,.95]).tolist())))
                builder.cache.clear()
                print(f'OPT24 outer {month}: causal LAD n={len(used)}, alpha={alpha:.9f}; iron exact, reversal PASS',flush=True)
        if budget.completed!=8 or lad!=6 or budget.forbidden or counter['attempted_target_fits']:
            raise ContractError('OPT24 budget/inference violation')
        atomic_write_json(output/'predictions_complete.json',dict(predictions=file_identities({str(p):p for p in (output/'predictions').glob('*.csv')}),
            completed_time_fits=8,completed_LAD_fits=6,inference_fit_attempts=0,holdout_consumed=True))
        # Separate audit process reconstructs predictions and provenance, never fits LAD.
        import subprocess, sys
        subprocess.run([sys.executable,'scripts/optimization_v11_cold_check.py','--source',str(output)],check=True)
        cold=read_json(output/'cold_validation.json')
        verify_manifest(manifest)
        append_ledger({**manifest['scope'],'authorization':'OPT24_scoring_after_frozen_predictions_and_cold_validation'},output)
        def provider(unit,cutoff,metadata,baselines):
            p=predictions[cutoff.month]; p=p.loc[p.sample_id.isin(metadata.sample_id)]
            equal(p,baselines['V1'],[PRED[0]],tolerance=0.)
            return {CANDIDATE:p}
        metrics,summary,errors,_=score(output,reg,provider)
        gate=acceptance(metrics,summary,reg,iron_exact,cold['status']=='PASS')
        atomic_write_json(output/'acceptance.json',gate)
        atomic_write_json(output/'week_bootstrap.json',week_intervals(errors,CANDIDATE,'V1',**reg['bootstrap']))
        atomic_write_json(output/'paired_changes.json',paired)
        diagnostic(output,errors)
        verify_manifest(manifest)
        counts(output,budget,lad,final=True,inference_fit_attempts=counter['attempted_target_fits'])
        atomic_write_json(output/'final_status.json',dict(G0='PASS_OPT24_DEVELOPMENT_AND_COLD',
            G1=gate['status'],selected=gate['selected'],holdout_consumed=True,active_release_unchanged=True,
            final_stage='AUTHORIZED_IF_FULL_TEST_SUITE_PASSES' if gate['passed'] else 'NOT_AUTHORIZED_QUALITY_FAIL'))
        print(__import__('json').dumps(gate,indent=2),flush=True)
    except Exception as exc:
        counts(output,budget,lad,final=not (output/'fit_counts.json').exists(),inference_fit_attempts=counter['attempted_target_fits'])
        atomic_write_json(output/'failure.json',dict(error=str(exc),holdout_consumed=True))
        raise


def diagnostic(output,errors):
    errors=errors.copy(); errors['month']=pd.to_datetime(errors.reference_time).dt.strftime('%Y-%m')
    rows=[]
    for keys,part in errors.groupby(['candidate','unit','month','spout_no']):
        for t in ('tap_iron','tap_time_len'):
            denominator=float(part[t].sum()); numerator=float(part['abs_error_'+t].sum())
            rows.append(dict(zip(('candidate','unit','month','spout_no'),keys))|dict(target=t,rows=len(part),
                absolute_error_sum=numerator,target_sum=denominator,wmape=numerator/denominator if denominator else None,
                signed_error_sum=float((part['pred_'+t]-part[t]).sum()),signed_bias=float((part['pred_'+t]-part[t]).mean())))
    atomic_write_json(output/'target_month_spout_diagnostics.json',rows)
    # Zero-fit base-time diagnostic, never supplied as a selectable score candidate.
    paired=[]
    for unit,part in errors.loc[errors.candidate==CANDIDATE].groupby('unit'):
        month=int(part.origin.iloc[0][-2:]); base=frame(output/'predictions'/f'{month}_inputs.csv').set_index('sample_id')
        b=base.loc[part.sample_id,PRED[1]].to_numpy(); y=part.tap_time_len.to_numpy()
        paired.append(dict(unit=unit,base_time_error_sum=float(np.abs(b-y).sum()),
            corrected_time_error_sum=float(part.abs_error_tap_time_len.sum()),time_denominator=float(y.sum())))
    atomic_write_json(output/'base_time_diagnostic.json',paired)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
