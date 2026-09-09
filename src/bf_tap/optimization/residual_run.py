"""OPT-23: twelve causal residual fits, then fixed V1-relative scoring."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import (atomic_write_json, file_identities, file_sha256,
                        stable_digest, verify_file_identities)
from ..config import load_yaml
from ..exceptions import ContractError
from ..offline import _load_process_sources
from .component_export import META, PRED, ComponentFeatures, read_json, units, forbid_fit
from .dual_ratio_common import frame, score, week_intervals, verify_manifest
from .dual_ratio_run import history_frame, assert_equal
from .final_lifecycle import algorithm
from .inverse_rate_model import InverseRateModel
from .refresh_factorial import stamp
from .residual_stack import (CANDIDATE, CONTEXT, RATIOS, CERT, registration,
                             feature_frame, select_training, ResidualModel,
                             ResidualFitBudget, predict, acceptance)

DATES = ['reference_time','label_available_at','fold_cutoff', 'train_reference_max',
         'train_available_max','history_available_max', *CERT]


def dated(path):
    result = frame(path)
    for column in DATES:
        result[column] = pd.to_datetime(result[column])
    return result


def freeze(output):
    reg = registration()
    scope = load_yaml('configs/optimization_v0_10/access_scope.yaml')
    load_yaml(scope['protection_contract'])
    source = Path(reg['source_v9'])
    prior = read_json(source/'manifest.json')
    verify_manifest(prior)
    parent = Path(reg['source_v8'])
    # v0.8 uses its own manifest layout; verify its complete frozen identities.
    old = read_json(parent/'manifest.json')
    for key in ('sources','inputs','evidence'):
        verify_file_identities(old[key])
    cold = Path(reg['source_v9_cold'])/'validation.json'
    if read_json(cold)['status'] != 'PASS' or read_json(cold)['source_manifest_sha256'] != file_sha256(source/'manifest.json'):
        raise ContractError('verified q OOF cold evidence required')
    if read_json(source/'final_status.json')['G0'] != 'PASS_OPT22':
        raise ContractError('completed source q run required')
    completion = Path('local/reports/optimization-v0.9-completion-r1.json')
    for path, digest in read_json(completion)['evidence_sha256'].items():
        if file_sha256(path) != digest:
            raise ContractError('v0.9 completed evidence changed')
    evidence = [completion, cold, source/'manifest.json', source/'final_status.json',
                source/'training_records.json', parent/'manifest.json', Path(scope['protection_contract']),
                Path('local/ledgers/optimization-v0.9-development.jsonl')]
    for month in reg['oof_months']:
        evidence += [source/'oof'/f'{month}.csv', parent/'oof'/f'{month}.csv',
                     parent/'models'/str(month)/'fit_record.json']
        evidence += list((source/'models'/str(month)).glob('*'))
        evidence += list((parent/'models'/str(month)/'rate').glob('*'))
    for month in reg['origins']:
        evidence += [source/'predictions'/f'{month}_q.csv', parent/'predictions'/f'{month}_inputs.csv',
                     parent/'predictions'/f'{month}_V1.csv']
    for unit,*_ in units(reg):
        evidence += [parent/'units'/unit/f'{c}_errors.csv' for c in ('U0','U1')]
    for name in ('active_release','fallback_release'):
        release = load_yaml(reg[name])
        if file_sha256(release['test_a_zip']) != release['test_a_zip_sha256']:
            raise ContractError('incumbent archive changed')
        evidence += [Path(reg[name]), Path(release['test_a_zip'])]
    desktop = Path('/mnt/c/Users/lqh22/Desktop/Luqhhh_bf_tap_predict_prelim.zip')
    if desktop.is_file():
        evidence.append(desktop)
    paths = {k:v['path'] for k,v in old['inputs'].items()
             if k in ('operation_hourly','burden_change','data_dictionary')}
    sources = [*Path('src/bf_tap').rglob('*.py'), *Path('configs/optimization_v0_10').glob('*.yaml'),
               Path('docs/optimization_v0_10/PLAN.md'),Path('scripts/optimization_v10_cold_check.py')]
    manifest = dict(registration=reg, scope=scope, purpose='OPT23_causal_R2_residual_training',
        evidence=file_identities({str(p):p for p in evidence}),
        sources=file_identities({str(p):p for p in sources}), inputs=file_identities(paths),
        old_ledger_sha256=file_sha256(scope['old_ledger']),test_inputs_used_for_selection=False)
    atomic_write_json(output/'manifest.json',manifest)
    ledger = Path(scope['new_ledger']); ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('a+') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX); f.seek(0); prior_lines=f.read().splitlines()
        f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),authorization=scope['authorization'],
            purpose=manifest['purpose'],manifest_sha256=file_sha256(output/'manifest.json'),
            previous_sha256=stable_digest(prior_lines[-1]) if prior_lines else None,holdout_consumed=True))+'\n')
    return manifest


def builder_for(manifest):
    a = algorithm()
    paths = {k:v['path'] for k,v in manifest['inputs'].items()}
    op, burden, _ = _load_process_sources(paths, a['semantic'], a['features'])
    return ComponentFeatures(a, op, burden)


def load_fold(reg, month):
    root = Path(reg['source_v9'])/'models'/str(month)
    record = read_json(root/'fit_record.json')
    if file_sha256(root/'bundle.json') != record['q_bundle_sha256']:
        raise ContractError('q bundle differs from completed fit')
    model = InverseRateModel.load(root)
    history = history_frame(root/'history_snapshot.csv')
    cutoff = stamp(month)
    md = model.metadata_
    if (pd.Timestamp(md['training']['fit_cutoff']) != cutoff or
        stable_digest(history.sample_id.tolist()) != md['training']['sample_ids_sha256'] or
        (history.reference_time >= cutoff).any() or (history.available_at > cutoff).any()):
        raise ContractError('q saved history provenance differs')
    entry = read_json(Path(reg['source_v8'])/'models'/str(month)/'fit_record.json')['base_components']['OR']
    oof_path = Path(reg['source_v9'])/'oof'/f'{month}.csv'
    if file_sha256(oof_path) != record['OOF_sha256']:
        raise ContractError('certified q OOF bytes changed')
    oof = dated(oof_path)
    if set(oof) & {'tap_iron','tap_time_len'}:
        raise ContractError('prediction archives must be label-free')
    if set(oof.sample_id) & set(history.sample_id):
        raise ContractError('OOF sample in own model training history')
    for c,value in zip(CERT,(cutoff,history.reference_time.max(),history.available_at.max(),history.available_at.max())):
        if not (oof[c] == value).all():
            raise ContractError('q temporal certificate differs from actual training history')
    assert_equal(oof, frame(Path(reg['source_v8'])/'oof'/f'{month}.csv'), PRED+['pred_rate'])
    return model, history, entry, oof


def context_for(builder, samples, entry, history):
    x = builder.X(samples[META], entry, history)
    result = x[CONTEXT].reset_index(drop=True).copy()
    result.insert(0, 'sample_id', samples.sample_id.to_numpy())
    return result


def prepare_oof(reg, builder):
    folds, contexts, histories = [], [], {}
    for month in reg['oof_months']:
        model, history, entry, oof = load_fold(reg, month)
        contexts.append(context_for(builder, oof[META], entry, history))
        histories[month] = history
        folds.append(oof)
        builder.cache.clear()
    return pd.concat(folds, ignore_index=True), pd.concat(contexts, ignore_index=True), histories


def training_at(combined, history, cutoff, minimum):
    labeled = combined.merge(history[['sample_id','tap_iron','tap_time_len']], on='sample_id', validate='one_to_one')
    available = history.set_index('sample_id').available_at
    if not np.array_equal(labeled.label_available_at.astype(str), available.loc[labeled.sample_id].astype(str)):
        raise ContractError('OOF labels do not match eligible history availability')
    return select_training(labeled, cutoff, minimum)


def outer_inputs(reg, month, combined, builder, history):
    cutoff=stamp(month)
    samples=combined.loc[(combined.reference_time >= cutoff) &
        (combined.reference_time < cutoff+pd.DateOffset(months=reg['origins'][month])), META]
    source=Path(reg['source_v8'])
    parts=frame(source/'predictions'/f'{month}_inputs.csv')[['sample_id',*PRED,'pred_rate']]
    q=frame(Path(reg['source_v9'])/'predictions'/f'{month}_q.csv')
    if set(parts.sample_id) != set(samples.sample_id) or set(q.sample_id) != set(samples.sample_id):
        raise ContractError('outer archived prediction identity differs')
    parts=parts.merge(q,on='sample_id',validate='one_to_one')[['sample_id',*PRED,*RATIOS]]
    entry=read_json(source/'models'/str(month)/'fit_record.json')['base_components']['OR']
    context=context_for(builder,samples,entry,history)
    return samples,parts,context


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    reg=registration(); budget=ResidualFitBudget(reg['budget']['development_residual_fits'])
    try:
        manifest=freeze(output); builder=builder_for(manifest)
        counter={'attempted_target_fits':0}
        with forbid_fit(counter):
            combined, context, histories=prepare_oof(reg,builder)
        context.to_csv(output/'OOF_context.csv',index=False)
        predictions={}; records=[]
        with budget:
            for month in reg['origins']:
                cutoff=stamp(month); history=histories[month]
                selected=training_at(combined,history,cutoff,reg['residual']['minimum_OOF_rows'])
                train_parts=selected[['sample_id',*PRED,*RATIOS]]
                train_context=context.loc[context.sample_id.isin(selected.sample_id)]
                x=feature_frame(train_parts,train_context)
                dest=output/'models'/str(month); dest.mkdir(parents=True)
                selected.to_csv(dest/'training_OOF.csv',index=False)
                train_context.to_csv(dest/'training_context.csv',index=False)
                training=dict(cutoff=str(cutoff),rows=len(selected),OOF_ids_sha256=stable_digest(selected.sample_id.tolist()),
                    OOF_reference_max=str(selected.reference_time.max()),OOF_label_available_max=str(selected.label_available_at.max()),
                    source_history_sha256=file_sha256(Path(reg['source_v9'])/'models'/str(month)/'history_snapshot.csv'),
                    OOF_sha256=file_sha256(dest/'training_OOF.csv'),context_sha256=file_sha256(dest/'training_context.csv'),
                    manifest_sha256=file_sha256(output/'manifest.json'))
                models={}
                for target,column in zip(('tap_iron','tap_time_len'),PRED):
                    model=ResidualModel().fit(x,selected[target],selected[column],target,budget)
                    model.save(dest/target,training)
                    loaded=ResidualModel.load(dest/target)
                    if not np.array_equal(model.predict_residual(x),loaded.predict_residual(x)):
                        raise ContractError('residual persistence changed prediction')
                    models[target]=loaded
                    print(f'OPT23 origin {month} {target}: {len(selected)} causal OOF rows; fits={budget.completed}/12',flush=True)
                with forbid_fit(counter):
                    samples,parts,ctx=outer_inputs(reg,month,combined,builder,history)
                    if set(samples.sample_id) & set(selected.sample_id):
                        raise ContractError('outer evaluation overlaps meta training')
                    pred=predict(models,parts,ctx)
                    reverse_context=context_for(builder,samples.iloc[::-1],
                        load_fold(reg,month)[2],history)
                    rev=predict(models,parts.iloc[::-1],reverse_context)
                    assert_equal(pred,rev,PRED)
                    d=output/'predictions';d.mkdir(exist_ok=True)
                    parts.to_csv(d/f'{month}_inputs.csv',index=False)
                    ctx.to_csv(d/f'{month}_context.csv',index=False)
                    pred.to_csv(d/f'{month}_V4.csv',index=False)
                    predictions[month]=pred
                    record=dict(month=month,training=training,rows=len(pred),reversal_exact=True,
                        bundle_sha256={t:file_sha256(dest/t/'bundle.json') for t in models},
                        prediction_sha256=file_sha256(d/f'{month}_V4.csv'))
                    records.append(record)
                builder.cache.clear()
            if budget.completed != 12 or budget.forbidden or counter['attempted_target_fits']:
                raise ContractError('OPT23 budget/inference violation')
            atomic_write_json(output/'predictions_complete.json',dict(**budget.counts(),**counter))
        def provider(unit,cutoff,metadata,baselines):
            p=predictions[cutoff.month]
            return {CANDIDATE:p.loc[p.sample_id.isin(metadata.sample_id)]}
        metrics,summary,errors,_=score(output,reg,provider)
        gate=acceptance(metrics,summary,reg)
        atomic_write_json(output/'acceptance.json',gate)
        atomic_write_json(output/'week_bootstrap.json',week_intervals(errors,CANDIDATE,'V1',**reg['bootstrap']))
        atomic_write_json(output/'training_records.json',records)
        verify_manifest(manifest)
        atomic_write_json(output/'final_status.json',dict(G0='PASS_OPT23_EXECUTION',G1=gate['status'],selected=gate['selected'],
            **budget.counts(),inference_fit_attempts=counter['attempted_target_fits'],new_base_rate_q_fits=0,new_LAD_fits=0,
            units=20,challenger_generated=False,final_fits=0))
        print(gate,flush=True)
    except Exception as exc:
        atomic_write_json(output/'failure.json',dict(error=str(exc),**budget.counts()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
