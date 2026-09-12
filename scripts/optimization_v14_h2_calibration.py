"""OPT-30/31: original-model forecast banks, bounded scalar fits, frozen scoring."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from bf_tap.artifacts import (atomic_write_json, file_identities, file_sha256, stable_digest,
    verify_file_identities, runtime_environment, validate_inference_source_contract)
from bf_tap.config import load_yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization.component_export import META, PRED, read_json, units
from bf_tap.optimization.dual_ratio_common import frame, score, week_intervals
from bf_tap.optimization.structural import INPUT, apply_correction
from bf_tap.optimization.structural_run import append_ledger, predict_inputs
from bf_tap.optimization.trajectory_run import load_fold, builder_for, equal
from bf_tap.optimization.v13_common import verify_receipt, zero_fit
from bf_tap.optimization.horizon_calibration import (CANDIDATE, CONTROL, DATES, month_start,
    calendar_window, verify_bank, verify_pair, select_matched, predict_time, ScalarBudget,
    no_tree_or_dual_fit, acceptance, read_bank)


def registration():
    reg = load_yaml('configs/optimization_v0_14/experiment.yaml')
    if (reg['candidate'] != CANDIDATE or reg['diagnostic_control'] != CONTROL or
        reg['bank_model_months'] != list(range(4,11)) or reg['oof_months'] != list(range(4,12)) or
        reg['budget'] != {'CatBoost':0,'candidate_time_LAD':6,'control_time_LAD':6,'iron_LAD':0,'conditional_final_time_LAD':1} or
        reg['rate_floor'] != 1e-6 or reg['minimum_OOF_rows'] != 100):
        raise ContractError('fixed OPT30 candidate/budget differs')
    return reg


def event(root, manifest, purpose, **extra):
    scope = manifest['scope']
    append_ledger({**scope,'authorization':scope['authorization']+':'+purpose},root)
    events = root/'access_events'; events.mkdir(exist_ok=True)
    atomic_write_json(events/f'{len(list(events.glob("*.json"))):03d}.json',
                      {'purpose':purpose,'at':datetime.now(timezone.utc).isoformat(),**extra})


def freeze(root):
    reg = registration(); scope = load_yaml('configs/optimization_v0_14/access_scope.yaml')
    protection = load_yaml(scope['protection_contract'])
    if not scope['holdout_consumed'] or not read_json('EVIDENCE_STATUS.json')['holdout_consumed']:
        raise ContractError('retrospective consumed-label scope required')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ContractError('clean registered working tree required')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip() != reg['branch']:
        raise ContractError('registered v14 branch required')
    subprocess.run(['git','merge-base','--is-ancestor',reg['base_commit'],'HEAD'],check=True)
    old = read_json(reg['inventory_manifest']); source = Path(reg['source_v8'])
    original = read_json(source/'manifest.json')
    for category in ('sources','inputs','evidence'): verify_file_identities(original[category])
    if read_json(source/'final_status.json')['G0'] != 'PASS_OPT20':
        raise ContractError('completed V1 source required')
    receipts = [verify_receipt(p) for p in reg['prior_receipts']]
    evidence = [Path(reg['inventory_manifest']),source/'manifest.json',source/'training_records.json',
                source/'final_status.json',source/'summary.json',Path(scope['protection_contract']),
                Path(reg['source_v8_cold'])/'validation.json',*map(Path,reg['prior_receipts'])]
    evidence += [Path(p) for r in receipts for p in r['evidence_sha256']]
    inventory = old['inventory']; sources_registry = []
    for month in reg['oof_months']:
        record_path = source/'models'/str(month)/'fit_record.json'
        if read_json(record_path) != inventory[str(month)]:
            raise ContractError('original monthly training registry differs')
        evidence += [record_path,source/'oof'/f'{month}.csv']
        for entry in inventory[str(month)]['base_components'].values():
            folder = Path(entry['path']); md = read_json(folder/'bundle.json')
            if file_sha256(folder/'bundle.json') != entry['bundle_sha256']:
                raise ContractError('BLOCKED_MISSING_SOURCE: restore original base bundle, no fit')
            evidence += [p for p in folder.rglob('*') if p.is_file()]
            sources_registry.append(md['inference_source_contract'])
        evidence += [p for p in (source/'models'/str(month)/'rate').rglob('*') if p.is_file()]
    for month in reg['origins']:
        evidence += [source/'predictions'/f'{month}_inputs.csv',source/'predictions'/f'{month}_V1.csv',source/'corrections'/f'{month}.json']
    for unit,*_ in units(reg): evidence += [source/'units'/unit/f'{k}_errors.csv' for k in ('U0','U1')]
    for key in ('active_release','fallback_release'):
        release = load_yaml(reg[key])
        if file_sha256(release['test_a_zip']) != release['test_a_zip_sha256']:
            raise ContractError('original published ZIP differs')
        evidence += [Path(reg[key]),Path(release['test_a_zip'])]
        evidence += [p for p in Path(release['bundle']).rglob('*') if p.is_file()]
    inputs = old['inputs']
    verify_file_identities(inputs)
    for contract in sources_registry:
        validate_inference_source_contract(contract,inputs,semantic_contract_sha256=contract['semantic_contract_sha256'])
    files = [*Path('src/bf_tap').rglob('*.py'),*Path('configs/optimization_v0_14').glob('*.yaml'),
             *Path('scripts').glob('optimization_v14*.py'),Path('tests/test_horizon_calibration.py'),
             Path('docs/optimization_v0_14/PLAN.md'),Path('uv.lock')]
    root.mkdir(parents=True,exist_ok=False);root.chmod(0o700)
    manifest = {'registration':reg,'scope':scope,'protection':protection,'algorithm':original['algorithm'],
                'inventory':inventory,'sources':file_identities({str(p):p for p in files}),
                'inputs':inputs,'evidence':file_identities({str(p):p for p in evidence}),
                'old_ledgers':file_identities({p:p for p in scope['old_ledgers']}),
                'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'environment':runtime_environment(),'test_inputs_for_selection':False,
                'official_target_column_reads':False,'holdout_consumed':True,
                'timestamp_contract_status':'ASSUMED','existing_same_definition_registry':False,
                'source_contracts':sources_registry}
    atomic_write_json(root/'manifest.json',manifest)
    event(root,manifest,'REGISTERED_BEFORE_FORECAST_BANK_AND_CALIBRATION_LABELS')
    return manifest


def verify(manifest):
    for key in ('sources','inputs','evidence','old_ledgers'): verify_file_identities(manifest[key])
    for contract in manifest['source_contracts']:
        validate_inference_source_contract(contract,manifest['inputs'],semantic_contract_sha256=contract['semantic_contract_sha256'])


def original_fold(manifest, month):
    fold, oof = load_fold(manifest,month)
    for name in ('reference_time','tap_end_time','available_at'):
        for _,history in fold[1].values():
            history[name] = pd.to_datetime(history[name],utc=True).dt.tz_convert('Asia/Shanghai')
        fold[3][name] = pd.to_datetime(fold[3][name],utc=True).dt.tz_convert('Asia/Shanghai')
    for name in DATES:
        if name in oof: oof[name] = pd.to_datetime(oof[name],utc=True).dt.tz_convert('Asia/Shanghai')
    return fold,oof


def forecast_bank(parts, metadata, manifest, month, horizon):
    record = manifest['inventory'][str(month)]; cutoff = month_start(month)
    start,end = calendar_window(cutoff,horizon)
    result = metadata[META+['label_available_at']].merge(parts,on='sample_id',validate='one_to_one')
    result['fold_cutoff'] = cutoff
    for k,v in {'train_reference_max':record['reference_max'],'train_available_max':record['label_available_max'],
                'history_available_max':record['history_available_max']}.items(): result[k] = pd.Timestamp(v)
    result['evaluation_month_start'],result['evaluation_month_end'],result['horizon'] = start,end,horizon
    result['fold_identity_sha256'] = stable_digest(record)
    result['train_ids_sha256'] = record['sample_ids_sha256']
    result['E09_bundle_sha256'] = record['base_components']['OR']['bundle_sha256']
    result['E04_bundle_sha256'] = record['base_components']['HR']['bundle_sha256']
    result['rate_bundle_sha256'] = record['rate_bundle_sha256']
    result['history_sha256'] = record['base_components']['OR']['history_snapshot_sha256']
    schema = read_json(Path(record['base_components']['OR']['path'])/'bundle.json')['feature_schema']
    result['schema_sha256'] = stable_digest(schema)
    result['source_contract_sha256'] = stable_digest(manifest['source_contracts'][0])
    return result


def p0(root,manifest):
    reg = manifest['registration'];builder = builder_for(manifest)
    banks={1:[],2:[]}; folds={}; checks=[]
    with zero_fit() as count:
        for month in reg['oof_months']:
            folds[month],_ = original_fold(manifest,month)
        for month in reg['bank_model_months']:
            _,meta = original_fold(manifest,month+1)
            for horizon,model_month in ((2,month),(1,month+1)):
                parts = predict_inputs(meta[META],folds[model_month],builder,reg)
                bank = forecast_bank(parts,meta,manifest,model_month,horizon)
                verify_bank(bank,horizon,manifest['inventory']);banks[horizon].append(bank)
                if horizon == 1:
                    delta = equal(parts,meta,PRED+['pred_rate'],reg['prediction_tolerance'])
                elif month >= 6:
                    old = frame(Path(reg['source_v8'])/'predictions'/f'{month}_inputs.csv')
                    delta = equal(parts,old.loc[old.sample_id.isin(parts.sample_id)],PRED+['pred_rate'],reg['prediction_tolerance'])
                else: delta = None
                checks.append({'model_month':model_month,'horizon':horizon,'rows':len(meta),'existing_prediction_max_delta':delta})
                builder.cache.clear()
            print(f'OPT30 bank {month}->{month+1}: genuine H2 + matched H1 PASS; fits=0',flush=True)
        # April H1 is not a matched control, but its original OOF identity is checked.
        _,april = original_fold(manifest,4)
        delta = equal(predict_inputs(april[META],folds[4],builder,reg),april,PRED+['pred_rate'],reg['prediction_tolerance'])
        checks.append({'model_month':4,'horizon':1,'rows':len(april),'existing_prediction_max_delta':delta})
        h2,h1 = pd.concat(banks[2],ignore_index=True),pd.concat(banks[1],ignore_index=True)
        verify_bank(h2,2,manifest['inventory']);verify_bank(h1,1,manifest['inventory']);verify_pair(h2,h1)
        folder=root/'paired_bank';folder.mkdir()
        h2.to_csv(folder/'H2.csv',index=False);h1.to_csv(folder/'H1_same_calendar.csv',index=False)
        # Outer metadata is prediction-only and is fixed without opening scoring targets.
        all_meta = pd.concat([original_fold(manifest,m)[1] for m in range(6,12)],ignore_index=True)
        outer = root/'outer_inputs';outer.mkdir()
        for month,n in reg['origins'].items():
            c = month_start(month)
            samples = all_meta.loc[(all_meta.reference_time>=c)&(all_meta.reference_time<c+pd.DateOffset(months=n)),META]
            parts = predict_inputs(samples,folds[month],builder,reg)
            source=Path(reg['source_v8']); old_inputs=frame(source/'predictions'/f'{month}_inputs.csv')
            old_v1=frame(source/'predictions'/f'{month}_V1.csv')
            diff=equal(parts,old_inputs,PRED+['pred_rate'],reg['prediction_tolerance'])
            v1fresh=apply_correction(parts,read_json(source/'corrections'/f'{month}.json')['alpha'])
            v1diff=equal(v1fresh,old_v1,PRED,reg['prediction_tolerance'])
            # Preserve original archived arrays after verifying genuine replay.
            samples.to_csv(outer/f'{month}_metadata.csv',index=False)
            old_inputs.to_csv(outer/f'{month}_inputs.csv',index=False);old_v1.to_csv(outer/f'{month}_V1.csv',index=False)
            checks.append({'outer_month':month,'rows':len(samples),'R2_rate_delta':diff,'V1_delta':v1diff})
            builder.cache.clear()
            print(f'OPT30 original outer {month}: V1/R2 replay PASS; fits=0',flush=True)
    bank_files=[p for p in folder.glob('*.csv')]+[p for p in outer.glob('*.csv')]
    atomic_write_json(root/'OOF_provenance.json',{'folds':manifest['inventory'],'checks':checks,
        'forecast_bank_identities':file_identities({str(p):p for p in bank_files}),
        'H1_H2_same_calendar_identity':True,'prediction_bank_contains_targets':False,
        'genuine_corresponding_fold_replayed':True,'holdout_consumed':True})
    atomic_write_json(root/'p0.json',{'status':'PASS','rows_per_matched_bank':len(h2),'checks':checks,**count})
    event(root,manifest,'FORECAST_BANK_COMPLETE_BEFORE_CALIBRATION_LABELS',rows=len(h2))
    return h2,h1


def fit_and_predict(root,manifest,h2,h1):
    reg=manifest['registration'];budget=ScalarBudget(root/'coefficients')
    provenance=[]; predictions=root/'predictions';predictions.mkdir()
    with no_tree_or_dual_fit() as count:
        for month in reg['origins']:
            c=month_start(month)
            event(root,manifest,'READ_CERTIFIED_PRIOR_CALIBRATION_LABELS',outer_cutoff=str(c),latest_allowed_month=month-1)
            fold,_=original_fold(manifest,month)
            s2,s1=select_matched(h2,h1,fold[3],c,reg['minimum_OOF_rows'])
            coefficients=[]
            for role,selected in ((CANDIDATE,s2),(CONTROL,s1)):
                selected.to_csv(root/'coefficients'/f'{role}-{month}.OOF.csv',index=False)
                saved=budget.fit(role,month,selected);coefficients.append(saved)
            base=frame(root/'outer_inputs'/f'{month}_inputs.csv')
            original=frame(root/'outer_inputs'/f'{month}_V1.csv')
            original.to_csv(predictions/f'{month}_V1.csv',index=False)
            for saved in coefficients:
                p,fallback=predict_time(base,original,saved['beta'])
                p.to_csv(predictions/f'{month}_{saved["role"]}.csv',index=False)
                provenance.append({'role':saved['role'],'outer_cutoff':str(c),'rows':len(s2),
                    'ids_sha256':saved['certificate']['ids_sha256'],'label_available_max':str(s2.label_available_at.max()),
                    'reference_max':str(s2.reference_time.max()),'allowed_calendar_months':sorted(s2.reference_time.dt.month.unique().tolist()),
                    'beta':saved['beta'],'rate_fallback':fallback,'label_source_history_sha256':
                    file_sha256(Path(manifest['inventory'][str(month)]['base_components']['OR']['path'])/'history_snapshot.csv')})
            print(f'OPT30 origin {month}: matched {len(s2)} rows; beta H2={coefficients[0]["beta"]:.12g}, H1={coefficients[1]["beta"]:.12g}',flush=True)
    if count['attempted_target_fits'] or count['attempted_dual_calibration_fits']:
        raise ContractError('forbidden fit attempted')
    atomic_write_json(root/'coefficient_provenance.json',{'origins':provenance,'fitting_target':'tap_time_len',
        'additional_LAD_weights':False,'rolling_past_outer_labels_may_be_used_by_later_origins':True})
    atomic_write_json(root/'fit_counts.json',{'CatBoost':0,'iron_LAD':0,'development_time_LAD':budget.counts(),
        'final_time_LAD':0,'new_challengers':0,**count})
    frozen=[p for p in predictions.glob('*.csv')]+[p for p in (root/'coefficients').glob('*')]
    atomic_write_json(root/'predictions_complete.json',{'status':'FROZEN_BEFORE_SCORING_LABEL_ACCESS',
        'at':datetime.now(timezone.utc).isoformat(),'identities':file_identities({str(p):p for p in frozen})})
    event(root,manifest,'ALL_OUTER_PREDICTIONS_FROZEN_BEFORE_SCORING_LABEL_ACCESS')


def score_saved(root,manifest):
    reg=manifest['registration']
    verify_file_identities(read_json(root/'predictions_complete.json')['identities'])
    if not read_json(root/'cold_validation.json')['engineering_valid']:
        raise ContractError('independent cold audit required before scoring')
    event(root,manifest,'READ_ARCHIVED_CONSUMED_OUTER_SCORING_LABELS_AFTER_PREDICTIONS_FROZEN')
    def provider(unit,cutoff,samples,parts):
        result={}
        for role in (CANDIDATE,CONTROL):
            p=frame(root/'predictions'/f'{cutoff.month}_{role}.csv').set_index('sample_id').loc[samples.sample_id].reset_index()
            reference=parts['V1'].set_index('sample_id').loc[p.sample_id]
            if not np.array_equal(p[PRED[0]],reference[PRED[0]]):
                raise ContractError('candidate iron differs from V1')
            result[role]=p
        return result
    with zero_fit() as counts:
        metrics,summary,errors,_=score(root,reg,provider)
        old_summary=read_json(Path(reg['source_v8'])/'summary.json')
        if abs(summary['V1']['J']-old_summary['V1']['J'])>reg['metric_tolerance']:
            raise ContractError('legacy full-precision J differs')
        for unit,v in metrics.items():
            for target in ('iron','time'):
                if v['candidates']['V1']['overall'][target]['actual_sum'] != v['candidates'][CANDIDATE]['overall'][target]['actual_sum']:
                    raise ContractError('target denominator differs')
        intervals={}
        for role in (CANDIDATE,CONTROL):
            intervals[role]=week_intervals(errors,role,'V1',**reg['bootstrap'])
            intervals[role]['invalid_repetitions']=reg['bootstrap']['repetitions']-intervals[role]['valid_repetitions']
        atomic_write_json(root/'bootstrap.json',intervals)
        result=acceptance(metrics,summary,reg,True,True)
        atomic_write_json(root/'acceptance.json',result)
        records=[]
        for (candidate,unit,month,spout),g in errors.assign(calendar_month=errors.reference_time.dt.strftime('%Y-%m')).groupby(['candidate','unit','calendar_month','spout_no']):
            for target in ('tap_iron','tap_time_len'):
                residual=g['pred_'+target]-g[target]
                records.append({'candidate':candidate,'unit':unit,'calendar_month':month,'spout_no':spout,'target':target,
                    'N':len(g),'absolute_error_sum':float(residual.abs().sum()),'target_sum':float(g[target].sum()),
                    'WMAPE':float(residual.abs().sum()/g[target].sum()),'signed_mean_residual':float(residual.mean()),
                    'median_residual':float(residual.median())})
        pd.DataFrame(records).to_csv(root/'target_month_spout.csv',index=False)
        changes=[]
        for month in reg['origins']:
            v=frame(root/'predictions'/f'{month}_V1.csv').set_index('sample_id')
            for role in (CANDIDATE,CONTROL):
                p=frame(root/'predictions'/f'{month}_{role}.csv').set_index('sample_id').loc[v.index]
                changes.append({'month':month,'role':role,'time_delta_quantiles':dict(zip(['min','p01','p05','p25','p50','p75','p95','p99','max'],np.quantile(p[PRED[1]]-v[PRED[1]],[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist()))})
        atomic_write_json(root/'prediction_change_diagnostics.json',changes)
    atomic_write_json(root/'scoring_zero_fit.json',counts)
    return result


def run(root):
    manifest=freeze(root)
    try:
        verify(manifest)
        h2,h1=p0(root,manifest)
        fit_and_predict(root,manifest,h2,h1)
        subprocess.run([sys.executable,'scripts/optimization_v14_cold_check.py','--run',str(root)],check=True)
        result=score_saved(root,manifest)
        verify(manifest)
        files=[p for p in root.rglob('*') if p.is_file()]
        atomic_write_json(root/'completion.json',{'status':result['status'],'G0':'PASS','G1':'PASS_HISTORICAL' if result['historical_quality_passed'] else 'FAIL_CLOSE_V7',
            'identities':file_identities({str(p):p for p in files}), 'ledger_sha256':file_sha256(manifest['scope']['new_ledger']),
            'new_CatBoost_fits':0,'development_time_LAD':12,'final_time_LAD':0,'challenger_ZIPs':0,
            'official_data_identity_verified':False,'platform_verified':False})
        print(result,flush=True)
    except Exception as exc:
        failure=root/'failure.json'
        if not failure.exists(): atomic_write_json(failure,{'error':str(exc),'fit_counts':ScalarBudget(root/'coefficients').counts()})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
