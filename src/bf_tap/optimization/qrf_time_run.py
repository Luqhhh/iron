"""OPT-32/33 thin isolated-worker lifecycle; original builders and scorer stay frozen."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from ..artifacts import (atomic_write_json, file_identities, file_sha256, stable_digest,
    verify_file_identities, runtime_environment, validate_inference_source_contract)
from ..config import load_yaml
from ..exceptions import ContractError
from ..metrics import score_predictions
from .component_export import META, PRED, read_json, units
from .dual_ratio_common import frame, score, week_intervals
from .rate_model import schema
from .refresh_factorial import Context, stamp
from .structural import apply_correction
from .structural_run import append_ledger, predict_inputs
from .trajectory_run import builder_for, load_fold, equal
from .v13_common import zero_fit, verify_receipt

CANDIDATE = 'V8_QRF_TIME'
DIAGNOSTIC = 'D2_QRF_WEIGHTED_MEAN_TIME'
WORKER_FILES = ('worker.py','qrf_model.py','preprocessing.py','pyproject.toml','uv.lock')


def registration():
    reg = load_yaml('configs/optimization_v0_15/experiment.yaml')
    expected = {'forest':6,'preprocessor':6,'internal_trees':1536,'CatBoost':0,'LAD':0,
                'conditional_final_forest':1,'conditional_final_preprocessor':1}
    if (reg['candidate'] != CANDIDATE or reg['diagnostic'] != DIAGNOSTIC or
        reg['protocol'] != 'QRF_FULLTRAIN_LEAF_v1' or reg['quantile'] != .5 or reg['budget'] != expected or
        reg['origins'] != {6:4,7:4,8:4,9:3,10:2,11:1}):
        raise ContractError('registered QRF identity/budget differs')
    return reg


def worker(reg, command, *args):
    here = Path(reg['worker']).resolve()
    return subprocess.check_output([str(here/'.venv/bin/python'),str(here/'worker.py'),command,*map(str,args)],text=True)


def event(root, manifest, purpose):
    append_ledger({**manifest['scope'],'authorization':manifest['scope']['authorization']+':'+purpose},root)


def verify(manifest):
    for key in ('sources','inputs','evidence','old_ledgers'): verify_file_identities(manifest[key])
    for contract in manifest['source_contracts']:
        validate_inference_source_contract(contract,manifest['inputs'],semantic_contract_sha256=contract['semantic_contract_sha256'])


def restore_manifest(root):
    manifest=read_json(root/'manifest.json')
    for key in ('origins','expected_training_rows'):
        manifest['registration'][key]={int(k):v for k,v in manifest['registration'][key].items()}
    return manifest


def freeze(root):
    reg=registration(); scope=load_yaml('configs/optimization_v0_15/access_scope.yaml')
    protection=load_yaml(scope['protection_contract'])
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip(): raise ContractError('clean registered working tree required')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=reg['branch']: raise ContractError('registered branch required')
    subprocess.run(['git','merge-base','--is-ancestor',reg['base_commit'],'HEAD'],check=True)
    if not scope['holdout_consumed'] or not read_json('EVIDENCE_STATUS.json')['holdout_consumed']: raise ContractError('consumed retrospective scope required')
    # Search pre-existing registries/manifests, never sensitive CSV contents, before any fit.
    matches=[]
    for p in Path('local/runs').glob('**/*manifest*.json'):
        if 'QRF_FULLTRAIN_LEAF_v1' in p.read_text() or 'V8_QRF_TIME' in p.read_text(): matches.append(str(p))
    if matches: raise ContractError('existing equivalent QRF registration; verify rather than retrain: '+str(matches))
    old=read_json(reg['inventory_manifest']);source=Path(reg['source_v8']); original=read_json(source/'manifest.json')
    for key in ('sources','inputs','evidence'): verify_file_identities(original[key])
    if read_json(source/'final_status.json')['G0']!='PASS_OPT20' or read_json(Path(reg['source_v8_cold'])/'validation.json')['status']!='PASS': raise ContractError('completed old V1 source/cold evidence required')
    receipts=[verify_receipt(p) for p in reg['prior_receipts']]
    evidence=[Path(reg['inventory_manifest']),source/'manifest.json',source/'final_status.json',source/'summary.json',source/'metrics.json',Path(scope['protection_contract']),Path(reg['source_v8_cold'])/'validation.json',*map(Path,reg['prior_receipts'])]
    evidence += [Path(p) for r in receipts for p in r['evidence_sha256']]
    evidence.append(Path('local/reports/optimization-v0.15-remote-observations-r1.json'))
    evidence.append(Path('local/reports/pytest-qrf-v015-worker-synthetic-r4.xml.synthetic_fit_counts.json'))
    for test_path in reg['required_test_evidence']:
        suites=ET.parse(test_path).getroot().iter('testsuite')
        if any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in suites):
            raise ContractError('required locked synthetic tests failed')
        evidence.append(Path(test_path))
    contracts=[]
    for month in reg['origins']:
        record=old['inventory'][str(month)]
        if record != read_json(source/'models'/str(month)/'fit_record.json'): raise ContractError('original training registry differs')
        evidence += [source/'models'/str(month)/'fit_record.json', source/'oof'/f'{month}.csv',
                     source/'predictions'/f'{month}_inputs.csv',source/'predictions'/f'{month}_V1.csv',source/'corrections'/f'{month}.json']
        evidence.append(Path(reg['inventory_manifest']).parent/'p0'/f'{month}_audit.json')
        for entry in record['base_components'].values():
            folder=Path(entry['path']); contracts.append(read_json(folder/'bundle.json')['inference_source_contract'])
            evidence += [p for p in folder.rglob('*') if p.is_file()]
        evidence += [p for p in (source/'models'/str(month)/'rate').rglob('*') if p.is_file()]
    for unit,*_ in units(reg): evidence += [source/'units'/unit/f'{k}_errors.csv' for k in ('U0','U1')]
    for key in ('active_release','fallback_release'):
        release=load_yaml(reg[key])
        if file_sha256(release['test_a_zip'])!=release['test_a_zip_sha256']: raise ContractError('original ZIP differs')
        evidence += [Path(reg[key]),Path(release['test_a_zip'])]
        evidence += [p for p in Path(release['bundle']).rglob('*') if p.is_file()]
        parent=Path(release['bundle']).parent
        evidence += [parent/'cold_predictions.csv',parent/('cold_data_repaired.yaml' if key=='active_release' else 'cold_data.yaml')]
    files=[*Path('src/bf_tap').rglob('*.py'),*Path('configs/optimization_v0_15').glob('*.yaml'),Path('tests/test_qrf_time_contract.py'),Path('scripts/optimization_v15_cold_check.py'),Path('docs/optimization_v0_15/PLAN.md'),Path('uv.lock'),Path('pyproject.toml')]
    files += [Path(reg['worker'])/name for name in WORKER_FILES]
    files += list((Path(reg['worker'])/'tests').glob('*.py'))
    env=json.loads(worker(reg,'environment'))
    if env['python']!='3.12.12' or env['numpy']!='2.2.6' or env['sklearn']!='1.8.0': raise ContractError('fixed worker versions unavailable')
    root.mkdir(parents=True,exist_ok=False);root.chmod(0o700)
    manifest=dict(registration=reg,scope=scope,protection=protection,algorithm=original['algorithm'],
        inventory=old['inventory'],inputs=old['inputs'],sources=file_identities({str(p):p for p in files}),
        evidence=file_identities({str(p):p for p in evidence}),old_ledgers=file_identities({p:p for p in scope['old_ledgers']}),
        environment=runtime_environment(),worker_environment=env,worker_sources={name:file_sha256(Path(reg['worker'])/name) for name in WORKER_FILES},
        source_contracts=contracts,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        equivalent_registry_matches=matches,test_inputs_for_selection=False,official_target_column_reads=False,
        holdout_consumed=True,timestamp_contract_status='ASSUMED')
    atomic_write_json(root/'manifest.json',manifest);verify(manifest);event(root,manifest,'REGISTERED_BEFORE_TRAINING_AND_NEW_DIAGNOSTIC_LABELS')
    return manifest


def fold_for(manifest, month):
    fold,oof=load_fold(manifest,month)
    for _,h in fold[1].values():
        for col in ('reference_time','available_at','tap_end_time'): h[col]=pd.to_datetime(h[col],utc=True).dt.tz_convert('Asia/Shanghai')
    return fold,oof


def raw_matrix(builder, samples, fold, training=False):
    entry=fold[0]['OR']; h=fold[1]['OR'][1]
    x=(Context.X(builder,samples,stamp(pd.Timestamp(entry['cutoff']).month),entry['component'],'R2',h)
       if training else builder.X(samples,entry,h))
    if schema(x)!=fold[1]['OR'][0].feature_schema_ or any('trajectory' in col or 'slope' in col for col in x): raise ContractError('original raw schema changed or V5 columns mixed in')
    return x


def pack(path,x,samples,cutoff,history=None):
    if len(x)!=len(samples) or samples.sample_id.duplicated().any() or samples[META].isna().any().any(): raise ContractError('raw feature/metadata IDs differ')
    if not x.index.equals(samples.index) or not np.array_equal(x.spout_no.astype(str).to_numpy(),samples.spout_no.astype(str).to_numpy()): raise ContractError('raw feature row order/spout differs from metadata')
    if len(set(x.columns))!=len(x.columns) or 'spout_no' not in x: raise ContractError('invalid raw schema')
    cols=[c for c in x if c!='spout_no']
    if any(not pd.api.types.is_numeric_dtype(x[c]) for c in cols): raise ContractError('illegal raw numeric dtype')
    numeric=x[cols].to_numpy(dtype=np.float64)
    if np.isinf(numeric).any(): raise ContractError('raw Inf rejected')
    if (np.abs(numeric[np.isfinite(numeric)])>np.finfo(np.float32).max).any(): raise ContractError('raw float32 overflow rejected before training')
    ids=samples.sample_id.astype(str).tolist()
    arrays=dict(ids=np.array(ids,dtype=str),numeric=numeric,spout=x.spout_no.astype(str).to_numpy(dtype=str),
                reference_ns=samples.reference_time.astype('int64').to_numpy())
    if history is not None:
        if history[['reference_time','available_at','tap_time_len']].isna().any().any() or history.sample_id.astype(str).tolist()!=ids or not np.array_equal(history.reference_time.to_numpy(),samples.reference_time.to_numpy()) or (history.reference_time>=cutoff).any() or (history.available_at>cutoff).any(): raise ContractError('training labels/ID/cutoff alignment differs')
        if not np.isfinite(history.tap_time_len).all() or (history.tap_time_len<0).any(): raise ContractError('invalid training time responses')
        arrays.update(y=history.tap_time_len.to_numpy(dtype=np.float64),available_ns=history.available_at.astype('int64').to_numpy(),
                      training_months=history.reference_time.dt.strftime('%Y-%m').to_numpy(dtype=str))
    elif (samples.reference_time<cutoff).any(): raise ContractError('future model for prediction reference')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise ContractError('never overwrite feature handoff')
    np.savez(path,**arrays)
    info=dict(sha256=file_sha256(path),ids=ids,numeric_columns=cols,raw_schema=schema(x),raw_schema_sha256=stable_digest(schema(x)),
              raw_matrix_sha256=stable_digest(dict(schema=schema(x),rows=pd.util.hash_pandas_object(x,index=False).astype(str).tolist())),
              cutoff_ns=cutoff.value,cutoff=str(cutoff),training=history is not None)
    atomic_write_json(str(path)+'.json',info)
    return info


def rebuild_old(root,manifest):
    reg=manifest['registration'];source=Path(reg['source_v8']);old=read_json(source/'metrics.json'); horizons={h:[] for h in range(1,5)}
    checks={}
    event(root,manifest,'P0_REBUILD_ALREADY_CONSUMED_OLD_V1_METRICS_ONLY')
    for unit,_,h,_,_ in units(reg):
        actual=frame(source/'units'/unit/'U1_errors.csv')
        got=score_predictions(actual,actual[['sample_id',*PRED]])
        expected=old[unit]['candidates']['U1']['overall']
        for target in ('iron','time'):
            for field in ('abs_error_sum','actual_sum','wmape'):
                if abs(got[target][field]-expected[target][field])>1e-12: raise ContractError('old metric numerator/denominator reconstruction differs')
        if abs(got['loss']-expected['loss'])>1e-12: raise ContractError('old E reconstruction differs')
        if h: horizons[h].append(got['loss'])
        checks[unit]=got
    j=float(np.mean([np.mean(v) for v in horizons.values()]))
    if abs(j-read_json(source/'summary.json')['U1']['J'])>1e-12: raise ContractError('old J reconstruction differs')
    atomic_write_json(root/'p0/old_metric_reconstruction.json',dict(status='PASS',J=j,units=checks,tolerance=1e-12))


def preflight(root,manifest):
    reg=manifest['registration']; dest=root/'p0';dest.mkdir(exist_ok=False)
    builder=builder_for(manifest);train_inputs={}; audits={}
    with zero_fit() as count:
        event(root,manifest,'P0_RESTORE_ORIGINAL_CUTOFF_HISTORIES_FOR_TRAIN_ONLY_FEATURES')
        for month in reg['origins']:
            fold,_=fold_for(manifest,month); h=fold[1]['OR'][1]
            if len(h)!=reg['expected_training_rows'][month]: raise ContractError('original rows differ; do not pad/truncate')
            x=raw_matrix(builder,h[META],fold,training=True)
            digest=stable_digest(dict(schema=schema(x),row_hashes=pd.util.hash_pandas_object(x,index=False).astype(str).tolist()))
            prior_audit=read_json(Path(reg['inventory_manifest']).parent/'p0'/f'{month}_audit.json')
            if digest!=prior_audit['old_feature_matrix_sha256']: raise ContractError('original training feature values changed')
            train_inputs[str(month)]=pack(root/'features'/str(month)/'train.npz',x,h[META],stamp(month),h)
            samples=frame(Path(reg['source_v8'])/'predictions'/f'{month}_inputs.csv',usecols=META)
            samples.reference_time=pd.to_datetime(samples.reference_time,utc=True).dt.tz_convert('Asia/Shanghai')
            old_inputs=predict_inputs(samples,fold,builder,{'rate':{'unusable_predicted_rate_max':1e-6}})
            old_v1=apply_correction(old_inputs,read_json(Path(reg['source_v8'])/'corrections'/f'{month}.json')['alpha'],1e-6)
            original_v1=frame(Path(reg['source_v8'])/'predictions'/f'{month}_V1.csv')
            difference=equal(old_v1,original_v1,PRED)
            xeval=raw_matrix(builder,samples,fold)
            evaluation=pack(root/'features'/str(month)/'evaluation.npz',xeval,samples,stamp(month))
            saved=root/'original_predictions'/f'{month}.csv';saved.parent.mkdir(exist_ok=True)
            original_v1.to_csv(saved,index=False)
            audits[str(month)]=dict(rows=len(h),evaluation_rows=len(samples),original_V1_max_difference=difference,
                training_reference_max=str(h.reference_time.max()),training_available_max=str(h.available_at.max()),
                original_OR=fold[0]['OR'],raw_schema=schema(x),evaluation=evaluation,
                feature_definition_values_unchanged=True)
            atomic_write_json(dest/f'{month}_audit.json',audits[str(month)])
            builder.cache.clear()
            print(f'P0 cutoff {month}: {len(h)} original rows, raw schema verified, V1 replay diff={difference}',flush=True)
        # Replay both original published packages under their original locked environment.
        for key,script in [('active_release','scripts/optimization_v8_cold_predict.py'),('fallback_release','scripts/optimization_v4_cold_predict.py')]:
            release=load_yaml(reg[key]);parent=Path(release['bundle']).parent
            config=parent/('cold_data_repaired.yaml' if key=='active_release' else 'cold_data.yaml')
            out=dest/f'{key}.csv'
            subprocess.run([sys.executable,script,'--bundle',release['bundle'],'--data-config',str(config),'--output',str(out)],check=True)
            equal(frame(out),frame(parent/'cold_predictions.csv'),PRED)
        rebuild_old(root,manifest)
    verify(manifest)
    atomic_write_json(dest/'complete.json',dict(status='PASS',training_inputs=train_inputs,audits=audits,zero_fit=count,
        root_lock_sha256=file_sha256('uv.lock'),worker_lock_sha256=file_sha256(Path(reg['worker'])/'uv.lock')))


def fit_counts(root):
    models=root/'models'
    result=dict(forest_attempted=len(list(models.glob('*/forest_intent.json'))),forest_completed=len(list(models.glob('*/bundle.json'))),
        preprocessor_attempted=len(list(models.glob('*/preprocessor_intent.json'))),preprocessor_completed=len(list(models.glob('*/preprocessor.json'))))
    if max(result.values())>6: raise ContractError('QRF development fit budget exceeded')
    return dict(**result,internal_trees_completed=256*result['forest_completed'],CatBoost=0,LAD=0,final_forest=0,final_preprocessor=0,challenger_ZIPs=0)


def train_predict(root,manifest):
    reg=manifest['registration']
    if read_json(root/'p0/complete.json')['status']!='PASS': raise ContractError('P0 gate not passed')
    event(root,manifest,'UNLOCK_SIX_TRAIN_ONLY_FOREST_AND_PREPROCESSOR_FITS')
    for month in reg['origins']:
        print(worker(reg,'fit','--root',root,'--month',month).strip(),flush=True)
    models={str(m):file_sha256(root/'models'/str(m)/'bundle.json') for m in reg['origins']}
    atomic_write_json(root/'models_complete.json',models)
    for month in reg['origins']:
        out=root/'worker_predictions'/f'{month}.npz'
        worker(reg,'predict','--root',root,'--month',month,'--output',out)
        with np.load(out,allow_pickle=False) as data:
            old=frame(root/'original_predictions'/f'{month}.csv')
            old=old.set_index('sample_id').loc[data['ids'].tolist()].reset_index()
            for role,name in [(CANDIDATE,'median'),(DIAGNOSTIC,'mean')]:
                pred=old[['sample_id',*PRED]].copy();pred.pred_tap_time_len=data[name]
                if not np.array_equal(pred.pred_tap_iron.to_numpy(),old.pred_tap_iron.to_numpy()): raise ContractError('iron changed')
                dest=root/'predictions';dest.mkdir(exist_ok=True);pred.to_csv(dest/f'{month}_{role}.csv',index=False)
        print(f'QRF cutoff {month}: both frozen outputs persisted, outer scoring labels unread',flush=True)
    ids=file_identities({str(p):p for dirname in ('predictions','worker_predictions','original_predictions','features','models') for p in (root/dirname).rglob('*') if p.is_file()})
    atomic_write_json(root/'predictions_complete.json',dict(identities=ids,fit_counts=fit_counts(root),before_outer_scoring_label_access=True))


def acceptance(metrics,summary,reg,engineering,iron_exact):
    a=reg['acceptance']; cells={};recent=[]
    for h in range(1,5):
        cells[h]=np.array([v['candidates'][CANDIDATE]['overall']['loss']-v['candidates']['V1']['overall']['loss']
                          for k,v in metrics.items() if k.startswith('O') and v['horizon']==h])
    if [len(cells[h]) for h in range(1,5)] != [6,5,4,3]: raise ContractError('incomplete grid')
    for k,v in metrics.items():
        if k.startswith('O') and v['horizon']==2 and int(k[5:7]) in (8,9,10):
            recent.append(v['candidates'][CANDIDATE]['overall']['loss']-v['candidates']['V1']['overall']['loss'])
    j=summary[CANDIDATE]['J']-summary['V1']['J']
    dev={k:metrics[k]['candidates'][CANDIDATE]['overall']['loss']-metrics[k]['candidates']['V1']['overall']['loss'] for k in ('DEV_LONG','DEV_SHORT')}
    gates=dict(engineering_causal_budget=bool(engineering),iron_exact=bool(iron_exact),H2_mean_E=float(cells[2].mean())<=a['H2_mean_E'],
        H2_improved_origins=int((cells[2]<0).sum())>=a['H2_improved_origins'],recent_H2_improved=sum(v<0 for v in recent)>=a['recent_H2_improved'],
        H2_single_origin=float(cells[2].max())<=a['H2_single_origin'],J_guardrail=j<=a['J_guardrail'])
    gates.update({f'H{h}_guardrail':float(cells[h].mean())<=a['other_horizon_guardrail'] for h in (1,3,4)})
    gates.update({f'{k}_guardrail':v<=a['DEV_guardrail'] for k,v in dev.items()})
    passed=all(gates.values())
    return dict(status='DEV_ACCEPTED_PENDING_OFFICIAL_IDENTITY' if passed else 'FAIL_CLOSE_V8_RETAIN_V1',gates=gates,
        H2_delta_E=float(cells[2].mean()),H2_improved=int((cells[2]<0).sum()),recent_H2_improved=sum(v<0 for v in recent),
        H2_single_max_delta_E=float(cells[2].max()),delta_J=j,horizon_delta_E={str(h):float(v.mean()) for h,v in cells.items()},DEV_delta_E=dev,
        engineering_valid=bool(engineering),historical_quality_passed=passed,official_data_identity_verified=False,platform_verified=False,
        final_model_fitted=False,ready_challenger=False,diagnostic_releasable=False,source_semantics_status='ORIGINAL_ASSUMED_CONTRACT_RETAINED')


def evaluate(root,manifest):
    verify_file_identities(read_json(root/'predictions_complete.json')['identities'])
    if read_json(root/'cold_validation.json')['status']!='PASS': raise ContractError('independent cold audit must pass')
    reg=manifest['registration'];event(root,manifest,'OUTER_SCORING_ALREADY_CONSUMED_LABELS_AFTER_ALL_PREDICTIONS_FROZEN')
    def provider(unit,cutoff,samples,parts):
        result={}
        for role in (CANDIDATE,DIAGNOSTIC):
            pred=frame(root/'predictions'/f'{cutoff.month}_{role}.csv')
            result[role]=pred.loc[pred.sample_id.isin(samples.sample_id)]
        return result
    with zero_fit() as count:
        metrics,summary,errors,_=score(root,reg,provider)
        iron=True
        for unit,*_ in units(reg):
            old=errors.loc[(errors.unit==unit)&(errors.candidate=='V1')].set_index('sample_id').sort_index()
            for role in (CANDIDATE,DIAGNOSTIC):
                got=errors.loc[(errors.unit==unit)&(errors.candidate==role)].set_index('sample_id').sort_index()
                iron &= old.index.equals(got.index) and np.array_equal(old.pred_tap_iron.to_numpy(),got.pred_tap_iron.to_numpy())
        atomic_write_json(root/'acceptance.json',acceptance(metrics,summary,reg,True,iron))
        intervals={role:week_intervals(errors,role,'V1',**{'repetitions':1000,'seed':2026}) for role in (CANDIDATE,DIAGNOSTIC)}
        atomic_write_json(root/'bootstrap.json',intervals)
        diagnostics=[];errors['calendar_month']=pd.to_datetime(errors.reference_time,utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m')
        for (role,unit,month,spout),part in errors.groupby(['candidate','unit','calendar_month','spout_no']):
            for target in ('tap_iron','tap_time_len'):
                residual=part['pred_'+target]-part[target]
                diagnostics.append(dict(candidate=role,unit=unit,month=month,spout=str(spout),target=target,n=len(part),
                    absolute_error_sum=float(residual.abs().sum()),target_sum=float(part[target].sum()),wmape=float(residual.abs().sum()/part[target].sum()),
                    signed_mean=float(residual.mean()),residual_median=float(residual.median())))
        atomic_write_json(root/'target_month_spout.json',diagnostics)
        changes={}
        for role in (CANDIDATE,DIAGNOSTIC):
            got=errors.loc[errors.candidate==role].set_index(['unit','sample_id']).sort_index()
            old=errors.loc[errors.candidate=='V1'].set_index(['unit','sample_id']).sort_index()
            changes[role]=np.quantile(got.pred_tap_time_len-old.pred_tap_time_len,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist()
        atomic_write_json(root/'prediction_change_quantiles.json',changes)
    verify(manifest);atomic_write_json(root/'scoring_zero_fit.json',count)
    atomic_write_json(root/'fit_counts.json',fit_counts(root))
    files=[p for p in root.rglob('*') if p.is_file()]
    atomic_write_json(root/'completion.json',dict(status=read_json(root/'acceptance.json')['status'],G0='PASS',G1='PASS' if read_json(root/'acceptance.json')['historical_quality_passed'] else 'FAIL',
        evidence=file_identities({str(p):p for p in files}),ledger_sha256=file_sha256(manifest['scope']['new_ledger']),holdout_consumed=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--action',choices=['run','cold'],default='run');args=parser.parse_args()
    root=args.output.resolve()
    try:
        if args.action=='cold':
            from subprocess import run
            run([sys.executable,'scripts/optimization_v15_cold_check.py','--run',str(root)],check=True)
            return
        manifest=freeze(root);preflight(root,manifest);train_predict(root,manifest)
        subprocess.run([sys.executable,'scripts/optimization_v15_cold_check.py','--run',str(root)],check=True)
        evaluate(root,manifest)
    except Exception as exc:
        if root.exists(): atomic_write_json(root/'failure.json',dict(error=repr(exc),fit_counts=fit_counts(root)))
        raise


if __name__=='__main__': main()
