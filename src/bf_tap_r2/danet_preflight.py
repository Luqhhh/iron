"""One-shot synthetic full-size DANet admission; no official-label loading."""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import torch
import yaml

from .data import FEATURES
from .dnnr_preflight import source_hashes as python_source_hashes,private_path,anchored,runtime as python_runtime
from .danet_model import Settings,ARMS,Regressor
from .danet_ledger import ReservationLedger,write_new,file_hash
from .danet_execution import execute_unit,audit_unit
from .danet_audit import ABSOLUTE,RELATIVE

SPEC='configs/danet_abstract/SPEC.yaml'
TASK=dict(target='tap_time_len',seed=42,fold=0)
LIMITS=dict(pair_unit=1,estimator=4,optimizer=4)
PROBE=dict(seed=57331,rows=2755,training_rows=2204,query_rows=551,spout_count=4,encoded_dimensions=26,
    formula='30+2*sin(x0)+x1*x2+0.5*x3+0.1*epsilon',worst_case_optimizer_epochs=960)
REQUIRED_FILES=(SPEC,'scripts/run_danet_preflight_frozen.py','scripts/monitor_danet_preflight_once.py',
    'tests/test_danet_preflight.py','licenses/DANet-MIT.txt','licenses/QHoptim-MIT.txt',
    *('src/bf_tap_r2/danet_'+name+'.py' for name in
      ('terms','optimizer','network','model','calibration','audit','ledger','execution','preflight')))


def source_hashes(workspace):
    workspace=Path(workspace);sources=python_source_hashes(workspace)
    for name in ('licenses/DANet-MIT.txt','licenses/QHoptim-MIT.txt'):
        path=workspace/name
        if path.is_symlink() or path.resolve()!=workspace.resolve()/name:raise ValueError('License source escaped worktree')
        sources[name]=file_hash(path)
    return dict(sorted(sources.items()))


def checked_receipt(workspace,path,sha256,sources,current_runtime):
    path=private_path(workspace,path);checked=anchored(path,sha256)
    if (checked['status']!='passed' or checked['exit_code']!=0 or checked['full_suite'] is not True
            or checked['source_changed'] is not False or checked['source_hashes']!=sources
            or checked['runtime']!=current_runtime or type(checked['passed']) is not int or checked['passed']<1
            or not set(REQUIRED_FILES).issubset(sources)):
        raise ValueError('Passed exact-source full locked Python3.12 DANet receipt required')
    log=private_path(workspace,checked['log'])
    if file_hash(log)!=checked['log_sha256'] or checked['summary'] not in log.read_text():
        raise ValueError('Checked full-suite log changed')
    return checked


def runtime():
    if torch.get_num_threads()!=1:raise ValueError('Single-threaded locked DANet runtime required')
    return dict(**python_runtime(),torch=torch.__version__,torch_threads=torch.get_num_threads())


def validate_spec(workspace):
    spec=yaml.safe_load((Path(workspace)/SPEC).read_text());stage=spec['future_formal_stage']
    if (spec['version']!='danet20-matched-mask-regression-v1' or spec['training']!=asdict(Settings())
            or spec['arms']!=list(ARMS) or spec['eligible_candidates']!=['DANET_LEARNED']
            or spec['matched_controls']!={'DANET_LEARNED':'DANET_FIXED'} or spec['postprocessing']!='none'
            or spec['control']!='identically_initialized_network_freeze_only_mask_logits'
            or spec['saved_audit']['tolerances']!=dict(absolute=ABSOLUTE,relative=RELATIVE)
            or stage['development_seeds']!=[42,3407] or stage['confirmation_seeds']!=[7777,12011]
            or stage['folds']!=[0,1,2,3,4] or stage['development_pair_units']!=20
            or stage['development_estimator_runs']!=80 or stage['development_optimizer_runs']!=80 or stage['workers']!=4
            or stage['development_gate']!=dict(both_complete_seeds_positive=True,mean_gain_minimum=.01,
                positive_mean_candidate_vs_matched_control=True,mean_package_score_minimum=96.25)
            or stage['confirmation_gate']!=dict(all_four_seeds_positive=True,seed_level_paired_lcb95_positive=True,
                t_multiplier=2.3533634348018264,sample_sd_ddof=1,positive_four_seed_candidate_vs_matched_control=True)
            or stage['fold_level']!='descriptive_only' or stage['cross_seed_oof_averaging']!='forbidden'
            or spec['resources']!=dict(full_size_probe_required=True,workers=4,numeric_threads=1,maximum_worker_mib=1536,
                free_memory_rule='4_times_measured_peak_plus_1024_MiB',probe_rows=dict(training=2204,query=551,dimensions=26),
                worst_case='both_inner_and_outer_per_arm_all_240_epochs',
                projection='20_times_worst_case_pair_fit_save_and_independent_audit_divided_by_4_times_1_5_plus_300',
                maximum_development_seconds=7200,failed_cost_probe='preserve_no_retry_shrink_or_gate_relaxation')
            or spec['monitoring']!=dict(interval_seconds=600,between_check_polling=False)
            or spec['release']!=dict(automatic_packages=False,full_data_fits=0,desktop_writes=0,agent_uploads=0)):
        raise ValueError('Prospective DANet model, controls, gates or resource contract changed')
    return spec


def synthetic_data():
    generator=np.random.default_rng(PROBE['seed']);raw=generator.normal(size=(PROBE['rows'],len(FEATURES)))
    y=30+2*np.sin(raw[:,0])+raw[:,1]*raw[:,2]+.5*raw[:,3]+.1*generator.normal(size=len(raw))
    frame=pd.DataFrame(raw,columns=FEATURES);frame['sample_id']=[f'danet-resource-{i:05d}' for i in range(len(frame))]
    frame['spout_no']=np.arange(len(frame))%4+1;n=PROBE['training_rows']
    return frame.iloc[:n].reset_index(drop=True),y[:n],frame.iloc[n:].reset_index(drop=True),y[n:]


def resource_decision(measurement,available_mib):
    m=measurement
    values=[m[k] for k in ('pair_fit_save_seconds','pair_audit_seconds','maximum_worker_mib','maximum_bound_fraction',
        'learned_mask_l1_change','fixed_mask_l1_change')]
    if (not np.isfinite(values+[available_mib,m['median_mae'],*m['mae'].values()]).all() or min(values[:3])<=0
            or min(values+[available_mib,m['median_mae'],*m['mae'].values()])<0
            or m['optimizer_epochs']!=960 or m['training_rows']!=2204 or m['query_rows']!=551
            or m['encoded_dimensions']!=26 or m['saved_models']!=4 or set(m['mae'])!=set(ARMS)):
        raise ValueError('Complete finite worst-case resource witness required')
    projected=20*(m['pair_fit_save_seconds']+m['pair_audit_seconds'])/4*1.5+300
    required=4*m['maximum_worker_mib']+1024
    checks=dict(cost=projected<=7200,worker_memory=m['maximum_worker_mib']<=1536,available_memory=available_mib>=required,
        numerical=m['maximum_bound_fraction']<=1,learnability=all(v<m['median_mae'] for v in m['mae'].values()),
        learned_mask_active=m['learned_mask_l1_change']>0,fixed_mask_unchanged=m['fixed_mask_l1_change']==0)
    return dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
        projected_development_seconds=projected,required_available_mib=required,available_mib=available_mib)


def freeze_probe(workspace,output,receipt_path,receipt_sha256):
    workspace=Path(workspace).resolve();output=private_path(workspace,output)
    sources=source_hashes(workspace);current_runtime=runtime()
    checked=checked_receipt(workspace,receipt_path,receipt_sha256,sources,current_runtime)
    if subprocess.check_output(['git','status','--porcelain','--',*sources],cwd=workspace,text=True).strip():
        raise ValueError('Commit complete tested resource implementation before freeze')
    spec=validate_spec(workspace);output.mkdir(parents=True,exist_ok=False)
    manifest=dict(format='danet-synthetic-admission-v1',workspace=str(workspace),source_hashes=sources,runtime=current_runtime,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip(),spec=spec,
        settings=spec['training'],probe=PROBE,limits=LIMITS,checked_tests=dict(path=str(Path(receipt_path).resolve()),sha256=receipt_sha256,receipt=checked),
        official_label_reads=0,official_estimator_fits=0,automatic_formal_execution=False,release_authorized=False)
    write_new(output/'manifest.json',manifest)
    return dict(manifest=str(output/'manifest.json'),manifest_sha256=file_hash(output/'manifest.json'))


def verify_manifest(path,sha256):
    manifest=anchored(path,sha256);workspace=Path(manifest['workspace']);private_path(workspace,path)
    if (manifest['format']!='danet-synthetic-admission-v1' or Path(__file__).resolve()!=workspace/'src/bf_tap_r2/danet_preflight.py'
            or manifest['source_hashes']!=source_hashes(workspace) or manifest['runtime']!=runtime()
            or manifest['spec']!=validate_spec(workspace) or manifest['settings']!=asdict(Settings())
            or manifest['probe']!=PROBE or manifest['limits']!=LIMITS or manifest['official_label_reads']!=0
            or manifest['official_estimator_fits']!=0 or manifest['automatic_formal_execution'] or manifest['release_authorized']):
        raise ValueError('Frozen DANet source/runtime/spec/scope changed')
    if checked_receipt(workspace,manifest['checked_tests']['path'],manifest['checked_tests']['sha256'],
            manifest['source_hashes'],manifest['runtime'])!=manifest['checked_tests']['receipt']:
        raise ValueError('Checked locked-suite receipt changed')
    return manifest


def measure(manifest_path,manifest_sha256,policy_sha256):
    manifest=verify_manifest(manifest_path,manifest_sha256);root=Path(manifest_path).resolve().parent/'preflight'
    training,y,query,truth=synthetic_data();start=time.perf_counter()
    result=execute_unit(TASK,training,y,query,root/'unit',root/'ledger',policy_sha256,
        settings=Settings(**manifest['settings']),resource_upper_bound=True)
    fit_save=time.perf_counter()-start;start=time.perf_counter()
    predictions,audit=audit_unit(root/'unit',result['complete_sha256'],TASK,training,y,query,
        settings=Settings(**manifest['settings']),resource_upper_bound=True,ledger_root=root/'ledger')
    audit_seconds=time.perf_counter()-start
    learned=Regressor.load(root/'unit/models/outer-DANET_LEARNED.npz',file_hash(root/'unit/models/outer-DANET_LEARNED.npz'))
    fixed=Regressor.load(root/'unit/models/outer-DANET_FIXED.npz',file_hash(root/'unit/models/outer-DANET_FIXED.npz'))
    measurement=dict(training_rows=len(training),query_rows=len(query),encoded_dimensions=learned.x_.shape[1],saved_models=audit['saved_models'],
        optimizer_epochs=audit['optimizer_epochs'],pair_fit_save_seconds=fit_save,pair_audit_seconds=audit_seconds,
        maximum_worker_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        maximum_bound_fraction=max(value['maximum_bound_fraction'] for report in audit['models'].values() for value in report['checks'].values()),
        mae={arm:float(np.abs(prediction-truth).mean()) for arm,prediction in predictions.items()},
        median_mae=float(np.abs(truth-np.median(y)).mean()),learned_mask_l1_change=float(learned.trace_['mask_change'][-1]),
        fixed_mask_l1_change=float(fixed.trace_['mask_change'][-1]),unit_complete_sha256=result['complete_sha256'],audit=audit)
    verify_manifest(manifest_path,manifest_sha256);write_new(root/'measurement.json',measurement)
    return dict(measurement_sha256=file_hash(root/'measurement.json'))


def run_probe(manifest_path,manifest_sha256):
    manifest=verify_manifest(manifest_path,manifest_sha256);root=Path(manifest_path).resolve().parent/'preflight'
    root.mkdir(exist_ok=False);ledger=ReservationLedger.create(root/'ledger',LIMITS)
    available=psutil.virtual_memory().available/1024**2
    write_new(root/'started.json',dict(manifest_sha256=manifest_sha256,policy_sha256=ledger.policy_sha256,available_mib=available))
    try:
        command=[sys.executable,'-m','bf_tap_r2.danet_preflight','worker','--manifest',str(manifest_path),
            '--manifest-sha256',manifest_sha256,'--policy-sha256',ledger.policy_sha256]
        with (root/'worker.log').open('x') as log:subprocess.run(command,cwd=manifest['workspace'],stdout=log,stderr=subprocess.STDOUT,check=True)
        verify_manifest(manifest_path,manifest_sha256)
        measurement=json.loads((root/'measurement.json').read_text());decision=resource_decision(measurement,available)
        counts=ledger.inspect()
        if counts['started']!=LIMITS or counts['completed']!=LIMITS or any(v for status in ('failed','incomplete') for v in counts[status].values()):
            raise ValueError('Resource fitting reservations incomplete or changed')
        hashes={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()}
        report=dict(**decision,manifest_sha256=manifest_sha256,measurement=measurement,policy_sha256=ledger.policy_sha256,
            counts=counts,artifact_hashes=hashes,official_label_reads=0,official_estimator_fits=0,automatic_formal_execution=False,release_authorized=False)
        write_new(root/'admission.json',report)
        if report['status']!='passed':raise ValueError('Frozen full-size DANet resource gate failed')
    except BaseException as exc:
        write_new(root/'failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return dict(admission_sha256=file_hash(root/'admission.json'),**decision)


def verify_admission(path,sha256,manifest_path,manifest_sha256):
    manifest=verify_manifest(manifest_path,manifest_sha256);path=private_path(manifest['workspace'],path);root=path.parent
    report=anchored(path,sha256)
    if (root!=Path(manifest_path).resolve().parent/'preflight' or (root/'failed.json').exists()
            or report['status']!='passed' or report['manifest_sha256']!=manifest_sha256
            or report['official_label_reads']!=0 or report['official_estimator_fits']!=0
            or report['automatic_formal_execution'] or report['release_authorized']):raise ValueError('Passed original DANet synthetic admission required')
    if {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}!=set(report['artifact_hashes'])|{'admission.json'}:
        raise ValueError('Closed synthetic probe artifact schema differs')
    for name,digest in report['artifact_hashes'].items():
        item=root/name
        if item.resolve()!=item or not item.is_relative_to(root) or file_hash(item)!=digest:raise ValueError('Resource artifact changed or escaped')
    measurement=anchored(root/'measurement.json',report['artifact_hashes']['measurement.json'])
    if measurement!=report['measurement']:raise ValueError('Resource measurement changed')
    ledger=ReservationLedger.open(root/'ledger',report['policy_sha256']);counts=ledger.inspect()
    if (ledger.limits!=LIMITS or counts!=report['counts'] or counts['started']!=LIMITS or counts['completed']!=LIMITS
            or any(v for status in ('failed','incomplete') for v in counts[status].values())):raise ValueError('Resource ledger differs')
    training,y,query,truth=synthetic_data()
    predictions,audit=audit_unit(root/'unit',measurement['unit_complete_sha256'],TASK,training,y,query,
        settings=Settings(**manifest['settings']),resource_upper_bound=True,ledger_root=root/'ledger')
    if audit!=measurement['audit'] or any(float(np.abs(predictions[a]-truth).mean())!=measurement['mae'][a] for a in ARMS):
        raise ValueError('Fresh no-fit resource audit or synthetic arithmetic differs')
    fresh=dict(training_rows=len(training),query_rows=len(query),saved_models=audit['saved_models'],optimizer_epochs=audit['optimizer_epochs'],
        median_mae=float(np.abs(truth-np.median(y)).mean()),
        maximum_bound_fraction=max(value['maximum_bound_fraction'] for model in audit['models'].values() for value in model['checks'].values()))
    for arm,key in (('DANET_LEARNED','learned_mask_l1_change'),('DANET_FIXED','fixed_mask_l1_change')):
        model=Regressor.load(root/f'unit/models/outer-{arm}.npz',report['artifact_hashes'][f'unit/models/outer-{arm}.npz'])
        fresh[key]=float(model.trace_['mask_change'][-1]);fresh['encoded_dimensions']=model.x_.shape[1]
    if any(measurement[k]!=v for k,v in fresh.items()):raise ValueError('Fresh dimensions/mask/numerical resource witness differs')
    decision=resource_decision(measurement,report['available_mib'])
    if any(report[k]!=v for k,v in decision.items()):raise ValueError('Resource admission arithmetic differs')
    if psutil.virtual_memory().available/1024**2<report['required_available_mib']:raise ValueError('Current memory below admitted requirement')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','worker'])
    parser.add_argument('--manifest',type=Path);parser.add_argument('--manifest-sha256');parser.add_argument('--policy-sha256')
    parser.add_argument('--workspace',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--receipt',type=Path);parser.add_argument('--receipt-sha256');args=parser.parse_args()
    if args.action=='freeze':result=freeze_probe(args.workspace,args.output,args.receipt,args.receipt_sha256)
    elif args.action=='run':result=run_probe(args.manifest,args.manifest_sha256)
    else:result=measure(args.manifest,args.manifest_sha256,args.policy_sha256)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
