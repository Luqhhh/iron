"""One fixed SHORT full-training procedure; explicit task admission required."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / 'src'))

import numpy as np
from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.ema_reference_artifacts import sha, write_new, cold_only
from bf_tap_r2.ema_reference_ledger import binding_sources, reference_bindings, native_hooks
from bf_tap_r2.ema_span_confirmation import partition, runtime
from bf_tap_r2.ema_training_scale import require_memory
from bf_tap_r2.v7_periodic import digest

SPEC = 'configs/ema_short_release/PREPARATION.json'
AUTH_SCOPE = dict(candidate='SHORT_SPAN_FULL_Q75', full_training_procedures=1,
                  torch_optimizer=2, packages=1, new_CV=0,
                  new_confirmation_seeds=0, desktop_writes=0, agent_uploads=0)


def read(path):
    return json.loads(Path(path).read_text())


def validate_scope(spec, confirmation):
    expected = dict(candidate='SHORT_SPAN_FULL_Q75', target='tap_time_len',
        candidate_beta=.9801, control_beta=.99, replacement_weight=.75,
        full_training_procedures=1, torch_optimizer=2, new_CV=0,
        new_confirmation_seeds=0, packages=1, desktop_writes=0, agent_uploads=0,
        workers=1, numerical_threads=1, monitor_seconds=600,
        max_worker_rss_mib=1536, cold_predict_atol=.0005,
        maximum_runtime_seconds=None, automatic_scientific_retries=False,
        authorization_mode='external_explicit_task_required', require_four_seed_gate=True,
        full_scope_identity_seed=-1, full_scope_identity_fold=-1,
        confirmation_spec_sha256='cb7ff2d78e1b70f5219c778304b9a77a8048e9142db8497c425214984d985c24')
    if any(spec.get(k) != v or type(spec.get(k)) is not type(v) for k,v in expected.items()):
        raise ValueError('Unregistered SHORT release scope')
    if (spec['training'] != confirmation['training']
            or spec['mechanisms'] != confirmation['mechanisms']
            or spec['runtime_versions'] != confirmation['runtime_versions']):
        raise ValueError('Original scientific training/runtime changed')
    if spec['platform_reference'] != confirmation['platform_reference']:
        raise ValueError('Proposed Q75 parent differs from the confirmation reference')


def sources(workspace):
    workspace=Path(workspace)
    paths=list((workspace/'src').rglob('*.py'))
    paths += [workspace/p for p in (SPEC, 'configs/ema_span_confirmation/SPEC.json', 'scripts/ema_short_release.py',
        'tests/test_ema_short_release.py', 'uv.lock', 'pyproject.toml')]
    return {str(p.relative_to(workspace)):sha(p) for p in sorted(set(paths))}


def verify_files(files):
    for path,expected in files.items():
        if sha(path)!=expected: raise ValueError(f'Frozen input/source changed: {path}')


def explicit_authority(path):
    if path is None: return None
    grant=read(path)
    scope=grant.get('scope',{})
    if (grant.get('source')!='explicit_user_task' or scope!=AUTH_SCOPE
            or any(type(scope.get(k)) is not type(v) for k,v in AUTH_SCOPE.items())
            or grant.get('user_authorized') is not True
            or not isinstance(grant.get('user_request'),str) or not grant['user_request'].strip()):
        raise ValueError('Explicit concrete full-release task authorization required')
    return dict(path=str(Path(path).resolve()),sha256=sha(path),request=grant['user_request'])


def confirmation_evidence(spec):
    out=Path(spec['confirmation_run']);audit=read(out/'audit.json')
    summary=read(out/'summary.json');completion=read(out/'completion-event.json')
    terminal=read(out/'process-terminal.json');verification=read(spec['confirmation_terminal_verification'])
    hashes={name:sha(out/name) for name in ('manifest.json','summary.json','audit.json',
        'completion-event.json','process-terminal.json')}
    if (audit['status']!='passed' or audit['four_seed_gate_passed'] is not True
            or audit['complete_units']!=10 or audit['new_retained_states']!=440
            or audit['manifest_sha256']!=hashes['manifest.json']
            or audit['summary_sha256']!=hashes['summary.json']
            or set(audit['seed_gains'])!={'42','3407','271828','314159'}
            or not all(v>0 for v in audit['seed_gains'].values())
            or summary['four_seed_gate_passed'] is not True or summary['paired']['lcb95']<=0
            or completion['status']!='completed'
            or completion['summary_sha256']!=hashes['summary.json']
            or completion['audit_sha256']!=hashes['audit.json']
            or terminal['exit_code']!=0 or verification['status']!='passed'
            or verification['manifest_sha256']!=hashes['manifest.json']
            or verification['audit_sha256']!=hashes['audit.json']
            or verification['summary_sha256']!=hashes['summary.json']
            or verification['process_terminal_sha256']!=hashes['process-terminal.json']
            or verification['completion_sha256']!=hashes['completion-event.json']):
        raise ValueError('Complete successful four-seed confirmation and actual terminal required')
    return {str(out/name):h for name,h in hashes.items()} | {
        spec['confirmation_terminal_verification']:sha(spec['confirmation_terminal_verification'])}


def current_reference(spec):
    current=read(Path(spec['main_root'])/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(current.get(k)!=v for k,v in spec['platform_reference'].items()):
        raise ValueError('Latest platform reference changed before activation')


def official_frames(spec):
    import pandas as pd
    from bf_tap_r2.v5_library import load_v5_training_frame
    from bf_tap_r2.v2_release import load_v2
    main=Path(spec['main_root']);training=load_v5_training_frame(main)
    query=load_v2(main/'复赛_test','test',322)[['sample_id','spout_no',*FEATURES]].copy()
    ids=pd.read_csv(main/'复赛_test/result_template.csv',dtype={'sample_id':str}).sample_id.tolist()
    if len(training)!=2754 or len(query)!=322 or query.sample_id.tolist()!=ids:
        raise ValueError('Official full training/query/template identity changed')
    return training,query


def input_bindings(spec):
    files=dict(spec['old_inputs'])
    for stage in ('train','test'):
        for kind in ('samples','features'):
            path=Path(spec['main_root'])/f'复赛_{stage}/{stage}_{kind}.csv'
            files[str(path)]=sha(path)
    template=Path(spec['main_root'])/'复赛_test/result_template.csv';files[str(template)]=sha(template)
    return files


def zip_payload(path,expected_sha,ids):
    from bf_tap_r2.submission import validate_result
    if sha(path)!=expected_sha:raise ValueError('Scored parent ZIP changed')
    with zipfile.ZipFile(path) as archive:
        if archive.namelist()!=['result.csv'] or archive.testzip() is not None:
            raise ValueError('Parent archive member/CRC changed')
        payload=archive.read('result.csv')
    validate_result(payload,ids)
    return payload


def release_payload(parent,ids,old,new):
    from bf_tap_r2.submission import validate_result
    from bf_tap_r2.v5_package import payload_with_parent_other_column
    rows=validate_result(parent,ids)
    old,new=np.asarray(old,float),np.asarray(new,float)
    if old.shape!=(len(ids),) or new.shape!=old.shape or not all(np.isfinite(v).all() for v in (old,new)):
        raise ValueError('Aligned finite original-unit component vectors required')
    values=np.asarray([float(r['pred_tap_time_len']) for r in rows])+.75*(new-old)
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Invalid affine SHORT replacement; no clipping allowed')
    result=payload_with_parent_other_column(ids,'tap_time_len',values,[r['pred_tap_iron'] for r in rows])
    actual=validate_result(result,ids)
    if [r['pred_tap_iron'] for r in actual]!=[r['pred_tap_iron'] for r in rows]:
        raise ValueError('Parent iron field strings changed')
    np.testing.assert_array_equal([float(r['pred_tap_time_len']) for r in actual],values)
    return result


def prepare(workspace,out,tests,authorization=None):
    workspace=Path(workspace).resolve();out=Path(out).resolve();spec=read(workspace/SPEC)
    confirmation=read(workspace/'configs/ema_span_confirmation/SPEC.json')
    if sha(workspace/'configs/ema_span_confirmation/SPEC.json')!=spec['confirmation_spec_sha256']:
        raise ValueError('Original frozen confirmation specification changed')
    validate_scope(spec,confirmation);current_reference(spec)
    if not out.is_relative_to(Path(spec['main_root'])/'local/runs') or out.exists():
        raise ValueError('Fresh private release directory required')
    source=sources(workspace);receipt=read(tests)
    if receipt.get('status')!='passed' or receipt.get('release_sources')!=source or receipt.get('full_suite') is not True:
        raise ValueError('Exact-source locked full release checks required')
    versions=runtime(spec);verify_files(spec['old_inputs'])
    grant=explicit_authority(authorization)
    evidence=confirmation_evidence(spec) if grant else {}
    training,query=official_frames(spec);part=partition(training,query,spec['training'])
    old=Path(spec['old_ema_directory']);fit=read(old/'fit.json')
    if fit['metadata']['fit_ids_digest']!=part['training'] or fit['metadata']['fit_rows']!=2754:
        raise ValueError('Old full EMA and new full training identities differ')
    import pandas as pd
    cached_query=pd.read_pickle(old/'query.pkl')
    if digest(cached_query.to_dict(orient='list'))!=part['query_frame']:
        raise ValueError('Old and new full query features/order differ')
    zip_payload(spec['parent_zip'],spec['platform_reference']['zip_sha256'],query.sample_id.tolist())
    available=require_memory(spec)
    inputs=input_bindings(spec)|evidence
    if grant:inputs[grant['path']]=grant['sha256']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()
    if grant:
        tracked=set(subprocess.check_output(['git','ls-files'],cwd=workspace,text=True).splitlines())
        if not set(source)<=tracked:
            raise ValueError('Committed scientific release source required')
        subprocess.run(['git','diff','--exit-code','HEAD','--',*source],cwd=workspace,check=True,capture_output=True)
    models={str(workspace/p):h for p,h in source.items()}
    models.update(binding_sources(reference_bindings()))
    out.mkdir(parents=True,exist_ok=False)
    with (out/'query.pkl').open('xb') as stream:query.to_pickle(stream)
    manifest=dict(spec=spec,workspace=str(workspace),output=str(out),source_commit=commit,sources=source,inputs=inputs,
        model_sources=models,
        versions=versions,partition=part,query_sha256=sha(out/'query.pkl'),authorization=grant,
        scientific_execution_admitted=grant is not None,tests_receipt_sha256=sha(tests),
        available_memory_mib=available,created_ns=time.time_ns())
    write_new(out/'manifest.json',manifest)
    return manifest


def context(out,*,require_activation=False,cold=False):
    out=Path(out).resolve();manifest=read(out/'manifest.json');spec=manifest['spec']
    workspace=Path(manifest['workspace'])
    if manifest['output']!=str(out) or not out.is_relative_to(Path(spec['main_root'])/'local/runs'):
        raise ValueError('Bound private release identity changed')
    verify_files({str(workspace/p):h for p,h in manifest['sources'].items()})
    if spec!=read(workspace/SPEC):raise ValueError('Frozen release specification differs from committed source')
    verify_files(manifest['model_sources'])
    validate_scope(spec,read(workspace/'configs/ema_span_confirmation/SPEC.json'))
    verify_files({p:h for p,h in manifest['inputs'].items() if not cold or not p.lower().endswith('.csv')})
    runtime(spec)
    if sha(out/'query.pkl')!=manifest['query_sha256']:raise ValueError('Frozen query changed')
    if require_activation:
        grant=manifest['authorization']
        if manifest['scientific_execution_admitted'] is not True or grant is None:
            raise ValueError('Engineering preparation does not authorize full training or packaging')
        explicit_authority(grant['path'])
        if read(out/'activation.json')['manifest_sha256']!=sha(out/'manifest.json'):
            raise ValueError('Bound explicit release activation required')
    return manifest


def peak(spec):
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if value>spec['max_worker_rss_mib']:raise ValueError('Original release memory gate failed')
    return value


def activated_context(out,*,cold=False):
    manifest=context(out,require_activation=True,cold=cold)
    if (Path(out)/'failure.json').exists():raise ValueError('Failed release evidence retained; no retry')
    return manifest


def warm_worker(out):
    from bf_tap_r2.ema_span_confirmation_models import fit_component
    out=Path(out);manifest=activated_context(out);spec=manifest['spec']
    training,query=official_frames(spec)
    if partition(training,query,spec['training'])!=manifest['partition']:
        raise ValueError('Frozen full/inner training and query partitions changed')
    # -1 is a declared full-training sentinel, never an outer split allocation.
    identity=dict(source_directory=str(out.resolve()),split_seed=spec['full_scope_identity_seed'],
        fold=spec['full_scope_identity_fold'],trial_id='SHORT_SPAN_FULL')
    source_hashes=manifest['model_sources']
    prediction,receipt=fit_component(out/'SHORT_SPAN_FULL',training,query,identity=identity,
        settings=spec['training'],mechanisms=dict(spec['mechanisms'],ema_beta=spec['candidate_beta']),
        source_hashes=source_hashes)
    if receipt['native_counts']!={'catboost_fit':0,'ebm_fit':0,'ebm_boost':0,'sklearn_mlp_fit':0,'torch_optimizer':2}:
        raise ValueError('Full SHORT native budget changed')
    activated_context(out)
    write_new(out/'warm-complete.json',dict(status='passed',pid=os.getpid(),
        manifest_sha256=sha(out/'manifest.json'),component_receipt_sha256=sha(out/'SHORT_SPAN_FULL/complete.json'),
        peak_rss_mib=peak(spec),full_training_procedures=1,new_CV=0,new_confirmation_seeds=0))
    return prediction


def cold_worker(out,receipt):
    from bf_tap_r2.ema_span_confirmation_models import audit_component
    out=Path(out);manifest=activated_context(out,cold=True)
    if sha(out/'warm-complete.json')!=receipt:raise ValueError('External full warm receipt changed')
    warm=read(out/'warm-complete.json')
    if warm['pid']==os.getpid() or warm['manifest_sha256']!=sha(out/'manifest.json'):
        raise ValueError('Independent cold process and original manifest required')
    report=audit_component(out/'SHORT_SPAN_FULL',warm['component_receipt_sha256'])
    if report['retained_states']!=2:raise ValueError('Two SHORT selected/refit states required')
    write_new(out/'cold-complete.json',dict(status='passed',pid=os.getpid(),
        warm_receipt_sha256=receipt,component_cold_receipt_sha256=sha(out/'SHORT_SPAN_FULL/cold-complete.json'),
        peak_rss_mib=peak(manifest['spec']),new_fits=0,training_csv_reads=0))


def native_audit(out):
    from bf_tap_r2.component_regularization_audit import verify_saved
    from bf_tap_r2.v3_4_bags import group_safe_inner_folds
    out=Path(out);manifest=activated_context(out);spec=manifest['spec']
    warm=read(out/'warm-complete.json');cold=read(out/'cold-complete.json')
    unit=out/'SHORT_SPAN_FULL';component=read(unit/'complete.json')
    if (cold['status']!='passed' or cold['warm_receipt_sha256']!=sha(out/'warm-complete.json')
            or cold['component_cold_receipt_sha256']!=sha(unit/'cold-complete.json')
            or warm['component_receipt_sha256']!=sha(unit/'complete.json')
            or warm['pid']==os.getpid()):raise ValueError('Independent closed SHORT cold receipt required')
    training,query=official_frames(spec)
    if partition(training,query,spec['training'])!=manifest['partition']:
        raise ValueError('Frozen native audit training/inner/query partitions changed')
    mask=np.asarray(group_safe_inner_folds(training,seed=spec['training']['inner_seed'])['fold'])!=0
    y=training[['tap_time_len']].to_numpy();mechanisms=dict(spec['mechanisms'],ema_beta=spec['candidate_beta'])
    for phase,item in component['checkpoints'].items():
        if sha(unit/(phase+'.pt'))!=item['sha256']:raise ValueError('Native SHORT checkpoint changed')
    # No active ledger: any numerical solver/Optimizer entry is rejected.
    # Original preprocessing statistics are independently recomputed by the auditor.
    with native_hooks(reference_bindings()):
        selection=verify_saved(unit/'selection.pt',training.loc[mask].reset_index(drop=True),y[mask],
            'EMA',spec['training'],mechanisms,training.loc[~mask])
        refit=verify_saved(unit/'refit.pt',training,y,'EMA',spec['training'],mechanisms,
            expected_epoch=selection.saved['trace']['selected_epoch'])
    for phase,state in [('selection',selection),('refit',refit)]:
        if state.saved['trace']!=component['model_metadata']['traces'][phase]:
            raise ValueError('Original selection/refit trace differs from saved native state')
    with np.load(unit/'predictions.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(refit.predict(query),saved['prediction'])
        np.testing.assert_array_equal(saved['query_ids'],query.sample_id.to_numpy(str))
    activated_context(out)
    write_new(out/'native-audit.json',dict(status='passed',pid=os.getpid(),retained_states=2,
        manifest_sha256=sha(out/'manifest.json'),cold_receipt_sha256=sha(out/'cold-complete.json'),
        component_receipt_sha256=sha(unit/'complete.json'),selected_epoch=selection.saved['trace']['selected_epoch'],
        peak_rss_mib=peak(spec),new_scientific_fits=0))


def model_payload(manifest,out):
    import pandas as pd
    from bf_tap_r2.component_regularization import ComponentRegressor
    out=Path(out);spec=manifest['spec'];query=pd.read_pickle(out/'query.pkl')
    if any(t in query for t in TARGETS):raise ValueError('Cold release query contains labels')
    ids=query.sample_id.tolist()
    parent=zip_payload(spec['parent_zip'],spec['platform_reference']['zip_sha256'],ids)
    with cold_only():
        old_model=ComponentRegressor.load(Path(spec['old_ema_directory'])/'refit.pt')
        new_model=ComponentRegressor.load(out/'SHORT_SPAN_FULL/refit.pt')
        if (old_model.settings!=spec['training'] or old_model.mechanisms!=spec['mechanisms']
                or new_model.settings!=spec['training']
                or new_model.mechanisms!=dict(spec['mechanisms'],ema_beta=spec['candidate_beta'])):
            raise ValueError('Release old/new scientific recipe changed')
        old=old_model.predict(query);new=new_model.predict(query)
        if old.shape!=(322,1) or new.shape!=old.shape:raise ValueError('Single-output 322-row full time models required')
        np.testing.assert_array_equal(old,np.load(Path(spec['old_ema_directory'])/'cold.npy',allow_pickle=False))
        with np.load(out/'SHORT_SPAN_FULL/predictions.npz',allow_pickle=False) as saved:
            np.testing.assert_array_equal(new,saved['prediction'])
            np.testing.assert_array_equal(saved['query_ids'],query.sample_id.to_numpy(str))
        result=release_payload(parent,ids,old[:,0],new[:,0])
    return result,ids


def require_native_audit(out):
    out=Path(out);audit=read(out/'native-audit.json')
    if (audit['status']!='passed' or audit['retained_states']!=2
            or audit['manifest_sha256']!=sha(out/'manifest.json')
            or audit['cold_receipt_sha256']!=sha(out/'cold-complete.json')
            or audit['component_receipt_sha256']!=sha(out/'SHORT_SPAN_FULL/complete.json')):
        raise ValueError('Independent native checkpoint/partition audit required')
    return audit


def build(out):
    from bf_tap_r2.submission import package,ZIP_NAME
    out=Path(out);manifest=activated_context(out,cold=True);audit=require_native_audit(out)
    payload,ids=model_payload(manifest,out)
    destination=out/'package';destination.mkdir(exist_ok=False)
    package(destination,payload,ids)
    with zipfile.ZipFile(destination/ZIP_NAME) as archive:
        if archive.namelist()!=['result.csv'] or archive.testzip() is not None or archive.read('result.csv')!=payload:
            raise ValueError('New SHORT archive readback failed')
    write_new(out/'release.json',dict(status='package_built_independent_verification_pending',pid=os.getpid(),
        manifest_sha256=sha(out/'manifest.json'),native_audit_sha256=sha(out/'native-audit.json'),
        zip_sha256=sha(destination/ZIP_NAME),csv_sha256=sha(destination/'result.csv'),rows=len(ids),
        unchanged_iron_string_mismatches=0,selected_epoch=audit['selected_epoch'],
        full_training_procedures=1,torch_optimizer=2,new_CV=0,new_confirmation_seeds=0,
        peak_rss_mib=peak(manifest['spec']),desktop_writes=0,agent_uploads=0,platform_score=None))


def verify_package(out,receipt):
    from bf_tap_r2.submission import ZIP_NAME
    out=Path(out);manifest=activated_context(out,cold=True);require_native_audit(out)
    if sha(out/'release.json')!=receipt:raise ValueError('External package receipt changed')
    released=read(out/'release.json')
    if (released['pid']==os.getpid() or released['manifest_sha256']!=sha(out/'manifest.json')
            or released['native_audit_sha256']!=sha(out/'native-audit.json')):
        raise ValueError('Independent package verifier/source identity required')
    payload,ids=model_payload(manifest,out);archive_path=out/'package'/ZIP_NAME
    if sha(archive_path)!=released['zip_sha256'] or sha(out/'package/result.csv')!=released['csv_sha256']:
        raise ValueError('Released SHORT CSV/ZIP identity changed')
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist()!=['result.csv'] or archive.testzip() is not None or archive.read('result.csv')!=payload:
            raise ValueError('Independent SHORT arithmetic or archive differs')
    if (out/'package/result.csv').read_bytes()!=payload:raise ValueError('Standalone CSV differs from independent payload')
    write_new(out/'package-audit.json',dict(status='passed',G0='passed',G1='four_seed_confirmed_platform_untested',
        release_receipt_sha256=receipt,manifest_sha256=sha(out/'manifest.json'),pid=os.getpid(),rows=len(ids),
        arithmetic_difference=0,unchanged_iron_string_mismatches=0,training_csv_reads=0,new_fits=0,
        peak_rss_mib=peak(manifest['spec']),desktop_writes=0,agent_uploads=0,platform_score=None))


def execute(out):
    out=Path(out);manifest=context(out);spec=manifest['spec']
    if manifest['scientific_execution_admitted'] is not True or manifest['authorization'] is None:
        raise ValueError('Engineering preparation does not authorize full training or packaging')
    explicit_authority(manifest['authorization']['path']);confirmation_evidence(spec);current_reference(spec)
    require_memory(spec)
    write_new(out/'activation.json',dict(manifest_sha256=sha(out/'manifest.json'),pid=os.getpid(),started_ns=time.time_ns()))
    command=[sys.executable,str(Path(manifest['workspace'])/'scripts/ema_short_release.py'),'--output',str(out)]
    try:
        for mode,receipt in [('warm',None),('cold','warm-complete.json'),('audit',None),('build',None),('verify','release.json')]:
            args=command+[mode]
            if receipt:args+=['--receipt',sha(out/receipt)]
            subprocess.run(args,cwd=manifest['workspace'],check=True)
        write_new(out/'completion-event.json',dict(status='completed',completed_ns=time.time_ns(),
            package_audit_sha256=sha(out/'package-audit.json'),release_sha256=sha(out/'release.json'),
            manifest_sha256=sha(out/'manifest.json')))
    except BaseException as error:
        write_new(out/'failure.json',dict(error=repr(error),automatic_retry=False,evidence_preserved=True));raise


def observe(out):
    out=Path(out);manifest=context(out)
    if manifest['scientific_execution_admitted'] is not True:
        raise ValueError('Engineering preparation cannot launch a scientific controller')
    command=[sys.executable,str(Path(manifest['workspace'])/'scripts/ema_short_release.py'),'--output',str(out),'execute']
    started=time.monotonic();next_check=started+600
    with (out/'observer.log').open('x') as stream:
        process=subprocess.Popen(command,cwd=manifest['workspace'],stdout=stream,stderr=subprocess.STDOUT)
        write_new(out/'process-launch.json',dict(pid=process.pid,observer_pid=os.getpid(),started_ns=time.time_ns(),command=command))
        while True:
            try:code=process.wait(timeout=min(60,max(.01,next_check-time.monotonic())))
            except subprocess.TimeoutExpired:
                if time.monotonic()<next_check:continue
                record=dict(event='scheduled_600_second_check',pid=process.pid,elapsed_seconds=time.monotonic()-started)
                with (out/'monitor.jsonl').open('a') as log:
                    log.write(json.dumps(record)+'\n');log.flush();os.fsync(log.fileno())
                print(json.dumps(record),flush=True);next_check+=600;continue
            terminal=dict(event='actual_controller_completion',pid=process.pid,observer_pid=os.getpid(),exit_code=code,completed_ns=time.time_ns())
            write_new(out/'process-terminal.json',terminal);print(json.dumps(terminal),flush=True);return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','execute','observe','warm','cold','audit','build','verify'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--tests',type=Path)
    parser.add_argument('--authorization',type=Path);parser.add_argument('--receipt')
    args=parser.parse_args()
    if args.mode=='prepare':
        if args.tests is None:parser.error('--tests required for preparation')
        prepare(WORKSPACE,args.output,args.tests,args.authorization)
    elif args.mode in ('cold','verify'):
        if args.receipt is None:parser.error('--receipt required for independent stage')
        {'cold':cold_worker,'verify':verify_package}[args.mode](args.output,args.receipt)
    elif args.mode=='observe':return observe(args.output)
    else:{'execute':execute,'warm':warm_worker,'audit':native_audit,'build':build}[args.mode](args.output)
    return 0


if __name__=='__main__':raise SystemExit(main())
