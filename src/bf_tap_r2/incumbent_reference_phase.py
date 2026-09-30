"""Frozen, bounded current-reference completion, followed by zero-fit audit.

No quality selection, platform package or full-data training is performed.
CLI execution requires committed exact-source locked tests and external hashes.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import subprocess
import sys

import numpy as np
import psutil
import yaml

from .data import FEATURES,TARGETS
from .incumbent_native import load_reference_cache
from .incumbent_reference_cache import (inspect_development,audit_development,verify_hashes,safe_child)
from .incumbent_reference_columns import Member,assemble_columns,SPLIT_SEEDS,TRAINING_SEEDS,ENDPOINT
from .incumbent_reference_core import execute_estimator,audit_estimator
from .incumbent_reference_ledger import ReservationLedger,file_hash,write_new
from .v7_periodic import digest

SPEC='configs/incumbent_de3_reference/SPEC.yaml'
LIMITS=dict(estimator=20,optimizer=40)
TASKS=tuple((s,f,t) for s in (7777,12011) for f in range(5) for t in (104729,130363))
SHARED_MODULES=('data','component_regularization','component_regularization_audit','component_regularization_run',
    'v12_joint','v7_periodic','v3_4_bags','v3_6_networks','v5_library','v5_spec','v5_replicate','v17_run','v7_confirm','v9_confirm')
RESOURCE=dict(workers=4,max_projected_seconds=7200,max_worker_rss_mib=1536,
              step_safety_multiplier=1.5,peak_safety_multiplier=1.5,fixed_overhead_seconds=300,free_memory_margin_mib=1024)


def read_json(path):return json.loads(Path(path).read_text())


def anchored(path,sha):
    if file_hash(path)!=sha:raise ValueError('External manifest/evidence anchor differs')
    return read_json(path)


def require_threads():
    if sys.version_info[:2]!=(3,12):raise ValueError('Locked Python 3.12 required')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('Set '+name+'=1')


def source_snapshot(workspace):
    root=Path(workspace)
    paths=[*(root/'src').rglob('*.py'),*(root/'tests').rglob('*.py'),*(root/'configs').rglob('*.yaml'),
           root/'uv.lock',root/'pyproject.toml',root/'scripts/run_incumbent_reference.py']
    return {str(p.relative_to(root)):file_hash(p) for p in sorted(paths)}


def runtime_snapshot():
    return dict(python=sys.version,packages={name:importlib.metadata.version(name) for name in
        ('torch','tabm','rtdl-num-embeddings','numpy','pandas','scikit-learn','scipy','PyYAML','psutil')})


def private_output(workspace,output):
    root=Path(workspace).resolve();p=Path(output).resolve()
    if not p.is_relative_to(root/'local') or p==root/'local':raise ValueError('Private isolated output required')
    return p


def validate_spec(spec,settings):
    if (spec['platform_reference']['candidate']!='DE3_IRON_USER_REQUESTED'
            or spec['platform_reference']['score']!=96.3749 or spec['parent']!='V32_TIME_A60V7_50'
            or spec['endpoint']!=ENDPOINT or spec['training_seeds']!=list(TRAINING_SEEDS)
            or spec['split_seeds']!=list(SPLIT_SEEDS) or spec['folds']!=5
            or spec['purpose']!='incumbent_reference_only_not_DE3_promotion'
            or spec['planned_budget_not_yet_reserved']!=dict(estimators=20,optimizer_runs=40,reference_factory_fits=0,
                full_data_fits=0,packages=0,desktop_writes=0,agent_uploads=0)
            or spec['resources']!=RESOURCE or spec['phase_admission']['implemented'] is not True
            or spec['monitoring']!=dict(future_interval_seconds=600,between_check_polling=False)):
        raise ValueError('Frozen incumbent scope/budget/resource contract differs')
    if spec['missing_members']!=dict(split_seeds=[7777,12011],training_seeds=[104729,130363],
            target_output='tap_iron',estimator_outputs=list(TARGETS),optimization=False):
        raise ValueError('Only the missing current-reference members may be fitted')
    if (spec['native_recipe']!=dict(original_spec='configs/strong_component_regularization/SPEC.yaml',
            settings_key='training.tap_iron',permitted_difference='training_random_seed_only',arm='BASE',mechanisms={},
            recipe=dict(backbone='tabm',frequency=.01),inner_seed=42,epoch_selection='mean_standardized_mae_both_outputs')
            or spec['core_audit']!=dict(full_batch_atol=0,order_chunk_absolute_atol=.0005,
                order_chunk_relative_atol=.000001,zero_optimizer_calls=True)):
        raise ValueError('Frozen native recipe/audit bounds differ')
    original=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/strong_component_regularization/SPEC.yaml').read_text())
    if settings!=original['training']['tap_iron']:raise ValueError('Original native settings changed')


def resource_admission(evidence,settings,available_mib):
    witnesses=evidence['cost_witnesses']
    if len(witnesses)!=10 or any(not all(np.isfinite(v) and v>0 for v in w.values()) for w in witnesses):
        raise ValueError('Ten finite positive historical cost witnesses required')
    per_step=max(w['seconds']/w['optimizer_updates'] for w in witnesses)
    worst_updates=2*math.ceil(2204/settings['batch_size'])*settings['max_epochs']
    seconds=20*worst_updates*per_step/RESOURCE['workers']*RESOURCE['step_safety_multiplier']+RESOURCE['fixed_overhead_seconds']
    rss=max(w['peak_rss_mib'] for w in witnesses)*RESOURCE['peak_safety_multiplier']
    memory_required=RESOURCE['workers']*RESOURCE['max_worker_rss_mib']+RESOURCE['free_memory_margin_mib']
    passed=seconds<=RESOURCE['max_projected_seconds'] and rss<=RESOURCE['max_worker_rss_mib'] and available_mib>=memory_required
    return dict(status='passed' if passed else 'failed',projected_seconds=seconds,projected_peak_worker_rss_mib=rss,
        available_mib=available_mib,required_available_mib=memory_required,maximum_seconds_per_optimizer_update=per_step,
        worst_optimizer_updates_per_estimator=worst_updates,source='same_native_recipe_complete_development_trace',
        new_probe_fits=0,limits=RESOURCE)


def native_members(reference_root,frame,folds,native_audit):
    root=Path(reference_root);j42={};members=[]
    for seed in SPLIT_SEEDS:
        j42[seed]=np.full(len(frame),np.nan)
        for fold in range(5):
            name=(f'local/runs/round2-v12-joint-tabm/development-r1/joint-joint_plr001-s{seed}-f{fold}.npy'
                if seed in (42,3407) else f'local/runs/round2-v12-joint-tabm/confirmation-r1/seed-{seed}-fold-{fold}.npy')
            if name not in native_audit['hashes'] or file_hash(root/name)!=native_audit['hashes'][name]:
                raise ValueError('Native J42 prediction not bound to original audit')
            values=np.load(root/name,allow_pickle=False);mask=folds[seed]==fold
            if values.shape!=(int(mask.sum()),2) or not np.isfinite(values).all():raise ValueError('Incomplete native joint reference')
            j42[seed][mask]=values[:,0]
            if seed in (7777,12011):members.append(Member(seed,fold,42,tuple(frame.loc[mask,'sample_id']),values[:,0]))
    return j42,members


def reference_identity(frame,folds,parent,audit):
    return dict(row_ids_digest=digest(frame.sample_id.tolist()),native_data_digest=audit['data_digest'],
        fold_digests={str(s):digest(folds[s].tolist()) for s in SPLIT_SEEDS},native_audit=audit,
        parent_digests={str(s):{t:hashlib.sha256(np.asarray(p,float).tobytes()).hexdigest() for t,p in parent[s].items()} for s in SPLIT_SEEDS})


def check_shared_sources(workspace,roots):
    for root in roots:
        for name in SHARED_MODULES:
            path=Path('src/bf_tap_r2')/(name+'.py')
            if (Path(workspace)/path).read_bytes().replace(b'\r\n',b'\n')!=(Path(root)/path).read_bytes().replace(b'\r\n',b'\n'):
                raise ValueError('Executing native source differs from historical source: '+name)


def prepare(workspace,reference_root,development_source_root,development_cache,output,tests_path,tests_sha256):
    require_threads();workspace=Path(workspace).resolve();output=private_output(workspace,output)
    if Path(__file__).resolve()!=workspace/'src/bf_tap_r2/incumbent_reference_phase.py':raise ValueError('Imported workspace differs')
    sources=source_snapshot(workspace);runtime=runtime_snapshot();tests=anchored(tests_path,tests_sha256)
    if (tests.get('status')!='passed' or tests.get('exit_code')!=0 or tests.get('full_suite') is not True
            or tests.get('source_hashes')!=sources or tests.get('runtime')!=runtime):
        raise ValueError('Passed full locked tests for exact phase sources/runtime required')
    changed=subprocess.check_output(['git','status','--porcelain','--',*sources],cwd=workspace,text=True)
    if changed.strip():raise ValueError('Commit and push validated phase before freeze')
    spec=yaml.safe_load((workspace/SPEC).read_text())
    anchors={name:spec['development_cache'][key] for name,key in
        [('manifest.json','manifest_sha256'),('summary.json','summary_sha256'),('audit.json','audit_sha256')]}
    evidence=inspect_development(development_source_root,development_cache,anchors)
    settings=evidence['original_settings'];validate_spec(spec,settings)
    check_shared_sources(workspace,[reference_root,development_source_root])
    output.mkdir(parents=True,exist_ok=False);write_new(output/'freeze.started.json',dict(workspace=str(workspace)))
    try:
        admission=resource_admission(evidence,settings,psutil.virtual_memory().available/2**20)
        write_new(output/'resource-admission.json',admission)
        if admission['status']!='passed':raise ValueError('Native reference resource admission failed')
        frame,folds,parent,_,native_audit=load_reference_cache(reference_root)
        j42,_=native_members(reference_root,frame,folds,native_audit)
        _,reused_audit=audit_development(development_cache,evidence,frame,folds,j42)
        manifest=dict(version=1,experiment='incumbent_de3_reference_completion',workspace=str(workspace),output=str(output),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip(),
            reference_root=str(Path(reference_root).resolve()),development_source_root=str(Path(development_source_root).resolve()),
            development_cache=str(Path(development_cache).resolve()),source_hashes=sources,runtime=runtime,spec=spec,settings=settings,
            development_evidence=evidence,pre_fit_reused_audit=reused_audit,reference=reference_identity(frame,folds,parent,native_audit),
            admission_sha256=file_hash(output/'resource-admission.json'),tasks=[list(k) for k in TASKS],limits=LIMITS,
            tests=dict(path=str(Path(tests_path).resolve()),sha256=tests_sha256),release_authorized=False)
        write_new(output/'manifest.json',manifest)
        write_new(output/'freeze.complete.json',dict(manifest_sha256=file_hash(output/'manifest.json')))
    except BaseException as exc:
        write_new(output/'freeze.failed.json',dict(type=type(exc).__name__,message=str(exc)));raise
    return dict(path=str(output/'manifest.json'),sha256=file_hash(output/'manifest.json'))


def verify_manifest(path,sha):
    require_threads();m=anchored(path,sha);workspace=Path(m['workspace']);output=private_output(workspace,m['output'])
    if (Path(path).resolve()!=output/'manifest.json' or Path(__file__).resolve()!=workspace/'src/bf_tap_r2/incumbent_reference_phase.py'
            or m['experiment']!='incumbent_de3_reference_completion' or m['release_authorized'] is not False
            or m['tasks']!=[list(k) for k in TASKS] or m['limits']!=LIMITS):raise ValueError('Frozen phase scope differs')
    if source_snapshot(workspace)!=m['source_hashes'] or runtime_snapshot()!=m['runtime']:raise ValueError('Phase sources/runtime changed')
    spec=yaml.safe_load((workspace/SPEC).read_text());validate_spec(spec,m['settings'])
    if spec!=m['spec']:raise ValueError('Phase spec changed')
    tests=anchored(m['tests']['path'],m['tests']['sha256'])
    if tests['source_hashes']!=m['source_hashes'] or tests['runtime']!=m['runtime']:raise ValueError('Frozen test evidence differs')
    verify_hashes(m['reference_root'],m['reference']['native_audit']['hashes'],shared_runs=True)
    verify_hashes(m['development_cache'],m['development_evidence']['hashes'])
    # Recheck original source/data/runtime metadata; no training labels read here.
    anchors={name:m['spec']['development_cache'][key] for name,key in
        [('manifest.json','manifest_sha256'),('summary.json','summary_sha256'),('audit.json','audit_sha256')]}
    if inspect_development(m['development_source_root'],m['development_cache'],anchors)!=m['development_evidence']:
        raise ValueError('Original development evidence changed')
    check_shared_sources(workspace,[m['reference_root'],m['development_source_root']])
    admission=anchored(output/'resource-admission.json',m['admission_sha256'])
    if admission['status']!='passed' or admission['limits']!=RESOURCE:raise ValueError('Passed resource admission required')
    return m


def reload_context(m):
    frame,folds,parent,_,audit=load_reference_cache(m['reference_root'])
    if reference_identity(frame,folds,parent,audit)!=m['reference']:raise ValueError('Frozen native rows/folds/columns differ')
    j42,native=native_members(m['reference_root'],frame,folds,audit)
    return frame,folds,parent,j42,native


def partitions(frame,fv,fold):
    return (frame.loc[fv!=fold].reset_index(drop=True),
            frame.loc[fv==fold,['sample_id','spout_no',*FEATURES]].reset_index(drop=True))


def execute_jobs(frame,folds,settings,output,ledger):
    """Twenty fixed tasks, spawn processes, at most four in-flight estimators."""
    output=Path(output);(output/'estimators').mkdir(exist_ok=False)
    anchors={}
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={};iterator=iter(TASKS)
        def submit_next():
            try:key=next(iterator)
            except StopIteration:return False
            seed,fold,training_seed=key;training,query=partitions(frame,folds[seed],fold)
            name=f'estimators/s{seed}-f{fold}-t{training_seed}'
            future=pool.submit(execute_estimator,training,query,dict(settings,random_seed=training_seed),key,
                               output/name,ledger.root,ledger.policy_sha256)
            pending[future]=name;return True
        for _ in range(4):submit_next()
        while pending:
            future=next(as_completed(pending));name=pending.pop(future)
            try:anchors[name]=future.result()
            except BaseException:
                for f in pending:f.cancel()
                raise
            submit_next()
    if len(anchors)!=20:raise ValueError('Incomplete fixed reference task coverage')
    return anchors


def execute(path,sha):
    m=verify_manifest(path,sha);out=Path(m['output'])
    write_new(out/'run.started.json',dict(manifest_sha256=sha))
    try:
        if psutil.virtual_memory().available/2**20<4*1536+1024:raise ValueError('Insufficient current available memory')
        frame,folds,_,_,_=reload_context(m)
        ledger=ReservationLedger.create(out/'ledger',LIMITS)
        write_new(out/'ledger-anchor.json',dict(policy_sha256=ledger.policy_sha256))
        anchors=execute_jobs(frame,folds,m['settings'],out,ledger)
        counts=ledger.inspect()
        if counts['started']!=LIMITS or counts['completed']!=LIMITS or any(counts['failed'].values()) or any(counts['incomplete'].values()):
            raise ValueError('Incomplete reference budget execution')
        write_new(out/'run.finished.json',dict(manifest_sha256=sha,anchors=anchors,counts=counts))
    except BaseException as exc:
        write_new(out/'run.failed.json',dict(type=type(exc).__name__,message=str(exc)));raise
    return dict(path=str(out/'run.finished.json'),sha256=file_hash(out/'run.finished.json'))


def audit_new_estimators(frame,folds,settings,output,ledger,anchors):
    expected={f'estimators/s{s}-f{f}-t{t}' for s,f,t in TASKS}
    if set(anchors)!=expected:raise ValueError('External estimator anchors incomplete')
    members=[];audits=[]
    for seed,fold,ts in TASKS:
        name=f'estimators/s{seed}-f{fold}-t{ts}';key=(seed,fold,ts);anchor=anchors[name]
        if anchor['key']!=list(key):raise ValueError('Estimator anchor key differs')
        training,query=partitions(frame,folds[seed],fold)
        pred,report=audit_estimator(Path(output)/name,training,query,dict(settings,random_seed=ts),key,
            anchor['complete_sha256'],ledger.root,ledger.policy_sha256)
        if max(read_json(Path(output)/name/'complete.json')['metadata']['traces'][p]['peak_rss_mib'] for p in ('selection','refit'))>1536:
            raise ValueError('Observed reference memory exceeds frozen worker cap')
        members.append(Member(seed,fold,ts,tuple(query.sample_id),pred[:,0]));audits.append(report)
    return members,audits


def audit(path,sha,run_sha):
    m=verify_manifest(path,sha);out=Path(m['output']);run=anchored(out/'run.finished.json',run_sha)
    if run['manifest_sha256']!=sha:raise ValueError('Executed manifest differs')
    write_new(out/'audit.started.json',dict(manifest_sha256=sha,run_sha256=run_sha))
    try:
        frame,folds,parent,j42,native=reload_context(m)
        ledger=ReservationLedger.open(out/'ledger',read_json(out/'ledger-anchor.json')['policy_sha256']);before=ledger.inspect()
        if before!=run['counts'] or before['started']!=LIMITS or before['completed']!=LIMITS:raise ValueError('Executed ledger differs')
        reused,reused_audit=audit_development(m['development_cache'],m['development_evidence'],frame,folds,j42)
        new,audits=audit_new_estimators(frame,folds,m['settings'],out,ledger,run['anchors'])
        columns=assemble_columns(frame.sample_id.tolist(),folds,parent,[*reused,*native,*new]);iron_columns={}
        for seed in SPLIT_SEEDS:
            # Independently reassemble the delivered endpoint from the member
            # vectors; never use cross-split averages or held-out labels.
            average=np.full(len(frame),np.nan)
            library={(u.split_seed,u.fold,u.training_seed):u for u in [*reused,*native,*new]}
            for fold in range(5):
                mask=folds[seed]==fold
                average[mask]=np.mean([library[seed,fold,t].iron for t in TRAINING_SEEDS],axis=0)
            expected=np.maximum(parent[seed]['tap_iron']+.5*(average-j42[seed]),0)
            np.testing.assert_array_equal(columns[seed]['tap_iron'],expected)
            np.testing.assert_array_equal(columns[seed]['tap_time_len'],parent[seed]['tap_time_len'])
            name=f'iron-seed-{seed}.npy'
            with (out/name).open('xb') as stream:np.save(stream,columns[seed]['tap_iron'],allow_pickle=False)
            iron_columns[str(seed)]=dict(path=name,sha256=file_hash(out/name))
        if ledger.inspect()!=before:raise ValueError('Audit consumed an optimizer or altered ledger')
        # The audit subprocess still has its stdout log open. Bind completed
        # artifacts only; hashing that live log would invalidate the overlay.
        artifacts={str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file() and p.name!='audit.log'}
        report=dict(status='passed',candidate='DE3_IRON_USER_REQUESTED',verified_split_seeds=list(SPLIT_SEEDS),
            new_audit_fits=0,scope='incumbent_reference_only_not_DE3_promotion',artifact_hashes=artifacts,
            new_reference_estimators=20,new_optimizer_runs=40,new_cold_models=40,reused_development_audit=reused_audit,
            derived_J42_original_audit_reuse_units=10,new_model_audits=audits,full_batch_difference=0.,
            unchanged_parent_time=True,source_hashes_unchanged=True,ledger_counts_before=before,ledger_counts_after=ledger.inspect(),
            historical_DE3_no_finalist_unchanged=True,packages=0,agent_uploads=0)
        write_new(out/'audit.json',report)
        complete=dict(candidate='DE3_IRON_USER_REQUESTED',parent='V32_TIME_A60V7_50',endpoint=ENDPOINT,
            training_seeds=list(TRAINING_SEEDS),iron_columns=iron_columns,audit_sha256=file_hash(out/'audit.json'),
            row_ids_digest=m['reference']['row_ids_digest'],native_data_digest=m['reference']['native_data_digest'],
            fold_digests=m['reference']['fold_digests'],manifest_sha256=sha,run_sha256=run_sha)
        write_new(out/'complete.json',complete)
    except BaseException as exc:
        write_new(out/'audit.failed.json',dict(type=type(exc).__name__,message=str(exc)));raise
    return dict(path=str(out/'complete.json'),sha256=file_hash(out/'complete.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['execute','audit'],required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--run-sha256')
    args=p.parse_args()
    if args.phase=='audit' and not args.run_sha256:p.error('--run-sha256 required for audit')
    result=execute(args.manifest,args.sha256) if args.phase=='execute' else audit(args.manifest,args.sha256,args.run_sha256)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
