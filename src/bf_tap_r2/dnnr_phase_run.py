"""Bounded complete-coverage fitting and fresh saved-model phase audit."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import multiprocessing
import resource
import time

import numpy as np
import yaml

from .data import TARGETS
from .dnnr_model import Settings, ARMS
from .dnnr_phase_protocol import DEV, CONFIRM, phase_tasks, phase_limits, required_arms, task_name
from .dnnr_earned_unit import execute_earned, audit_earned
from .dnnr_phase_freeze import verify_manifest, reload_references
from .dnnr_preflight import anchored, private_path
from .dnnr_ledger import ReservationLedger, write_new, file_hash
from .dnnr_scoring import score_seed, decide_development, decide_confirmation
from .candidate_tiers import classify_candidates


def earned_context(root, phase, development_audit_sha256=None, development_arithmetic_sha256=None):
    if phase == 'development':
        if development_audit_sha256 is not None or development_arithmetic_sha256 is not None:raise ValueError('Development has no prior selector')
        return None,None
    if phase != 'confirmation' or development_audit_sha256 is None or development_arithmetic_sha256 is None:
        raise ValueError('Confirmation needs anchored independent development audit')
    report=anchored(Path(root)/'development/audit.json',development_audit_sha256)
    if report['status']!='passed' or report['phase']!='development' or report['new_audit_fits']!=0:
        raise ValueError('Passed independent development audit required')
    development=Path(root)/'development'
    if (development/'failed.json').exists() or (development/'audit-failed.json').exists() or (development/'arithmetic-failed.json').exists():
        raise ValueError('Failed development evidence cannot earn confirmation')
    complete=anchored(development/'complete.json',report['phase_complete_sha256'])
    if complete['phase']!='development' or complete['manifest_sha256']!=report['manifest_sha256']:
        raise ValueError('Development manifest identity differs')
    for name,sha in report['artifact_hashes'].items():
        path=development/name
        if not path.resolve().is_relative_to(development.resolve()) or file_hash(path)!=sha:
            raise ValueError('Audited development fitting artifact changed')
    decision=decide_development(report['records'])
    if decision!=report['decision']:raise ValueError('Development gate decision changed')
    # Confirmation needs both the cold model audit and independent vector/gate arithmetic.
    arithmetic=anchored(Path(root)/'development/arithmetic.json',development_arithmetic_sha256)
    if (arithmetic['status']!='passed' or arithmetic['phase']!='development' or arithmetic['new_fits']!=0
            or arithmetic['manifest_sha256']!=report['manifest_sha256']
            or arithmetic['audit_sha256']!=development_audit_sha256 or arithmetic['selected_pairs']!=decision['eligible_pairs']):
        raise ValueError('Confirmation requires independently verified gate arithmetic')
    return decision['eligible_pairs'],report


def partition(frame,folds,task):
    mask=folds[task['seed']]==task['fold']
    training=frame.loc[~mask].reset_index(drop=True)
    query=frame.loc[mask].drop(columns=list(TARGETS)).reset_index(drop=True)
    y=training[task['target']].to_numpy(dtype=np.float64)
    return training.drop(columns=list(TARGETS)),y,query


def worker(manifest_path,manifest_sha256,phase,task,eligible,policy_sha256,development_audit_sha256=None,development_arithmetic_sha256=None):
    m=verify_manifest(manifest_path,manifest_sha256)
    root=Path(manifest_path).resolve().parent
    earned,_=earned_context(root,phase,development_audit_sha256,development_arithmetic_sha256)
    if earned!=eligible or task not in phase_tasks(phase,earned):raise ValueError('Unqualified or undeclared worker task')
    output=root/phase
    started=json.loads((output/'started.json').read_text())
    if (started['manifest_sha256']!=manifest_sha256 or started['phase']!=phase
            or started['tasks']!=phase_tasks(phase,earned) or started['eligible_pairs']!=earned
            or started['development_audit_sha256']!=development_audit_sha256
            or started['development_arithmetic_sha256']!=development_arithmetic_sha256):
        raise ValueError('Worker phase identity differs')
    ledger=ReservationLedger.open(output/'ledger',policy_sha256)
    if ledger.limits!=phase_limits(phase,earned):raise ValueError('Worker ledger policy differs')
    frame,folds,_,_,_=reload_references(m)
    training,y,query=partition(frame,folds,task)
    result=execute_earned(task,training,y,query,output/'units'/task_name(task),output/'ledger',policy_sha256,
        required_arms(task['target'],phase,eligible),Settings(**m['settings']))
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    witness=output/'worker-resources'/f'{task_name(task)}.json'
    write_new(witness,dict(peak_mib=peak,maximum_mib=1536))
    if peak>1536:raise ValueError('Official DNNR worker exceeded admitted peak bound')
    verify_manifest(manifest_path,manifest_sha256)
    return task_name(task),result['complete_sha256'],file_hash(witness)


def run_phase(manifest_path,manifest_sha256,phase,development_audit_sha256=None,development_arithmetic_sha256=None):
    m=verify_manifest(manifest_path,manifest_sha256)
    root=private_path(m['workspace'],Path(manifest_path).resolve().parent)
    eligible,prior=earned_context(root,phase,development_audit_sha256,development_arithmetic_sha256)
    if phase=='confirmation' and not eligible:
        raise ValueError('No qualified candidate; no confirmation directory or fit may be created')
    tasks=phase_tasks(phase,eligible);output=root/phase
    output.mkdir(exist_ok=False);(output/'units').mkdir();(output/'worker-resources').mkdir()
    limits=phase_limits(phase,eligible);ledger=ReservationLedger.create(output/'ledger',limits)
    write_new(output/'started.json',dict(manifest_sha256=manifest_sha256,phase=phase,tasks=tasks,
        eligible_pairs=eligible,limits=limits,development_audit_sha256=development_audit_sha256,development_arithmetic_sha256=development_arithmetic_sha256))
    futures={};anchors={};resource_anchors={};start=time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as executor:
            futures={executor.submit(worker,str(manifest_path),manifest_sha256,phase,task,eligible,ledger.policy_sha256,development_audit_sha256,development_arithmetic_sha256):task for task in tasks}
            try:
                for future in as_completed(futures):
                    name,sha,resource_sha=future.result()
                    if name in anchors:raise ValueError('Repeated DNNR unit completion')
                    anchors[name]=sha;resource_anchors[name]=resource_sha
            except BaseException:
                for future in futures:future.cancel()
                raise
        verify_manifest(manifest_path,manifest_sha256)
        complete=dict(phase=phase,manifest_sha256=manifest_sha256,tasks=tasks,eligible_pairs=eligible,
            unit_anchors=anchors,worker_resource_anchors=resource_anchors,policy_sha256=ledger.policy_sha256,limits=limits,
            development_audit_sha256=development_audit_sha256,development_arithmetic_sha256=development_arithmetic_sha256,elapsed_seconds=time.perf_counter()-start)
        write_new(output/'complete.json',complete)
    except BaseException as exc:
        write_new(output/'failed.json',dict(type=type(exc).__name__,message=str(exc),completed_unit_anchors=anchors))
        raise
    return dict(phase=phase,complete_sha256=file_hash(output/'complete.json'))


def recover_members(output,complete,frame,folds,phase,eligible,*,cold,settings):
    tasks=phase_tasks(phase,eligible);expected={task_name(t) for t in tasks}
    if complete['tasks']!=tasks or set(complete['unit_anchors'])!=expected or set(complete['worker_resource_anchors'])!=expected:
        raise ValueError('Missing/extra/reordered complete phase units')
    if ({p.name for p in (output/'units').iterdir()}!=expected
            or {p.name for p in (output/'worker-resources').iterdir()}!={n+'.json' for n in expected}):
        raise ValueError('Unexpected/missing closed unit/resource paths')
    members,reports={},{}
    for task in tasks:
        name=task_name(task);unit=output/'units'/name
        info=anchored(unit/'complete.json',complete['unit_anchors'][name])
        training,y,query=partition(frame,folds,task)
        arms=required_arms(task['target'],phase,eligible)
        if info['task']!=task or info['policy_sha256']!=complete['policy_sha256'] or info['query_ids']!=query.sample_id.astype(str).tolist():
            raise ValueError('Unit task, ledger or held-out row identity differs')
        peak=anchored(output/'worker-resources'/f'{name}.json',complete['worker_resource_anchors'][name])
        if not np.isfinite(peak['peak_mib']) or peak['peak_mib']>1536 or peak['maximum_mib']!=1536:
            raise ValueError('Worker resource witness fails')
        if cold:
            values,reports[name]=audit_earned(unit,complete['unit_anchors'][name],task,training,y,query,arms,
                Settings(**settings),ledger_root=output/'ledger')
        else:
            values={}
            for a in arms:
                path=unit/f'prediction-{a}.npy'
                if file_hash(path)!=info['prediction_hashes'][path.name]:raise ValueError('Held-out prediction changed')
                values[a]=np.load(path,allow_pickle=False)
        mask=folds[task['seed']]==task['fold']
        for a,value in values.items():
            key=(task['target'],a,task['seed'])
            vector=members.setdefault(key,np.full(len(frame),np.nan))
            if value.shape!=(int(mask.sum()),) or not np.isfinite(value).all() or np.isfinite(vector[mask]).any():
                raise ValueError('Invalid/duplicate held-out coverage')
            vector[mask]=value
    if any(not np.isfinite(v).all() for v in members.values()):raise ValueError('Incomplete OOF column')
    return members,reports


def summarize(frame,folds,current,parent,members,phase,eligible):
    records=[];metrics={t:{} for t in TARGETS}
    targets=TARGETS if phase=='development' else tuple(t for t in TARGETS if any(p[0]==t for p in eligible))
    seeds=DEV if phase=='development' else CONFIRM
    y={t:frame[t].to_numpy(dtype=np.float64) for t in TARGETS}
    for target in targets:
        for seed in seeds:
            arms=required_arms(target,phase,eligible)
            rows,detail=score_seed(y,folds[seed],frame.spout_no.to_numpy(),current[seed],parent[seed],
                {a:members[target,a,seed] for a in arms},target,seed)
            records.extend(rows)
            for arm,value in detail.items():metrics[target].setdefault(arm,{})[str(seed)]=value
    return records,metrics


def audit_phase(manifest_path,manifest_sha256,phase,complete_sha256,development_audit_sha256=None,development_arithmetic_sha256=None):
    m=verify_manifest(manifest_path,manifest_sha256);root=Path(manifest_path).resolve().parent;output=root/phase
    eligible,prior=earned_context(root,phase,development_audit_sha256,development_arithmetic_sha256)
    complete=anchored(output/'complete.json',complete_sha256)
    if (complete['manifest_sha256']!=manifest_sha256 or complete['phase']!=phase or complete['eligible_pairs']!=eligible
            or complete['development_audit_sha256']!=development_audit_sha256
            or complete['development_arithmetic_sha256']!=development_arithmetic_sha256 or (output/'failed.json').exists()):
        raise ValueError('Failed/foreign phase completion cannot be audited')
    write_new(output/'audit-started.json',dict(complete_sha256=complete_sha256,new_fits=0))
    try:
        frame,folds,current,parent,_=reload_references(m)
        members,unit_reports=recover_members(output,complete,frame,folds,phase,eligible,cold=True,settings=m['settings'])
        records,metrics=summarize(frame,folds,current,parent,members,phase,eligible)
        decision=decide_development(records) if phase=='development' else decide_confirmation(prior['records'],records)
        ledger=ReservationLedger.open(output/'ledger',complete['policy_sha256']);counts=ledger.inspect()
        limits=phase_limits(phase,eligible)
        actual=dict(pair_unit=len(unit_reports),estimator=sum(r['counts']['estimator_runs'] for r in unit_reports.values()),
            derivative_bank=sum(r['counts']['derivative_bank_runs'] for r in unit_reports.values()),
            metric_epoch=sum(r['counts']['metric_epoch_runs'] for r in unit_reports.values()))
        if (ledger.limits!=limits or complete['limits']!=limits or counts['started']!=actual or counts['completed']!=actual
                or any(v for state in ('failed','incomplete') for v in counts[state].values())
                or any(actual[k]!=limits[k] for k in ('pair_unit','estimator','derivative_bank'))):
            raise ValueError('Phase actual fitting reservations differ from frozen budget')
        policy=yaml.safe_load((Path(m['workspace'])/'configs/candidate_tiers.yaml').read_text())
        tiers=classify_candidates(metrics,m['execution_spec'],policy) if phase=='development' else None
        verify_manifest(manifest_path,manifest_sha256)
        # Exclude live controller logs and evidence notifications; only closed
        # immutable fitting artifacts are included in the audit's hash inventory.
        hashes={str(p.relative_to(output)):file_hash(p) for folder in ('units','ledger','worker-resources')
                for p in sorted((output/folder).rglob('*')) if p.is_file()}
        report=dict(status='passed',phase=phase,manifest_sha256=manifest_sha256,phase_complete_sha256=complete_sha256,
            development_audit_sha256=development_audit_sha256,development_arithmetic_sha256=development_arithmetic_sha256,records=records,metrics=metrics,decision=decision,
            candidate_tiers_descriptive=tiers,unit_reports=unit_reports,counts=counts,artifact_hashes=hashes,
            new_audit_fits=0,release_authorized=False)
        write_new(output/'audit.json',report)
    except BaseException as exc:
        write_new(output/'audit-failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return dict(phase=phase,audit_sha256=file_hash(output/'audit.json'),decision=decision)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','audit'])
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--phase',choices=['development','confirmation'],required=True)
    p.add_argument('--complete-sha256');p.add_argument('--development-audit-sha256');p.add_argument('--development-arithmetic-sha256');a=p.parse_args()
    result=run_phase(a.manifest,a.manifest_sha256,a.phase,a.development_audit_sha256,a.development_arithmetic_sha256) if a.action=='run' else audit_phase(
        a.manifest,a.manifest_sha256,a.phase,a.complete_sha256,a.development_audit_sha256,a.development_arithmetic_sha256)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
