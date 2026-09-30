"""Fixed-worker PTaRL scheduling without a wall-clock cap; complete cold audit."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import multiprocessing
from pathlib import Path

from .data import FEATURES
from .ptarl_execution import execute_unit, task_name
from .ptarl_phase import collect_audited_phase
from .ptarl_unbudgeted_freeze import load_anchored, private_path, reload_references, verify_manifest, require_threads,require_serial_ready
from .ptarl_protocol import ReservationLedger, file_hash, phase_limits, phase_tasks, write_new
from .ptarl_scoring import decide_development
from .rfm_run import bounded_map, initialize_worker


def _worker(job):
    manifest_path, manifest_sha256, task, training, query, settings, output, ledger, policy = job
    manifest=verify_manifest(manifest_path,manifest_sha256,references=False)
    require_serial_ready(manifest)
    if settings!=manifest['settings']:
        raise ValueError('Worker settings differ from frozen PTaRL spec')
    result = execute_unit(task,training,query,settings,output,ledger,policy)
    verify_manifest(manifest_path,manifest_sha256,references=False)
    return result


def _context(manifest_path, manifest_sha256, preflight_sha256):
    manifest = verify_manifest(manifest_path,manifest_sha256)
    require_serial_ready(manifest)
    root = private_path(manifest["workspace"],Path(manifest_path).resolve().parent)
    # Lazy import keeps tiny scheduler tests independent of full-size admission.
    from .ptarl_unbudgeted_freeze import verify_resource
    verify_resource(manifest,preflight_sha256)
    return manifest,root


def verify_audited_artifacts(output,audit):
    actual={str(p.relative_to(output)) for folder in ('units','ledger')
        for p in (output/folder).rglob('*') if p.is_file()}
    if actual!=set(audit['artifact_hashes']):
        raise ValueError('Cold audited artifact inventory changed')
    for name,digest in audit['artifact_hashes'].items():
        path=output/name
        if path.resolve()!=path or not path.is_relative_to(output) or file_hash(path)!=digest:
            raise ValueError('Cold audited development artifact changed')


def development_records(root, manifest_sha256, audit_sha256,arithmetic_sha256):
    audit = load_anchored(root/'development'/'audit.json',audit_sha256)
    if (audit.get('status') != 'passed' or audit.get('phase') != 'development'
            or audit.get('manifest_sha256') != manifest_sha256
            or audit.get('new_audit_fits') != 0):
        raise ValueError('passed development audit for this manifest required')
    development=root/'development'
    if any((development/name).exists() for name in ('failed.json','audit-failed.json','arithmetic-failed.json')):
        raise ValueError('Failed development cannot earn confirmation')
    load_anchored(development/'complete.json',audit['phase_complete_sha256'])
    verify_audited_artifacts(development,audit)
    arithmetic=load_anchored(development/'arithmetic.json',arithmetic_sha256)
    decision=decide_development(audit['records'])
    if (arithmetic['status']!='passed' or arithmetic['phase']!='development' or arithmetic['new_fits']!=0
            or arithmetic['manifest_sha256']!=manifest_sha256
            or arithmetic['audit_sha256']!=audit_sha256 or arithmetic['selected_targets']!=decision['eligible_targets']):
        raise ValueError('Confirmation requires independent development arithmetic')
    return audit['records']


def run_phase(manifest_path, manifest_sha256, preflight_sha256, phase, development_sha256=None,development_arithmetic_sha256=None):
    manifest,root = _context(manifest_path,manifest_sha256,preflight_sha256)
    dev = None if phase == 'development' else development_records(root,manifest_sha256,development_sha256,development_arithmetic_sha256)
    if phase == 'development' and (development_sha256 is not None or development_arithmetic_sha256 is not None):
        raise ValueError('development cannot consume a prior audit')
    eligible = None if phase == 'development' else decide_development(dev)['eligible_targets']
    tasks = phase_tasks(phase,eligible)
    if not tasks:
        raise ValueError('no earned confirmation tasks; controller must skip this phase')
    output = root/phase
    output.mkdir(exist_ok=False)
    write_new(output/'started.json',dict(phase=phase,manifest_sha256=manifest_sha256,
              preflight_sha256=preflight_sha256,development_sha256=development_sha256,
              development_arithmetic_sha256=development_arithmetic_sha256,tasks=tasks))
    try:
        frame,folds,current,historical,_ = reload_references(manifest)
        ledger=ReservationLedger.create(output/'ledger',phase_limits(phase,eligible))
        def jobs():
            for task in tasks:
                mask=folds[task['seed']]==task['fold']
                train=frame.loc[~mask].reset_index(drop=True)
                query=frame.loc[mask,['sample_id','spout_no',*FEATURES]].reset_index(drop=True)
                yield (manifest_path,manifest_sha256,task,train,query,manifest['settings'],output/'units'/task_name(task),
                       ledger.root,ledger.policy_sha256)
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_worker) as pool:
            results=bounded_map(pool,_worker,jobs())
        verify_manifest(manifest_path,manifest_sha256)
        anchors={r['name']:r['complete_sha256'] for r in results}
        if len(anchors)!=len(tasks):
            raise ValueError('missing/duplicate worker results')
        complete=dict(phase=phase,started_sha256=file_hash(output/'started.json'),
            manifest_sha256=manifest_sha256,preflight_sha256=preflight_sha256,
            development_sha256=development_sha256,unit_anchors=anchors,
            development_arithmetic_sha256=development_arithmetic_sha256,
            ledger_policy_sha256=ledger.policy_sha256,counts=ledger.inspect())
        write_new(output/'complete.json',complete)
    except BaseException as exc:
        write_new(output/'failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return dict(phase=phase,complete_sha256=file_hash(output/'complete.json'))


def audit_phase(manifest_path, manifest_sha256, preflight_sha256, phase,
                complete_sha256, development_sha256=None,development_arithmetic_sha256=None):
    manifest,root = _context(manifest_path,manifest_sha256,preflight_sha256)
    output=root/phase
    if phase not in ('development','confirmation'):
        raise ValueError('invalid phase')
    if (output/'failed.json').exists():
        raise ValueError('failed phase cannot pass admission')
    # An audit attempt, including a failed one, is never silently repeated.
    write_new(output/'audit-started.json',dict(phase_complete_sha256=complete_sha256,
              manifest_sha256=manifest_sha256))
    try:
        complete=load_anchored(output/'complete.json',complete_sha256)
        started=load_anchored(output/'started.json',complete['started_sha256'])
        dev=None if phase=='development' else development_records(root,manifest_sha256,development_sha256,development_arithmetic_sha256)
        eligible=None if phase=='development' else decide_development(dev)['eligible_targets']
        expected=dict(phase=phase,manifest_sha256=manifest_sha256,preflight_sha256=preflight_sha256,
                      development_sha256=development_sha256,development_arithmetic_sha256=development_arithmetic_sha256,
                      tasks=phase_tasks(phase,eligible))
        if started!=expected or any(complete[k]!=v for k,v in expected.items() if k!='tasks'):
            raise ValueError('phase source/admission/parent identity mismatch')
        frame,folds,current,historical,_=reload_references(manifest)
        report=collect_audited_phase(manifest['workspace'],output,phase,frame,folds,current,historical,
                    manifest['settings'],complete['unit_anchors'],complete['ledger_policy_sha256'],dev)
        if report['counts']!=complete['counts']:
            raise ValueError('ledger changed after phase completion')
        verify_manifest(manifest_path,manifest_sha256)
        report.update(manifest_sha256=manifest_sha256,preflight_sha256=preflight_sha256,
                      phase_complete_sha256=complete_sha256,development_sha256=development_sha256,
                      development_arithmetic_sha256=development_arithmetic_sha256,
                      artifact_hashes={str(p.relative_to(output)):file_hash(p) for folder in ('units','ledger')
                          for p in sorted((output/folder).rglob('*')) if p.is_file()})
        write_new(output/'audit.json',report)
    except BaseException as exc:
        write_new(output/'audit-failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return dict(phase=phase,audit_sha256=file_hash(output/'audit.json'),decision=report['decision'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','audit'])
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--preflight-sha256',required=True)
    p.add_argument('--phase',choices=['development','confirmation'],required=True)
    p.add_argument('--complete-sha256')
    p.add_argument('--development-sha256')
    p.add_argument('--development-arithmetic-sha256')
    a=p.parse_args()
    args=(a.manifest,a.manifest_sha256,a.preflight_sha256,a.phase)
    if a.action=='run':
        result=run_phase(*args,development_sha256=a.development_sha256,development_arithmetic_sha256=a.development_arithmetic_sha256)
    else:
        result=audit_phase(*args,complete_sha256=a.complete_sha256,development_sha256=a.development_sha256,
            development_arithmetic_sha256=a.development_arithmetic_sha256)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
