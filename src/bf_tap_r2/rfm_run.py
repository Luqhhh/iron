"""Bounded RFM scheduling and fresh-process complete-phase audit entrypoints."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import json
import multiprocessing
from pathlib import Path

from .data import FEATURES, TARGETS
from .rfm_execution import execute_unit, collect_audited_phase, task_name
from .rfm_freeze import load_anchored, private_path, reload_references, verify_manifest, require_threads
from .rfm_protocol import ReservationLedger, file_hash, phase_limits, phase_tasks, write_new
from .rfm_scoring import decide_development


def bounded_map(executor, worker, jobs, workers=4):
    """Keep at most four submitted jobs; after any failure dispatch nothing new.

    Already running jobs may finish. The executor owner waits for those jobs;
    their append-only artifacts remain even when this function raises.
    """
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("worker count must be between one and four")
    iterator, pending, results = iter(jobs), set(), []
    def fill():
        while len(pending) < workers:
            try:
                job = next(iterator)
            except StopIteration:
                break
            pending.add(executor.submit(worker, job))
    fill()
    while pending:
        done, pending = wait(pending,return_when=FIRST_COMPLETED)
        # Inspect the whole completed group before refilling any slots.
        errors = [f.exception() for f in done if f.exception() is not None]
        if errors:
            for future in pending:
                future.cancel()
            raise errors[0]
        results.extend(f.result() for f in done)
        fill()
    return results


def initialize_worker():
    require_threads()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _worker(job):
    manifest_path, manifest_sha256, task, training, query, output, ledger, policy = job
    verify_manifest(manifest_path,manifest_sha256,references=False)
    result = execute_unit(task,training,query,output,ledger,policy)
    verify_manifest(manifest_path,manifest_sha256,references=False)
    return result


def _context(manifest_path, manifest_sha256, preflight_sha256):
    manifest = verify_manifest(manifest_path,manifest_sha256)
    root = private_path(manifest["workspace"],Path(manifest_path).resolve().parent)
    # Lazy import keeps tiny scheduler tests independent of full-size admission.
    from .rfm_preflight import verify_admission
    verify_admission(root/'preflight'/'admission.json',preflight_sha256,manifest_path,manifest_sha256)
    return manifest,root


def development_records(root, manifest_sha256, audit_sha256):
    audit = load_anchored(root/'development'/'audit.json',audit_sha256)
    if (audit.get('status') != 'passed' or audit.get('phase') != 'development'
            or audit.get('manifest_sha256') != manifest_sha256
            or audit.get('new_audit_fits') != 0):
        raise ValueError('passed development audit for this manifest required')
    load_anchored(root/'development'/'complete.json',audit['phase_complete_sha256'])
    return audit['records']


def run_phase(manifest_path, manifest_sha256, preflight_sha256, phase, development_sha256=None):
    manifest,root = _context(manifest_path,manifest_sha256,preflight_sha256)
    dev = None if phase == 'development' else development_records(root,manifest_sha256,development_sha256)
    if phase == 'development' and development_sha256 is not None:
        raise ValueError('development cannot consume a prior audit')
    eligible = None if phase == 'development' else decide_development(dev)['eligible_targets']
    tasks = phase_tasks(phase,eligible)
    if not tasks:
        raise ValueError('no earned confirmation tasks; controller must skip this phase')
    output = root/phase
    output.mkdir(exist_ok=False)
    write_new(output/'started.json',dict(phase=phase,manifest_sha256=manifest_sha256,
              preflight_sha256=preflight_sha256,development_sha256=development_sha256,tasks=tasks))
    try:
        frame,folds,current,historical,_ = reload_references(manifest)
        ledger=ReservationLedger.create(output/'ledger',phase_limits(phase,eligible))
        def jobs():
            for task in tasks:
                mask=folds[task['seed']]==task['fold']
                train=frame.loc[~mask].reset_index(drop=True)
                query=frame.loc[mask,['sample_id','spout_no',*FEATURES]].reset_index(drop=True)
                yield (manifest_path,manifest_sha256,task,train,query,output/'units'/task_name(task),
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
            ledger_policy_sha256=ledger.policy_sha256,counts=ledger.inspect())
        write_new(output/'complete.json',complete)
    except BaseException as exc:
        write_new(output/'failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return dict(phase=phase,complete_sha256=file_hash(output/'complete.json'))


def audit_phase(manifest_path, manifest_sha256, preflight_sha256, phase,
                complete_sha256, development_sha256=None):
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
        dev=None if phase=='development' else development_records(root,manifest_sha256,development_sha256)
        eligible=None if phase=='development' else decide_development(dev)['eligible_targets']
        expected=dict(phase=phase,manifest_sha256=manifest_sha256,preflight_sha256=preflight_sha256,
                      development_sha256=development_sha256,tasks=phase_tasks(phase,eligible))
        if started!=expected or any(complete[k]!=v for k,v in expected.items() if k!='tasks'):
            raise ValueError('phase source/admission/parent identity mismatch')
        frame,folds,current,historical,_=reload_references(manifest)
        report=collect_audited_phase(manifest['workspace'],output,phase,frame,folds,current,historical,
                    complete['unit_anchors'],complete['ledger_policy_sha256'],dev)
        if report['counts']!=complete['counts']:
            raise ValueError('ledger changed after phase completion')
        verify_manifest(manifest_path,manifest_sha256)
        report.update(manifest_sha256=manifest_sha256,preflight_sha256=preflight_sha256,
                      phase_complete_sha256=complete_sha256,development_sha256=development_sha256)
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
    a=p.parse_args()
    args=(a.manifest,a.manifest_sha256,a.preflight_sha256,a.phase)
    if a.action=='run':
        result=run_phase(*args,development_sha256=a.development_sha256)
    else:
        result=audit_phase(*args,complete_sha256=a.complete_sha256,development_sha256=a.development_sha256)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
