"""Explicit host-reboot recovery, preserving frozen scientific sources and evidence."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from bf_tap_r2.data import FEATURES
from bf_tap_r2.ptarl_execution import audit_unit, task_name
from bf_tap_r2.ptarl_protocol import ReservationLedger, file_hash, phase_tasks, phase_limits, write_new
from bf_tap_r2.ptarl_unbudgeted_freeze import verify_manifest, require_serial_ready, private_path
from bf_tap_r2.ptarl_unbudgeted_run import _context, _worker
from bf_tap_r2.ptarl_unbudgeted_controller import sequence
from bf_tap_r2.rfm_run import bounded_map, initialize_worker


def inventory(root):
    root = Path(root).resolve()
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('Recovery input must have no symlinks')
        if p.is_file():
            result[str(p.relative_to(root))] = file_hash(p)
    return result


def verify_inventory(root, expected):
    if inventory(root) != expected:
        raise ValueError('Interrupted evidence changed')


def closed_events(phase, name, policy_sha256):
    expected = {('pair_unit', (name,))}
    for role in ('teacher_inner', 'teacher_outer'):
        expected.add(('optimizer', (name, role)))
        expected.add(('kmeans', (name, role)))
    for arm in ('CONTROL', 'PTARL_AUX'):
        for stage in ('selector', 'refit'):
            expected.add(('optimizer', (name, arm+'_'+stage)))
    observed, files = set(), []
    for p in sorted((Path(phase)/'ledger/events').glob('*.started.json')):
        r = json.loads(p.read_text())
        if not r['key'] or r['key'][0] != name:
            continue
        key = r['kind'], tuple(r['key'])
        end = p.with_name(p.name.replace('started.json', 'complete.json'))
        failed = p.with_name(p.name.replace('started.json', 'failed.json'))
        if (key not in expected or key in observed or not end.exists() or failed.exists()
                or r['policy_sha256'] != policy_sha256
                or json.loads(end.read_text())['start_sha256'] != file_hash(p)):
            raise ValueError('Reusable unit has incomplete or mismatched reservations')
        observed.add(key)
        files.extend((p, end))
    if observed != expected:
        raise ValueError('Reusable unit reservation coverage incomplete')
    return files


def partition_tasks(phase):
    tasks = phase_tasks('development')
    names = {task_name(t) for t in tasks}
    actual = {p.name for p in (Path(phase)/'units').iterdir()}
    if not actual <= names:
        raise ValueError('Unexpected interrupted units')
    reusable, missing = [], []
    for t in tasks:
        unit = Path(phase)/'units'/task_name(t)
        (reusable if (unit/'complete.json').exists() else missing).append(t)
    return reusable, missing


def entry_refusal_scope(root, terminal_sha256):
    """Refuse every failed/incomplete fit; allow only untouched missing tasks."""
    root = Path(root)
    terminal = json.loads((root/'completion-event.json').read_text())
    if (file_hash(root/'completion-event.json') != terminal_sha256
            or terminal.get('status') != 'failed' or terminal.get('type') != 'ValueError'
            or terminal.get('message') != 'Current memory below original admitted requirement'
            or terminal.get('packages') != 0 or terminal.get('uploads') != 0):
        raise ValueError('Externally anchored worker-entry memory refusal required')
    phase = root/'development'
    if any((phase/n).exists() for n in ('complete.json', 'failed.json', 'audit-started.json')):
        raise ValueError('Failed/completed/audited phase cannot use entry-refusal retry')
    reusable, missing = partition_tasks(phase)
    if not reusable or not missing:
        raise ValueError('Partial closed coverage and untouched missing tasks required')
    names = {task_name(t) for t in reusable}
    if {p.name for p in (phase/'units').iterdir()} != names:
        raise ValueError('Missing tasks have model artifacts or consumed reservations')
    policy_sha = file_hash(phase/'ledger/policy.json')
    ledger = ReservationLedger.open(phase/'ledger', policy_sha); counts = ledger.inspect()
    required = dict(pair_unit=len(reusable), optimizer=6*len(reusable), kmeans=2*len(reusable))
    if (ledger.limits != phase_limits('development') or counts['started'] != required
            or counts['completed'] != required
            or any(counts[s][k] for s in ('failed', 'incomplete') for k in required)):
        raise ValueError('Failed/incomplete fit reservations cannot be retried')
    for task in reusable:
        closed_events(phase, task_name(task), policy_sha)
    return reusable, missing, policy_sha, counts


def prepare_entry_refusal(manifest_path, manifest_sha256, proof_path, proof_sha256,
                          tests_path, tests_sha256, output):
    manifest_path, proof_path, tests_path = map(Path, (manifest_path, proof_path, tests_path))
    if file_hash(proof_path) != proof_sha256 or file_hash(tests_path) != tests_sha256:
        raise ValueError('External terminal/test proof anchors required')
    proof = json.loads(proof_path.read_text())
    if (proof['status'] != 'resource_gate_refusal_at_worker_entry_before_any_missing_task_reservation'
            or not proof['original_evidence_unchanged'] or proof['new_audit_fits'] != 0
            or proof['phase_complete'] or proof['missing_task_directories'] != 0):
        raise ValueError('Verified pre-reservation resource refusal required')
    wrapper = Path(__file__).resolve(); workspace = wrapper.parent.parent
    tests = json.loads(tests_path.read_text())
    if (tests['status'] != 'passed' or tests['exit_code'] != 0 or not tests['full_suite']
            or tests['source_changed'] or tests['source_hashes'].get('scripts/'+wrapper.name) != file_hash(wrapper)
            or file_hash(tests['log']) != tests['log_sha256']):
        raise ValueError('Checked exact-source full-suite retry wrapper required')
    for name, sha in tests['source_hashes'].items():
        if file_hash(workspace/name) != sha:
            raise ValueError('Checked retry source changed')
    manifest = verify_manifest(manifest_path, manifest_sha256)
    old = manifest_path.resolve().parent
    if proof_path.resolve() != old/'terminal-verification.json':
        raise ValueError('Original terminal verification location required')
    original = dict(proof['artifact_hashes'], **{'terminal-verification.json': proof_sha256})
    verify_inventory(old, original)
    reusable, missing, policy_sha, counts = entry_refusal_scope(old, proof['terminal_sha256'])
    started = json.loads((old/'development/started.json').read_text())
    if (started['manifest_sha256'] != manifest_sha256 or started['tasks'] != phase_tasks('development')
            or counts != proof['closed_counts'] or missing != proof['remaining_tasks']):
        raise ValueError('Verified phase/remaining task identities changed')
    output = private_path(manifest['workspace'], output); output.mkdir(parents=True, exist_ok=False)
    with (output/'manifest.json').open('xb') as f:
        f.write(manifest_path.read_bytes())
    plan = dict(version=2, original_root=str(old), original_artifacts=original,
        manifest_sha256=manifest_sha256, output=str(output),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        terminal_verification_path=str(proof_path.resolve()), terminal_verification_sha256=proof_sha256,
        recovery_wrapper=str(wrapper), recovery_wrapper_sha256=file_hash(wrapper),
        tests_path=str(tests_path.resolve()), tests_sha256=tests_sha256,
        source_workspace=manifest['workspace'], original_counts=counts,
        reused_tasks=reusable, fresh_tasks=missing, original_policy_sha256=policy_sha,
        retry_authority='继续重试，刚才是别的任务在进行、现在结束了',
        predecessor_service='iron-ptarl-reboot-recovery-r1.service', execution_workers=4,
        scientific_recipe='unchanged', maximum_runtime_seconds=None,
        failed_model_retries=0, automatic_packages=False, uploads=0)
    write_new(output/'recovery-plan.json', plan)
    return dict(plan=str(output/'recovery-plan.json'), sha256=file_hash(output/'recovery-plan.json'))


def prepare(manifest_path, manifest_sha256, interruption_path, interruption_sha256,
            tests_path, tests_sha256, output):
    manifest_path, interruption_path, tests_path = map(Path, (manifest_path, interruption_path, tests_path))
    if file_hash(interruption_path) != interruption_sha256 or file_hash(tests_path) != tests_sha256:
        raise ValueError('External interruption/test anchors required')
    proof = json.loads(interruption_path.read_text())
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if (proof['status'] != 'host_reboot_interruption_confirmed_not_model_failure'
            or proof['current_boot_id'] != boot
            or proof['last_scheduled_observation']['observed_ns']/1e9 >= proof['current_boot_epoch']):
        raise ValueError('Specific host reboot evidence required')
    tests = json.loads(tests_path.read_text())
    wrapper = Path(__file__).resolve()
    checked_workspace = wrapper.parent.parent
    if (tests['status'] != 'passed' or tests['exit_code'] != 0
            or not tests['full_suite'] or tests['source_changed']
            or tests['source_hashes'].get('scripts/'+wrapper.name) != file_hash(wrapper)):
        raise ValueError('Checked recovery wrapper required')
    for name, digest in tests['source_hashes'].items():
        if file_hash(checked_workspace/name) != digest:
            raise ValueError('Recovery checked sources changed')
    if file_hash(tests['log']) != tests['log_sha256']:
        raise ValueError('Checked recovery test log changed')
    manifest = verify_manifest(manifest_path, manifest_sha256)
    old = manifest_path.resolve().parent
    verify_inventory(old, proof['artifact_hashes'])
    if any((old/n).exists() for n in ('completion-event.json', 'supervisor-terminal.json')):
        raise ValueError('Terminal execution cannot use reboot recovery')
    phase = old/'development'
    if any((phase/n).exists() for n in ('complete.json', 'failed.json', 'audit-started.json')):
        raise ValueError('Only interrupted training before audit may recover')
    started = json.loads((phase/'started.json').read_text())
    if started['manifest_sha256'] != manifest_sha256 or started['tasks'] != phase_tasks('development'):
        raise ValueError('Interrupted phase freeze mismatch')
    policy_sha = file_hash(phase/'ledger/policy.json')
    ledger = ReservationLedger.open(phase/'ledger', policy_sha)
    counts = ledger.inspect()
    if ledger.limits != phase_limits('development') or any(counts['failed'].values()):
        raise ValueError('Model failure cannot be silently retried as reboot recovery')
    reusable, missing = partition_tasks(phase)
    for t in reusable:
        closed_events(phase, task_name(t), policy_sha)
    output = private_path(manifest['workspace'], output)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'manifest.json').open('xb') as f:
        f.write(manifest_path.read_bytes())
    plan = dict(version=1, original_root=str(old), original_artifacts=proof['artifact_hashes'],
        manifest_sha256=manifest_sha256, output=str(output), boot_id=boot,
        interruption_path=str(interruption_path.resolve()), interruption_sha256=interruption_sha256,
        recovery_wrapper=str(wrapper), recovery_wrapper_sha256=file_hash(wrapper),
        tests_path=str(tests_path.resolve()), tests_sha256=tests_sha256,
        source_workspace=manifest['workspace'], original_counts=counts,
        reused_tasks=reusable, fresh_tasks=missing, original_policy_sha256=policy_sha,
        scientific_recipe='unchanged', maximum_runtime_seconds=None,
        failed_model_retries=0, automatic_packages=False, uploads=0)
    write_new(output/'recovery-plan.json', plan)
    return dict(plan=str(output/'recovery-plan.json'), sha256=file_hash(output/'recovery-plan.json'))


def verify_plan(path, expected_sha256):
    if file_hash(path) != expected_sha256:
        raise ValueError('External recovery plan anchor required')
    plan = json.loads(Path(path).read_text())
    if (file_hash(plan['recovery_wrapper']) != plan['recovery_wrapper_sha256']
            or file_hash(plan['tests_path']) != plan['tests_sha256']
            or Path('/proc/sys/kernel/random/boot_id').read_text().strip() != plan['boot_id']):
        raise ValueError('Recovery code/test/boot changed')
    verify_inventory(plan['original_root'], plan['original_artifacts'])
    return plan


def run(plan_path, plan_sha256):
    plan = verify_plan(plan_path, plan_sha256)
    root = Path(plan['output']); manifest_path = root/'manifest.json'
    mh = plan['manifest_sha256']
    manifest = verify_manifest(manifest_path, mh)
    ph = manifest['resource_admission_sha256']
    _context(manifest_path, mh, ph)
    require_serial_ready(manifest)
    status = subprocess.check_output(['systemctl', '--user', 'show', 'iron-ptarl-unbudgeted-formal-r1.service',
        '-p', 'MainPID', '-p', 'ActiveState'], text=True)
    if 'MainPID=0\n' not in status or not any(s in status for s in ('ActiveState=inactive', 'ActiveState=failed')):
        raise ValueError('Original PTaRL process must be absent before explicit recovery')
    if 'predecessor_service' in plan:
        previous = subprocess.check_output(['systemctl', '--user', 'show', plan['predecessor_service'],
            '-p', 'MainPID', '-p', 'ActiveState'], text=True)
        if 'MainPID=0\n' not in previous or not any(s in previous for s in ('ActiveState=inactive', 'ActiveState=failed')):
            raise ValueError('Predecessor recovery must actually be terminal')
    write_new(root/'orchestration-started.json', dict(recovery_plan_sha256=plan_sha256, time_ns=time.time_ns()))
    try:
        fit = recover_development(plan, manifest_path, mh, ph)
        results = continue_sequence(root, manifest, mh, ph, fit)
        verify_plan(plan_path, plan_sha256)
        verify_manifest(manifest_path, mh)
        terminal = dict(status='completed', results=results, recovery_plan_sha256=plan_sha256,
            reused_units=len(plan['reused_tasks']), fresh_units=len(plan['fresh_tasks']), packages=0, uploads=0)
    except BaseException as exc:
        write_new(root/'completion-event.json', dict(status='failed', type=type(exc).__name__, message=str(exc),
            recovery_plan_sha256=plan_sha256, packages=0, uploads=0))
        raise
    write_new(root/'completion-event.json', terminal)
    return terminal


def recover_development(plan, manifest_path, mh, ph):
    manifest = verify_manifest(manifest_path, mh)
    from bf_tap_r2.ptarl_unbudgeted_freeze import reload_references
    frame, folds, current, historical, _ = reload_references(manifest)
    output = Path(plan['output'])/'development'; output.mkdir(exist_ok=False)
    old = Path(plan['original_root'])/'development'
    # Original starts/partial models remain in the old root and continue to consume its ledger.
    write_new(output/'started.json', json.loads((old/'started.json').read_text()))
    ledger = ReservationLedger.create(output/'ledger', phase_limits('development'))
    if ledger.policy_sha256 != plan['original_policy_sha256']:
        raise ValueError('Identical closed-phase ledger policy required')
    (output/'units').mkdir()
    anchors, reuse_audits = {}, []
    for task in plan['reused_tasks']:
        name = task_name(task); mask = folds[task['seed']] == task['fold']
        training = frame.loc[~mask].reset_index(drop=True)
        query = frame.loc[mask, ['sample_id', 'spout_no', *FEATURES]].reset_index(drop=True)
        anchor = file_hash(old/'units'/name/'complete.json')
        _, audit = audit_unit(old/'units'/name, task, training, query, manifest['settings'],
            expected_sha256=anchor, ledger_root=old/'ledger')
        reuse_audits.append(audit)
        shutil.copytree(old/'units'/name, output/'units'/name)
        for path in closed_events(old, name, ledger.policy_sha256):
            shutil.copyfile(path, ledger.root/'events'/path.name)
        anchors[name] = anchor
    write_new(Path(plan['output'])/'reuse-audit.json', dict(status='passed', units=reuse_audits,
        original_counts=plan['original_counts'], new_fits=0, copied_units=len(anchors)))
    def jobs():
        for task in plan['fresh_tasks']:
            mask = folds[task['seed']] == task['fold']
            training = frame.loc[~mask].reset_index(drop=True)
            query = frame.loc[mask, ['sample_id', 'spout_no', *FEATURES]].reset_index(drop=True)
            yield (manifest_path, mh, task, training, query, manifest['settings'], output/'units'/task_name(task),
                ledger.root, ledger.policy_sha256)
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker) as pool:
        results = bounded_map(pool, _worker, jobs())
    for result in results:
        if result['name'] in anchors:
            raise ValueError('Recovery refit a completed unit')
        anchors[result['name']] = result['complete_sha256']
    if set(anchors) != {task_name(t) for t in phase_tasks('development')}:
        raise ValueError('Recovery coverage incomplete')
    verify_manifest(manifest_path, mh)
    write_new(output/'complete.json', dict(phase='development', started_sha256=file_hash(output/'started.json'),
        manifest_sha256=mh, preflight_sha256=ph, development_sha256=None, development_arithmetic_sha256=None,
        unit_anchors=anchors, ledger_policy_sha256=ledger.policy_sha256, counts=ledger.inspect()))
    return dict(phase='development', complete_sha256=file_hash(output/'complete.json'))


def continue_sequence(root, manifest, mh, ph, development_fit):
    def invoke(name, module, args):
        if name == 'development-run':
            return development_fit
        verify_manifest(root/'manifest.json', mh)
        command = [sys.executable, '-m', module, *args, '--manifest', str(root/'manifest.json'), '--manifest-sha256', mh]
        if module.endswith('ptarl_unbudgeted_run'):
            command += ['--preflight-sha256', ph]
        write_new(root/(name+'-started.json'), dict(command=command, time_ns=time.time_ns()))
        with (root/(name+'.log')).open('x') as log:
            proc = subprocess.run(command, cwd=manifest['workspace'], stdout=log, stderr=subprocess.STDOUT)
        write_new(root/(name+'-event.json'), dict(returncode=proc.returncode, time_ns=time.time_ns()))
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, command)
        value = json.loads((root/(name+'.log')).read_text().splitlines()[-1])
        write_new(root/(name+'-result.json'), value)
        return value
    return sequence(invoke)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'prepare-entry-refusal', 'run'])
    for name in ('manifest', 'interruption', 'tests', 'output', 'plan', 'proof'):
        p.add_argument('--'+name, type=Path)
    for name in ('manifest', 'interruption', 'tests', 'plan', 'proof'):
        p.add_argument('--'+name+'-sha256')
    a = p.parse_args()
    if a.action == 'prepare':
        result = prepare(a.manifest, a.manifest_sha256, a.interruption, a.interruption_sha256, a.tests, a.tests_sha256, a.output)
    elif a.action == 'prepare-entry-refusal':
        result = prepare_entry_refusal(a.manifest, a.manifest_sha256, a.proof, a.proof_sha256,
            a.tests, a.tests_sha256, a.output)
    else:
        result = run(a.plan, a.plan_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
