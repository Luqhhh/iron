"""Independent append-only terminal audit; imports no scientific training code."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def expected_tasks():
    result = []
    for seed in (42, 3407):
        for fold in range(5):
            result.append(f'{len(result):03d}-reuse-s{seed}-f{fold}-i-1')
            for init in (42, 1042, 2042):
                for stage in ('worker', 'cold'):
                    result.append(f'{len(result):03d}-{stage}-s{seed}-f{fold}-i{init}')
    return result + ['070-report-s42-f0-i-1', '071-audit-s42-f0-i-1']


def verify_events(terminal):
    if terminal.get('status') != 'passed' or not terminal.get('dependencies_unchanged'):
        raise ValueError('Successful real controller terminal required before quality reads')
    events = terminal['events']
    if [e['task'] for e in events] != expected_tasks():
        raise ValueError('Missing, duplicate or reordered child event')
    for e in events:
        if (e['exit_code'] != 0 or not math.isfinite(e['peak_rss_mib'])
                or not 0 < e['peak_rss_mib'] <= 1536):
            raise ValueError('Child execution/memory gate failed')
    times = [e['completed_ns'] for e in events]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('Child completion order differs')
    return events


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def verify_native(c, start, receipts, traces, plan):
    if (c['constructors'] != ['selection', 'refit'] or set(traces) != {'selection', 'refit'}
            or start['identity'] != c['identity'] or start['pid'] != c['pid']
            or c['training_ids'] != start['training_ids'] or c['training_ids'] != plan['training_ids']
            or c['query_ids'] != start['query_ids'] or c['query_ids'] != plan['query_ids']
            or set(c['training_ids']) & set(c['query_ids'])):
        raise ValueError('Estimator/partition/native optimizer identity differs')
    last = start['time_ns']
    for role in ('selection', 'refit'):
        receipt, trace = receipts[role], traces[role]
        if (receipt['identity'] != c['identity'] or receipt['role'] != role
                or receipt['time_ns'] <= last):
            raise ValueError('Optimizer pre-construction ordering differs')
        last = receipt['time_ns']
        history = trace['history']
        updates = math.ceil(trace['fit_rows'] / c['settings']['batch_size'])
        if (c['steps'][role] != trace['updates'] or not c['steps'][role] > 0
                or len(history) != trace['stopped_epoch']
                or [r['epoch'] for r in history] != list(range(1, len(history)+1))
                or any(r['updates'] != updates or r['gradient_evaluations'] != updates for r in history)
                or sum(r['updates'] for r in history) != c['steps'][role]):
            raise ValueError('Saved trajectory and native step ledger disagree')
    selection, refit = traces['selection'], traces['refit']
    best, epoch = math.inf, None
    for row in selection['history']:
        value = row['validation_mae']
        if not math.isfinite(value):
            raise ValueError('Nonfinite calibration trajectory')
        if value < best-c['settings']['min_delta']:
            best, epoch = value, row['epoch']
    if (epoch != selection['selected_epoch'] or refit['stopped_epoch'] != epoch
            or refit['selected_epoch'] != epoch or refit['fit_rows'] != len(plan['training_ids'])
            or refit['fit_ids_digest'] != digest(plan['training_ids'])):
        raise ValueError('Train-only selected epoch/refit identity differs')
    return sum(c['steps'].values())


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    execution = root/'execution'
    if output.exists() or output.parent != execution:
        raise ValueError('Fresh append-only path in original execution directory required')
    terminal = read(execution/'terminal.json')
    events = verify_events(terminal)
    manifest = read(root/'manifest.json')
    # Prediction, trajectory and quality reads occur only after all 72 actual exits.
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if torch.__version__ != '2.14.0+cpu':
        raise ValueError('Frozen CPU environment differs')
    for name, h in manifest['files'].items():
        if sha(name) != h:
            raise ValueError('Frozen dependency changed: '+name)
    report, audit = read(root/'report.json'), read(root/'independent-audit.json')
    if (terminal['report_sha256'] != sha(root/'report.json')
            or terminal['independent_audit_sha256'] != sha(root/'independent-audit.json')
            or audit['status'] != 'passed' or audit['report_sha256'] != sha(root/'report.json')
            or audit['manifest_sha256'] != sha(root/'manifest.json')
            or report['manifest_sha256'] != sha(root/'manifest.json')):
        raise ValueError('Terminal/report/independent arithmetic audit chain differs')
    starts = [read(execution/'started.json')]
    for e in events:
        if read(execution/(e['task']+'-terminal.json')) != e:
            raise ValueError('Aggregated event differs from original child exit')
        start = read(execution/(e['task']+'-start.json')); starts.append(start)
        if start['time_ns'] >= e['completed_ns']:
            raise ValueError('Child timestamp order differs')
    for start in starts:
        path = Path('/proc')/str(start['pid'])/'cmdline'
        if path.exists() and b'bf_tap_r2.ema_silu' in path.read_bytes():
            raise ValueError('Owned controller or child is still active')
    states = optimizers = steps = 0
    for seed in (42, 3407):
        for fold in range(5):
            plan = manifest['plans'][f's{seed}-f{fold}']
            reused = read(root/f'reuse-s{seed}-f{fold}.json')
            if (reused['status'] != 'passed' or reused['manifest_sha256'] != sha(root/'manifest.json')
                    or reused['cold_states'] != 6 or reused['new_optimizers'] != 0
                    or set(reused['receipts']) != {'42', '1042', '2042'}
                    or any(r['status'] != 'passed' or r['states'] != 2 for r in reused['receipts'].values())):
                raise ValueError('Original EMA cold coverage differs')
            states += 6
            for init in (42, 1042, 2042):
                d = root/f's{seed}-f{fold}-init{init}'
                c, cold = read(d/'complete.json'), read(d/'cold.json')
                identity = dict(source_directory=str(d), split_seed=seed, fold=fold, trial_id=f'SILU_EMA_INIT{init}')
                if (c['identity'] != identity or c['manifest_sha256'] != sha(root/'manifest.json')
                        or c['prediction_sha256'] != sha(d/'predictions.npz') or (d/'failure.json').exists()
                        or cold['status'] != 'passed' or cold['states'] != 2 or cold['pid'] == c['pid']
                        or cold['complete_sha256'] != sha(d/'complete.json')
                        or cold['native_optimizer_constructors'] != 2
                        or cold['native_steps'] != sum(c['steps'].values())
                        or c['settings'] != dict(manifest['training'], random_seed=init,
                                                arch_type='tabm', hidden_activation='SiLU')):
                    raise ValueError('New model/cold identity chain differs')
                traces = {}
                for role, h in c['state_hashes'].items():
                    path = d/(role+'.pt')
                    if sha(path) != h:
                        raise ValueError('Native state changed')
                    traces[role] = torch.load(path, map_location='cpu', weights_only=True)['trace']
                receipts = {r:read(d/(r+'-optimizer-start.json')) for r in ('selection', 'refit')}
                steps += verify_native(c, read(d/'estimator-start.json'), receipts, traces, plan)
                optimizers += 2; states += 2
    if ((states, optimizers, steps) != (120, 60, report['native_steps'])
            or (report['cold_base_states'], report['new_optimizer_constructors']) != (states, optimizers)):
        raise ValueError('Original/native model inventory differs')
    for seed, h in report['oof_sha256'].items():
        if sha(root/f'oof-s{seed}.npz') != h:
            raise ValueError('Independently audited OOF changed')
    gains = report['gains']
    if list(gains) != ['SILU_A100', 'SILU_A20']:
        raise ValueError('Frozen candidate order differs')
    for name, values in gains.items():
        if set(values) != {'42', '3407'} or any(not math.isfinite(v) for v in values.values()):
            raise ValueError('Complete within-seed development required')
        if any(abs(v-audit['gains'][name][s]) > 1e-10 for s,v in values.items()):
            raise ValueError('Independent arithmetic decision differs')
    eligible = [k for k in gains if min(gains[k].values()) > 0]
    chosen = max(eligible, key=lambda k:sum(gains[k].values())) if eligible else None
    if chosen != report['selected_for_confirmation'] or report['formal_promoted']:
        raise ValueError('Development selection/formal promotion scope differs')
    value = dict(status='passed', time_ns=time.time_ns(), actual_child_exit_codes=[e['exit_code'] for e in events],
        controller_and_owned_children_absent=True, frozen_files=len(manifest['files']), new_optimizer_constructors=optimizers,
        native_steps=steps, cold_states=states, selected_for_confirmation=chosen, formal_promoted=False,
        maximum_child_rss_mib=max(e['peak_rss_mib'] for e in events),
        original_terminal_sha256=sha(execution/'terminal.json'), report_sha256=sha(root/'report.json'),
        arithmetic_audit_sha256=sha(root/'independent-audit.json'), manifest_sha256=sha(root/'manifest.json'),
        auditor_sha256=sha(__file__), oof_sha256=report['oof_sha256'], new_fits=0,new_predictions=0,new_packages=0)
    with output.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k != 'actual_child_exit_codes'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3,12) or any(os.environ.get(k) != '1' for k in
            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')):
        raise ValueError('Locked Python3.12 with pre-import numerical threads=1 required')
    run(args.root, args.output)
