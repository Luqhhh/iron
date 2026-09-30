"""Frozen, one-shot complete-size synthetic DNNR cost/cold admission.

No official table or reference cache is read. Admission is a resource
prerequisite only; it does not launch any official-data phase automatically.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import yaml

from .data import FEATURES
from .dnnr_model import ARMS, Regressor, Settings
from .dnnr_execution import execute_unit, audit_unit
from .dnnr_audit import verify_saved
from .dnnr_ledger import ReservationLedger, file_hash, write_new

SPEC_PATH = 'configs/dnnr_taylor/SPEC.yaml'
TASK = dict(target='tap_time_len', seed=42, fold=0)
LIMITS = dict(pair_unit=1, estimator=7, metric_epoch=3, derivative_bank=5)
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
PACKAGES = ('numpy', 'pandas', 'scikit-learn', 'pytest', 'psutil', 'PyYAML')
PROBE = dict(seed=57331, rows=2755, train_rows=2204, query_rows=551, spout_count=4,
    encoded_dimension=26, formula='30+2*sin(x0)+x1*x2+0.5*x3+0.1*epsilon',
    all_arms_must_beat_training_median=True, cold_tolerance=1e-8,
    optional_upper_bound='if_outer_epoch0_selected_fit_and_audit_one_extra_epoch1_model', limits=LIMITS)
RESOURCES = dict(workers=4, numeric_threads=1, maximum_worker_mib=1536,
    free_memory_rule='4*maximum_measured_peak+1024',
    projection_seconds='20*upper_pair_execution_and_cold_audit_seconds/4*1.5+300',
    projection_maximum_seconds=7200, formal_encoded_dimension_must_not_exceed_probe=True)


def anchored(path, sha):
    if not sha or file_hash(path) != sha:
        raise ValueError('Externally anchored JSON hash required')
    return json.loads(Path(path).read_text())


def runtime():
    if sys.version_info[:2] != (3, 12) or any(os.environ.get(k) != '1' for k in THREADS):
        raise ValueError('Locked Python3.12 with four numeric thread limits required')
    return dict(python=platform.python_version(), executable=sys.executable,
                versions={k: importlib.metadata.version(k) for k in PACKAGES},
                thread_variables={k: os.environ[k] for k in THREADS})


def source_hashes(workspace):
    workspace = Path(workspace)
    paths = [p for folder in ('src', 'tests', 'scripts') for p in (workspace/folder).rglob('*.py')]
    paths += [p for p in (workspace/'configs').rglob('*.yaml')
              if not p.name.endswith('.local.yaml') and not p.is_symlink()]
    paths += [workspace/'pyproject.toml', workspace/'uv.lock']
    if any(p.is_symlink() or not p.resolve().is_relative_to(workspace.resolve()) for p in paths):
        raise ValueError('Source paths escape the actual worktree')
    return {str(p.relative_to(workspace)): file_hash(p) for p in sorted(paths)}


def private_path(workspace, path):
    workspace, path = Path(workspace).resolve(), Path(path).absolute()
    if (not path.is_relative_to(workspace/'local')
            or path.resolve() != path or not path.resolve().is_relative_to(workspace/'local')):
        raise ValueError('Direct private worktree-local output path required')
    return path


def freeze_probe(workspace, output, engineering_receipt, receipt_sha256):
    workspace = Path(workspace).resolve()
    output = private_path(workspace, output)
    receipt = anchored(engineering_receipt, receipt_sha256)
    snapshot_path = private_path(workspace, workspace/receipt['source_snapshot'])
    snapshot = anchored(snapshot_path, receipt['source_snapshot_sha256'])
    current = source_hashes(workspace)
    if (receipt['status'] != 'passed' or receipt['source_files'] != len(current)
            or receipt['focused_tests'] < 37 or receipt['locked_full_tests'] < 1347
            or snapshot != current or receipt['python'] != platform.python_version()
            or receipt['executable'] != sys.executable):
        raise ValueError('Exact-source locked engineering receipt required')
    for name, sha in receipt['logs'].items():
        p = private_path(workspace, workspace/name)
        if file_hash(p) != sha:
            raise ValueError('Engineering log changed')
    actual_runtime = runtime()
    if (any(actual_runtime['versions'][k] != v for k,v in receipt['versions'].items())
            or actual_runtime['thread_variables'] != receipt['thread_variables']):
        raise ValueError('Checked runtime differs')
    spec = yaml.safe_load((workspace/SPEC_PATH).read_text())
    if (spec['training'] != asdict(Settings()) or spec['arms'] != list(ARMS)
            or spec['synthetic_resource_probe'] != PROBE or spec['resources'] != RESOURCES):
        raise ValueError('Probe/model/resource spec differs from frozen implementation')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(version='dnnr-synthetic-admission-v1', workspace=str(workspace),
        source_hashes=current, runtime=actual_runtime, settings=spec['training'],
        engineering_receipt=str(Path(engineering_receipt).resolve()), engineering_receipt_sha256=receipt_sha256,
        probe=PROBE, resources=RESOURCES, official_label_reads=0, official_fits=0,
        automatic_formal_execution=False, release_authorized=False)
    write_new(output/'manifest.json', manifest)
    return dict(manifest=str(output/'manifest.json'), sha256=file_hash(output/'manifest.json'))


def verify_manifest(path, sha256):
    manifest = anchored(path, sha256)
    workspace = Path(manifest['workspace']).resolve()
    private_path(workspace, path)
    if (manifest['version'] != 'dnnr-synthetic-admission-v1'
            or manifest['source_hashes'] != source_hashes(workspace)
            or manifest['runtime'] != runtime() or manifest['settings'] != asdict(Settings())
            or manifest['probe'] != PROBE or manifest['resources'] != RESOURCES
            or manifest['official_label_reads'] != 0 or manifest['official_fits'] != 0
            or manifest['automatic_formal_execution'] or manifest['release_authorized']):
        raise ValueError('Synthetic manifest source/runtime/spec/scope changed')
    anchored(manifest['engineering_receipt'], manifest['engineering_receipt_sha256'])
    return manifest


def synthetic_data():
    rng = np.random.default_rng(PROBE['seed'])
    x = rng.normal(size=(PROBE['rows'], len(FEATURES)))
    y = 30+2*np.sin(x[:, 0])+x[:, 1]*x[:, 2]+.5*x[:, 3]+.1*rng.normal(size=len(x))
    frame = pd.DataFrame(x, columns=FEATURES)
    frame['sample_id'] = [f'dnnr-resource-{i:05d}' for i in range(len(x))]
    frame['spout_no'] = np.arange(len(x)) % PROBE['spout_count']+1
    n = PROBE['train_rows']
    return frame.iloc[:n].reset_index(drop=True), y[:n], frame.iloc[n:].reset_index(drop=True), y[n:]


def resource_decision(measurement, available_mib):
    if (measurement['train_rows'] != PROBE['train_rows'] or measurement['query_rows'] != PROBE['query_rows']
            or measurement['encoded_dimension'] != PROBE['encoded_dimension']
            or measurement['models_checked'] not in (6, 7) or set(measurement['mae']) != set(ARMS)
            or type(measurement['selected_metric_epochs']) is not int
            or measurement['selected_metric_epochs'] not in (0, 1)
            or measurement['models_checked'] != 7-measurement['selected_metric_epochs']):
        raise ValueError('Probe coverage/upper-bound model coverage differs')
    values = [available_mib, measurement['peak_mib'], measurement['maximum_audit_difference'],
        measurement['median_mae'], measurement['pair_execution_seconds'], measurement['pair_audit_seconds'],
        measurement['upper_bound_extra_seconds'], *measurement['mae'].values()]
    if (not np.isfinite(values).all() or min(values) < 0
            or measurement['peak_mib'] <= 0 or measurement['pair_execution_seconds'] <= 0
            or measurement['pair_audit_seconds'] <= 0
            or (not measurement['selected_metric_epochs'] and measurement['upper_bound_extra_seconds'] <= 0)
            or (measurement['selected_metric_epochs'] and measurement['upper_bound_extra_seconds'] != 0)):
        raise ValueError('Invalid resource or accuracy measurements')
    upper = measurement['pair_execution_seconds']+measurement['pair_audit_seconds']+measurement['upper_bound_extra_seconds']
    projection = 20*upper/4*1.5+300
    required = 4*measurement['peak_mib']+1024
    checks = dict(synthetic_quality=all(v < measurement['median_mae'] for v in measurement['mae'].values()),
        cold_inference=measurement['maximum_audit_difference'] <= PROBE['cold_tolerance'],
        worker_peak=measurement['peak_mib'] <= RESOURCES['maximum_worker_mib'],
        available_memory=available_mib >= required,
        development_cost=projection <= RESOURCES['projection_maximum_seconds'])
    return dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
        upper_pair_execution_and_cold_audit_seconds=upper, projected_development_seconds=projection,
        maximum_worker_mib=measurement['peak_mib'], available_mib=available_mib, required_available_mib=required)


def measure(manifest_path, manifest_sha256, policy_sha256):
    manifest = verify_manifest(manifest_path, manifest_sha256)
    root = private_path(manifest['workspace'], Path(manifest_path).resolve().parent/'preflight')
    ledger = ReservationLedger.open(root/'ledger', policy_sha256)
    training, y, query, truth = synthetic_data()
    settings = Settings(**manifest['settings'])
    start = time.perf_counter()
    complete = execute_unit(TASK, training, y, query, root/'unit', ledger.root, policy_sha256, settings=settings)
    execution_seconds = time.perf_counter()-start
    start = time.perf_counter()
    predictions, audit = audit_unit(root/'unit', complete['complete_sha256'], TASK, training, y, query,
                                    settings=settings, ledger_root=ledger.root)
    audit_seconds = time.perf_counter()-start
    selected = audit['selected_metric_epochs']
    extra_seconds, extra_audit = 0., None
    if not selected:
        # A zero-epoch decision is cheaper than the maximum official procedure.
        # Exercise the missing whole-outer-training metric epoch before cost admission.
        def observer(kind, payload):
            return ledger.event(kind, ('synthetic_upper_bound', TASK['target'], TASK['seed'], TASK['fold'], kind), payload)
        start = time.perf_counter()
        extra = Regressor('DNNR_LEARNED', settings).fit(training, y, metric_epochs=1, observer=observer)
        path = root/'upper-bound-epoch1.npz'; sha = extra.save(path)
        prediction = extra.predict(query)
        with (root/'upper-bound-prediction.npy').open('xb') as stream:
            np.save(stream, prediction, allow_pickle=False)
        extra_audit = verify_saved(path, sha, training, y, query, prediction)
        extra_seconds = time.perf_counter()-start
    differences = [r[k] for r in audit['models'].values() for k in
        ('cold_order_chunk_independent_max_difference', 'derivative_witness_max_difference', 'metric_witness_max_difference')]
    if extra_audit:
        differences += [extra_audit[k] for k in
            ('cold_order_chunk_independent_max_difference', 'derivative_witness_max_difference', 'metric_witness_max_difference')]
    measured = dict(train_rows=len(training), query_rows=len(query), encoded_dimension=PROBE['encoded_dimension'],
        models_checked=6+int(extra_audit is not None), selected_metric_epochs=selected,
        mae={a: float(np.abs(predictions[a]-truth).mean()) for a in ARMS},
        median_mae=float(np.abs(truth-np.median(y)).mean()), maximum_audit_difference=max(differences),
        peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        pair_execution_seconds=execution_seconds, pair_audit_seconds=audit_seconds,
        upper_bound_extra_seconds=extra_seconds, unit_complete_sha256=complete['complete_sha256'],
        pair_audit=audit, extra_audit=extra_audit, manifest_sha256=manifest_sha256, policy_sha256=policy_sha256)
    verify_manifest(manifest_path, manifest_sha256)
    write_new(root/'measurement.json', measured)
    return measured


def verify_probe_counts(root, policy_sha256, selected):
    ledger = ReservationLedger.open(Path(root)/'ledger', policy_sha256)
    counts = ledger.inspect()
    expected = dict(pair_unit=1, estimator=7-selected, metric_epoch=2, derivative_bank=5-selected)
    if (ledger.limits != LIMITS or counts['started'] != expected or counts['completed'] != expected
            or any(v for k in ('failed','incomplete') for v in counts[k].values())):
        raise ValueError('Extra, failed, missing or incomplete synthetic reservations')
    return counts


def run_probe(manifest_path, manifest_sha256):
    manifest = verify_manifest(manifest_path, manifest_sha256)
    root = private_path(manifest['workspace'], Path(manifest_path).resolve().parent/'preflight')
    root.mkdir(exist_ok=False)
    write_new(root/'started.json', dict(manifest_sha256=manifest_sha256, official_fits=0))
    try:
        ledger = ReservationLedger.create(root/'ledger', LIMITS)
        command = [sys.executable, '-m', 'bf_tap_r2.dnnr_preflight', 'worker', '--manifest', str(manifest_path),
            '--manifest-sha256', manifest_sha256, '--policy-sha256', ledger.policy_sha256]
        with (root/'worker.log').open('x') as log:
            subprocess.run(command, cwd=manifest['workspace'], check=True, stdout=log, stderr=subprocess.STDOUT)
        measured = json.loads((root/'measurement.json').read_text())
        if measured['manifest_sha256'] != manifest_sha256 or measured['policy_sha256'] != ledger.policy_sha256:
            raise ValueError('Wrong synthetic measurement identity')
        counts = verify_probe_counts(root, ledger.policy_sha256, measured['selected_metric_epochs'])
        decision = resource_decision(measured, psutil.virtual_memory().available/1024**2)
        hashes = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()}
        verify_manifest(manifest_path, manifest_sha256)
        report = dict(**decision, measurement=measured, counts=counts, artifact_hashes=hashes,
            manifest_sha256=manifest_sha256, policy_sha256=ledger.policy_sha256, official_fits=0,
            automatic_formal_execution=False, release_authorized=False)
        write_new(root/'admission.json', report)
        if decision['status'] != 'passed':
            raise ValueError('Frozen DNNR synthetic resource admission failed')
    except BaseException as exc:
        write_new(root/'failed.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    return dict(admission_sha256=file_hash(root/'admission.json'), **decision)


def verify_admission(path, sha256, manifest_path, manifest_sha256):
    manifest = verify_manifest(manifest_path, manifest_sha256)
    path = private_path(manifest['workspace'], path)
    root = path.parent
    if root != Path(manifest_path).resolve().parent/'preflight' or (root/'failed.json').exists():
        raise ValueError('Failed or foreign probe directory')
    report = anchored(path, sha256)
    if (report['status'] != 'passed' or report['manifest_sha256'] != manifest_sha256
            or report['official_fits'] != 0 or report['automatic_formal_execution'] or report['release_authorized']):
        raise ValueError('Failed/foreign/scope-changed synthetic admission')
    if {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} != set(report['artifact_hashes']) | {'admission.json'}:
        raise ValueError('Missing or unexpected probe artifact')
    for name, sha in report['artifact_hashes'].items():
        p = root/name
        if p.resolve() != p or not p.is_relative_to(root) or file_hash(p) != sha:
            raise ValueError('Probe artifact changed or escaped')
    measured = anchored(root/'measurement.json', report['artifact_hashes']['measurement.json'])
    if measured != report['measurement']:
        raise ValueError('Probe measurement changed')
    if verify_probe_counts(root, report['policy_sha256'], measured['selected_metric_epochs']) != report['counts']:
        raise ValueError('Probe ledger changed')
    training, y, query, truth = synthetic_data()
    predictions, audit = audit_unit(root/'unit', measured['unit_complete_sha256'], TASK, training, y, query,
        settings=Settings(**manifest['settings']), ledger_root=root/'ledger')
    if audit != measured['pair_audit']:
        raise ValueError('Fresh pair cold audit differs from measured audit')
    for arm in ARMS:
        if float(np.abs(predictions[arm]-truth).mean()) != measured['mae'][arm]:
            raise ValueError('Synthetic MAE arithmetic differs')
    if not measured['selected_metric_epochs']:
        path = root/'upper-bound-epoch1.npz'
        extra = verify_saved(path, report['artifact_hashes'][path.name], training, y, query,
            np.load(root/'upper-bound-prediction.npy', allow_pickle=False))
        if extra != measured['extra_audit']:
            raise ValueError('Fresh upper-bound cold audit differs')
    decision = resource_decision(measured, report['available_mib'])
    if any(report[k] != v for k,v in decision.items()):
        raise ValueError('Synthetic resource arithmetic differs')
    if psutil.virtual_memory().available/1024**2 < report['required_available_mib']:
        raise ValueError('Current available memory below admitted bound')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['freeze', 'run', 'worker'])
    p.add_argument('--workspace', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--receipt', type=Path); p.add_argument('--receipt-sha256')
    p.add_argument('--manifest', type=Path); p.add_argument('--manifest-sha256'); p.add_argument('--policy-sha256')
    a = p.parse_args()
    if a.action == 'freeze':
        result = freeze_probe(a.workspace, a.output, a.receipt, a.receipt_sha256)
    elif a.action == 'run':
        result = run_probe(a.manifest, a.manifest_sha256)
    else:
        result = measure(a.manifest, a.manifest_sha256, a.policy_sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
