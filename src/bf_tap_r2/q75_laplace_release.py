"""Gated full-data fixed Laplace time fit and audited single-column Q75 release."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import yaml

from .data import FEATURES, TARGETS
from .ema_evaluation_diagnostics import sha, write_new
from .ema_fusion_selection import runtime, verify_files
from .ema_reference_ledger import KINDS, NativeLedger, binding_sources, native_hooks
from .q75_gaussian_confirmation import gate, hooks, read
from .q75_gaussian_time import cold_state, compose, forbidden, selected_epoch
from .submission import ZIP_NAME, deny_training_reads, package, validate_result
from .v2_release import load_v2
from .v5_library import load_v5_training_frame
from .v5_package import payload_with_parent_other_column
from .laplace_time_model import LaplaceRegressor
from .q75_laplace_confirmation import laplace_cold_state
from .v33_run import group_safe_inner_folds

WORK = Path(__file__).resolve().parents[2]
SPEC = 'configs/q75_laplace_release/SPEC.json'
PROTOCOL = 'docs/q75_laplace_release/PREREGISTRATION.md'


def validate_spec(spec):
    expected = dict(candidate='LAPLACE_FIXED_A20', arm='LAPLACE_FIXED', weight=.2, training_changes=[],
        full_data_procedures=1, optimizer_runs=2, new_states=2, engineering_optimizer_runs=0,
        new_cv_fits=0, reference_fits=0, workers=1, numerical_threads=1,
        torch_interop_threads=1, maximum_runtime_seconds=None, automatic_retries=False,
        requires_confirmation_formal_promotion=True, packages=1, desktop_writes=0, agent_uploads=0)
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Unregistered release scope or budget')


def zip_payload(path):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ['result.csv'] or archive.testzip() is not None:
            raise ValueError('Unexpected ZIP members or invalid CRC')
        return archive.read('result.csv')


def verify_payload(payload, parent, ids, member):
    rows, old = validate_result(payload, ids), validate_result(parent, ids)
    expected = compose(np.array([float(r['pred_tap_time_len']) for r in old]), member)
    if any(row['pred_tap_iron'] != previous['pred_tap_iron'] for row, previous in zip(rows, old)):
        raise ValueError('Unchanged iron field string differs')
    actual = np.array([float(r['pred_tap_time_len']) for r in rows])
    if not np.array_equal(actual, expected):
        raise ValueError('Serialized fixed endpoint differs')
    return dict(rows=len(rows), unique_ids=len(set(ids)), iron_string_mismatches=0,
        fixed_weight=.2, finite_nonnegative=True, min_time=float(actual.min()), max_time=float(actual.max()))


def access(out, stage):
    with (out / 'access.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(stage=stage, time_ns=time.time_ns(),
            frozen_files_sha256=sha(out / 'frozen-files.json'),
            scope='authorized_round2_only', protected_prelim_targets_read=False)) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def freeze(out, checks):
    spec = read(WORK / SPEC)
    validate_spec(spec)
    main, confirm = Path(spec['main_root']), Path(spec['main_root']) / spec['confirmation']
    if out.exists() or not out.is_relative_to(main / 'local/runs'):
        raise ValueError('Fresh private output required')
    versions = runtime(spec)
    torch.set_num_interop_threads(1)
    current = read(main / 'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(current[k] != v for k, v in spec['reference'].items()):
        raise ValueError('Current reference changed')
    terminal, report, independent = [read(confirm / f'{n}.json') for n in ('terminal', 'report', 'independent-score')]
    if (terminal['status'] != 'passed' or terminal['actual_exit_codes'] != [0]*4
            or terminal['optimizer_runs'] != 20 or terminal['cold_states'] != 20
            or not gate(report['gains'])['formal_promoted'] or not report['formal_promoted']
            or independent['status'] != 'passed' or not independent['formal_promoted']
            or independent['report_sha256'] != sha(confirm / 'report.json')):
        raise ValueError('Audited formal four-seed promotion required')
    verify_files({str(confirm / name): h for name, h in terminal['artifacts'].items()})
    cm = read(confirm / 'manifest.json')
    verify_files(cm['files'])
    engineering = main / spec['engineering']
    et = read(engineering / 'terminal.json')
    if et['status'] != 'passed' or et['actual_exit_codes'] != [0, 0] or et['optimizer_runs'] != 4 or et['cold_states'] != 4:
        raise ValueError('Original full-shape G0 required')
    verify_files({str(engineering / name): h for name, h in et['artifacts'].items()})
    for rel in ('src/bf_tap_r2/laplace_time_model.py', 'src/bf_tap_r2/v33_mixture.py', 'src/bf_tap_r2/v3_6_networks.py'):
        original = next(h for p, h in cm['sources'].items() if p.endswith('/'+rel))
        if sha(WORK / rel) != original:
            raise ValueError('Scientific source differs from confirmation')
    receipt = read(checks)
    if (receipt['exit_code'] != 0 or receipt['python_version'][:4] != '3.12' or receipt['optimizer_runs'] != 0
            or receipt['optimizer_constructor_attempts'] != 0 or sha(receipt['junit_path']) != receipt['junit_sha256']):
        raise ValueError('Locked Python3.12 zero-fit release checks required')
    verify_files(receipt['source_hashes'])
    parent = main / spec['parent_package']
    if sha(parent) != spec['reference']['zip_sha256']:
        raise ValueError('Parent package identity differs')
    paths = list((WORK / 'src').rglob('*.py')) + [WORK / p for p in (SPEC, PROTOCOL,
        'uv.lock', 'pyproject.toml', spec['original_specification'],
        'scripts/q75_laplace_release.py', 'scripts/check_q75_laplace_release.py', 'scripts/observe_ema_fusion_selection.py',
        'tests/test_q75_laplace_release.py')]
    paths += [main / p for p in ('configs/protection.yaml', 'configs/data.local.yaml',
        'local/authorizations/optimization-standing-20261002-r1.json')]
    paths += [main / f'复赛_{stage}/{stage}_{kind}.csv' for stage in ('train','test') for kind in ('samples','features')]
    paths += [main / '复赛_test/result_template.csv', parent, checks, Path(receipt['junit_path'])]
    paths += [confirm / n for n in ('terminal.json','manifest.json','report.json','independent-score.json','cold-audit.json')]
    paths += [engineering / n for n in ('terminal.json','manifest.json','cold-audit.json')]
    files = dict(cm['files'])
    files.update({str(p): sha(p) for p in paths})
    verify_files(files)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'frozen-files.json', files)
    access(out, 'freeze_before_authorized_round2_labels')
    training, query = load_v5_training_frame(main), load_v2(main / '复赛_test', 'test', 322)
    if len(training) != 2754 or set(training.sample_id) & set(query.sample_id) or any(t in query for t in TARGETS):
        raise ValueError('Full training/query identity invalid')
    ids = query.sample_id.to_numpy(str)
    validate_result(zip_payload(parent), ids)
    for name, frame in [('training', training), ('query', query)]:
        arrays = dict(x=frame[list(FEATURES)].to_numpy(float), spouts=frame.spout_no.to_numpy(), ids=frame.sample_id.to_numpy(str))
        if name == 'training':
            arrays['y'] = frame.tap_time_len.to_numpy(float)
        with (out / f'{name}.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
    with (out / 'parent.csv').open('xb') as stream:
        stream.write(zip_payload(parent))
    native = cm['native']
    sources = {str(p): h for p, h in files.items() if Path(p).is_relative_to(WORK)}
    native_sources = {**binding_sources(hooks()), **{str(WORK / p): sha(WORK / p) for p in (
        'src/bf_tap_r2/laplace_time_model.py', 'src/bf_tap_r2/v33_mixture.py', 'src/bf_tap_r2/v3_6_networks.py', 'src/bf_tap_r2/q75_laplace_release.py')}}
    write_new(out / 'manifest.json', dict(spec=spec, native=native, files=files, sources=sources,
        native_sources=native_sources, versions=versions, python=sys.version, frozen_ns=time.time_ns(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=WORK, text=True).strip(),
        frozen_files_sha256=sha(out / 'frozen-files.json'),
        inputs={name: sha(out / name) for name in ('training.npz','query.npz','parent.csv')}))


def context(out, *, no_fit=True, label_free=False):
    manifest = read(out / 'manifest.json')
    validate_spec(manifest['spec'])
    runtime(manifest['spec'])
    torch.set_num_interop_threads(1)
    if label_free:
        sys.addaudithook(deny_training_reads)
        def deny_local_training(event, args):
            if event == 'open' and isinstance(args[0], (str, bytes)) and Path(os.fsdecode(args[0])).resolve() == out / 'training.npz':
                raise PermissionError('Label-free inference forbids frozen training input')
        sys.addaudithook(deny_local_training)
        verify_files(manifest['sources'])
    else:
        verify_files(manifest['files'])
    if sha(out / 'frozen-files.json') != manifest['frozen_files_sha256']:
        raise ValueError('Frozen dependency record changed')
    for name, value in manifest['inputs'].items():
        if label_free and name == 'training.npz':
            continue
        if sha(out / name) != value:
            raise ValueError('Frozen input changed')
    if no_fit:
        LaplaceRegressor.initialize = LaplaceRegressor.train = forbidden
        torch.optim.Adam = torch.optim.AdamW = forbidden
    access(out, 'label_free_query_only' if label_free else 'authorized_round2_training_audit')
    return manifest


def frame_from(path):
    with np.load(path, allow_pickle=False) as saved:
        frame = pd.DataFrame(saved['x'], columns=FEATURES)
        frame['sample_id'], frame['spout_no'] = saved['ids'], saved['spouts']
        if 'y' in saved:
            frame['tap_time_len'] = saved['y']
    return frame


def partitions(training, native):
    settings = native['calibration']
    folds = group_safe_inner_folds(training, seed=settings['split_seed'], n_splits=settings['n_splits'])['fold']
    mask = folds == settings['held_fold']
    fitting, calibration = training.loc[~mask].reset_index(drop=True), training.loc[mask].reset_index(drop=True)
    groups = lambda f: set(pd.util.hash_pandas_object(f[list(FEATURES)], index=False))
    if groups(fitting) & groups(calibration) or set(fitting.sample_id) & set(calibration.sample_id):
        raise ValueError('Inner partition overlap')
    return fitting, calibration


def release_ledger(out, training_ids, query_ids, sources):
    return NativeLedger(out / 'model/native', identity=dict(source_directory=str(out),
        split_seed=-1, fold=-1, trial_id='LAPLACE_FIXED_A20'),
        expected={k:2 if k=='torch_optimizer' else 0 for k in KINDS},
        training_ids=training_ids, query_ids=query_ids, source_hashes=sources)


def train(out):
    manifest = context(out, no_fit=False)
    training, query = frame_from(out / 'training.npz'), frame_from(out / 'query.npz')
    fitting, calibration = partitions(training, manifest['native'])
    settings = manifest['native']['training']
    unit = out / 'model'
    unit.mkdir(exist_ok=False)
    ledger = release_ledger(out, training.sample_id.tolist(), query.sample_id.tolist(), manifest['native_sources'])
    try:
        with native_hooks(hooks()):
            with ledger.partition(fitting.sample_id.tolist(), calibration.sample_id.tolist()):
                selector = LaplaceRegressor('LAPLACE_FIXED', settings).initialize(fitting, fitting.tap_time_len.to_numpy())
                epoch = selector.train(settings['max_epochs'], (calibration, calibration.tap_time_len.to_numpy()))
            cp = selector.predict(calibration.drop(columns=list(TARGETS), errors='ignore'))
            selector.save(unit / 'calibration_model.pt')
            with ledger.partition(training.sample_id.tolist()):
                model = LaplaceRegressor('LAPLACE_FIXED', settings).initialize(training, training.tap_time_len.to_numpy())
                model.train(epoch)
            prediction = model.predict(query)
            compose(np.array([float(r['pred_tap_time_len']) for r in validate_result((out / 'parent.csv').read_bytes(), query.sample_id)]), prediction)
            model.save(unit / 'model.pt')
        receipt = ledger.close()
        write_new(unit / 'metadata.json', dict(calibration=selector.metadata(), refit=model.metadata()))
        with (unit / 'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, member=prediction, query_ids=query.sample_id.to_numpy(str),
                calibration_prediction=cp, calibration_ids=calibration.sample_id.to_numpy(str))
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if rss > manifest['spec']['max_worker_rss_mib']:
            raise ValueError('Training RSS gate failed')
        write_new(out / 'warm-complete.json', dict(pid=os.getpid(), selected_epoch=epoch,
            counts=receipt['counts'], manifest_sha256=sha(out / 'manifest.json'), peak_rss_mib=rss,
            hashes={str(p.relative_to(out)): sha(p) for p in unit.rglob('*') if p.is_file()}))
    except BaseException as error:
        write_new(out / 'failure.json', dict(error=repr(error), automatic_retry=False))
        raise


def warm_identity(out):
    warm = read(out / 'warm-complete.json')
    if warm['manifest_sha256'] != sha(out / 'manifest.json') or warm['pid'] == os.getpid():
        raise ValueError('Independent process or warm identity invalid')
    verify_files({str(out / p): h for p, h in warm['hashes'].items()})
    return warm


def cold(out):
    manifest = context(out)
    warm = warm_identity(out)
    training, query = frame_from(out / 'training.npz'), frame_from(out / 'query.npz')
    fitting, calibration = partitions(training, manifest['native'])
    unit, settings = out / 'model', manifest['native']['training']
    meta = read(unit / 'metadata.json')
    epoch, best = selected_epoch(meta['calibration']['history'], settings)
    if (epoch != warm['selected_epoch'] or epoch != meta['calibration']['selected_epoch']
            or epoch != meta['refit']['selected_epoch'] or epoch != meta['refit']['stopped_epoch']
            or meta['calibration']['stopped_epoch'] != len(meta['calibration']['history'])
            or [r['epoch'] for r in meta['refit']['history']] != list(range(1,epoch+1))):
        raise ValueError('Selector/refit epoch identity invalid')
    native = read(unit / 'native/scope-complete.json')
    if native['counts'] != warm['counts'] or native['counts'] != {k: 2 if k == 'torch_optimizer' else 0 for k in KINDS}:
        raise ValueError('Actual native budget invalid')
    for index, fit, held in [(1,fitting,calibration),(2,training,None)]:
        call = read(unit / f'native/call-{index:04d}/start.json')
        if (call['partition']['training_ids'] != fit.sample_id.tolist()
                or call['partition']['calibration_ids'] != ([] if held is None else held.sample_id.tolist())):
            raise ValueError('Native training/calibration partition mismatch')
    differences = []
    with np.load(unit / 'predictions.npz', allow_pickle=False) as saved:
        for filename, fit, held, field, ids, md in [
            ('calibration_model.pt',fitting,calibration,'calibration_prediction','calibration_ids',meta['calibration']),
            ('model.pt',training,query,'member','query_ids',meta['refit'])]:
            np.testing.assert_array_equal(saved[ids], held.sample_id.to_numpy(str))
            prediction, difference = laplace_cold_state(unit / filename, fit,
                held.drop(columns=list(TARGETS), errors='ignore'), 'tap_time_len', md, settings, saved[field])
            if filename == 'calibration_model.pt':
                score = np.abs(prediction-calibration.tap_time_len.to_numpy()).mean()/md['target_std']
                if abs(score-best)>1e-10:
                    raise ValueError('Selected checkpoint does not match best traced metric')
            differences.append(difference)
    if max(differences)>manifest['spec']['cold_predict_atol']:
        raise ValueError('Cold prediction tolerance exceeded')
    write_new(out / 'cold-audit.json', dict(status='passed', cold_states=2, optimizer_runs=2,
        max_difference=max(differences), warm_sha256=sha(out / 'warm-complete.json'),
        selected_epoch=epoch, manifest_sha256=sha(out / 'manifest.json')))


def predict(out):
    manifest = context(out, label_free=True)
    warm_identity(out)
    cold_receipt = read(out / 'cold-audit.json')
    if cold_receipt['status'] != 'passed' or cold_receipt['warm_sha256'] != sha(out / 'warm-complete.json'):
        raise ValueError('Independent cold state audit required')
    with np.load(out / 'query.npz', allow_pickle=False) as values:
        if set(values.files) != {'x','spouts','ids'}:
            raise ValueError('Inference input must be label-free')
    query = frame_from(out / 'query.npz')
    model = LaplaceRegressor.load(out / 'model/model.pt')
    if model.recipe != 'GAUSS1' or model.arm != 'LAPLACE_FIXED' or model.settings != manifest['native']['training']:
        raise ValueError('Cold model recipe/settings differ')
    member = model.predict(query)
    with np.load(out / 'model/predictions.npz', allow_pickle=False) as previous:
        np.testing.assert_array_equal(query.sample_id.to_numpy(str), previous['query_ids'])
        difference = float(np.max(np.abs(member-previous['member'])))
    if difference > manifest['spec']['cold_predict_atol']:
        raise ValueError('Label-free cold inference differs')
    with (out / 'cold-predictions.npz').open('xb') as stream:
        np.savez_compressed(stream, ids=query.sample_id.to_numpy(str), member=member)
    write_new(out / 'prediction-audit.json', dict(status='passed', pid=os.getpid(),
        no_training_reads=True, new_fits=0, max_difference=difference,
        model_sha256=sha(out / 'model/model.pt'), query_sha256=sha(out / 'query.npz'),
        predictions_sha256=sha(out / 'cold-predictions.npz'), cold_audit_sha256=sha(out / 'cold-audit.json')))


def release(out):
    context(out, label_free=True)
    warm_identity(out)
    receipt = read(out / 'prediction-audit.json')
    if (receipt['status'] != 'passed' or not receipt['no_training_reads']
            or receipt['predictions_sha256'] != sha(out / 'cold-predictions.npz')
            or receipt['model_sha256'] != sha(out / 'model/model.pt')
            or receipt['cold_audit_sha256'] != sha(out / 'cold-audit.json')):
        raise ValueError('Label-free cold prediction binding required')
    with np.load(out / 'cold-predictions.npz', allow_pickle=False) as saved:
        ids, member = saved['ids'], saved['member']
        query = frame_from(out / 'query.npz')
        np.testing.assert_array_equal(ids, query.sample_id.to_numpy(str))
    parent = (out / 'parent.csv').read_bytes()
    rows = validate_result(parent, ids)
    endpoint = compose(np.array([float(r['pred_tap_time_len']) for r in rows]), member)
    payload = payload_with_parent_other_column(ids, 'tap_time_len', endpoint, [r['pred_tap_iron'] for r in rows])
    evidence = verify_payload(payload, parent, ids, member)
    directory = out / 'LAPLACE_FIXED_A20'
    directory.mkdir(exist_ok=False)
    package(directory, payload, ids)
    write_new(out / 'package.json', dict(status='written_pending_independent_audit', pid=os.getpid(),
        candidate='LAPLACE_FIXED_A20', result_sha256=sha(directory / 'result.csv'),
        zip_sha256=sha(directory / ZIP_NAME), evidence=evidence,
        prediction_audit_sha256=sha(out / 'prediction-audit.json')))


def audit(out):
    manifest = context(out, label_free=True)
    warm_identity(out)
    written = read(out / 'package.json')
    receipt = read(out / 'prediction-audit.json')
    if (written['pid'] == os.getpid() or written['prediction_audit_sha256'] != sha(out / 'prediction-audit.json')
            or receipt['predictions_sha256'] != sha(out / 'cold-predictions.npz')):
        raise ValueError('Independent package/prediction identity required')
    path = out / 'LAPLACE_FIXED_A20'
    if written['zip_sha256'] != sha(path / ZIP_NAME) or written['result_sha256'] != sha(path / 'result.csv'):
        raise ValueError('Written package identity changed')
    payload, parent = zip_payload(path / ZIP_NAME), (out / 'parent.csv').read_bytes()
    if payload != (path / 'result.csv').read_bytes():
        raise ValueError('ZIP/CSV bytes differ')
    with np.load(out / 'cold-predictions.npz', allow_pickle=False) as saved:
        ids, member = saved['ids'], saved['member']
        np.testing.assert_array_equal(ids, frame_from(out / 'query.npz').sample_id.to_numpy(str))
    evidence = verify_payload(payload, parent, ids, member)
    # Independent scalar reconstruction also guards against a shared composition error.
    rows, old = validate_result(payload, ids), validate_result(parent, ids)
    independent = [.8*float(row['pred_tap_time_len'])+.2*float(value) for row,value in zip(old, member)]
    if [float(row['pred_tap_time_len']) for row in rows] != independent:
        raise ValueError('Independent scalar package arithmetic mismatch')
    write_new(out / 'package-audit.json', dict(status='passed', **evidence,
        zip_sha256=sha(path / ZIP_NAME), result_sha256=sha(path / 'result.csv'),
        parent_zip_sha256=manifest['spec']['reference']['zip_sha256'],
        package_receipt_sha256=sha(out / 'package.json'), manifest_sha256=sha(out / 'manifest.json'),
        optimizer_runs=2, cold_states=2, packages=1, desktop_writes=0, uploads=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['freeze','train','cold','predict','release','audit'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checks', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.operation == 'freeze':
        if args.checks is None:
            raise ValueError('Locked check receipt required')
        freeze(out, args.checks.resolve())
    else:
        globals()[args.operation](out)


if __name__ == '__main__':
    main()
