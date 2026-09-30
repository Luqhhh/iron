"""One exclusively reserved matched unit with saved predictions and receipt."""
from pathlib import Path
import json
import time

import numpy as np

from .data import TARGETS
from .dnnr_calibration import fit_pair
from .dnnr_model import ARMS, Settings, validate_frame
from .dnnr_ledger import ReservationLedger, file_hash, write_new
from .dnnr_audit import verify_pair


def validate_task(task):
    if (set(task) != {"target", "seed", "fold"} or task['target'] not in TARGETS
            or type(task['seed']) is not int or task['seed'] not in (42, 3407, 7777, 12011)
            or type(task['fold']) is not int or not 0 <= task['fold'] < 5):
        raise ValueError('Unknown DNNR unit identity')
    return (task['target'], task['seed'], task['fold'])


def execute_unit(task, training, y, query, directory, ledger_root, policy_sha256,
                 *, settings=None):
    scope = validate_task(task)
    validate_frame(training); validate_frame(query)
    if set(training.sample_id.astype(str)) & set(query.sample_id.astype(str)):
        raise ValueError('Outer training/query identity overlap')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    ledger = ReservationLedger.open(ledger_root, policy_sha256)
    with ledger.event('pair_unit', scope, dict(task=task, train_rows=len(training), query_rows=len(query))) as result:
        started = time.perf_counter()
        def observer(kind, payload):
            key = (*scope, payload['role'], payload['arm'], kind)
            return ledger.event(kind, key, payload)
        pair = fit_pair(training, y, task['target'], settings=settings, observer=observer)
        hashes = pair.save(directory/'models')
        prediction_hashes = {}
        for arm, model in pair.outer.items():
            path = directory/f'prediction-{arm}.npy'
            with path.open('xb') as stream:
                np.save(stream, model.predict(query), allow_pickle=False)
            prediction_hashes[path.name] = file_hash(path)
        complete = dict(task=task, policy_sha256=policy_sha256,
            query_ids=query.sample_id.astype(str).tolist(),
            train_ids=training.sample_id.astype(str).tolist(),
            model_hashes=hashes, prediction_hashes=prediction_hashes,
            settings=pair.outer['KNN_FIXED'].metadata()['settings'],
            counts={k: pair.receipt[k] for k in ('estimator_runs', 'derivative_bank_runs',
                'derivative_local_solutions', 'metric_epoch_runs', 'metric_updates', 'metric_local_solutions')},
            selected_metric_epochs=pair.receipt['selected_metric_epochs'],
            elapsed_seconds=time.perf_counter()-started)
        write_new(directory/'complete.json', complete)
        result.update(complete_sha256=file_hash(directory/'complete.json'), **complete['counts'])
    return dict(complete_sha256=file_hash(directory/'complete.json'))


def audit_unit(directory, complete_sha256, task, training, y, query, *, settings=None,
               ledger_root=None):
    validate_task(task)
    directory = Path(directory)
    path = directory/'complete.json'
    if not complete_sha256 or file_hash(path) != complete_sha256:
        raise ValueError('Externally anchored completed unit required')
    complete = json.loads(path.read_text())
    settings = settings or Settings()
    from dataclasses import asdict
    if (complete['task'] != task or complete['settings'] != asdict(settings)
            or complete['query_ids'] != query.sample_id.astype(str).tolist()
            or complete['train_ids'] != training.sample_id.astype(str).tolist()
            or set(complete['prediction_hashes']) != {f'prediction-{a}.npy' for a in ARMS}
            or set(complete['model_hashes']) != {'receipt.json'} | {f'{r}-{a}.npz' for r in ('inner','outer') for a in ARMS}):
        raise ValueError('Unit identity, settings or artifact schema differs')
    expected_paths = {'complete.json'} | set(complete['prediction_hashes']) | {'models/'+k for k in complete['model_hashes']}
    if {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} != expected_paths:
        raise ValueError('Unexpected/missing DNNR unit artifacts')
    predictions = {}
    for name, sha in complete['model_hashes'].items():
        path = directory/'models'/name
        if path.is_symlink() or file_hash(path) != sha:
            raise ValueError('Saved model artifact changed')
    for name, sha in complete['prediction_hashes'].items():
        path = directory/name
        if path.is_symlink() or file_hash(path) != sha:
            raise ValueError('Saved prediction artifact changed')
        value = np.load(path, allow_pickle=False)
        if value.shape != (len(query),) or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError('Saved prediction array differs')
        predictions[name[len('prediction-'):-4]] = value
    report = verify_pair(directory/'models', complete['model_hashes']['receipt.json'], training, y, query, predictions)
    for key, value in complete['counts'].items():
        if report['counts'].get(key) != value:
            raise ValueError('Completed unit operation counts differ')
    if complete['selected_metric_epochs'] != report['selected_metric_epochs']:
        raise ValueError('Completed unit endpoint selection differs')
    if ledger_root is not None:
        verify_unit_reservations(ledger_root, complete['policy_sha256'], task,
                                 complete_sha256, complete['counts'])
    return predictions, report


def verify_unit_reservations(root, policy_sha256, task, complete_sha256, counts):
    ledger = ReservationLedger.open(root, policy_sha256)
    ledger.inspect()
    scope = validate_task(task)
    expected = {('pair_unit', scope)}
    for role in ('inner', 'outer'):
        for arm in ARMS:
            expected.add(('estimator', (*scope, role, arm, 'estimator')))
            if arm != 'KNN_FIXED':
                expected.add(('derivative_bank', (*scope, role, arm, 'derivative_bank')))
            if arm == 'DNNR_LEARNED' and (role == 'inner' or counts['metric_epoch_runs'] == 2):
                expected.add(('metric_epoch', (*scope, role, arm, 'metric_epoch')))
    seen = set()
    measured_solutions = dict(derivative_local_solutions=0, metric_updates=0, metric_local_solutions=0)
    for path in (Path(root)/'events').glob('*.started.json'):
        record = json.loads(path.read_text())
        if tuple(record['key'][:3]) != scope:
            continue
        identity = (record['kind'], tuple(record['key']))
        if identity not in expected:
            raise ValueError('Unexpected unit fitting reservation')
        endpoint = path.with_name(path.name.replace('started.json', 'complete.json'))
        if not endpoint.exists() or path.with_name(path.name.replace('started.json','failed.json')).exists():
            raise ValueError('Unit reservation failed or incomplete')
        terminal = json.loads(endpoint.read_text())
        if identity[0] == 'pair_unit':
            if (terminal['result'].get('complete_sha256') != complete_sha256
                    or any(terminal['result'].get(k) != v for k,v in counts.items())):
                raise ValueError('Pair completion and reservation result differ')
        elif identity[0] == 'metric_epoch':
            measured_solutions['metric_updates'] += record['payload']['updates']
            measured_solutions['metric_local_solutions'] += record['payload']['local_solutions']
        elif identity[0] == 'derivative_bank':
            measured_solutions['derivative_local_solutions'] += record['payload']['local_solutions']
        seen.add(identity)
    if seen != expected or any(counts[k] != v for k,v in measured_solutions.items()):
        raise ValueError('Missing/different unit computational reservations')
