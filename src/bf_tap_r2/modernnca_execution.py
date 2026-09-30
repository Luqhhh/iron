"""A complete paired partition with four saved states and two optimizer runs.

Scheduling, frozen reference admission and release remain separate. All
construction/preprocessing/training occurs inside append-only reservations.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .modernnca_model import ARMS, NeighborRegressor, Settings, clean, validate_pair
from .modernnca_audit import audit_partition, read_state
from .modernnca_protocol import ReservationLedger, file_hash, write_new, task_name
from .v3_4_bags import group_safe_inner_folds
from .v7_periodic import digest


def frame_digest(frame):
    return digest(dict(columns=frame.columns.tolist(), dtypes={k:str(v) for k,v in frame.dtypes.items()},
        rows=pd.util.hash_pandas_object(frame, index=True).tolist()))


def inner_parts(training):
    info = group_safe_inner_folds(clean(training), seed=42, n_splits=5)
    mask = info['fold'] != 0
    fitting = training.loc[mask].reset_index(drop=True)
    calibration = training.loc[~mask].reset_index(drop=True)
    validate_pair(clean(fitting), clean(calibration))
    return fitting, calibration, info


def execute_unit(task, training, query, settings, output, ledger_root, policy_sha256):
    if not isinstance(settings, Settings):
        raise ValueError('Validated immutable settings required')
    name = task_name(task); settings_dict = asdict(settings)
    ledger = ReservationLedger.open(ledger_root, policy_sha256)
    payload = dict(task=task, settings_digest=digest(settings_dict))
    with ledger.event('pair_unit', (name,), payload) as pair:
        output = Path(output); output.mkdir(parents=True, exist_ok=False)
        validate_pair(clean(training), query)
        fitting, calibration, info = inner_parts(training)
        target = task['target']; artifacts, costs = {}, {}
        predictions = dict(query_ids=query.sample_id.to_numpy(dtype=str),
            calibration_ids=calibration.sample_id.to_numpy(dtype=str))
        for arm in ARMS:
            selected = None
            for stage, part in (('selector', fitting), ('refit', training)):
                role = f'{arm}_{stage}'
                model_payload = dict(arm=arm, stage=stage, fit_ids_digest=digest(part.sample_id.astype(str).tolist()),
                    settings_digest=digest(settings_dict))
                with ledger.event('estimator', (name, role), model_payload) as estimator:
                    model = NeighborRegressor(arm, settings).initialize(clean(part), part[target].to_numpy())
                    epochs = settings.max_epochs if stage == 'selector' else selected
                    cal = (clean(calibration), calibration[target].to_numpy()) if stage == 'selector' else None
                    started = time.perf_counter()
                    if arm == 'LEARNED_ENCODER':
                        with ledger.event('optimizer', (name, role), model_payload) as optimizer:
                            epoch = model.train(epochs, cal)
                            artifacts[role] = model.save(output/(role+'.npz'))
                            result = dict(model_sha256=artifacts[role], selected_epoch=epoch,
                                actual_epochs=model.actual_epochs_, optimizer_steps=model.optimizer_steps_)
                            optimizer.update(result)
                    else:
                        epoch = model.train(epochs, cal)
                        artifacts[role] = model.save(output/(role+'.npz'))
                        result = dict(model_sha256=artifacts[role], selected_epoch=epoch,
                            actual_epochs=model.actual_epochs_, optimizer_steps=model.optimizer_steps_)
                    costs[role] = dict(seconds=time.perf_counter()-started, **result)
                    estimator.update(result)
                    if stage == 'selector':
                        selected = epoch
                        predictions[role+'_calibration'] = model.predict(clean(calibration))
                        predictions[role+'_query'] = model.predict(query)
                    else:
                        predictions[arm] = model.predict(query)
        with (output/'predictions.npz').open('xb') as stream:
            np.savez(stream, **predictions)
        complete = dict(format='modernnca-pair-v1', task=task, name=name, settings=settings_dict,
            ledger_policy_sha256=policy_sha256, training_sha256=frame_digest(training), query_sha256=frame_digest(query),
            fitting_sha256=frame_digest(fitting), calibration_sha256=frame_digest(calibration),
            inner_fold_hash=info['inner_fold_hash'], group_hash=info['group_hash'],
            artifacts=artifacts, costs=costs, prediction_sha256=file_hash(output/'predictions.npz'))
        write_new(output/'complete.json', complete)
        anchor = file_hash(output/'complete.json'); pair.update(unit_complete_sha256=anchor)
    return dict(name=name, complete_sha256=anchor)


def close(actual, expected, label):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Invalid '+label)
    difference = float(np.max(np.abs(actual-expected)))
    if difference > 1e-8:
        raise ValueError('Independent '+label+' differs')
    return difference


def audit_unit(directory, task, training, query, settings, *, expected_sha256, ledger_root):
    directory = Path(directory); name = task_name(task)
    if directory.is_symlink() or file_hash(directory/'complete.json') != expected_sha256:
        raise ValueError('Caller-anchored paired completion changed')
    validate_pair(clean(training), query)
    fitting, calibration, info = inner_parts(training)
    complete = json.loads((directory/'complete.json').read_text()); settings_dict = asdict(settings)
    roles = {f'{a}_{s}' for a in ARMS for s in ('selector', 'refit')}
    expected = dict(format='modernnca-pair-v1', task=task, name=name, settings=settings_dict,
        training_sha256=frame_digest(training), query_sha256=frame_digest(query),
        fitting_sha256=frame_digest(fitting), calibration_sha256=frame_digest(calibration),
        inner_fold_hash=info['inner_fold_hash'], group_hash=info['group_hash'])
    if (any(complete[k] != v for k,v in expected.items()) or set(complete['artifacts']) != roles
            or set(complete['costs']) != roles
            or {p.name for p in directory.iterdir()} != {'complete.json','predictions.npz',*[r+'.npz' for r in roles]}
            or any(p.is_symlink() for p in directory.iterdir())):
        raise ValueError('Paired artifact/partition schema differs')
    if file_hash(directory/'predictions.npz') != complete['prediction_sha256']:
        raise ValueError('Saved paired predictions changed')
    with np.load(directory/'predictions.npz', allow_pickle=False) as archive:
        predictions = {k:archive[k].copy() for k in archive.files}
    expected_fields = {'query_ids','calibration_ids',*ARMS,
        *[f'{a}_selector_{part}' for a in ARMS for part in ('query','calibration')]}
    if (set(predictions) != expected_fields
            or not np.array_equal(predictions['query_ids'], query.sample_id.to_numpy(dtype=str))
            or not np.array_equal(predictions['calibration_ids'], calibration.sample_id.to_numpy(dtype=str))):
        raise ValueError('Saved prediction row/schema identity mismatch')
    ledger = ReservationLedger.open(ledger_root, complete['ledger_policy_sha256']); ledger.inspect()
    events = {('pair_unit',(name,)):dict(payload=dict(task=task,settings_digest=digest(settings_dict)),
        result=dict(unit_complete_sha256=expected_sha256))}
    maximum, model_reports, initial_states, selected_epochs = 0., [], {}, {}
    for arm in ARMS:
        for stage, part in (('selector',fitting),('refit',training)):
            role = f'{arm}_{stage}'; path = directory/(role+'.npz'); sha = complete['artifacts'][role]
            metadata, _, initial, _ = read_state(path, sha)
            if metadata['arm'] != arm or metadata['settings'] != settings_dict:
                raise ValueError('Saved model arm/recipe changed')
            if stage == 'selector':
                cal = (clean(calibration), calibration[task['target']].to_numpy())
                selected_epochs[arm] = metadata['selected_epoch']
                expected_prediction = predictions[role+'_query']
            else:
                cal = None; expected_prediction = predictions[arm]
                if metadata['selected_epoch'] != selected_epochs[arm]:
                    raise ValueError('Fresh refit changed selected epoch')
            values, report = audit_partition(path, sha, clean(part), part[task['target']].to_numpy(), query, cal)
            maximum = max(maximum, close(values, expected_prediction, 'cold query prediction'))
            if stage == 'selector':
                values, _ = audit_partition(path, sha, clean(part), part[task['target']].to_numpy(), clean(calibration), cal)
                maximum = max(maximum, close(values, predictions[role+'_calibration'], 'cold calibration prediction'))
            cost = complete['costs'][role]
            result = dict(model_sha256=sha, selected_epoch=metadata['selected_epoch'],
                actual_epochs=metadata['actual_epochs'], optimizer_steps=metadata['optimizer_steps'])
            if (set(cost) != {'seconds',*result} or not np.isfinite(cost['seconds']) or cost['seconds'] <= 0
                    or any(cost[k] != v for k,v in result.items())):
                raise ValueError('Saved estimator/optimizer count trace differs')
            payload = dict(arm=arm, stage=stage, fit_ids_digest=digest(part.sample_id.astype(str).tolist()),
                settings_digest=digest(settings_dict))
            events['estimator',(name,role)] = dict(payload=payload,result=result)
            if arm == 'LEARNED_ENCODER':
                events['optimizer',(name,role)] = dict(payload=payload,result=result)
            initial_states[arm,stage] = initial
            model_reports.append(dict(role=role, **report))
    for stage in ('selector','refit'):
        a,b = (initial_states[arm,stage] for arm in ARMS)
        if any(not np.array_equal(a[k],b[k]) for k in a):
            raise ValueError('Matched control initial parameters differ')
    observed = set()
    for path in (ledger.root/'events').glob('*.started.json'):
        start = json.loads(path.read_text())
        if not start['key'] or start['key'][0] != name:
            continue
        key = start['kind'],tuple(start['key'])
        end = path.with_name(path.name.replace('started.json','complete.json'))
        if key not in events or key in observed or not end.exists():
            raise ValueError('Incomplete/unexpected paired reservations')
        if start['payload'] != events[key]['payload'] or json.loads(end.read_text())['result'] != events[key]['result']:
            raise ValueError('Paired reservation provenance differs')
        observed.add(key)
    if observed != set(events):
        raise ValueError('Missing paired reservations')
    return {a:predictions[a] for a in ARMS}, dict(status='passed',name=name,models_checked=4,
        maximum_difference=maximum,models=model_reports,ledger_policy_sha256=ledger.policy_sha256,
        new_estimator_fits=0,new_optimizer_calls=0,scope='one paired unit; full-phase admission remains separate')
