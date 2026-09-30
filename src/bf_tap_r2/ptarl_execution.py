"""One paired training partition, six optimizer reservations, zero scheduling.

Complete formal phase/source/reference/resource admission remains separate.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import TARGETS
from .ptarl_model import ARMS, PrototypeRegressor, prototype_bank
from .ptarl_protocol import ReservationLedger, file_hash, write_new
from .ptarl_verify import saved, oracle, close, verify_model, verify_bank
from .rfm_execution import validate_partition
from .v3_4_bags import group_safe_inner_folds
from .v49_gradients import GradientRegressor, clean
from .v49_verify import verify_model as verify_teacher
from .v7_periodic import digest


def task_name(task):
    if (set(task) != {'target','seed','fold'} or task['target'] not in TARGETS
            or type(task['seed']) is not int or task['seed'] not in (42,3407,7777,12011)
            or type(task['fold']) is not int or task['fold'] not in range(5)):
        raise ValueError('Invalid paired outer task')
    return f"{task['target']}-s{task['seed']}-f{task['fold']}"


def inner_parts(training):
    info=group_safe_inner_folds(clean(training),seed=42,n_splits=5)
    mask=info['fold'] != 0
    fitting=training.loc[mask].reset_index(drop=True)
    calibration=training.loc[~mask].reset_index(drop=True)
    validate_partition(fitting,clean(calibration))
    return fitting,calibration,info


def frame_digest(frame):
    return digest(frame.to_dict('list'))


def execute_unit(task, training, query, settings, output, ledger_root, policy_sha256):
    name=task_name(task)
    ledger=ReservationLedger.open(ledger_root,policy_sha256)
    with ledger.event('pair_unit',(name,),dict(task=task,settings_digest=digest(settings))) as pair:
        output=Path(output)
        output.mkdir(parents=True,exist_ok=False)
        validate_partition(training,query)
        fitting,calibration,info=inner_parts(training)
        target=task['target']
        artifacts,metadata,teachers,banks,costs={},{},{},{},{}
        for role,part in [('teacher_inner',fitting),('teacher_outer',training)]:
            with ledger.event('optimizer',(name,role),dict(fit_ids_digest=digest(part.sample_id.tolist()))) as event:
                with torch.random.fork_rng(devices=[]):
                    teacher=GradientRegressor('BASE',settings).initialize(part,part[target].to_numpy())
                    started=time.perf_counter()
                    if role=='teacher_inner':
                        teacher.train(settings['max_epochs'],(clean(calibration),calibration[target].to_numpy()))
                    else:
                        teacher.train(teachers['teacher_inner'].selected_epoch_)
                    seconds=time.perf_counter()-started
                steps=teacher.stopped_epoch_*int(np.ceil(len(part)/settings['batch_size']))
                costs[role]=dict(training_seconds=seconds,training_steps=steps)
                teacher.save(output/f'{role}.pt')
                artifacts[role]=file_hash(output/f'{role}.pt')
                metadata[role]=teacher.metadata()
                teachers[role]=teacher
                event.update(model_sha256=artifacts[role],selected_epoch=teacher.selected_epoch_,**costs[role])
            with ledger.event('kmeans',(name,role),dict(teacher_sha256=artifacts[role])) as event:
                bank=prototype_bank(teacher,clean(part),settings['prototype_count'],settings['random_seed'])
                banks[role]=bank
                event.update(prototype_receipt=bank[1])
        predictions={'query_ids':query.sample_id.to_numpy(dtype=str),
                     'calibration_ids':calibration.sample_id.to_numpy(dtype=str)}
        for arm in ARMS:
            selected=None
            for stage,part,bank in [('selector',fitting,banks['teacher_inner']),
                                     ('refit',training,banks['teacher_outer'])]:
                role=f'{arm}_{stage}'
                with ledger.event('optimizer',(name,role),dict(fit_ids_digest=digest(part.sample_id.tolist()))) as event:
                    model=PrototypeRegressor(arm,settings).initialize(part,part[target].to_numpy(),*bank)
                    started=time.perf_counter()
                    if stage=='selector':
                        selected=model.train(settings['max_epochs'],(clean(calibration),calibration[target].to_numpy()))
                        seconds=time.perf_counter()-started
                        predictions[f'{arm}_calibration']=model.predict(clean(calibration))
                    else:
                        model.train(selected)
                        seconds=time.perf_counter()-started
                        predictions[arm]=model.predict(query)
                    artifacts[role]=model.save(output/f'{role}.pt')
                    metadata[role]=model.metadata()
                    steps=model.stopped_epoch_*int(np.ceil(len(part)/settings['batch_size']))
                    costs[role]=dict(training_seconds=seconds,training_steps=steps)
                    event.update(model_sha256=artifacts[role],selected_epoch=model.selected_epoch_,**costs[role])
        with (output/'predictions.npz').open('xb') as stream:np.savez(stream,**predictions)
        complete=dict(task=task,name=name,settings=settings,ledger_policy_sha256=policy_sha256,
            training_sha256=frame_digest(training),query_sha256=frame_digest(query),
            fitting_sha256=frame_digest(fitting),calibration_sha256=frame_digest(calibration),
            inner_fold_hash=info['inner_fold_hash'],group_hash=info['group_hash'],
            artifacts=artifacts,metadata=metadata,costs=costs,prediction_sha256=file_hash(output/'predictions.npz'))
        write_new(output/'complete.json',complete)
        anchor=file_hash(output/'complete.json')
        pair.update(unit_complete_sha256=anchor)
    return dict(name=name,complete_sha256=anchor)


def audit_unit(directory, task, training, query, settings, *, expected_sha256, ledger_root):
    """Externally anchored unit; caller still must bind the phase and references."""
    directory=Path(directory)
    name=task_name(task)
    validate_partition(training,query)
    if file_hash(directory/'complete.json') != expected_sha256:
        raise ValueError('Paired completion identity mismatch')
    complete=json.loads((directory/'complete.json').read_text())
    roles={'teacher_inner','teacher_outer',*[f'{a}_{s}' for a in ARMS for s in ('selector','refit')]}
    if (set(complete['artifacts']) != roles or set(complete['metadata']) != roles or set(complete['costs']) != roles
            or {p.name for p in directory.iterdir()} != {'complete.json','predictions.npz',*[f'{r}.pt' for r in roles]}):
        raise ValueError('Unexpected/incomplete paired unit artifacts')
    fitting,calibration,info=inner_parts(training)
    expected=dict(task=task,name=name,settings=settings,training_sha256=frame_digest(training),
        query_sha256=frame_digest(query),fitting_sha256=frame_digest(fitting),
        calibration_sha256=frame_digest(calibration),inner_fold_hash=info['inner_fold_hash'],group_hash=info['group_hash'])
    if any(complete[k] != v for k,v in expected.items()):
        raise ValueError('Paired source/partition identity mismatch')
    ledger=ReservationLedger.open(ledger_root,complete['ledger_policy_sha256'])
    ledger.inspect()
    expected_events={('pair_unit',(name,)):dict(payload=dict(task=task,settings_digest=digest(settings)),
        result=dict(unit_complete_sha256=expected_sha256))}
    for role in roles:
        part=fitting if role.endswith('inner') or role.endswith('selector') else training
        cost=complete['costs'][role]
        if (set(cost)!={'training_seconds','training_steps'} or not np.isfinite(cost['training_seconds'])
                or cost['training_seconds']<=0 or cost['training_steps'] !=
                complete['metadata'][role]['stopped_epoch']*int(np.ceil(len(part)/settings['batch_size']))):
            raise ValueError('Training cost trace mismatch')
        expected_events['optimizer',(name,role)]=dict(
            payload=dict(fit_ids_digest=digest(part.sample_id.tolist())),
            result=dict(model_sha256=complete['artifacts'][role],selected_epoch=complete['metadata'][role]['selected_epoch'],**cost))
    for role,arm_role in [('teacher_inner','CONTROL_selector'),('teacher_outer','CONTROL_refit')]:
        expected_events['kmeans',(name,role)]=dict(payload=dict(teacher_sha256=complete['artifacts'][role]),
            result=dict(prototype_receipt=complete['metadata'][arm_role]['prototype_initialization']))
    observed=set()
    for path in (ledger.root/'events').glob('*.started.json'):
        start=json.loads(path.read_text())
        if not start['key'] or start['key'][0] != name:continue
        key=start['kind'],tuple(start['key'])
        end=path.with_name(path.name.replace('started.json','complete.json'))
        if key not in expected_events or key in observed or not end.exists():
            raise ValueError('Incomplete/unexpected paired reservation')
        result=json.loads(end.read_text())
        if (start['payload'] != expected_events[key]['payload']
                or result['result'] != expected_events[key]['result']):
            raise ValueError('Paired reservation result/provenance mismatch')
        observed.add(key)
    if observed != set(expected_events):
        raise ValueError('Missing paired reservations')
    if file_hash(directory/'predictions.npz') != complete['prediction_sha256']:
        raise ValueError('Saved paired predictions changed')
    with np.load(directory/'predictions.npz',allow_pickle=False) as archive:
        predictions={k:archive[k].copy() for k in archive.files}
    if set(predictions) != {'query_ids','calibration_ids',*ARMS,*[f'{a}_calibration' for a in ARMS]}:
        raise ValueError('Prediction schema mismatch')
    if (not np.array_equal(predictions['query_ids'],query.sample_id.to_numpy(dtype=str))
            or not np.array_equal(predictions['calibration_ids'],calibration.sample_id.to_numpy(dtype=str))):
        raise ValueError('Saved prediction row identity mismatch')
    target=task['target']
    teachers,max_diff={},0.
    for role,part in [('teacher_inner',fitting),('teacher_outer',training)]:
        data=saved(directory/f'{role}.pt',complete['artifacts'][role])
        verify_teacher(directory/f'{role}.pt',part,part[target].to_numpy(),'BASE',settings,complete['metadata'][role])
        teachers[role]=data
        if any(('calibration_standardized_mae' in row) != (role=='teacher_inner')
               for row in data['metadata']['history']):
            raise ValueError('Teacher selector/refit trace mismatch')
        if role=='teacher_inner':
            prediction,_=oracle(data,clean(calibration))
            selected=data['metadata']['selected_epoch']
            observed=np.abs(prediction-calibration[target].to_numpy()).mean()/data['metadata']['target_std']
            max_diff=max(max_diff,close(observed,data['metadata']['history'][selected-1]['calibration_standardized_mae'],
                                       'Teacher selected calibration state'))
        elif data['metadata']['selected_epoch'] != teachers['teacher_inner']['metadata']['selected_epoch']:
            raise ValueError('Fresh outer teacher epoch mismatch')
    for arm in ARMS:
        for stage,part,q,teacher_role in [('selector',fitting,clean(calibration),'teacher_inner'),
                                         ('refit',training,query,'teacher_outer')]:
            role=f'{arm}_{stage}'
            data=verify_model(directory/f'{role}.pt',part,part[target].to_numpy(),arm,settings,
                complete['artifacts'][role],selector=stage=='selector',trace=complete['metadata'][role])
            centers=data['state']['initial_prototypes'].numpy()
            verify_bank(teachers[teacher_role],part,data['metadata']['prototype_initialization'],centers)
            cold=PrototypeRegressor.load(directory/f'{role}.pt',complete['artifacts'][role])
            independent,_=oracle(data,q)
            expected=predictions[f'{arm}_calibration' if stage=='selector' else arm]
            reverse=cold.predict(q.iloc[::-1])[::-1]
            chunks=np.concatenate([cold.predict(q.iloc[i:i+7]) for i in range(0,len(q),7)])
            for p in [independent,cold.predict(q),reverse,chunks]:
                max_diff=max(max_diff,close(p,expected,'Independent cold/order/chunk prediction'))
            if stage=='selector':
                value=np.abs(independent-calibration[target].to_numpy()).mean()/data['metadata']['target_std']
                row=data['metadata']['history'][data['metadata']['selected_epoch']-1]
                max_diff=max(max_diff,close(value,row['calibration_standardized_mae'],'Selector checkpoint calibration MAE'))
            elif data['metadata']['selected_epoch'] != complete['metadata'][f'{arm}_selector']['selected_epoch']:
                raise ValueError('Fresh stage2 refit epoch mismatch')
    for stage in ('selector','refit'):
        metas=[complete['metadata'][f'{a}_{stage}'] for a in ARMS]
        if (metas[0]['initial_native_digest'] != metas[1]['initial_native_digest']
                or metas[0]['prototype_initialization'] != metas[1]['prototype_initialization']):
            raise ValueError('Matched initialization/prototype receipt mismatch')
    report=dict(status='passed',name=name,models_checked=6,maximum_difference=max_diff,
        ledger_policy_sha256=complete['ledger_policy_sha256'],new_optimizer_calls=0,new_kmeans_calls=0,
        scope='one paired outer unit; phase/source/reference/gate/resource admission remains separate')
    return {a:predictions[a] for a in ARMS},report
