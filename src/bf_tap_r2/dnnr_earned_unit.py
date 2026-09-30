"""Only earned confirmation candidates plus their necessary causal controls.

The original full development pair and checked estimator implementations stay
byte-identical. Omitted failed candidates consume no confirmation fit.
"""
from dataclasses import asdict
from pathlib import Path
import json
import time

import numpy as np

from .dnnr_model import ARMS, Regressor, Settings, validate_frame
from .dnnr_execution import execute_unit, audit_unit, validate_task
from .dnnr_audit import verify_saved
from .dnnr_ledger import ReservationLedger, file_hash, write_new
from .v3_4_bags import group_safe_inner_folds


def operation_counts(arms, inner_rows, outer_rows, selected):
    derivatives = sum(a != 'KNN_FIXED' for a in arms)
    learned = 'DNNR_LEARNED' in arms
    return dict(estimator_runs=2*len(arms),derivative_bank_runs=2*derivatives,
        derivative_local_solutions=derivatives*(inner_rows+outer_rows),
        metric_epoch_runs=(1+selected)*learned,metric_updates=(inner_rows+selected*outer_rows)*learned,
        metric_local_solutions=(inner_rows+selected*outer_rows)*learned)


def execute_earned(task,training,y,query,directory,ledger_root,policy_sha256,arms,settings=None):
    arms = tuple(arms); settings = settings or Settings()
    if arms == ARMS:
        return execute_unit(task,training,y,query,directory,ledger_root,policy_sha256,settings=settings)
    if arms not in (('KNN_FIXED','DNNR_FIXED'),('DNNR_FIXED','DNNR_LEARNED')):
        raise ValueError('Only declared candidate/control subsets may be fitted')
    scope = validate_task(task)
    validate_frame(training);validate_frame(query)
    if set(training.sample_id.astype(str)) & set(query.sample_id.astype(str)):
        raise ValueError('Outer train/query overlap')
    y = np.asarray(y,float)
    if y.shape != (len(training),) or not np.isfinite(y).all():
        raise ValueError('Finite aligned training-only target required')
    directory = Path(directory);directory.mkdir(parents=True,exist_ok=False)
    ledger = ReservationLedger.open(ledger_root,policy_sha256)
    with ledger.event('pair_unit',scope,dict(task=task,arms=arms,train_rows=len(training),query_rows=len(query))) as terminal:
        started = time.perf_counter()
        assignment = group_safe_inner_folds(training,n_splits=5,seed=42);mask = assignment['fold'] != 0
        fitting,calibration = training.loc[mask].reset_index(drop=True),training.loc[~mask].reset_index(drop=True)
        inner,outer,calibration_predictions,maes = {},{},{},{}
        def observer(role,arm):
            return lambda kind,payload: ledger.event(kind,(*scope,role,arm,kind),dict(payload,role=role,arm=arm,target=task['target']))
        for arm in arms:
            inner[arm] = Regressor(arm,settings).fit(fitting,y[mask],metric_epochs=int(arm=='DNNR_LEARNED'),observer=observer('inner',arm))
            calibration_predictions[arm] = inner[arm].predict(calibration)
            maes[arm] = float(np.abs(calibration_predictions[arm]-y[~mask]).mean())
        selected = int('DNNR_LEARNED' in arms and maes['DNNR_LEARNED'] < maes['DNNR_FIXED'])
        for arm in arms:
            outer[arm] = Regressor(arm,settings).fit(training,y,metric_epochs=selected if arm=='DNNR_LEARNED' else 0,observer=observer('outer',arm))
        models = directory/'models';models.mkdir()
        hashes,prediction_hashes = {},{}
        for role,members in (('inner',inner),('outer',outer)):
            for arm,model in members.items():
                name=f'{role}-{arm}.npz';hashes[name]=model.save(models/name)
        receipt = dict(format='dnnr-earned-subset-v1',arms=list(arms),task=task,inner_folds=assignment['fold'].tolist(),
            fitting_ids=fitting.sample_id.astype(str).tolist(),calibration_ids=calibration.sample_id.astype(str).tolist(),
            inner_group_hash=assignment['group_hash'],inner_fold_hash=assignment['inner_fold_hash'],
            calibration_predictions={a:p.tolist() for a,p in calibration_predictions.items()},calibration_maes=maes,
            selected_metric_epochs=selected,settings=asdict(settings))
        write_new(models/'receipt.json',receipt);hashes['receipt.json']=file_hash(models/'receipt.json')
        for arm in arms:
            path=directory/f'prediction-{arm}.npy'
            with path.open('xb') as stream:np.save(stream,outer[arm].predict(query),allow_pickle=False)
            prediction_hashes[path.name]=file_hash(path)
        counts=operation_counts(arms,len(fitting),len(training),selected)
        complete=dict(task=task,policy_sha256=policy_sha256,arms=list(arms),
            query_ids=query.sample_id.astype(str).tolist(),train_ids=training.sample_id.astype(str).tolist(),
            model_hashes=hashes,prediction_hashes=prediction_hashes,settings=asdict(settings),counts=counts,
            selected_metric_epochs=selected,elapsed_seconds=time.perf_counter()-started)
        write_new(directory/'complete.json',complete)
        terminal.update(complete_sha256=file_hash(directory/'complete.json'),**counts)
    return dict(complete_sha256=file_hash(directory/'complete.json'))


def audit_earned(directory,sha256,task,training,y,query,arms,settings=None,ledger_root=None):
    validate_task(task); validate_frame(training); validate_frame(query)
    arms=tuple(arms);settings=settings or Settings()
    if arms==ARMS:
        return audit_unit(directory,sha256,task,training,y,query,settings=settings,ledger_root=ledger_root)
    if arms not in (('KNN_FIXED','DNNR_FIXED'),('DNNR_FIXED','DNNR_LEARNED')):
        raise ValueError('Unknown earned subset')
    directory=Path(directory)
    if not sha256 or file_hash(directory/'complete.json') != sha256:
        raise ValueError('Anchored earned completion required')
    complete=json.loads((directory/'complete.json').read_text())
    expected={f'{role}-{a}.npz' for role in ('inner','outer') for a in arms}|{'receipt.json'}
    if (complete['task']!=task or complete['arms']!=list(arms) or complete['settings']!=asdict(settings)
            or complete['train_ids']!=training.sample_id.astype(str).tolist() or complete['query_ids']!=query.sample_id.astype(str).tolist()
            or set(complete['model_hashes'])!=expected or set(complete['prediction_hashes'])!={f'prediction-{a}.npy' for a in arms}):
        raise ValueError('Earned task/arm/row/artifact identity differs')
    all_paths={'complete.json'}|set(complete['prediction_hashes'])|{'models/'+k for k in expected}
    if {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}!=all_paths:
        raise ValueError('Unexpected/missing earned artifact')
    for name,sha in {**{'models/'+k:v for k,v in complete['model_hashes'].items()},**complete['prediction_hashes']}.items():
        p=directory/name
        if p.resolve()!=p.absolute() or file_hash(p)!=sha:raise ValueError('Earned artifact changed or escaped')
    receipt=json.loads((directory/'models/receipt.json').read_text())
    assignment=group_safe_inner_folds(training,n_splits=5,seed=42);mask=assignment['fold']!=0
    fitting,calibration=training.loc[mask].reset_index(drop=True),training.loc[~mask].reset_index(drop=True)
    check=dict(format='dnnr-earned-subset-v1',arms=list(arms),task=task,inner_folds=assignment['fold'].tolist(),
        fitting_ids=fitting.sample_id.astype(str).tolist(),calibration_ids=calibration.sample_id.astype(str).tolist(),
        inner_group_hash=assignment['group_hash'],inner_fold_hash=assignment['inner_fold_hash'],settings=asdict(settings))
    if any(receipt[k]!=v for k,v in check.items()):raise ValueError('Earned inner partition/settings differ')
    predictions,reports,maes={}, {}, {}
    y=np.asarray(y,float)
    for arm in arms:
        path=directory/'models'/f'inner-{arm}.npz';sha=complete['model_hashes'][path.name]
        model=Regressor.load(path,sha)
        if model.arm!=arm or model.settings!=settings or model.metric_epochs_!=int(arm=='DNNR_LEARNED'):
            raise ValueError('Earned inner arm/epoch/settings differ')
        reports[path.name]=verify_saved(path,sha,fitting,y[mask],calibration,receipt['calibration_predictions'][arm])
        # Also checked by independent NumPy prediction above, retaining the
        # original reduction order for exact endpoint ties.
        maes[arm]=float(np.abs(model.predict(calibration)-y[~mask]).mean())
        if abs(maes[arm]-receipt['calibration_maes'][arm])>1e-8:raise ValueError('Earned inner MAE differs')
    selected=int('DNNR_LEARNED' in arms and maes['DNNR_LEARNED']<maes['DNNR_FIXED'])
    counts=operation_counts(arms,len(fitting),len(training),selected)
    if complete['counts']!=counts or complete['selected_metric_epochs']!=selected or receipt['selected_metric_epochs']!=selected:
        raise ValueError('Earned selection/counts differ')
    for arm in arms:
        path=directory/'models'/f'outer-{arm}.npz';sha=complete['model_hashes'][path.name]
        model=Regressor.load(path,sha)
        if model.arm!=arm or model.settings!=settings or model.metric_epochs_!=(selected if arm=='DNNR_LEARNED' else 0):
            raise ValueError('Earned outer arm/epoch/settings differ')
        predictions[arm]=np.load(directory/f'prediction-{arm}.npy',allow_pickle=False)
        if (predictions[arm].shape != (len(query),) or predictions[arm].dtype != np.float64
                or not np.isfinite(predictions[arm]).all()):
            raise ValueError('Earned saved prediction array differs')
        reports[path.name]=verify_saved(path,sha,training,y,query,predictions[arm])
    if ledger_root is not None:
        ledger=ReservationLedger.open(ledger_root,complete['policy_sha256']);ledger.inspect()
        scope=validate_task(task);expected_events={('pair_unit',scope)}
        for role in ('inner','outer'):
            for a in arms:
                expected_events.add(('estimator',(*scope,role,a,'estimator')))
                if a!='KNN_FIXED':expected_events.add(('derivative_bank',(*scope,role,a,'derivative_bank')))
                if a=='DNNR_LEARNED' and (role=='inner' or selected):expected_events.add(('metric_epoch',(*scope,role,a,'metric_epoch')))
        seen=set(); measured=dict(derivative_local_solutions=0,metric_updates=0,metric_local_solutions=0)
        for p in (Path(ledger_root)/'events').glob('*.started.json'):
            event=json.loads(p.read_text())
            if tuple(event['key'][:3])!=scope:continue
            identity=(event['kind'],tuple(event['key']))
            endpoint=p.with_name(p.name.replace('started.json','complete.json'))
            if identity not in expected_events or not endpoint.exists():raise ValueError('Extra/missing earned reservation')
            terminal=json.loads(endpoint.read_text())
            if identity[0]=='pair_unit' and (terminal['result'].get('complete_sha256')!=sha256
                    or any(terminal['result'].get(k)!=v for k,v in counts.items())):raise ValueError('Earned completion reservation differs')
            if identity[0]=='metric_epoch':
                measured['metric_updates']+=event['payload']['updates']
                measured['metric_local_solutions']+=event['payload']['local_solutions']
            elif identity[0]=='derivative_bank':
                measured['derivative_local_solutions']+=event['payload']['local_solutions']
            seen.add(identity)
        if seen!=expected_events or any(counts[k]!=v for k,v in measured.items()):
            raise ValueError('Missing/different earned fitting reservation')
    return predictions,dict(status='passed',saved_models=2*len(arms),counts=counts,models=reports,
        selected_metric_epochs=selected,new_estimator_fits=0,new_metric_epoch_runs=0)
