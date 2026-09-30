"""One immutable matched pair with separate estimator/optimizer reservations."""
from pathlib import Path
import json

import numpy as np

from .dnnr_execution import validate_task
from .dnnr_model import validate_frame
from .danet_ledger import ReservationLedger,write_new,file_hash
from .danet_model import ARMS,Settings
from .danet_calibration import fit_pair
from .danet_audit import verify_pair


def execute_unit(task,training,y,query,directory,ledger_root,policy_sha256,*,settings=None,resource_upper_bound=False):
    scope=validate_task(task);validate_frame(training);validate_frame(query)
    if set(training.sample_id.astype(str))&set(query.sample_id.astype(str)):raise ValueError('Outer train/query overlap')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    ledger=ReservationLedger.open(ledger_root,policy_sha256)
    with ledger.event('pair_unit',scope,dict(task=task,training_rows=len(training),query_rows=len(query),resource_upper_bound=resource_upper_bound)) as terminal:
        def observer(kind,payload):
            return ledger.event(kind,(*scope,payload['role'],payload['arm'],kind),payload)
        pair=fit_pair(training,y,task['target'],settings=settings,observer=observer,resource_upper_bound=resource_upper_bound)
        hashes=pair.save(directory/'models');prediction_hashes={}
        for arm in ARMS:
            path=directory/f'prediction-{arm}.npy'
            with path.open('xb') as stream:np.save(stream,pair.outer[arm].predict(query),allow_pickle=False)
            prediction_hashes[path.name]=file_hash(path)
        complete=dict(task=task,policy_sha256=policy_sha256,resource_upper_bound=resource_upper_bound,
            training_ids=training.sample_id.astype(str).tolist(),query_ids=query.sample_id.astype(str).tolist(),
            settings=pair.outer[ARMS[0]].metadata()['settings'],model_hashes=hashes,prediction_hashes=prediction_hashes,
            estimator_runs=4,optimizer_runs=4,optimizer_epochs=pair.receipt['optimizer_epochs'])
        write_new(directory/'complete.json',complete)
        terminal.update(complete_sha256=file_hash(directory/'complete.json'),estimator_runs=4,optimizer_runs=4,optimizer_epochs=complete['optimizer_epochs'])
    return dict(complete_sha256=file_hash(directory/'complete.json'))


def audit_unit(directory,sha256,task,training,y,query,*,settings=None,resource_upper_bound=False,ledger_root=None):
    from dataclasses import asdict
    validate_task(task);directory=Path(directory)
    if not sha256 or file_hash(directory/'complete.json')!=sha256:raise ValueError('Anchored DANet unit required')
    complete=json.loads((directory/'complete.json').read_text());settings=settings or Settings()
    expected={'receipt.json'}|{f'{r}-{a}.npz' for r in ('inner','outer') for a in ARMS}
    if (complete['task']!=task or complete['settings']!=asdict(settings) or complete['resource_upper_bound'] is not resource_upper_bound
            or complete['training_ids']!=training.sample_id.astype(str).tolist() or complete['query_ids']!=query.sample_id.astype(str).tolist()
            or set(complete['model_hashes'])!=expected or set(complete['prediction_hashes'])!={f'prediction-{a}.npy' for a in ARMS}):
        raise ValueError('DANet unit task/row/schema identity differs')
    expected_paths={'complete.json'}|{'models/'+k for k in expected}|set(complete['prediction_hashes'])
    if {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}!=expected_paths:raise ValueError('Unexpected/missing DANet unit artifact')
    predictions={}
    for name,digest in {**{'models/'+k:v for k,v in complete['model_hashes'].items()},**complete['prediction_hashes']}.items():
        path=directory/name
        if path.is_symlink() or file_hash(path)!=digest:raise ValueError('DANet unit artifact changed')
    for arm in ARMS:
        value=np.load(directory/f'prediction-{arm}.npy',allow_pickle=False)
        if value.shape!=(len(query),) or value.dtype!=np.float64 or not np.isfinite(value).all():raise ValueError('Invalid saved DANet prediction')
        predictions[arm]=value
    report=verify_pair(directory/'models',complete['model_hashes']['receipt.json'],training,y,query,predictions,settings=settings)
    for key in ('estimator_runs','optimizer_runs','optimizer_epochs'):
        if report[key]!=complete[key]:raise ValueError('DANet completed operation counts differ')
    receipt=json.loads((directory/'models/receipt.json').read_text())['receipt']
    if receipt['resource_upper_bound'] is not resource_upper_bound:raise ValueError('Resource-only upper-bound mode changed')
    if ledger_root is not None:
        ledger=ReservationLedger.open(ledger_root,complete['policy_sha256']);ledger.inspect();scope=validate_task(task)
        wanted={('pair_unit',scope)}|{(kind,(*scope,role,arm,kind)) for kind in ('estimator','optimizer') for role in ('inner','outer') for arm in ARMS}
        seen=set();epochs=0
        for path in (Path(ledger_root)/'events').glob('*.started.json'):
            event=json.loads(path.read_text())
            if tuple(event['key'][:3])!=scope:continue
            identity=event['kind'],tuple(event['key']);endpoint=path.with_name(path.name.replace('started.json','complete.json'))
            if identity not in wanted or not endpoint.exists():raise ValueError('Missing/extra DANet fitting reservation')
            if identity[0]=='pair_unit':
                result=json.loads(endpoint.read_text())['result']
                if result.get('complete_sha256')!=sha256 or any(result.get(k)!=report[k] for k in ('estimator_runs','optimizer_runs','optimizer_epochs')):
                    raise ValueError('DANet pair terminal result differs')
            else:
                role,arm=event['key'][3:5]
                expected_epochs=settings.max_epochs if role=='inner' or resource_upper_bound else receipt['selected_epochs'][arm]
                expected_rows=len(receipt['fitting_ids']) if role=='inner' else len(training)
                if event['payload']['epochs']!=expected_epochs or event['payload']['arm']!=arm or event['payload']['role']!=role:
                    raise ValueError('DANet estimator/optimizer reservation recipe differs')
                if event['payload']['rows']!=expected_rows or event['payload']['target']!=task['target']:
                    raise ValueError('DANet estimator/optimizer reservation training scope differs')
                if identity[0]=='optimizer':epochs+=event['payload']['epochs']
            seen.add(identity)
        if seen!=wanted or epochs!=report['optimizer_epochs']:raise ValueError('Missing/different DANet computational reservations')
    return predictions,report
