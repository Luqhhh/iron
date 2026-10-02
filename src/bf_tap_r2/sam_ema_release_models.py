"""Observe original SAM-EMA selector/refit methods with bounded native accounting.

This module does not admit a scientific batch. The confirmation controller
must bind source, data, recipes and all partitions before calling it.
"""
from __future__ import annotations

from copy import deepcopy
from functools import wraps
import json
import os
from pathlib import Path

import numpy as np

from .sam_ema import SAMEMARegressor as ComponentRegressor
from .component_regularization_run import RECIPE
from .data import TARGETS
from .ema_reference_artifacts import cold_only,save_witness,sha,write_new
from .ema_reference_ledger import KINDS,NativeLedger,native_hooks,reference_bindings
from .v12_joint import JointRegressor
from .v3_4_bags import group_safe_inner_folds
from .v7_periodic import digest


def fit_component(directory,training,query,*,identity,settings,mechanisms,source_hashes):
    """Preserve ComponentRegressor.fit and _train; observe only initialization/save.

    Its selector retains selected EMA weights, unlike the original V7/V12
    terminal selector. Additional selector audit inference uses a separate copy.
    The final witness uses the one actual original query prediction.
    """
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    write_new(directory/'start.json',dict(identity=identity,settings=settings,mechanisms=mechanisms,
        training_ids=training.sample_id.tolist(),query_ids=query.sample_id.tolist()))
    try:
        if any(t in query for t in TARGETS) or set(training.sample_id)&set(query.sample_id):
            raise ValueError('EMA confirmation outer partition/query changed')
        expected={k:int(k=='torch_optimizer')*2 for k in KINDS}
        ledger=NativeLedger(directory/'native',identity=identity,expected=expected,
            training_ids=training.sample_id.tolist(),query_ids=query.sample_id.tolist(),source_hashes=source_hashes)
        assignment=np.asarray(group_safe_inner_folds(training,seed=settings['inner_seed'])['fold'])
        fitting=training.loc[assignment!=0].reset_index(drop=True)
        calibration=training.loc[assignment==0].reset_index(drop=True)
        frames=[fitting,training];calibrations=[calibration.sample_id.tolist(),[]]
        model=ComponentRegressor(RECIPE,settings,'SAM_EMA',mechanisms,directory)
        original_initialize=JointRegressor._initialize;original_save=ComponentRegressor.save
        initialization_count=0;checkpoints={};witnesses={}

        @wraps(original_initialize)
        def initialize(instance,frame,y):
            nonlocal initialization_count
            if instance is not model:
                raise ValueError('Unexpected model initialization during EMA scope')
            position=initialization_count
            if position>=2 or digest(frame.to_dict(orient='list'))!=digest(frames[position].to_dict(orient='list')):
                raise ValueError('Original SAM-EMA initialization partition changed')
            initialization_count+=1
            expected_y=frames[position][['tap_time_len']].to_numpy()
            if not np.array_equal(y,expected_y):raise ValueError('Original SAM-EMA initialization targets changed')
            with ledger.partition(frame.sample_id.tolist(),calibrations[position]):
                return original_initialize(instance,frame,y)

        @wraps(original_save)
        def save(instance,path,trace):
            if instance is not model:raise ValueError('Unexpected model checkpoint during EMA scope')
            phase=Path(path).stem
            if phase not in {'selection','refit'} or phase in checkpoints:
                raise ValueError('Unexpected original SAM-EMA checkpoint call')
            required_position=1 if phase=='selection' else 2
            fit=frames[required_position-1]
            if initialization_count!=required_position or trace['fit_ids_digest']!=digest(fit.sample_id.tolist()):
                raise ValueError('Original SAM-EMA checkpoint fit identity changed')
            result=original_save(instance,path,trace)
            checkpoints[phase]=dict(sha256=sha(path),trace=deepcopy(trace))
            if phase=='selection':
                request=calibration.drop(columns=list(TARGETS),errors='ignore')
                with cold_only():
                    selected=deepcopy(instance);observed=selected.predict(request)
                witnesses['selection']=save_witness(selected,request,observed,directory/'selection-witness',
                    identity=dict(identity,fit_call_id='selection'),training_ids=fit.sample_id.tolist(),
                    source_hashes=source_hashes,full_batch_atol=0,row_atol=.0005,
                    fit_metadata=dict(state_kind='original_selected_EMA_network',trace=trace,
                        observation='additional label-free audit inference on an independent selected-state copy',
                        original_factory_prediction=False))
            return result

        try:
            JointRegressor._initialize=initialize;ComponentRegressor.save=save
            with native_hooks(reference_bindings()):
                model.fit(training,training[['tap_time_len']].to_numpy())
                prediction=model.predict(query)
        finally:
            JointRegressor._initialize=original_initialize;ComponentRegressor.save=original_save
        if initialization_count!=2 or set(checkpoints)!={'selection','refit'}:
            raise ValueError('Original SAM-EMA selector/refit checkpoint scope incomplete')
        metadata=model.metadata_
        if (metadata['optimizer_runs']!=2 or metadata['fit_ids_digest']!=digest(training.sample_id.tolist())
                or checkpoints['selection']['trace']['selected_epoch']!=metadata['selected_epoch']
                or checkpoints['refit']['trace']['selected_epoch']!=metadata['selected_epoch']):
            raise ValueError('Original SAM-EMA selected epoch/refit metadata changed')
        if prediction.shape!=(len(query),1) or not np.isfinite(prediction).all():
            raise ValueError('Original SAM-EMA query prediction invalid')
        witnesses['refit']=save_witness(model,query,prediction,directory/'refit-witness',
            identity=dict(identity,fit_call_id='refit'),training_ids=training.sample_id.tolist(),
            source_hashes=source_hashes,full_batch_atol=0,row_atol=.0005,
            fit_metadata=dict(state_kind='original_fresh_refit_EMA_network',model_metadata=metadata,
                original_factory_prediction=True))
        native=ledger.close()
        with (directory/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream,prediction=prediction,query_ids=query.sample_id.to_numpy(str))
        payload=dict(identity=identity,initializations=initialization_count,model_metadata=metadata,
            checkpoints=checkpoints,witnesses=witnesses,native_counts=native['counts'],
            native_receipt_sha256=sha(directory/'native/scope-complete.json'),
            start_sha256=sha(directory/'start.json'),prediction_sha256=sha(directory/'predictions.npz'),
            status='original_SAM_EMA_warm_closed_cold_audits_pending')
        write_new(directory/'complete.json',payload)
        return prediction[:,0],payload
    except BaseException as error:
        write_new(directory/'failure.json',dict(error=repr(error),identity=identity))
        raise


def audit_component(directory,receipt_sha256):
    """Called in the controller's independent cold process, without CSV reads."""
    from .ema_reference_artifacts import audit_witness
    directory=Path(directory)
    if (directory/'cold-complete.json').exists():raise FileExistsError('EMA state already cold audited')
    if sha(directory/'complete.json')!=receipt_sha256:raise ValueError('EMA external warm receipt changed')
    receipt=json.loads((directory/'complete.json').read_text())
    if receipt['status']!='original_SAM_EMA_warm_closed_cold_audits_pending' or set(receipt['witnesses'])!={'selection','refit'}:
        raise ValueError('EMA warm state coverage changed')
    if (directory/'failure.json').exists():raise ValueError('EMA warm failure preserved')
    if sha(directory/'start.json')!=receipt['start_sha256'] or sha(directory/'predictions.npz')!=receipt['prediction_sha256']:
        raise ValueError('EMA warm partition or prediction changed')
    for phase,record in receipt['checkpoints'].items():
        if sha(directory/(phase+'.pt'))!=record['sha256']:raise ValueError('EMA native checkpoint changed')
    native_path=directory/'native/scope-complete.json'
    if sha(native_path)!=receipt['native_receipt_sha256']:raise ValueError('EMA native ledger changed')
    native=json.loads(native_path.read_text())
    expected={k:int(k=='torch_optimizer')*2 for k in KINDS}
    native_start=directory/'native/scope-start.json'
    if sha(native_start)!=native['scope_start_sha256']:raise ValueError('EMA native scope start changed')
    scope=json.loads(native_start.read_text());start=json.loads((directory/'start.json').read_text())
    if os.getpid()==scope['pid']:raise ValueError('EMA cold audit must run in an independent process')
    if (native['status']!='passed' or native['identity']!=receipt['identity'] or scope['identity']!=receipt['identity']
            or native['counts']!=expected or receipt['native_counts']!=expected or scope['expected']!=expected
            or scope['training_ids']!=start['training_ids'] or scope['query_ids']!=start['query_ids']):
        raise ValueError('EMA native optimizer budget changed')
    actual={str(p.relative_to(directory/'native')) for d in (directory/'native').glob('call-*') for p in d.iterdir() if p.is_file()}
    if actual!=set(native['call_hashes']) or len(list((directory/'native').glob('call-*')))!=2:
        raise ValueError('EMA native call coverage changed')
    for relative,h in native['call_hashes'].items():
        if sha(directory/'native'/relative)!=h:raise ValueError('EMA native call artifact changed')
    audits={}
    for phase,h in receipt['witnesses'].items():
        path=directory/(phase+'-witness');record=json.loads((path/'complete.json').read_text())
        if record['identity']!=dict(receipt['identity'],fit_call_id=phase):
            raise ValueError('EMA cold model source/seed/trial/fit identity changed')
        if record['source_hashes']!=scope['source_hashes'] or record['full_batch_atol']!=0 or record['row_atol']!=.0005:
            raise ValueError('EMA cold source or tolerance changed')
        fit_ids=record['training_ids'];query_ids=record['query_ids']
        if phase=='refit':
            if fit_ids!=start['training_ids'] or query_ids!=start['query_ids']:
                raise ValueError('EMA final model outer partition changed')
        elif set(fit_ids)&set(query_ids) or set(fit_ids)|set(query_ids)!=set(start['training_ids']):
            raise ValueError('EMA selected-state partition changed')
        audits[phase]=audit_witness(path,h)
    payload=dict(status='passed',warm_receipt_sha256=receipt_sha256,identity=receipt['identity'],
        retained_states=2,audits=audits,cold_pid=os.getpid(),new_fits=0,training_csv_reads=0)
    write_new(directory/'cold-complete.json',payload)
    return payload
