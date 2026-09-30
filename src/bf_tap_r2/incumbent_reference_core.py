"""Reserved native BASE reference fits; no incumbent promotion or release.

The original ComponentRegressor and joint modules are unchanged. Reservation
hooks wrap optimizer creation and training without consuming numerical RNG.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .component_regularization import ComponentRegressor
from .component_regularization_audit import verify_saved
from .data import FEATURES, TARGETS
from .incumbent_reference_ledger import ReservationLedger, file_hash, write_new
from .v3_4_bags import group_safe_inner_folds
from .v7_periodic import digest

RECIPE=dict(backbone='tabm',frequency=.01)


class ReservedRegressor(ComponentRegressor):
    def __init__(self,settings,directory,ledger,key):
        super().__init__(RECIPE,settings,'BASE',{},directory)
        self.ledger,self.key=ledger,tuple(key)
        self.initializations=0
        self.active=None
        self.fit_started=False

    def _initialize(self,frame,y):
        if self.initializations>=2 or self.active is not None:
            raise ValueError('No repeated fit or checkpoint resume')
        phase=('selection','refit')[self.initializations]
        self.initializations+=1
        context=self.ledger.event('optimizer',(*self.key,phase),
            dict(phase=phase,fit_ids_digest=digest(frame.sample_id.tolist()),settings_digest=digest(self.settings)))
        receipt=context.__enter__()
        self.active=(context,receipt,phase)
        try:super()._initialize(frame,y)
        except BaseException as exc:
            self._end(exc)
            raise

    def _end(self,exception=None):
        if self.active is None:return
        context,_,_=self.active;self.active=None
        context.__exit__(type(exception) if exception is not None else None,exception,
                         exception.__traceback__ if exception is not None else None)

    def _train(self,frame,y,epochs,validation=None):
        if self.active is None:raise ValueError('Optimizer reservation must precede training')
        _,receipt,phase=self.active
        if (validation is not None)!=(phase=='selection'):
            exc=ValueError('Selector/refit reservation mismatch');self._end(exc);raise exc
        try:
            selected=super()._train(frame,y,epochs,validation)
            receipt.update(model_sha256=file_hash(self.directory/f'{phase}.pt'),
                selected_epoch=selected,stopped_epoch=self.traces[phase]['stopped_epoch'])
            self._end()
            return selected
        except BaseException as exc:
            self._end(exc)
            raise

    def fit(self,frame,y):
        if self.fit_started:raise ValueError('No repeated fit or checkpoint resume')
        self.fit_started=True
        try:return super().fit(frame,y)
        except BaseException as exc:
            self._end(exc)
            raise


def validate_rows(training,query):
    for frame in (training,query):
        if (not len(frame) or frame.columns.duplicated().any() or frame.sample_id.isna().any()
                or frame.sample_id.duplicated().any() or not frame.sample_id.map(lambda v:isinstance(v,str)).all()):
            raise ValueError('Nonempty unique string row identities required')
        x=frame[[*FEATURES,'spout_no']].to_numpy(float)
        if not np.isfinite(x).all() or not np.equal(x[:,-1],np.floor(x[:,-1])).all():
            raise ValueError('Finite features and integer categories required')
    if set(training.sample_id)&set(query.sample_id) or set(query.columns)&set(TARGETS):
        raise ValueError('Training/query overlap or query labels')
    hashes=[set(pd.util.hash_pandas_object(f[list(FEATURES)],index=False)) for f in (training,query)]
    if hashes[0]&hashes[1]:raise ValueError('Training/query duplicate-feature leakage')


def frame_digest(frame):return digest(frame.to_dict('list'))


def execute_estimator(training,query,settings,key,output,ledger_root,policy_sha256):
    ledger=ReservationLedger.open(ledger_root,policy_sha256)
    with ledger.event('estimator',key,dict(settings_digest=digest(settings))) as receipt:
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
        validate_rows(training,query)
        y=training[list(TARGETS)].to_numpy(float)
        if not np.isfinite(y).all() or not (y.std(0)>0).all():raise ValueError('Invalid joint training targets')
        model=ReservedRegressor(settings,output,ledger,key).fit(training,y)
        pred=model.predict(query)
        with (output/'predictions.npz').open('xb') as stream:
            np.savez(stream,query_ids=query.sample_id.to_numpy(dtype=str),prediction=pred)
        complete=dict(key=list(key),settings=settings,policy_sha256=policy_sha256,
            training_digest=frame_digest(training),query_digest=frame_digest(query),metadata=model.metadata_,
            artifacts={name:file_hash(output/name) for name in ('selection.pt','refit.pt','predictions.npz')})
        write_new(output/'complete.json',complete)
        anchor=file_hash(output/'complete.json');receipt.update(complete_sha256=anchor)
    return dict(complete_sha256=anchor,key=list(key))


def audit_estimator(output,training,query,settings,key,expected_sha256,ledger_root,policy_sha256):
    output=Path(output)
    validate_rows(training,query)
    if file_hash(output/'complete.json')!=expected_sha256:raise ValueError('External estimator anchor mismatch')
    complete=json.loads((output/'complete.json').read_text())
    expected=dict(key=list(key),settings=settings,policy_sha256=policy_sha256,
                  training_digest=frame_digest(training),query_digest=frame_digest(query))
    if any(complete[k]!=v for k,v in expected.items()):raise ValueError('Reference estimator partition/spec mismatch')
    if {p.name for p in output.iterdir()}!={'selection.pt','refit.pt','predictions.npz','complete.json'}:
        raise ValueError('Unexpected/incomplete reference estimator artifacts')
    if set(complete['artifacts'])!={'selection.pt','refit.pt','predictions.npz'}:
        raise ValueError('Incomplete reference artifact anchors')
    for name,sha in complete['artifacts'].items():
        if file_hash(output/name)!=sha:raise ValueError('Reference model/prediction changed')
    inner=np.asarray(group_safe_inner_folds(training,seed=settings['inner_seed'])['fold'])
    fit=training.loc[inner!=0].reset_index(drop=True);cal=training.loc[inner==0]
    y=training[list(TARGETS)].to_numpy(float)
    # Match the native boolean-sliced reduction layout; do not re-extract the
    # selector targets from a new DataFrame and invent an equality tolerance.
    selector=verify_saved(output/'selection.pt',fit,y[inner!=0],'BASE',settings,{},cal)
    final=verify_saved(output/'refit.pt',training,y,'BASE',settings,{},
                       expected_epoch=selector.saved['trace']['selected_epoch'])
    with np.load(output/'predictions.npz',allow_pickle=False) as saved:
        if set(saved.files)!={'query_ids','prediction'} or not np.array_equal(saved['query_ids'],query.sample_id.to_numpy(dtype=str)):
            raise ValueError('Reference prediction identity/schema mismatch')
        prediction=saved['prediction'].copy()
    if prediction.shape!=(len(query),2) or not np.isfinite(prediction).all():
        raise ValueError('Invalid joint reference prediction')
    observed=final.predict(query)
    if not np.array_equal(prediction,observed):raise ValueError('Full-batch cold reference difference')
    scale=max(1.,float(np.abs(observed).max()))
    variants=[final.predict(query.iloc[::-1])[::-1],np.concatenate([final.predict(query.iloc[i:i+37]) for i in range(0,len(query),37)])]
    absolute=max(float(np.abs(p-observed).max()) for p in variants)
    delta=absolute/scale
    if delta>1e-6 or absolute>5e-4:raise ValueError('Cold reference order/chunk difference')
    for phase,m in [('selection',selector),('refit',final)]:
        if m.saved['trace']!=complete['metadata']['traces'][phase]:raise ValueError('Reference saved trace mismatch')
    ledger=ReservationLedger.open(ledger_root,policy_sha256);ledger.inspect()
    expected_events={('estimator',tuple(key)):dict(payload=dict(settings_digest=digest(settings)),result=dict(complete_sha256=expected_sha256))}
    for phase,part,m in [('selection',fit,selector),('refit',training,final)]:
        expected_events['optimizer',(*key,phase)]=dict(payload=dict(phase=phase,fit_ids_digest=digest(part.sample_id.tolist()),settings_digest=digest(settings)),
            result=dict(model_sha256=complete['artifacts'][f'{phase}.pt'],selected_epoch=m.saved['trace']['selected_epoch'],
                        stopped_epoch=m.saved['trace']['stopped_epoch']))
    found=set()
    for path in (ledger.root/'events').glob('*.started.json'):
        start=json.loads(path.read_text());event=start['kind'],tuple(start['key'])
        if tuple(start['key'][:len(key)])!=tuple(key):continue
        if event not in expected_events:raise ValueError('Unexpected estimator reservation')
        end=path.with_name(path.name.replace('started.json','complete.json'))
        if not end.exists():raise ValueError('Incomplete estimator reservation')
        result=json.loads(end.read_text())
        if start['payload']!=expected_events[event]['payload'] or result['result']!=expected_events[event]['result']:
            raise ValueError('Reference reservation result mismatch')
        found.add(event)
    if found!=set(expected_events):raise ValueError('Missing estimator reservations')
    return prediction,dict(status='passed',cold_models=2,full_batch_difference=0.,
        maximum_order_chunk_relative_difference=delta,maximum_order_chunk_absolute_difference=absolute,
        new_optimizer_calls=0,scope='one supplied reference estimator_not_incumbent_promotion')
