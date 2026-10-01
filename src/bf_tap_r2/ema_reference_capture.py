"""Observe the unchanged matching reference factory's fits and predictions.

Runtime wrappers preserve the original classes, methods, recipes and returned
arrays. Each original worker creates its own pipeline ledger. The factory still
uses its original single-worker ProcessPoolExecutor and original composition.
This module supplies infrastructure, not scientific execution admission.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import inspect
import json
import math
import multiprocessing
from pathlib import Path
import sys
import weakref

import numpy as np

from .data import TARGETS
from .ema_reference_artifacts import cold_only, save_witness, sha, write_new, _query
from .ema_reference_ledger import KINDS, NativeLedger, native_hooks, reference_bindings
from .ema_reference_selector import observe_selector
from .v7_periodic import digest


_MANAGER=ContextVar('ema_reference_capture_manager',default=None)
_ROLE=ContextVar('ema_reference_capture_role',default=None)
_FIT_DEPTH=ContextVar('ema_reference_capture_fit_depth',default=0)
_PREDICT_DEPTH=ContextVar('ema_reference_capture_predict_depth',default=0)


def _same(left,right):
    return digest(left)==digest(right)


def _frame_ids(frame):
    ids=frame.sample_id.tolist()
    if not ids or any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('Reference capture requires unique string frame IDs')
    return ids


class Pipeline:
    def __init__(self,manager,name):
        self.manager=manager;self.name=name;self.plan=manager.roles[name]
        self.directory=manager.directory/name
        self.directory.mkdir(exist_ok=False)
        self.identity=dict(source_directory=manager.source_directory,
            split_seed=manager.split_seed,fold=manager.fold,trial_id=name)
        write_new(self.directory/'start.json',dict(identity=self.identity,plan=self.plan,
            training_ids=manager.training_ids,query_ids=manager.query_ids,
            query_frame_digest=digest(manager.query.to_dict(orient='list'))))
        self.start_sha=sha(self.directory/'start.json')
        self.ledger=NativeLedger(self.directory/'native',identity=self.identity,
            expected=self.plan['expected_native_calls'],training_ids=manager.training_ids,
            query_ids=manager.query_ids,source_hashes=manager.sources)
        self.root_ref=None;self.top_fits=0;self.selected_tree_fields=None;self.witnesses={}
        self.fit_frames=[];self.closed=False

    def validate_top(self,model,frame,eval_frame):
        if self.closed:raise ValueError('Completed reference pipeline cannot be refitted')
        ids=_frame_ids(frame);self.top_fits+=1
        expected=self.plan['recipe'];family=self.plan['family']
        if hasattr(model,'trial'):
            actual=model.trial
        elif hasattr(model,'parameters'):
            actual=model.parameters
        elif hasattr(model,'spec'):
            actual=model.spec
        else:
            actual=dict(recipe=model.recipe,training=model.settings)
        if family=='legacy_inner_select_refit_tree':
            from .v3_1_models import _clone_trial_with_iterations
            if self.top_fits==1:
                if eval_frame is None:raise ValueError('Original tree selector lacks calibration')
                from .splits import make_folds
                assignment=make_folds(self.manager.training_features,seed=self.plan['inner_seed'],n_splits=5)
                mask=assignment.set_index('sample_id').loc[self.manager.training_ids,'fold'].to_numpy()==0
                expected_fit=np.asarray(self.manager.training_ids)[~mask].tolist()
                expected_cal=np.asarray(self.manager.training_ids)[mask].tolist()
                if ids!=expected_fit or _frame_ids(eval_frame)!=expected_cal:
                    raise ValueError('Original tree selector partition changed')
            elif self.top_fits==2 and self.selected_tree_fields is not None:
                expected=_clone_trial_with_iterations(expected,self.selected_tree_fields['selected_num_boost_round'])
                if eval_frame is not None or ids!=self.manager.training_ids:
                    raise ValueError('Original tree refit did not use the full ordered training pool')
            else:raise ValueError('Unexpected original tree selector/refit call')
        elif self.top_fits!=1 or ids!=self.manager.training_ids or eval_frame is not None:
            raise ValueError('Original reference top-level fit partition changed')
        if not _same(actual,expected):
            raise ValueError('Original reference constructor recipe changed')
        self.root_ref=weakref.ref(model)
        self.manager.models[id(model)]=(weakref.ref(model),self)

    def record_fit(self,model,frame,eval_frame):
        for candidate in (frame,eval_frame):
            if candidate is None:continue
            ids=_frame_ids(candidate)
            if not set(ids)<=set(self.manager.training_ids):
                raise ValueError('Original fit frame escapes the outer training pool')
            actual=candidate.drop(columns=list(TARGETS),errors='ignore')
            expected=self.manager.training_features.set_index('sample_id',drop=False).loc[ids]
            if not _same(actual.to_dict(orient='list'),expected.to_dict(orient='list')):
                raise ValueError('Original fit feature values or order changed')
        self.fit_frames.append(dict(model_class=type(model).__module__+'.'+type(model).__qualname__,
            training_ids=_frame_ids(frame),calibration_ids=[] if eval_frame is None else _frame_ids(eval_frame)))

    def save(self,key,model,query,observed,training_ids,metadata):
        if key in self.witnesses:raise ValueError('Reference witness already captured')
        identity=dict(self.identity,fit_call_id=key)
        receipt=save_witness(model,query,observed,self.directory/key,identity=identity,
            training_ids=training_ids,source_hashes=self.manager.sources,
            full_batch_atol=self.manager.full_atol,row_atol=self.manager.row_atol,
            fit_metadata=metadata)
        self.witnesses[key]=receipt
        return receipt

    def tree_probe(self,model,frame,calibration):
        from .v3_1_models import iteration_fields,configured_rounds
        fields=iteration_fields(model.family,model._delegate.estimator_,configured_rounds(model.trial))
        self.selected_tree_fields=fields
        query=calibration.drop(columns=list(TARGETS),errors='ignore')
        # The original helper never predicts with its probe. Audit a separate
        # copy, leaving the object used for its selected-round reflection intact.
        with cold_only():
            copy=deepcopy(model);observed=copy.predict(query)
        self.save('tree-selector',copy,query,observed,_frame_ids(frame),dict(
            state_kind='original_retained_inner_tree',selection_fields=fields,
            observation='additional label-free audit inference on an independent model copy',
            original_factory_prediction=False))

    def finish(self,model,query,observed):
        if _frame_ids(query)!=self.manager.query_ids or not _same(query.to_dict(orient='list'),self.manager.query.to_dict(orient='list')):
            raise ValueError('Original reference prediction query changed')
        expected_fits=2 if self.plan['family']=='legacy_inner_select_refit_tree' else 1
        if self.top_fits!=expected_fits:raise ValueError('Original reference pipeline fit count changed')
        self.save('final',model,query,observed,self.manager.training_ids,dict(
            original_factory_prediction=True,fit_frames=self.fit_frames,
            native_model_metadata=_json_metadata(model)))
        if self.plan['family']=='matching_selector_fresh_refit_network':
            selector=self.directory/'selector-terminal/complete.json'
            receipt=json.loads(selector.read_text())
            self.witnesses['selector-terminal/model-witness']=receipt['witness_receipt_sha256']
        if self.plan['family']=='legacy_inner_select_refit_tree' and 'tree-selector' not in self.witnesses:
            raise ValueError('Original inner tree state was not captured')
        native=self.ledger.close()
        self.closed=True
        write_new(self.directory/'complete.json',dict(identity=self.identity,start_sha256=self.start_sha,
            plan_digest=digest(self.plan),top_fits=self.top_fits,fit_frames=self.fit_frames,
            witnesses=self.witnesses,native_receipt_sha256=sha(self.directory/'native/scope-complete.json'),
            native_counts=native['counts']))


def _json_metadata(model):
    result={}
    for obj in (model,getattr(model,'impl',None)):
        if obj is None:continue
        for name in ('metadata_','fit_meta_','fit_report_'):
            if hasattr(obj,name):result[type(obj).__name__+'.'+name]=getattr(obj,name)
    return json.loads(json.dumps(result,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))


class ReferenceCapture:
    def __init__(self,directory,*,source_directory,split_seed,fold,training,query,plan,
                 source_hashes,full_batch_atol=0,row_atol=.0005):
        plan=deepcopy(plan)
        _query(query)
        self.training_ids=_frame_ids(training);self.query_ids=_frame_ids(query)
        if set(self.training_ids)&set(self.query_ids):raise ValueError('Reference capture outer ID overlap')
        self.roles={r['name']:r for r in plan['roles']}
        if len(self.roles)!=32 or len(plan['roles'])!=32 or plan['pipelines']!=32:
            raise ValueError('Reference capture requires all 32 frozen roles')
        if any(not isinstance(n,str) or not n or Path(n).name!=n or n in {'.','..'} for n in self.roles):
            raise ValueError('Invalid reference role directory name')
        totals={k:sum(r['expected_native_calls'].get(k,0) for r in self.roles.values()) for k in KINDS}
        if totals!=plan['expected_native_calls_per_factory']:
            raise ValueError('Reference plan native totals changed')
        for role in self.roles.values():
            counts=role['expected_native_calls']
            if set(counts)!=set(KINDS) or any(type(n) is not int or n<0 for n in counts.values()) or not any(counts.values()):
                raise ValueError('Invalid reference role budget')
        if type(split_seed) is not int or type(fold) is not int or fold<0:
            raise ValueError('Invalid reference split identity')
        if any(not math.isfinite(x) or x<0 for x in (full_batch_atol,row_atol)):
            raise ValueError('Invalid reference cold tolerance')
        self.source_directory=str(Path(source_directory).resolve())
        self.split_seed=split_seed;self.fold=fold;self.query=query.copy(deep=True)
        self.training_features=training.drop(columns=list(TARGETS),errors='ignore').copy(deep=True)
        self.sources={str(Path(p).resolve()):h for p,h in source_hashes.items()}
        if not self.sources or any(sha(p)!=h for p,h in self.sources.items()):
            raise ValueError('Reference capture source identity missing or changed')
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.plan_digest=digest(plan);self.expected_totals=totals
        self.full_atol=full_batch_atol;self.row_atol=row_atol;self.models={}
        write_new(self.directory/'start.json',dict(source_directory=self.source_directory,
            split_seed=split_seed,fold=fold,plan_digest=digest(plan),plan=plan,source_hashes=self.sources,
            training_ids=self.training_ids,query_ids=self.query_ids,
            training_feature_digest=digest(self.training_features.to_dict(orient='list')),
            query_frame_digest=digest(self.query.to_dict(orient='list')),
            full_batch_atol=full_batch_atol,row_atol=row_atol))
        self.start_sha=sha(self.directory/'start.json')

    def new_role(self,name):
        if name not in self.roles:raise ValueError('Unregistered reference pipeline')
        return Pipeline(self,name)

    @contextmanager
    def role(self,name):
        if _ROLE.get() is not None:raise ValueError('Nested top-level reference roles')
        role=self.new_role(name);token=_ROLE.set(role)
        try:
            yield role
            if not role.closed:raise ValueError('Original reference did not emit its final prediction')
        except BaseException as error:
            if not (role.directory/'failure.json').exists():
                write_new(role.directory/'failure.json',dict(error=repr(error),identity=role.identity))
            raise
        finally:_ROLE.reset(token)

    def auto_name(self,model):
        name=type(model).__name__
        if name in ('V34Regressor','V36Regressor'):
            trial=model.trial['trial_id']
            return 'N0048' if trial=='v36-s1-N-0048' else trial
        if name=='JointRegressor':return 'V12_joint'
        if name=='PeriodicRegressor':return 'V7_periodic'
        raise ValueError('Legacy reference fit has no registered named role')

    def context_for(self,model):
        entry=self.models.get(id(model))
        return entry[1] if entry is not None and entry[0]() is model else None

    def close(self):
        if sha(self.directory/'start.json')!=self.start_sha:
            raise ValueError('Reference capture start identity changed')
        expected_dirs=set(self.roles)
        if {p.name for p in self.directory.iterdir() if p.is_dir()}!=expected_dirs:
            raise ValueError('Reference capture role coverage changed')
        summaries={};totals={k:0 for k in next(iter(self.roles.values()))['expected_native_calls']}
        for name,plan in self.roles.items():
            directory=self.directory/name;complete=json.loads((directory/'complete.json').read_text())
            identity=dict(source_directory=self.source_directory,split_seed=self.split_seed,fold=self.fold,trial_id=name)
            start=json.loads((directory/'start.json').read_text())
            if complete['identity']!=identity or start['identity']!=identity or start['plan']!=plan:
                raise ValueError('Reference pipeline identity changed')
            if start['training_ids']!=self.training_ids or start['query_ids']!=self.query_ids:
                raise ValueError('Reference pipeline outer partition changed')
            if start['query_frame_digest']!=digest(self.query.to_dict(orient='list')):
                raise ValueError('Reference pipeline query features changed')
            if (directory/'failure.json').exists() or complete['plan_digest']!=digest(plan):
                raise ValueError('Reference pipeline failure or recipe identity changed')
            if sha(directory/'start.json')!=complete['start_sha256']:
                raise ValueError('Reference pipeline start identity changed')
            if sha(directory/'native/scope-complete.json')!=complete['native_receipt_sha256']:
                raise ValueError('Reference native ledger identity changed')
            native=json.loads((directory/'native/scope-complete.json').read_text())
            native_start=json.loads((directory/'native/scope-start.json').read_text())
            if sha(directory/'native/scope-start.json')!=native['scope_start_sha256']:
                raise ValueError('Reference native scope start changed')
            if (native['status']!='passed' or native['identity']!=identity
                    or native['counts']!=plan['expected_native_calls'] or native['counts']!=complete['native_counts']
                    or native_start['identity']!=identity or native_start['expected']!=plan['expected_native_calls']
                    or native_start['training_ids']!=self.training_ids or native_start['query_ids']!=self.query_ids
                    or native_start['source_hashes']!=self.sources):
                raise ValueError('Reference native scope partition, sources or budget changed')
            for p,h in native['call_hashes'].items():
                if sha(directory/'native'/p)!=h:raise ValueError('Reference native call artifact changed')
            expected_witnesses={'final'}
            if plan['family']=='legacy_inner_select_refit_tree':expected_witnesses.add('tree-selector')
            if plan['family']=='matching_selector_fresh_refit_network':expected_witnesses.add('selector-terminal/model-witness')
            if set(complete['witnesses'])!=expected_witnesses:
                raise ValueError('Reference retained model state coverage changed')
            for key,h in complete['witnesses'].items():
                if sha(directory/key/'complete.json')!=h:raise ValueError('Reference model witness identity changed')
            for k,n in complete['native_counts'].items():totals[k]+=n
            summaries[name]=dict(receipt_sha256=sha(directory/'complete.json'),witnesses=complete['witnesses'])
        if totals!=self.expected_totals:raise ValueError('Reference factory native totals changed')
        for p,h in self.sources.items():
            if sha(p)!=h:raise ValueError('Reference capture frozen source changed')
        payload=dict(status='warm_original_factory_capture_closed_cold_audits_pending',
            pipelines=len(summaries),native_counts=totals,roles=summaries,
            source_directory=self.source_directory,split_seed=self.split_seed,fold=self.fold,
            start_sha256=self.start_sha,plan_digest=self.plan_digest)
        write_new(self.directory/'complete.json',payload)
        return payload


def _fit_wrapper(original):
    @wraps(original)
    def wrapped(model,*args,**kwargs):
        manager=_MANAGER.get()
        if manager is None:return original(model,*args,**kwargs)
        bound=inspect.signature(original).bind(model,*args,**kwargs);bound.apply_defaults()
        frame=bound.arguments['frame'];cal=bound.arguments.get('eval_frame')
        parent=_ROLE.get();top=_FIT_DEPTH.get()==0
        role=parent or manager.new_role(manager.auto_name(model))
        token=_ROLE.set(role);depth=_FIT_DEPTH.set(_FIT_DEPTH.get()+1)
        try:
            if top:role.validate_top(model,frame,cal)
            role.record_fit(model,frame,cal)
            ids=_frame_ids(frame);cal_ids=[] if cal is None else _frame_ids(cal)
            if type(model).__name__=='V36NetworkRegressor':
                from .v3_4_bags import group_safe_inner_folds
                folds=group_safe_inner_folds(frame,n_splits=model.params.get('inner_validation_folds',5),
                    seed=model.params.get('inner_validation_seed',42))['fold']
                cal_ids=frame.loc[np.asarray(folds)==0,'sample_id'].tolist()
                ids=frame.loc[np.asarray(folds)!=0,'sample_id'].tolist()
            with role.ledger.partition(ids,cal_ids):result=original(model,*args,**kwargs)
            if top and cal is not None:role.tree_probe(model,frame,cal)
            return result
        except BaseException as error:
            if not (role.directory/'failure.json').exists():
                write_new(role.directory/'failure.json',dict(error=repr(error),identity=role.identity))
            raise
        finally:_FIT_DEPTH.reset(depth);_ROLE.reset(token)
    return wrapped


def _predict_wrapper(original):
    @wraps(original)
    def wrapped(model,*args,**kwargs):
        manager=_MANAGER.get();role=None if manager is None else manager.context_for(model)
        capture=role is not None and _FIT_DEPTH.get()==0 and _PREDICT_DEPTH.get()==0
        depth=_PREDICT_DEPTH.set(_PREDICT_DEPTH.get()+1)
        try:
            result=original(model,*args,**kwargs)
            if capture:
                bound=inspect.signature(original).bind(model,*args,**kwargs)
                role.finish(model,bound.arguments['frame'],result)
            return result
        except BaseException as error:
            if capture and not (role.directory/'failure.json').exists():
                write_new(role.directory/'failure.json',dict(error=repr(error),identity=role.identity))
            raise
        finally:_PREDICT_DEPTH.reset(depth)
    return wrapped


def _initialize_wrapper(original):
    @wraps(original)
    def wrapped(model,frame,y):
        role=_ROLE.get()
        if role is None:return original(model,frame,y)
        ids=_frame_ids(frame);pool=role.manager.training_ids
        cal=[x for x in pool if x not in set(ids)]
        with role.ledger.partition(ids,cal):return original(model,frame,y)
    return wrapped


def _train_wrapper(original):
    @wraps(original)
    def wrapped(model,frame,y,epochs,validation=None):
        role=_ROLE.get()
        if role is None:return original(model,frame,y,epochs,validation)
        ids=_frame_ids(frame);cal=[] if validation is None else _frame_ids(validation[0])
        with role.ledger.partition(ids,cal):
            return observe_selector(original,model,frame,y,epochs,validation,
                role.directory/'selector-terminal',identity=dict(role.identity,fit_call_id='selector-terminal'),
                source_hashes=role.manager.sources,full_batch_atol=role.manager.full_atol,row_atol=role.manager.row_atol)
    return wrapped


@contextmanager
def capture_original_factory(manager):
    """Preserve original source and class identities while observing calls."""
    if _MANAGER.get() is not None:raise ValueError('Reference capture already active')
    if sys.version_info[:2]!=(3,12) or multiprocessing.get_context().get_start_method()!='fork':
        raise ValueError('Original reference worker needs the frozen Python3.12 fork path')
    from . import v4_1_reference
    from .v3_local_search import TrialRegressor
    from .v3_1_models import V31Regressor
    from .normalized_models import JointSnapshotRegressor
    from .nonlinear_models import NonlinearRegressor
    from .v3_4_models import V34Regressor
    from .v3_6_models import V36Regressor
    from .v3_6_networks import V36NetworkRegressor
    from .v12_joint import JointRegressor
    from .v7_periodic import PeriodicRegressor
    classes=(TrialRegressor,V31Regressor,JointSnapshotRegressor,NonlinearRegressor,
             V34Regressor,V36Regressor,V36NetworkRegressor,JointRegressor,PeriodicRegressor)
    originals=[];token=_MANAGER.set(manager)
    def install(owner,name,wrapper):
        original=getattr(owner,name);own=name in vars(owner)
        originals.append((owner,name,original,own));setattr(owner,name,wrapper(original))
    def recovery_loader(original):
        @wraps(original)
        def load(root):
            recovery=original(root);fit=recovery.fit_model
            @wraps(fit)
            def named(name,tr,va):
                with manager.role(name):return fit(name,tr,va)
            recovery.fit_model=named
            return recovery
        return load
    try:
        for cls in classes:
            install(cls,'fit',_fit_wrapper)
            if cls is not V36NetworkRegressor:install(cls,'predict',_predict_wrapper)
        for cls in (JointRegressor,PeriodicRegressor):
            install(cls,'_initialize',_initialize_wrapper);install(cls,'_train',_train_wrapper)
        install(v4_1_reference,'_load_l1_recovery',recovery_loader)
        with native_hooks(reference_bindings()):yield
    finally:
        for owner,name,original,own in reversed(originals):
            if own:setattr(owner,name,original)
            else:delattr(owner,name)
        _MANAGER.reset(token)
