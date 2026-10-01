"""Fail-closed native-call accounting for an isolated reference adapter.

This infrastructure does not choose a recipe, dispatch work or authorize fits.
The adapter must supply a frozen per-pipeline budget and ID partition before
calling the original estimators. Native methods retain their original results.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from .ema_reference_artifacts import sha, write_new


KINDS = ('catboost_fit', 'ebm_fit', 'ebm_boost', 'sklearn_mlp_fit', 'torch_optimizer')
_ACTIVE = ContextVar('ema_reference_native_ledger', default=None)
_PARTITION = ContextVar('ema_reference_native_partition', default=None)
_INSTALLED = False


def _ids(values):
    values = tuple(values)
    if not values or any(not isinstance(x, str) or not x for x in values) or len(set(values)) != len(values):
        raise ValueError('Native ledger needs unique nonempty string IDs')
    return values


class NativeLedger:
    """A fresh, permanently consumed pipeline scope; caught failures stay fatal."""

    def __init__(self, directory, *, identity, expected, training_ids, query_ids, source_hashes):
        required = {'source_directory', 'split_seed', 'fold', 'trial_id'}
        if set(identity) != required or not identity['trial_id']:
            raise ValueError('Incomplete native pipeline identity')
        owner = Path(identity['source_directory'])
        if not owner.is_absolute() or str(owner.resolve()) != str(owner):
            raise ValueError('Native ledger fit origin must be canonical')
        if type(identity['split_seed']) is not int or type(identity['fold']) is not int:
            raise ValueError('Nonintegral native split identity')
        if (set(expected) != set(KINDS) or any(type(n) is not int or n < 0 for n in expected.values())
                or not any(expected.values())):
            raise ValueError('Invalid frozen native call budget')
        self.training_ids = _ids(training_ids); self.query_ids = _ids(query_ids)
        if set(self.training_ids) & set(self.query_ids):
            raise ValueError('Native ledger training/query overlap')
        self.sources = {str(Path(p).resolve()): h for p, h in source_hashes.items()}
        if not self.sources:
            raise ValueError('Missing native ledger source identity')
        self.verify_sources()
        self.directory = Path(directory); self.identity = dict(identity); self.expected = dict(expected)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.counts = {k: 0 for k in KINDS}; self.ordinal = 0
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.closed = False
        write_new(self.directory/'scope-start.json', dict(identity=self.identity, expected=self.expected,
            training_ids=self.training_ids, query_ids=self.query_ids, source_hashes=self.sources,
            pid=self.pid, thread=self.thread, started_ns=time.time_ns()))
        self.start_sha = sha(self.directory/'scope-start.json')

    def verify_sources(self):
        for p, h in self.sources.items():
            if sha(p) != h:
                raise ValueError('Native ledger frozen source changed')

    def verify_owner(self):
        if (os.getpid(), threading.get_ident()) != (self.pid, self.thread):
            raise ValueError('Native ledger cannot be shared across workers or threads')
        if self.closed:
            raise ValueError('Native ledger is already closed')
        if sha(self.directory/'scope-start.json') != self.start_sha:
            raise ValueError('Native ledger scope identity changed')

    @contextmanager
    def partition(self, training_ids, calibration_ids=()):
        self.verify_owner()
        ids = _ids(training_ids); cal = tuple(calibration_ids)
        cal = _ids(cal) if cal else ()
        if (set(ids) & set(cal) or not (set(ids) | set(cal)) <= set(self.training_ids)):
            raise ValueError('Native fit/calibration partition escapes outer training')
        token = _PARTITION.set(dict(training_ids=ids, calibration_ids=cal))
        active = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(active); _PARTITION.reset(token)

    def call(self, kind, function, *, metadata=None, result_failed=None):
        self.verify_owner()
        partition = _PARTITION.get()
        if _ACTIVE.get() is not self or partition is None:
            raise ValueError('Native call has no declared training partition')
        self.ordinal += 1
        directory = self.directory/f'call-{self.ordinal:04d}'
        directory.mkdir(exist_ok=False)
        write_new(directory/'start.json', dict(identity=self.identity, kind=kind,
            partition=partition, metadata=metadata or {}, started_ns=time.time_ns(),
            pid=os.getpid(), thread=threading.get_ident()))
        invoked = False
        try:
            self.verify_sources()
            if kind not in self.expected:
                raise ValueError('Undeclared native call kind')
            self.counts[kind] += 1
            if self.counts[kind] > self.expected[kind]:
                raise ValueError('Frozen native call budget exceeded')
            invoked = True
            result = function()
            self.verify_sources()
            if result_failed is not None and result_failed(result):
                write_new(directory/'failure.json', dict(error='Native solver returned a failure value',
                    native_invoked=True, ended_ns=time.time_ns()))
            else:
                write_new(directory/'complete.json', dict(native_invoked=True, ended_ns=time.time_ns()))
            return result
        except BaseException as error:
            write_new(directory/'failure.json', dict(error=repr(error), native_invoked=invoked,
                ended_ns=time.time_ns()))
            raise

    def close(self):
        """Read actual files, including any failure swallowed by legacy fallbacks."""
        self.verify_owner(); self.verify_sources()
        counts = {k: 0 for k in KINDS}; files = {}
        failures = []; boost_stages = {'main':0, 'interaction':0}
        for d in sorted(self.directory.glob('call-*')):
            start = json.loads((d/'start.json').read_text()); kind = start['kind']
            if kind in counts:
                counts[kind] += 1
            if kind == 'ebm_boost':
                stage = start['metadata'].get('stage')
                if stage not in boost_stages:
                    failures.append(d.name+':unknown_boost_stage')
                else:
                    boost_stages[stage] += 1
            if start['identity'] != self.identity or kind not in counts or not (d/'complete.json').is_file() or (d/'failure.json').exists():
                failures.append(d.name)
            files.update({str(p.relative_to(self.directory)):sha(p) for p in d.iterdir() if p.is_file()})
        expected_stages = {k:self.expected['ebm_boost']//2 for k in boost_stages}
        if (counts != self.expected or failures or boost_stages != expected_stages
                or self.expected['ebm_boost'] % 2 or len(list(self.directory.glob('call-*'))) != self.ordinal):
            self.closed = True
            write_new(self.directory/'scope-failure.json', dict(counts=counts, expected=self.expected,
                failed_or_incomplete_calls=failures, boost_stages=boost_stages, ended_ns=time.time_ns()))
            raise ValueError('Native scope failed or did not close its frozen budget')
        self.closed = True
        payload = dict(status='passed', identity=self.identity, counts=counts,
            scope_start_sha256=self.start_sha, call_hashes=files, boost_stages=boost_stages,
            ended_ns=time.time_ns())
        write_new(self.directory/'scope-complete.json', payload)
        return payload


@dataclass(frozen=True)
class Binding:
    owner: object
    name: str
    kind: str
    metadata: object = None
    result_failed: object = None


@contextmanager
def native_hooks(bindings):
    """Install exact public/native entry hooks, restoring originals on exit.

    A caller in each worker needs its own NativeLedger partition. This is a
    single-thread adapter contract; an unscoped native call fails before entry.
    """
    global _INSTALLED
    if _INSTALLED:
        raise ValueError('Native hooks are already installed')
    originals = []; _INSTALLED = True
    try:
        for binding in bindings:
            if binding.kind not in KINDS:
                raise ValueError('Unknown native hook kind')
            original = getattr(binding.owner, binding.name)
            if any(owner is binding.owner and name == binding.name for owner, name, _, _, _ in originals):
                raise ValueError('Duplicate native hook binding')
            def wrapper_for(original, binding):
                @wraps(original)
                def wrapped(*args, **kwargs):
                    ledger = _ACTIVE.get()
                    if ledger is None:
                        raise ValueError('Unscoped native fitting/optimizer construction')
                    source = inspect.getsourcefile(inspect.unwrap(original))
                    if source is None or str(Path(source).resolve()) not in ledger.sources:
                        raise ValueError('Actual native entry source was not frozen')
                    metadata = {'entry':original.__module__+'.'+original.__qualname__,
                                'source':str(Path(source).resolve())}
                    if binding.metadata is not None:
                        metadata.update(binding.metadata(args, kwargs, _PARTITION.get()))
                    return ledger.call(binding.kind, lambda:original(*args, **kwargs),
                        metadata=metadata, result_failed=binding.result_failed)
                return wrapped
            wrapper = wrapper_for(original, binding)
            own_attribute = binding.name in vars(binding.owner)
            originals.append((binding.owner, binding.name, original, wrapper, own_attribute))
            setattr(binding.owner, binding.name, wrapper)
        yield
    finally:
        for owner, name, original, wrapper, own_attribute in reversed(originals):
            if own_attribute:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        _INSTALLED = False


def _boost_metadata(args, kwargs, partition):
    from interpret.glassbox._ebm._boost import boost
    bound = inspect.signature(boost).bind(*args, **kwargs).arguments
    bag = bound.get('bag'); terms = bound['term_features']
    if bag is None:
        raise ValueError('Unexpected unbagged EBM solver route; freeze it before use')
    bag = np.asarray(bag)
    ids = partition['training_ids']
    if bag.shape != (len(ids),) or not np.isin(bag, (-1, 1)).all():
        raise ValueError('Native EBM bag does not match declared fit IDs')
    if not terms or not (all(len(t) == 1 for t in terms) or all(len(t) == 2 for t in terms)):
        raise ValueError('Unexpected native EBM term stage')
    return dict(stage='main' if all(len(t) == 1 for t in terms) else 'interaction',
        term_features=[list(t) for t in terms], n_inner_bags=int(bound['n_inner_bags']),
        gradient_ids=[x for x, v in zip(ids, bag) if v > 0],
        calibration_ids=[x for x, v in zip(ids, bag) if v < 0])


def reference_bindings():
    """Exact entry list, not a fit or confirmation-run constructor."""
    from catboost import CatBoostRegressor
    from interpret.glassbox import ExplainableBoostingRegressor
    from interpret.glassbox._ebm import _ebm
    from sklearn.neural_network import MLPRegressor
    from torch.optim import Optimizer
    return [Binding(CatBoostRegressor, 'fit', 'catboost_fit'),
            Binding(ExplainableBoostingRegressor, 'fit', 'ebm_fit'),
            Binding(_ebm, 'boost', 'ebm_boost', _boost_metadata,
                    lambda result:result[0] is not None),
            Binding(MLPRegressor, 'fit', 'sklearn_mlp_fit'),
            Binding(Optimizer, '__init__', 'torch_optimizer')]


def binding_sources(bindings):
    result = {}
    for b in bindings:
        p = inspect.getsourcefile(inspect.unwrap(getattr(b.owner, b.name)))
        if p is None:
            raise ValueError('Native hook has no auditable source')
        result[str(Path(p).resolve())] = sha(p)
    return result
