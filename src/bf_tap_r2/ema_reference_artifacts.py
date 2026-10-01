"""Immutable predictor witnesses for the matching EMA confirmation reference.

This module never fits an estimator or chooses a recipe. The caller supplies
an already-fitted object, an actual observed prediction and externally held
receipt identity. Original estimators and reference training are untouched.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import builtins
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd

from .data import TARGETS


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def _pickle_new(path, value):
    with Path(path).open('xb') as stream:
        pickle.dump(value, stream, protocol=5)
        stream.flush(); os.fsync(stream.fileno())


def _query(frame):
    if not isinstance(frame, pd.DataFrame) or 'sample_id' not in frame:
        raise ValueError('Predictor witness needs an explicit query ID frame')
    if any(name in frame for name in TARGETS):
        raise ValueError('Predictor witness query contains targets')
    if frame.empty or frame.sample_id.isna().any() or not frame.sample_id.is_unique:
        raise ValueError('Missing, duplicate or empty witness query IDs')


def _prediction(value, rows, shape=None):
    value = np.asarray(value, dtype=float)
    if (value.ndim not in (1, 2) or len(value) != rows
            or not np.isfinite(value).all() or (shape is not None and value.shape != shape)):
        raise ValueError('Invalid predictor witness shape or values')
    return value


def save_witness(model, query, observed, directory, *, identity, training_ids,
                 source_hashes, full_batch_atol, row_atol, fit_metadata):
    """Save states without predicting, refitting, deleting or changing attrs.

    A fresh directory is permanently consumed, including serialization failure.
    Prediction is the caller's original observed output, never another warm call.
    Inference-only copies must be supplied by the caller if an estimator has a
    non-pickleable training object; this function does not guess what to remove.
    """
    directory = Path(directory)
    _query(query); observed = _prediction(observed, len(query))
    required = {'source_directory', 'split_seed', 'fold', 'trial_id', 'fit_call_id'}
    if set(identity) != required or not identity['trial_id'] or not identity['fit_call_id']:
        raise ValueError('Incomplete source/seed/fold/trial/fit identity')
    if not Path(identity['source_directory']).is_absolute():
        raise ValueError('Source directory must be explicit and absolute')
    owner = Path(identity['source_directory']).resolve()
    if str(owner) != identity['source_directory']:
        raise ValueError('Original fit source directory must be canonical')
    if type(identity['split_seed']) is not int or type(identity['fold']) is not int:
        raise ValueError('Split identity must be integral')
    ids = list(training_ids)
    if not ids or len(set(ids)) != len(ids) or set(ids) & set(query.sample_id):
        raise ValueError('Invalid or overlapping training/query identity')
    tolerances = (full_batch_atol, row_atol)
    if any(not math.isfinite(x) or x < 0 for x in tolerances):
        raise ValueError('Invalid cold tolerance')
    if not source_hashes:
        raise ValueError('Missing predictor source identity')
    normalized = {str(Path(p).resolve()): value for p, value in source_hashes.items()}
    source = inspect.getsourcefile(type(model))
    if source is None or str(Path(source).resolve()) not in normalized:
        raise ValueError('Actual predictor class source is not frozen')
    for path, expected in normalized.items():
        if sha(path) != expected:
            raise ValueError('Predictor source changed before saving')
    directory.mkdir(parents=True, exist_ok=False)
    start = dict(identity=identity, started_ns=time.time_ns(), source_hashes=normalized,
                 training_ids=ids, query_ids=query.sample_id.tolist(), fit_metadata=fit_metadata,
                 full_batch_atol=full_batch_atol, row_atol=row_atol)
    try:
        write_new(directory/'start.json', start)
        _pickle_new(directory/'model.pkl', model)
        _pickle_new(directory/'query.pkl', query)
        with (directory/'observed.npy').open('xb') as stream:
            np.save(stream, observed, allow_pickle=False)
            stream.flush(); os.fsync(stream.fileno())
        receipt = dict(identity=identity, artifact_directory=str(directory.resolve()), source_hashes=normalized,
            model_class=type(model).__module__+'.'+type(model).__qualname__,
            model_source=str(Path(source).resolve()), model_source_sha256=sha(source),
            query_ids=query.sample_id.tolist(), training_ids=ids, fit_metadata=fit_metadata,
            shape=list(observed.shape), full_batch_atol=full_batch_atol, row_atol=row_atol,
            hashes={name:sha(directory/name) for name in ('start.json','model.pkl','query.pkl','observed.npy')})
        write_new(directory/'complete.json', receipt)
        return sha(directory/'complete.json')
    except BaseException as error:
        write_new(directory/'failure.json', dict(error=repr(error), identity=identity))
        raise


@contextmanager
def cold_only():
    """Reject fitting and CSV reads during imports, loading and inference."""
    original_open = builtins.open; original_io_open = io.open
    original_read_csv = pd.read_csv; original_profile = sys.getprofile()
    forbidden = {'fit','fit_transform','fit_predict','partial_fit','_train',
                 'fit_model','fit_with_inner_early_stop'}
    def guarded_open(original):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, os.PathLike)) and Path(path).suffix.lower() == '.csv':
                raise ValueError('Cold predictor attempted a CSV read/write')
            return original(path, *args, **kwargs)
        return call
    def no_csv(*args, **kwargs):
        raise ValueError('Cold predictor attempted a CSV read')
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code.co_name in forbidden:
            raise ValueError('Cold predictor attempted fitting: '+frame.f_code.co_name)
    builtins.open = guarded_open(original_open); io.open = guarded_open(original_io_open)
    pd.read_csv = no_csv; sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(original_profile); pd.read_csv = original_read_csv
        builtins.open = original_open; io.open = original_io_open


def audit_witness(directory, expected_receipt_sha256):
    """Load only the externally bound model/query and compare independent cold output."""
    directory = Path(directory)
    if (directory/'cold-audit.json').exists():
        raise FileExistsError('This predictor witness already has a cold audit')
    if sha(directory/'complete.json') != expected_receipt_sha256:
        raise ValueError('External predictor receipt identity mismatch')
    receipt = json.loads((directory/'complete.json').read_text())
    if any(not math.isfinite(receipt[k]) or receipt[k] < 0
           for k in ('full_batch_atol', 'row_atol')):
        raise ValueError('Invalid frozen cold tolerance')
    for path, expected in receipt['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Frozen predictor source changed before cold loading')
    for name, expected in receipt['hashes'].items():
        if sha(directory/name) != expected:
            raise ValueError('Predictor witness artifact changed')
    observed = np.load(directory/'observed.npy', allow_pickle=False)
    with cold_only():
        with (directory/'model.pkl').open('rb') as stream:
            model = pickle.load(stream)
        with (directory/'query.pkl').open('rb') as stream:
            query = pickle.load(stream)
        _query(query)
        if (query.sample_id.tolist() != receipt['query_ids']
                or set(query.sample_id) & set(receipt['training_ids'])
                or type(model).__module__+'.'+type(model).__qualname__ != receipt['model_class']):
            raise ValueError('Cold model/query/partition identity mismatch')
        actual_source = inspect.getsourcefile(type(model))
        if actual_source is None or sha(actual_source) != receipt['model_source_sha256']:
            raise ValueError('Actual cold predictor class source changed')
        predict = lambda q:_prediction(model.predict(q), len(q))
        result = predict(query)
        if result.shape != tuple(receipt['shape']) or observed.shape != result.shape:
            raise ValueError('Cold output shape changed')
        reverse = predict(query.iloc[::-1])[::-1]
        pieces = [predict(query.iloc[i:i+7]) for i in range(0, len(query), 7)]
        chunks = np.concatenate(pieces, axis=0)
        singleton = predict(query.iloc[:1])
        if reverse.shape != result.shape or chunks.shape != result.shape:
            raise ValueError('Cold order/chunk output shape changed')
        differences = dict(full=float(np.max(np.abs(result-observed))),
            reverse=float(np.max(np.abs(result-reverse))),
            chunk=float(np.max(np.abs(result-chunks))),
            singleton=float(np.max(np.abs(result[:1]-singleton))))
        if (differences['full'] > receipt['full_batch_atol']
                or max(differences[k] for k in ('reverse','chunk','singleton')) > receipt['row_atol']):
            raise ValueError('Saved-state cold or row-independent prediction mismatch')
    if sha(directory/'complete.json') != expected_receipt_sha256:
        raise ValueError('Predictor receipt changed during cold inference')
    for name, expected in receipt['hashes'].items():
        if sha(directory/name) != expected:
            raise ValueError('Predictor artifact changed during cold inference')
    for path, expected in receipt['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Predictor source changed during cold inference')
    payload = dict(status='passed', identity=receipt['identity'], receipt_sha256=expected_receipt_sha256,
                   differences=differences, rows=len(query), new_fits=0, training_csv_reads=0,
                   actual_cold_model_source=str(Path(actual_source).resolve()))
    write_new(directory/'cold-audit.json', payload)
    return payload


def main():
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError('Cold numerical thread contract mismatch')
    import torch
    if str(torch.__version__) != '2.14.0+cpu':
        raise ValueError('Cold CPU runtime identity mismatch')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(audit_witness(args.directory, args.receipt_sha256)), flush=True)


if __name__ == '__main__':
    main()
