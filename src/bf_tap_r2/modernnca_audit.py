"""Independent no-fit inference, preprocessing and epoch-selection audit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURES
from .modernnca_model import Settings,validate_features,validate_pair
from .v7_periodic import digest,file_hash


def independent_inputs(frame, preprocessor):
    raw = frame[list(FEATURES)].to_numpy(float)
    means,stds = np.array(preprocessor['means']),np.array(preprocessor['stds'])
    numeric = ((raw-means)/stds).astype(np.float32).astype(np.float64)
    codes = [preprocessor['spout_vocabulary'].get(str(int(s)),0) for s in frame.spout_no]
    cat = np.zeros((len(frame),preprocessor['n_spout_categories']),dtype=np.float64)
    cat[np.arange(len(frame)),codes] = 1
    return np.column_stack([numeric,cat])


def encode_numpy(x,state):
    angles = 2*np.pi*state['num_embeddings.0.frequencies'][None]*x[:,:len(FEATURES),None]
    periodic = np.concatenate([np.cos(angles),np.sin(angles)],axis=2)
    embedded = np.maximum(periodic @ state['num_embeddings.1.weight'].T + state['num_embeddings.1.bias'],0)
    columns = np.column_stack([embedded.reshape(len(x),-1),x[:,len(FEATURES):]])
    return columns @ state['encoder.weight'].T + state['encoder.bias']


def numpy_predict(frame,metadata,state,bank_x,bank_y,chunk=16):
    validate_features(frame)
    if type(chunk) is not int or chunk < 1:
        raise ValueError('Positive query chunk required')
    queries = encode_numpy(independent_inputs(frame,metadata['preprocessor']),state)
    bank = encode_numpy(bank_x,state)
    out = []
    for start in range(0,len(queries),chunk):
        q = queries[start:start+chunk]
        distances = np.sqrt(np.sum((q[:,None,:]-bank[None,:,:])**2,axis=2))
        logits = -distances/metadata['settings']['temperature']
        weights = np.exp(logits-logits.max(axis=1,keepdims=True))
        weights /= weights.sum(axis=1,keepdims=True)
        out.extend((weights @ bank_y)*metadata['std']+metadata['mean'])
    prediction = np.asarray(out)
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise ValueError('Independent neighbor inference is invalid')
    return prediction


def read_state(path,expected_sha256):
    path = Path(path)
    if path.is_symlink() or file_hash(path) != expected_sha256:
        raise ValueError('Caller-anchored saved state changed')
    with np.load(path,allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        state = {k[7:]:archive[k].copy() for k in archive.files if k.startswith('state::')}
        initial = {k[9:]:archive[k].copy() for k in archive.files if k.startswith('initial::')}
        others = {k:archive[k].copy() for k in ('bank_x','bank_y','calibration')}
        if set(archive.files) != {'metadata',*others}|{f'state::{k}' for k in state}|{f'initial::{k}' for k in initial}:
            raise ValueError('Unknown saved-state fields')
    settings = Settings(**metadata['settings'])
    categories = metadata['preprocessor']['n_spout_categories']
    shapes = {'encoder.weight':(settings.dim,len(FEATURES)*settings.embedding+categories),
        'encoder.bias':(settings.dim,),'num_embeddings.0.frequencies':(len(FEATURES),settings.frequencies),
        'num_embeddings.1.weight':(settings.embedding,2*settings.frequencies),'num_embeddings.1.bias':(settings.embedding,)}
    if set(state) != set(shapes) or set(initial) != set(shapes):
        raise ValueError('Exact author-default parameter fields required')
    for values in (state,initial):
        if any(values[k].shape != shape or values[k].dtype != np.float64 or not np.isfinite(values[k]).all() for k,shape in shapes.items()):
            raise ValueError('Invalid float64 saved parameters')
    if metadata['format'] != 'modernnca-partition-v1' or metadata['arm'] not in ('FIXED_ENCODER','LEARNED_ENCODER'):
        raise ValueError('Wrong saved-model format/arm')
    return metadata,state,initial,others


def audit_partition(path,expected_sha256,training,y,query,calibration=None):
    validate_pair(training,query)
    metadata,state,initial,arrays = read_state(path,expected_sha256)
    y = np.asarray(y,float);raw=training[list(FEATURES)].to_numpy(float)
    if (metadata['fit_ids'] != training.sample_id.astype(str).tolist() or metadata['fit_y_digest'] != digest(y.tolist())
            or metadata['fit_frame_digest'] != digest(pd.util.hash_pandas_object(training,index=True).tolist())):
        raise ValueError('Saved fitting rows or responses changed')
    vocabulary = {str(s):i for i,s in enumerate(sorted({int(s) for s in training.spout_no}),1)}
    pp = metadata['preprocessor']
    if (pp['structure'] != 'raw_tabm' or pp['feature_names'] != list(FEATURES)
            or pp['spout_vocabulary'] != vocabulary or pp['n_spout_categories'] != len(vocabulary)+1
            or not np.array_equal(pp['means'],raw.mean(0)) or not np.array_equal(pp['stds'],raw.std(0))
            or metadata['mean'] != float(y.mean()) or metadata['std'] != float(y.std())):
        raise ValueError('Saved preprocessing/target scale is not training-only')
    bank_x,bank_y = arrays['bank_x'],arrays['bank_y']
    if (bank_x.dtype != np.float64 or bank_y.dtype != np.float64
            or not np.array_equal(bank_x,independent_inputs(training,pp))
            or not np.array_equal(bank_y,(y-y.mean())/y.std())):
        raise ValueError('Saved neighbor context differs from fitting partition')
    epochs,selected = metadata['actual_epochs'],metadata['selected_epoch']
    settings = Settings(**metadata['settings'])
    if not 1 <= selected <= epochs <= settings.max_epochs:
        raise ValueError('Invalid saved epoch protocol')
    if metadata['arm'] == 'FIXED_ENCODER':
        if epochs != 1 or selected != 1 or metadata['optimizer_steps'] != 0 or any(not np.array_equal(state[k],initial[k]) for k in state):
            raise ValueError('Fixed encoder changed or performed optimization')
    elif metadata['optimizer_steps'] != epochs*((len(training)+settings.batch_size-1)//settings.batch_size):
        raise ValueError('Actual optimizer-step witness differs')
    if calibration is None:
        if metadata['history'] or arrays['calibration'].shape != (0,) or selected != epochs:
            raise ValueError('Fresh outer refit cannot inherit calibration history')
    else:
        features,truth = calibration;validate_pair(training,features);truth=np.asarray(truth,float)
        if truth.shape != (len(features),) or not np.isfinite(truth).all() or len(metadata['history']) != epochs:
            raise ValueError('Matching finite calibration responses/history required')
        predictions = arrays['calibration']
        if predictions.shape != (epochs,len(features)) or not np.isfinite(predictions).all():
            raise ValueError('Complete calibration prediction history required')
        best,best_epoch,stale = float('inf'),0,0
        for index,p in enumerate(predictions,1):
            error = float(np.abs(p-truth).mean())
            if metadata['history'][index-1] != dict(epoch=index,mae=error):
                raise ValueError('Calibration score history is inconsistent')
            if error < best-settings.min_delta:best,best_epoch,stale=error,index,0
            else:stale += 1
            if index < epochs and (stale >= settings.patience or metadata['arm'] == 'FIXED_ENCODER'):
                raise ValueError('Training continued past calibration stop')
        if best_epoch != selected or (epochs < settings.max_epochs and stale < settings.patience and metadata['arm'] != 'FIXED_ENCODER'):
            raise ValueError('Selected/stopped epoch differs from fixed rule')
        cold_calibration = numpy_predict(features,metadata,state,bank_x,bank_y)
        if np.max(np.abs(cold_calibration-predictions[selected-1])) > 1e-8:
            raise ValueError('Selected saved state differs from its calibration predictions')
    prediction = numpy_predict(query,metadata,state,bank_x,bank_y)
    reverse = numpy_predict(query.iloc[::-1],metadata,state,bank_x,bank_y)[::-1]
    single = numpy_predict(query,metadata,state,bank_x,bank_y,chunk=1)
    difference = max(float(np.max(np.abs(prediction-reverse))),float(np.max(np.abs(prediction-single))))
    if difference > 1e-8:
        raise ValueError('Independent inference depends on query order/chunk')
    return prediction,dict(status='passed',selected_epoch=selected,actual_epochs=epochs,
        optimizer_steps=metadata['optimizer_steps'],train_rows=len(training),query_rows=len(query),
        maximum_batch_difference=difference,new_estimator_fits=0,new_optimizer_calls=0,
        parameters_changed=any(not np.array_equal(state[k],initial[k]) for k in state))
