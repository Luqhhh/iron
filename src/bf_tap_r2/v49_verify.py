"""Audit saved gradient-regularized states and traces without new model fits."""
import math
import numpy as np
import torch
from .data import FEATURES
from .v7_periodic import digest
from .v49_gradients import architecture


def verify_model(path, training, y, recipe, settings, trace=None):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    meta = saved['metadata']
    if saved['recipe'] != recipe or saved['settings'] != settings or meta['recipe'] != recipe:
        raise ValueError('Model spec mismatch')
    if meta['fit_ids_digest'] != digest(training.sample_id.astype(str).tolist()) or meta['fit_rows'] != len(training):
        raise ValueError('Fit identity mismatch')
    # Independent NumPy reconstruction of the train-only preprocessing metadata.
    x = training[list(FEATURES)].to_numpy(float)
    std = x.std(axis=0)
    if not np.isfinite(x).all() or np.isclose(std, 0.).any():
        raise ValueError('Invalid training features')
    categories = sorted(set(training.spout_no.astype(int)))
    info = dict(structure='raw_tabm', feature_names=list(FEATURES),
                means=x.mean(axis=0).tolist(), stds=std.tolist(), n_bins=16, d_embedding=8,
                spout_vocabulary={str(k): i for i, k in enumerate(categories, 1)},
                n_spout_categories=len(categories)+1, uses_ple=False, ple_bins_lengths=None)
    if meta['preprocessing'] != info:
        raise ValueError('Training-only preprocessing mismatch')
    y = np.asarray(y, float)
    if meta['target_mean'] != float(y.mean()) or meta['target_std'] != float(y.std()):
        raise ValueError('Target scaling mismatch')
    with torch.random.fork_rng(devices=[]):
        expected = architecture(settings, len(categories)+1)
    state = saved['state']
    if set(state) != set(expected.state_dict()):
        raise ValueError('Parameter keys mismatch')
    for key, reference in expected.state_dict().items():
        value = state[key]
        if value.shape != reference.shape or value.dtype != reference.dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid parameter shape/dtype/value')
    if meta['parameter_count'] != sum(p.numel() for p in expected.parameters()):
        raise ValueError('Parameter count mismatch')
    if trace is not None:
        if {k:v for k,v in meta.items() if k!='peak_rss_mib'} != {k:v for k,v in trace.items() if k!='peak_rss_mib'}:
            raise ValueError('Saved training trace mismatch')
    history = meta['history']
    if not 1 <= meta['selected_epoch'] <= meta['stopped_epoch'] <= settings['max_epochs']:
        raise ValueError('Invalid epoch range')
    if [r['epoch'] for r in history] != list(range(1, meta['stopped_epoch']+1)):
        raise ValueError('Missing/reordered training epochs')
    batches = math.ceil(len(training)/settings['batch_size'])
    active = recipe == 'TANGOS' and (settings['lambda_specialization']>0 or settings['lambda_orthogonalization']>0)
    total = 0
    for row in history:
        start = (row['epoch']-1)*batches
        count = sum(step % settings['auxiliary_every_steps'] == 0 for step in range(start, start+batches)) if active else 0
        total += count
        if row['auxiliary_updates'] != count:
            raise ValueError('Auxiliary update count mismatch')
        for key in ('standardized_training_mse', 'specialization', 'orthogonalization'):
            if not np.isfinite(row[key]) or row[key] < 0:
                raise ValueError('Nonfinite/negative training trace')
        if row['orthogonalization'] > 1+1e-12 or (not count and (row['specialization'] or row['orthogonalization'])):
            raise ValueError('Invalid auxiliary penalty trace')
    if total != meta['auxiliary_updates']:
        raise ValueError('Total auxiliary update count mismatch')
    if 'calibration_standardized_mae' in history[0]:
        best, selected, stale = float('inf'), 0, 0
        for row in history:
            if stale >= settings['patience']:
                raise ValueError('Training continued after patience')
            value = row['calibration_standardized_mae']
            if not np.isfinite(value) or value < 0:
                raise ValueError('Invalid calibration trace')
            if value < best-settings['min_delta_standardized_mae']:
                best, selected, stale = value, row['epoch'], 0
            else:
                stale += 1
        if meta['selected_epoch'] != selected:
            raise ValueError('Wrong selected epoch')
        if meta['stopped_epoch'] != settings['max_epochs'] and stale != settings['patience']:
            raise ValueError('Premature selector stop')
    elif any('calibration_standardized_mae' in row for row in history) or meta['selected_epoch'] != meta['stopped_epoch']:
        raise ValueError('Refit epoch mismatch')
    return meta
