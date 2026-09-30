"""Independent NumPy inference and zero-fit prototype/trace verification.

Uses saved tensors directly: no production model, preprocessor, prototype
builder, training or clustering call is used by the inference oracle.
"""
import math

import numpy as np
import torch

from .data import FEATURES, TARGETS
from .v7_periodic import digest, file_hash


def saved(path, expected_sha256):
    if not expected_sha256 or file_hash(path) != expected_sha256:
        raise ValueError('External model hash mismatch')
    return torch.load(path, map_location='cpu', weights_only=True)


def close(left, right, message, atol=1e-9):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(message+': shape/value mismatch')
    delta = float(np.abs(a-b).max()) if a.size else 0.
    if delta > atol:
        raise ValueError(f'{message}: difference {delta}')
    return delta


def oracle(data, frame):
    if any(t in frame for t in TARGETS):
        raise ValueError('Query labels forbidden')
    s, meta = data['settings'], data['metadata']
    info = meta['preprocessing']
    # Native preprocessor rounds standardized inputs to float32 before casting
    # them to the float64 training/inference tensors. Preserve that contract.
    x = ((frame[list(FEATURES)].to_numpy(float)-np.array(info['means'])) /
         np.array(info['stds'])).astype(np.float32).astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Invalid query values')
    codes = [info['spout_vocabulary'].get(str(int(v)), 0) for v in frame.spout_no]
    onehot = np.eye(info['n_spout_categories'])[codes]
    prefix = 'native.' if 'arm' in data else ''
    p = {k[len(prefix):]: v.numpy() for k,v in data['state'].items() if k.startswith(prefix)}
    if s['lite'] is not True:
        raise ValueError('Only frozen lite periodic embedding is implemented')
    angle = 2*math.pi*p['num_module.periodic.weight'][None,:,:]*x[:,:,None]
    trig = np.concatenate([np.cos(angle), np.sin(angle)], axis=2)
    emb = np.maximum(trig @ p['num_module.linear.weight'].T+p['num_module.linear.bias'], 0.)
    inputs = np.column_stack([emb.reshape(len(x), -1), onehot])
    h = np.broadcast_to(inputs[:,None,:], (len(x),s['tabm_k'],inputs.shape[1])).copy()
    for i in range(s['blocks']):
        name = f'backbone.blocks.{i}.0.'
        h = np.maximum(((h*p[name+'r']) @ p[name+'weight'].T)*p[name+'s']+p[name+'bias'], 0.)
    members = np.einsum('bkd,kdo->bko', h, p['output.weight'])+p['output.bias']
    prediction = members[:,:,0].mean(1)*meta['target_std']+meta['target_mean']
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite independent prediction')
    return prediction, h.mean(1)


def verify_model(path, training, y, arm, settings, expected_sha256, *, selector, trace=None):
    data = saved(path, expected_sha256)
    meta, state = data['metadata'], data['state']
    if data['settings'] != settings or data['arm'] != arm or meta['arm'] != arm:
        raise ValueError('Frozen model recipe mismatch')
    if meta['task_head'] != 'native_direct_latent' or meta['recipe'] != 'BASE':
        raise ValueError('Task head mismatch')
    x = training[list(FEATURES)].to_numpy(float)
    categories = sorted(set(training.spout_no.astype(int)))
    prep = dict(structure='raw_tabm', feature_names=list(FEATURES), means=x.mean(0).tolist(),
        stds=x.std(0).tolist(), n_bins=16, d_embedding=8,
        spout_vocabulary={str(k):i for i,k in enumerate(categories,1)},
        n_spout_categories=len(categories)+1, uses_ple=False, ple_bins_lengths=None)
    if not np.isfinite(x).all() or (np.isclose(x.std(0),0)).any() or meta['preprocessing'] != prep:
        raise ValueError('Training-only preprocessing mismatch')
    ids = training.sample_id.astype(str).tolist()
    if len(set(ids)) != len(ids) or meta['fit_rows'] != len(ids) or meta['fit_ids_digest'] != digest(ids):
        raise ValueError('Fitting identity mismatch')
    y = np.asarray(y, float)
    if y.shape != (len(ids),) or not np.isfinite(y).all() or y.std() <= 0:
        raise ValueError('Invalid original training targets')
    if meta['target_mean'] != float(y.mean()) or meta['target_std'] != float(y.std()):
        raise ValueError('Training-only target scale mismatch')
    width,k,e,f = settings['width'],settings['tabm_k'],settings['embedding_dim'],settings['n_frequencies']
    shapes = {'native.num_module.periodic.weight':(len(FEATURES),f),
        'native.num_module.linear.weight':(e,2*f), 'native.num_module.linear.bias':(e,),
        'native.output.weight':(k,width,1), 'native.output.bias':(k,1),
        'prototypes':(settings['prototype_count'],width),
        'initial_prototypes':(settings['prototype_count'],width)}
    d = len(FEATURES)*e+len(categories)+1
    for i in range(settings['blocks']):
        n=f'native.backbone.blocks.{i}.0.'
        shapes.update({n+'weight':(width,d),n+'r':(k,d),n+'s':(k,width),n+'bias':(k,width)})
        d=width
    for i in range(3):
        shapes[f'coordinates.{3*i}.weight']=(width,width)
        shapes[f'coordinates.{3*i}.bias']=(width,)
    shapes['coordinates.9.weight']=(settings['prototype_count'],width)
    shapes['coordinates.9.bias']=(settings['prototype_count'],)
    if set(state) != set(shapes):
        raise ValueError('Saved parameter keys mismatch')
    for name,shape in shapes.items():
        value=state[name]
        if tuple(value.shape) != shape or value.dtype != torch.float64 or not torch.isfinite(value).all():
            raise ValueError('Saved parameter shape/dtype/value mismatch')
    # initial_prototypes is a buffer, not an optimizable parameter.
    count = sum(np.prod(v) for name,v in shapes.items() if name != 'initial_prototypes')
    if meta['parameter_count'] != count:
        raise ValueError('Saved parameter count mismatch')
    receipt=meta['prototype_initialization']
    if (receipt['fit_ids_digest'] != digest(ids) or receipt['latent_rows'] != len(ids)
            or receipt['latent_width'] != width or receipt['prototype_count'] != settings['prototype_count']
            or digest(state['initial_prototypes'].numpy().tolist()) != receipt['centers_digest']
            or digest(state['prototypes'].numpy().tolist()) != meta['prototype_state_digest']):
        raise ValueError('Prototype receipt mismatch')
    history=meta['history']
    if (not history or not 1 <= meta['selected_epoch'] <= meta['stopped_epoch'] <= settings['max_epochs']
            or [r['epoch'] for r in history] != list(range(1,meta['stopped_epoch']+1))):
        raise ValueError('Missing/out-of-budget epochs')
    keys={'epoch','standardized_training_mse','auxiliary_updates','projection','diversity','orthogonalization'}
    if selector:keys.add('calibration_standardized_mae')
    active = arm == 'PTARL_AUX' and any(settings['auxiliary_weights'].values())
    updates=math.ceil(len(ids)/settings['batch_size']) if active else 0
    if meta['auxiliary_updates'] != updates*len(history):
        raise ValueError('Total auxiliary update mismatch')
    best,selected,stale=float('inf'),0,0
    for row in history:
        if set(row) != keys or row['auxiliary_updates'] != updates:
            raise ValueError('Training trace schema/update mismatch')
        for name in keys-{'epoch','auxiliary_updates'}:
            if not np.isfinite(row[name]) or row[name]<0:
                raise ValueError('Invalid training trace value')
        terms=[row[n] for n in ('projection','diversity','orthogonalization')]
        if (not active and any(terms)) or (active and (row['projection']>2+1e-12 or row['orthogonalization']<1-1e-10)):
            raise ValueError('Invalid auxiliary term range')
        if selector:
            if stale >= settings['patience']:
                raise ValueError('Continued after patience')
            if row['calibration_standardized_mae']<best-settings['min_delta_standardized_mae']:
                best,selected,stale=row['calibration_standardized_mae'],row['epoch'],0
            else:stale+=1
    if selector:
        if meta['selected_epoch'] != selected or (len(history)<settings['max_epochs'] and stale != settings['patience']):
            raise ValueError('Wrong epoch selection/stop')
    elif meta['selected_epoch'] != meta['stopped_epoch']:
        raise ValueError('Refit selected epoch mismatch')
    if trace is not None and meta != trace:
        raise ValueError('Saved training metadata mismatch')
    return data


def verify_bank(teacher_data, fitting, receipt, centers):
    """Zero-fit nearest-cluster/centroid/inertia witnesses; not a KMeans replay."""
    _,hidden=oracle(teacher_data,fitting.drop(columns=list(TARGETS),errors='ignore'))
    labels=np.asarray(receipt['cluster_labels'])
    n=receipt['prototype_count']
    if (labels.shape != (len(fitting),) or not np.issubdtype(labels.dtype,np.integer)
            or set(labels) != set(range(n)) or not 1<=receipt['kmeans_iterations']<=100):
        raise ValueError('Cluster witness mismatch')
    distances=np.square(hidden[:,None,:]-centers[None,:,:]).sum(2)
    if not np.array_equal(labels,distances.argmin(1)):
        raise ValueError('Cluster nearest-center membership mismatch')
    means=np.stack([hidden[labels==i].mean(0) for i in range(n)])
    close(np.abs(means-centers).max(),receipt['centroid_mean_max_difference'],'Centroid residual witness')
    close(distances[np.arange(len(hidden)),labels].sum(),receipt['inertia'],'Cluster inertia witness',1e-7)
    if (receipt['fit_ids_digest'] != teacher_data['metadata']['fit_ids_digest']
            or receipt['teacher_selected_epoch'] != teacher_data['metadata']['selected_epoch']):
        raise ValueError('Teacher/prototype partition binding mismatch')
    return hidden
