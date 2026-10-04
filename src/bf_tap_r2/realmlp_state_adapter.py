"""Capture V9's native selector/refit objects without replacing its fit method.

This module provides persistence primitives, not scientific fitting admission.
Only hash-bound private snapshots produced by this module are loadable.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np

from . import v9_realmlp as original
from .data import TARGETS
from .v3_4_bags import group_safe_inner_folds


VERSION = 'realmlp-native-state-v1'
PRIVATE_ROOT = Path('/home/lux1/iron/local')
ROLES = ('selection', 'refit')
DEPENDENCIES = ('pytabkit', 'pytorch-lightning', 'torchmetrics', 'torch',
                'numpy', 'pandas', 'scikit-learn')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def validate_identity(identity, parent):
    if (set(identity) != {'source_directory', 'split_seed', 'fold', 'trial_id'}
            or identity['source_directory'] != str(Path(parent).resolve())
            or type(identity['split_seed']) is not int or type(identity['fold']) is not int
            or not isinstance(identity['trial_id'], str) or not identity['trial_id']):
        raise ValueError('Physical source/split/fold/trial identity required')


@dataclass
class CapturedFit:
    regressor: object
    estimators: tuple
    encoders: tuple
    fitting_ids: tuple
    target_digests: tuple


def capture_fit(recipe, frame, y, *, inner_seed=42):
    """Call the original fit once and retain its two returned native objects.

    Temporary factory interception is for one serial worker only. It creates no
    additional encoder or estimator and does not call a replacement fit loop.
    """
    y = np.asarray(y, dtype=float)
    if (frame.sample_id.duplicated().any() or y.shape != (len(frame),)
            or not np.isfinite(y).all()):
        raise ValueError('Unique fitting identities and aligned finite target required')
    inner_mask = group_safe_inner_folds(frame, seed=inner_seed)['fold'] != 0
    estimators, encoders = [], []
    make, encoder_type = original.make_estimator, original.InputEncoder

    def make_native(*args, **kwargs):
        if len(estimators) >= 2:
            raise ValueError('Unexpected third native estimator')
        value = make(*args, **kwargs)
        estimators.append(value)
        return value

    def make_encoder(*args, **kwargs):
        if len(encoders) >= 2:
            raise ValueError('Unexpected third input encoder')
        value = encoder_type(*args, **kwargs)
        encoders.append(value)
        return value

    regressor = original.RealMLPRegressor(deepcopy(recipe), inner_seed=inner_seed)
    with patch.object(original, 'make_estimator', side_effect=make_native), \
            patch.object(original, 'InputEncoder', side_effect=make_encoder):
        regressor.fit(frame, y)
    if (len(estimators) != 2 or len(encoders) != 2
            or regressor.model_ is not estimators[1] or regressor.encoder_ is not encoders[1]):
        raise ValueError('Original selector/refit object contract differs')
    epoch = int(estimators[0].fit_params_['stop_epoch']['mae'])
    if regressor.metadata_['selected_epoch'] != epoch:
        raise ValueError('Original native selected epoch differs')
    return CapturedFit(regressor, tuple(estimators), tuple(encoders),
        (tuple(frame.loc[inner_mask, 'sample_id']), tuple(frame.sample_id)),
        (digest(y[inner_mask].tolist()), digest(y.tolist())))


def expected_native_config(recipe, role, epoch):
    if role not in ROLES:
        raise ValueError('Expected selection or refit role')
    result = deepcopy(recipe['resolved'])
    if role == 'refit':
        result.update(stop_epoch=epoch, val_fraction=0.)
    return result


def validate_cleanup(module):
    # LightningModule.__getstate__ restores the empty trainer slot when pickling.
    # A None slot is not a live Trainer; retained data loaders remain forbidden.
    if getattr(module, '_trainer', None) is not None or any(
            hasattr(module, k) for k in ('train_dl', 'val_dl', 'callbacks')):
        raise ValueError('Native post-fit cleanup incomplete')


def validate_native(estimator, recipe, role, epoch):
    import torch
    from pytabkit import RealMLP_TD_Regressor
    if recipe['class'] != 'RealMLP_TD_Regressor' or type(estimator) is not RealMLP_TD_Regressor:
        raise ValueError('Only the frozen native full TD recipe is supported')
    if estimator.get_config() != expected_native_config(recipe, role, epoch):
        raise ValueError('Native configuration differs from frozen recipe/role')
    if not hasattr(estimator, 'alg_interface_'):
        raise ValueError('Native fitted state is absent')
    module = estimator.alg_interface_.model
    validate_cleanup(module)
    state = module.model.state_dict()
    if not state or any(v.device.type != 'cpu' or (v.is_floating_point() and not torch.isfinite(v).all())
                        for v in state.values()):
        raise ValueError('Finite CPU native state required')
    if role == 'selection' and int(estimator.fit_params_['stop_epoch']['mae']) != epoch:
        raise ValueError('Native selector epoch differs')


def header_for(captured, role, identity, parent):
    validate_identity(identity, parent)
    if role not in ROLES:
        raise ValueError('Expected selection or refit role')
    i = ROLES.index(role)
    recipe = captured.regressor.recipe
    metadata = captured.regressor.metadata_
    validate_native(captured.estimators[i], recipe, role, metadata['selected_epoch'])
    return dict(version=VERSION, role=role, identity=deepcopy(identity), recipe=deepcopy(recipe),
        metadata=deepcopy(metadata), fitting_ids_digest=digest(list(captured.fitting_ids[i])),
        fitting_rows=len(captured.fitting_ids[i]), target_digest=captured.target_digests[i],
        encoder_medians=captured.encoders[i].medians_.tolist(),
        encoder_categories=captured.encoders[i].categories_.tolist(),
        runtime_versions={p:importlib.metadata.version(p) for p in DEPENDENCIES},
        original_source_sha256=sha(original.__file__), adapter_source_sha256=sha(__file__))


def save_snapshot(captured, role, path, identity):
    path = Path(path).resolve()
    if not path.is_relative_to(PRIVATE_ROOT):
        raise ValueError('Private snapshot path required')
    sidecar = path.with_suffix(path.suffix + '.json')
    if path.exists() or sidecar.exists():
        raise FileExistsError('Snapshot path already consumed')
    header = header_for(captured, role, identity, path.parent)
    # Validate JSON serialization before creating any binary artifact.
    json.dumps(header, allow_nan=False)
    i = ROLES.index(role)
    payload = dict(header=header, estimator=captured.estimators[i], encoder=captured.encoders[i])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        pickle.dump(payload, stream, protocol=5)
    record = dict(header=header, state_sha256=sha(path))
    with sidecar.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False); stream.write('\n')
    return record


def load_snapshot(path, *, expected_identity, expected_role, expected_recipe):
    path = Path(path).resolve()
    if not path.is_relative_to(PRIVATE_ROOT):
        raise ValueError('Private snapshot path required')
    validate_identity(expected_identity, path.parent)
    record = json.loads(path.with_suffix(path.suffix + '.json').read_text())
    h = record['header']
    if (h['version'] != VERSION or h['identity'] != expected_identity
            or h['role'] != expected_role or h['role'] not in ROLES or h['recipe'] != expected_recipe
            or h['original_source_sha256'] != sha(original.__file__)
            or h['adapter_source_sha256'] != sha(__file__)
            or h['runtime_versions'] != {p:importlib.metadata.version(p) for p in DEPENDENCIES}
            or record['state_sha256'] != sha(path)):
        raise ValueError('Snapshot identity, source, runtime or byte hash differs')
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    if set(payload) != {'header', 'estimator', 'encoder'} or payload['header'] != h:
        raise ValueError('Snapshot payload/header differs')
    validate_native(payload['estimator'], h['recipe'], h['role'], h['metadata']['selected_epoch'])
    np.testing.assert_array_equal(payload['encoder'].medians_, h['encoder_medians'])
    np.testing.assert_array_equal(payload['encoder'].categories_, h['encoder_categories'])
    return payload


def predict_snapshot(payload, query):
    if any(name in query for name in TARGETS):
        raise ValueError('Prediction query contains targets')
    result = np.asarray(payload['estimator'].predict(payload['encoder'].transform(query)), dtype=float)
    if result.shape != (len(query),) or not np.isfinite(result).all():
        raise ValueError('Invalid saved native prediction')
    return result
