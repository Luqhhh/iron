"""Fold-local positive numeric geometry for the frozen V2.8 kernel experiment."""
from __future__ import annotations

import json
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURES, TARGETS
from .models import inputs


def numeric_weights(importances, floor=0.1):
    values = np.asarray(importances, dtype=float)
    if values.shape != (len(FEATURES),) or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError('Invalid numeric importance vector')
    if not 0 < floor <= 1:
        raise ValueError('Invalid positive uniform floor')
    return floor + (1-floor)*len(values)*values/values.sum()


class GeometryRegressor:
    def __init__(self, route, spec, target, weights):
        if route not in ('KSM', 'KWT') or target not in TARGETS:
            raise ValueError('Invalid geometry route/target')
        self.route, self.spec, self.target = route, dict(spec), target
        self.numeric_weights_ = np.asarray(weights, dtype=float).copy()
        if self.numeric_weights_.shape != (len(FEATURES),) or not np.isfinite(self.numeric_weights_).all() or (self.numeric_weights_ <= 0).any():
            raise ValueError('Positive finite numeric weights required')
        if route == 'KSM' and not np.array_equal(self.numeric_weights_, np.ones(len(FEATURES))):
            raise ValueError('KSM requires uniform geometry')
        self.target_fields_ = (target,)

    def fit(self, frame, labels):
        y = np.asarray(labels, dtype=float)
        if y.shape != (len(frame),) or not len(y) or not np.isfinite(y).all():
            raise ValueError('Invalid single target')
        self.scales_ = np.array([y.mean()])
        if self.scales_[0] <= 0:
            raise ValueError('Positive target mean required')
        self.preprocessor_ = ColumnTransformer([
            ('numeric', StandardScaler(), list(FEATURES)),
            ('spout', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['spout_no'])])
        x = self.preprocessor_.fit_transform(inputs(frame))
        x[:, :len(FEATURES)] *= np.sqrt(self.numeric_weights_)
        self.estimator_ = KernelRidge(**self.spec)
        self.estimator_.fit(x, (y-self.scales_[0])/self.scales_[0])
        self.actual_parameters_ = json.loads(json.dumps(self.estimator_.get_params(deep=False)))
        return self

    def predict(self, frame):
        x = self.preprocessor_.transform(inputs(frame))
        x[:, :len(FEATURES)] *= np.sqrt(self.numeric_weights_)
        pred = self.scales_[0]*(1+np.asarray(self.estimator_.predict(x), dtype=float))
        if pred.shape != (len(frame),) or not np.isfinite(pred).all():
            raise ValueError('Invalid geometry predictions')
        return pred
