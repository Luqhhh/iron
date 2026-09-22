"""Smooth fold-local polynomial baseline with CatBoost residual correction."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from .data import FEATURES, TARGETS
from .models import inputs
from .normalized_models import NormalizedSnapshotRegressor


class ResidualPolynomialRegressor:
    def __init__(self, parameters, quadratic, target):
        if target not in TARGETS or parameters['loss_function'] != 'RMSE':
            raise ValueError('Invalid residual specification')
        self.parameters, self.quadratic, self.target = dict(parameters), dict(quadratic), target

    def fit(self, frame, target):
        y = np.asarray(target, dtype=float)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or not len(y) or y.mean() <= 0:
            raise ValueError('Invalid residual labels')
        self.target_scale_ = float(y.mean())
        z = (y-self.target_scale_)/self.target_scale_
        polynomial = Pipeline([('input_scale', StandardScaler()),
                               ('poly', PolynomialFeatures(degree=self.quadratic['degree'], include_bias=False)),
                               ('output_scale', StandardScaler())])
        transform = ColumnTransformer([('numeric', polynomial, list(FEATURES)),
                     ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['spout_no'])])
        self.smooth_ = Pipeline([('transform', transform),
                      ('ridge', Ridge(alpha=self.quadratic['ridge_alpha'], solver=self.quadratic['solver']))])
        x = inputs(frame)
        self.smooth_.fit(x, z)
        residual = z - self.smooth_.predict(x)
        self.estimator_ = CatBoostRegressor(**self.parameters)
        self.estimator_.fit(inputs(frame, True), residual)
        self.actual_parameters_ = self.estimator_.get_all_params()
        self.input_fields_ = (*FEATURES, 'spout_no')
        return self

    def predict(self, frame):
        if self.input_fields_ != (*FEATURES, 'spout_no'):
            raise ValueError('Feature order mismatch')
        z = self.smooth_.predict(inputs(frame)) + self.estimator_.predict(inputs(frame, True))
        result = self.target_scale_*(1+z)
        if result.shape != (len(frame),) or not np.isfinite(result).all():
            raise ValueError('Invalid residual prediction')
        return result


def make_model(route, target, spec):
    parameters = {**spec['common'], **spec['routes'][route]}
    if route == 'QRC':
        return ResidualPolynomialRegressor(parameters, spec['quadratic'], target)
    return NormalizedSnapshotRegressor(parameters, target)
