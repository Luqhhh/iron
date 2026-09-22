"""Training-fold mean-centered relative targets for independent and joint trees."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor

from .data import FEATURES, TARGETS
from .models import inputs


class NormalizedSnapshotRegressor:
    def __init__(self, parameters, target):
        if target not in TARGETS or parameters["loss_function"] not in ("RMSE", "Huber:delta=0.05"):
            raise ValueError("Invalid independent normalized specification")
        self.parameters = dict(parameters)
        self.target_fields_ = (target,)

    def _fit(self, frame, labels):
        y = np.asarray(labels, dtype=float)
        if y.shape != (len(frame), len(self.target_fields_)) or not len(frame) or not np.isfinite(y).all():
            raise ValueError("Invalid normalized training labels")
        self.target_scales_ = y.mean(axis=0)
        if not np.isfinite(self.target_scales_).all() or (self.target_scales_ <= 0).any():
            raise ValueError("Training target means must be positive")
        self.input_fields_ = (*FEATURES, "spout_no")
        z = (y - self.target_scales_) / self.target_scales_
        self.estimator_ = CatBoostRegressor(**self.parameters)
        self.estimator_.fit(inputs(frame, True), z if len(self.target_fields_) == 2 else z[:, 0])
        self.actual_parameters_ = self.estimator_.get_all_params()
        return self

    def fit(self, frame, target):
        y = np.asarray(target, dtype=float)
        if y.ndim != 1:
            raise ValueError("Independent target must be one-dimensional")
        return self._fit(frame, y[:, None])

    def _predict(self, frame):
        if tuple(self.input_fields_) != (*FEATURES, "spout_no"):
            raise ValueError("Stored feature order mismatch")
        raw = np.asarray(self.estimator_.predict(inputs(frame, True)), dtype=float)
        if len(self.target_fields_) == 1:
            raw = raw.reshape(-1, 1)
        expected = (len(frame), len(self.target_fields_))
        if raw.shape != expected:
            raise ValueError("Prediction target shape mismatch")
        values = self.target_scales_ * (1 + raw)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite restored prediction")
        return values

    def predict(self, frame):
        return self._predict(frame)[:, 0]


class JointSnapshotRegressor(NormalizedSnapshotRegressor):
    def __init__(self, parameters):
        if parameters["loss_function"] != "MultiRMSE":
            raise ValueError("Joint model requires MultiRMSE")
        self.parameters = dict(parameters)
        self.target_fields_ = TARGETS

    def fit(self, frame, targets):
        if not hasattr(targets, "columns") or tuple(targets.columns) != TARGETS:
            raise ValueError("Joint label order must be tap_iron, tap_time_len")
        return self._fit(frame, targets.to_numpy(dtype=float))

    def predict(self, frame):
        if tuple(self.target_fields_) != TARGETS:
            raise ValueError("Stored target order mismatch")
        return self._predict(frame)
