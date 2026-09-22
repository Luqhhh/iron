"""Explicit, label-free four-feature augmentation for V2.12 only."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor
from .data import FEATURES, TARGETS
from .v2_residual_structure import diagnostic_inputs

DERIVED_FEATURES=('oxygen_per_air_volume','pressure_per_air_volume','thermal_difference','upper_pressure_fraction')
ENGINEERED_FIELDS=(*FEATURES,*DERIVED_FEATURES,'spout_no')


def engineered_inputs(frame):
    x=diagnostic_inputs(frame)
    x['spout_no']=frame.spout_no.astype(str)
    if tuple(x.columns)!=ENGINEERED_FIELDS:
        raise ValueError('Engineered feature order mismatch')
    return x


class EngineeredRegressor:
    def __init__(self,parameters,target):
        if target not in TARGETS or parameters['loss_function']!='RMSE':
            raise ValueError('Invalid engineered target/loss')
        self.parameters=dict(parameters)
        self.target_fields_=(target,)

    def fit(self,frame,target):
        y=np.asarray(target,dtype=float)
        if y.shape!=(len(frame),) or not len(y) or not np.isfinite(y).all() or y.mean()<=0:
            raise ValueError('Invalid engineered labels')
        self.target_scales_=np.array([y.mean()])
        self.input_fields_=ENGINEERED_FIELDS
        self.estimator_=CatBoostRegressor(**self.parameters)
        self.estimator_.fit(engineered_inputs(frame),(y-y.mean())/y.mean())
        self.actual_parameters_=self.estimator_.get_all_params()
        return self

    def predict(self,frame):
        if tuple(self.input_fields_)!=ENGINEERED_FIELDS:
            raise ValueError('Stored engineered feature order mismatch')
        pred=self.target_scales_[0]*(1+np.asarray(self.estimator_.predict(engineered_inputs(frame)),dtype=float))
        if pred.shape!=(len(frame),) or not np.isfinite(pred).all():
            raise ValueError('Invalid engineered predictions')
        return pred
