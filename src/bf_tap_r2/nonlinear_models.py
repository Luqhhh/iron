"""Fold-local smooth nonlinear candidates, without test preprocessing or labels."""
from __future__ import annotations

import json
import warnings
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.svm import SVR

from .data import FEATURES,TARGETS
from .models import inputs


class NonlinearRegressor:
    def __init__(self,route,spec,target=None):
        if route not in ('JM1','KR1','KS1') or (route=='KS1' and target not in TARGETS):
            raise ValueError('Invalid nonlinear route/target')
        self.route,self.spec,self.target=route,dict(spec),target
        self.target_fields_=TARGETS if route!='KS1' else (target,)

    def fit(self,frame,labels):
        if len(self.target_fields_)==2:
            if not hasattr(labels,'columns') or tuple(labels.columns)!=TARGETS:
                raise ValueError('Joint target order mismatch')
            y=labels.to_numpy(dtype=float)
        else:
            y=np.asarray(labels,dtype=float)
            if y.shape!=(len(frame),):
                raise ValueError('Independent target shape mismatch')
            y=y[:,None]
        if y.shape!=(len(frame),len(self.target_fields_)) or not len(y) or not np.isfinite(y).all():
            raise ValueError('Invalid nonlinear labels')
        self.scales_=y.mean(axis=0)
        if (self.scales_<=0).any():
            raise ValueError('Positive target means required')
        self.preprocessor_=ColumnTransformer([
            ('numeric',StandardScaler(),list(FEATURES)),
            ('spout',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['spout_no'])])
        x=self.preprocessor_.fit_transform(inputs(frame))
        z=(y-self.scales_)/self.scales_
        parameters=dict(self.spec)
        if self.route=='JM1':
            parameters['hidden_layer_sizes']=tuple(parameters['hidden_layer_sizes'])
            self.estimator_=MLPRegressor(**parameters)
        elif self.route=='KR1':
            self.estimator_=KernelRidge(**parameters)
        else:
            self.estimator_=SVR(**parameters)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.estimator_.fit(x,z if len(self.target_fields_)==2 else z[:,0])
        self.fit_report_={'warnings':[str(w.message) for w in caught],
            'convergence_warning':any(issubclass(w.category,ConvergenceWarning) for w in caught),
            'iterations':int(getattr(self.estimator_,'n_iter_',0)),
            'fit_status':int(getattr(self.estimator_,'fit_status_',0))}
        if self.fit_report_['convergence_warning'] or self.fit_report_['fit_status']:
            raise RuntimeError('Nonlinear convergence gate failed: '+str(self.fit_report_))
        self.actual_parameters_=json.loads(json.dumps(self.estimator_.get_params(deep=False)))
        return self

    def predict(self,frame):
        raw=np.asarray(self.estimator_.predict(self.preprocessor_.transform(inputs(frame))),dtype=float)
        expected=(len(frame),len(self.target_fields_))
        if len(self.target_fields_)==1:
            raw=raw.reshape(-1,1)
        if raw.shape!=expected:
            raise ValueError('Nonlinear prediction target shape mismatch')
        pred=self.scales_*(1+raw)
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite prediction')
        return pred if len(self.target_fields_)==2 else pred[:,0]
