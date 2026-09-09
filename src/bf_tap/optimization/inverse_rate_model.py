"""Bounded new q=time/iron fits; all existing model fit entrypoints are forbidden."""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np
from catboost import CatBoostRegressor
from ..artifacts import atomic_write_json,file_sha256
from ..exceptions import ContractError
from ..models.baseline import DualTargetBaseline,FROZEN_PARAMETERS
from .component_export import read_json
from .rate_model import RateModel,schema
from .refresh_factorial import Context


class InverseFitBudget:
    def __init__(self,limit):
        self.limit=limit;self.allowed=set();self.seen=set();self.models=[]
        self.attempted=0;self.completed=0;self.forbidden=0

    def allow(self,model):
        if id(model) in self.seen:raise ContractError('inverse model registered twice')
        self.allowed.add(id(model));self.seen.add(id(model));self.models.append(model)

    def __enter__(self):
        original=CatBoostRegressor.fit;self.stack=ExitStack()
        def reject(*args,**kwargs):
            self.forbidden+=1;raise ContractError('OPT22 forbids fitting existing R2/rate models')
        def fit(model,*args,**kwargs):
            if id(model) not in self.allowed:
                self.forbidden+=1;raise ContractError('unregistered CatBoost fit')
            self.allowed.remove(id(model));self.attempted+=1
            if self.attempted>self.limit:raise ContractError('inverse-rate model fit budget exceeded')
            result=original(model,*args,**kwargs);self.completed+=1;return result
        for cls in (DualTargetBaseline,RateModel,Context):self.stack.enter_context(patch.object(cls,'fit',reject))
        self.stack.enter_context(patch.object(CatBoostRegressor,'fit',fit));return self

    def __exit__(self,*args):return self.stack.__exit__(*args)

    def counts(self):return dict(attempted_inverse_fits=self.attempted,completed_inverse_fits=self.completed,forbidden_fit_attempts=self.forbidden)


class InverseRateModel:
    def fit(self,X,iron,time,parameters,budget):
        if parameters!=FROZEN_PARAMETERS:raise ContractError('inverse-rate parameter drift')
        iron,time=np.asarray(iron,dtype=float),np.asarray(time,dtype=float)
        if iron.shape!=(len(X),) or time.shape!=iron.shape or not np.isfinite(iron).all() or not np.isfinite(time).all() or (iron<0).any() or (time<0).any():raise ContractError('invalid inverse-rate targets')
        positive=iron>0
        if not positive.any():raise ContractError('no positive-iron rows for q training')
        target=time[positive]/iron[positive]
        if not np.isfinite(target).all():raise ContractError('nonfinite inverse-rate target')
        self.schema_=schema(X);self.model=CatBoostRegressor(**parameters);budget.allow(self.model)
        self.model.fit(X.loc[positive],target,sample_weight=iron[positive]/iron[positive].mean(),cat_features=['spout_no'])
        self.excluded_zero_iron=int((~positive).sum());self.training_rows=int(positive.sum());return self

    def predict(self,X):
        if schema(X)!=self.schema_:raise ContractError('inverse-rate schema differs')
        q=np.asarray(self.model.predict(X),dtype=float)
        if not np.isfinite(q).all():raise ContractError('nonfinite inverse-rate prediction')
        return np.maximum(q,0.)

    def save(self,root,metadata,history):
        root=Path(root);root.mkdir(parents=True,exist_ok=False)
        self.model.save_model(root/'inverse_rate.cbm');history.to_csv(root/'history_snapshot.csv',index=False)
        atomic_write_json(root/'bundle.json',dict(**metadata,model_kind='inverse_rate_time_per_iron',feature_schema=self.schema_,
            excluded_zero_iron=self.excluded_zero_iron,inverse_training_rows=self.training_rows,
            model_sha256=file_sha256(root/'inverse_rate.cbm'),history_sha256=file_sha256(root/'history_snapshot.csv')))

    @classmethod
    def load(cls,root):
        root=Path(root);md=read_json(root/'bundle.json')
        if md['model_kind']!='inverse_rate_time_per_iron' or md['parameters']!=FROZEN_PARAMETERS:raise ContractError('inverse-rate bundle contract differs')
        if file_sha256(root/'inverse_rate.cbm')!=md['model_sha256'] or file_sha256(root/'history_snapshot.csv')!=md['history_sha256']:raise ContractError('inverse-rate bundle identity differs')
        obj=cls();obj.model=CatBoostRegressor();obj.model.load_model(root/'inverse_rate.cbm');obj.schema_=md['feature_schema'];obj.metadata_=md;return obj
