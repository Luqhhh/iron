"""Single rate regressor with explicit identity and zero-duration semantics."""
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
from ..artifacts import atomic_write_json, file_sha256
from ..exceptions import ContractError
from ..models.baseline import FROZEN_PARAMETERS
from .component_export import read_json


def schema(X):
    return [dict(name=c,dtype=str(X[c].dtype),categorical=c=='spout_no') for c in X]


class RateModel:
    def fit(self, X, iron, duration, parameters):
        if parameters != FROZEN_PARAMETERS: raise ContractError('rate parameter drift')
        iron,duration = np.asarray(iron,dtype=float),np.asarray(duration,dtype=float)
        if iron.shape!=(len(X),) or duration.shape!=iron.shape or not np.isfinite(iron).all() or not np.isfinite(duration).all() or (iron<0).any() or (duration<0).any():
            raise ContractError('invalid rate labels')
        valid = duration>0
        if not valid.any(): raise ContractError('no positive-duration training rows')
        target = iron[valid]/duration[valid]
        if not np.isfinite(target).all(): raise ContractError('invalid rate target')
        self.schema_ = schema(X)
        self.model = CatBoostRegressor(**parameters)
        self.model.fit(X.loc[valid], target, sample_weight=duration[valid]/duration[valid].mean(),cat_features=['spout_no'])
        self.excluded_zero_duration = int((~valid).sum())
        self.training_rows = int(valid.sum())
        return self

    def predict(self, X):
        if schema(X)!=self.schema_: raise ContractError('rate feature schema changed')
        raw=np.asarray(self.model.predict(X),dtype=float)
        if not np.isfinite(raw).all(): raise ContractError('nonfinite rate prediction')
        return np.maximum(raw,0.)

    def save(self, directory, metadata, history):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        self.model.save_model(directory/'rate.cbm')
        history.to_csv(directory/'history_snapshot.csv',index=False)
        atomic_write_json(directory/'bundle.json',dict(**metadata,feature_schema=self.schema_,
            excluded_zero_duration=self.excluded_zero_duration,rate_training_rows=self.training_rows,
            model_sha256=file_sha256(directory/'rate.cbm'),history_sha256=file_sha256(directory/'history_snapshot.csv')))

    @classmethod
    def load(cls,directory):
        directory=Path(directory);md=read_json(directory/'bundle.json')
        if file_sha256(directory/'rate.cbm')!=md['model_sha256'] or file_sha256(directory/'history_snapshot.csv')!=md['history_sha256']:
            raise ContractError('rate bundle identity mismatch')
        if md['parameters']!=FROZEN_PARAMETERS:raise ContractError('rate bundle parameter drift')
        obj=cls();obj.model=CatBoostRegressor();obj.model.load_model(directory/'rate.cbm')
        obj.schema_=md['feature_schema'];obj.metadata_=md
        return obj
