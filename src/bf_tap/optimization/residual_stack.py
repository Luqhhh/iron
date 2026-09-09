"""Fixed low-capacity residual model and label-free structural meta features."""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from ..artifacts import atomic_write_json, file_sha256
from ..config import load_yaml
from ..exceptions import ContractError
from ..models.baseline import DualTargetBaseline
from .component_export import PRED, read_json
from .rate_model import RateModel, schema
from .inverse_rate_model import InverseRateModel
from .refresh_factorial import Context
from .structural import select_oof

CANDIDATE = 'V4_R2_CAUSAL_RESIDUAL'
RATIOS = ['pred_rate', 'pred_inverse_rate']
CONTEXT = ['spout_no', 'operation__air_volume__6h__mean',
           'operation__total_press_diff__6h__mean']
FEATURES = PRED + RATIOS + ['direction_iron', 'direction_time'] + CONTEXT
CERT = ['inverse_fit_cutoff', 'inverse_train_reference_max',
        'inverse_train_available_max', 'inverse_history_available_max']


def registration():
    return load_yaml('configs/optimization_v0_10/experiment.yaml')


def feature_frame(predictions, context):
    if list(predictions) != ['sample_id', *PRED, *RATIOS] or list(context) != ['sample_id', *CONTEXT]:
        raise ContractError('meta features accept only registered prediction/context columns')
    for frame in (predictions, context):
        if frame.empty or frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
            raise ContractError('meta features require unique complete IDs')
    p = predictions.set_index('sample_id')
    c = context.set_index('sample_id')
    if set(p.index) != set(c.index):
        raise ContractError('meta prediction/context IDs differ')
    c = c.loc[p.index]
    values = p.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ContractError('invalid base/ratio predictions')
    if c.spout_no.isna().any() or np.isinf(c[CONTEXT[1:]].to_numpy(dtype=float)).any():
        raise ContractError('invalid public context')
    x = p.astype(float).copy()
    x['direction_iron'] = x.pred_rate*x.pred_tap_time_len-x.pred_tap_iron
    x['direction_time'] = x.pred_inverse_rate*x.pred_tap_iron-x.pred_tap_time_len
    if not np.isfinite(x.to_numpy()).all():
        raise ContractError('structural feature overflow')
    x['spout_no'] = c.spout_no.astype('string')
    for column in CONTEXT[1:]:
        x[column] = c[column].astype(float)
    return x[FEATURES].reset_index(drop=True)


def select_training(oof, cutoff, minimum=500):
    if set(CERT)-set(oof) or oof[CERT].isna().any().any():
        raise ContractError('missing q OOF certificates')
    if ((oof.inverse_fit_cutoff != oof.fold_cutoff) |
        (oof.inverse_fit_cutoff > oof.reference_time) |
        (oof.inverse_train_reference_max >= oof.inverse_fit_cutoff) |
        (oof.inverse_train_available_max > oof.inverse_fit_cutoff) |
        (oof.inverse_history_available_max > oof.inverse_fit_cutoff)).any():
        raise ContractError('inverse-rate OOF temporal leakage')
    selected = select_oof(oof, cutoff, minimum)
    values = selected[['tap_iron', 'tap_time_len', *PRED, *RATIOS]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ContractError('invalid residual training values')
    return selected.reset_index(drop=True)


class ResidualFitBudget:
    """Allow only newly registered residual CatBoost objects, exactly once each."""
    def __init__(self, limit):
        self.limit = limit
        self.allowed = set()
        self.models = []
        self.attempted = self.completed = self.forbidden = 0

    def allow(self, model):
        if any(model is old for old in self.models):
            raise ContractError('duplicate residual model registration')
        self.models.append(model)
        self.allowed.add(id(model))

    def __enter__(self):
        self.stack = ExitStack()
        original = CatBoostRegressor.fit

        def reject(*args, **kwargs):
            self.forbidden += 1
            raise ContractError('OPT23 forbids fitting existing base/rate/q models')

        def fit(model, *args, **kwargs):
            if id(model) not in self.allowed:
                return reject()
            self.allowed.remove(id(model))
            self.attempted += 1
            if self.attempted > self.limit:
                raise ContractError('residual fit budget exceeded')
            result = original(model, *args, **kwargs)
            self.completed += 1
            return result

        for cls in (DualTargetBaseline, RateModel, InverseRateModel, Context):
            self.stack.enter_context(patch.object(cls, 'fit', reject))
        self.stack.enter_context(patch.object(CatBoostRegressor, 'fit', fit))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def counts(self):
        return dict(attempted_residual_fits=self.attempted,
                    completed_residual_fits=self.completed,
                    forbidden_fit_attempts=self.forbidden)


class ResidualModel:
    def fit(self, x, actual, base, target, budget):
        if list(x) != FEATURES or target not in ('tap_iron', 'tap_time_len'):
            raise ContractError('residual feature/target contract differs')
        y, b = np.asarray(actual, dtype=float), np.asarray(base, dtype=float)
        if y.shape != (len(x),) or b.shape != y.shape or not np.isfinite([y,b]).all():
            raise ContractError('unaligned residual labels')
        self.target = target
        self.schema = schema(x)
        self.parameters = registration()['residual']['parameters']
        self.model = CatBoostRegressor(**self.parameters)
        budget.allow(self.model)
        self.model.fit(x, y-b, cat_features=['spout_no'])
        return self

    def predict_residual(self, x):
        if schema(x) != self.schema:
            raise ContractError('residual feature schema differs')
        residual = np.asarray(self.model.predict(x), dtype=float)
        if not np.isfinite(residual).all():
            raise ContractError('nonfinite residual prediction')
        return residual

    def save(self, root, metadata):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        self.model.save_model(root/'residual.cbm')
        atomic_write_json(root/'bundle.json', dict(metadata=metadata, target=self.target,
            model_kind=CANDIDATE, parameters=self.parameters, feature_schema=self.schema,
            model_sha256=file_sha256(root/'residual.cbm')))

    @classmethod
    def load(cls, root):
        root = Path(root)
        md = read_json(root/'bundle.json')
        if (md['model_kind'] != CANDIDATE or md['parameters'] != registration()['residual']['parameters']
                or file_sha256(root/'residual.cbm') != md['model_sha256']):
            raise ContractError('residual bundle identity differs')
        obj = cls()
        obj.model = CatBoostRegressor()
        obj.model.load_model(root/'residual.cbm')
        obj.target, obj.schema, obj.metadata = md['target'], md['feature_schema'], md
        return obj


def predict(models, parts, context):
    if set(models) != {'tap_iron', 'tap_time_len'} or any(m.target != t for t,m in models.items()):
        raise ContractError('residual target models differ')
    x = feature_frame(parts, context)
    result = parts[['sample_id', *PRED]].copy()
    for target, column in zip(('tap_iron', 'tap_time_len'), PRED):
        result[column] = np.maximum(0., result[column].to_numpy()+models[target].predict_residual(x))
    return result


def acceptance(metrics, summary, reg):
    policy = reg['acceptance']
    current, reference = summary[CANDIDATE], summary['V1']
    delta = {h: {t: current['horizons'][h][t]-reference['horizons'][h][t]
                 for t in ('mean_loss','iron_mean_wmape','time_mean_wmape')}
             for h in ('H1','H2','H3','H4')}
    origins = {u: v['candidates'][CANDIDATE]['overall']['loss']-v['candidates']['V1']['overall']['loss']
               for u,v in metrics.items() if v['horizon'] == 1}
    if len(origins) != 6:
        raise ContractError('six H1 origins required')
    dev = {u: metrics[u]['candidates'][CANDIDATE]['overall']['loss']-metrics[u]['candidates']['V1']['overall']['loss']
           for u in ('DEV_LONG','DEV_SHORT')}
    checks = dict(
        J_improvement=reference['J']-current['J'] >= policy['J_min_improvement'],
        H1_E=delta['H1']['mean_loss'] <= policy['H1_E_max_regression'],
        H1_time=-delta['H1']['time_mean_wmape'] >= policy['H1_time_min_improvement'],
        H1_origins=sum(d <= 0 for d in origins.values()) >= policy['H1_min_nonworse_origins'],
        recent_H1=sum(origins[f'O2024{m:02d}_H1'] < 0 for m in policy['recent_H1_months']) >= policy['recent_H1_min_improved'],
        H234_E=max(delta[h]['mean_loss'] for h in ('H2','H3','H4')) <= policy['H234_E_max_regression'],
        H234_time=max(delta[h]['time_mean_wmape'] for h in ('H2','H3','H4')) <= policy['H234_time_max_regression'],
        iron=max(d['iron_mean_wmape'] for d in delta.values()) <= policy['iron_max_horizon_regression'],
        DEV=max(dev.values()) <= policy['DEV_max_regression'])
    passed = all(checks.values())
    return dict(candidate=CANDIDATE, reference='V1', passed=passed, checks=checks,
                selected=CANDIDATE if passed else None, J=current['J'],
                J_delta_vs_V1=current['J']-reference['J'], horizon_deltas=delta,
                H1_origin_deltas=origins, DEV_deltas=dev,
                status='PASS' if passed else 'FAIL_CLOSE_REGISTERED_V4')
