"""V4.1 E0/E1/E3 residual components with nested coefficient learning.

REAL V36 ADAPTER NOT INCLUDED: pass a factory that rebuilds the complete frozen
V36 baseline on ONLY the training rows received by fit. Never substitute A,
reuse globally trained models, or slice global OOF for an inner fit.

Backend protocol:
  model = factory()
  model.fit(X_dataframe, Y_dataframe, groups=group_array)
  model.predict_state(X_query) -> DataFrame with exactly STATE_COLUMNS,
                                  matching X_query.index and spout_no.
The gap columns are fixed L1-minus-EBM predictions in original target units:
iron uses v36-s1-D-0029 and time uses v36-s1-O-0057 as the EBM endpoints. No true target may enter predict_state.

X.index must contain unique immutable sample IDs. Y has the exact two TARGETS.
The outer evaluator must keep its validation labels entirely outside this class.
This core does not select among variants or write a competition submission.
"""
from __future__ import annotations
from collections.abc import Callable, Mapping
import hashlib
import weakref
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler, SplineTransformer
from .v4_1_r2_increment import exact_l1_alpha

TARGETS = ('tap_iron', 'tap_time_len')
PRED_COLUMNS = ('pred_iron', 'pred_time')
NUMERIC_STATES = ('pred_iron', 'pred_time', 'gap_iron', 'gap_time')
STATE_COLUMNS = (*NUMERIC_STATES, 'spout_no')


def _state(s: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(s, pd.DataFrame) or set(s.columns) != set(STATE_COLUMNS):
        raise ValueError(f'State must have exactly {STATE_COLUMNS}; no labels or extra features')
    if not s.index.is_unique or len(s) == 0 or not np.isfinite(s.loc[:, STATE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError('Invalid state values or sample IDs')
    if (s.loc[:, PRED_COLUMNS].to_numpy() < 0).any():
        raise ValueError('Baseline predictions must be nonnegative')
    return s.loc[:, STATE_COLUMNS]


def _residual(s: pd.DataFrame, r: Any) -> np.ndarray:
    out = np.asarray(r, dtype=float)
    if out.shape != (len(s),) or not np.isfinite(out).all():
        raise ValueError('Invalid residual vector')
    return out


class MedianResidual:
    """E0 constant control. Residuals are allowed to be negative."""
    def fit(self, state: pd.DataFrame, residual: Any) -> 'MedianResidual':
        s = _state(state)
        self.value_ = float(np.median(_residual(s, residual)))
        return self

    def predict(self, state: pd.DataFrame) -> np.ndarray:
        s = _state(state)
        return np.full(len(s), self.value_)


class SplineMedianResidual:
    """E1 fixed low-capacity conditional median in prediction-state space.

    This r2 implementation is explicitly state-only, not a claimed reproduction
    of an unexecuted raw-feature E1. Knots, scaling, target scale and vocabulary
    are fitted solely on the rows passed to fit.
    """
    def __init__(self, n_knots: int = 4, alpha: float = .01):
        self.n_knots, self.alpha = int(n_knots), float(alpha)
        if self.n_knots < 2 or self.alpha < 0:
            raise ValueError('Invalid spline parameters')

    def fit(self, state: pd.DataFrame, residual: Any) -> 'SplineMedianResidual':
        s = _state(state)
        r = _residual(s, residual)
        self.active_numeric_ = [c for c in NUMERIC_STATES if s[c].nunique() > 1]
        self.scaler_ = RobustScaler().fit(s[self.active_numeric_]) if self.active_numeric_ else None
        z = self.scaler_.transform(s[self.active_numeric_]) if self.scaler_ is not None else None
        self.spline_ = (SplineTransformer(n_knots=self.n_knots, degree=2,
                                        knots='quantile', include_bias=False,
                                        extrapolation='constant').fit(z)
                        if z is not None else None)
        self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(s[['spout_no']])
        self.center_ = float(np.median(r))
        self.scale_ = max(float(np.median(np.abs(r - self.center_))), 1e-8)
        design = self._design(s)
        self.model_ = QuantileRegressor(quantile=.5, alpha=self.alpha, solver='highs').fit(
            design, (r - self.center_) / self.scale_)
        return self

    def _design(self, s: pd.DataFrame) -> np.ndarray:
        encoded = self.encoder_.transform(s[['spout_no']])
        if self.spline_ is None:
            return encoded
        z = self.scaler_.transform(s[self.active_numeric_])
        return np.column_stack((self.spline_.transform(z), encoded))

    def predict(self, state: pd.DataFrame) -> np.ndarray:
        s = _state(state)
        return self.center_ + self.scale_ * self.model_.predict(self._design(s))


class LocalMedianResidual:
    """E3 local residual median on frozen prediction states, not raw features.

    k=64 is an initial fixed recipe, not a fitted knob. All distance-tied boundary
    points are included, avoiding training-order sensitivity at the k boundary.
    A robust scaler and spout one-hot encoder are fitted only on training states.
    """
    def __init__(self, n_neighbors: int = 64):
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors < 1:
            raise ValueError('n_neighbors must be positive')

    def fit(self, state: pd.DataFrame, residual: Any) -> 'LocalMedianResidual':
        s = _state(state)
        self.r_ = _residual(s, residual).copy()
        self.scaler_ = RobustScaler().fit(s.loc[:, NUMERIC_STATES])
        self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(s[['spout_no']])
        self.z_ = self._design(s)
        return self

    def _design(self, s: pd.DataFrame) -> np.ndarray:
        return np.column_stack((self.scaler_.transform(s.loc[:, NUMERIC_STATES]),
                                self.encoder_.transform(s[['spout_no']])))

    def predict(self, state: pd.DataFrame) -> np.ndarray:
        s = _state(state)
        out = []
        for q in self._design(s):
            distances = np.linalg.norm(self.z_ - q, axis=1)
            k = min(self.n_neighbors, len(distances))
            radius = float(np.partition(distances, k - 1)[k - 1])
            # Zero-distance ties contain all identical states, never arbitrary k.
            mask = (distances == 0) if np.any(distances == 0) else (distances <= radius)
            values = self.r_[mask]
            if np.any(distances == 0):
                weights = np.ones(len(values))
            else:
                weights = np.exp(-np.square(distances[mask] / max(radius, 1e-12)))
            order = np.argsort(values, kind='stable')
            cumulative = np.cumsum(weights[order])
            j = int(np.searchsorted(cumulative, .5 * cumulative[-1], side='left'))
            out.append(float(values[order[min(j, len(order) - 1)]]))
        return np.asarray(out)


def _ids_hash(index: pd.Index) -> str:
    h = hashlib.sha256()
    for x in index:
        h.update(str(x).encode('utf-8')); h.update(b'\0')
    return h.hexdigest()


class NestedResidualBank:
    """Fit several correction recipes with shared, fully nested baseline fits.

    For coefficient-learning folds U/W inside T:
      base inner-OOF on U -> residual model fit on U;
      base refit on U -> predict W -> correction prediction on W.
    All W predictions combined -> exact L1 alpha per target and recipe.
    Reuse U-refit/W predictions as the ordinary baseline OOF on T to fit the
    final correctors; then refit base on all T. No global OOF files accepted.

    With 3x3 folds, the backend is fitted 3*(3+1)+1=13 times, shared across
    all correction recipes and both targets. Backend-internal fits are separate.
    Choice among recipes still belongs to the outer evaluation, not this class.
    """
    def __init__(self, baseline_factory: Callable[[], Any],
                 correction_factories: Mapping[str, Callable[[], Any]] | None = None,
                 n_splits: int = 3, seed: int = 314159,
                 split_function: Callable | None = None):
        if n_splits < 2 or not callable(baseline_factory):
            raise ValueError('A real baseline factory and at least two splits are required')
        self.baseline_factory = baseline_factory
        self.correction_factories = dict(correction_factories or {
            'E0_median': MedianResidual, 'E1_state_spline_median': SplineMedianResidual,
            'E3_state_local_median': LocalMedianResidual,
        })
        if not self.correction_factories or any(not callable(f) for f in self.correction_factories.values()):
            raise ValueError('Invalid correction factories')
        self.n_splits, self.seed, self.split_function = int(n_splits), int(seed), split_function

    def _splits(self, x, y, g, seed):
        if self.split_function is None:
            # For production, supply the project's group+spout splitter here.
            raw = list(GroupKFold(n_splits=self.n_splits, shuffle=True,
                                  random_state=seed).split(x, y, groups=g))
        else:
            raw = list(self.split_function(x, y, g, seed))
        if len(raw) != self.n_splits:
            raise ValueError('Splitter returned an unexpected number of folds')
        count = np.zeros(len(x), dtype=int)
        for tr, va in raw:
            tr, va = np.asarray(tr, dtype=int), np.asarray(va, dtype=int)
            if (not len(tr) or not len(va) or len(set(tr)) != len(tr) or len(set(va)) != len(va)
                or min(tr.min(), va.min()) < 0 or max(tr.max(), va.max()) >= len(x)
                or set(tr) & set(va) or set(tr) | set(va) != set(range(len(x)))
                or set(g[tr]) & set(g[va])):
                raise ValueError('Invalid fold or a group crosses a training/evaluation boundary')
            count[va] += 1
        if not np.all(count == 1):
            raise ValueError('Evaluation rows must be covered exactly once')
        return raw

    def _fit_base(self, x, y, g, stage):
        model = self.baseline_factory()
        if any(model is prior() for prior in self._backend_instances):
            raise ValueError('baseline_factory reused an estimator instance')
        self._backend_instances.append(weakref.ref(model))
        model.fit(x.copy(deep=True), y.copy(deep=True), groups=np.asarray(g).copy())
        self.audit_['base_calls'].append({'stage': stage, 'n_train': len(x),
                                          'train_ids_sha256': _ids_hash(x.index)})
        return model

    def _predict_base(self, model, x):
        s = _state(model.predict_state(x.copy(deep=True)))
        if not s.index.equals(x.index) or not np.array_equal(s.spout_no.to_numpy(), x.spout_no.to_numpy()):
            raise ValueError('Backend changed prediction row order/IDs or spout values')
        return s

    def _base_oof(self, x, y, g, seed, stage):
        out = pd.DataFrame(np.nan, index=x.index, columns=STATE_COLUMNS)
        for fold, (tr, va) in enumerate(self._splits(x, y, g, seed)):
            model = self._fit_base(x.iloc[tr], y.iloc[tr], g[tr], f'{stage}/inner-{fold}')
            out.iloc[va] = self._predict_base(model, x.iloc[va]).to_numpy()
        return _state(out)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, *, groups: Any) -> 'NestedResidualBank':
        if not isinstance(x, pd.DataFrame) or not x.index.is_unique or 'spout_no' not in x:
            raise ValueError('X needs unique immutable sample IDs as index and spout_no column')
        if any(t in x.columns for t in TARGETS):
            raise ValueError('True target column in X')
        if not isinstance(y, pd.DataFrame) or tuple(y.columns) != TARGETS or not y.index.equals(x.index):
            raise ValueError('Target schema/index mismatch')
        if not np.isfinite(y.to_numpy(dtype=float)).all() or (y.to_numpy() < 0).any():
            raise ValueError('Invalid targets')
        g = np.asarray(groups)
        if g.shape != (len(x),) or pd.isna(g).any():
            raise ValueError('Invalid groups')
        self.train_ids_ = set(x.index)
        self.audit_ = {'base_calls': [], 'corrector_fits': 0, 'scalar_optimizations': 0,
                       'uses_global_oof': False, 'platform_authority': False}
        self._backend_instances = []
        b_oof = pd.DataFrame(np.nan, index=x.index, columns=STATE_COLUMNS)
        corrections = {name: np.full((len(x), 2), np.nan) for name in self.correction_factories}
        for fold, (tr, va) in enumerate(self._splits(x, y, g, self.seed)):
            u, yu, gu = x.iloc[tr], y.iloc[tr], g[tr]
            su = self._base_oof(u, yu, gu, self.seed + fold + 1, f'calibration-{fold}')
            base = self._fit_base(u, yu, gu, f'calibration-{fold}/refit')
            sv = self._predict_base(base, x.iloc[va])
            b_oof.iloc[va] = sv.to_numpy()
            for name, factory in self.correction_factories.items():
                for t, target in enumerate(TARGETS):
                    r = yu[target].to_numpy() - su[PRED_COLUMNS[t]].to_numpy()
                    model = factory().fit(su, r)
                    pred = np.asarray(model.predict(sv), dtype=float)
                    if pred.shape != (len(va),) or not np.isfinite(pred).all():
                        raise ValueError('Invalid cross-fitted correction')
                    corrections[name][va, t] = pred
                    self.audit_['corrector_fits'] += 1
        b_oof = _state(b_oof)
        self.alphas_, self.final_correctors_ = {}, {}
        for name, factory in self.correction_factories.items():
            self.alphas_[name], self.final_correctors_[name] = {}, {}
            for t, target in enumerate(TARGETS):
                b = b_oof[PRED_COLUMNS[t]].to_numpy()
                c = b + corrections[name][:, t]
                self.alphas_[name][target] = exact_l1_alpha(y[target].to_numpy(), b, c)
                self.audit_['scalar_optimizations'] += 1
                self.final_correctors_[name][target] = factory().fit(b_oof, y[target].to_numpy() - b)
                self.audit_['corrector_fits'] += 1
        self.baseline_ = self._fit_base(x, y, g, 'all-training-refit')
        self.audit_['base_fit_calls'] = len(self.audit_['base_calls'])
        # Keep only the final trained base; diagnostic audit contains hashes, not labels.
        del self._backend_instances
        return self

    def predict_variants(self, x: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if set(x.index) & self.train_ids_:
            raise ValueError('Evaluation IDs overlap this model\'s training IDs')
        s = self._predict_base(self.baseline_, x)
        b = s.loc[:, PRED_COLUMNS].to_numpy(dtype=float)
        out = {'baseline': pd.DataFrame(b, columns=TARGETS, index=x.index)}
        for name, models in self.final_correctors_.items():
            pred = b.copy()
            for t, target in enumerate(TARGETS):
                alpha = self.alphas_[name][target]
                if alpha != 0:
                    pred[:, t] += alpha * np.asarray(models[target].predict(s), dtype=float)
            if not np.isfinite(pred).all():
                raise ValueError('Nonfinite residual prediction')
            # No clipping: signed residual learning is not replaced by an implicit rule.
            out[name] = pd.DataFrame(pred, columns=TARGETS, index=x.index)
        return out
