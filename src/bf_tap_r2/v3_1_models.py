"""V3.1 trial estimators with the task-book inner-early-stop/refit recipe."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .metrics import wmape
from .splits import make_folds
from .v3_local_search import TrialRegressor, expression_frame, fit_target_transform, inverse_target_transform

TREE_FAMILIES = {"catboost", "lightgbm", "xgboost"}
SPECIAL_FAMILIES = {"spline", "poly", "residual"}


def _clone_trial_with_iterations(trial: Mapping[str, Any], best_iteration: int) -> dict:
    out = deepcopy(dict(trial))
    family = out.get("base_family", out["family"])
    params = dict(out["parameters"])
    if family == "catboost":
        params["iterations"] = int(max(1, best_iteration))
        params.pop("use_best_model", None)
        params.pop("od_type", None)
        params.pop("od_wait", None)
    elif family == "lightgbm":
        params["n_estimators"] = int(max(1, best_iteration))
    elif family == "xgboost":
        params["n_estimators"] = int(max(1, best_iteration))
        params.pop("early_stopping_rounds", None)
    else:
        raise ValueError(f"Tree iteration reflection unsupported for {family}")
    out["parameters"] = params
    return out


class V31Regressor:
    """One V3.1 trial; tree families delegate to V3 TrialRegressor."""

    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        self.family = self.trial.get("base_family", self.trial["family"])
        self.feature_set = self.trial["feature_set"]
        self.target_transform = self.trial["target_transform"]
        self.target_state_: dict = {}
        if self.family in {"catboost", "lightgbm", "xgboost", "mlp", "kernel"}:
            self._delegate = TrialRegressor(self.trial)
        elif self.family in SPECIAL_FAMILIES:
            self._delegate = None
        else:
            raise ValueError(f"Unsupported V3.1 family: {self.family}")

    def _special_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        return expression_frame(frame, self.feature_set)

    def fit(self, frame: pd.DataFrame, target: np.ndarray, eval_frame: pd.DataFrame | None = None,
            eval_target: np.ndarray | None = None) -> "V31Regressor":
        if self._delegate is not None:
            self._delegate.fit(frame, target, eval_frame=eval_frame, eval_target=eval_target)
            self.target_state_ = getattr(self._delegate, "target_state_", {})
            self.input_columns_ = getattr(self._delegate, "input_columns_", None)
            return self
        if self.family == "residual":
            return self._fit_residual(frame, target)
        if self.family in {"spline", "poly"}:
            return self._fit_regularized(frame, target)
        raise ValueError(self.family)

    def _fit_residual(self, frame: pd.DataFrame, target: np.ndarray) -> "V31Regressor":
        y = np.asarray(target, dtype=float)
        if y.ndim != 1 or len(y) != len(frame) or not np.isfinite(y).all():
            raise ValueError("Invalid residual labels")
        params = dict(self.trial["parameters"])
        base_params = dict(params["base"])
        residual_params = dict(params["residual"])
        alpha = float(params.get("alpha", 0.5))
        inner_seed = int(params.get("inner_seed", 777))
        self.base_params_ = base_params
        self.residual_params_ = residual_params
        self.alpha_ = alpha
        x_raw = expression_frame(frame, self.feature_set)
        # Inner fold-out residuals only; no outer validation labels are used.
        inner = make_folds(frame, seed=inner_seed, n_splits=5).set_index("sample_id").loc[frame.sample_id]
        inner_fold = inner.fold.to_numpy()
        base_oof = np.full(len(frame), np.nan)
        for fold in range(5):
            train_mask = inner_fold != fold
            valid_mask = inner_fold == fold
            tr = frame.loc[train_mask]
            va = frame.loc[valid_mask]
            m = CatBoostRegressor(**base_params)
            m.fit(expression_frame(tr, self.feature_set), y[train_mask])
            base_oof[valid_mask] = m.predict(expression_frame(va, self.feature_set))
        if not np.isfinite(base_oof).all():
            raise ValueError("Residual inner OOF coverage failed")
        residual = y - base_oof
        x_res = x_raw.copy()
        x_res["_v31_base_oof"] = base_oof
        self.residual_model_ = CatBoostRegressor(**residual_params)
        self.residual_model_.fit(x_res, residual)
        self.base_model_ = CatBoostRegressor(**base_params)
        self.base_model_.fit(x_raw, y)
        self.input_columns_ = tuple(x_raw.columns) + ("_v31_base_oof",)
        self.target_state_ = {}
        return self

    def _fit_regularized(self, frame: pd.DataFrame, target: np.ndarray) -> "V31Regressor":
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, SplineTransformer, StandardScaler

        y = np.asarray(target, dtype=float)
        z, self.target_state_ = fit_target_transform(y, self.target_transform)
        x = expression_frame(frame, self.feature_set)
        params = dict(self.trial["parameters"])
        numeric = [c for c in x.columns if c != "spout_no"]
        if self.family == "spline":
            num = Pipeline([
                ("scale", StandardScaler()),
                ("spline", SplineTransformer(n_knots=int(params.get("n_knots", 5)),
                                              degree=int(params.get("degree", 3)),
                                              include_bias=False)),
            ])
        else:
            num = Pipeline([
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=int(params.get("degree", 2)),
                                             interaction_only=bool(params.get("interaction_only", False)),
                                             include_bias=False)),
            ])
        pre = ColumnTransformer([
            ("num", num, numeric),
            ("spout", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["spout_no"]),
        ], remainder="drop")
        self.pipeline_ = Pipeline([("pre", pre), ("ridge", Ridge(alpha=float(params.get("alpha", 10.0)), solver="svd"))])
        self.pipeline_.fit(x, z)
        self.input_columns_ = tuple(x.columns)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._delegate is not None:
            return self._delegate.predict(frame)
        if self.family == "residual":
            x_raw = expression_frame(frame, self.feature_set)
            x_res = x_raw.copy()
            x_res["_v31_base_oof"] = self.base_model_.predict(x_raw)
            pred = self.base_model_.predict(x_raw) + self.alpha_ * self.residual_model_.predict(x_res)
            if pred.shape != (len(frame),) or not np.isfinite(pred).all():
                raise ValueError("Invalid residual predictions")
            return pred
        x = expression_frame(frame, self.feature_set)
        if tuple(x.columns) != self.input_columns_:
            raise ValueError("V3.1 feature order mismatch")
        pred = inverse_target_transform(np.asarray(self.pipeline_.predict(x), dtype=float),
                                        self.target_transform, self.target_state_)
        if pred.shape != (len(frame),) or not np.isfinite(pred).all():
            raise ValueError("Invalid regularized predictions")
        return pred


def configured_rounds(trial: Mapping[str, Any]) -> int:
    family = trial.get("base_family", trial["family"])
    params = trial["parameters"]
    if family == "catboost":
        return int(params.get("iterations", 1000))
    if family in {"lightgbm", "xgboost"}:
        return int(params.get("n_estimators", 1000))
    return 0


def iteration_fields(family: str, estimator: Any, configured: int) -> dict:
    """Return zero-based best index and actual one-based boosting rounds.

    CatBoost and XGBoost expose a zero-based best index.  LightGBM exposes the
    number of retained iterations directly.  A valid best index of 0 must not
    be confused with "no early stopping".
    """
    configured = max(1, int(configured))
    index: int | None
    selected: int
    actual: int
    if family == "catboost":
        raw = estimator.get_best_iteration() if hasattr(estimator, "get_best_iteration") else None
        try:
            index = int(raw)
        except (TypeError, ValueError):
            index = None
        if index is not None and index < 0:
            index = None
        selected = configured if index is None else index + 1
        actual = int(getattr(estimator, "tree_count_", selected) or selected)
    elif family == "lightgbm":
        raw = getattr(estimator, "best_iteration_", None)
        try:
            retained = int(raw)
        except (TypeError, ValueError):
            retained = 0
        if retained <= 0:
            index = None
            selected = configured
            actual = configured
        else:
            selected = retained
            index = retained - 1
            actual = retained
    elif family == "xgboost":
        raw = getattr(estimator, "best_iteration", None)
        try:
            index = int(raw)
        except (TypeError, ValueError):
            index = None
        if index is not None and index < 0:
            index = None
        selected = configured if index is None else index + 1
        actual = selected
        if hasattr(estimator, "get_booster"):
            try:
                actual = int(estimator.get_booster().num_boosted_rounds())
            except Exception:
                pass
    else:
        index = None
        selected = configured
        actual = configured
    return {
        "best_iteration_index": index,
        "selected_num_boost_round": int(selected),
        "actual_num_boost_round": int(actual),
        "configured_num_boost_round": int(configured),
    }


def _best_iteration(model: V31Regressor) -> int:
    fields = iteration_fields(model.family, model._delegate.estimator_, configured_rounds(model.trial))
    return int(fields["selected_num_boost_round"])


def fit_with_inner_early_stop(trial: Mapping[str, Any], training: pd.DataFrame,
                              target: str, inner_seed: int) -> tuple[V31Regressor, dict]:
    """Fit inner early-stop model, then refit on the full outer training part.

    For spline/poly/residual the method falls back to a single full-training fit
    because their current recipes do not expose a tree-count early-stop knob.
    """
    family = trial.get("base_family", trial["family"])
    inner_assignment = make_folds(training, seed=inner_seed, n_splits=5).set_index("sample_id").loc[training.sample_id]
    inner_valid_mask = inner_assignment.fold.to_numpy() == 0
    inner_train = training.loc[~inner_valid_mask].reset_index(drop=True)
    inner_valid = training.loc[inner_valid_mask].reset_index(drop=True)
    if family in TREE_FAMILIES:
        probe = V31Regressor(trial)
        probe.fit(inner_train, inner_train[target].to_numpy(),
                  eval_frame=inner_valid, eval_target=inner_valid[target].to_numpy())
        fields = iteration_fields(family, probe._delegate.estimator_, configured_rounds(trial))
        selected = int(fields["selected_num_boost_round"])
        refit_trial = _clone_trial_with_iterations(trial, selected)
        model = V31Regressor(refit_trial)
        model.fit(training, training[target].to_numpy())
        return model, {"inner_seed": int(inner_seed),
                       "best_iteration": int(selected),
                       **fields,
                       "refit_full_outer_training": True, "family": family}
    model = V31Regressor(trial)
    model.fit(training, training[target].to_numpy())
    return model, {"inner_seed": None, "best_iteration": None,
                   "refit_full_outer_training": False, "family": family}


def evaluate_v31_outer_folds(train: pd.DataFrame, folds: np.ndarray, trial: Mapping[str, Any],
                             fold_ids: Sequence[int], inner_seed: int) -> dict:
    target = trial["target"]
    predictions = np.full(len(train), np.nan)
    fold_scores: dict[str, float] = {}
    fit_meta = []
    for fold in fold_ids:
        training = train.loc[folds != fold].reset_index(drop=True)
        valid = train.loc[folds == fold].reset_index(drop=True)
        model, meta = fit_with_inner_early_stop(trial, training, target, inner_seed=inner_seed + int(fold))
        pred = model.predict(valid)
        predictions[folds == fold] = pred
        fold_scores[str(int(fold))] = wmape(valid[target], pred)
        fit_meta.append({"fold": int(fold), **meta})
    mask = np.isin(folds, list(fold_ids))
    if not np.isfinite(predictions[mask]).all():
        raise ValueError("V3.1 OOF coverage failed")
    pooled = wmape(train.loc[mask, target], predictions[mask])
    return {
        "trial_id": trial["trial_id"],
        "family": trial["family"],
        "target": target,
        "pooled_wmape": float(pooled),
        "mean_wmape": float(np.mean([fold_scores[str(int(f))] for f in fold_ids])),
        "fold_scores": fold_scores,
        "fit_meta": fit_meta,
        "predictions": predictions,
    }
