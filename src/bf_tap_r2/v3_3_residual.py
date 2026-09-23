"""V3.3 full-recipe residual wrapper.

Unlike the V3.2 residual sampler, this wrapper preserves the complete base
trial dictionary, including feature set, target transform, loss, seeds and
training protocol.  Base predictions are restored to original units before
residual construction.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .splits import make_folds
from .v3_1_models import V31Regressor
from .v3_local_search import expression_frame


class FullRecipeResidualRegressor:
    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        if self.trial.get("family") not in {"full_residual", "v33_residual"}:
            raise ValueError("Not a V3.3 full-recipe residual trial")
        params = self.trial.get("parameters", {})
        if "base_trial" not in params:
            raise ValueError("V3.3 residual requires a full base_trial")

    def _residual_features(self, frame: pd.DataFrame, base_prediction: np.ndarray) -> pd.DataFrame:
        x = expression_frame(frame, "raw").copy()
        x["_v33_base_prediction"] = np.asarray(base_prediction, dtype=float)
        if not np.isfinite(x.to_numpy(dtype=float)).all():
            raise ValueError("V3.3 residual features must be finite")
        return x

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "FullRecipeResidualRegressor":
        y = np.asarray(target, dtype=float)
        if y.ndim != 1 or len(y) != len(frame) or not np.isfinite(y).all():
            raise ValueError("Invalid V3.3 residual labels")
        params = dict(self.trial.get("parameters", self.trial))
        base_trial = params["base_trial"]
        residual_spec = dict(params["residual"])
        self.alpha_ = float(params.get("alpha", 0.25))
        self.kind_ = str(residual_spec["kind"])
        inner_seed = int(params.get("inner_seed", 777))
        n_splits = int(params.get("inner_splits", 5))
        inner = make_folds(frame, seed=inner_seed, n_splits=n_splits).set_index("sample_id").loc[frame.sample_id]
        fold_vector = inner.fold.to_numpy()
        base_oof = np.full(len(frame), np.nan)
        for fold in range(n_splits):
            train_mask = fold_vector != fold
            valid_mask = fold_vector == fold
            base = V31Regressor(base_trial)
            base.fit(frame.loc[train_mask].reset_index(drop=True), y[train_mask])
            base_oof[valid_mask] = base.predict(frame.loc[valid_mask].reset_index(drop=True))
        if not np.isfinite(base_oof).all():
            raise ValueError("V3.3 base OOF coverage failed")
        residual = y - base_oof
        x_res = self._residual_features(frame, base_oof)
        if self.kind_ == "catboost":
            self.residual_model_ = CatBoostRegressor(**dict(residual_spec["params"]))
            self.residual_model_.fit(x_res.assign(spout_no=x_res.spout_no.astype(str)), residual)
            self.residual_cat_features_ = ["spout_no"]
        elif self.kind_ == "ridge":
            from sklearn.compose import ColumnTransformer
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            numeric = [c for c in x_res.columns if c != "spout_no"]
            pre = ColumnTransformer([
                ("num", StandardScaler(), numeric),
                ("spout", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["spout_no"]),
            ], remainder="drop")
            self.residual_model_ = Pipeline([("pre", pre), ("ridge", Ridge(**dict(residual_spec["params"])))])
            self.residual_model_.fit(x_res, residual)
        else:
            raise ValueError(f"Unsupported V3.3 residual kind: {self.kind_}")
        self.base_model_ = V31Regressor(base_trial)
        self.base_model_.fit(frame, y)
        self.input_columns_ = tuple(x_res.columns)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        base_pred = np.asarray(self.base_model_.predict(frame), dtype=float)
        x_res = self._residual_features(frame, base_pred)
        if tuple(x_res.columns) != self.input_columns_:
            raise ValueError("V3.3 residual feature order mismatch")
        if self.kind_ == "catboost":
            x_res = x_res.assign(spout_no=x_res.spout_no.astype(str))
        correction = np.asarray(self.residual_model_.predict(x_res), dtype=float)
        pred = base_pred + self.alpha_ * correction
        if pred.shape != (len(frame),) or not np.isfinite(pred).all():
            raise ValueError("Invalid V3.3 residual predictions")
        return pred
