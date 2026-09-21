"""Fixed first-round estimators with training-fold-only preprocessing."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from .data import FEATURES


def inputs(frame, categorical_strings=False):
    x = frame.loc[:, [*FEATURES, "spout_no"]].copy()
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite input; no imputation rule authorized")
    if categorical_strings:
        x["spout_no"] = x.spout_no.astype(str)
    return x


class SnapshotRegressor:
    def __init__(self, route: str, config: dict):
        self.route = route
        self.config = config

    def fit(self, frame, target):
        y = np.asarray(target, dtype=float)
        if y.ndim != 1 or len(y) != len(frame) or not np.isfinite(y).all():
            raise ValueError("Invalid training target")
        spec = self.config["models"][self.route]
        execution = self.config["execution"]
        self.target_scale_ = float(np.median(y)) if self.route == "L2" else 1.0
        if self.target_scale_ <= 0:
            raise ValueError("L2 target median must be positive")
        is_tree = self.route in ("C1", "C2")
        x = inputs(frame, is_tree)
        if is_tree:
            self.estimator_ = CatBoostRegressor(
                **{k: spec[k] for k in ("loss_function", "depth", "iterations", "learning_rate", "l2_leaf_reg")},
                random_seed=execution["model_seed"], thread_count=execution["catboost_thread_count"],
                cat_features=["spout_no"], verbose=False, allow_writing_files=False,
                use_best_model=False)
        else:
            numeric = (SplineTransformer(n_knots=spec["numeric"]["n_knots"],
                                         degree=spec["numeric"]["degree"],
                                         knots=execution["spline_knots"],
                                         extrapolation=execution["spline_extrapolation"],
                                         include_bias=execution["spline_include_bias"])
                       if self.route == "S1" else StandardScaler())
            preprocessor = ColumnTransformer([
                ("numeric", numeric, list(FEATURES)),
                ("spout", OneHotEncoder(handle_unknown=execution["one_hot_handle_unknown"],
                                         sparse_output=False), ["spout_no"])], remainder="drop")
            regression = (QuantileRegressor(quantile=spec["quantile"], alpha=spec["alpha"], solver=spec["solver"])
                          if self.route == "L2" else Ridge(alpha=spec["alpha"], solver=execution["ridge_solver"]))
            self.estimator_ = Pipeline([("preprocess", preprocessor), ("regression", regression)])
        self.estimator_.fit(x, y / self.target_scale_)
        return self

    def predict(self, frame):
        prediction = np.asarray(self.estimator_.predict(inputs(frame, self.route in ("C1", "C2"))), dtype=float) * self.target_scale_
        if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
            raise ValueError("Invalid predictions")
        return prediction
