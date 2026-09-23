"""V3.4 estimators: group-safe EBM, EBM residual, and shrink spout experts.

The classes in this module deliberately accept the private trial dictionaries
produced by :mod:`bf_tap_r2.v3_4_sampler`.  They never read outer validation
labels during fit.  Base residual OOF predictions are generated with a
duplicate-group-safe inner fold split on the current training part only.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .data import FEATURES
from .metrics import wmape
from .v3_4_bags import (
    BAG_PROTOCOL,
    assert_group_isolated,
    build_group_safe_bags,
    group_safe_inner_folds,
)
from .v3_local_search import expression_frame, fit_target_transform, inverse_target_transform
from .v3_1_models import V31Regressor

EBM_KINDS = {"ebm", "ebm_boundary", "ebm_base"}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite_1d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"Invalid {name}")
    return array


def _ebm_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in (*FEATURES, "spout_no") if name not in frame.columns]
    if missing:
        raise ValueError(f"EBM frame is missing columns: {missing}")
    x = frame.loc[:, [*FEATURES, "spout_no"]].copy()
    if x.isna().any().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("EBM feature frame must be finite")
    return x


def _validate_custom_bags(bags: np.ndarray, n_samples: int, n_outer_bags: int) -> np.ndarray:
    matrix = np.asarray(bags)
    if matrix.ndim != 2 or matrix.shape != (int(n_outer_bags), int(n_samples)):
        raise ValueError(f"bags must have shape ({n_outer_bags}, {n_samples})")
    if not np.isin(matrix, (-1, 1)).all():
        raise ValueError("bags may only contain -1 and +1")
    for row in range(matrix.shape[0]):
        if not (matrix[row] == 1).any() or not (matrix[row] == -1).any():
            raise ValueError("Every bag needs nonempty training and validation parts")
    return matrix.astype(np.int8, copy=False)


class V34EBMRegressor:
    """Complete EBM recipe with explicit group-safe-bags-v1."""

    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        kind = str(self.trial.get("kind", "ebm"))
        if kind not in EBM_KINDS:
            raise ValueError(f"Not a V3.4 EBM trial: {kind}")
        self.kind = kind
        self.target = self.trial["target"]
        self.target_transform = self.trial["target_transform"]
        self.parameters = dict(self.trial["parameters"])
        self.protocol = dict(self.trial.get("protocol", {}))
        self.inner_splits = int(self.protocol.get("inner_splits", 5))
        self.bag_seed = int(self.protocol.get("bag_seed", self.parameters.get("random_state", 42)))

    def _params(self, n_outer_bags: int) -> dict[str, Any]:
        params = dict(self.parameters)
        if int(params.get("outer_bags", n_outer_bags)) != int(n_outer_bags):
            raise ValueError("Trial outer_bags does not match generated bag matrix")
        params["outer_bags"] = int(n_outer_bags)
        if "objective" not in params:
            params["objective"] = "rmse"
        if bool(params.get("objective") != "rmse"):
            raise ValueError("V3.4 EBM boundary/recipe currently requires objective=rmse")
        return params

    def fit(self, frame: pd.DataFrame, target: np.ndarray,
            bags: np.ndarray | None = None) -> "V34EBMRegressor":
        from interpret.glassbox import ExplainableBoostingRegressor

        y = _finite_1d(np.asarray(target, dtype=float), "EBM labels")
        if len(frame) != len(y):
            raise ValueError("EBM frame/label length mismatch")
        z, self.target_state_ = fit_target_transform(y, self.target_transform)
        x = _ebm_feature_frame(frame)
        if bags is None:
            bag_info = build_group_safe_bags(frame, n_outer_bags=4,
                                             n_inner_splits=self.inner_splits,
                                             seed=self.bag_seed)
            matrix = bag_info["bags"]
            self.bag_info_ = bag_info
        else:
            matrix = _validate_custom_bags(np.asarray(bags), len(frame), 4)
            # When custom bags are supplied, still verify their group
            # isolation using the current frame's frozen duplicate groups.
            folded = group_safe_inner_folds(frame, n_splits=self.inner_splits, seed=self.bag_seed)
            assert_group_isolated(matrix, folded["group_id"])
            self.bag_info_ = {
                "bags": matrix,
                "fold": folded["fold"],
                "group_id": folded["group_id"],
                "bag_hash": _sha256_bytes(np.ascontiguousarray(matrix, dtype=np.int8).tobytes()),
                "group_hash": folded["group_hash"],
                "inner_fold_hash": folded["inner_fold_hash"],
                "metadata": {
                    "protocol": BAG_PROTOCOL,
                    "n_outer_bags": int(matrix.shape[0]),
                    "n_inner_splits": int(self.inner_splits),
                    "bag_seed": int(self.bag_seed),
                    "bag_hash": _sha256_bytes(np.ascontiguousarray(matrix, dtype=np.int8).tobytes()),
                    "group_hash": folded["group_hash"],
                    "inner_fold_hash": folded["inner_fold_hash"],
                    "n_samples": int(len(frame)),
                    "bag_train_counts": [int((matrix[i] == 1).sum()) for i in range(matrix.shape[0])],
                    "bag_validation_counts": [int((matrix[i] == -1).sum()) for i in range(matrix.shape[0])],
                },
            }
        params = self._params(matrix.shape[0])
        feature_types = ["continuous"] * len(FEATURES) + ["nominal"]
        self.estimator_ = ExplainableBoostingRegressor(feature_types=feature_types, **params)
        self.estimator_.fit(x, z, bags=matrix)
        self.input_columns_ = tuple(x.columns)
        self.bag_hash_ = str(self.bag_info_["bag_hash"])
        self.group_hash_ = str(self.bag_info_["group_hash"])
        self.inner_fold_hash_ = str(self.bag_info_["inner_fold_hash"])
        try:
            best_iteration = np.asarray(self.estimator_.best_iteration_, dtype=int).ravel().tolist()
        except Exception:  # pragma: no cover - library attribute guard
            best_iteration = []
        self.best_iteration_ = [int(v) for v in best_iteration]
        self.fit_meta_ = {
            "model_kind": self.kind,
            "target_transform": self.target_transform,
            "bag_protocol": BAG_PROTOCOL,
            "bag_hash": self.bag_hash_,
            "group_hash": self.group_hash_,
            "inner_fold_hash": self.inner_fold_hash_,
            "n_outer_bags": int(matrix.shape[0]),
            "n_inner_splits": int(self.inner_splits),
            "bag_seed": int(self.bag_seed),
            "bag_train_counts": list(self.bag_info_["metadata"]["bag_train_counts"]),
            "bag_validation_counts": list(self.bag_info_["metadata"]["bag_validation_counts"]),
            "best_iteration": list(self.best_iteration_),
            "configured_max_rounds": int(params.get("max_rounds", 0)),
        }
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("EBM must be fitted before prediction")
        x = _ebm_feature_frame(frame)
        if tuple(x.columns) != self.input_columns_:
            raise ValueError("V3.4 EBM feature order mismatch")
        raw = np.asarray(self.estimator_.predict(x), dtype=float)
        pred = inverse_target_transform(raw, self.target_transform, self.target_state_)
        if pred.shape != (len(frame),) or not np.isfinite(pred).all():
            raise ValueError("Invalid V3.4 EBM predictions")
        return pred


class V34ResidualRegressor:
    """EBM base prediction plus a signed original/log residual corrector."""

    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        if str(self.trial.get("kind")) != "ebm_residual":
            raise ValueError("Not a V3.4 EBM residual trial")
        params = dict(self.trial.get("parameters", {}))
        if "base_trial" not in params:
            raise ValueError("EBM residual requires parameters.base_trial")
        self.base_trial = deepcopy(dict(params["base_trial"]))
        self.coordinate = str(params.get("coordinate", "original_unit"))
        if self.coordinate not in {"original_unit", "log_unit"}:
            raise ValueError(f"Unsupported residual coordinate: {self.coordinate}")
        corrector = dict(params.get("corrector", {}))
        self.corrector_kind = str(corrector.get("kind", ""))
        self.corrector_params = dict(corrector.get("params", {}))
        if self.corrector_kind not in {"ridge", "catboost"}:
            raise ValueError(f"Unsupported residual corrector: {self.corrector_kind}")
        self.alpha = float(params.get("alpha", 0.25))
        if self.alpha < 0:
            raise ValueError("Residual correction alpha must be nonnegative")
        self.inner_seed = int(params.get("inner_seed", 777))
        self.inner_splits = int(params.get("inner_splits", 5))
        self.target = self.trial.get("target", self.base_trial.get("target"))

    def _feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = expression_frame(frame, "raw").copy()
        if x.isna().any().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
            raise ValueError("Residual corrector features must be finite")
        return x

    def _base_oof(self, frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        folded = group_safe_inner_folds(frame, n_splits=self.inner_splits, seed=self.inner_seed)
        fold = folded["fold"]
        oof = np.full(len(frame), np.nan, dtype=float)
        self.oof_bag_hashes_: list[str] = []
        self.oof_best_iterations_: list[list[int]] = []
        for inner in range(self.inner_splits):
            train_mask = fold != inner
            valid_mask = fold == inner
            if not train_mask.any() or not valid_mask.any():
                raise ValueError(f"Residual inner fold {inner} has an empty side")
            base = V34EBMRegressor(self.base_trial)
            base.fit(frame.loc[train_mask].reset_index(drop=True), y[train_mask])
            oof[valid_mask] = base.predict(frame.loc[valid_mask].reset_index(drop=True))
            self.oof_bag_hashes_.append(str(base.bag_hash_))
            self.oof_best_iterations_.append(list(getattr(base, "best_iteration_", [])))
        if not np.isfinite(oof).all():
            raise ValueError("Residual base OOF coverage failed")
        self.oof_group_hash_ = str(folded["group_hash"])
        self.oof_inner_fold_hash_ = str(folded["inner_fold_hash"])
        return oof

    def _fit_corrector(self, x: pd.DataFrame, residual: np.ndarray) -> None:
        if self.corrector_kind == "ridge":
            from sklearn.compose import ColumnTransformer
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            numeric = [c for c in x.columns if c != "spout_no"]
            pre = ColumnTransformer([
                ("num", StandardScaler(), numeric),
                ("spout", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["spout_no"]),
            ], remainder="drop")
            self.correction_model_ = Pipeline([
                ("pre", pre),
                ("ridge", Ridge(**self.corrector_params)),
            ])
            self.correction_model_.fit(x, residual)
            self.corrector_cat_features_ = None
        elif self.corrector_kind == "catboost":
            params = dict(self.corrector_params)
            params["cat_features"] = ["spout_no"]
            params.setdefault("allow_writing_files", False)
            params.setdefault("verbose", False)
            params.setdefault("thread_count", 1)
            self.correction_model_ = CatBoostRegressor(**params)
            x_cat = x.assign(spout_no=x["spout_no"].astype(str))
            self.correction_model_.fit(x_cat, residual)
            self.corrector_cat_features_ = ["spout_no"]
        else:  # pragma: no cover - constructor validation
            raise ValueError(self.corrector_kind)

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "V34ResidualRegressor":
        started = time.perf_counter()
        y = _finite_1d(np.asarray(target, dtype=float), "residual labels")
        if len(frame) != len(y):
            raise ValueError("Residual frame/label length mismatch")
        base_oof = self._base_oof(frame, y)
        if self.coordinate == "original_unit":
            residual = y - base_oof
        else:
            if np.any(y < 0) or np.any(base_oof < 0):
                raise ValueError("log_unit residual coordinate requires nonnegative labels and predictions")
            residual = np.log1p(y) - np.log1p(np.maximum(base_oof, 0.0))
        if not np.isfinite(residual).all():
            raise ValueError("Residual target is nonfinite")
        x = self._feature_frame(frame)
        self._fit_corrector(x, residual)
        self.input_columns_ = tuple(x.columns)
        self.base_oof_ = base_oof
        # Full current-training-part refit of the parent model.
        self.base_model_ = V34EBMRegressor(self.base_trial)
        self.base_model_.fit(frame, y)
        self.fit_meta_ = {
            "model_kind": "ebm_residual",
            "coordinate": self.coordinate,
            "corrector_kind": self.corrector_kind,
            "corrector_params": deepcopy(self.corrector_params),
            "alpha": self.alpha,
            "inner_seed": int(self.inner_seed),
            "inner_splits": int(self.inner_splits),
            "oof_group_hash": self.oof_group_hash_,
            "oof_inner_fold_hash": self.oof_inner_fold_hash_,
            "oof_bag_hashes": list(self.oof_bag_hashes_),
            "oof_best_iterations": [list(values) for values in self.oof_best_iterations_],
            "full_base_bag_hash": str(self.base_model_.bag_hash_),
            "full_base_best_iteration": list(getattr(self.base_model_, "best_iteration_", [])),
            "seconds": float(time.perf_counter() - started),
        }
        return self

    def _transform_correction(self, base_pred: np.ndarray, correction: np.ndarray) -> np.ndarray:
        if self.alpha == 0.0:
            return np.asarray(base_pred, dtype=float).copy()
        if self.coordinate == "original_unit":
            return np.asarray(base_pred, dtype=float) + self.alpha * np.asarray(correction, dtype=float)
        base_nonneg = np.maximum(np.asarray(base_pred, dtype=float), 0.0)
        return np.expm1(np.log1p(base_nonneg) + self.alpha * np.asarray(correction, dtype=float))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "base_model_"):
            raise RuntimeError("Residual model must be fitted before prediction")
        base_pred = np.asarray(self.base_model_.predict(frame), dtype=float)
        if self.alpha == 0.0:
            return base_pred
        x = self._feature_frame(frame)
        if tuple(x.columns) != self.input_columns_:
            raise ValueError("Residual feature order mismatch")
        if self.corrector_kind == "catboost":
            x = x.assign(spout_no=x["spout_no"].astype(str))
        correction = np.asarray(self.correction_model_.predict(x), dtype=float)
        pred = self._transform_correction(base_pred, correction)
        if pred.shape != (len(frame),) or not np.isfinite(pred).all():
            raise ValueError("Invalid V3.4 residual predictions")
        return pred


class ShrunkSpoutRegressor:
    """Global CatBoost parent plus optional per-spout shrunk specialists."""

    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        if str(self.trial.get("kind")) != "global_spout_shrink":
            raise ValueError("Not a V3.4 global/spout shrink trial")
        params = dict(self.trial.get("parameters", {}))
        parent = params.get("parent_trial", params.get("base_trial"))
        if not isinstance(parent, Mapping):
            raise ValueError("Global/spout shrink requires parameters.parent_trial")
        self.parent_trial = deepcopy(dict(parent))
        self.local_l2_multiplier = float(params.get("local_l2_multiplier", 1.0))
        self.beta = float(params.get("beta", 0.25))
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("Shrink beta must be in [0, 1]")
        self.min_spout_samples = int(params.get("min_spout_samples", 200))
        if self.min_spout_samples < 1:
            raise ValueError("min_spout_samples must be positive")
        self.local_include_spout = bool(params.get("local_include_spout", True))
        if not self.local_include_spout:
            raise ValueError("V3.4 freezes local_include_spout=true for this batch")
        self.target = self.trial.get("target", self.parent_trial.get("target"))

    def _local_trial(self) -> dict:
        trial = deepcopy(self.parent_trial)
        params = dict(trial["parameters"])
        params["l2_leaf_reg"] = float(params.get("l2_leaf_reg", 10.0)) * self.local_l2_multiplier
        trial["parameters"] = params
        return trial

    @staticmethod
    def _fit_one(trial: Mapping[str, Any], frame: pd.DataFrame, y: np.ndarray) -> V31Regressor:
        model = V31Regressor(trial)
        model.fit(frame.reset_index(drop=True), np.asarray(y, dtype=float))
        if not np.isfinite(model.predict(frame.reset_index(drop=True))).all():
            raise ValueError("Shrink component produced nonfinite predictions")
        return model

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "ShrunkSpoutRegressor":
        started = time.perf_counter()
        y = _finite_1d(np.asarray(target, dtype=float), "shrink labels")
        if len(frame) != len(y):
            raise ValueError("Shrink frame/label length mismatch")
        if "spout_no" not in frame:
            raise ValueError("Shrink model requires spout_no")
        self.global_model_ = self._fit_one(self.parent_trial, frame, y)
        counts = frame["spout_no"].astype(int).value_counts().to_dict()
        self.spout_counts_ = {int(k): int(v) for k, v in counts.items()}
        self.local_models_: dict[int, V31Regressor] = {}
        self.local_available_: dict[int, bool] = {}
        self.local_errors_: dict[int, str] = {}
        local_trial = self._local_trial()
        for spout, count in sorted(self.spout_counts_.items()):
            if count < self.min_spout_samples:
                self.local_available_[spout] = False
                continue
            mask = frame["spout_no"].astype(int).to_numpy() == int(spout)
            subset = frame.loc[mask]
            try:
                self.local_models_[spout] = self._fit_one(local_trial, subset, y[mask])
                self.local_available_[spout] = True
            except Exception as exc:  # noqa: BLE001 - fallback is intentional
                self.local_available_[spout] = False
                self.local_errors_[spout] = f"{type(exc).__name__}: {exc}"
        self.fit_meta_ = {
            "model_kind": "global_spout_shrink",
            "local_l2_multiplier": self.local_l2_multiplier,
            "beta": self.beta,
            "min_spout_samples": self.min_spout_samples,
            "local_include_spout": self.local_include_spout,
            "spout_counts": dict(self.spout_counts_),
            "local_fitted_spouts": sorted(self.local_models_),
            "local_fallback_spouts": sorted(
                spout for spout, available in self.local_available_.items() if not available
            ),
            "local_errors": dict(self.local_errors_),
            "seconds": float(time.perf_counter() - started),
        }
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "global_model_"):
            raise RuntimeError("Shrink model must be fitted before prediction")
        global_pred = np.asarray(self.global_model_.predict(frame), dtype=float)
        if not np.isfinite(global_pred).all():
            raise ValueError("Global shrink predictions are nonfinite")
        if self.beta == 0.0 or not self.local_models_:
            return global_pred
        result = global_pred.astype(float, copy=True)
        spouts = frame["spout_no"].astype(int).to_numpy()
        for spout in np.unique(spouts):
            if int(spout) not in self.local_models_:
                continue
            mask = spouts == int(spout)
            local_pred = np.asarray(self.local_models_[int(spout)].predict(
                frame.loc[mask].reset_index(drop=True)
            ), dtype=float)
            result[mask] = (1.0 - self.beta) * global_pred[mask] + self.beta * local_pred
        if not np.isfinite(result).all():
            raise ValueError("Shrunk spout predictions are nonfinite")
        return result


class V34Regressor:
    """Trial-kind dispatcher used by the V3.4 runner and tests."""

    def __init__(self, trial: Mapping[str, Any]):
        kind = str(trial.get("kind"))
        self.trial = deepcopy(dict(trial))
        if kind in EBM_KINDS:
            self.impl = V34EBMRegressor(trial)
        elif kind == "ebm_residual":
            self.impl = V34ResidualRegressor(trial)
        elif kind == "global_spout_shrink":
            self.impl = ShrunkSpoutRegressor(trial)
        else:
            raise ValueError(f"Unknown V3.4 trial kind: {kind}")

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "V34Regressor":
        self.impl.fit(frame, target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.impl.predict(frame)


def evaluate_v34_outer_folds(train: pd.DataFrame, folds: np.ndarray, trial: Mapping[str, Any],
                             fold_ids: Sequence[int]) -> dict:
    """Evaluate one V3.4 recipe on outer training/validation folds.

    The function passes only outer-training rows to model fits and never to
    inner preprocessing.  It returns full-length predictions with NaN outside
    ``fold_ids``, exactly like the V3.1/V3.3 local runners.
    """
    target = trial["target"]
    predictions = np.full(len(train), np.nan, dtype=float)
    fold_scores: dict[str, float] = {}
    fit_meta: list[dict[str, Any]] = []
    for fold in fold_ids:
        fold = int(fold)
        started = time.perf_counter()
        training = train.loc[folds != fold].reset_index(drop=True)
        valid = train.loc[folds == fold].reset_index(drop=True)
        model = V34Regressor(trial)
        model.fit(training, training[target].to_numpy(dtype=float))
        pred = model.predict(valid)
        predictions[folds == fold] = pred
        fold_scores[str(fold)] = float(wmape(valid[target], pred))
        meta = deepcopy(getattr(model.impl, "fit_meta_", {}))
        meta["outer_fold"] = fold
        meta["n_train"] = int(len(training))
        meta["n_valid"] = int(len(valid))
        meta["seconds"] = float(time.perf_counter() - started)
        fit_meta.append(meta)
    mask = np.isin(folds, list(fold_ids))
    if not np.isfinite(predictions[mask]).all():
        raise ValueError("V3.4 OOF coverage failed")
    return {
        "trial_id": trial["trial_id"],
        "kind": trial["kind"],
        "target": target,
        "pooled_wmape": float(wmape(train.loc[mask, target], predictions[mask])),
        "mean_wmape": float(np.mean([fold_scores[str(int(f))] for f in fold_ids])),
        "fold_scores": fold_scores,
        "fit_meta": fit_meta,
        "predictions": predictions,
    }
