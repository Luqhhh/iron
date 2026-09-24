"""V3.5 estimators: regularized/expression EBMs and global-plus-local EBM.

The implementation reuses the frozen V3.4 group-safe bag protocol and target
transforms.  V3.5 adds two orthogonal capabilities required by the task book:

1. feature packs (``raw`` / ``four``) with pairwise explicit interactions;
2. global EBM plus per-spout local EBM with original-unit shrinkage.

No model reads outer-validation labels during fit.  Local models are fitted only
on the current training part for each outer fold.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import FEATURES
from .metrics import wmape
from .v3_4_bags import (
    BAG_PROTOCOL,
    assert_group_isolated,
    build_group_safe_bags,
    group_safe_inner_folds,
)
from .v3_local_search import expression_frame, fit_target_transform, inverse_target_transform

EBM_KINDS = {
    "ebm", "ebm_boundary", "ebm_base", "ebm_regularized", "ebm_expression",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite_1d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"Invalid {name}")
    return array


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


def _frame_columns(feature_set: str) -> list[str]:
    if feature_set == "raw":
        return [*FEATURES, "spout_no"]
    if feature_set == "four":
        return expression_frame(pd.DataFrame(columns=[*FEATURES, "spout_no"]), "four").columns.tolist()
    raise ValueError(f"Unknown V3.5 feature_set: {feature_set}")


def _feature_frame(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    x = expression_frame(frame, feature_set).copy()
    expected = _frame_columns(feature_set)
    if list(x.columns) != expected:
        raise ValueError(f"V3.5 feature order mismatch for {feature_set}")
    if x.isna().any().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("V3.5 EBM feature frame must be finite")
    return x


def _normalise_interactions(interactions: Any) -> int | list[tuple[int, int]]:
    if isinstance(interactions, bool):
        raise ValueError("Boolean interactions are not a valid V3.5 interaction specification")
    if isinstance(interactions, (int, np.integer)):
        value = int(interactions)
        if value < 0:
            raise ValueError("Automatic interaction count must be nonnegative")
        return value
    if isinstance(interactions, (list, tuple)):
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for pair in interactions:
            if len(pair) != 2:
                raise ValueError(f"Explicit interaction must be a pair, got {pair!r}")
            i, j = int(pair[0]), int(pair[1])
            if i < 0 or j < 0 or i == j:
                raise ValueError(f"Invalid explicit interaction pair: {pair!r}")
            key = tuple(sorted((i, j)))
            if key not in seen:
                seen.add((int(key[0]), int(key[1])))
                out.append((int(key[0]), int(key[1])))
        if not out:
            raise ValueError("Explicit interaction list may not be empty; use 0 for additive EBM")
        return out
    raise ValueError(f"Unsupported V3.5 interactions specification: {interactions!r}")


class V35EBMRegressor:
    """Complete regularized/expression EBM recipe with explicit interactions."""

    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        kind = str(self.trial.get("kind", "ebm"))
        if kind not in EBM_KINDS:
            raise ValueError(f"Not a V3.5 EBM trial: {kind}")
        self.kind = kind
        self.target = str(self.trial["target"])
        self.target_transform = str(self.trial["target_transform"])
        self.feature_set = str(self.trial.get("feature_set", "raw"))
        self.parameters = dict(self.trial.get("parameters", {}))
        self.protocol = dict(self.trial.get("protocol", {}))
        self.inner_splits = int(self.protocol.get("inner_splits", 5))
        self.bag_seed = int(self.protocol.get("bag_seed", self.parameters.get("random_state", 42)))
        self._interactions = _normalise_interactions(self.parameters.get("interactions", 0))

    def _params(self, n_outer_bags: int) -> dict[str, Any]:
        allowed = {
            "max_bins", "min_samples_leaf", "interactions", "max_interaction_bins",
            "max_leaves", "objective", "learning_rate", "outer_bags", "inner_bags",
            "max_rounds", "early_stopping_rounds", "random_state", "n_jobs",
        }
        params = {key: deepcopy(value) for key, value in self.parameters.items() if key in allowed}
        if int(params.get("outer_bags", n_outer_bags)) != int(n_outer_bags):
            raise ValueError("Trial outer_bags does not match generated bag matrix")
        params["outer_bags"] = int(n_outer_bags)
        params["interactions"] = deepcopy(self._interactions)
        params.setdefault("objective", "rmse")
        if str(params["objective"]) != "rmse":
            raise ValueError("V3.5 EBM requires objective=rmse")
        return params

    def fit(self, frame: pd.DataFrame, target: np.ndarray,
            bags: np.ndarray | None = None) -> "V35EBMRegressor":
        from interpret.glassbox import ExplainableBoostingRegressor

        y = _finite_1d(np.asarray(target, dtype=float), "EBM labels")
        if len(frame) != len(y):
            raise ValueError("V3.5 EBM frame/label length mismatch")
        z, self.target_state_ = fit_target_transform(y, self.target_transform)
        x = _feature_frame(frame, self.feature_set)
        if bags is None:
            bag_info = build_group_safe_bags(
                frame, n_outer_bags=4, n_inner_splits=self.inner_splits, seed=self.bag_seed
            )
            matrix = bag_info["bags"]
            self.bag_info_ = bag_info
        else:
            matrix = _validate_custom_bags(np.asarray(bags), len(frame), 4)
            folded = group_safe_inner_folds(frame, n_splits=self.inner_splits, seed=self.bag_seed)
            assert_group_isolated(matrix, folded["group_id"])
            bag_hash = _sha256_bytes(np.ascontiguousarray(matrix, dtype=np.int8).tobytes())
            self.bag_info_ = {
                "bags": matrix,
                "fold": folded["fold"],
                "group_id": folded["group_id"],
                "bag_hash": bag_hash,
                "group_hash": folded["group_hash"],
                "inner_fold_hash": folded["inner_fold_hash"],
                "metadata": {
                    "protocol": BAG_PROTOCOL,
                    "n_outer_bags": int(matrix.shape[0]),
                    "n_inner_splits": int(self.inner_splits),
                    "bag_seed": int(self.bag_seed),
                    "bag_hash": bag_hash,
                    "group_hash": folded["group_hash"],
                    "inner_fold_hash": folded["inner_fold_hash"],
                    "n_samples": int(len(frame)),
                    "bag_train_counts": [int((matrix[i] == 1).sum()) for i in range(matrix.shape[0])],
                    "bag_validation_counts": [int((matrix[i] == -1).sum()) for i in range(matrix.shape[0])],
                },
            }
        params = self._params(matrix.shape[0])
        feature_types = ["continuous"] * len(x.columns)
        spout_index = list(x.columns).index("spout_no")
        feature_types[spout_index] = "nominal"
        # Validate explicit interaction indices against the current frame.
        if isinstance(self._interactions, list):
            for i, j in self._interactions:
                if max(i, j) >= len(x.columns):
                    raise ValueError("Explicit interaction index exceeds feature count")
        self.estimator_ = ExplainableBoostingRegressor(
            feature_names=list(x.columns), feature_types=feature_types, **params
        )
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
            "feature_set": self.feature_set,
            "interactions": deepcopy(self._interactions),
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
            raise RuntimeError("V3.5 EBM must be fitted before prediction")
        x = _feature_frame(frame, self.feature_set)
        if tuple(x.columns) != self.input_columns_:
            raise ValueError("V3.5 EBM feature order mismatch")
        raw = np.asarray(self.estimator_.predict(x), dtype=float)
        pred = inverse_target_transform(raw, self.target_transform, self.target_state_)
        if pred.shape != (len(frame),) or not np.isfinite(pred).all():
            raise ValueError("Invalid V3.5 EBM predictions")
        return pred


class V35GlobalSpoutEBMRegressor:
    """Global EBM plus optional per-spout local EBM shrinking in original units."""

    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        if str(self.trial.get("kind")) != "global_spout_ebm":
            raise ValueError("Not a V3.5 global/local EBM trial")
        params = dict(self.trial.get("parameters", {}))
        if not isinstance(params.get("parent_trial"), Mapping):
            raise ValueError("V3.5 global/local EBM requires parameters.parent_trial")
        self.parent_trial = deepcopy(dict(params["parent_trial"]))
        self.local_min_samples_leaf = int(params.get("local_min_samples_leaf", 60))
        self.local_interactions = _normalise_interactions(params.get("local_interactions", 0))
        self.beta = float(params.get("beta", 0.30))
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("V3.5 shrink beta must be in [0, 1]")
        self.min_spout_samples = int(params.get("min_spout_samples", 200))
        if self.min_spout_samples < 1:
            raise ValueError("min_spout_samples must be positive")
        self.local_include_spout = bool(params.get("local_include_spout", True))
        if not self.local_include_spout:
            raise ValueError("V3.5 freezes local_include_spout=true for this batch")
        self.feature_set = str(params.get("feature_set", self.parent_trial.get("feature_set", "raw")))
        self.target = str(self.trial.get("target", self.parent_trial.get("target")))
        self.component_key = str(params.get("component_key", "")) or None

    def _local_trial(self) -> dict:
        trial = deepcopy(self.parent_trial)
        trial["kind"] = "ebm_regularized"
        params = dict(trial.get("parameters", {}))
        params["min_samples_leaf"] = int(self.local_min_samples_leaf)
        params["interactions"] = deepcopy(self.local_interactions)
        trial["parameters"] = params
        trial["feature_set"] = self.feature_set
        return trial

    @staticmethod
    def _fit_ebm(trial: Mapping[str, Any], frame: pd.DataFrame, y: np.ndarray) -> V35EBMRegressor:
        model = V35EBMRegressor(trial)
        model.fit(frame.reset_index(drop=True), np.asarray(y, dtype=float))
        return model

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "V35GlobalSpoutEBMRegressor":
        started = time.perf_counter()
        y = _finite_1d(np.asarray(target, dtype=float), "global/local EBM labels")
        if len(frame) != len(y):
            raise ValueError("V3.5 global/local EBM frame/label length mismatch")
        if "spout_no" not in frame:
            raise ValueError("V3.5 global/local EBM requires spout_no")
        self.global_model_ = self._fit_ebm(self.parent_trial, frame, y)
        counts = frame["spout_no"].astype(int).value_counts().to_dict()
        self.spout_counts_ = {int(k): int(v) for k, v in counts.items()}
        self.local_models_: dict[int, V35EBMRegressor] = {}
        self.local_available_: dict[int, bool] = {}
        self.local_errors_: dict[int, str] = {}
        if self.beta > 0.0:
            local_trial = self._local_trial()
            for spout, count in sorted(self.spout_counts_.items()):
                if count < self.min_spout_samples:
                    self.local_available_[spout] = False
                    self.local_errors_[spout] = "below_min_spout_samples"
                    continue
                mask = frame["spout_no"].astype(int).to_numpy() == int(spout)
                subset = frame.loc[mask]
                try:
                    self.local_models_[spout] = self._fit_ebm(local_trial, subset, y[mask])
                    self.local_available_[spout] = True
                except Exception as exc:  # noqa: BLE001 - fallback is intentional
                    self.local_available_[spout] = False
                    self.local_errors_[spout] = f"{type(exc).__name__}: {exc}"
        else:
            for spout in sorted(self.spout_counts_):
                self.local_available_[spout] = False
                self.local_errors_[spout] = "beta_zero_global_only"
        self.fit_meta_ = {
            "model_kind": "global_spout_ebm",
            "parent_trial_id": str(self.parent_trial.get("trial_id", "")),
            "component_key": self.component_key,
            "feature_set": self.feature_set,
            "local_min_samples_leaf": int(self.local_min_samples_leaf),
            "local_interactions": deepcopy(self.local_interactions),
            "beta": self.beta,
            "min_spout_samples": int(self.min_spout_samples),
            "local_include_spout": bool(self.local_include_spout),
            "spout_counts": dict(self.spout_counts_),
            "local_fitted_spouts": sorted(self.local_models_),
            "local_fallback_spouts": sorted(
                spout for spout, available in self.local_available_.items() if not available
            ),
            "local_errors": dict(self.local_errors_),
            "global_bag_hash": str(getattr(self.global_model_, "bag_hash_", "")),
            "seconds": float(time.perf_counter() - started),
        }
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "global_model_"):
            raise RuntimeError("V3.5 global/local EBM must be fitted before prediction")
        global_pred = np.asarray(self.global_model_.predict(frame), dtype=float)
        if not np.isfinite(global_pred).all():
            raise ValueError("Global EBM predictions are nonfinite")
        if self.beta == 0.0 or not self.local_models_:
            return global_pred
        result = global_pred.astype(float, copy=True)
        spouts = frame["spout_no"].astype(int).to_numpy()
        for spout in np.unique(spouts):
            if int(spout) not in self.local_models_:
                continue
            mask = spouts == int(spout)
            local_pred = np.asarray(
                self.local_models_[int(spout)].predict(frame.loc[mask].reset_index(drop=True)),
                dtype=float,
            )
            result[mask] = (1.0 - self.beta) * global_pred[mask] + self.beta * local_pred
        if not np.isfinite(result).all():
            raise ValueError("V3.5 global/local EBM predictions are nonfinite")
        return result


class V35Regressor:
    """Trial-kind dispatcher for the V3.5 runner and tests."""

    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        kind = str(trial.get("kind"))
        if kind in EBM_KINDS:
            self.impl = V35EBMRegressor(trial)
        elif kind == "global_spout_ebm":
            self.impl = V35GlobalSpoutEBMRegressor(trial)
        else:
            raise ValueError(f"Unknown V3.5 trial kind: {kind}")

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "V35Regressor":
        self.impl.fit(frame, target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.impl.predict(frame)


def evaluate_v35_outer_folds(train: pd.DataFrame, folds: np.ndarray, trial: Mapping[str, Any],
                             fold_ids: Sequence[int]) -> dict:
    """Evaluate one V3.5 recipe on outer training/validation folds."""
    target = trial["target"]
    predictions = np.full(len(train), np.nan, dtype=float)
    fold_scores: dict[str, float] = {}
    fit_meta: list[dict[str, Any]] = []
    for fold in fold_ids:
        fold = int(fold)
        started = time.perf_counter()
        training = train.loc[folds != fold].reset_index(drop=True)
        valid = train.loc[folds == fold].reset_index(drop=True)
        model = V35Regressor(trial)
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
        raise ValueError("V3.5 OOF coverage failed")
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
