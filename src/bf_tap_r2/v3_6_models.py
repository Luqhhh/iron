"""V3.6 EBM loss/training-protocol estimator and outer-fold evaluator.

This module opens only the dimensions that were outside the V3.5 wrapper
whitelist: objective/loss, leaf-output L2 regularization, greedy-vs-cyclic
training, and initial smoothing.  It does not change the V3.5 feature packs,
interaction semantics, target transforms, or bag protocol.

Every fitted model stores the requested parameters and the parameters read back
from the actual InterpretML estimator.  A requested parameter that cannot be
observed on the estimator is a hard error, so a dropped parameter can never be
mistaken for a completed V3.6 trial.
"""
from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import wmape
from .v3_5_models import EBM_KINDS, V35EBMRegressor


V36_EBM_ALLOWED = {
    "max_bins",
    "min_samples_leaf",
    "interactions",
    "max_interaction_bins",
    "max_leaves",
    "objective",
    "learning_rate",
    "outer_bags",
    "inner_bags",
    "max_rounds",
    "early_stopping_rounds",
    "random_state",
    "n_jobs",
    "reg_lambda",
    "greedy_ratio",
    "cyclic_progress",
    "smoothing_rounds",
    "interaction_smoothing_rounds",
}

V36_EBM_OBJECTIVES = {
    "rmse",
    "pseudo_huber:delta=0.01",
    "pseudo_huber:delta=0.03",
    "pseudo_huber:delta=0.10",
}

V36_EBM_EFFECTIVENESS_KEYS = (
    "max_bins",
    "min_samples_leaf",
    "interactions",
    "max_interaction_bins",
    "max_leaves",
    "objective",
    "learning_rate",
    "outer_bags",
    "inner_bags",
    "max_rounds",
    "early_stopping_rounds",
    "random_state",
    "n_jobs",
    "reg_lambda",
    "greedy_ratio",
    "cyclic_progress",
    "smoothing_rounds",
    "interaction_smoothing_rounds",
)


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return list(left) == list(right)
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        try:
            return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))
        except (TypeError, ValueError):
            return False
    return left == right


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class V36EBMRegressor(V35EBMRegressor):
    """V3.5-compatible EBM with the V3.6 loss/training controls."""

    def __init__(self, trial: Mapping[str, Any]):
        super().__init__(trial)
        objective = str(self.parameters.get("objective", "rmse"))
        if objective not in V36_EBM_OBJECTIVES:
            raise ValueError(f"Unsupported V3.6 EBM objective: {objective!r}")
        unknown = sorted(set(self.parameters) - V36_EBM_ALLOWED)
        if unknown:
            raise ValueError(f"V3.6 EBM trial carries unsupported parameters: {unknown}")

    def _params(self, n_outer_bags: int) -> dict[str, Any]:
        params = {
            key: deepcopy(value)
            for key, value in self.parameters.items()
            if key in V36_EBM_ALLOWED
        }
        if int(params.get("outer_bags", n_outer_bags)) != int(n_outer_bags):
            raise ValueError("V3.6 trial outer_bags does not match the generated bag matrix")
        params["outer_bags"] = int(n_outer_bags)
        params["interactions"] = deepcopy(self._interactions)
        params.setdefault("objective", "rmse")
        if str(params["objective"]) not in V36_EBM_OBJECTIVES:
            raise ValueError(f"Unsupported V3.6 EBM objective: {params['objective']!r}")
        return params

    def _validate_effective_params(self) -> None:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("V3.6 EBM has not been fitted")
        actual_all = self.estimator_.get_params(deep=False)
        checks: dict[str, dict[str, Any]] = {}
        requested = {
            key: deepcopy(value)
            for key, value in self.parameters.items()
            if key in V36_EBM_ALLOWED
        }
        requested["outer_bags"] = int(self.estimator_.outer_bags)
        missing = sorted(set(requested) - set(actual_all))
        if missing:
            raise ValueError(f"InterpretML silently dropped V3.6 EBM parameters: {missing}")
        for key, requested_value in requested.items():
            actual_value = actual_all[key]
            if not _values_equal(requested_value, actual_value):
                raise ValueError(
                    f"V3.6 EBM parameter {key!r} did not take effect: "
                    f"requested={requested_value!r}, actual={actual_value!r}"
                )
            checks[key] = {
                "requested": _jsonable(requested_value),
                "actual": _jsonable(actual_value),
                "effective": True,
            }
        ignored = sorted(set(self.parameters) - set(requested))
        if ignored:
            raise ValueError(f"V3.6 EBM ignored unsupported parameters: {ignored}")
        self.requested_params_ = requested
        self.effective_params_ = {
            key: _jsonable(actual_all[key]) for key in V36_EBM_EFFECTIVENESS_KEYS if key in actual_all
        }
        self.parameter_checks_ = checks

    def fit(self, frame: pd.DataFrame, target: np.ndarray,
            bags: np.ndarray | None = None) -> "V36EBMRegressor":
        result = super().fit(frame, target, bags=bags)
        self._validate_effective_params()
        self.fit_meta_ = dict(self.fit_meta_)
        self.fit_meta_.update({
            "requested_params": _jsonable(self.requested_params_),
            "effective_params": _jsonable(self.effective_params_),
            "parameter_checks": _jsonable(self.parameter_checks_),
            "ignored_params": [],
        })
        return result


class V36Regressor:
    """Trial-kind dispatcher for V3.6 EBM and numeric-encoding network trials."""

    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        kind = str(trial.get("kind"))
        if kind in EBM_KINDS or kind == "ebm_loss":
            self.impl = V36EBMRegressor(trial)
        elif kind in {"numeric_mlp", "numeric_tabm"}:
            from .v3_6_networks import V36NetworkRegressor

            self.impl = V36NetworkRegressor(trial)
        else:
            raise ValueError(f"Unknown V3.6 trial kind: {kind}")

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "V36Regressor":
        self.impl.fit(frame, target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.impl.predict(frame)


def evaluate_v36_outer_folds(train: pd.DataFrame, folds: np.ndarray, trial: Mapping[str, Any],
                             fold_ids: Sequence[int]) -> dict[str, Any]:
    """Evaluate one V3.6 recipe on outer training/validation folds.

    Only the current outer-training part is passed to the model.  The evaluator
    never sees validation labels during fitting.
    """
    target = str(trial["target"])
    predictions = np.full(len(train), np.nan, dtype=float)
    fold_scores: dict[str, float] = {}
    fit_meta: list[dict[str, Any]] = []
    for fold in fold_ids:
        fold = int(fold)
        started = time.perf_counter()
        training = train.loc[folds != fold].reset_index(drop=True)
        valid = train.loc[folds == fold].reset_index(drop=True)
        model = V36Regressor(trial)
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
        raise ValueError("V3.6 OOF coverage failed")
    return {
        "trial_id": str(trial["trial_id"]),
        "kind": str(trial["kind"]),
        "target": target,
        "pooled_wmape": float(wmape(train.loc[mask, target], predictions[mask])),
        "mean_wmape": float(np.mean([fold_scores[str(int(f))] for f in fold_ids])),
        "fold_scores": fold_scores,
        "fit_meta": fit_meta,
        "predictions": predictions,
    }


__all__ = [
    "V36_EBM_ALLOWED",
    "V36_EBM_EFFECTIVENESS_KEYS",
    "V36_EBM_OBJECTIVES",
    "V36EBMRegressor",
    "V36Regressor",
    "evaluate_v36_outer_folds",
]
