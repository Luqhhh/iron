"""V3.3 original-unit tree-count selection bridge.

Mode A keeps the existing library-metric early stopping.  Mode B fits an inner
CatBoost model without early stopping, evaluates a pre-fixed tree-count grid on
the inner validation split after inverse-transforming both predictions and
labels, and refits on the full outer-training part with the selected tree count.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .metrics import wmape
from .splits import make_folds
from .v3_1_models import V31Regressor, fit_with_inner_early_stop
from .v3_local_search import TrialRegressor, fit_target_transform, inverse_target_transform


def checkpoint_tree_counts(max_iter: int, step: int = 100) -> list[int]:
    maximum = int(max_iter)
    if maximum < 1:
        raise ValueError("max_iter must be positive")
    if step < 1:
        raise ValueError("step must be positive")
    values = list(range(step, maximum + 1, step))
    if not values or values[-1] != maximum:
        values.append(maximum)
    return sorted(set(int(v) for v in values))


def _fit_catboost_raw(trial: Mapping[str, Any], frame: pd.DataFrame, target: str,
                      iterations: int, *, with_eval: bool = False,
                      eval_frame: pd.DataFrame | None = None) -> TrialRegressor:
    model = TrialRegressor(trial)
    x = model._x(frame)
    y = frame[target].to_numpy(dtype=float)
    z, state = fit_target_transform(y, model.target_transform)
    params = dict(trial["parameters"])
    params["iterations"] = int(iterations)
    for key in ("use_best_model", "od_type", "od_wait"):
        params.pop(key, None)
    estimator = CatBoostRegressor(**params)
    if with_eval and eval_frame is not None:
        eval_x = model._x(eval_frame)
        eval_z = fit_target_transform(eval_frame[target].to_numpy(dtype=float), model.target_transform)[0] if False else None
        # Library-metric mode is handled by V31Regressor; this direct path is
        # reserved for the original-unit checkpoint selection.
    estimator.fit(x, z)
    model.estimator_ = estimator
    model.target_state_ = state
    model.input_columns_ = tuple(x.columns)
    return model


def fit_catboost_original_unit_selection(trial: Mapping[str, Any], training: pd.DataFrame,
                                         target: str, inner_seed: int,
                                         checkpoint_step: int = 100) -> tuple[TrialRegressor, dict]:
    """Mode B: select tree count on inner-valid original-unit WMAPE."""
    family = trial.get("base_family", trial["family"])
    if family != "catboost":
        raise ValueError("V3.3 original-unit bridge currently supports CatBoost backbones only")
    inner = make_folds(training, seed=inner_seed, n_splits=5).set_index("sample_id").loc[training.sample_id]
    inner_fold = inner.fold.to_numpy()
    valid_mask = inner_fold == 0
    inner_train = training.loc[~valid_mask].reset_index(drop=True)
    inner_valid = training.loc[valid_mask].reset_index(drop=True)
    probe = _fit_catboost_raw(trial, inner_train, target, int(trial["parameters"].get("iterations", 1000)))
    z_valid, _ = fit_target_transform(inner_valid[target].to_numpy(dtype=float), probe.target_transform) if False else (None, None)
    x_valid = probe._x(inner_valid)
    raw_valid = np.asarray(probe.estimator_.predict(x_valid), dtype=float)
    max_iter = int(probe.estimator_.tree_count_)
    grid = checkpoint_tree_counts(max_iter, checkpoint_tree_counts_step(checkpoint_step))
    best = None
    curve = []
    for trees in grid:
        raw = np.asarray(probe.estimator_.predict(x_valid, ntree_end=trees), dtype=float)
        pred = inverse_target_transform(raw, probe.target_transform, probe.target_state_)
        score = wmape(inner_valid[target].to_numpy(dtype=float), pred)
        curve.append({"trees": int(trees), "wmape": float(score)})
        if best is None or score < best[0]:
            best = (score, int(trees))
    if best is None:
        raise RuntimeError("Original-unit checkpoint selection failed")
    selected = int(best[1])
    final_model = _fit_catboost_raw(trial, training, target, selected)
    return final_model, {
        "selection_mode": "original_unit_inner_wmape",
        "best_iteration_index": selected - 1,
        "selected_num_boost_round": selected,
        "actual_num_boost_round": int(final_model.estimator_.tree_count_),
        "configured_num_boost_round": int(trial["parameters"].get("iterations", selected)),
        "checkpoint_curve": curve,
    }


def checkpoint_tree_counts_step(step: int) -> int:
    return int(step)


def evaluate_tree_selection_mode(train: pd.DataFrame, folds: np.ndarray, trial: Mapping[str, Any],
                                 mode: str, fold_ids: Sequence[int], inner_seed: int) -> dict:
    target = trial["target"]
    predictions = np.full(len(train), np.nan)
    fold_scores = {}
    meta = []
    for fold in fold_ids:
        training = train.loc[folds != fold].reset_index(drop=True)
        valid = train.loc[folds == fold].reset_index(drop=True)
        if mode == "A":
            model, record = fit_with_inner_early_stop(trial, training, target, inner_seed=inner_seed + int(fold))
        elif mode == "B":
            model, record = fit_catboost_original_unit_selection(trial, training, target, inner_seed=inner_seed + int(fold))
        else:
            raise ValueError(mode)
        pred = model.predict(valid)
        predictions[folds == fold] = pred
        fold_scores[str(int(fold))] = float(wmape(valid[target], pred))
        meta.append({"fold": int(fold), **record})
    mask = np.isin(folds, list(fold_ids))
    if not np.isfinite(predictions[mask]).all():
        raise ValueError("V3.3 metric-mode OOF coverage failed")
    return {
        "trial_id": trial["trial_id"],
        "target": target,
        "mode": mode,
        "pooled_wmape": float(wmape(train.loc[mask, target], predictions[mask])),
        "fold_scores": fold_scores,
        "fit_meta": meta,
        "predictions": predictions,
    }
