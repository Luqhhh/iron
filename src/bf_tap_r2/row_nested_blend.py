"""Row-disjoint blend selection engine, separate from historical V5 scoring.

No outer evaluation labels are accepted by the selection/prediction API. Inner
OOF is refitted within the outer training partition, never imported from global
OOF caches. Factories must be audited, freshly initialized, train-only estimators;
an in-process boundary cannot sandbox arbitrary callbacks or their closures.
This module is not a competition runner, release gate or model-quality result.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping
import weakref

import numpy as np
import pandas as pd

from .data import FEATURES


@dataclass(frozen=True)
class NestedPrediction:
    candidate: str | None
    alpha: float
    inner_absolute_error: float
    prediction: np.ndarray
    reference_prediction: np.ndarray
    evaluation_ids: tuple[str, ...]
    fit_ledger: tuple[dict, ...]


def _frame(frame, role):
    required = {"sample_id", "spout_no", *FEATURES}
    if not isinstance(frame, pd.DataFrame) or set(frame.columns) != required:
        raise ValueError(f"{role} must contain only IDs, spout and frozen numeric features")
    if frame.columns.duplicated().any() or frame.empty:
        raise ValueError(f"{role} has duplicate columns or no rows")
    ids = frame.sample_id.tolist()
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{role} requires unique nonempty string IDs")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in (*FEATURES, "spout_no")):
        raise ValueError(f"{role} features must be numeric")
    return tuple(ids), pd.util.hash_pandas_object(frame[list(FEATURES)], index=False).to_numpy()


def _aligned_labels(values, ids, role):
    if (not isinstance(values, pd.Series) or not values.index.is_unique
            or set(values.index) != set(ids)):
        raise ValueError(f"{role} labels require a unique sample-ID index with exact coverage")
    actual = values.loc[list(ids)].to_numpy(dtype=float, copy=True)
    if actual.shape != (len(ids),) or not np.isfinite(actual).all() or (actual <= 0).any():
        raise ValueError(f"{role} labels must be finite and positive")
    return actual


def select_and_predict(
    training_frame: pd.DataFrame,
    training_labels,
    evaluation_frame: pd.DataFrame,
    *,
    inner_folds,
    reference_factory: Callable,
    candidate_factories: Mapping[str, Callable],
    alpha_grid,
    max_fit_calls: int,
) -> NestedPrediction:
    """Select one candidate/alpha on T-only inner OOF, then predict E once.

    Fold assignments and factory ordering must be frozen before execution. IDs
    are metadata for audited adapters, never model features. Preprocessing,
    bins, early stopping and target transforms belong inside each fit boundary.
    T/E duplicate feature groups are refused, including duplicates across spouts.
    No existing predictions are accepted. Returned arrays retain E input order.
    The caller needs an append-only private runner before official-data use.
    """
    train_ids, train_groups = _frame(training_frame, "Training")
    eval_ids, eval_groups = _frame(evaluation_frame, "Evaluation")
    if set(train_ids) & set(eval_ids):
        raise ValueError("Outer training/evaluation IDs overlap")
    if set(train_groups) & set(eval_groups):
        raise ValueError("Duplicate feature group crosses the outer boundary")
    labels = _aligned_labels(training_labels, train_ids, "Training")
    folds = np.asarray(inner_folds)
    if folds.shape != labels.shape or folds.dtype.kind not in "iu" or (folds < 0).any():
        raise ValueError("Invalid inner fold vector")
    fold_ids = sorted(set(folds.tolist()))
    if len(fold_ids) < 2:
        raise ValueError("At least two nonempty inner folds are required")
    for group in set(train_groups):
        if len(set(folds[train_groups == group])) != 1:
            raise ValueError("Duplicate feature group crosses an inner boundary")
    grid = np.asarray(alpha_grid, dtype=float)
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
            or (grid < 0).any() or (grid > 1).any()
            or len(np.unique(grid)) != len(grid) or 0.0 not in grid):
        raise ValueError("Alpha grid must be unique, finite, bounded and include zero")
    grid = np.sort(grid)
    candidates = list(candidate_factories.items())
    if not candidates or any(not isinstance(n, str) or not n or n == "reference" for n, _ in candidates):
        raise ValueError("Require named frozen candidates, excluding reserved 'reference'")
    factories = [("reference", reference_factory), *candidates]
    if any(not callable(f) for _, f in factories):
        raise ValueError("Factories are required; global OOF arrays are not accepted")
    required_calls = len(fold_ids) * len(factories) + 2
    if type(max_fit_calls) is not int or max_fit_calls < required_calls:
        raise ValueError(f"Fit budget must reserve {required_calls} calls before execution")
    ledger, previous = [], []

    def fit_predict(name, factory, fit_mask, query, phase, fold):
        estimator = factory()
        if any(ref() is estimator for ref in previous):
            raise ValueError("Factories must return fresh estimators, not reused instances")
        try:
            previous.append(weakref.ref(estimator))
        except TypeError as exc:
            raise ValueError("Estimator adapters must support weak references") from exc
        if not callable(getattr(estimator, "fit", None)) or not callable(getattr(estimator, "predict", None)):
            raise ValueError("Estimator requires fit and predict")
        fit_frame = training_frame.iloc[np.flatnonzero(fit_mask)].copy(deep=True)
        fit_labels = labels[fit_mask].copy()
        # No E targets are in either frame; query labels are never passed to fit.
        estimator.fit(fit_frame, fit_labels)
        values = np.asarray(estimator.predict(query.copy(deep=True)), dtype=float)
        if values.shape != (len(query),) or not np.isfinite(values).all():
            raise ValueError("Predictions must be aligned one-dimensional finite values")
        ledger.append({"phase": phase, "model": name, "inner_fold": fold,
                       "training_ids": tuple(training_frame.loc[fit_mask, "sample_id"]),
                       "prediction_ids": tuple(query.sample_id)})
        # Match nonnegative competition output semantics inside selection too.
        return np.maximum(values, 0.0)

    oof = {name: np.full(len(labels), np.nan) for name, _ in factories}
    for fold in fold_ids:
        held = folds == fold
        query = training_frame.iloc[np.flatnonzero(held)]
        for name, factory in factories:
            oof[name][held] = fit_predict(name, factory, ~held, query, "inner_oof", int(fold))
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise ValueError("Incomplete inner OOF coverage")
    reference = oof["reference"]
    best_error = float(np.abs(labels - reference).sum())
    best_name, best_alpha = None, 0.0
    # Exact ties prefer reference-only, then frozen candidate order/smaller alpha.
    for name, _ in candidates:
        for alpha in grid[grid > 0]:
            prediction = (1 - alpha) * reference + alpha * oof[name]
            error = float(np.abs(labels - prediction).sum())
            if error < best_error:
                best_error, best_name, best_alpha = error, name, float(alpha)
    full_mask = np.ones(len(labels), dtype=bool)
    base = fit_predict("reference", reference_factory, full_mask, evaluation_frame, "outer_refit", None)
    prediction = base.copy()
    if best_name is not None:
        member = fit_predict(best_name, candidate_factories[best_name], full_mask,
                             evaluation_frame, "outer_refit", None)
        prediction = (1 - best_alpha) * base + best_alpha * member
    return NestedPrediction(best_name, best_alpha, best_error, prediction, base,
                            eval_ids, tuple(ledger))


def score_prediction(evaluation_labels, result: NestedPrediction) -> dict:
    """Score only after selection/prediction, with a separate held-label API."""
    actual = _aligned_labels(evaluation_labels, result.evaluation_ids, "Evaluation")
    if (result.prediction.shape != actual.shape or result.reference_prediction.shape != actual.shape
            or not np.isfinite(result.prediction).all()
            or not np.isfinite(result.reference_prediction).all()):
        raise ValueError("Invalid prediction arrays")
    denominator = float(actual.sum())
    base = float(np.abs(actual - result.reference_prediction).sum() / denominator)
    blended = float(np.abs(actual - result.prediction).sum() / denominator)
    return {"reference_wmape": base, "blended_wmape": blended,
            "isolated_column_score_gain": 50.0 * (base - blended)}


def synthetic_probe() -> dict:
    """Executable invariance check; scores are not competition evidence."""
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression

    class Adapter:
        def __init__(self, estimator):
            self.estimator = estimator

        def fit(self, frame, labels):
            self.estimator.fit(frame[[FEATURES[0]]], labels)
            return self

        def predict(self, frame):
            return self.estimator.predict(frame[[FEATURES[0]]])

    values = np.arange(24, dtype=float)
    frame = pd.DataFrame({c: values + j for j, c in enumerate(FEATURES)})
    frame["sample_id"] = [f"synthetic-{i:02}" for i in range(24)]
    frame["spout_no"] = 1 + np.arange(24) % 2
    train, evaluation = frame.iloc[:18].copy(), frame.iloc[18:].copy()
    labels = pd.Series(100.0 + 4.0 * values, index=frame.sample_id)
    arguments = dict(inner_folds=np.arange(18) % 3,
                     reference_factory=lambda: Adapter(DummyRegressor()),
                     candidate_factories={"linear": lambda: Adapter(LinearRegression())},
                     alpha_grid=[0.0, 0.5, 1.0], max_fit_calls=8)
    first = select_and_predict(train, labels.iloc[:18], evaluation, **arguments)
    changed = labels.iloc[18:] + 50.0
    second = select_and_predict(train, labels.iloc[:18], evaluation, **arguments)
    return {"synthetic_only": True, "official_target_fits": 0,
            "outer_evaluation_rows": len(evaluation),
            "choice": {"candidate": first.candidate, "alpha": first.alpha},
            "prediction_equal_after_evaluation_label_change": bool(np.array_equal(first.prediction, second.prediction)),
            "choice_equal_after_evaluation_label_change": (first.candidate, first.alpha) == (second.candidate, second.alpha),
            "fit_ledger_equal": first.fit_ledger == second.fit_ledger,
            "fit_calls_per_run": len(first.fit_ledger),
            "original_score": score_prediction(labels.iloc[18:], first),
            "changed_label_score": score_prediction(changed, second),
            "fit_ledger": first.fit_ledger,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-output", type=Path, required=True)
    args = parser.parse_args()
    output = args.synthetic_output.resolve()
    if not output.is_relative_to((Path.cwd() / "local").resolve()):
        raise ValueError("Synthetic evidence must stay under local/")
    if output.exists():
        raise FileExistsError(output)
    result = synthetic_probe()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "fit_ledger"}, indent=2))


if __name__ == "__main__":
    main()
