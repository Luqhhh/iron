from __future__ import annotations

import numpy as np
import pandas as pd

from ....exceptions import ContractError
from ...baseline import TARGETS
from ..estimator import FrozenCandidate, validate_training


def recency_weights(reference_times, cutoff, half_life_days):
    cutoff = pd.Timestamp(cutoff)
    if (cutoff.tzinfo is None or not np.isfinite(half_life_days)
            or half_life_days <= 0 or len(reference_times) == 0):
        raise ContractError("recency requires a positive half-life and aware cutoff")
    if reference_times.isna().any() or reference_times.dt.tz is None:
        raise ContractError("reference times must be complete and timezone-aware")
    age = (cutoff - reference_times).dt.total_seconds() / 86400
    if (age <= 0).any():
        raise ContractError("recency rows must precede cutoff")
    weights = np.exp2(-age / half_life_days)
    if (weights <= 0).any() or not np.isfinite(weights).all():
        raise ContractError("recency weights underflow or are nonfinite")
    return weights / weights.mean()


class RecencyModel(FrozenCandidate):
    family = "M2_RECENCY"

    def __init__(self, half_life_days=60):
        if not np.isfinite(half_life_days) or half_life_days <= 0:
            raise ContractError("half-life must be positive and finite")
        self.options = {"half_life_days": half_life_days}

    def fit(self, samples, X, cutoff):
        order = validate_training(samples, X, cutoff)
        data = samples.loc[order].reset_index(drop=True)
        features = X.loc[order].reset_index(drop=True)
        weights = recency_weights(data["reference_time"], cutoff, self.options["half_life_days"])
        self.state_ = {
            "fit_cutoff": str(cutoff), "fit_rows": len(data),
            "weight_min": float(weights.min()), "weight_max": float(weights.max()),
            "weight_mean": float(weights.mean()),
            "effective_sample_size": float(weights.sum() ** 2 / (weights ** 2).sum()),
        }
        return self._fit(features, data[list(TARGETS)], weights)

    def predict_raw(self, samples, X):
        if not samples.index.equals(X.index):
            raise ContractError("prediction samples/features not aligned")
        if (samples["reference_time"] < pd.Timestamp(self.state_["fit_cutoff"])).any():
            raise ContractError("prediction rows precede fitting cutoff")
        return self._raw(X)

    def predict(self, samples, X):
        return self.predict_raw(samples, X).clip(lower=0.0)
