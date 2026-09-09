"""Shared estimator mechanics; frozen baseline files are never modified."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..baseline import DualTargetBaseline, FROZEN_PARAMETERS, TARGETS
from ...artifacts import atomic_write_json, file_sha256, stable_digest
from ...exceptions import ContractError


def validate_training(samples, X, cutoff):
    cutoff = pd.Timestamp(cutoff)
    required = {"sample_id", "tap_no", "spout_no", "reference_time",
                "label_available_at", *TARGETS}
    if required - set(samples) or not samples.index.equals(X.index):
        raise ContractError("training metadata and features must be complete and aligned")
    if cutoff.tzinfo is None or samples.empty or samples["sample_id"].duplicated().any():
        raise ContractError("training requires timezone-aware cutoff and unique nonempty samples")
    for col in ("reference_time", "label_available_at"):
        if samples[col].isna().any() or samples[col].dt.tz is None:
            raise ContractError("training times must be complete and timezone-aware")
    if (samples["reference_time"] >= cutoff).any() or (samples["label_available_at"] > cutoff).any():
        raise ContractError("training rows or labels follow fit cutoff")
    values = samples[list(TARGETS)].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ContractError("original targets must be finite and nonnegative")
    return samples.sort_values(["reference_time", "tap_no", "sample_id"], kind="stable").index


class FrozenCandidate:
    """Dual-target MAE estimator with explicit local candidate bundle semantics."""

    def _fit(self, X, y, weights=None):
        if X.empty or len(X) != len(y) or not X.index.equals(y.index):
            raise ContractError("estimator inputs must be aligned and nonempty")
        if X.columns.duplicated().any() or "spout_no" not in X:
            raise ContractError("invalid feature columns")
        if X["spout_no"].isna().any() or X["spout_no"].nunique() > 64:
            raise ContractError("invalid categorical values")
        if not np.isfinite(y[list(TARGETS)].to_numpy(float)).all():
            raise ContractError("estimator targets must be finite")
        if weights is not None:
            if (len(weights) != len(X) or not np.isfinite(weights).all()
                    or (np.asarray(weights) <= 0).any()):
                raise ContractError("sample weights must be aligned, finite and positive")
        self.schema_ = [(c, str(X[c].dtype)) for c in X]
        self.models_ = {}
        for target in TARGETS:
            model = DualTargetBaseline._catboost_regressor()(**FROZEN_PARAMETERS)
            model.fit(X, y[target], cat_features=["spout_no"], sample_weight=weights)
            self.models_[target] = model
        return self

    def _raw(self, X):
        if [(c, str(X[c].dtype)) for c in X] != self.schema_:
            raise ContractError("prediction feature schema differs from bundle")
        result = pd.DataFrame(
            {f"pred_{t}": self.models_[t].predict(X) for t in TARGETS}, index=X.index)
        if not np.isfinite(result.to_numpy()).all():
            raise ContractError("nonfinite model predictions")
        return result

    def save(self, directory, *, metadata, history_snapshot=None):
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for target, model in self.models_.items():
            name = f"{target}.cbm"
            model.save_model(destination / name)
            hashes[name] = file_sha256(destination / name)
        if history_snapshot is not None:
            history_snapshot.to_csv(destination / "history_snapshot.csv", index=False)
            hashes["history_snapshot.csv"] = file_sha256(destination / "history_snapshot.csv")
        payload = {
            "schema_version": "local-optimization-v02-model-1",
            "family": self.family, "parameters": FROZEN_PARAMETERS,
            "options": self.options, "feature_schema": self.schema_,
            "state": self.state_, "metadata": metadata,
            "components": hashes, "postprocess": {"lower_bound": 0.0},
        }
        atomic_write_json(destination / "bundle.json",
                          {**payload, "payload_sha256": stable_digest(payload)})

    @classmethod
    def load(cls, directory):
        root = Path(directory)
        payload = json.loads((root / "bundle.json").read_text())
        digest = payload.pop("payload_sha256")
        if stable_digest(payload) != digest:
            raise ContractError("bundle metadata digest mismatch")
        if (payload["schema_version"] != "local-optimization-v02-model-1"
                or payload["parameters"] != FROZEN_PARAMETERS
                or payload["family"] != cls.family
                or payload["postprocess"] != {"lower_bound": 0.0}
                or not {f"{t}.cbm" for t in TARGETS} <= set(payload["components"])
                or set(payload["components"]) - {f"{t}.cbm" for t in TARGETS} - {"history_snapshot.csv"}):
            raise ContractError("candidate bundle contract mismatch")
        instance = cls(**payload["options"])
        instance.schema_ = [tuple(row) for row in payload["feature_schema"]]
        instance.state_ = payload["state"]
        instance.metadata_ = payload["metadata"]
        for name, expected in payload["components"].items():
            if file_sha256(root / name) != expected:
                raise ContractError("bundle component digest mismatch")
        instance.models_ = {}
        for target in TARGETS:
            name = f"{target}.cbm"
            if file_sha256(root / name) != payload["components"][name]:
                raise ContractError("bundle model component digest mismatch")
            model = DualTargetBaseline._catboost_regressor()()
            model.load_model(root / name)
            instance.models_[target] = model
        return instance
