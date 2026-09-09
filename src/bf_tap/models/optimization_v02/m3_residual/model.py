from __future__ import annotations

import numpy as np
import pandas as pd

from ....exceptions import ContractError
from ...baseline import TARGETS
from ...sanity import MedianControls
from ..estimator import FrozenCandidate, validate_training


def expanding_anchors(samples, *, min_group_count=20):
    """Past completed training labels only; unknown initial anchor remains NaN."""
    if not isinstance(min_group_count, int) or min_group_count < 1:
        raise ContractError("minimum group count must be a positive integer")
    required = {"sample_id", "reference_time", "label_available_at", "spout_no", *TARGETS}
    if required - set(samples) or samples["sample_id"].duplicated().any():
        raise ContractError("anchor samples need complete metadata and unique IDs")
    for col in ("reference_time", "label_available_at"):
        if samples[col].isna().any() or samples[col].dt.tz is None:
            raise ContractError("anchor times must be complete and timezone-aware")
    anchors = pd.DataFrame(np.nan, index=samples.index, columns=list(TARGETS))
    evidence = []
    for idx, row in samples.iterrows():
        visible = samples.loc[
            (samples["reference_time"] < row["reference_time"])
            & (samples["label_available_at"] <= row["reference_time"])
            & (samples["sample_id"] != row["sample_id"])
        ]
        group = visible.loc[visible["spout_no"].astype(str) == str(row["spout_no"])]
        use_group = len(group) >= min_group_count
        chosen = group if use_group else visible
        if not chosen.empty:
            anchors.loc[idx] = chosen[list(TARGETS)].median()
        evidence.append({
            "sample_id": str(row["sample_id"]), "visible_rows": len(visible),
            "spout_rows": len(group), "anchor_source": "B1" if use_group else (
                "B0" if len(visible) else "WARMUP"),
            "max_label_available_at": str(visible["label_available_at"].max()) if len(visible) else None,
        })
    return anchors, pd.DataFrame(evidence, index=samples.index)


class ResidualModel(FrozenCandidate):
    family = "M3_RESIDUAL"

    def __init__(self, min_group_count=20):
        if not isinstance(min_group_count, int) or min_group_count < 1:
            raise ContractError("minimum group count must be positive")
        self.options = {"min_group_count": min_group_count}

    def fit(self, samples, X, cutoff):
        order = validate_training(samples, X, cutoff)
        data = samples.loc[order].reset_index(drop=True)
        features = X.loc[order].reset_index(drop=True)
        anchors, self.anchor_audit_ = expanding_anchors(data, **self.options)
        keep = anchors.notna().all(axis=1)
        if not keep.any():
            raise ContractError("no residual training rows remain after anchor warmup")
        controls = MedianControls(**self.options).fit(data)
        self.state_ = {
            "fit_cutoff": str(cutoff), "fit_rows": int(keep.sum()),
            "warmup_rows": int((~keep).sum()),
            "global_anchor": controls.global_, "by_spout_anchor": controls.by_spout_,
        }
        residual = data[list(TARGETS)] - anchors
        return self._fit(features.loc[keep].reset_index(drop=True),
                         residual.loc[keep].reset_index(drop=True))

    def predict_raw(self, samples, X):
        if not samples.index.equals(X.index):
            raise ContractError("prediction samples/features not aligned")
        if (samples["reference_time"] < pd.Timestamp(self.state_["fit_cutoff"])).any():
            raise ContractError("prediction precedes frozen anchor cutoff")
        result = self._raw(X)
        for target in TARGETS:
            result[f"pred_{target}"] += np.asarray([
                self.state_["by_spout_anchor"].get(str(spout), self.state_["global_anchor"])[target]
                for spout in samples["spout_no"]
            ])
        return result

    def predict(self, samples, X):
        return self.predict_raw(samples, X).clip(lower=0.0)
