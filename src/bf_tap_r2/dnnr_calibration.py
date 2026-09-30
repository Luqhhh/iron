"""Matched inner calibration and fresh outer refits; no outer query labels."""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .data import TARGETS
from .dnnr_model import ARMS, Regressor, Settings, validate_frame
from .v3_4_bags import group_safe_inner_folds
from .v7_periodic import digest, file_hash


@dataclass
class Pair:
    inner: dict
    outer: dict
    receipt: dict

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for role, members in (("inner", self.inner), ("outer", self.outer)):
            for arm, model in members.items():
                path = directory / f"{role}-{arm}.npz"
                hashes[path.name] = model.save(path)
        path = directory / "receipt.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"format": "dnnr-matched-pair-v1", "receipt": self.receipt,
                       "model_hashes": hashes}, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        hashes[path.name] = file_hash(path)
        return hashes


def fit_pair(frame, y, target, *, settings=None, observer=None):
    """Calibration may select epoch 0 or 1; all outer states start fresh.

    Three inner plus three outer estimators, four derivative banks, one or two
    metric epochs. The learned inner state ALWAYS records epoch 1; its initial
    endpoint is the matching fixed Taylor arm. No outer score enters selection.
    """
    validate_frame(frame)
    y = np.asarray(y, dtype=np.float64)
    if target not in TARGETS or y.shape != (len(frame),) or not np.isfinite(y).all():
        raise ValueError("One recognized target with matching finite training labels required")
    settings = settings or Settings()
    observer = observer or (lambda _kind, _metadata: None)
    assignment = group_safe_inner_folds(frame, n_splits=5, seed=42)
    folds = assignment["fold"]
    mask = folds != 0
    if not mask.any() or mask.all():
        raise ValueError("Nonempty disjoint fitting/calibration parts required")
    fitting, calibration = frame.loc[mask].reset_index(drop=True), frame.loc[~mask].reset_index(drop=True)
    inner, outer = {}, {}

    def reserve(role, arm):
        return lambda kind, meta: observer(kind, {**meta, "role": role, "target": target, "arm": arm})

    for arm in ARMS:
        inner[arm] = Regressor(arm, settings).fit(fitting, y[mask],
            metric_epochs=int(arm == "DNNR_LEARNED"), observer=reserve("inner", arm))
    predictions = {arm: model.predict(calibration) for arm, model in inner.items()}
    maes = {arm: float(np.abs(value-y[~mask]).mean()) for arm, value in predictions.items()}
    # Initial scale wins exact ties. This rule is fixed before official fits.
    selected = int(maes["DNNR_LEARNED"] < maes["DNNR_FIXED"])
    for arm in ARMS:
        outer[arm] = Regressor(arm, settings).fit(frame, y,
            metric_epochs=selected if arm == "DNNR_LEARNED" else 0,
            observer=reserve("outer", arm))
    if not selected:
        for key in ("x", "y", "scale", "derivative_graph", "derivatives"):
            if not np.array_equal(outer["DNNR_FIXED"].arrays_[key], outer["DNNR_LEARNED"].arrays_[key]):
                raise ValueError("Epoch-zero learned arm must reproduce matching fixed Taylor state")
    receipt = {"target": target, "fitting_ids": fitting.sample_id.astype(str).tolist(),
        "calibration_ids": calibration.sample_id.astype(str).tolist(),
        "outer_ids": frame.sample_id.astype(str).tolist(), "inner_folds": folds.tolist(),
        "inner_seed": 42, "held_fold": 0, "inner_group_hash": assignment["group_hash"],
        "inner_fold_hash": assignment["inner_fold_hash"], "selected_metric_epochs": selected,
        "selection": "lower_raw_calibration_MAE_initial_wins_ties", "calibration_maes": maes,
        "calibration_predictions": {arm: value.tolist() for arm, value in predictions.items()},
        "calibration_prediction_digests": {arm: digest(value.tolist()) for arm, value in predictions.items()},
        "estimator_runs": 6, "derivative_bank_runs": 4, "derivative_local_solutions": 2*len(fitting)+2*len(frame),
        "metric_epoch_runs": 1+selected, "metric_updates": len(fitting)+selected*len(frame),
        "metric_local_solutions": len(fitting)+selected*len(frame),
        "total_fit_seconds": sum(m.elapsed_seconds_ for models in (inner, outer) for m in models.values()),
        "outer_refit": "fresh_unit_scale_relearn_selected_epoch_count", "outer_query_labels": "absent"}
    return Pair(inner, outer, receipt)
