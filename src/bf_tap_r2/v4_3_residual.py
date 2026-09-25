"""V4.3 strong-base time residual calibration primitives.

V4.1 calibrated residuals around a *publicly reconstructed* V34 EBM anchor.  V4.3
keeps the same residual machinery but replaces the anchor with the real frozen
V36 composition that produced the current user-reported platform best, using the
private V3.4/V3.6 development caches present in this checkout.

The module is deliberately split so that the residual learners are pure and
testable: every learner consumes an already cross-fitted base prediction and
never sees the outer validation labels.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import FEATURES

CONFIG_VERSION = "round2-v4.3-strong-base-time-residual"
TARGET = "tap_time_len"
FAMILIES = ("shrink_bins", "lightgbm_residual", "histgb_residual")


def load_v43_config(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    config = yaml.safe_load((root / "configs/round2_v4_3/experiment.yaml").read_text(encoding="utf-8"))
    if config.get("version") != CONFIG_VERSION:
        raise ValueError("Unexpected V4.3 configuration version")
    if int(config["candidate_budget"]["per_target"]) != 12:
        raise ValueError("V4.3 candidate budget must remain 12")
    if str(config["parent"]["target_in_scope"]) != TARGET:
        raise ValueError("V4.3 is frozen to the time target")
    if str(config["parent"]["iron_side"]) != "unchanged_and_out_of_scope":
        raise ValueError("V4.3 must leave the iron side untouched")
    return config


def candidate_specs(config: Mapping[str, Any], target: str = TARGET) -> list[dict[str, Any]]:
    if target != TARGET:
        raise ValueError(f"Unsupported V4.3 target: {target}")
    defaults = dict(config["fixed_corrector_defaults"])
    specs: list[dict[str, Any]] = []
    for family in FAMILIES:
        for raw in config["candidates"][family]:
            item = deepcopy(dict(raw))
            item.update({
                "family": family,
                "target": target,
                "candidate_id": f"v43-{target}-{item['name']}",
                "clip_quantile": float(defaults["correction_clip_absolute_quantile"]),
            })
            if family == "lightgbm_residual":
                item["defaults"] = deepcopy(defaults["lightgbm"])
            elif family == "histgb_residual":
                item["defaults"] = deepcopy(defaults["histgb"])
            specs.append(item)
    if len(specs) != 12 or len({row["candidate_id"] for row in specs}) != 12:
        raise AssertionError("V4.3 must produce 12 unique candidates")
    return specs


def residual_feature_frame(frame: pd.DataFrame, base_prediction: np.ndarray) -> pd.DataFrame:
    """Residual design matrix: frozen features, spout id, and the base prediction."""
    base = np.asarray(base_prediction, dtype=float)
    if base.shape != (len(frame),) or not np.isfinite(base).all():
        raise ValueError("Invalid base prediction for residual features")
    missing = [name for name in (*FEATURES, "spout_no") if name not in frame.columns]
    if missing:
        raise ValueError(f"Residual frame is missing columns: {missing}")
    out = frame.loc[:, [*FEATURES, "spout_no"]].astype(float).copy()
    out.insert(0, "base_prediction", base)
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("Residual features must be finite")
    return out


def _quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if int(bins) < 2:
        raise ValueError("At least two bins are required")
    interior = np.quantile(values, np.linspace(0.0, 1.0, int(bins) + 1)[1:-1])
    return np.r_[-np.inf, np.unique(interior), np.inf].astype(float)


def _group_stat(keys: Sequence[Any], values: np.ndarray,
                statistic: str) -> dict[Any, tuple[int, float]]:
    table = pd.DataFrame({"key": list(keys), "value": np.asarray(values, dtype=float)})
    grouped = table.groupby("key", sort=True, observed=True)["value"]
    if statistic == "mean":
        stats = grouped.mean()
    elif statistic == "median":
        stats = grouped.median()
    else:
        raise ValueError(f"Unsupported residual statistic: {statistic}")
    counts = dict(grouped.size().items())
    return {key: (int(counts[key]), float(value)) for key, value in stats.items()}


class ShrinkBinnedResidualCorrector:
    """Hierarchical base-quantile and spout/base-quantile residual shrinkage."""

    def __init__(self, *, bins: int, shrinkage: float, statistic: str, alpha: float,
                 clip_quantile: float):
        self.bins = int(bins)
        self.shrinkage = float(shrinkage)
        self.statistic = str(statistic)
        self.alpha = float(alpha)
        self.clip_quantile = float(clip_quantile)

    def fit(self, frame: pd.DataFrame, base_prediction: np.ndarray,
            target: np.ndarray) -> "ShrinkBinnedResidualCorrector":
        base = np.asarray(base_prediction, dtype=float)
        y = np.asarray(target, dtype=float)
        residual = y - base
        if residual.shape != (len(frame),) or not np.isfinite(residual).all():
            raise ValueError("Invalid shrink-binned residual target")
        self.edges_ = _quantile_edges(base, self.bins)
        bin_id = np.digitize(base, self.edges_[1:-1], right=True).astype(int)
        spout = frame["spout_no"].astype(str).to_numpy()
        self.global_ = float(np.mean(residual) if self.statistic == "mean" else np.median(residual))
        self.bin_stats_ = _group_stat(bin_id.tolist(), residual, self.statistic)
        self.local_stats_ = _group_stat(list(zip(spout.tolist(), bin_id.tolist())), residual,
                                        self.statistic)
        self.clip_ = float(np.quantile(np.abs(residual), self.clip_quantile))
        self.fit_meta_ = {
            "family": "shrink_bins",
            "bins_requested": self.bins,
            "bins_effective": int(len(self.edges_) - 1),
            "shrinkage": self.shrinkage,
            "statistic": self.statistic,
            "alpha": self.alpha,
            "clip": self.clip_,
            "residual_source": "nested_cross_fitted_parent_only",
        }
        return self

    def predict_correction(self, frame: pd.DataFrame, base_prediction: np.ndarray) -> np.ndarray:
        if not hasattr(self, "edges_"):
            raise RuntimeError("Residual corrector must be fitted before prediction")
        base = np.asarray(base_prediction, dtype=float)
        bin_id = np.digitize(base, self.edges_[1:-1], right=True).astype(int)
        spout = frame["spout_no"].astype(str).to_numpy()
        correction = np.empty(len(frame), dtype=float)
        for index, (spout_value, bin_value) in enumerate(zip(spout, bin_id)):
            bin_count, bin_stat = self.bin_stats_.get(int(bin_value), (0, self.global_))
            parent = (bin_count * bin_stat + self.shrinkage * self.global_) / (bin_count + self.shrinkage)
            local_count, local_stat = self.local_stats_.get((str(spout_value), int(bin_value)),
                                                            (0, parent))
            correction[index] = (
                local_count * local_stat + self.shrinkage * parent
            ) / (local_count + self.shrinkage)
        correction = np.clip(self.alpha * correction, -self.clip_, self.clip_)
        if correction.shape != (len(frame),) or not np.isfinite(correction).all():
            raise ValueError("Invalid shrink-binned correction")
        return correction


class TreeResidualCorrector:
    """LightGBM or histogram-gradient residual learner with fixed shrinkage."""

    def __init__(self, spec: Mapping[str, Any]):
        self.spec = deepcopy(dict(spec))
        self.family = str(self.spec["family"])
        self.alpha = float(self.spec["alpha"])
        self.clip_quantile = float(self.spec["clip_quantile"])

    def fit(self, frame: pd.DataFrame, base_prediction: np.ndarray,
            target: np.ndarray) -> "TreeResidualCorrector":
        x = residual_feature_frame(frame, base_prediction)
        residual = np.asarray(target, dtype=float) - np.asarray(base_prediction, dtype=float)
        if self.family == "lightgbm_residual":
            params = deepcopy(dict(self.spec["defaults"]))
            params.update({
                "objective": str(self.spec["objective"]),
                "num_leaves": int(self.spec["num_leaves"]),
            })
            self.model_ = LGBMRegressor(**params)
        elif self.family == "histgb_residual":
            params = deepcopy(dict(self.spec["defaults"]))
            params.update({
                "loss": str(self.spec["loss"]),
                "max_leaf_nodes": int(self.spec["max_leaf_nodes"]),
            })
            self.model_ = HistGradientBoostingRegressor(**params)
        else:
            raise ValueError(f"Unsupported tree residual family: {self.family}")
        self.model_.fit(x, residual)
        self.columns_ = tuple(x.columns)
        self.clip_ = float(np.quantile(np.abs(residual), self.clip_quantile))
        self.fit_meta_ = {
            "family": self.family,
            "alpha": self.alpha,
            "clip": self.clip_,
            "columns": list(self.columns_),
            "residual_source": "nested_cross_fitted_parent_only",
        }
        return self

    def predict_correction(self, frame: pd.DataFrame, base_prediction: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Residual corrector must be fitted before prediction")
        x = residual_feature_frame(frame, base_prediction)
        if tuple(x.columns) != self.columns_:
            raise ValueError("Residual feature order mismatch")
        correction = self.alpha * np.asarray(self.model_.predict(x), dtype=float)
        correction = np.clip(correction, -self.clip_, self.clip_)
        if correction.shape != (len(frame),) or not np.isfinite(correction).all():
            raise ValueError("Invalid tree residual correction")
        return correction


def make_corrector(spec: Mapping[str, Any]) -> ShrinkBinnedResidualCorrector | TreeResidualCorrector:
    family = str(spec["family"])
    if family == "shrink_bins":
        return ShrinkBinnedResidualCorrector(
            bins=int(spec["bins"]),
            shrinkage=float(spec["shrinkage"]),
            statistic=str(spec["statistic"]),
            alpha=float(spec["alpha"]),
            clip_quantile=float(spec["clip_quantile"]),
        )
    if family in {"lightgbm_residual", "histgb_residual"}:
        return TreeResidualCorrector(spec)
    raise ValueError(f"Unsupported V4.3 candidate family: {family}")


@dataclass
class FoldEvaluation:
    """One outer fold's candidate predictions against the recorded parent base."""

    parent_prediction: np.ndarray
    candidate_predictions: dict[str, np.ndarray]
    parent_wmape: float
    candidate_wmape: dict[str, float]
    package_delta_single: dict[str, float]


def evaluate_candidate_corrections(
    frame: pd.DataFrame,
    target: np.ndarray,
    parent_prediction: np.ndarray,
    correction: np.ndarray,
    *,
    clip_at_zero: bool = True,
) -> np.ndarray:
    """Apply a correction to the parent prediction with the export clipping rule."""
    base = np.asarray(parent_prediction, dtype=float)
    values = base + np.asarray(correction, dtype=float)
    if clip_at_zero:
        values = np.maximum(values, 0.0)
    if values.shape != np.asarray(target).shape or not np.isfinite(values).all():
        raise ValueError("Invalid corrected prediction")
    return values


__all__ = [
    "CONFIG_VERSION",
    "FAMILIES",
    "FoldEvaluation",
    "ShrinkBinnedResidualCorrector",
    "TARGET",
    "TreeResidualCorrector",
    "candidate_specs",
    "evaluate_candidate_corrections",
    "load_v43_config",
    "make_corrector",
    "residual_feature_frame",
]
