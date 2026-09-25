"""V4.1 public-anchor orthogonal residual calibration primitives.

The module deliberately separates the strong base prediction from residual
learning.  Every residual target must be formed from predictions produced for
rows that were excluded from the corresponding base-model fit.  The public
anchor is reconstructible without the private V3.4/V3.6 cache tree; it is not
claimed to be an exact replay of V34_A or V36.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import FEATURES
from .v3_4_bags import group_safe_inner_folds
from .v3_4_models import V34EBMRegressor
from .v3_4_sampler import build_ebm_boundary_trials, load_v34_config, mark_duplicate_slots

CONFIG_VERSION = "round2-v4.1-orthogonal-residual-search"
TARGETS = ("tap_iron", "tap_time_len")


def load_v41_config(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    path = root / "configs/round2_v4_1/experiment.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("version") != CONFIG_VERSION:
        raise ValueError("Unexpected V4.1 configuration version")
    if int(config["candidate_budget"]["per_target"]) != 12:
        raise ValueError("V4.1 candidate budget must remain 12 per target")
    if int(config["candidate_budget"]["total"]) != 24:
        raise ValueError("V4.1 total candidate budget must remain 24")
    return config


def candidate_specs(config: Mapping[str, Any], target: str) -> list[dict[str, Any]]:
    if target not in TARGETS:
        raise ValueError(f"Unsupported target: {target}")
    defaults = dict(config["fixed_corrector_defaults"])
    specs: list[dict[str, Any]] = []
    for family in ("shrink_bins", "lightgbm_residual", "histgb_residual"):
        for raw in config["candidates"][family]:
            item = deepcopy(dict(raw))
            item.update({
                "family": family,
                "target": target,
                "candidate_id": f"v41-{target}-{item['name']}",
                "clip_quantile": float(defaults["correction_clip_absolute_quantile"]),
            })
            if family == "lightgbm_residual":
                item["defaults"] = deepcopy(defaults["lightgbm"])
            elif family == "histgb_residual":
                item["defaults"] = deepcopy(defaults["histgb"])
            specs.append(item)
    if len(specs) != 12 or len({row["candidate_id"] for row in specs}) != 12:
        raise AssertionError("V4.1 must produce 12 unique candidates per target")
    return specs


def public_anchor_trials(root: Path | str, target: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Rebuild the frozen public V34 EBM trial dictionaries without private ledgers."""
    root = Path(root)
    config = load_v41_config(root)
    reference = dict(config["reference"][target])
    v34 = load_v34_config(root)
    # V34's public boundary order is deterministic.  The historical time
    # center was log1p; this value is frozen explicitly in the V4.1 config.
    trials = mark_duplicate_slots(
        build_ebm_boundary_trials(v34, str(config["reference"]["target_transform"]))
    )
    by_id = {str(row["trial_id"]): row for row in trials}
    requested = [str(value) for value in reference["trial_ids"]]
    missing = [trial_id for trial_id in requested if trial_id not in by_id]
    if missing:
        raise ValueError(f"Missing public V34 anchor trials: {missing}")
    selected = [deepcopy(by_id[trial_id]) for trial_id in requested]
    if any(row["target"] != target for row in selected):
        raise ValueError("Public anchor target mismatch")
    weights = np.asarray(reference["weights"], dtype=float)
    if weights.shape != (len(selected),) or not np.isfinite(weights).all():
        raise ValueError("Invalid public anchor weights")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Public anchor weights must be nonnegative and sum to one")
    return selected, weights


def cross_fitted_predictions(
    frame: pd.DataFrame,
    target: np.ndarray,
    folds: np.ndarray,
    model_factory: Callable[[], Any],
) -> tuple[np.ndarray, Any]:
    """Generic row-excluded predictions plus one full-data fitted model.

    This small primitive is independently testable with a memorizing sentinel.
    The caller is responsible for supplying a group-isolated fold vector.
    """
    y = np.asarray(target, dtype=float)
    fold = np.asarray(folds, dtype=int)
    if y.shape != (len(frame),) or fold.shape != (len(frame),):
        raise ValueError("Cross-fit frame, target, and fold shapes do not match")
    if not np.isfinite(y).all() or len(set(fold.tolist())) < 2:
        raise ValueError("Cross-fit inputs are invalid")
    prediction = np.full(len(frame), np.nan, dtype=float)
    for fold_id in sorted(set(fold.tolist())):
        validation = fold == int(fold_id)
        training = ~validation
        if not validation.any() or not training.any():
            raise ValueError("Cross-fit fold has an empty side")
        model = model_factory()
        model.fit(frame.loc[training].reset_index(drop=True), y[training])
        prediction[validation] = np.asarray(
            model.predict(frame.loc[validation].reset_index(drop=True)), dtype=float
        )
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError("Cross-fit predictions are incomplete or nonfinite")
    full = model_factory()
    full.fit(frame.reset_index(drop=True), y)
    return prediction, full


class PublicAnchorRegressor:
    """Weighted ensemble of publicly reconstructible V34 EBM recipes."""

    def __init__(self, root: Path | str, target: str):
        self.root = Path(root)
        self.target = str(target)
        self.trials, self.weights = public_anchor_trials(self.root, self.target)

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "PublicAnchorRegressor":
        y = np.asarray(target, dtype=float)
        self.models_ = [V34EBMRegressor(trial).fit(frame, y) for trial in self.trials]
        self.fit_meta_ = {
            "reference": "PUBLIC_REBUILDABLE_V34_EBM_ANCHOR",
            "target": self.target,
            "trial_ids": [str(row["trial_id"]) for row in self.trials],
            "weights": self.weights.tolist(),
            "exact_v34_a_replay": False,
            "private_cache_used": False,
        }
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise RuntimeError("Public anchor must be fitted before prediction")
        matrix = np.column_stack([model.predict(frame) for model in self.models_])
        prediction = matrix @ self.weights
        if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
            raise ValueError("Invalid public anchor prediction")
        return prediction


@dataclass
class CrossFittedAnchor:
    oof_prediction: np.ndarray
    fitted_model: PublicAnchorRegressor
    fold: np.ndarray
    group_hash: str
    fold_hash: str


def fit_cross_fitted_anchor(
    root: Path | str,
    frame: pd.DataFrame,
    target: str,
    *,
    n_splits: int,
    seed: int,
) -> CrossFittedAnchor:
    folded = group_safe_inner_folds(frame, n_splits=int(n_splits), seed=int(seed))
    y = frame[target].to_numpy(dtype=float)
    prediction, full = cross_fitted_predictions(
        frame,
        y,
        folded["fold"],
        lambda: PublicAnchorRegressor(root, target),
    )
    return CrossFittedAnchor(
        oof_prediction=prediction,
        fitted_model=full,
        fold=np.asarray(folded["fold"], dtype=int),
        group_hash=str(folded["group_hash"]),
        fold_hash=str(folded["inner_fold_hash"]),
    )


def residual_feature_frame(frame: pd.DataFrame, base_prediction: np.ndarray) -> pd.DataFrame:
    base = np.asarray(base_prediction, dtype=float)
    if base.shape != (len(frame),) or not np.isfinite(base).all():
        raise ValueError("Invalid base prediction for residual features")
    missing = [name for name in (*FEATURES, "spout_no") if name not in frame.columns]
    if missing:
        raise ValueError(f"Residual frame is missing columns: {missing}")
    out = frame.loc[:, [*FEATURES, "spout_no"]].astype(float).copy()
    out.insert(0, "anchor_prediction", base)
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("Residual features must be finite")
    return out


def _quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if int(bins) < 2:
        raise ValueError("At least two bins are required")
    interior = np.quantile(values, np.linspace(0.0, 1.0, int(bins) + 1)[1:-1])
    return np.r_[-np.inf, np.unique(interior), np.inf].astype(float)


def _group_stat(keys: Sequence[Any], values: np.ndarray, statistic: str) -> dict[Any, tuple[int, float]]:
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
        self.local_stats_ = _group_stat(list(zip(spout.tolist(), bin_id.tolist())), residual, self.statistic)
        self.clip_ = float(np.quantile(np.abs(residual), self.clip_quantile))
        self.fit_meta_ = {
            "family": "shrink_bins",
            "bins_requested": self.bins,
            "bins_effective": int(len(self.edges_) - 1),
            "shrinkage": self.shrinkage,
            "statistic": self.statistic,
            "alpha": self.alpha,
            "clip": self.clip_,
            "residual_source": "cross_fitted_anchor_only",
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
            local_count, local_stat = self.local_stats_.get((str(spout_value), int(bin_value)), (0, parent))
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
            "residual_source": "cross_fitted_anchor_only",
            "model_parameters": self.model_.get_params(),
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
    raise ValueError(f"Unsupported V4.1 candidate family: {family}")


__all__ = [
    "CONFIG_VERSION",
    "CrossFittedAnchor",
    "PublicAnchorRegressor",
    "ShrinkBinnedResidualCorrector",
    "TreeResidualCorrector",
    "candidate_specs",
    "cross_fitted_predictions",
    "fit_cross_fitted_anchor",
    "load_v41_config",
    "make_corrector",
    "public_anchor_trials",
    "residual_feature_frame",
]
