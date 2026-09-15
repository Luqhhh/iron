"""Causal, state-local calibration for the iron/time structural correction.

The existing V1 correction fits one global LAD coefficient for each target.  This
module keeps the same correction direction, but lets the coefficient vary with
the observable furnace state.  A local estimate is always shrunk toward the
global V1 coefficient and is suppressed for out-of-distribution states.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from ..exceptions import ContractError
from .structural import INPUT, PRED, directions, lad_coefficient, select_oof

TARGETS = ("tap_iron", "tap_time_len")


def _weighted_lower_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise ContractError("state-local LAD has no positive finite weight")
    values, weights = values[valid], weights[valid]
    order = np.argsort(values, kind="mergesort")
    position = np.searchsorted(
        np.cumsum(weights[order]), 0.5 * weights.sum(), side="left"
    )
    return float(values[order[position]])


def _row_distances(
    left: np.ndarray, right: np.ndarray, *, chunk_size: int = 256
) -> np.ndarray:
    """Return Euclidean distances without allocating one giant 3-D tensor."""
    parts: list[np.ndarray] = []
    for start in range(0, len(left), chunk_size):
        block = left[start : start + chunk_size]
        squared = (
            np.square(block).sum(axis=1)[:, None]
            + np.square(right).sum(axis=1)[None, :]
            - 2.0 * block @ right.T
        )
        parts.append(np.sqrt(np.maximum(squared, 0.0)))
    return np.vstack(parts)


@dataclass(frozen=True)
class AnalogDiagnostics:
    alpha: np.ndarray
    reliability: np.ndarray
    radius: np.ndarray
    effective_neighbors: np.ndarray


@dataclass(frozen=True)
class AlphaModelDiagnostics:
    alpha: np.ndarray
    model_alpha: np.ndarray
    beta: np.ndarray


class StateAnalogCalibrator:
    """Nearest-state LAD with global shrinkage and an OOD reliability gate."""

    def __init__(
        self,
        state_columns: tuple[str, ...],
        *,
        neighbors: int = 64,
        prior_strength: float = 32.0,
        ood_quantile: float = 0.9,
        rate_floor: float = 1.0e-6,
    ):
        if not state_columns or len(set(state_columns)) != len(state_columns):
            raise ContractError("state columns must be non-empty and unique")
        if neighbors < 8 or prior_strength <= 0:
            raise ContractError("state analog neighbor/prior settings are invalid")
        if not 0.5 < ood_quantile < 1.0:
            raise ContractError("state analog OOD quantile must be in (0.5, 1)")
        self.state_columns = state_columns
        self.neighbors = int(neighbors)
        self.prior_strength = float(prior_strength)
        self.ood_quantile = float(ood_quantile)
        self.rate_floor = float(rate_floor)

    def _state_values(self, frame: pd.DataFrame, *, fitting: bool) -> np.ndarray:
        missing = set(self.state_columns) - set(frame)
        if missing:
            raise ContractError(f"state analog inputs missing columns: {sorted(missing)}")
        values = frame.loc[:, self.state_columns].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ContractError("state analog inputs contain infinity")
        if fitting:
            self.center_ = np.nanmedian(values, axis=0)
            q25, q75 = np.nanquantile(values, [0.25, 0.75], axis=0)
            scale = q75 - q25
            fallback = np.nanstd(values, axis=0)
            scale = np.where(scale > 1.0e-12, scale, fallback)
            keep = np.isfinite(self.center_) & np.isfinite(scale) & (scale > 1.0e-12)
            if not keep.any():
                raise ContractError("state analog inputs contain no varying finite feature")
            self.keep_ = keep
            self.center_ = self.center_[keep]
            self.scale_ = scale[keep]
        values = values[:, self.keep_]
        values = np.where(np.isfinite(values), values, self.center_)
        return (values - self.center_) / self.scale_

    def fit(self, oof: pd.DataFrame, cutoff: pd.Timestamp) -> "StateAnalogCalibrator":
        selected = select_oof(oof, cutoff, minimum=max(100, self.neighbors))
        self.bank_ = self._state_values(selected, fitting=True)
        self.bank_ids_ = selected["sample_id"].astype(str).to_numpy()
        delta, _ = directions(selected[INPUT], self.rate_floor)
        self.delta_ = delta
        self.ratio_ = np.zeros_like(delta)
        self.direction_weight_ = np.abs(delta)
        self.global_alpha_ = np.asarray(
            [
                lad_coefficient(selected[target], selected[pred], delta[:, index])
                for index, (target, pred) in enumerate(zip(TARGETS, PRED))
            ],
            dtype=float,
        )
        for index, (target, pred) in enumerate(zip(TARGETS, PRED)):
            usable = self.direction_weight_[:, index] > 1.0e-12
            self.ratio_[usable, index] = (
                selected.loc[usable, target].to_numpy(dtype=float)
                - selected.loc[usable, pred].to_numpy(dtype=float)
            ) / delta[usable, index]

        # A leave-one-out radius defines what "familiar" means using training
        # states alone.  Prediction-time radii beyond it progressively close the
        # local gate and recover the globally validated V1 correction.
        distances = _row_distances(self.bank_, self.bank_)
        np.fill_diagonal(distances, np.inf)
        k = min(self.neighbors, len(selected) - 1)
        if k < 8:
            raise ContractError("state analog OOF bank is too small")
        radii = np.partition(distances, k - 1, axis=1)[:, k - 1]
        self.reference_radius_ = float(np.quantile(radii, self.ood_quantile))
        if not np.isfinite(self.reference_radius_) or self.reference_radius_ <= 0:
            raise ContractError("state analog reference radius is invalid")
        return self

    def coefficients(self, states: pd.DataFrame) -> AnalogDiagnostics:
        if not hasattr(self, "bank_"):
            raise ContractError("state analog calibrator is not fitted")
        query = self._state_values(states, fitting=False)
        distances = _row_distances(query, self.bank_)
        k = min(self.neighbors, len(self.bank_))
        nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
        row = np.arange(len(query))[:, None]
        local_distance = distances[row, nearest]
        radius = np.maximum(local_distance.max(axis=1), 1.0e-12)
        relative = np.clip(local_distance / radius[:, None], 0.0, 1.0)
        kernel = np.square(1.0 - np.power(relative, 3.0))
        kernel_sum = kernel.sum(axis=1)
        effective = np.square(kernel_sum) / np.maximum(
            np.square(kernel).sum(axis=1), 1.0e-12
        )
        evidence_gate = effective / (effective + self.prior_strength)
        ood_gate = np.minimum(1.0, self.reference_radius_ / radius) ** 2
        reliability = evidence_gate * ood_gate

        alpha = np.tile(self.global_alpha_, (len(query), 1))
        for i in range(len(query)):
            for target_index in range(2):
                indexes = nearest[i]
                weights = kernel[i] * self.direction_weight_[indexes, target_index]
                if (weights > 0).sum() < 4:
                    continue
                local = np.clip(
                    _weighted_lower_median(self.ratio_[indexes, target_index], weights),
                    0.0,
                    1.0,
                )
                alpha[i, target_index] += reliability[i] * (
                    local - self.global_alpha_[target_index]
                )
        return AnalogDiagnostics(alpha, reliability, radius, effective)

    def predict(
        self, predictions: pd.DataFrame, states: pd.DataFrame
    ) -> tuple[pd.DataFrame, AnalogDiagnostics]:
        if len(predictions) != len(states) or not predictions.index.equals(states.index):
            raise ContractError("state analog predictions and states are misaligned")
        delta, _ = directions(predictions[INPUT], self.rate_floor)
        diagnostics = self.coefficients(states)
        values = np.maximum(
            0.0,
            predictions[PRED].to_numpy(dtype=float) + diagnostics.alpha * delta,
        )
        if not np.isfinite(values).all():
            raise ContractError("state analog prediction is nonfinite")
        result = pd.DataFrame(values, columns=PRED, index=predictions.index)
        result.insert(0, "sample_id", predictions["sample_id"].astype(str).to_numpy())
        return result, diagnostics


class StateAlphaCalibrator:
    """Learn a varying structural coefficient, with a causal last-month gate.

    For a structural direction ``d`` the constrained MAE problem is equivalent
    to predicting ``clip((y - base) / d, 0, 1)`` with sample weight ``abs(d)``.
    The final local-vs-global strength is itself selected by exact LAD on the
    latest available OOF month, before the outer prediction cutoff.
    """

    DEFAULT_PARAMETERS = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 200,
        "depth": 2,
        "learning_rate": 0.03,
        "l2_leaf_reg": 20.0,
        "random_seed": 2026,
        "task_type": "CPU",
        "thread_count": 8,
        "bootstrap_type": "No",
        "random_strength": 0.0,
        "rsm": 1.0,
        "boosting_type": "Plain",
        "has_time": True,
        "nan_mode": "Min",
        "use_best_model": False,
        "allow_writing_files": False,
        "verbose": False,
    }

    def __init__(
        self,
        state_columns: tuple[str, ...],
        *,
        parameters: dict | None = None,
        rate_floor: float = 1.0e-6,
    ):
        if not state_columns or len(set(state_columns)) != len(state_columns):
            raise ContractError("state alpha columns must be non-empty and unique")
        self.state_columns = state_columns
        self.parameters = dict(parameters or self.DEFAULT_PARAMETERS)
        self.rate_floor = float(rate_floor)

    def _X(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.state_columns) - set(frame)
        if missing:
            raise ContractError(f"state alpha inputs missing columns: {sorted(missing)}")
        result = frame.loc[:, self.state_columns].astype(float)
        if np.isinf(result.to_numpy()).any():
            raise ContractError("state alpha inputs contain infinity")
        return result

    def _fit_target(
        self, frame: pd.DataFrame, delta: np.ndarray, target_index: int
    ) -> CatBoostRegressor:
        target, pred = TARGETS[target_index], PRED[target_index]
        weight = np.abs(delta[:, target_index])
        usable = weight > 1.0e-12
        if usable.sum() < 50:
            raise ContractError("state alpha has insufficient structural direction rows")
        ratio = np.clip(
            (
                frame.loc[usable, target].to_numpy(dtype=float)
                - frame.loc[usable, pred].to_numpy(dtype=float)
            )
            / delta[usable, target_index],
            0.0,
            1.0,
        )
        model = CatBoostRegressor(**self.parameters)
        model.fit(self._X(frame).loc[usable], ratio, sample_weight=weight[usable])
        return model

    @staticmethod
    def _global(frame: pd.DataFrame, delta: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                lad_coefficient(frame[target], frame[pred], delta[:, index])
                for index, (target, pred) in enumerate(zip(TARGETS, PRED))
            ],
            dtype=float,
        )

    def fit(self, oof: pd.DataFrame, cutoff: pd.Timestamp) -> "StateAlphaCalibrator":
        selected = select_oof(oof, cutoff, minimum=300)
        delta, _ = directions(selected[INPUT], self.rate_floor)
        self.global_alpha_ = self._global(selected, delta)

        local_time = selected.reference_time.dt.tz_convert("Asia/Shanghai")
        latest = local_time.max()
        validation_start = pd.Timestamp(
            year=latest.year, month=latest.month, day=1, tz="Asia/Shanghai"
        )
        inner_mask = selected.reference_time < validation_start
        validation_mask = ~inner_mask
        if inner_mask.sum() < 200 or validation_mask.sum() < 100:
            raise ContractError("state alpha causal gate lacks an inner temporal split")
        inner = selected.loc[inner_mask].copy()
        validation = selected.loc[validation_mask].copy()
        inner_delta, _ = directions(inner[INPUT], self.rate_floor)
        validation_delta, _ = directions(validation[INPUT], self.rate_floor)
        inner_global = self._global(inner, inner_delta)
        beta = np.zeros(2, dtype=float)
        for target_index, (target, pred) in enumerate(zip(TARGETS, PRED)):
            inner_model = self._fit_target(inner, inner_delta, target_index)
            local = np.clip(
                inner_model.predict(self._X(validation)), 0.0, 1.0
            )
            globally_corrected = (
                validation[pred].to_numpy(dtype=float)
                + inner_global[target_index] * validation_delta[:, target_index]
            )
            local_direction = (
                local - inner_global[target_index]
            ) * validation_delta[:, target_index]
            beta[target_index] = lad_coefficient(
                validation[target], globally_corrected, local_direction
            )

        self.beta_ = beta
        self.models_ = [self._fit_target(selected, delta, index) for index in range(2)]
        self.gate_validation_ = {
            "start": str(validation_start),
            "train_rows": int(inner_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "inner_global_alpha": inner_global.tolist(),
            "beta": beta.tolist(),
        }
        return self

    def coefficients(self, states: pd.DataFrame) -> AlphaModelDiagnostics:
        if not hasattr(self, "models_"):
            raise ContractError("state alpha calibrator is not fitted")
        X = self._X(states)
        model_alpha = np.column_stack(
            [np.clip(model.predict(X), 0.0, 1.0) for model in self.models_]
        )
        alpha = self.global_alpha_ + self.beta_ * (
            model_alpha - self.global_alpha_
        )
        alpha = np.clip(alpha, 0.0, 1.0)
        return AlphaModelDiagnostics(alpha, model_alpha, self.beta_.copy())

    def predict(
        self, predictions: pd.DataFrame, states: pd.DataFrame
    ) -> tuple[pd.DataFrame, AlphaModelDiagnostics]:
        if len(predictions) != len(states) or not predictions.index.equals(states.index):
            raise ContractError("state alpha predictions and states are misaligned")
        delta, _ = directions(predictions[INPUT], self.rate_floor)
        diagnostics = self.coefficients(states)
        values = np.maximum(
            0.0,
            predictions[PRED].to_numpy(dtype=float) + diagnostics.alpha * delta,
        )
        if not np.isfinite(values).all():
            raise ContractError("state alpha prediction is nonfinite")
        result = pd.DataFrame(values, columns=PRED, index=predictions.index)
        result.insert(0, "sample_id", predictions["sample_id"].astype(str).to_numpy())
        return result, diagnostics
