"""Fixed history-centered target transforms for optimization v0.25."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from ..artifacts import atomic_write_json, file_sha256
from ..exceptions import ContractError
from ..models.baseline import FROZEN_PARAMETERS


CANDIDATE_A = "V25I_HISTORY_CENTERED_RECENCY_IRON"
CANDIDATE_B = "V25T_HISTORY_CENTERED_QRF_TIME"
PREDICTION_COLUMNS = ["pred_tap_iron", "pred_tap_time_len"]
BASELINE_SOURCES = {0: "zero", 1: "all", 2: "spout"}


def history_baseline(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Select spout last100, then all-furnace last100, then the fixed zero fallback."""
    if target not in ("tap_iron", "tap_time_len"):
        raise ContractError("registered centered target required")
    names = {
        "spout_median": f"history__spout__{target}__last100_median",
        "spout_count": f"history__spout__{target}__last100_count",
        "all_median": f"history__all__{target}__last100_median",
        "all_count": f"history__all__{target}__last100_count",
    }
    if any(name not in frame for name in names.values()):
        raise ContractError("certified last100 baseline columns missing")
    values = {key: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
              for key, name in names.items()}
    for key in ("spout_count", "all_count"):
        count = values[key]
        if not np.isfinite(count).all() or (count < 0).any() or not np.array_equal(count, np.floor(count)):
            raise ContractError("last100 counts must be finite nonnegative integers")
    spout_valid = values["spout_count"] >= 1
    all_valid = values["all_count"] >= 1
    if (spout_valid & ~np.isfinite(values["spout_median"])).any():
        raise ContractError("positive spout history count requires finite median")
    if (all_valid & ~np.isfinite(values["all_median"])).any():
        raise ContractError("positive all-history count requires finite median")
    baseline = np.zeros(len(frame), dtype=float)
    source = np.zeros(len(frame), dtype=np.int8)
    use_all = ~spout_valid & all_valid
    baseline[use_all] = values["all_median"][use_all]
    source[use_all] = 1
    baseline[spout_valid] = values["spout_median"][spout_valid]
    source[spout_valid] = 2
    if not np.isfinite(baseline).all() or (baseline < 0).any():
        raise ContractError("history baselines must be finite nonnegative")
    return baseline, source


def baseline_audit(baseline, source, raw_target=None) -> dict:
    baseline = np.asarray(baseline, dtype=float)
    source = np.asarray(source, dtype=np.int8)
    if baseline.ndim != 1 or source.shape != baseline.shape or not set(np.unique(source)) <= set(BASELINE_SOURCES):
        raise ContractError("invalid baseline audit arrays")
    result = {
        "rows": int(len(baseline)),
        "sources": {BASELINE_SOURCES[code]: int(np.sum(source == code)) for code in BASELINE_SOURCES},
        "baseline_quantiles": {str(q): float(np.quantile(baseline, q)) for q in (0, .05, .25, .5, .75, .95, 1)},
    }
    if raw_target is not None:
        raw = np.asarray(raw_target, dtype=float)
        if raw.shape != baseline.shape or not np.isfinite(raw).all() or (raw < 0).any():
            raise ContractError("invalid raw target audit")
        signed = raw - baseline
        result["signed_quantiles"] = {str(q): float(np.quantile(signed, q)) for q in (0, .05, .25, .5, .75, .95, 1)}
        result["negative_signed_rows"] = int(np.sum(signed < 0))
        result["reconstruction_max_abs"] = float(np.max(np.abs((baseline + signed) - raw)))
    return result


def reconstruct(baseline, signed_prediction) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=float)
    signed = np.asarray(signed_prediction, dtype=float)
    if baseline.ndim != 1 or signed.shape != baseline.shape or not np.isfinite(baseline).all() or not np.isfinite(signed).all() or (baseline < 0).any():
        raise ContractError("aligned finite baseline and signed prediction required")
    value = np.maximum(0.0, baseline + signed)
    if not np.isfinite(value).all():
        raise ContractError("nonfinite reconstructed prediction")
    return value


def _schema(frame: pd.DataFrame) -> list[dict]:
    return [dict(name=column, dtype=str(frame[column].dtype), categorical=column == "spout_no")
            for column in frame]


def _validate_original_frame(frame: pd.DataFrame) -> None:
    forbidden = ("burden_lag__", "trajectory__", "baseline__", "residual__")
    if len(frame.columns) != 210 or "spout_no" not in frame or any(column.startswith(forbidden) for column in frame):
        raise ContractError("original certified 210-column E09/R2 frame required")


class CenteredRecencyIronModel:
    """Frozen recency60 CatBoost fitted to a signed iron deviation."""

    def fit(self, frame: pd.DataFrame, raw_target: pd.Series, baseline, weight: pd.Series) -> "CenteredRecencyIronModel":
        _validate_original_frame(frame)
        if not frame.index.equals(raw_target.index) or not frame.index.equals(weight.index) or raw_target.name != "tap_iron":
            raise ContractError("centered iron rows/target/weights are misaligned")
        raw = raw_target.to_numpy(dtype=float)
        baseline = np.asarray(baseline, dtype=float)
        weights = weight.to_numpy(dtype=float)
        if raw.shape != baseline.shape or not np.isfinite(raw).all() or (raw < 0).any() or not np.isfinite(baseline).all() or (baseline < 0).any():
            raise ContractError("finite nonnegative raw iron and baseline required")
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ContractError("finite positive recency weights required")
        signed = raw - baseline
        if not np.isfinite(signed).all() or not (signed < 0).any() or not (signed > 0).any():
            raise ContractError("centered iron response must retain finite signed variation")
        self.feature_names = list(frame)
        self.feature_schema = _schema(frame)
        self.model = CatBoostRegressor(**FROZEN_PARAMETERS)
        self.model.fit(frame, signed, cat_features=["spout_no"], sample_weight=weight)
        return self

    def predict_signed(self, frame: pd.DataFrame) -> np.ndarray:
        _validate_original_frame(frame)
        if list(frame) != self.feature_names or _schema(frame) != self.feature_schema:
            raise ContractError("centered iron inference schema differs")
        value = np.asarray(self.model.predict(frame), dtype=float)
        if not np.isfinite(value).all():
            raise ContractError("nonfinite signed iron prediction")
        return value

    def predict(self, frame: pd.DataFrame, baseline) -> np.ndarray:
        return reconstruct(baseline, self.predict_signed(frame))

    def save(self, root: str | Path, metadata: dict) -> dict:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        self.model.save_model(root / "direct.cbm")
        bundle = {
            "kind": "V25_HISTORY_CENTERED_RECENCY60_IRON_v1",
            "response_transform": "tap_iron_minus_legal_last100_baseline",
            "parameters": FROZEN_PARAMETERS,
            "feature_names": self.feature_names,
            "feature_schema": self.feature_schema,
            "model_sha256": file_sha256(root / "direct.cbm"),
            "metadata": metadata,
        }
        atomic_write_json(root / "bundle.json", bundle)
        return bundle

    @classmethod
    def load(cls, root: str | Path, expected_bundle_sha256: str) -> "CenteredRecencyIronModel":
        root = Path(root)
        if file_sha256(root / "bundle.json") != expected_bundle_sha256:
            raise ContractError("untrusted centered iron bundle metadata")
        value = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
        if (value.get("kind") != "V25_HISTORY_CENTERED_RECENCY60_IRON_v1"
                or value.get("response_transform") != "tap_iron_minus_legal_last100_baseline"
                or value.get("parameters") != FROZEN_PARAMETERS
                or file_sha256(root / "direct.cbm") != value.get("model_sha256")):
            raise ContractError("centered iron model identity differs")
        result = cls()
        result.feature_names = value["feature_names"]
        result.feature_schema = value["feature_schema"]
        result.model = CatBoostRegressor()
        result.model.load_model(root / "direct.cbm")
        result.metadata = value
        return result


def compose_iron(new_direct, old_e04, old_rate, old_r2_time, beta: float) -> tuple[np.ndarray, dict]:
    arrays = [np.asarray(item, dtype=float) for item in (new_direct, old_e04, old_rate, old_r2_time)]
    if len({item.shape for item in arrays}) != 1 or arrays[0].ndim != 1 or not all(np.isfinite(item).all() for item in arrays):
        raise ContractError("aligned finite centered iron composition inputs required")
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ContractError("certified V6I beta must be in [0,1]")
    base = .8 * arrays[0] + .2 * arrays[1]
    usable = arrays[2] > 1e-6
    value = base.copy()
    value[usable] = base[usable] + beta * (arrays[2][usable] * arrays[3][usable] - base[usable])
    value = np.maximum(0.0, value)
    return value, {"rate_unusable": int((~usable).sum()), "beta": float(beta)}


def roundtrip_six(values) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ContractError("finite nonnegative predictions required for six-decimal roundtrip")
    return np.asarray([float(f"{value:.6f}") for value in raw], dtype=float)


def compose_time(new_qrf, old_gate, old_median) -> np.ndarray:
    qrf = roundtrip_six(new_qrf)
    gate = np.asarray(old_gate)
    median = np.asarray(old_median, dtype=float)
    if gate.dtype != np.bool_ or gate.shape != qrf.shape or median.shape != qrf.shape or not np.isfinite(median).all() or (median < 0).any():
        raise ContractError("aligned original V21 gate and median required")
    result = qrf.copy()
    result[gate] = .75 * qrf[gate] + .25 * median[gate]
    return result


def isolate(parent: pd.DataFrame, *, iron=None, time=None, candidate: str) -> pd.DataFrame:
    if list(parent) != ["sample_id", *PREDICTION_COLUMNS] or parent.sample_id.isna().any() or parent.sample_id.astype(str).duplicated().any():
        raise ContractError("valid ordered V21 parent required")
    if (iron is None) == (time is None) or candidate not in (CANDIDATE_A, CANDIDATE_B):
        raise ContractError("exactly one registered centered target replacement required")
    result = parent.copy()
    if iron is not None:
        result["pred_tap_iron"] = np.asarray(iron, dtype=float)
        if not np.array_equal(result.pred_tap_time_len.to_numpy(), parent.pred_tap_time_len.to_numpy()):
            raise ContractError("candidate A changed time")
    else:
        result["pred_tap_time_len"] = np.asarray(time, dtype=float)
        if not np.array_equal(result.pred_tap_iron.to_numpy(), parent.pred_tap_iron.to_numpy()):
            raise ContractError("candidate B changed iron")
    values = result[PREDICTION_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ContractError("invalid isolated centered candidate output")
    return result

