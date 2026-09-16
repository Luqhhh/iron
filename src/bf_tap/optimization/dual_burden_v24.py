"""Target-isolated v0.24 composition and persistence primitives."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from ..artifacts import atomic_write_json, file_sha256
from ..exceptions import ContractError
from ..models.baseline import FROZEN_PARAMETERS
from .burden_lag_v24 import FEATURE_COLUMNS


CANDIDATE_A = "V24I_BURDEN_LAG_RECENCY_IRON"
CANDIDATE_B = "V24T_BURDEN_LAG_QRF_TIME"
PREDICTION_COLUMNS = ["pred_tap_iron", "pred_tap_time_len"]


class BurdenRecencyIron:
    """The frozen recency60 direct iron learner with exactly 30 appended columns."""
    def fit(self, x: pd.DataFrame, y: pd.Series, weight: pd.Series) -> "BurdenRecencyIron":
        if list(x.columns[-30:]) != list(FEATURE_COLUMNS) or "spout_no" not in x:
            raise ContractError("v0.24 expanded CatBoost schema required")
        if not x.index.equals(y.index) or not x.index.equals(weight.index) or y.name != "tap_iron":
            raise ContractError("v0.24 iron rows/target/weights are misaligned")
        values = y.to_numpy(dtype=float); weights = weight.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ContractError("finite nonnegative iron and positive weights required")
        self.feature_names = list(x)
        self.feature_schema = [dict(name=c, dtype=str(x[c].dtype), categorical=c == "spout_no") for c in x]
        self.model = CatBoostRegressor(**FROZEN_PARAMETERS)
        self.model.fit(x, y, cat_features=["spout_no"], sample_weight=weight)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        schema = [dict(name=c, dtype=str(x[c].dtype), categorical=c == "spout_no") for c in x]
        if list(x) != self.feature_names or schema != self.feature_schema:
            raise ContractError("v0.24 iron inference schema differs")
        value = np.maximum(0.0, np.asarray(self.model.predict(x), dtype=float))
        if not np.isfinite(value).all():
            raise ContractError("nonfinite v0.24 direct iron prediction")
        return value

    def save(self, root: str | Path, metadata: dict) -> dict:
        root = Path(root); root.mkdir(parents=True, exist_ok=False)
        self.model.save_model(root / "direct.cbm")
        bundle = dict(kind="V24_BURDEN_LAG_RECENCY60_IRON_v1", parameters=FROZEN_PARAMETERS,
                      feature_names=self.feature_names, feature_schema=self.feature_schema,
                      model_sha256=file_sha256(root / "direct.cbm"), metadata=metadata)
        atomic_write_json(root / "bundle.json", bundle)
        return bundle

    @classmethod
    def load(cls, root: str | Path, expected_bundle_sha256: str) -> "BurdenRecencyIron":
        root = Path(root)
        if file_sha256(root / "bundle.json") != expected_bundle_sha256:
            raise ContractError("untrusted v0.24 iron bundle metadata")
        value = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
        if value.get("kind") != "V24_BURDEN_LAG_RECENCY60_IRON_v1" or value.get("parameters") != FROZEN_PARAMETERS or file_sha256(root / "direct.cbm") != value.get("model_sha256"):
            raise ContractError("v0.24 iron model identity differs")
        result = cls(); result.feature_names = value["feature_names"]; result.feature_schema = value["feature_schema"]
        result.model = CatBoostRegressor(); result.model.load_model(root / "direct.cbm")
        result.metadata = value
        return result


def compose_iron(new_direct, old_e04, old_rate, old_r2_time, beta: float) -> tuple[np.ndarray, dict]:
    arrays = [np.asarray(x, dtype=float) for x in (new_direct, old_e04, old_rate, old_r2_time)]
    if len({x.shape for x in arrays}) != 1 or arrays[0].ndim != 1 or not all(np.isfinite(x).all() for x in arrays):
        raise ContractError("aligned finite v0.24 iron composition inputs required")
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ContractError("certified V6I beta must be in [0,1]")
    base = 0.8 * arrays[0] + 0.2 * arrays[1]
    usable = arrays[2] > 1e-6
    value = base.copy()
    value[usable] = base[usable] + beta * (arrays[2][usable] * arrays[3][usable] - base[usable])
    value = np.maximum(0.0, value)
    if not np.isfinite(value).all():
        raise ContractError("nonfinite v0.24 iron output")
    return value, {"rate_unusable": int((~usable).sum()), "beta": float(beta)}


def roundtrip_six(values) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ContractError("finite nonnegative predictions required for six-decimal roundtrip")
    return np.asarray([float(f"{v:.6f}") for v in raw], dtype=float)


def compose_time(new_qrf, old_gate, old_median) -> np.ndarray:
    q = roundtrip_six(new_qrf)
    gate = np.asarray(old_gate)
    median = np.asarray(old_median, dtype=float)
    if gate.dtype != np.bool_ or gate.shape != q.shape or median.shape != q.shape or not np.isfinite(median).all() or (median < 0).any():
        raise ContractError("aligned old V21 gate/median inputs required")
    result = q.copy(); result[gate] = 0.75 * q[gate] + 0.25 * median[gate]
    return result


def isolate(parent: pd.DataFrame, *, iron=None, time=None, candidate: str) -> pd.DataFrame:
    if list(parent) != ["sample_id", *PREDICTION_COLUMNS] or parent.sample_id.isna().any() or parent.sample_id.astype(str).duplicated().any():
        raise ContractError("valid ordered V21 parent required")
    if (iron is None) == (time is None) or candidate not in (CANDIDATE_A, CANDIDATE_B):
        raise ContractError("exactly one registered target replacement required")
    result = parent.copy()
    if iron is not None:
        result["pred_tap_iron"] = np.asarray(iron, dtype=float)
        if not np.array_equal(result.pred_tap_time_len.to_numpy(), parent.pred_tap_time_len.to_numpy()):
            raise ContractError("candidate A changed time")
    else:
        result["pred_tap_time_len"] = np.asarray(time, dtype=float)
        if not np.array_equal(result.pred_tap_iron.to_numpy(), parent.pred_tap_iron.to_numpy()):
            raise ContractError("candidate B changed iron")
    if not np.isfinite(result[PREDICTION_COLUMNS].to_numpy()).all() or (result[PREDICTION_COLUMNS].to_numpy() < 0).any():
        raise ContractError("invalid isolated candidate output")
    return result
