"""V3.3 structural-search estimators: feature packs, EBM and residual wrappers."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data import FEATURES
from .metrics import wmape
from .v3_3_residual import FullRecipeResidualRegressor
from .v3_local_search import fit_target_transform, inverse_target_transform

GROUPS = {
    "A": ["air_volume", "oxygen", "air_speed", "humidity"],
    "P": ["cold_air_press", "hot_air_press", "furnace_top_press", "upper_press_diff", "lower_press_diff", "total_press_diff", "air_press_ratio"],
    "T": ["hot_air_temp", "furnace_top_temp_avg", "furnace_throat_temp"],
    "F": ["coal_rate", "gas_rate", "fuel_rate", "coke_rate"],
    "B": ["pig", "all_quality", "consumption"],
}
DERIVED = ["oxygen_per_air_volume", "pressure_per_air_volume", "thermal_difference", "upper_pressure_fraction"]


def _derived(frame: pd.DataFrame) -> pd.DataFrame:
    if (frame[["air_volume", "total_press_diff"]] <= 0).any().any():
        raise ValueError("V3.3 derived features require positive denominators")
    out = pd.DataFrame(index=frame.index)
    out["oxygen_per_air_volume"] = frame["oxygen"] / (60.0 * frame["air_volume"])
    out["pressure_per_air_volume"] = frame["total_press_diff"] / frame["air_volume"]
    out["thermal_difference"] = frame["hot_air_temp"] - frame["furnace_throat_temp"]
    out["upper_pressure_fraction"] = frame["upper_press_diff"] / frame["total_press_diff"]
    return out


class PackCatBoostRegressor:
    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        self.pack = self.trial["feature_pack"]
        self.target = self.trial["target"]
        self.target_transform = self.trial["target_transform"]
        self.parameters = dict(self.trial["parameters"])
        self.l2_multiplier = float(self.trial.get("l2_multiplier", 1.0))

    def _build(self, frame: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        numeric = list(FEATURES)
        if self.pack == "F1":
            out = frame.loc[:, numeric].copy()
            out = pd.concat([out, _derived(frame)], axis=1)
        elif self.pack == "F7":
            out = frame.loc[:, [c for c in numeric if c not in GROUPS["P"] and c not in GROUPS["T"]]].copy()
            if fit:
                self.scaler_p_ = StandardScaler().fit(frame[GROUPS["P"]])
                self.pca_p_ = PCA(n_components=2, random_state=42).fit(self.scaler_p_.transform(frame[GROUPS["P"]]))
                self.scaler_t_ = StandardScaler().fit(frame[GROUPS["T"]])
                self.pca_t_ = PCA(n_components=1, random_state=42).fit(self.scaler_t_.transform(frame[GROUPS["T"]]))
            pcs = self.pca_p_.transform(self.scaler_p_.transform(frame[GROUPS["P"]]))
            tcs = self.pca_t_.transform(self.scaler_t_.transform(frame[GROUPS["T"]]))
            out["pca_p1"] = pcs[:, 0]; out["pca_p2"] = pcs[:, 1]; out["pca_t1"] = tcs[:, 0]
        else:
            drop = {"F2": "P", "F3": "T", "F4": "F", "F5": "B", "F6": "A"}.get(self.pack)
            keep = [c for c in numeric if drop is None or c not in GROUPS[drop]]
            out = frame.loc[:, keep].copy()
        out["spout_no"] = frame["spout_no"].to_numpy()
        if not np.isfinite(out.to_numpy(dtype=float)).all():
            raise ValueError("V3.3 feature pack produced nonfinite values")
        self.input_columns_ = tuple(out.columns)
        return out

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "PackCatBoostRegressor":
        y = np.asarray(target, dtype=float)
        if y.ndim != 1 or len(y) != len(frame) or not np.isfinite(y).all():
            raise ValueError("Invalid pack labels")
        z, self.target_state_ = fit_target_transform(y, self.target_transform)
        x = self._build(frame, fit=True)
        params = dict(self.parameters)
        params["l2_leaf_reg"] = float(params.get("l2_leaf_reg", 10.0)) * self.l2_multiplier
        params["iterations"] = int(params.get("iterations", 1000))
        params.setdefault("cat_features", ["spout_no"])
        params["allow_writing_files"] = False
        params["verbose"] = False
        params["thread_count"] = 1
        x = x.assign(spout_no=x["spout_no"].astype(str))
        self.estimator_ = CatBoostRegressor(**params)
        self.estimator_.fit(x, z)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = self._build(frame)
        if tuple(x.columns) != self.input_columns_:
            raise ValueError("V3.3 pack feature order mismatch")
        x = x.assign(spout_no=x["spout_no"].astype(str))
        pred = inverse_target_transform(np.asarray(self.estimator_.predict(x), dtype=float),
                                        self.target_transform, self.target_state_)
        if pred.shape != (len(frame),) or not np.isfinite(pred).all():
            raise ValueError("Invalid V3.3 pack predictions")
        return pred


class EBMRegressor:
    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        self.target = self.trial["target"]
        self.target_transform = self.trial["target_transform"]
        self.parameters = dict(self.trial["parameters"])

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "EBMRegressor":
        from interpret.glassbox import ExplainableBoostingRegressor
        y = np.asarray(target, dtype=float)
        if y.ndim != 1 or len(y) != len(frame) or not np.isfinite(y).all():
            raise ValueError("Invalid EBM labels")
        z, self.target_state_ = fit_target_transform(y, self.target_transform)
        x = frame.loc[:, [*FEATURES, "spout_no"]].copy()
        feature_types = ["continuous"] * len(FEATURES) + ["nominal"]
        params = dict(self.parameters)
        self.estimator_ = ExplainableBoostingRegressor(feature_types=feature_types, **params)
        self.estimator_.fit(x, z)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, [*FEATURES, "spout_no"]].copy()
        pred = inverse_target_transform(np.asarray(self.estimator_.predict(x), dtype=float),
                                        self.target_transform, self.target_state_)
        if pred.shape != (len(frame),) or not np.isfinite(pred).all():
            raise ValueError("Invalid EBM predictions")
        return pred


class V33Regressor:
    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        kind = self.trial["kind"]
        if kind == "pack_catboost":
            self.impl = PackCatBoostRegressor(self.trial)
        elif kind == "ebm":
            self.impl = EBMRegressor(self.trial)
        elif kind == "full_residual":
            self.impl = FullRecipeResidualRegressor(self.trial)
        else:
            raise ValueError(f"Unknown V3.3 trial kind: {kind}")

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "V33Regressor":
        self.impl.fit(frame, target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.impl.predict(frame)


def evaluate_v33_outer_folds(train: pd.DataFrame, folds: np.ndarray, trial: Mapping[str, Any],
                             fold_ids: Sequence[int]) -> dict:
    target = trial["target"]
    predictions = np.full(len(train), np.nan)
    fold_scores = {}
    for fold in fold_ids:
        training = train.loc[folds != fold].reset_index(drop=True)
        valid = train.loc[folds == fold].reset_index(drop=True)
        model = V33Regressor(trial)
        model.fit(training, training[target].to_numpy())
        pred = model.predict(valid)
        predictions[folds == fold] = pred
        fold_scores[str(int(fold))] = float(wmape(valid[target], pred))
    mask = np.isin(folds, list(fold_ids))
    if not np.isfinite(predictions[mask]).all():
        raise ValueError("V3.3 OOF coverage failed")
    return {
        "trial_id": trial["trial_id"], "kind": trial["kind"], "target": target,
        "pooled_wmape": float(wmape(train.loc[mask, target], predictions[mask])),
        "mean_wmape": float(np.mean([fold_scores[str(int(f))] for f in fold_ids])),
        "fold_scores": fold_scores, "predictions": predictions,
    }
