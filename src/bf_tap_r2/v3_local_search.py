"""Round2 V3 large local search primitives.

This module intentionally separates cheap search decisions from release
evidence.  It provides deterministic parameter sampling, legal CatBoost /
LightGBM / XGBoost / MLP / kernel trial specs, a fold-only target transform,
and a nested OOF fusion utility.  Nothing here uploads a package or relaxes the
existing submission contract.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .data import FEATURES, TARGETS
from .metrics import wmape
from .models import inputs
from .splits import make_folds

try:  # optional, used only when a trial explicitly asks for it
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - exercised in environments without xgboost
    XGBRegressor = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - LightGBM is a declared dependency
    lgb = None

from catboost import CatBoostRegressor

DERIVED_FEATURES = (
    "oxygen_per_air_volume",
    "pressure_per_air_volume",
    "thermal_difference",
    "upper_pressure_fraction",
)

TOTAL_BUDGET = 400
TARGET_PLATFORM_SCORE = 96.2
CURRENT_PLATFORM_BEST_SCORE = 96.1259
CURRENT_LOCAL_REFERENCE_SCORE = 96.0123883047359

FAMILY_KEYS = ("catboost", "lightgbm", "xgboost", "mlp", "kernel", "expression")


@dataclass(frozen=True)
class SearchStage:
    seeds: tuple[int, ...]
    folds_per_seed: int
    keep_per_target: int


def load_spec(root: Path | str) -> dict:
    root = Path(root)
    path = root / "configs/round2_v3/experiment.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if spec.get("version") != "round2-v3-local-search":
        raise ValueError("Unexpected V3 search spec")
    validate_budget(spec["search_budget"])
    if spec["current_platform_best"]["score"] != CURRENT_PLATFORM_BEST_SCORE:
        raise ValueError("The frozen V3 platform reference changed")
    return spec


def validate_budget(budget: Mapping[str, int]) -> None:
    if int(budget.get("total", -1)) != TOTAL_BUDGET:
        raise ValueError("V3 first batch must total 400 trials")
    counted = sum(int(budget[k]) for k in FAMILY_KEYS)
    if counted != TOTAL_BUDGET:
        raise ValueError(f"Family budgets sum to {counted}, expected {TOTAL_BUDGET}")
    if int(budget.get("per_target", 0)) * 2 != TOTAL_BUDGET:
        raise ValueError("V3 budget must be split equally across two targets")
    for key in FAMILY_KEYS:
        if int(budget[key]) <= 0:
            raise ValueError(f"Empty family budget: {key}")


def _log_uniform(rng: np.random.Generator, low_log10: float, high_log10: float) -> float:
    return float(10 ** rng.uniform(float(low_log10), float(high_log10)))


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(float(low), float(high)))


def _choice(rng: np.random.Generator, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("Cannot choose from an empty sequence")
    return values[int(rng.integers(0, len(values)))]


def _legal_catboost(rng: np.random.Generator, space: Mapping[str, Any]) -> dict:
    grow = _choice(rng, space["grow_policies"])
    params: dict[str, Any] = {
        "task_type": "CPU",
        "loss_function": _choice(rng, space["loss_functions"]),
        "iterations": int(_choice(rng, space["iterations"])),
        "learning_rate": _log_uniform(rng, *space["learning_rate_log10"]),
        "l2_leaf_reg": _log_uniform(rng, *space["l2_leaf_reg_log10"]),
        "random_strength": _log_uniform(rng, *space["random_strength_log10"]),
        "boosting_type": _choice(rng, space["boosting_types"]),
        "grow_policy": grow,
        "thread_count": 1,
        "allow_writing_files": False,
        "verbose": False,
        "random_seed": int(_choice(rng, (42, 2026, 2027))),
        "cat_features": ["spout_no"],
    }
    if grow == "SymmetricTree":
        params["depth"] = int(rng.integers(space["depth"][0], space["depth"][1] + 1))
    elif grow == "Depthwise":
        params["depth"] = int(rng.integers(space["depth"][0], space["depth"][1] + 1))
        params["min_data_in_leaf"] = int(_choice(rng, (10, 20, 50, 100)))
    else:
        params["max_leaves"] = int(_choice(rng, (15, 31, 63, 127)))
        params["min_data_in_leaf"] = int(_choice(rng, (10, 20, 50, 100)))
    if grow != "SymmetricTree" and params["boosting_type"] == "Ordered":
        params["boosting_type"] = "Plain"
    bootstrap = _choice(rng, space["bootstrap_types"])
    if bootstrap == "MVS":
        params["bootstrap_type"] = "MVS"
        params["subsample"] = _uniform(rng, 0.6, 1.0)
    elif bootstrap == "Bernoulli":
        params["bootstrap_type"] = "Bernoulli"
        params["subsample"] = _uniform(rng, 0.6, 1.0)
    elif bootstrap == "Bayesian":
        params["bootstrap_type"] = "Bayesian"
        params["bagging_temperature"] = _uniform(rng, 0.0, 2.0)
    else:
        params["bootstrap_type"] = "No"
    return params


def _legal_lightgbm(rng: np.random.Generator, space: Mapping[str, Any]) -> dict:
    subsample = _uniform(rng, 0.6, 1.0)
    params = {
        "objective": "regression",
        "n_estimators": int(_choice(rng, space["n_estimators"])),
        "learning_rate": _log_uniform(rng, *space["learning_rate_log10"]),
        "num_leaves": int(_choice(rng, space["num_leaves"])),
        "max_depth": int(_choice(rng, space["max_depth"])),
        "min_child_samples": max(5, int(round(_log_uniform(rng, *space["min_child_samples_log10"])))),
        "reg_alpha": _log_uniform(rng, *space["reg_alpha_log10"]),
        "reg_lambda": _log_uniform(rng, *space["reg_lambda_log10"]),
        "subsample": subsample,
        "subsample_freq": 1 if subsample < 0.999 else 0,
        "colsample_bytree": _uniform(rng, 0.5, 1.0),
        "random_state": int(_choice(rng, (42, 2026, 2027))),
        "n_jobs": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "device_type": "cpu",
    }
    return params


def _legal_xgboost(rng: np.random.Generator, space: Mapping[str, Any]) -> dict:
    return {
        "objective": "reg:squarederror",
        "n_estimators": int(_choice(rng, space["n_estimators"])),
        "learning_rate": _log_uniform(rng, *space["learning_rate_log10"]),
        "max_depth": int(_choice(rng, space["max_depth"])),
        "min_child_weight": _log_uniform(rng, *space["min_child_weight_log10"]),
        "reg_alpha": _log_uniform(rng, *space["reg_alpha_log10"]),
        "reg_lambda": _log_uniform(rng, *space["reg_lambda_log10"]),
        "subsample": _uniform(rng, *space["subsample"]),
        "colsample_bytree": _uniform(rng, *space["colsample_bytree"]),
        "random_state": int(_choice(rng, (42, 2026, 2027))),
        "n_jobs": 1,
        "tree_method": "hist",
        "verbosity": 0,
    }


def _legal_mlp(rng: np.random.Generator, space: Mapping[str, Any]) -> dict:
    hidden = _choice(rng, space["mlp_hidden_layers"])
    return {
        "hidden_layer_sizes": tuple(int(v) for v in hidden),
        "activation": _choice(rng, ("relu", "tanh")),
        "solver": "adam",
        "alpha": _log_uniform(rng, *space["mlp_alpha_log10"]),
        "max_iter": 3000,
        "early_stopping": True,
        "n_iter_no_change": 50,
        "random_state": int(_choice(rng, (42, 2026, 2027))),
    }


def _legal_kernel(rng: np.random.Generator, space: Mapping[str, Any]) -> dict:
    kernel = _choice(rng, space["kernel"])
    out = {"kernel": kernel, "alpha": _log_uniform(rng, *space["kernel_alpha_log10"])}
    if kernel == "rbf":
        out["gamma"] = _log_uniform(rng, -3.0, 0.0)
    else:
        out["degree"] = int(_choice(rng, (2, 3)))
        out["coef0"] = _uniform(rng, 0.0, 2.0)
    return out


def sample_trials(spec: Mapping[str, Any], seed: int = 20260923) -> list[dict]:
    """Return exactly the frozen 400-trial V3 first-batch schedule.

    Budget is charged per (configuration, target) item, as requested.  The
    function contains no observation values and is therefore safe to freeze
    before any model fitting.
    """
    budget = spec["search_budget"]
    rng = np.random.default_rng(seed)
    family_allocations: list[tuple[str, int]] = []
    for family in FAMILY_KEYS:
        family_allocations.append((family, int(budget[family])))
    trials: list[dict] = []
    for family, count in family_allocations:
        # Half per target, exactly as the budget contract states.
        for target_index, target in enumerate(TARGETS):
            per_target = count // 2
            if count % 2:
                per_target += target_index
            for offset in range(per_target):
                trial_id = f"v3-{family}-{target}-{offset:04d}"
                if family == "catboost":
                    entry = {
                        "trial_id": trial_id,
                        "family": "catboost",
                        "target": target,
                        "feature_set": _choice(rng, spec["catboost_space"]["feature_sets"]),
                        "target_transform": _choice(rng, spec["catboost_space"]["target_transforms"]),
                        "parameters": _legal_catboost(rng, spec["catboost_space"]),
                    }
                elif family == "lightgbm":
                    entry = {
                        "trial_id": trial_id,
                        "family": "lightgbm",
                        "target": target,
                        "feature_set": _choice(rng, spec["lightgbm_space"]["feature_sets"]),
                        "target_transform": _choice(rng, spec["lightgbm_space"]["target_transforms"]),
                        "parameters": _legal_lightgbm(rng, spec["lightgbm_space"]),
                    }
                elif family == "xgboost":
                    entry = {
                        "trial_id": trial_id,
                        "family": "xgboost",
                        "target": target,
                        "feature_set": _choice(rng, spec["xgboost_space"]["feature_sets"]),
                        "target_transform": _choice(rng, spec["xgboost_space"]["target_transforms"]),
                        "parameters": _legal_xgboost(rng, spec["xgboost_space"]),
                    }
                elif family == "mlp":
                    entry = {
                        "trial_id": trial_id,
                        "family": "mlp",
                        "target": target,
                        "feature_set": _choice(rng, spec["mlp_kernel_space"]["feature_sets"]),
                        "target_transform": _choice(rng, spec["mlp_kernel_space"]["target_transforms"]),
                        "parameters": _legal_mlp(rng, spec["mlp_kernel_space"]),
                    }
                elif family == "kernel":
                    entry = {
                        "trial_id": trial_id,
                        "family": "kernel",
                        "target": target,
                        "feature_set": _choice(rng, spec["mlp_kernel_space"]["feature_sets"]),
                        "target_transform": _choice(rng, spec["mlp_kernel_space"]["target_transforms"]),
                        "parameters": _legal_kernel(rng, spec["mlp_kernel_space"]),
                    }
                else:
                    # Feature/objective-expression budget uses a tree backbone
                    # because it is currently the only family with broad local
                    # support.  The family label remains `expression` for the
                    # budget contract; `base_family` controls the estimator.
                    entry = {
                        "trial_id": trial_id,
                        "family": "expression",
                        "base_family": "catboost",
                        "target": target,
                        "feature_set": _choice(rng, ("raw", "raw", "four", "four")),
                        "target_transform": _choice(rng, ("mean", "mean_std", "log1p", "identity")),
                        "parameters": _legal_catboost(rng, spec["catboost_space"]),
                    }
                trials.append(entry)
    if len(trials) != TOTAL_BUDGET:
        raise AssertionError(f"Internal V3 schedule produced {len(trials)} trials")
    if sorted(pd.Series([t["trial_id"] for t in trials]).value_counts().tolist())[-1] != 1:
        raise AssertionError("Duplicate V3 trial IDs")
    counts = pd.Series([t["family"] for t in trials]).value_counts().to_dict()
    expected = {k: int(budget[k]) for k in FAMILY_KEYS}
    if counts != expected:
        raise AssertionError(f"V3 family counts mismatch: {counts}")
    return trials


def expression_frame(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "raw":
        x = frame.loc[:, [*FEATURES, "spout_no"]].copy()
    elif feature_set == "four":
        if (frame[["air_volume", "total_press_diff"]] <= 0).any().any():
            raise ValueError("V3 ratio features require positive denominators")
        x = frame.loc[:, [*FEATURES, "spout_no"]].copy()
        x["oxygen_per_air_volume"] = frame["oxygen"] / (60.0 * frame["air_volume"])
        x["pressure_per_air_volume"] = frame["total_press_diff"] / frame["air_volume"]
        x["thermal_difference"] = frame["hot_air_temp"] - frame["furnace_throat_temp"]
        x["upper_pressure_fraction"] = frame["upper_press_diff"] / frame["total_press_diff"]
    else:
        raise ValueError(f"Unknown V3 feature set: {feature_set}")
    if x.isna().any().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("V3 expression frame must be finite")
    return x


def fit_target_transform(values: np.ndarray, kind: str) -> tuple[np.ndarray, dict]:
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("Invalid V3 target")
    state: dict[str, float] = {}
    if kind == "identity":
        return y.copy(), state
    if np.any(y <= 0):
        raise ValueError(f"Target transform {kind} requires positive labels")
    if kind == "mean":
        mean = float(y.mean())
        state["mean"] = mean
        return (y - mean) / mean, state
    if kind == "mean_std":
        mean = float(y.mean())
        std = float(y.std(ddof=0))
        if std <= 0:
            raise ValueError("Target standard deviation must be positive")
        state.update({"mean": mean, "std": std})
        return (y - mean) / std, state
    if kind == "log1p":
        state["shift"] = 0.0
        return np.log1p(y), state
    raise ValueError(f"Unknown V3 target transform: {kind}")


def apply_target_transform(values: np.ndarray, kind: str, state: Mapping[str, float]) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("Invalid V3 target values")
    if kind == "identity":
        return y.copy()
    if kind == "mean":
        return (y - state["mean"]) / state["mean"]
    if kind == "mean_std":
        return (y - state["mean"]) / state["std"]
    if kind == "log1p":
        if np.any(y <= 0):
            raise ValueError("log1p target requires positive labels")
        return np.log1p(y)
    raise ValueError(f"Unknown V3 target transform: {kind}")


def inverse_target_transform(prediction: np.ndarray, kind: str, state: Mapping[str, float]) -> np.ndarray:
    z = np.asarray(prediction, dtype=float)
    if kind == "identity":
        return z.copy()
    if kind == "mean":
        return state["mean"] * (1.0 + z)
    if kind == "mean_std":
        return state["mean"] + state["std"] * z
    if kind == "log1p":
        return np.expm1(z + state["shift"])
    raise ValueError(f"Unknown V3 target transform: {kind}")


class TrialRegressor:
    """A single V3 trial, with all preprocessing learned on the training fold."""

    def __init__(self, trial: Mapping[str, Any]):
        self.trial = dict(trial)
        self.family = self.trial.get("base_family", self.trial["family"])
        self.feature_set = self.trial["feature_set"]
        self.target_transform = self.trial["target_transform"]
        if self.family not in {"catboost", "lightgbm", "xgboost", "mlp", "kernel"}:
            raise ValueError(f"Unknown V3 family: {self.family}")

    def _x(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = expression_frame(frame, self.feature_set)
        if self.family == "catboost":
            x["spout_no"] = x["spout_no"].astype(str)
        return x

    def _fit_estimator(self, x: pd.DataFrame, z: np.ndarray, eval_x: pd.DataFrame | None = None,
                       eval_z: np.ndarray | None = None) -> Any:
        params = dict(self.trial["parameters"])
        if self.family == "catboost":
            if eval_x is not None:
                params.update({"use_best_model": True, "od_type": "Iter", "od_wait": 100})
            estimator = CatBoostRegressor(**params)
            estimator.fit(x, z, eval_set=(eval_x, eval_z) if eval_x is not None else None)
            return estimator
        if self.family == "lightgbm":
            if lgb is None:
                raise RuntimeError("LightGBM is not importable")
            estimator = lgb.LGBMRegressor(**params)
            eval_set = [(eval_x, eval_z)] if eval_x is not None else None
            callbacks = [lgb.early_stopping(100, verbose=False)] if eval_set is not None else None
            estimator.fit(x, z, eval_set=eval_set, callbacks=callbacks,
                          categorical_feature=["spout_no"] if "spout_no" in x.columns else None)
            return estimator
        if self.family == "xgboost":
            if XGBRegressor is None:
                raise RuntimeError("xgboost is not installed; install the declared optional dependency to run these trials")
            estimator = XGBRegressor(**params)
            estimator.fit(x, z, eval_set=[(eval_x, eval_z)] if eval_x is not None else None,
                          verbose=False)
            return estimator
        # MLP / kernel use a train-fold-only scaler and one-hot spout.
        from sklearn.compose import ColumnTransformer
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        numeric = [c for c in x.columns if c != "spout_no"]
        pre = ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("spout", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["spout_no"]),
        ], remainder="drop")
        if self.family == "mlp":
            estimator = MLPRegressor(**params)
        else:
            estimator = KernelRidge(**params)
        pipe = Pipeline([("pre", pre), ("model", estimator)])
        pipe.fit(x, z)
        return pipe

    def fit(self, frame: pd.DataFrame, target: np.ndarray, eval_frame: pd.DataFrame | None = None,
            eval_target: np.ndarray | None = None) -> "TrialRegressor":
        self.target_state_ = {}
        z, self.target_state_ = fit_target_transform(np.asarray(target, dtype=float), self.target_transform)
        x = self._x(frame)
        eval_x = eval_z = None
        if eval_frame is not None and eval_target is not None:
            eval_x = self._x(eval_frame)
            eval_z = apply_target_transform(np.asarray(eval_target, dtype=float), self.target_transform, self.target_state_)
        self.estimator_ = self._fit_estimator(x, z, eval_x, eval_z)
        self.input_columns_ = tuple(x.columns)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = self._x(frame)
        if tuple(x.columns) != self.input_columns_:
            raise ValueError("V3 prediction feature order mismatch")
        prediction = inverse_target_transform(np.asarray(self.estimator_.predict(x), dtype=float),
                                              self.target_transform, self.target_state_)
        if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
            raise ValueError("Invalid V3 predictions")
        return prediction


def evaluate_trial_folds(train: pd.DataFrame, folds: np.ndarray, trial: Mapping[str, Any],
                         fold_ids: Sequence[int], early_stopping: bool = True,
                         inner_seed: int = 12345) -> dict:
    """Fit a trial on held-out folds; optional early stopping uses an inner split.

    The outer fold labels are never passed to the early-stopping call.
    """
    target = trial["target"]
    predictions = np.full(len(train), np.nan)
    fold_scores: dict[str, float] = {}
    for fold in fold_ids:
        training = train.loc[folds != fold].reset_index(drop=True)
        valid = train.loc[folds == fold].reset_index(drop=True)
        model = TrialRegressor(trial)
        if early_stopping and len(training) >= 100:
            rng = np.random.default_rng(inner_seed + int(fold))
            order = rng.permutation(len(training))
            cut = max(1, int(round(0.9 * len(training))))
            inner = training.iloc[order[:cut]].reset_index(drop=True)
            inner_valid = training.iloc[order[cut:]].reset_index(drop=True)
            model.fit(inner, inner[target].to_numpy(), inner_valid, inner_valid[target].to_numpy())
        else:
            model.fit(training, training[target].to_numpy())
        values = model.predict(valid)
        predictions[folds == fold] = values
        fold_scores[str(int(fold))] = wmape(valid[target], values)
    mask = np.isin(folds, list(fold_ids))
    if not np.isfinite(predictions[mask]).all():
        raise ValueError("V3 trial OOF coverage failed")
    return {
        "trial_id": trial["trial_id"],
        "family": trial["family"],
        "target": target,
        "feature_set": trial["feature_set"],
        "target_transform": trial["target_transform"],
        "fold_scores": fold_scores,
        "mean_wmape": float(np.mean([fold_scores[str(int(f))] for f in fold_ids])),
        "predictions": predictions,
    }


def package_local_score(iron_wmape: float, time_wmape: float) -> float:
    """The frozen local complete-package score: equal-target WMAPE."""
    return float(100.0 - 100.0 * ((float(iron_wmape) + float(time_wmape)) / 2.0))


def triage_delta(delta_local: float, thresholds: Mapping[str, Any]) -> str:
    if delta_local < float(thresholds["below_independent"]):
        return "not_independent"
    if delta_local < float(thresholds["fallback_upper"]):
        return "fallback"
    if delta_local < float(thresholds["primary_candidate"][0]):
        return "candidate_pool"
    primary = thresholds["primary_candidate"]
    if float(primary[0]) <= delta_local <= float(primary[1]):
        return "primary"
    return "candidate_pool"


def _align_series(series: pd.Series, ids: pd.Index) -> np.ndarray:
    if set(series.index) != set(ids):
        raise ValueError("OOF candidate ID set mismatch")
    out = series.loc[ids].to_numpy(dtype=float)
    if not np.isfinite(out).all():
        raise ValueError("OOF candidate contains nonfinite predictions")
    return out


def load_oof_candidates(root: Path | str, target: str, ids: pd.Index,
                        patterns: Iterable[str] = ("round2-v2*",)) -> dict[str, dict[str, np.ndarray]]:
    """Load all aligned `*-{target}.csv` OOF columns under local runs."""
    root = Path(root)
    library: dict[str, dict[str, np.ndarray]] = {}
    for pattern in patterns:
        for directory in sorted(root.glob(f"local/runs/{pattern}")):
            for path in sorted(directory.rglob(f"*/oof/*-{target}.csv")):
                frame = pd.read_csv(path, dtype={"sample_id": "string"}).set_index("sample_id")
                if frame.index.has_duplicates:
                    raise ValueError(f"Duplicate OOF IDs: {path}")
                for column in frame.columns:
                    if column in {"spout_no", "fold", target}:
                        continue
                    if not pd.api.types.is_numeric_dtype(frame[column]):
                        continue
                    key = f"{directory.name}:{path.parent.parent.name}:{column}"
                    seed = path.name.split("-", 1)[0]
                    library.setdefault(key, {})[seed] = _align_series(frame[column], ids)
    return library


def optimize_simplex_weights(y_by_seed: Mapping[str, np.ndarray],
                             p_by_seed: Mapping[str, np.ndarray],
                             starts: int = 24, seed: int = 123) -> tuple[float, np.ndarray]:
    """Minimize mean WMAPE over a nonnegative, sum-to-one weight vector.

    Uses SciPy's constrained SLSQP when available.  The simple projected
    subgradient fallback exists only so the utility stays importable in minimal
    environments; search/reporting should use SciPy.
    """
    seeds = sorted(y_by_seed)
    if list(seeds) != sorted(p_by_seed):
        raise ValueError("Prediction and label seed sets differ")
    n = p_by_seed[seeds[0]].shape[1]
    if n == 0:
        raise ValueError("No fusion candidates")
    for s in seeds:
        if p_by_seed[s].shape[1] != n or y_by_seed[s].shape[0] != p_by_seed[s].shape[0]:
            raise ValueError("Fusion matrix shape mismatch")

    def objective(w: np.ndarray) -> float:
        return float(np.mean([wmape(y_by_seed[s], p_by_seed[s] @ w) for s in seeds]))

    try:
        from scipy.optimize import minimize
    except Exception:  # pragma: no cover
        # Small projected subgradient fallback.
        w = np.ones(n) / n
        best_w, best = w.copy(), objective(w)
        rng = np.random.default_rng(seed)
        for step in range(4000):
            alpha = 0.2 / np.sqrt(step + 1.0)
            for s in seeds:
                residual = p_by_seed[s] @ w - y_by_seed[s]
                grad = (p_by_seed[s].T @ np.sign(residual)) / (len(residual) * np.abs(y_by_seed[s]).sum())
                w = w - alpha * grad
                w = np.maximum(w, 0.0)
                total = w.sum()
                if total <= 0:
                    w = np.ones(n) / n
                else:
                    w = w / total
            value = objective(w)
            if value < best:
                best, best_w = value, w.copy()
        return best, best_w

    rng = np.random.default_rng(seed)
    starts_list = [np.ones(n) / n]
    for _ in range(max(0, starts - 1)):
        if n <= 6:
            starts_list.append(rng.dirichlet(np.ones(n)))
        else:
            k = int(rng.integers(1, 6))
            subset = rng.choice(n, size=min(k, n), replace=False)
            w = np.zeros(n)
            w[subset] = 1.0 / len(subset)
            starts_list.append(w)
    bounds = [(0.0, 1.0)] * n
    constraints = ({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},)
    best = (np.inf, None)
    for start in starts_list:
        result = minimize(objective, start, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"maxiter": 800, "ftol": 1e-12})
        if result.success and result.fun < best[0]:
            weights = np.asarray(result.x, dtype=float)
            weights[weights < 1e-12] = 0.0
            if weights.sum() <= 0:
                continue
            weights /= weights.sum()
            best = (float(result.fun), weights)
    if best[1] is None:
        raise RuntimeError("Fusion weight optimization failed")
    return best


def greedy_forward_select(y_by_seed: Mapping[str, np.ndarray],
                          p_by_seed: Mapping[str, np.ndarray],
                          candidate_names: Sequence[str] | None = None,
                          max_members: int = 5,
                          starts: int = 24) -> dict:
    """Greedy forward member selection with simplex weight refitting."""
    seeds = sorted(y_by_seed)
    names = list(candidate_names) if candidate_names is not None else [str(i) for i in range(next(iter(p_by_seed.values())).shape[1])]
    if len(names) != next(iter(p_by_seed.values())).shape[1]:
        raise ValueError("Candidate name count mismatch")
    chosen: list[int] = []
    history: list[dict] = []
    for _ in range(min(max_members, len(names))):
        best_candidate = None
        best = None
        for j in range(len(names)):
            if j in chosen:
                continue
            columns = [*chosen, j]
            p = {s: p_by_seed[s][:, columns] for s in seeds}
            value, weights = optimize_simplex_weights(y_by_seed, p, starts=starts)
            if best is None or value < best["fit_score"]:
                best = {"fit_score": value, "columns": columns, "weights": weights}
                best_candidate = j
        if best_candidate is None:
            break
        if history and best["fit_score"] >= history[-1]["fit_score"] - 1e-12:
            break
        chosen = list(best["columns"])
        history.append({
            "members": [names[j] for j in chosen],
            "columns": chosen,
            "fit_score": float(best["fit_score"]),
            "weights": {names[j]: float(w) for j, w in zip(chosen, best["weights"])},
        })
    if not chosen:
        raise RuntimeError("No fusion members selected")
    return {"members": [names[j] for j in chosen], "columns": chosen, "weights": history[-1]["weights"],
            "fit_score": history[-1]["fit_score"], "history": history}


def nested_fusion(library: Mapping[str, Mapping[str, np.ndarray]], target: str,
                  labels_by_seed: Mapping[str, np.ndarray],
                  max_pool: int = 30, max_members: int = 5, starts: int = 24) -> dict:
    """Leave-one-seed-out fusion: rank and fit on the fit seed, score on the held-out seed."""
    seeds = sorted(labels_by_seed)
    if len(seeds) < 2:
        raise ValueError("Nested fusion requires at least two seeds")
    common = [key for key, values in library.items() if all(s in values for s in seeds)]
    if not common:
        raise ValueError("No fusion candidate covers all evaluation seeds")
    heldout: dict[str, dict] = {}
    for held in seeds:
        fit_s = [s for s in seeds if s != held]
        y_fit = {s: labels_by_seed[s] for s in fit_s}
        p_fit_all = {s: np.stack([library[key][s] for key in common], axis=1) for s in fit_s}
        # Rank on the fit seed(s) only: no held-out labels are used here.
        ranking = []
        for j, key in enumerate(common):
            score = float(np.mean([wmape(y_fit[s], p_fit_all[s][:, j]) for s in fit_s]))
            ranking.append((score, j))
        ranking.sort()
        pool = [common[j] for _, j in ranking[: min(max_pool, len(ranking))]]
        if len(pool) == 0:
            raise RuntimeError("Empty fusion pool")
        p_fit = {s: p_fit_all[s][:, [common.index(key) for key in pool]] for s in fit_s}
        selection = greedy_forward_select(y_fit, p_fit, pool, max_members=max_members, starts=starts)
        selected = [key for key in selection["members"]]
        weights = {key: selection["weights"][key] for key in selected}
        p_held = np.stack([library[key][held] for key in selected], axis=1)
        w = np.array([weights[key] for key in selected], dtype=float)
        heldout[held] = {
            "selected": selected,
            "weights": weights,
            "fit_score": selection["fit_score"],
            "heldout_wmape": wmape(labels_by_seed[held], p_held @ w),
        }
    return {
        "target": target,
        "seeds": seeds,
        "pool_size": len(common),
        "fold": heldout,
        "mean_heldout_wmape": float(np.mean([v["heldout_wmape"] for v in heldout.values()])),
    }


def build_run_manifest(spec: Mapping[str, Any], root: Path, seed: int = 20260923) -> dict:
    trials = sample_trials(spec, seed=seed)
    return {
        "version": spec["version"],
        "root": str(Path(root).resolve()),
        "seed": int(seed),
        "trials": len(trials),
        "family_counts": pd.Series([t["family"] for t in trials]).value_counts().sort_index().to_dict(),
        "target_counts": pd.Series([t["target"] for t in trials]).value_counts().sort_index().to_dict(),
        "trial_ids": [t["trial_id"] for t in trials],
        "agent_uploads": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args(argv)
    spec = load_spec(args.root)
    manifest = build_run_manifest(spec, args.root, seed=args.seed)
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    if args.manifest_output:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
