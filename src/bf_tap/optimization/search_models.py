"""Versioned single-target models for OPT-08; baseline remains immutable."""
from __future__ import annotations

import json
from importlib.metadata import version
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..artifacts import atomic_write_json, file_sha256, stable_digest
from ..exceptions import ContractError

REGISTRY_VERSION = "optimization-model-registry-v1"
MODEL_TYPES = {"tunable_catboost_l1", "lightgbm_l1"}


def select_promotions(records: list[dict], configs: list[dict]) -> list[dict]:
    expected = {(c["id"], t, fold) for c in configs for t in ("tap_iron", "tap_time_len")
                for fold in ("DEV_LONG", "DEV_SHORT")}
    received = {(r["candidate"], r["target"], r["fold"]) for r in records}
    if len(records) != len(expected) or received != expected:
        raise ContractError("promotion requires all configurations on both distinct DEV folds and targets")
    if any(not np.isfinite(r["wmape"]) or r["wmape"] < 0 for r in records):
        raise ContractError("promotion metrics must be finite and nonnegative")
    promotions = []
    for model_type in ("tunable_catboost_l1", "lightgbm_l1"):
        for target in ("tap_iron", "tap_time_len"):
            scored = []
            for config in configs:
                if config["model_type"] != model_type:
                    continue
                rows = [r for r in records if r["candidate"] == config["id"] and r["target"] == target]
                if any(r["model_type"] != model_type for r in rows):
                    raise ContractError("promotion metric model identity mismatch")
                scored.append((float(np.mean([r["wmape"] for r in rows])), config["capacity"], config["id"]))
            score, _, candidate = min(scored)
            promotions.append({"model_type": model_type, "target": target, "candidate": candidate,
                               "mean_two_fold_wmape": score})
    return promotions


def search_configurations(directory: str | Path) -> list[dict]:
    result = []
    for filename in ("catboost_l1.yaml", "lightgbm_l1.yaml"):
        cfg = yaml.safe_load((Path(directory) / filename).read_text())
        if (set(cfg) != {"schema_version", "model_type", "parameters", "grid", "early_stopping_patience"}
                or cfg["schema_version"] != "optimization-model-search-v1"
                or cfg["model_type"] not in MODEL_TYPES or cfg["early_stopping_patience"] != 150):
            raise ContractError("invalid v0.3 search model schema")
        if cfg["model_type"] == "tunable_catboost_l1":
            if cfg["grid"] != {"depth": [3, 4, 5], "l2_leaf_reg": [5, 20, 50]}:
                raise ContractError("CatBoost finite search grid differs from registration")
            variants = [dict(depth=d, l2_leaf_reg=l2) for d, l2 in product([3, 4, 5], [5, 20, 50])]
            if cfg["parameters"].get("loss_function") != "MAE":
                raise ContractError("v0.3 CatBoost requires MAE")
            capacity = lambda p: [p["depth"], -p["l2_leaf_reg"]]
        else:
            pairs = [[7, 40], [7, 80], [15, 40], [15, 80], [31, 80], [31, 160]]
            if cfg["grid"] != {"pairs_num_leaves_min_data_in_leaf": pairs}:
                raise ContractError("LightGBM finite search grid differs from registration")
            variants = [dict(num_leaves=n, min_data_in_leaf=m) for n, m in pairs]
            if cfg["parameters"].get("objective") != "regression_l1":
                raise ContractError("v0.3 LightGBM requires regression_l1")
            capacity = lambda p: [p["num_leaves"], -p["min_data_in_leaf"]]
        for n, variant in enumerate(variants, 1):
            params = {**cfg["parameters"], **variant}
            result.append({"id": f"{'CB' if 'catboost' in cfg['model_type'] else 'LG'}{n:02d}",
                           "model_type": cfg["model_type"], "parameters": params,
                           "patience": 150, "capacity": capacity(params)})
    return result


class SingleTargetModel:
    def __init__(self, model_type: str, parameters: dict, target: str, patience: int = 150):
        if model_type not in MODEL_TYPES or target not in {"tap_iron", "tap_time_len"}:
            raise ContractError("unknown registered model type or target")
        self.model_type, self.parameters, self.target = model_type, dict(parameters), target
        self.patience = patience
        self.mapping: dict[str, list[str]] = {}

    def _encode(self, X: pd.DataFrame, *, fit: bool = False) -> pd.DataFrame:
        if fit:
            if "spout_no" not in X or X.columns.duplicated().any():
                raise ContractError("model feature schema is invalid")
            self.columns = list(X.columns)
            self.feature_schema = [{"name": c, "dtype": str(X[c].dtype)} for c in X]
            self.mapping = {"spout_no": sorted(X.spout_no.dropna().astype(str).unique())}
        if list(X.columns) != self.columns:
            raise ContractError("model feature schema/order mismatch")
        frame = X.copy()
        for column, values in self.mapping.items():
            strings = frame[column].astype("string")
            if self.model_type == "lightgbm_l1":
                # Unknown and missing are -1, which LightGBM treats as missing.
                frame[column] = strings.map({v: i for i, v in enumerate(values)}).fillna(-1).astype("int32")
            else:
                frame[column] = strings.fillna("__MISSING__").astype(str)
        return frame

    def fit(self, X: pd.DataFrame, y: pd.Series, *,
            inner_validation: tuple[pd.DataFrame, pd.Series] | None = None,
            iterations: int | None = None) -> "SingleTargetModel":
        if X.empty or len(X) != len(y) or not X.index.equals(y.index) or not np.isfinite(y.to_numpy(float)).all():
            raise ContractError("single-target training rows/labels are invalid or misaligned")
        if inner_validation is None and iterations is None:
            raise ContractError("new models require inner selection or explicit selected iterations")
        if inner_validation is not None and iterations is not None:
            raise ContractError("choose either inner selection or fixed-iteration refit")
        if iterations is not None and (isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1):
            raise ContractError("selected iterations must be a positive integer")
        if (y.to_numpy(float) < 0).any():
            raise ContractError("training targets must be nonnegative")
        train = self._encode(X, fit=True)
        validation = None
        if inner_validation is not None:
            vx, vy = inner_validation
            if (vx.empty or not vx.index.equals(vy.index) or not np.isfinite(vy.to_numpy(float)).all()
                    or (vy.to_numpy(float) < 0).any()):
                raise ContractError("inner validation is empty or misaligned")
            validation = self._encode(vx), vy
        params = dict(self.parameters)
        if self.model_type == "tunable_catboost_l1":
            from catboost import CatBoostRegressor
            if iterations is not None:
                params["iterations"] = int(iterations)
            params["use_best_model"] = validation is not None
            self.model = CatBoostRegressor(**params)
            self.model.fit(train, y, cat_features=["spout_no"], eval_set=validation,
                           **({"early_stopping_rounds": self.patience} if validation is not None else {}))
            self.selected_iterations = int(self.model.tree_count_)
            self.effective_parameters = self.model.get_all_params()
        else:
            import lightgbm as lgb
            maximum = int(params.pop("num_boost_round"))
            dataset = lgb.Dataset(train, label=y, categorical_feature=["spout_no"], free_raw_data=False)
            valid_sets = ([lgb.Dataset(validation[0], label=validation[1], reference=dataset,
                                       categorical_feature=["spout_no"])] if validation is not None else None)
            self.model = lgb.train(params, dataset, num_boost_round=int(iterations or maximum),
                                   valid_sets=valid_sets,
                                   callbacks=[lgb.early_stopping(self.patience, verbose=False)] if valid_sets else [])
            self.selected_iterations = int(self.model.best_iteration or self.model.current_iteration())
            self.effective_parameters = dict(self.model.params)
        self.actual_tree_count = self.selected_iterations
        if iterations is not None:
            self.selected_iterations = iterations
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self._encode(X)), dtype=float)

    def save(self, directory: str | Path, *, identity: dict) -> None:
        required = {"fit_cutoff", "history_cutoff", "label_available_cutoff", "train_samples_sha256",
                    "inner_split", "source_sha256", "data_sha256"}
        if required - set(identity) or not all(identity[k] for k in required):
            raise ContractError("model bundle is missing training provenance")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        model_path = path / "model.bin"
        self.model.save_model(str(model_path))
        metadata = {"schema_version": REGISTRY_VERSION, "model_type": self.model_type,
                    "target": self.target, "parameters": self.parameters, "patience": self.patience,
                    "effective_parameters": self.effective_parameters,
                    "selected_iterations": self.selected_iterations, "feature_columns": self.columns,
                    "actual_tree_count": self.actual_tree_count,
                    "feature_schema": self.feature_schema, "categorical_mapping": self.mapping,
                    "unknown_category_policy": "negative_one_missing" if self.model_type == "lightgbm_l1" else "native_catboost",
                    "identity": identity, "dependencies": {name: version(name) for name in ("catboost", "lightgbm", "numpy", "pandas")},
                    "model_sha256": file_sha256(model_path)}
        atomic_write_json(path / "bundle.json", metadata)
        atomic_write_json(path / "bundle_identity.json", {"metadata_sha256": stable_digest(metadata)})

    @classmethod
    def load(cls, directory: str | Path) -> "SingleTargetModel":
        path = Path(directory)
        metadata = json.loads((path / "bundle.json").read_text())
        digest = json.loads((path / "bundle_identity.json").read_text())
        if metadata["schema_version"] != REGISTRY_VERSION or stable_digest(metadata) != digest["metadata_sha256"]:
            raise ContractError("model bundle metadata mismatch")
        if file_sha256(path / "model.bin") != metadata["model_sha256"]:
            raise ContractError("model file digest mismatch")
        if any(version(name) != expected for name, expected in metadata["dependencies"].items()):
            raise ContractError("model bundle dependency version mismatch")
        obj = cls(metadata["model_type"], metadata["parameters"], metadata["target"], metadata["patience"])
        obj.columns, obj.mapping = metadata["feature_columns"], metadata["categorical_mapping"]
        obj.feature_schema = metadata["feature_schema"]
        obj.selected_iterations = metadata["selected_iterations"]
        obj.actual_tree_count = metadata.get("actual_tree_count", metadata["selected_iterations"])
        obj.effective_parameters = metadata["effective_parameters"]
        if obj.model_type == "tunable_catboost_l1":
            from catboost import CatBoostRegressor
            obj.model = CatBoostRegressor()
            obj.model.load_model(str(path / "model.bin"))
        else:
            import lightgbm as lgb
            obj.model = lgb.Booster(model_file=str(path / "model.bin"))
        obj.identity = metadata["identity"]
        return obj
