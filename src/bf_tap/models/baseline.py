from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..audit import sha256_file
from ..artifacts import stable_digest
from ..exceptions import ContractError

TARGETS = ("tap_iron", "tap_time_len")
FROZEN_PARAMETERS = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "iterations": 800,
    "depth": 5,
    "learning_rate": 0.03,
    "l2_leaf_reg": 5.0,
    "random_seed": 2026,
    "task_type": "CPU",
    "thread_count": 8,
    "bootstrap_type": "No",
    "random_strength": 0.0,
    "rsm": 1.0,
    "boosting_type": "Plain",
    "has_time": True,
    "one_hot_max_size": 64,
    "nan_mode": "Min",
    "use_best_model": False,
    "allow_writing_files": False,
    "verbose": False,
}


class DualTargetBaseline:
    def __init__(self, parameters: dict, categorical: tuple[str, ...] = ("spout_no",)):
        self.parameters = dict(parameters)
        self.categorical = tuple(categorical)
        differences = {
            key: (self.parameters.get(key), expected)
            for key, expected in FROZEN_PARAMETERS.items()
            if self.parameters.get(key) != expected
        }
        extras = set(self.parameters) - set(FROZEN_PARAMETERS)
        if differences or extras:
            raise ContractError(
                f"parameters differ from frozen baseline: differences={differences}, extras={sorted(extras)}"
            )

    @staticmethod
    def _catboost_regressor():
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ContractError("CatBoost is not installed") from exc
        return CatBoostRegressor

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "DualTargetBaseline":
        if set(TARGETS) - set(y.columns):
            raise ContractError("both target columns are required")
        if set(self.categorical) - set(X.columns):
            raise ContractError("categorical feature is missing")
        values = y[list(TARGETS)].to_numpy(dtype=float)
        if len(X) == 0 or len(X) != len(y) or not np.isfinite(values).all():
            raise ContractError("training rows and finite targets must be non-empty and aligned")
        if (values < 0).any():
            raise ContractError("training targets must be nonnegative")
        if X[list(self.categorical)].isna().any().any():
            raise ContractError("categorical values must use the configured missing token")
        for column in self.categorical:
            if X[column].nunique(dropna=False) > int(self.parameters["one_hot_max_size"]):
                raise ContractError(f"categorical cardinality exceeds contract: {column}")
        self.feature_names_ = list(X.columns)
        self.feature_schema_ = [
            {
                "name": column,
                "dtype": str(X[column].dtype),
                "categorical": column in self.categorical,
            }
            for column in X.columns
        ]
        self.models_ = {}
        model_type = self._catboost_regressor()
        for target in TARGETS:
            model = model_type(**self.parameters)
            model.fit(X, y[target], cat_features=list(self.categorical))
            self.models_[target] = model
        return self

    def predict_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        if list(X.columns) != self.feature_names_:
            raise ContractError("prediction feature order differs from bundle")
        actual_schema = [
            {
                "name": column,
                "dtype": str(X[column].dtype),
                "categorical": column in self.categorical,
            }
            for column in X.columns
        ]
        if actual_schema != self.feature_schema_:
            raise ContractError("prediction feature types differ from bundle")
        values = {f"pred_{t}": self.models_[t].predict(X) for t in TARGETS}
        result = pd.DataFrame(values, index=X.index)
        if not np.isfinite(result.to_numpy()).all():
            raise ContractError("model emitted non-finite prediction")
        return result

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.predict_raw(X).clip(lower=0.0)

    def save(
        self,
        directory: str | Path,
        *,
        metadata: dict,
        history_snapshot: pd.DataFrame,
    ) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        required_metadata = {
            "baseline_config",
            "semantic_contract",
            "feature_config",
            "contract_digests",
            "inference_source_contract",
            "training",
            "code_identity",
            "environment",
            "lockfile_sha256",
        }
        missing = required_metadata - set(metadata)
        if missing:
            raise ContractError(f"bundle metadata missing keys: {sorted(missing)}")
        if history_snapshot.empty:
            raise ContractError("bundle history snapshot must not be empty")
        for target, model in self.models_.items():
            model.save_model(destination / f"{target}.cbm")
        history_path = destination / "history_snapshot.csv"
        history_snapshot.to_csv(history_path, index=False, encoding="utf-8")
        canonical_history = history_snapshot.copy()
        for column in ("reference_time", "tap_end_time", "available_at"):
            canonical_history[column] = pd.to_datetime(
                canonical_history[column], utc=True
            ).dt.tz_convert("Asia/Shanghai")
        metadata = {**metadata, "training": dict(metadata["training"])}
        metadata["training"]["history_origin_sha256"] = stable_digest(
            canonical_history[["sample_id", "available_at"]]
            .astype(str)
            .to_dict("records")
        )
        component_sha256 = {
            f"{target}.cbm": sha256_file(destination / f"{target}.cbm")
            for target in TARGETS
        }
        component_sha256[history_path.name] = sha256_file(history_path)
        bundle = {
            "bundle_schema_version": 3,
            "parameters": self.parameters,
            "categorical": self.categorical,
            "feature_names": self.feature_names_,
            "feature_schema": self.feature_schema_,
            "targets": TARGETS,
            "postprocess": {"lower_bound": 0.0},
            "component_sha256": component_sha256,
            **metadata,
        }
        (destination / "bundle.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "DualTargetBaseline":
        source = Path(directory)
        metadata = json.loads((source / "bundle.json").read_text(encoding="utf-8"))
        if metadata.get("bundle_schema_version") != 3:
            raise ContractError("unsupported bundle schema")
        required_metadata = {
            "baseline_config",
            "semantic_contract",
            "feature_config",
            "contract_digests",
            "inference_source_contract",
            "training",
            "code_identity",
            "environment",
            "lockfile_sha256",
            "component_sha256",
        }
        missing = required_metadata - set(metadata)
        if missing:
            raise ContractError(f"bundle metadata missing keys: {sorted(missing)}")
        expected_components = {
            "tap_iron.cbm",
            "tap_time_len.cbm",
            "history_snapshot.csv",
        }
        if set(metadata["component_sha256"]) != expected_components:
            raise ContractError("bundle component manifest is incomplete")
        for filename, expected in metadata["component_sha256"].items():
            component = source / filename
            if not component.is_file() or sha256_file(component) != expected:
                raise ContractError(f"bundle component hash mismatch: {filename}")
        result = cls(metadata["parameters"], tuple(metadata["categorical"]))
        result.feature_names_ = metadata["feature_names"]
        result.feature_schema_ = metadata["feature_schema"]
        result.bundle_metadata_ = metadata
        result.bundle_directory_ = source
        model_type = result._catboost_regressor()
        result.models_ = {}
        for target in TARGETS:
            model = model_type()
            model.load_model(source / f"{target}.cbm")
            result.models_[target] = model
        return result

    def load_history_snapshot(self) -> pd.DataFrame:
        if not hasattr(self, "bundle_directory_"):
            raise ContractError("history snapshot is only available on a loaded bundle")
        frame = pd.read_csv(
            self.bundle_directory_ / "history_snapshot.csv",
            dtype={"sample_id": "string"},
        )
        for column in ("reference_time", "tap_end_time", "available_at"):
            frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_convert("Asia/Shanghai")
        expected = self.bundle_metadata_["training"].get("history_origin_sha256")
        actual = stable_digest(
            frame[["sample_id", "available_at"]].astype(str).to_dict("records")
        )
        if expected != actual:
            raise ContractError("history snapshot identity differs from training metadata")
        return frame
