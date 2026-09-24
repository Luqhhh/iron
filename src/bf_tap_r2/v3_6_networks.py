"""V3.6 numeric-encoding networks: raw/PLE MLP and raw/PLE TabM.

Everything is trained from random initialization on the current training part
only.  Scalers, PLE bin edges, the categorical vocabulary, and the target
scale are fitted on that part alone.  TabM training averages the losses of the
parallel submodels; TabM inference averages their predictions.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import FEATURES
from .metrics import wmape
from .v3_4_bags import group_safe_inner_folds

_NETWORK_STRUCTURES = {"raw_mlp", "ple_mlp", "raw_tabm", "ple_tabm"}


def _torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("V3.6 network trials require torch") from exc
    return torch


def _set_seed(seed: int) -> None:
    torch = _torch()
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():  # pragma: no cover - CPU is the frozen path
        torch.cuda.manual_seed_all(int(seed) % (2**32 - 1))


class NumericPreprocessor:
    """Train-part-only preprocessing for numeric encodings and spout vocabulary."""

    def __init__(self, *, structure: str, n_bins: int = 16,
                 d_embedding: int = 8):
        if structure not in _NETWORK_STRUCTURES:
            raise ValueError(f"Unknown numeric-encoding structure: {structure!r}")
        self.structure = str(structure)
        self.n_bins = int(n_bins)
        self.d_embedding = int(d_embedding)
        self.feature_names_ = tuple(FEATURES)
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None
        self.bins_: list[Any] | None = None
        self.spout_to_index_: dict[int, int] = {}
        self.n_spout_categories_ = 0

    @property
    def uses_ple(self) -> bool:
        return self.structure in {"ple_mlp", "ple_tabm"}

    def fit(self, frame: pd.DataFrame) -> "NumericPreprocessor":
        if "spout_no" not in frame:
            raise ValueError("V3.6 numeric networks require spout_no")
        x = frame.loc[:, list(self.feature_names_)].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("V3.6 network features must be finite")
        if np.isclose(x.std(axis=0, ddof=0), 0.0).any():
            raise ValueError("V3.6 network features contain a constant column")
        self.means_ = x.mean(axis=0)
        self.stds_ = x.std(axis=0, ddof=0)
        self.stds_[np.isclose(self.stds_, 0.0)] = 1.0
        values = sorted({int(v) for v in frame["spout_no"].astype(int).tolist()})
        if not values:
            raise ValueError("V3.6 network training part has no spout_no values")
        self.spout_to_index_ = {int(v): i for i, v in enumerate(values, start=1)}
        self.n_spout_categories_ = len(values) + 1  # index 0 is reserved for unseen
        if self.uses_ple:
            torch = _torch()
            from rtdl_num_embeddings import compute_bins

            self.bins_ = compute_bins(
                torch.as_tensor(x, dtype=torch.float32),
                n_bins=int(self.n_bins),
            )
        return self

    def _standardized(self, frame: pd.DataFrame) -> np.ndarray:
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("NumericPreprocessor has not been fitted")
        x = frame.loc[:, list(self.feature_names_)].to_numpy(dtype=np.float64)
        return (x - self.means_) / self.stds_

    def _cat_codes(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [self.spout_to_index_.get(int(v), 0) for v in frame["spout_no"].astype(int).tolist()],
            dtype=np.int64,
        )

    def transform_mlp(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_std = self._standardized(frame)
        if self.uses_ple:
            torch = _torch()
            from rtdl_num_embeddings import PiecewiseLinearEncoding

            if self.bins_ is None:
                raise RuntimeError("PLE bins are unavailable")
            raw = frame.loc[:, list(self.feature_names_)].to_numpy(dtype=np.float32)
            encoder = PiecewiseLinearEncoding(self.bins_)
            with torch.no_grad():
                x_num = encoder(torch.as_tensor(raw, dtype=torch.float32)).numpy()
        else:
            x_num = x_std
        x_cat = np.zeros((len(frame), self.n_spout_categories_), dtype=np.float32)
        codes = self._cat_codes(frame)
        x_cat[np.arange(len(frame)), codes] = 1.0
        return np.asarray(x_num, dtype=np.float32), x_cat

    def transform_tabm(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_std = self._standardized(frame)
        x_num = x_std.astype(np.float32)
        if self.uses_ple:
            if self.bins_ is None:
                raise RuntimeError("PLE bins are unavailable")
            # TabM consumes raw numerical values together with the PLE module.
            x_num = frame.loc[:, list(self.feature_names_)].to_numpy(dtype=np.float32)
        else:
            x_num = x_std.astype(np.float32)
        x_cat = self._cat_codes(frame).reshape(-1, 1)
        return x_num, x_cat

    def metadata(self) -> dict[str, Any]:
        return {
            "structure": self.structure,
            "feature_names": list(self.feature_names_),
            "means": None if self.means_ is None else self.means_.tolist(),
            "stds": None if self.stds_ is None else self.stds_.tolist(),
            "n_bins": int(self.n_bins),
            "d_embedding": int(self.d_embedding),
            "spout_vocabulary": {str(k): int(v) for k, v in self.spout_to_index_.items()},
            "n_spout_categories": int(self.n_spout_categories_),
            "uses_ple": bool(self.uses_ple),
            "ple_bins_lengths": None if self.bins_ is None else [int(len(b)) for b in self.bins_],
        }


def _make_mlp(input_dim: int, hidden: Sequence[int], dropout: float):
    torch = _torch()
    layers: list[Any] = []
    previous = int(input_dim)
    for width in hidden:
        layers.extend([
            torch.nn.Linear(previous, int(width)),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)),
        ])
        previous = int(width)
    layers.append(torch.nn.Linear(previous, 1))
    return torch.nn.Sequential(*layers)


def _make_tabm(preprocessor: NumericPreprocessor, params: Mapping[str, Any]):
    torch = _torch()
    from tabm import TabM

    kwargs: dict[str, Any] = {
        "n_num_features": len(preprocessor.feature_names_),
        "cat_cardinalities": [int(preprocessor.n_spout_categories_)],
        "d_out": 1,
        "k": int(params["k"]),
        "n_blocks": int(params["n_blocks"]),
        "d_block": int(params["d_block"]),
        "dropout": float(params["dropout"]),
    }
    if preprocessor.uses_ple:
        from rtdl_num_embeddings import PiecewiseLinearEmbeddings

        if preprocessor.bins_ is None:
            raise RuntimeError("PLE bins are unavailable")
        kwargs["num_embeddings"] = PiecewiseLinearEmbeddings(
            preprocessor.bins_,
            d_embedding=int(preprocessor.d_embedding),
            activation=True,
            version="B",
        )
    return TabM.make(**kwargs)


def _batch_tensor(array: np.ndarray):
    torch = _torch()
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


class V36NetworkRegressor:
    """One numeric-encoding network recipe on a current training part."""

    def __init__(self, trial: Mapping[str, Any]):
        self.trial = deepcopy(dict(trial))
        self.structure = str(self.trial.get("structure"))
        if self.structure not in _NETWORK_STRUCTURES:
            raise ValueError(f"Unknown V3.6 network structure: {self.structure!r}")
        self.target = str(self.trial["target"])
        self.params = deepcopy(dict(self.trial.get("parameters", {})))
        if str(self.trial.get("target_transform")) != "train_mean_std":
            raise ValueError("V3.6 networks require target_transform=train_mean_std")
        required = {"loss", "optimizer", "learning_rate", "batch_size", "max_epochs",
                    "early_stopping_patience", "min_delta", "n_bins", "d_embedding"}
        missing = sorted(required - set(self.params))
        if missing:
            raise ValueError(f"V3.6 network parameters are missing: {missing}")
        if "tabm" in self.structure:
            for key in ("k", "n_blocks", "d_block", "dropout"):
                if key not in self.params:
                    raise ValueError(f"V3.6 TabM parameter is missing: {key}")
        else:
            for key in ("hidden", "dropout"):
                if key not in self.params:
                    raise ValueError(f"V3.6 MLP parameter is missing: {key}")

    def _criterion(self):
        torch = _torch()
        loss = str(self.params["loss"])
        if loss == "mse":
            return torch.nn.MSELoss()
        if loss == "mae":
            return torch.nn.L1Loss()
        if loss == "huber":
            return torch.nn.HuberLoss(delta=float(self.params.get("huber_delta", 1.0)))
        raise ValueError(f"Unknown V3.6 network loss: {loss!r}")

    @staticmethod
    def _predict_raw(model, structure: str, x_num, x_cat, *, train_mode: bool = False):
        torch = _torch()
        if structure in {"raw_mlp", "ple_mlp"}:
            x = torch.cat([x_num, x_cat], dim=1)
            if train_mode:
                return model(x)
            return model(x)
        out = model(x_num=x_num, x_cat=x_cat)
        return out

    def _loss_for_tabm(self, prediction, target):
        torch = _torch()
        criterion = self._criterion()
        if prediction.ndim == 2:
            prediction = prediction.unsqueeze(1)
        target = target.reshape(-1)
        terms = [criterion(prediction[:, i, 0], target) for i in range(prediction.shape[1])]
        return torch.stack(terms).mean()

    def _loss_for_mlp(self, prediction, target):
        criterion = self._criterion()
        return criterion(prediction.reshape(-1), target.reshape(-1))

    def _to_device(self, *tensors):
        # The frozen V3.6 environment is CPU.  Keeping the explicit device path
        # makes the code fail loudly rather than silently moving tensor devices.
        return tensors

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "V36NetworkRegressor":
        torch = _torch()
        started = time.perf_counter()
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(frame) != len(y):
            raise ValueError("V3.6 network frame/label length mismatch")
        if not np.isfinite(y).all():
            raise ValueError("V3.6 network labels must be finite")
        self.target_mean_ = float(y.mean())
        self.target_std_ = float(y.std(ddof=0))
        if self.target_std_ <= 0.0:
            raise ValueError("V3.6 network target standard deviation must be positive")
        _set_seed(int(self.params.get("random_seed", 42)))
        z = (y - self.target_mean_) / self.target_std_
        self.preprocessor_ = NumericPreprocessor(
            structure=self.structure,
            n_bins=int(self.params["n_bins"]),
            d_embedding=int(self.params["d_embedding"]),
        ).fit(frame)
        if self.structure in {"raw_mlp", "ple_mlp"}:
            x_num_np, x_cat_np = self.preprocessor_.transform_mlp(frame)
            input_dim = int(x_num_np.shape[1] + x_cat_np.shape[1])
            self.model_ = _make_mlp(input_dim, self.params["hidden"], float(self.params["dropout"]))
        else:
            self.model_ = _make_tabm(self.preprocessor_, self.params)
        # Group-safe internal validation, training labels only.
        folded = group_safe_inner_folds(
            frame, n_splits=int(self.params.get("inner_validation_folds", 5)),
            seed=int(self.params.get("inner_validation_seed", 42)),
        )
        val_fold = np.asarray(folded["fold"], dtype=int) == 0
        if not val_fold.any() or val_fold.all():
            raise ValueError("V3.6 network inner validation split is degenerate")
        train_index = np.flatnonzero(~val_fold)
        valid_index = np.flatnonzero(val_fold)
        train_frame = frame.iloc[train_index].reset_index(drop=True)
        valid_frame = frame.iloc[valid_index].reset_index(drop=True)
        train_z = z[train_index]
        valid_z = z[valid_index]
        if self.structure in {"raw_mlp", "ple_mlp"}:
            train_x_num, train_x_cat = self.preprocessor_.transform_mlp(train_frame)
            valid_x_num, valid_x_cat = self.preprocessor_.transform_mlp(valid_frame)
        else:
            train_x_num, train_x_cat = self.preprocessor_.transform_tabm(train_frame)
            valid_x_num, valid_x_cat = self.preprocessor_.transform_tabm(valid_frame)
        train_x_num_t = _batch_tensor(train_x_num)
        train_x_cat_t = torch.as_tensor(train_x_cat, dtype=torch.long if "tabm" in self.structure else torch.float32)
        valid_x_num_t = _batch_tensor(valid_x_num)
        valid_x_cat_t = torch.as_tensor(valid_x_cat, dtype=torch.long if "tabm" in self.structure else torch.float32)
        train_y_t = torch.as_tensor(train_z, dtype=torch.float32)
        valid_y_t = torch.as_tensor(valid_z, dtype=torch.float32)
        optimizer_name = str(self.params["optimizer"]).lower()
        lr = float(self.params["learning_rate"])
        wd = float(self.params.get("weight_decay", 0.0))
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.model_.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown V3.6 network optimizer: {optimizer_name!r}")
        batch_size = max(1, int(self.params["batch_size"]))
        max_epochs = max(1, int(self.params["max_epochs"]))
        patience = max(1, int(self.params["early_stopping_patience"]))
        min_delta = float(self.params["min_delta"])
        rng = np.random.default_rng(int(self.params.get("random_seed", 42)))
        best_state = deepcopy(self.model_.state_dict())
        best_val = math.inf
        best_epoch = 0
        stale = 0
        history: list[dict[str, float]] = []
        for epoch in range(1, max_epochs + 1):
            self.model_.train()
            order = rng.permutation(len(train_frame))
            train_losses: list[float] = []
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                xb_num = train_x_num_t[batch]
                xb_cat = train_x_cat_t[batch]
                yb = train_y_t[batch]
                optimizer.zero_grad(set_to_none=True)
                prediction = self._predict_raw(self.model_, self.structure, xb_num, xb_cat, train_mode=True)
                if "tabm" in self.structure:
                    loss = self._loss_for_tabm(prediction, yb)
                else:
                    loss = self._loss_for_mlp(prediction, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
            self.model_.eval()
            with torch.no_grad():
                valid_prediction = self._predict_raw(
                    self.model_, self.structure, valid_x_num_t, valid_x_cat_t, train_mode=False
                )
                if "tabm" in self.structure:
                    val_loss = float(self._loss_for_tabm(valid_prediction, valid_y_t).cpu().item())
                else:
                    val_loss = float(self._loss_for_mlp(valid_prediction, valid_y_t).cpu().item())
            train_loss = float(np.mean(train_losses)) if train_losses else math.inf
            history.append({"epoch": int(epoch), "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_epoch = int(epoch)
                best_state = deepcopy(self.model_.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        self.model_.load_state_dict(best_state)
        self.best_epoch_ = int(best_epoch)
        self.best_validation_loss_ = float(best_val)
        self.stopped_epoch_ = int(history[-1]["epoch"])
        self.history_ = history
        self.fit_meta_ = {
            "model_kind": self.structure,
            "target": self.target,
            "target_mean": float(self.target_mean_),
            "target_std": float(self.target_std_),
            "parameters": deepcopy(self.params),
            "preprocessing": self.preprocessor_.metadata(),
            "input_num_dim": int(train_x_num.shape[1]) if self.structure in {"raw_mlp", "ple_mlp"} else len(self.preprocessor_.feature_names_),
            "input_cat_dim": int(train_x_cat.shape[1]) if self.structure in {"raw_mlp", "ple_mlp"} else 1,
            "best_epoch": int(self.best_epoch_),
            "stopped_epoch": int(self.stopped_epoch_),
            "best_validation_loss": float(self.best_validation_loss_),
            "history": history,
            "seconds": float(time.perf_counter() - started),
            "training_protocol": "numeric-network-train-fold-only-v1",
        }
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_") or not hasattr(self, "preprocessor_"):
            raise RuntimeError("V3.6 network must be fitted before prediction")
        torch = _torch()
        self.model_.eval()
        if self.structure in {"raw_mlp", "ple_mlp"}:
            x_num_np, x_cat_np = self.preprocessor_.transform_mlp(frame)
            x_num = _batch_tensor(x_num_np)
            x_cat = torch.as_tensor(x_cat_np, dtype=torch.float32)
            with torch.no_grad():
                prediction = self._predict_raw(self.model_, self.structure, x_num, x_cat, train_mode=False)
            raw = prediction.detach().cpu().numpy().reshape(-1)
        else:
            x_num_np, x_cat_np = self.preprocessor_.transform_tabm(frame)
            x_num = _batch_tensor(x_num_np)
            x_cat = torch.as_tensor(x_cat_np, dtype=torch.long)
            with torch.no_grad():
                prediction = self._predict_raw(self.model_, self.structure, x_num, x_cat, train_mode=False)
            if prediction.ndim != 3:
                raise ValueError("V3.6 TabM prediction must have shape (n, k, 1)")
            raw = prediction.detach().cpu().numpy().mean(axis=1).reshape(-1)
        result = raw * self.target_std_ + self.target_mean_
        if result.shape != (len(frame),) or not np.isfinite(result).all():
            raise ValueError("Invalid V3.6 network predictions")
        return result.astype(float, copy=False)


__all__ = ["NumericPreprocessor", "V36NetworkRegressor"]
