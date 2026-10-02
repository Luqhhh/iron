"""GRANDE-derived hard-tree regressor on the frozen round-two folds.

The model and training settings are taken verbatim from ``configs/round2_v37``;
only the data plumbing (train-fold standardisation, spout one-hot, epoch
selection, refit) follows the project's numeric-network protocol.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import FEATURES
from .v37_hard_tree import HardTreeEnsemble

TARGETS = ("tap_iron", "tap_time_len")


class HardTreeRegressor:
    """Single-target hard-tree regressor with inner-validation epoch selection."""

    def __init__(self, arm: str, settings: dict):
        self.arm = arm
        self.settings = dict(settings)
        self.training_ = dict(self.settings["training"])
        self.model_settings_ = dict(self.settings["model"])
        self.metadata_: dict = {}
        self.snapshot_predictions_ = None

    # ------------------------------------------------------------------ inputs
    def _build(self, frame):
        numeric = frame[list(FEATURES)].to_numpy(float)
        spout = frame["spout_no"].to_numpy()
        categories = np.zeros((len(frame), 3))
        for index, value in enumerate(spout):
            categories[index, 1 if int(value) == 1 else 2] = 1.0
        return np.column_stack([numeric, categories])

    def _initialize(self, frame, y):
        x = self._build(frame)
        self.means_ = x[:, :len(FEATURES)].mean(0)
        self.stds_ = x[:, :len(FEATURES)].std(0)
        self.stds_[self.stds_ == 0] = 1.0
        x = x.copy()
        x[:, :len(FEATURES)] = (x[:, :len(FEATURES)] - self.means_) / self.stds_
        self.target_mean_ = float(np.mean(y))
        self.target_std_ = float(np.std(y))
        if self.target_std_ <= 0:
            raise ValueError("Constant target")
        torch.set_num_threads(1)
        torch.manual_seed(int(self.training_.get("random_seed", 42)))
        self.model_ = HardTreeEnsemble(x.shape[1], self.arm, self.model_settings_)
        rates = self.training_["learning_rates"]
        self.optimizer_ = torch.optim.Adam(
            [{"params": [getattr(self.model_, name)], "lr": rate} for name, rate in rates.items()],
            betas=tuple(self.training_["betas"]), eps=self.training_["eps"],
            weight_decay=self.training_["weight_decay"])
        return x

    def _transform(self, frame):
        x = self._build(frame)
        x[:, :len(FEATURES)] = (x[:, :len(FEATURES)] - self.means_) / self.stds_
        return x

    # ----------------------------------------------------------------- training
    def _train(self, frame, y, epochs, validation=None):
        x = torch.as_tensor(self._build_scaled(frame), dtype=torch.float32)
        target = torch.as_tensor((y - self.target_mean_) / self.target_std_, dtype=torch.float32)
        rng = np.random.default_rng(int(self.training_.get("random_seed", 42)))
        batch = int(self.training_["batch_size"])
        patience = int(self.training_["patience"])
        min_delta = float(self.training_["min_delta"])
        if validation is not None:
            vx = torch.as_tensor(self._build_scaled(validation[0]), dtype=torch.float32)
            vy = torch.as_tensor((validation[1] - self.target_mean_) / self.target_std_,
                                 dtype=torch.float32)
        best, best_epoch, stale = float("inf"), 0, 0
        for epoch in range(1, epochs + 1):
            self.model_.train()
            order = rng.permutation(len(frame))
            for start in range(0, len(order), batch):
                idx = order[start:start + batch]
                self.optimizer_.zero_grad(set_to_none=True)
                loss = (self.model_(x[idx]) - target[idx]).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                self.optimizer_.step()
            if validation is not None:
                self.model_.eval()
                with torch.no_grad():
                    value = float((self.model_(vx) - vy).abs().mean())
                if value < best - min_delta:
                    best, best_epoch, stale = value, epoch, 0
                else:
                    stale += 1
                if stale >= patience:
                    break
        self.stopped_epoch_ = epoch
        return best_epoch if validation is not None else epochs

    def _build_scaled(self, frame):
        return self._transform(frame).astype(np.float32)

    def fit(self, frame, target):
        from .v3_4_bags import group_safe_inner_folds

        y = np.asarray(target, dtype=float)
        if y.ndim != 1 or len(y) != len(frame) or not np.isfinite(y).all():
            raise ValueError("Invalid training target")
        folds = group_safe_inner_folds(frame, seed=int(self.settings.get("inner_seed", 42)))["fold"]
        mask = folds != 0
        inner = frame.loc[mask].reset_index(drop=True)
        self._initialize(inner, y[mask])
        inner_mean, inner_std = self.target_mean_, self.target_std_
        epoch = self._train(inner, y[mask], int(self.training_["max_epochs"]),
                            (frame.loc[~mask].reset_index(drop=True), y[~mask]))
        if epoch < 1:
            raise ValueError("No finite inner-validation epoch")
        self._initialize(frame, y)
        self._train(frame, y, epoch)
        self.model_.eval()
        self.metadata_ = {"selected_epoch": int(epoch),
                          "selection_stopped_epoch": int(self.stopped_epoch_),
                          "budget_limited": bool(self.stopped_epoch_ >= int(self.training_["max_epochs"])),
                          "fit_rows": int(len(frame)),
                          "inner_fit_rows": int(mask.sum()),
                          "inner_target_mean": inner_mean,
                          "inner_target_std": inner_std,
                          "outer_target_mean": self.target_mean_,
                          "outer_target_std": self.target_std_,
                          "optimizer_runs": 2}
        return self

    def predict(self, frame):
        x = torch.as_tensor(self._build_scaled(frame), dtype=torch.float32)
        self.model_.eval()
        chunk = int(self.training_.get("prediction_chunk_rows", 256))
        parts = []
        with torch.no_grad():
            for start in range(0, len(x), chunk):
                parts.append(self.model_(x[start:start + chunk]).numpy().astype(float))
        result = np.concatenate(parts) * self.target_std_ + self.target_mean_
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite prediction")
        return result
