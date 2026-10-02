"""Engineered-feature and mixup arms for the joint TabM recipe.

Engineered columns are deterministic functions of the raw features only; the
preprocessing (train-part standardisation) and every other protocol detail stay
identical to the incumbent recipe.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import FEATURES
from .q75_schedule_screen import ScheduledJointRegressor
from .v3_6_networks import NumericPreprocessor

LOG_COLUMNS = ("air_volume", "oxygen", "coke_rate", "fuel_rate", "consumption",
               "pig", "air_speed", "total_press_diff")
LOG_RATIOS = (
    ("oxygen", "air_volume"), ("coal_rate", "air_volume"), ("coke_rate", "air_volume"),
    ("fuel_rate", "air_volume"), ("air_volume", "air_speed"),
    ("hot_air_press", "cold_air_press"), ("furnace_top_press", "total_press_diff"),
    ("consumption", "pig"), ("hot_air_temp", "furnace_top_temp_avg"),
)
PRODUCTS = (
    ("air_volume", "oxygen"), ("hot_air_press", "hot_air_temp"),
)


def engineered_columns() -> tuple[str, ...]:
    names = [f"log_{name}" for name in LOG_COLUMNS]
    names.append("log1p_humidity")
    names += [f"logratio_{left}_over_{right}" for left, right in LOG_RATIOS]
    names.append("logratio_humidity1_over_air_volume")
    names += [f"product_{left}_x_{right}" for left, right in PRODUCTS]
    names.append("product_air_volume_x_fuel_sum")
    names.append("product_air_volume_x_humidity1")
    return tuple(names)


ENGINEERED = engineered_columns()


def augment(frame):
    """Return a copy of ``frame`` with the engineered feature columns appended."""
    out = frame.copy()
    values = frame[list(FEATURES)]
    for name in LOG_COLUMNS:
        out[f"log_{name}"] = np.log(values[name].to_numpy(float))
    out["log1p_humidity"] = np.log1p(values["humidity"].to_numpy(float))
    for left, right in LOG_RATIOS:
        out[f"logratio_{left}_over_{right}"] = np.log(
            values[left].to_numpy(float) / values[right].to_numpy(float))
    out["logratio_humidity1_over_air_volume"] = np.log(
        (values["humidity"].to_numpy(float) + 1.0) / values["air_volume"].to_numpy(float))
    for left, right in PRODUCTS:
        out[f"product_{left}_x_{right}"] = (values[left].to_numpy(float)
                                            * values[right].to_numpy(float))
    out["product_air_volume_x_fuel_sum"] = values["air_volume"].to_numpy(float) * (
        values["coal_rate"].to_numpy(float) + values["coke_rate"].to_numpy(float))
    out["product_air_volume_x_humidity1"] = values["air_volume"].to_numpy(float) * (
        values["humidity"].to_numpy(float) + 1.0)
    return out


class ExtendedPreprocessor(NumericPreprocessor):
    """NumericPreprocessor with an explicit feature list."""

    def __init__(self, feature_names, *, structure: str = "raw_tabm"):
        super().__init__(structure=structure)
        self.feature_names_ = tuple(feature_names)


def make_network(recipe, settings, n_categories, n_outputs, n_num_features):
    from rtdl_num_embeddings import PeriodicEmbeddings
    from tabm import TabM

    embedding = None
    if recipe["frequency"] is not None:
        embedding = PeriodicEmbeddings(
            n_num_features, d_embedding=settings["embedding_dim"],
            n_frequencies=settings["n_frequencies"],
            frequency_init_scale=recipe["frequency"], lite=settings["lite"])
    return TabM.make(
        n_num_features=n_num_features, cat_cardinalities=[n_categories],
        d_out=n_outputs, k=settings["tabm_k"], n_blocks=settings["blocks"],
        d_block=settings["width"], dropout=settings["dropout"],
        num_embeddings=embedding)


class AugmentedJointRegressor(ScheduledJointRegressor):
    """Joint Regressor over an explicit feature list with optional mixup."""

    def __init__(self, recipe, settings, feature_names):
        super().__init__(recipe, settings)
        self.feature_names_ = tuple(feature_names)

    def _initialize(self, frame, y):
        torch.set_num_threads(1)
        torch.manual_seed(self.settings["random_seed"])
        self.preprocessor_ = ExtendedPreprocessor(self.feature_names_).fit(frame)
        self.mean_, self.std_ = np.mean(y, axis=0), np.std(y, axis=0)
        if not (self.std_ > 0).all():
            raise ValueError("Constant target")
        self.model_ = make_network(self.recipe, self.settings,
                                   self.preprocessor_.n_spout_categories_, y.shape[1],
                                   len(self.feature_names_))
        self.optimizer_ = torch.optim.AdamW(self.model_.parameters(),
                                            lr=self.settings["learning_rate"],
                                            weight_decay=self.settings["weight_decay"])

    def _train(self, frame, y, epochs, validation=None):
        alpha = float(self.settings.get("mixup_alpha", 0.0) or 0.0)
        if alpha <= 0:
            return super()._train(frame, y, epochs, validation)
        loss_name = self.settings.get("loss", "mse")
        head_loss = self.settings.get("head_loss")
        x, cat = self._inputs(frame)
        target = torch.as_tensor((y - self.mean_) / self.std_, dtype=torch.float32)
        rng = np.random.default_rng(self.settings["random_seed"])
        best, best_epoch, stale = float("inf"), 0, 0
        if validation is not None:
            vx, vc = self._inputs(validation[0])
            vy = torch.as_tensor((validation[1] - self.mean_) / self.std_, dtype=torch.float32)
        for epoch in range(1, epochs + 1):
            self.model_.train()
            order = rng.permutation(len(frame))
            for start in range(0, len(order), self.settings["batch_size"]):
                idx = order[start:start + self.settings["batch_size"]]
                mixed_x, mixed_y = x[idx], target[idx]
                if len(idx) > 1:
                    lam = float(rng.beta(alpha, alpha))
                    permutation = rng.permutation(len(idx))
                    mixed_x = lam * x[idx] + (1 - lam) * x[idx][permutation]
                    mixed_y = lam * target[idx] + (1 - lam) * target[idx][permutation]
                self.optimizer_.zero_grad(set_to_none=True)
                pred = self.model_(mixed_x, cat[idx])
                if head_loss:
                    error = pred - mixed_y[:, None, :]
                    terms = [(error[..., index].square().mean() if str(kind) == "mse"
                              else error[..., index].abs().mean())
                             for index, kind in enumerate(head_loss)]
                    loss = sum(terms) / len(terms)
                elif loss_name == "mse":
                    loss = (pred - mixed_y[:, None, :]).square().mean()
                else:
                    loss = (pred - mixed_y[:, None, :]).abs().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                self.optimizer_.step()
            if validation is not None:
                self.model_.eval()
                with torch.no_grad():
                    pred = self.model_(vx, vc).mean(1)
                    value = float((pred - vy).abs().mean())
                if value < best - self.settings["min_delta"]:
                    best, best_epoch, stale = value, epoch, 0
                else:
                    stale += 1
                if stale >= self.settings["patience"]:
                    break
        if validation is not None:
            self.selection_stopped_epoch_ = epoch
        return best_epoch if validation is not None else epochs
