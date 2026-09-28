"""Continuous conditional Gaussian mixtures with median point prediction."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtr
import torch
from torch import nn
from torch.nn import functional as F

from .data import FEATURES, TARGETS
from .v3_6_networks import NumericPreprocessor
from .v7_periodic import digest
from .v30_deep_kernel import select_weight


def mixture_nll(y, logits, means, scales):
    log_density = -.5 * ((y[:, None] - means) / scales).square()
    log_density = log_density - scales.log() - .5 * math.log(2 * math.pi)
    return -torch.logsumexp(F.log_softmax(logits, dim=1) + log_density, dim=1).mean()


def mixture_median(weights, means, scales, steps=48):
    """The mixture median lies between the smallest/largest component means."""
    weights, means, scales = [np.asarray(v, dtype=float) for v in (weights, means, scales)]
    if (weights.ndim != 2 or weights.shape != means.shape or scales.shape != means.shape
            or not all(np.isfinite(v).all() for v in (weights, means, scales))
            or (weights < 0).any() or not np.allclose(weights.sum(1), 1)
            or (scales <= 0).any() or steps < 1):
        raise ValueError("Invalid Gaussian mixture")
    low, high = means.min(1), means.max(1)
    for _ in range(steps):
        mid = (low + high) / 2
        cdf = (weights * ndtr((mid[:, None] - means) / scales)).sum(1)
        low, high = np.where(cdf < .5, mid, low), np.where(cdf < .5, high, mid)
    return (low + high) / 2


class MixtureNetwork(nn.Module):
    def __init__(self, recipe, settings, categories):
        super().__init__()
        from rtdl_num_embeddings import PeriodicEmbeddings
        if recipe not in ("GAUSS1", "MDN3"):
            raise ValueError("Unknown mixture recipe")
        self.components = 1 if recipe == "GAUSS1" else 3
        self.floor = settings["scale_floor"]
        self.embedding = PeriodicEmbeddings(len(FEATURES),
            d_embedding=settings["periodic_embedding_dim"],
            n_frequencies=settings["periodic_n_frequencies"],
            frequency_init_scale=settings["periodic_frequency_init_scale"], lite=True)
        width = len(FEATURES) * settings["periodic_embedding_dim"] + categories
        layers = []
        for hidden in settings["hidden_widths"]:
            layers.extend([nn.Linear(width, hidden), nn.SiLU()])
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, 3 * self.components)
        nn.init.normal_(self.head.weight, std=.01)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[self.components:2*self.components] = torch.tensor(
                [0.] if self.components == 1 else settings["initial_component_means"])
            self.head.bias[2*self.components:] = math.log(math.expm1(settings["initial_scale"] - self.floor))

    def forward(self, x):
        z = torch.cat([self.embedding(x[:, :len(FEATURES)]).flatten(1), x[:, len(FEATURES):]], 1)
        logits, means, raw_scales = self.head(self.trunk(z)).chunk(3, dim=1)
        return logits, means, F.softplus(raw_scales) + self.floor


class MixtureRegressor:
    def __init__(self, recipe, settings):
        self.recipe, self.settings = recipe, dict(settings)

    def _inputs(self, frame):
        numeric, cat = self.preprocessor_.transform_mlp(frame)
        x = np.concatenate([numeric, cat], axis=1).astype(np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite mixture inputs")
        return torch.as_tensor(x, dtype=torch.float64)

    def initialize(self, frame, y):
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or not y.std() > 0:
            raise ValueError("Invalid training target")
        torch.set_num_threads(1)
        torch.manual_seed(self.settings["random_seed"])
        self.preprocessor_ = NumericPreprocessor(structure="raw_mlp").fit(frame)
        self.mean_, self.std_ = float(y.mean()), float(y.std())
        self.fit_ids_ = frame.sample_id.astype(str).tolist()
        self.model_ = MixtureNetwork(self.recipe, self.settings, self.preprocessor_.n_spout_categories_).double()
        self.x_train_, self.y_train_ = self._inputs(frame), torch.as_tensor((y-self.mean_)/self.std_)
        self.optimizer_ = torch.optim.AdamW(self.model_.parameters(), lr=self.settings["learning_rate"],
                                            weight_decay=self.settings["weight_decay"])
        return self

    def train(self, epochs, validation=None):
        rng = np.random.default_rng(self.settings["random_seed"])
        best, stale, best_epoch, best_state = float("inf"), 0, 0, None
        self.history_ = []
        for epoch in range(1, epochs+1):
            self.model_.train()
            order, loss_sum = rng.permutation(len(self.y_train_)), 0.
            for start in range(0, len(order), self.settings["batch_size"]):
                idx = order[start:start+self.settings["batch_size"]]
                self.optimizer_.zero_grad(set_to_none=True)
                loss = mixture_nll(self.y_train_[idx], *self.model_(self.x_train_[idx]))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite mixture likelihood")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), self.settings["gradient_norm_clip"], error_if_nonfinite=True)
                self.optimizer_.step()
                loss_sum += float(loss.detach()) * len(idx)
            row = {"epoch": epoch, "negative_log_likelihood": loss_sum/len(order)}
            if validation is not None:
                value = float(np.abs(self.predict(validation[0]) - validation[1]).mean()/self.std_)
                row["calibration_standardized_mae"] = value
                if value < best - self.settings["min_delta_standardized_mae"]:
                    best, best_epoch, stale = value, epoch, 0
                    best_state = deepcopy(self.model_.state_dict())
                else:
                    stale += 1
            self.history_.append(row)
            if validation is not None and stale >= self.settings["patience"]:
                break
        if validation is not None:
            if best_state is None:
                raise ValueError("No finite selected epoch")
            self.model_.load_state_dict(best_state)
        else:
            best_epoch = epoch
        self.selected_epoch_, self.stopped_epoch_ = best_epoch, epoch
        return best_epoch

    def predict(self, frame):
        self.model_.eval()
        with torch.no_grad():
            logits, means, scales = self.model_(self._inputs(frame))
            weights = torch.softmax(logits, 1).numpy()
            prediction = mixture_median(weights, means.numpy(), scales.numpy(), self.settings["median_bisection_steps"])
        values = prediction * self.std_ + self.mean_
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ValueError("Invalid mixture predictions")
        return values

    def metadata(self):
        return {"recipe": self.recipe, "selected_epoch": self.selected_epoch_,
                "stopped_epoch": self.stopped_epoch_, "fit_ids_digest": digest(self.fit_ids_),
                "fit_rows": len(self.fit_ids_), "preprocessing": self.preprocessor_.metadata(),
                "target_mean": self.mean_, "target_std": self.std_, "history": self.history_,
                "parameter_count": sum(p.numel() for p in self.model_.parameters())}

    def save(self, path):
        payload = {"recipe": self.recipe, "settings": self.settings,
                   "preprocessing": self.preprocessor_.metadata(), "mean": self.mean_,
                   "std": self.std_, "state": self.model_.state_dict()}
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(payload["recipe"], payload["settings"])
        info = payload["preprocessing"]
        model.preprocessor_ = NumericPreprocessor(structure="raw_mlp")
        model.preprocessor_.means_ = np.array(info["means"])
        model.preprocessor_.stds_ = np.array(info["stds"])
        model.preprocessor_.spout_to_index_ = {int(k): v for k, v in info["spout_vocabulary"].items()}
        model.preprocessor_.n_spout_categories_ = info["n_spout_categories"]
        model.model_ = MixtureNetwork(model.recipe, model.settings, info["n_spout_categories"]).double()
        model.model_.load_state_dict(payload["state"])
        model.mean_, model.std_ = payload["mean"], payload["std"]
        return model


def fit_partition(fitting, calibration, outer_training, query, target, recipe,
                  settings, calibration_base, grid):
    if set(fitting.sample_id) & set(calibration.sample_id):
        raise ValueError("Fitting/calibration overlap")
    if set(outer_training.sample_id) != set(fitting.sample_id) | set(calibration.sample_id):
        raise ValueError("Incorrect outer training union")
    if set(outer_training.sample_id) & set(query.sample_id):
        raise ValueError("Outer query overlaps training")
    if any(c in query.columns for c in TARGETS):
        raise ValueError("Query labels must be removed")
    selector = MixtureRegressor(recipe, settings).initialize(fitting, fitting[target].to_numpy())
    epoch = selector.train(settings["max_epochs"], (calibration, calibration[target].to_numpy()))
    cp = selector.predict(calibration)
    weight, losses = select_weight(calibration[target], calibration_base, cp, grid)
    final = MixtureRegressor(recipe, settings).initialize(outer_training, outer_training[target].to_numpy())
    final.train(epoch)
    prediction = final.predict(query)
    final.calibration_model_ = selector
    return final, prediction, {"weight": weight, "calibration_mae_by_weight": losses,
                              "calibration": selector.metadata(), "refit": final.metadata()}, cp
