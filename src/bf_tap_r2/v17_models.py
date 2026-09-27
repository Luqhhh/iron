"""Frozen V17 mechanisms: iron-led joint optimization and dual periodic embeddings."""
from __future__ import annotations

import numpy as np
import torch
from rtdl_num_embeddings import PeriodicEmbeddings
from tabm import TabM

from .data import FEATURES, TARGETS
from .v12_joint import JointRegressor
from .v7_periodic import PeriodicRegressor
from .v3_6_networks import NumericPreprocessor


class DualPeriodicEmbeddings(torch.nn.Module):
    """Two eight-dimensional branches; the TabM input remains 16 dimensions."""

    def __init__(self, scales):
        super().__init__()
        self.branches = torch.nn.ModuleList([
            PeriodicEmbeddings(len(FEATURES), d_embedding=8, n_frequencies=8,
                               frequency_init_scale=float(scale), lite=True)
            for scale in scales
        ])

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=-1)

    def diagnostics(self):
        return {
            "frequency_weight_std": [float(branch.periodic.weight.detach().std())
                                     for branch in self.branches],
            "frequency_weight_norm": [float(branch.periodic.weight.detach().norm())
                                      for branch in self.branches],
            "frequency_weight_abs_mean": [float(branch.periodic.weight.detach().abs().mean())
                                          for branch in self.branches],
        }

    def activation_diagnostics(self, x):
        with torch.no_grad():
            outputs = [branch(x).reshape(len(x), -1) for branch in self.branches]
            return {
                "activation_rms": [float(v.square().mean().sqrt()) for v in outputs],
                "activation_mean_abs": [float(v.abs().mean()) for v in outputs],
            }


def dual_network(settings, n_categories, n_outputs, scales):
    # TabM's constructor accepts only its registered embedding classes. Build
    # the unchanged 16-dimension backbone, then install an equal-width module.
    placeholder = PeriodicEmbeddings(len(FEATURES), d_embedding=16, n_frequencies=16,
                                     frequency_init_scale=.01, lite=True)
    model = TabM.make(
        n_num_features=len(FEATURES), cat_cardinalities=[n_categories],
        d_out=n_outputs, k=settings["tabm_k"], n_blocks=settings["blocks"],
        d_block=settings["width"], dropout=settings["dropout"],
        num_embeddings=placeholder,
    )
    model.num_module = DualPeriodicEmbeddings(scales)
    return model


def projected_task_grads(shared, private, iron_loss, time_loss, eps=1e-20):
    """Project only a conflicting auxiliary gradient on shared parameters.

    Output parameters keep their original per-target 0.5 MSE gradients. The
    caller must compute each task loss over all TabM submodels before calling.
    """
    params = list(shared) + list(private)
    gi = torch.autograd.grad(iron_loss, params, retain_graph=True)
    gt = torch.autograd.grad(time_loss, params)
    n_shared = len(shared)
    dot = sum((a.detach() * b.detach()).sum() for a, b in zip(gi[:n_shared], gt[:n_shared]))
    norm_i = sum(a.detach().square().sum() for a in gi[:n_shared])
    norm_t = sum(a.detach().square().sum() for a in gt[:n_shared])
    conflict = bool(dot < 0 and norm_i > eps)
    coefficient = dot / norm_i if conflict else 0.0
    with torch.no_grad():
        for p, a, b in zip(shared, gi[:n_shared], gt[:n_shared]):
            p.grad = (0.5 * (a + b - coefficient * a)).detach()
        for p, a, b in zip(private, gi[n_shared:], gt[n_shared:]):
            p.grad = (0.5 * (a + b)).detach()
    ni, nt = float(torch.sqrt(norm_i)), float(torch.sqrt(norm_t))
    cosine = float(dot / torch.sqrt(norm_i * norm_t)) if ni and nt else 0.0
    return {"conflict": conflict, "cosine": cosine, "iron_norm": ni,
            "time_norm": nt, "projection_norm": float(abs(coefficient) * torch.sqrt(norm_i)) if conflict else 0.0}


class V17JointRegressor(JointRegressor):
    def __init__(self, recipe, settings):
        super().__init__({"backbone": "tabm", "frequency": .01}, settings)
        self.v17_recipe = dict(recipe)
        if self.v17_recipe["kind"] != "joint":
            raise ValueError("Joint recipe required")

    def _initialize(self, frame, y):
        if self.v17_recipe["dual_scales"] is None:
            return super()._initialize(frame, y)
        torch.set_num_threads(1)
        torch.manual_seed(self.settings["random_seed"])
        self.preprocessor_ = NumericPreprocessor(structure="raw_tabm").fit(frame)
        self.mean_, self.std_ = np.mean(y, axis=0), np.std(y, axis=0)
        if not (self.std_ > 0).all():
            raise ValueError("Constant target")
        self.model_ = dual_network(self.settings, self.preprocessor_.n_spout_categories_, 2,
                                   self.v17_recipe["dual_scales"])
        self.optimizer_ = torch.optim.AdamW(self.model_.parameters(),
                                           lr=self.settings["learning_rate"],
                                           weight_decay=self.settings["weight_decay"])

    def _train(self, frame, y, epochs, validation=None):
        x, cat = self._inputs(frame)
        target = torch.as_tensor((y - self.mean_) / self.std_, dtype=torch.float32)
        rng = np.random.default_rng(self.settings["random_seed"])
        best, best_epoch, stale = float("inf"), 0, 0
        trace, gradients = [], []
        if validation is not None:
            vx, vc = self._inputs(validation[0])
            vy = torch.as_tensor(validation[1], dtype=torch.float32)
        if self.v17_recipe["projection"]:
            shared = [p for name, p in self.model_.named_parameters() if not name.startswith("output.")]
            private = [p for name, p in self.model_.named_parameters() if name.startswith("output.")]
            if not shared or len(private) != 2:
                raise ValueError("Unexpected TabM shared/output parameter partition")
        for epoch in range(1, epochs + 1):
            self.model_.train()
            order = rng.permutation(len(frame))
            for start in range(0, len(frame), self.settings["batch_size"]):
                idx = order[start:start + self.settings["batch_size"]]
                self.optimizer_.zero_grad(set_to_none=True)
                prediction = self.model_(x[idx], cat[idx])
                iron_loss = (prediction[:, :, 0] - target[idx, None, 0]).square().mean()
                time_loss = (prediction[:, :, 1] - target[idx, None, 1]).square().mean()
                if not torch.isfinite(iron_loss + time_loss):
                    raise ValueError("Nonfinite joint loss")
                if self.v17_recipe["projection"]:
                    gradients.append(projected_task_grads(shared, private, iron_loss, time_loss))
                else:
                    (0.5 * (iron_loss + time_loss)).backward()
                self.optimizer_.step()
            if validation is not None:
                self.model_.eval()
                with torch.no_grad():
                    pred = self.model_(vx, vc).mean(1).numpy()
                    raw = pred * self.std_ + self.mean_
                    iron_mae = float(np.abs(raw[:, 0] - vy.numpy()[:, 0]).mean())
                    time_mae = float(np.abs(raw[:, 1] - vy.numpy()[:, 1]).mean())
                    trace.append({"epoch": epoch, "iron_raw_mae": iron_mae, "time_raw_mae": time_mae})
                    value = iron_mae
                # V12 min_delta is standardized; convert to raw iron units.
                if value < best - self.settings["min_delta"] * self.std_[0]:
                    best, best_epoch, stale = value, epoch, 0
                else:
                    stale += 1
                if stale >= self.settings["patience"]:
                    break
        if validation is not None:
            self.selection_stopped_epoch_ = epoch
            self.selection_trace_ = trace
            self.gradient_diagnostics_ = {
                "batches": len(gradients),
                "conflict_fraction": float(np.mean([g["conflict"] for g in gradients])) if gradients else None,
                "mean_cosine": float(np.mean([g["cosine"] for g in gradients])) if gradients else None,
                "mean_iron_norm": float(np.mean([g["iron_norm"] for g in gradients])) if gradients else None,
                "mean_time_norm": float(np.mean([g["time_norm"] for g in gradients])) if gradients else None,
                "mean_projection_norm": float(np.mean([g["projection_norm"] for g in gradients])) if gradients else None,
            }
        return best_epoch if validation is not None else epochs

    def fit(self, frame, y):
        if np.asarray(y).shape != (len(frame), 2):
            raise ValueError("V17 joint model requires ordered iron/time targets")
        super().fit(frame, y)
        self.metadata_["selection_trace"] = self.selection_trace_
        self.metadata_["gradient_diagnostics"] = self.gradient_diagnostics_
        self.metadata_["parameter_count"] = sum(p.numel() for p in self.model_.parameters())
        if self.v17_recipe["dual_scales"] is not None:
            x, _ = self._inputs(frame)
            self.metadata_["dual_periodic"] = {
                **self.model_.num_module.diagnostics(),
                **self.model_.num_module.activation_diagnostics(x),
            }
        return self


class V17TimeRegressor(PeriodicRegressor):
    def __init__(self, recipe, settings):
        super().__init__({"backbone": "tabm", "frequency": .01}, settings)
        self.v17_recipe = dict(recipe)
        if self.v17_recipe["kind"] != "single":
            raise ValueError("Single-output recipe required")

    def _initialize(self, frame, y):
        torch.set_num_threads(1)
        torch.manual_seed(self.settings["random_seed"])
        self.preprocessor_ = NumericPreprocessor(structure="raw_tabm").fit(frame)
        self.mean_, self.std_ = float(np.mean(y)), float(np.std(y))
        if not self.std_ > 0:
            raise ValueError("Constant target")
        self.model_ = dual_network(self.settings, self.preprocessor_.n_spout_categories_, 1,
                                   self.v17_recipe["dual_scales"])
        self.optimizer_ = torch.optim.AdamW(self.model_.parameters(),
                                           lr=self.settings["learning_rate"],
                                           weight_decay=self.settings["weight_decay"])

    def _train(self, frame, y, epochs, validation=None):
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
                self.optimizer_.zero_grad(set_to_none=True)
                prediction = self.model_(x[idx], cat[idx])[:, :, 0]
                loss = (prediction - target[idx, None]).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite time loss")
                loss.backward()
                self.optimizer_.step()
            if validation is not None:
                self.model_.eval()
                with torch.no_grad():
                    pred = self.model_(vx, vc).mean(1).flatten()
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

    def fit(self, frame, y):
        super().fit(frame, y)
        self.metadata_["parameter_count"] = sum(p.numel() for p in self.model_.parameters())
        self.metadata_["selection_stopped_epoch"] = self.selection_stopped_epoch_
        self.metadata_["budget_limited"] = self.selection_stopped_epoch_ >= self.settings["max_epochs"]
        x, _ = self._inputs(frame)
        self.metadata_["dual_periodic"] = {
            **self.model_.num_module.diagnostics(),
            **self.model_.num_module.activation_diagnostics(x),
        }
        return self


def fit_fold(frame, folds, name, recipe, settings, fold):
    mask = folds == fold
    training = frame.loc[~mask].reset_index(drop=True)
    query = frame.loc[mask].drop(columns=list(TARGETS)).reset_index(drop=True)
    if recipe["kind"] == "joint":
        model = V17JointRegressor(recipe, settings).fit(training, training[list(TARGETS)].to_numpy())
        pred = model.predict(query)[:, 0]
    else:
        model = V17TimeRegressor(recipe, settings).fit(training, training["tap_time_len"].to_numpy())
        pred = model.predict(query)
    if pred.shape != (int(mask.sum()),) or not np.isfinite(pred).all():
        raise ValueError("Invalid V17 fold prediction")
    return pred, model.metadata_
