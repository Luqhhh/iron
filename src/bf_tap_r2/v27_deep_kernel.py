"""V27 sparse variational deep kernels, fitted only on supplied partitions."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import FEATURES
from .v3_6_networks import NumericPreprocessor
from .v7_periodic import digest


def farthest_rows(x: np.ndarray, count: int, seed: int) -> np.ndarray:
    """A deterministic, label-free inducing subset of the current training rows."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or count < 1:
        raise ValueError("Invalid inducing inputs")
    count = min(count, len(x))
    selected = np.empty(count, dtype=int)
    selected[0] = np.random.default_rng(seed).integers(len(x))
    distance = np.full(len(x), np.inf)
    for i in range(count):
        distance = np.minimum(distance, ((x - x[selected[i]]) ** 2).sum(1))
        distance[selected[:i + 1]] = -np.inf
        if i + 1 < count:
            selected[i + 1] = int(np.argmax(distance))
    return selected


def collapsed_bound(kmm, kmn, y, noise, diagonal_sum, jitter):
    """Negative Titsias bound and posterior coefficient; no n-by-n allocation.

    The trace penalty distinguishes this variational objective from DTC/FITC.
    Returns the unnormalised scalar bound and K(query, inducing) coefficients.
    """
    m, n = kmn.shape
    eye = torch.eye(m, dtype=kmm.dtype, device=kmm.device)
    lm = torch.linalg.cholesky(kmm + jitter * eye)
    a = torch.linalg.solve_triangular(lm, kmn, upper=False) / noise.sqrt()
    lb = torch.linalg.cholesky(eye + a @ a.T)
    rhs = (a @ y) / noise.sqrt()
    c = torch.linalg.solve_triangular(lb, rhs[:, None], upper=False).ravel()
    trace = diagonal_sum - noise * a.square().sum()
    objective = .5 * (n * math.log(2 * math.pi) + n * noise.log()
                      + 2 * lb.diag().log().sum()
                      + y.square().sum() / noise - c.square().sum() + trace / noise)
    coefficient = torch.linalg.solve_triangular(
        lm.T, torch.cholesky_solve(rhs[:, None], lb), upper=True).ravel()
    return objective, coefficient


class Encoder(nn.Module):
    def __init__(self, recipe, settings, n_categories):
        super().__init__()
        self.recipe = recipe
        self.embedding = None
        dimension = len(FEATURES) + n_categories
        if recipe == "GP_ARD":
            self.network = nn.Identity()
            self.output_dim = dimension
            return
        if recipe == "DKL_PLR":
            from rtdl_num_embeddings import PeriodicEmbeddings
            self.embedding = PeriodicEmbeddings(
                len(FEATURES), d_embedding=settings["periodic_embedding_dim"],
                n_frequencies=settings["periodic_n_frequencies"],
                frequency_init_scale=settings["periodic_frequency_init_scale"], lite=True)
            dimension = len(FEATURES) * settings["periodic_embedding_dim"] + n_categories
        elif recipe != "DKL_RAW":
            raise ValueError(f"Unknown V27 recipe: {recipe}")
        layers = []
        for width in settings["hidden_widths"]:
            layers.extend([nn.Linear(dimension, width), nn.SiLU()])
            dimension = width
        layers.append(nn.Linear(dimension, settings["latent_dim"]))
        self.network = nn.Sequential(*layers)
        self.output_dim = settings["latent_dim"]

    def forward(self, x):
        if self.embedding is not None:
            x = torch.cat([self.embedding(x[:, :len(FEATURES)]).flatten(1),
                           x[:, len(FEATURES):]], dim=1)
        return self.network(x)


class SparseKernel(nn.Module):
    def __init__(self, recipe, settings, n_categories):
        super().__init__()
        self.encoder = Encoder(recipe, settings, n_categories)
        self.log_length = nn.Parameter(torch.full((self.encoder.output_dim,),
                                                  math.log(math.sqrt(self.encoder.output_dim))))
        self.log_variance = nn.Parameter(torch.tensor(math.log(settings["output_variance_init"])))
        self.log_noise = nn.Parameter(torch.tensor(math.log(
            settings["noise_variance_init"] - settings["noise_variance_floor"])))
        self.noise_floor = settings["noise_variance_floor"]

    @property
    def noise(self):
        return self.log_noise.exp() + self.noise_floor

    def kernel(self, a, b):
        distance = torch.cdist(a / self.log_length.exp(), b / self.log_length.exp(),
                               compute_mode="donot_use_mm_for_euclid_dist")
        scaled = math.sqrt(5) * distance
        return self.log_variance.exp() * (1 + scaled + scaled.square() / 3) * (-scaled).exp()

    def objective(self, x, inducing, y, jitter):
        z, u = self.encoder(x), self.encoder(inducing)
        return collapsed_bound(self.kernel(u, u), self.kernel(u, z), y,
                               self.noise, len(x) * self.log_variance.exp(), jitter)


class DeepKernelRegressor:
    def __init__(self, recipe, settings):
        self.recipe, self.settings = recipe, dict(settings)

    def _inputs(self, frame):
        numeric, cat = self.preprocessor_.transform_mlp(frame)
        values = np.concatenate([numeric, cat], axis=1).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite V27 input")
        return torch.as_tensor(values, dtype=torch.float64)

    def initialize(self, frame, y):
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or not y.std() > 0:
            raise ValueError("Invalid training target")
        torch.set_num_threads(1)
        torch.manual_seed(self.settings["random_seed"])
        self.preprocessor_ = NumericPreprocessor(structure="raw_mlp").fit(frame)
        self.mean_, self.std_ = float(y.mean()), float(y.std())
        self.fit_ids_ = frame.sample_id.astype(str).tolist()
        self.model_ = SparseKernel(self.recipe, self.settings,
                                   self.preprocessor_.n_spout_categories_).double()
        x = self._inputs(frame)
        indices = farthest_rows(x.numpy(), self.settings["inducing_points"], self.settings["random_seed"])
        self.inducing_ = x[indices].clone()
        self.inducing_ids_ = [self.fit_ids_[i] for i in indices]
        self.x_train_, self.y_train_ = x, torch.as_tensor((y - self.mean_) / self.std_)
        kernel_params = [self.model_.log_length, self.model_.log_variance, self.model_.log_noise]
        groups = [{"params": kernel_params, "lr": self.settings["kernel_learning_rate"], "weight_decay": 0.0}]
        encoder_params = list(self.model_.encoder.parameters())
        if encoder_params:
            groups.append({"params": encoder_params, "lr": self.settings["encoder_learning_rate"],
                           "weight_decay": self.settings["encoder_weight_decay"]})
        self.optimizer_ = torch.optim.AdamW(groups)
        return self

    def _posterior(self):
        with torch.no_grad():
            _, self.coefficient_ = self.model_.objective(
                self.x_train_, self.inducing_, self.y_train_, self.settings["jitter"])
            self.encoded_inducing_ = self.model_.encoder(self.inducing_).detach().clone()

    def train(self, epochs, validation=None):
        best, best_epoch, stale = math.inf, 0, 0
        best_state = None
        self.history_ = []
        for epoch in range(1, int(epochs) + 1):
            self.optimizer_.zero_grad(set_to_none=True)
            loss, _ = self.model_.objective(self.x_train_, self.inducing_, self.y_train_,
                                             self.settings["jitter"])
            loss = loss / len(self.y_train_)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite GP variational objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.settings["gradient_norm_clip"],
                                          error_if_nonfinite=True)
            self.optimizer_.step()
            row = {"epoch": epoch, "negative_elbo_per_row": float(loss.detach())}
            if validation is not None:
                self._posterior()
                prediction = self.predict(validation[0])
                value = float(np.mean(np.abs(prediction - validation[1])) / self.std_)
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
        self._posterior()
        self.selected_epoch_ = best_epoch
        self.stopped_epoch_ = epoch
        return best_epoch

    def predict(self, frame):
        with torch.no_grad():
            x = self.model_.encoder(self._inputs(frame))
            values = self.model_.kernel(x, self.encoded_inducing_) @ self.coefficient_
            result = values.numpy() * self.std_ + self.mean_
        if result.shape != (len(frame),) or not np.isfinite(result).all():
            raise ValueError("Invalid V27 predictions")
        return result

    def metadata(self):
        return {"recipe": self.recipe, "selected_epoch": self.selected_epoch_,
                "stopped_epoch": self.stopped_epoch_, "fit_ids_digest": digest(self.fit_ids_),
                "inducing_ids_digest": digest(self.inducing_ids_),
                "fit_rows": len(self.fit_ids_), "inducing_rows": len(self.inducing_ids_),
                "preprocessing": self.preprocessor_.metadata(),
                "target_mean": self.mean_, "target_std": self.std_,
                "noise_variance": float(self.model_.noise.detach()),
                "output_variance": float(self.model_.log_variance.exp().detach()),
                "lengthscales": self.model_.log_length.exp().detach().tolist(),
                "history": self.history_}

    def save(self, path):
        """Persist inference state without training labels or optimizer state."""
        payload = {"recipe": self.recipe, "settings": self.settings,
                   "preprocessing": self.preprocessor_.metadata(),
                   "mean": self.mean_, "std": self.std_, "state": self.model_.state_dict(),
                   "coefficient": self.coefficient_, "encoded_inducing": self.encoded_inducing_}
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
        model.model_ = SparseKernel(model.recipe, model.settings, info["n_spout_categories"]).double()
        model.model_.load_state_dict(payload["state"])
        model.mean_, model.std_ = payload["mean"], payload["std"]
        model.coefficient_, model.encoded_inducing_ = payload["coefficient"], payload["encoded_inducing"]
        return model


def select_weight(y, baseline, member, grid):
    arrays = [np.asarray(v, dtype=float) for v in (y, baseline, member)]
    if any(v.shape != arrays[0].shape or v.ndim != 1 or not len(v)
           or not np.isfinite(v).all() for v in arrays):
        raise ValueError("Invalid calibration arrays")
    if not grid or list(grid) != sorted(set(grid)) or min(grid) < 0 or max(grid) > 1:
        raise ValueError("Sorted unique convex blend grid required")
    y, baseline, member = arrays
    loss = [float(np.abs(y - ((1 - a) * baseline + a * member)).mean()) for a in grid]
    return float(grid[int(np.argmin(loss))]), loss


def fit_partition(fitting, calibration, outer_training, query, target, recipe,
                  settings, calibration_base, grid):
    """Labels are supplied only for fitting/calibration, never through query."""
    if set(fitting.sample_id) & set(calibration.sample_id):
        raise ValueError("Fitting/calibration overlap")
    if set(outer_training.sample_id) != set(fitting.sample_id) | set(calibration.sample_id):
        raise ValueError("Incorrect outer training union")
    if set(outer_training.sample_id) & set(query.sample_id):
        raise ValueError("Outer query overlaps training")
    if any(c in query.columns for c in ("tap_iron", "tap_time_len")):
        raise ValueError("Query labels must be removed")
    selector = DeepKernelRegressor(recipe, settings).initialize(fitting, fitting[target].to_numpy())
    epoch = selector.train(settings["max_epochs"], (calibration, calibration[target].to_numpy()))
    calibration_prediction = selector.predict(calibration)
    weight, losses = select_weight(calibration[target], calibration_base, calibration_prediction, grid)
    calibration_meta = selector.metadata()
    final = DeepKernelRegressor(recipe, settings).initialize(outer_training, outer_training[target].to_numpy())
    final.train(epoch)
    prediction = final.predict(query)
    final.calibration_model_ = selector
    return final, prediction, {"weight": weight, "calibration_mae_by_weight": losses,
                              "calibration": calibration_meta, "refit": final.metadata()}, calibration_prediction
