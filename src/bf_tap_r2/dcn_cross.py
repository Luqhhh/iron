"""Original full-matrix DCN-v2 adaptation and a matched additive control.

No external weights, feature enumeration, or official-data entrypoint.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import FEATURES, TARGETS
from .v7_periodic import digest, file_hash

ARMS = ("ADDITIVE", "CROSS")


def clean(frame):
    return frame.drop(columns=[t for t in TARGETS if t in frame])


def validate_rows(frame):
    if not len(frame) or frame.sample_id.isna().any() or frame.sample_id.astype(str).duplicated().any():
        raise ValueError("Nonempty unique sample identities required")


class CrossPreprocessor:
    """Training-only numeric moments and spout vocabulary; unseen spout is zero."""

    def fit(self, frame):
        validate_rows(frame)
        if any(t in frame for t in TARGETS):
            raise ValueError("Preprocessor receives unlabeled inputs")
        x = frame[list(FEATURES)].to_numpy(dtype=float)
        if not np.isfinite(x).all() or frame.spout_no.isna().any():
            raise ValueError("Finite inputs required")
        self.mean_ = x.mean(0)
        self.std_ = x.std(0)
        self.std_[self.std_ < 1e-12] = 1.
        self.categories_ = sorted(int(c) for c in frame.spout_no.unique())
        if not np.isin(frame.spout_no, self.categories_).all():
            raise ValueError("Integer spout categories required")
        self.fit_digest_ = digest(frame.sample_id.astype(str).tolist())
        return self

    def transform(self, frame):
        if any(t in frame for t in TARGETS):
            raise ValueError("Prediction inputs carry targets")
        numeric = (frame[list(FEATURES)].to_numpy(float) - self.mean_) / self.std_
        if not np.isfinite(numeric).all() or frame.spout_no.isna().any():
            raise ValueError("Finite query inputs required")
        categories = np.column_stack([frame.spout_no.to_numpy() == c for c in self.categories_])
        return np.column_stack((numeric, categories)).astype(np.float64)

    def metadata(self):
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist(),
                "categories": self.categories_, "fit_ids_digest": self.fit_digest_}

    @classmethod
    def restore(cls, data):
        obj = cls()
        obj.mean_, obj.std_ = np.array(data["mean"]), np.array(data["std"])
        obj.categories_, obj.fit_digest_ = data["categories"], data["fit_ids_digest"]
        return obj


class CrossNetwork(nn.Module):
    def __init__(self, width, settings, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Unknown cross arm")
        self.arm = arm
        # Identical parameters, head, RNG consumption and identity initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings["random_seed"])
            layers = []
            dimension = width
            for hidden in settings["deep_widths"]:
                layers.extend([nn.Linear(dimension, hidden, dtype=torch.float64), nn.ReLU(),
                               nn.Dropout(settings["dropout"])])
                dimension = hidden
            self.deep = nn.Sequential(*layers)
            self.head = nn.Linear(width + dimension, 1, dtype=torch.float64)
            self.cross = nn.ModuleList([nn.Linear(width, width, dtype=torch.float64)
                                        for _ in range(settings["cross_layers"])])
        for layer in self.cross:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def cross_branch(self, x):
        h = x
        for layer in self.cross:
            update = layer(h)
            h = h + (x * update if self.arm == "CROSS" else update)
        return h

    def forward(self, x):
        return self.head(torch.cat([self.cross_branch(x), self.deep(x)], dim=1)).flatten()


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


class CrossRegressor:
    def __init__(self, arm, settings):
        self.arm, self.settings = arm, deepcopy(settings)

    def initialize(self, frame, y):
        validate_rows(frame)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or y.std() <= 0:
            raise ValueError("Finite nonconstant target with matching rows required")
        torch.set_num_threads(1)
        self.preprocessor_ = CrossPreprocessor().fit(clean(frame))
        self.mean_, self.std_ = float(y.mean()), float(y.std())
        self.fit_ids_ = frame.sample_id.astype(str).tolist()
        self.model_ = CrossNetwork(len(FEATURES) + len(self.preprocessor_.categories_), self.settings, self.arm)
        self.initial_state_digest_ = state_digest(self.model_.state_dict())
        self.x_ = torch.tensor(self.preprocessor_.transform(clean(frame)), dtype=torch.float64)
        self.y_ = torch.tensor((y - self.mean_) / self.std_, dtype=torch.float64)
        self.optimizer_ = torch.optim.AdamW(self.model_.parameters(), lr=self.settings["learning_rate"],
                                           weight_decay=self.settings["weight_decay"])
        self.trained_ = False
        return self

    def train(self, epochs, validation=None):
        if self.trained_:
            raise ValueError("No implicit resume or repeated fit")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= self.settings["max_epochs"]:
            raise ValueError("Epoch count outside frozen budget")
        if validation is not None:
            validate_rows(validation[0])
            if set(self.fit_ids_) & set(validation[0].sample_id.astype(str)):
                raise ValueError("Calibration overlaps fitting rows")
            vy = np.asarray(validation[1], float)
            if vy.shape != (len(validation[0]),) or not np.isfinite(vy).all():
                raise ValueError("Invalid calibration target")
        self.trained_ = True
        best, stale, selected, checkpoint = float("inf"), 0, 0, None
        self.history_ = []
        rng = np.random.default_rng(self.settings["random_seed"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.settings["random_seed"])
            for epoch in range(1, epochs + 1):
                self.model_.train()
                order = rng.permutation(len(self.y_))
                total = 0.
                for start in range(0, len(order), self.settings["batch_size"]):
                    idx = order[start:start + self.settings["batch_size"]]
                    self.optimizer_.zero_grad(set_to_none=True)
                    loss = (self.model_(self.x_[idx]) - self.y_[idx]).square().mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite cross loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.settings["gradient_norm_clip"],
                                             error_if_nonfinite=True)
                    self.optimizer_.step()
                    total += float(loss.detach()) * len(idx)
                row = {"epoch": epoch, "standardized_training_mse": total / len(order)}
                if validation is not None:
                    value = float(np.abs(self.predict(validation[0]) - vy).mean() / self.std_)
                    row["calibration_standardized_mae"] = value
                    if value < best - self.settings["min_delta_standardized_mae"]:
                        best, selected, stale = value, epoch, 0
                        checkpoint = deepcopy(self.model_.state_dict())
                    else:
                        stale += 1
                self.history_.append(row)
                if validation is not None and stale >= self.settings["patience"]:
                    break
        if validation is not None:
            if checkpoint is None:
                raise ValueError("No selected checkpoint")
            self.model_.load_state_dict(checkpoint)
        else:
            selected = epoch
        self.model_.eval()
        self.selected_epoch_, self.stopped_epoch_ = selected, epoch
        return selected

    def predict(self, frame):
        self.model_.eval()
        with torch.no_grad():
            x = torch.tensor(self.preprocessor_.transform(frame), dtype=torch.float64)
            result = self.model_(x).numpy() * self.std_ + self.mean_
        if result.shape != (len(frame),) or not np.isfinite(result).all():
            raise ValueError("Invalid cross predictions")
        return result

    def metadata(self):
        return {"preprocessing": self.preprocessor_.metadata(), "target_mean": self.mean_,
                "target_std": self.std_, "fit_ids_digest": digest(self.fit_ids_), "fit_rows": len(self.fit_ids_),
                "selected_epoch": self.selected_epoch_, "stopped_epoch": self.stopped_epoch_,
                "history": self.history_, "initial_state_digest": self.initial_state_digest_,
                "state_digest": state_digest(self.model_.state_dict()),
                "parameter_count": sum(p.numel() for p in self.model_.parameters())}

    def save(self, path):
        data = {"arm": self.arm, "settings": self.settings, "metadata": self.metadata(),
                "state": {k: v.detach().numpy().tolist() for k, v in self.model_.state_dict().items()}}
        with Path(path).open("x") as stream:
            json.dump(data, stream, allow_nan=False)
        return file_hash(path)

    @classmethod
    def load(cls, path, expected_sha256):
        if not expected_sha256 or file_hash(path) != expected_sha256:
            raise ValueError("Externally anchored model hash required")
        data = json.loads(Path(path).read_text())
        obj = cls(data["arm"], data["settings"])
        meta = data["metadata"]
        obj.saved_metadata_ = meta
        obj.preprocessor_ = CrossPreprocessor.restore(meta["preprocessing"])
        obj.mean_, obj.std_ = meta["target_mean"], meta["target_std"]
        if not np.isfinite([obj.mean_, obj.std_]).all() or obj.std_ <= 0:
            raise ValueError("Invalid saved target scale")
        obj.model_ = CrossNetwork(len(FEATURES) + len(obj.preprocessor_.categories_), obj.settings, obj.arm)
        state = {k: torch.tensor(v, dtype=torch.float64) for k, v in data["state"].items()}
        if not all(torch.isfinite(v).all() for v in state.values()) or state_digest(state) != meta["state_digest"]:
            raise ValueError("Invalid saved parameter state")
        obj.model_.load_state_dict(state, strict=True)
        obj.model_.eval()
        return obj


def fit_partition(fitting, calibration, outer_training, query, target, arm, settings):
    """Select on calibration then fresh refit; no weight learning in preparation."""
    if target not in TARGETS:
        raise ValueError("Unknown target")
    for frame in (fitting, calibration, outer_training, query):
        validate_rows(frame)
    ids = [set(p.sample_id.astype(str)) for p in (fitting, calibration, outer_training, query)]
    if ids[0] & ids[1] or ids[2] != ids[0] | ids[1] or ids[2] & ids[3]:
        raise ValueError("Invalid disjoint fit/calibration/outer/query partition")
    # Row content and order in the supplied full training partition must be genuine.
    combined = pd.concat([fitting, calibration]).set_index("sample_id").sort_index()
    if not combined.equals(outer_training.set_index("sample_id").sort_index()):
        raise ValueError("Training union changed row contents")
    if any(t in query for t in TARGETS):
        raise ValueError("Query labels forbidden")
    selector = CrossRegressor(arm, settings).initialize(fitting, fitting[target].to_numpy())
    epoch = selector.train(settings["max_epochs"], (clean(calibration), calibration[target].to_numpy()))
    cp = selector.predict(clean(calibration))
    refit = CrossRegressor(arm, settings).initialize(outer_training, outer_training[target].to_numpy())
    refit.train(epoch)
    if refit.initial_state_digest_ != selector.initial_state_digest_:
        # A different vocabulary is legitimate, so compare only when widths match.
        if refit.preprocessor_.categories_ == selector.preprocessor_.categories_:
            raise ValueError("Fresh refit initial state differs")
    return refit, selector, refit.predict(query), cp
